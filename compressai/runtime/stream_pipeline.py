# compressai/runtime/stream_pipeline.py
"""
Two-stream batch-level pipeline for overlapped TRT inference and GPU ANS coding.

Architecture constraints:
  - GPU ANS kernels use ``at::cuda::getDefaultCUDAStream()`` (hardcoded in C++)
  - TRTModule reads ``torch.cuda.current_stream()`` at call time

Strategy:
  - TRT inference runs on a custom non-default stream
  - GPU ANS runs on the default stream

Two modes
────────
``compress_pipelined`` (legacy)
    Uses the monolithic ``codec.compress(y_fp32)``. Partial overlap via
    interleaved host launches.

``compress_pipelined_session`` (recommended)
    Uses ``PipelinedAnsEncoder`` whose ``launch()`` is non-blocking.
    Phase 1 launches all (ga + enc) work without host sync; Phase 2
    finalizes.  Gives true GPU overlap.

Timeline (session-based, 3 batches)::

    GPU TRT:    ga0███   ga1███   ga2███
    GPU ANS:        █████████████████████████████enc0██████████████████████...
                         ↑ real overlap ↑
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import torch

from compressai.runtime.ans_pipeline import PipelinedAnsEncoder, PipelinedAnsDecoder


Pack = Dict[str, Any]
Runner = Callable[..., torch.Tensor]
CodecLike = Any
PrepFn = Callable[[torch.Tensor], tuple]  # y -> (symbols_i32, indexes_i32)


# ═══════════════════════════════════════════════════════════════
# Legacy interleaved compress
# ═══════════════════════════════════════════════════════════════

def compress_pipelined(
    ga_runner: Runner,
    codec: CodecLike,
    blocks: List[torch.Tensor],
    *,
    ga_input_dtype: Optional[torch.dtype] = None,
    codec_input_dtype: torch.dtype = torch.float32,
) -> List[Pack]:
    """
    Batch-pipelined compress with interleaved launch (legacy).

    Prefer ``compress_pipelined_session`` for better overlap.
    """
    N = len(blocks)
    if N == 0:
        return []
    if N == 1:
        y = blocks[0]
        if ga_input_dtype is not None and y.dtype != ga_input_dtype:
            y = y.to(ga_input_dtype)
        y = ga_runner(y)
        y = y.to(codec_input_dtype) if y.dtype != codec_input_dtype else y
        return [codec.compress(y)]

    trt_stream = torch.cuda.Stream()
    ans_stream = torch.cuda.default_stream()

    def _launch_ga(x):
        with torch.cuda.stream(trt_stream):
            if ga_input_dtype is not None and x.dtype != ga_input_dtype:
                x = x.to(ga_input_dtype)
            if not x.is_contiguous():
                x = x.contiguous()
            y = ga_runner(x)
            y_fp32 = y.to(codec_input_dtype) if y.dtype != codec_input_dtype else y
            y_fp32 = y_fp32.contiguous()
            y_fp32.record_stream(ans_stream)
            ev = torch.cuda.Event()
            ev.record(trt_stream)
        return y_fp32, ev

    packs: List[Optional[Pack]] = [None] * N
    y_prev, ev_prev = _launch_ga(blocks[0])

    for i in range(1, N):
        y_curr, ev_curr = _launch_ga(blocks[i])
        with torch.cuda.stream(ans_stream):
            ans_stream.wait_event(ev_prev)
            packs[i - 1] = codec.compress(y_prev)
        y_prev, ev_prev = y_curr, ev_curr

    with torch.cuda.stream(ans_stream):
        ans_stream.wait_event(ev_prev)
        packs[N - 1] = codec.compress(y_prev)

    torch.cuda.synchronize()
    return packs  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════
# Session-based compress (recommended)
# ═══════════════════════════════════════════════════════════════

def compress_pipelined_session(
    ga_runner: Runner,
    prep_for_encode: PrepFn,
    encoders: List[PipelinedAnsEncoder],
    blocks: List[torch.Tensor],
    *,
    ga_input_dtype: Optional[torch.dtype] = None,
) -> List[Pack]:
    """
    Batch-pipelined compress with session-based async encoder.

    Phase 1 launches all batches' (ga + enc) work, submitting kernels
    without host synchronisation.  Phase 2 finalizes each batch's
    compressed output.  The GPU scheduler can overlap ``ga(batch_{i+1})``
    on the TRT stream with ``enc(batch_i)`` on the ANS stream.

    Each batch needs its own ``PipelinedAnsEncoder`` (separate buffers).

    Args:
        ga_runner:   ``x -> y`` (TRTModule wrapping g_a).
        prep_for_encode:  ``y -> (symbols_i32, indexes_i32)``.
        encoders:    list of ``PipelinedAnsEncoder`` (same length as blocks).
        blocks:      list of CUDA tensors, one per batch.
        ga_input_dtype:  optional dtype to cast x to.

    Returns:
        packs: list of pack dicts ``{"strings": TightWarpANS, "state": {...}}``.

    Example::

        def prep(y):
            symbols = eb.quantize(y, "symbols", means=None)
            B, C, H, W = symbols.shape
            idx = torch.arange(C, device=y.device, dtype=torch.int32) \\
                      .view(1, C, 1, 1).expand(B, C, H, W)
            return symbols.to(torch.int32).contiguous(), idx

        encoders = [
            PipelinedAnsEncoder.from_entropy_bottleneck(eb, batchsize, N, P)
            for _ in range(num_batches)
        ]
        packs = compress_pipelined_session(ga, prep, encoders, blocks)
    """
    N = len(blocks)
    if N == 0:
        return []
    if len(encoders) != N:
        raise ValueError(f"Need {N} encoders (one per batch), got {len(encoders)}")

    trt_stream = torch.cuda.Stream()
    ans_stream = torch.cuda.default_stream()

    # ── Phase 1: Launch ALL (ga + encode) ──
    sizes_hw: List[tuple] = [None] * N  # type: ignore[assignment]

    for i, x in enumerate(blocks):
        # --- ga + quantize on TRT stream ---
        with torch.cuda.stream(trt_stream):
            if ga_input_dtype is not None and x.dtype != ga_input_dtype:
                x = x.to(ga_input_dtype)
            if not x.is_contiguous():
                x = x.contiguous()
            y = ga_runner(x)
            symbols, indexes = prep_for_encode(y)

            if not symbols.is_contiguous():
                symbols = symbols.contiguous()
            if not indexes.is_contiguous():
                indexes = indexes.contiguous()

            sizes_hw[i] = tuple(y.shape[-2:])

            symbols.record_stream(ans_stream)
            indexes.record_stream(ans_stream)
            ev = torch.cuda.Event()
            ev.record(trt_stream)

        # --- encode launch on ANS stream (non-blocking!) ---
        with torch.cuda.stream(ans_stream):
            ans_stream.wait_event(ev)
            encoders[i].launch(symbols, indexes)

    # ── Phase 2: Finalize all ──
    packs: List[Pack] = []
    for i in range(N):
        packed = encoders[i].finalize()
        packs.append({"strings": packed, "state": {"size_hw": sizes_hw[i]}})

    torch.cuda.synchronize()
    return packs


# ═══════════════════════════════════════════════════════════════
# Decompress pipelines (unchanged)
# ═══════════════════════════════════════════════════════════════

def decompress_pipelined(
    gs_runner: Runner,
    codec: CodecLike,
    packs: List[Pack],
    *,
    gs_input_dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Batch-pipelined decompress with interleaved launch."""
    N = len(packs)
    if N == 0:
        return []
    if N == 1:
        y_hat = codec.decompress(packs[0])
        if y_hat.dtype != gs_input_dtype:
            y_hat = y_hat.to(gs_input_dtype)
        return [gs_runner(y_hat)]

    trt_stream = torch.cuda.Stream()
    ans_stream = torch.cuda.default_stream()

    x_hats: List[Optional[torch.Tensor]] = [None] * N

    def _launch_gs(y_hat, ev):
        with torch.cuda.stream(trt_stream):
            trt_stream.wait_event(ev)
            if y_hat.dtype != gs_input_dtype:
                y_hat = y_hat.to(gs_input_dtype)
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()
            return gs_runner(y_hat).contiguous()

    with torch.cuda.stream(ans_stream):
        y_prev = codec.decompress(packs[0])
        if not isinstance(y_prev, torch.Tensor):
            raise TypeError("codec.decompress must return a torch.Tensor")
        y_prev = y_prev.contiguous()
        y_prev.record_stream(trt_stream)
        ev_prev = torch.cuda.Event()
        ev_prev.record(ans_stream)

    for i in range(1, N):
        x_hats[i - 1] = _launch_gs(y_prev, ev_prev)
        with torch.cuda.stream(ans_stream):
            y_curr = codec.decompress(packs[i])
            if not isinstance(y_curr, torch.Tensor):
                raise TypeError("codec.decompress must return a torch.Tensor")
            y_curr = y_curr.contiguous()
            y_curr.record_stream(trt_stream)
            ev_curr = torch.cuda.Event()
            ev_curr.record(ans_stream)
        y_prev, ev_prev = y_curr, ev_curr

    x_hats[N - 1] = _launch_gs(y_prev, ev_prev)

    torch.cuda.synchronize()
    return x_hats  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════
# Session-based decompress
# ═══════════════════════════════════════════════════════════════

def decompress_pipelined_session(
    gs_runner: Runner,
    dequantize_for_gs: Callable[[torch.Tensor], torch.Tensor],
    decoders: List[PipelinedAnsDecoder],
    packs: List[Pack],
    *,
    gs_input_dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """
    Batch-pipelined decompress with session-based async decoder.

    Interleaves decode finalize + gs launch so that ``gs(batch_{i-1})`` on
    TRT stream overlaps with ``dec(batch_i)`` on ANS stream.

    Each batch needs its own ``PipelinedAnsDecoder`` (separate buffers).

    Timeline::

        ANS stream:  dec₀ ████  dec₁ ████  dec₂ ████  ...
        TRT stream:         ░░░░ gs(y₀) ████ gs(y₁) ████ ...
                                 ↑ overlap ↑

    Args:
        gs_runner:        ``y_hat -> x_hat`` (TRTModule wrapping g_s).
        dequantize_for_gs: ``decoded_int32 [B,N] -> y_hat [B,C,H,W] float``.
        decoders:          list of ``PipelinedAnsDecoder`` (same length as packs).
        packs:             list of pack dicts from compress.
        gs_input_dtype:    dtype to cast y_hat to before calling gs.

    Returns:
        x_hats: list of reconstructed tensors, one per batch.
    """
    N = len(packs)
    if N == 0:
        return []
    if len(decoders) != N:
        raise ValueError(f"Need {N} decoders, got {len(decoders)}")

    trt_stream = torch.cuda.Stream()
    ans_stream = torch.cuda.default_stream()
    x_hats: List[Optional[torch.Tensor]] = [None] * N

    def _build_indexes(pack):
        strings = pack["strings"]
        size_hw = pack["state"]["size_hw"]
        B = strings.max_rounds_u32.size(0)
        C = decoders[0]._cdfs.size(0)  # all decoders share same CDFs
        return torch.arange(C, dtype=torch.int32, device=strings.packed.device) \
                    .view(1, C, 1, 1) \
                    .expand(B, C, size_hw[0], size_hw[1]) \
                    .reshape(B, -1).contiguous()

    def _launch_decode(i):
        """Launch decode kernel for batch i on ANS stream (non-blocking)."""
        strings = packs[i]["strings"]
        idx = _build_indexes(packs[i])
        strings.packed.record_stream(ans_stream)
        strings.max_rounds_u32.record_stream(ans_stream)
        idx.record_stream(ans_stream)
        with torch.cuda.stream(ans_stream):
            decoders[i].launch(
                packed_u8=strings.packed,
                max_rounds_u32=strings.max_rounds_u32,
                header_bytes=int(strings.header_bytes_cpu.item()),
                indexes_bxn=idx,
            )

    def _finalize_and_gs(i):
        """Finalize decode i (sync ANS stream), then launch gs on TRT stream."""
        out_i32 = decoders[i].finalize()
        y_hat = dequantize_for_gs(out_i32)
        with torch.cuda.stream(trt_stream):
            if y_hat.dtype != gs_input_dtype:
                y_hat = y_hat.to(gs_input_dtype)
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()
            x_hat = gs_runner(y_hat)
            return x_hat.contiguous()

    # ── Batch 0: launch decode first, nothing to overlap with ──
    _launch_decode(0)

    # ── Batches 1..N-1: interleave ──
    # Pattern: gs(i-1) || dec(i)
    for i in range(1, N):
        # Wait for dec(i-1), then gs(i-1) on TRT
        x_hats[i - 1] = _finalize_and_gs(i - 1)
        # Launch dec(i) on ANS — its kernels run while gs(i-1) is on TRT
        _launch_decode(i)

    # ── Last batch: gs only ──
    x_hats[N - 1] = _finalize_and_gs(N - 1)

    torch.cuda.synchronize()
    assert all(x is not None for x in x_hats)
    return x_hats  # type: ignore[return-value]
