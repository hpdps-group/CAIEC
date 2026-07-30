"""Session-based CUDA pipeline for :class:`DCAEEngine`.

The inference side is submitted to a private CUDA stream; ANS sessions use the
PyTorch default stream, as required by the C++ extension.  DCAE's slice context
is inherently sequential, but encoding work already launched for z and earlier
slices may overlap subsequent inference kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from compressai.entropy_models.entropy_models import TightWarpANS
from compressai.runtime.ans_pipeline import PipelinedAnsDecoder, PipelinedAnsEncoder
from compressai.runtime.engines.dcae_engine import DCAEEngine


@dataclass
class DCAEEncodeSessions:
    """One independent z session and ``num_slices`` y sessions per batch."""

    z: List[PipelinedAnsEncoder]
    y: List[List[PipelinedAnsEncoder]]
    batch_size: int
    z_shape: Tuple[int, int, int]
    y_shape: Tuple[int, int, int]


@dataclass
class DCAEDecodeSessions:
    """Decode-session counterpart of :class:`DCAEEncodeSessions`."""

    z: List[PipelinedAnsDecoder]
    y: List[List[PipelinedAnsDecoder]]
    batch_size: int
    z_shape: Tuple[int, int, int]
    y_shape: Tuple[int, int, int]


def _shape3(name: str, shape: Sequence[int]) -> Tuple[int, int, int]:
    if not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 3:
        raise ValueError(f"{name} must be (C, H, W)")
    out = tuple(int(v) for v in shape)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} dimensions must be positive")
    return out  # type: ignore[return-value]


def _check_engine(engine: DCAEEngine) -> None:
    if not isinstance(engine, DCAEEngine):
        raise TypeError("engine must be a DCAEEngine")
    if engine.codec is None or engine.codec.gaussian_conditional is None:
        raise ValueError("DCAE pipeline requires EB and GaussianConditional codecs")
    if getattr(engine.codec, "_variant", None) != "warp_smem":
        raise ValueError("session pipeline requires codec variant 'warp_smem'")


def create_dcae_encode_sessions(
    engine: DCAEEngine,
    num_batches: int,
    batch_size: int,
    z_shape: Sequence[int],
    y_shape: Sequence[int],
    *,
    parallelism: Optional[int] = None,
) -> DCAEEncodeSessions:
    """Allocate fixed-shape encode sessions.

    ``z_shape`` and ``y_shape`` are per-sample ``(C,H,W)`` shapes.  ``y_shape``
    is the full y latent; its channel count must divide ``engine.num_slices``.
    """
    _check_engine(engine)
    if num_batches < 0 or batch_size <= 0:
        raise ValueError("num_batches must be non-negative and batch_size positive")
    zc, zh, zw = _shape3("z_shape", z_shape)
    yc, yh, yw = _shape3("y_shape", y_shape)
    if yc % engine.num_slices:
        raise ValueError("y channels must be divisible by engine.num_slices")
    p = int(engine.codec.P if parallelism is None else parallelism)
    if p <= 0:
        raise ValueError("parallelism must be positive")
    eb, gc = engine.codec.eb, engine.codec.gaussian_conditional
    eb_cdfs = int(eb._quantized_cdf.shape[0])
    gc_cdfs = int(gc._quantized_cdf.shape[0])
    if zc != eb_cdfs:
        raise ValueError(f"z channels ({zc}) must match EntropyBottleneck CDFs ({eb_cdfs})")
    y_slice_channels = yc // engine.num_slices
    z_n = zc * zh * zw
    ys_n = y_slice_channels * yh * yw
    if ys_n % gc_cdfs:
        raise ValueError(
            f"y slice symbols ({ys_n}) must be divisible by GaussianConditional CDFs ({gc_cdfs})"
        )
    return DCAEEncodeSessions(
        z=[PipelinedAnsEncoder.from_entropy_bottleneck(eb, batch_size, z_n, p)
           for _ in range(num_batches)],
        y=[[PipelinedAnsEncoder.from_gaussian_conditional(gc, batch_size, ys_n, p)
            for _ in range(engine.num_slices)] for _ in range(num_batches)],
        batch_size=batch_size, z_shape=(zc, zh, zw), y_shape=(yc, yh, yw),
    )


def _warp_meta(value: Any, name: str) -> Tuple[int, int, int, int, int]:
    if not isinstance(value, TightWarpANS):
        raise TypeError(f"{name} must be TightWarpANS (session decoder cannot read TightANS)")
    if value.packed.device.type != "cuda" or value.packed.dtype != torch.uint8:
        raise ValueError(f"{name}.packed must be CUDA uint8")
    if value.max_rounds_u32.device.type != "cuda" or value.max_rounds_u32.dtype != torch.uint32:
        raise ValueError(f"{name}.max_rounds_u32 must be CUDA uint32")
    if value.max_rounds_u32.ndim != 2:
        raise ValueError(f"{name}.max_rounds_u32 must have shape (B, K)")
    try:
        b = int(value.max_rounds_u32.shape[0])
        k = int(value.max_rounds_u32.shape[1])
        header = int(value.header_bytes_cpu.reshape(-1)[0].item())
        chunk = int(value.chunk_len_cpu.reshape(-1)[0].item())
        p = int(value.P_cpu.reshape(-1)[0].item())
    except (IndexError, RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid TightWarpANS metadata in {name}") from exc
    if min(b, k, header, chunk, p) <= 0:
        raise ValueError(f"invalid non-positive TightWarpANS metadata in {name}")
    expected_header = (32 + 4 * b * k + 15) & ~15
    if header != expected_header:
        raise ValueError(f"invalid TightWarpANS header size in {name}")
    expected_bytes = header + int(value.max_rounds_u32.sum().item()) * 32 * 4
    if value.packed.numel() != expected_bytes:
        raise ValueError(f"invalid TightWarpANS payload size in {name}")
    return b, k, header, chunk, p


def create_dcae_decode_sessions(
    engine: DCAEEngine,
    packs: Sequence[Dict[str, Any]],
) -> DCAEDecodeSessions:
    """Allocate fixed-shape decoders from pack metadata.

    A separate factory is necessary because the C++ decoder contract fixes K,
    chunk length, HW, B and N at session creation time.
    """
    _check_engine(engine)
    if not packs:
        return DCAEDecodeSessions([], [], 0, (0, 0, 0), (0, 0, 0))
    z_decoders: List[PipelinedAnsDecoder] = []
    y_decoders: List[List[PipelinedAnsDecoder]] = []
    expected = None
    eb, gc = engine.codec.eb, engine.codec.gaussian_conditional
    for bi, pack in enumerate(packs):
        if not isinstance(pack, dict) or "z" not in pack or "y" not in pack:
            raise ValueError(f"packs[{bi}] must contain y and z")
        zp = pack["z"]
        yp = pack["y"]
        if not isinstance(zp, dict) or "strings" not in zp or "state" not in zp:
            raise ValueError(f"packs[{bi}]['z'] has invalid pack structure")
        if not isinstance(yp, dict) or "strings" not in yp:
            raise ValueError(f"packs[{bi}]['y'] has invalid pack structure")
        ys = yp["strings"]
        if not isinstance(ys, (list, tuple)) or len(ys) != engine.num_slices:
            raise ValueError(f"packs[{bi}]['y']['strings'] must have {engine.num_slices} entries")
        z_hw = tuple(zp["state"].get("size_hw", ())) if isinstance(zp["state"], dict) else ()
        if len(z_hw) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in z_hw):
            raise ValueError(f"packs[{bi}]['z']['state']['size_hw'] is invalid")
        z_b, z_k, _, z_chunk, z_p = _warp_meta(zp["strings"], f"packs[{bi}].z.strings")
        y_hw = engine._get_y_shape(yp, torch.empty((z_b, 1, *z_hw)))
        y_meta = [_warp_meta(v, f"packs[{bi}].y.strings[{si}]") for si, v in enumerate(ys)]
        if any(m[0] != z_b for m in y_meta):
            raise ValueError(f"packs[{bi}] y/z batch sizes differ")
        zc = int(engine.codec.eb._quantized_cdf.shape[0])
        z_hw_flat = z_hw[0] * z_hw[1]
        zn = zc * z_hw_flat
        if z_chunk % z_hw_flat or z_p != z_chunk // z_hw_flat:
            raise ValueError(f"packs[{bi}] z TightWarpANS chunk metadata is inconsistent with shape")
        if z_k != (zc + z_p - 1) // z_p:
            raise ValueError(f"packs[{bi}] z TightWarpANS K is inconsistent with shape")
        yc = getattr(engine.net, "M", None)
        if not isinstance(yc, int) or isinstance(yc, bool) or yc <= 0:
            raise ValueError("engine.net.M must provide the fixed y channel count")
        if yc % engine.num_slices:
            raise ValueError("engine.net.M must be divisible by num_slices")
        ycs = yc // engine.num_slices
        y_hw_flat = y_hw[0] * y_hw[1]
        for meta in y_meta:
            _, y_k, _, y_chunk, y_p = meta
            if y_chunk % y_hw_flat or y_p != y_chunk // y_hw_flat:
                raise ValueError(f"packs[{bi}] y TightWarpANS chunk metadata is inconsistent with shape")
            if y_k != (ycs + y_p - 1) // y_p:
                raise ValueError(f"packs[{bi}] y TightWarpANS K is inconsistent with shape")
        current = (z_b, (zc, *z_hw), (yc, *y_hw))
        if expected is None:
            expected = current
        elif current != expected:
            raise ValueError("all packs must have identical fixed B/z/y shapes")
        z_decoders.append(PipelinedAnsDecoder.from_entropy_bottleneck(
            eb, z_b, zc * z_hw[0] * z_hw[1], z_k, z_chunk, z_hw[0] * z_hw[1]))
        y_decoders.append([
            PipelinedAnsDecoder.from_gaussian_conditional(
                gc, z_b, ycs * y_hw[0] * y_hw[1], m[1], m[3], y_hw[0] * y_hw[1])
            for m in y_meta
        ])
    assert expected is not None
    return DCAEDecodeSessions(z_decoders, y_decoders, expected[0], expected[1], expected[2])


def _validate_blocks(blocks: Sequence[torch.Tensor], sessions: DCAEEncodeSessions) -> None:
    if len(blocks) != len(sessions.z) or len(sessions.y) != len(blocks):
        raise ValueError("one independent session set is required per input block")
    shape = None
    for i, x in enumerate(blocks):
        if not isinstance(x, torch.Tensor) or not x.is_cuda or x.ndim != 4:
            raise ValueError(f"blocks[{i}] must be a BCHW CUDA tensor")
        if x.shape[0] != sessions.batch_size:
            raise ValueError(f"blocks[{i}] batch size does not match sessions")
        if shape is None:
            shape = tuple(x.shape)
        elif tuple(x.shape) != shape:
            raise ValueError("all blocks must have one fixed shape")


@torch.no_grad()
def compress_dcae_pipelined_session(
    engine: DCAEEngine,
    sessions: DCAEEncodeSessions,
    blocks: Sequence[torch.Tensor],
) -> List[Dict[str, Any]]:
    """Compress blocks with custom inference and default ANS streams."""
    _check_engine(engine)
    _validate_blocks(blocks, sessions)
    if not blocks:
        return []
    trt_stream, ans_stream = torch.cuda.Stream(), torch.cuda.default_stream()
    pending: List[Tuple[PipelinedAnsEncoder, torch.Tensor, torch.Tensor]] = []
    states: List[List[Dict[str, Any]]] = []
    eb, gc = engine.codec.eb, engine.codec.gaussian_conditional

    for bi, input_x in enumerate(blocks):
        with torch.cuda.stream(trt_stream):
            x = input_x.contiguous()
            x = engine._cast(x, engine.ga_input_dtype)
            y = engine._ensure_cuda_contiguous(engine.runners["ga"](x))
            z = engine._ensure_cuda_contiguous(engine.runners["ha"](engine._cast(y, engine.ha_input_dtype)))
            if tuple(z.shape[1:]) != sessions.z_shape or tuple(y.shape[1:]) != sessions.y_shape:
                raise ValueError(f"block {bi} runner output shape does not match fixed sessions")
            z_fp32 = z.float().contiguous()
            med = eb._extend_ndims(eb._get_medians().detach(), z.ndim - 2).expand_as(z_fp32)
            z_symbols = eb.quantize(z_fp32, "symbols", med).contiguous()
            z_indexes = eb._build_indexes(z.shape, device=z.device).contiguous()
            z_hat = eb.dequantize(z_symbols, med).contiguous()
            ready = torch.cuda.Event()
            ready.record(trt_stream)
            z_symbols.record_stream(ans_stream); z_indexes.record_stream(ans_stream)
        with torch.cuda.stream(ans_stream):
            ans_stream.wait_event(ready)
            sessions.z[bi].launch(z_symbols, z_indexes)
        pending.append((sessions.z[bi], z_symbols, z_indexes))

        with torch.cuda.stream(trt_stream):
            scales = engine._ensure_cuda_contiguous(engine.runners["h_z_s1"](engine._cast(z_hat, engine.h_z_s1_input_dtype)))
            means = engine._ensure_cuda_contiguous(engine.runners["h_z_s2"](engine._cast(z_hat, engine.h_z_s2_input_dtype)))
            dt = engine._repeat_dt(sessions.batch_size, y.device, means.dtype)
            hats: List[torch.Tensor] = []
            batch_states: List[Dict[str, Any]] = []
            for si, y_slice in enumerate(y.chunk(engine.num_slices, 1)):
                support_slices = hats if engine.max_support_slices < 0 else hats[:engine.max_support_slices]
                query = torch.cat([scales, means] + support_slices, 1)
                info = engine.runners[f"dt_cross_attention_{si}"](engine._cast(query, engine.dt_ca_input_dtypes[si]), dt)
                support = torch.cat([query, info], 1)
                mu = engine.runners[f"cc_mean_{si}"](engine._cast(support, engine.cc_mean_input_dtypes[si]))[:, :, :y.shape[2], :y.shape[3]].float()
                scale = engine.runners[f"cc_scale_{si}"](engine._cast(support, engine.cc_scale_input_dtypes[si]))[:, :, :y.shape[2], :y.shape[3]].float()
                indexes = gc.build_indexes(scale).to(torch.int32).contiguous()
                symbols = gc.quantize(y_slice, "symbols", mu).to(torch.int32).contiguous()
                y_hat = gc.dequantize(symbols, mu)
                ready = torch.cuda.Event(); ready.record(trt_stream)
                symbols.record_stream(ans_stream); indexes.record_stream(ans_stream)
                with torch.cuda.stream(ans_stream):
                    ans_stream.wait_event(ready)
                    sessions.y[bi][si].launch(symbols, indexes)
                pending.append((sessions.y[bi][si], symbols, indexes))
                lrp_in = torch.cat([engine._cast(support, engine.lrp_input_dtypes[si]), engine._cast(y_hat, engine.lrp_input_dtypes[si])], 1)
                lrp = 0.5 * torch.tanh(engine.runners[f"lrp_transforms_{si}"](lrp_in))
                hats.append((y_hat + lrp.to(y_hat.dtype)).contiguous())
                batch_states.append({"size_hw": tuple(y.shape[-2:])})
            states.append(batch_states)

    # Every encode has been launched before any synchronization/finalization.
    finalized = [entry[0].finalize() for entry in pending]
    out: List[Dict[str, Any]] = []
    cursor = 0
    for bi in range(len(blocks)):
        z_string = finalized[cursor]; cursor += 1
        y_strings = finalized[cursor:cursor + engine.num_slices]; cursor += engine.num_slices
        out.append({"y": {"strings": y_strings, "state": states[bi]},
                    "z": {"strings": z_string, "state": {"size_hw": sessions.z_shape[-2:]}}})
    torch.cuda.synchronize()
    return out


@torch.no_grad()
def decompress_dcae_pipelined_session(
    engine: DCAEEngine,
    sessions: DCAEDecodeSessions,
    packs: Sequence[Dict[str, Any]],
) -> List[torch.Tensor]:
    """Decode with ANS on the default stream and inference on a private stream."""
    _check_engine(engine)
    if len(packs) != len(sessions.z) or len(sessions.y) != len(packs):
        raise ValueError("one independent decoder session set is required per pack")
    if not packs:
        return []
    ans_stream = torch.cuda.default_stream()
    trt_stream = torch.cuda.Stream()
    outputs: List[torch.Tensor] = []
    eb, gc = engine.codec.eb, engine.codec.gaussian_conditional
    zc, zh, zw = sessions.z_shape
    yc, yh, yw = sessions.y_shape
    ycs = yc // engine.num_slices

    # z indexes are independent of inference.  Submit every batch up front so
    # its ANS work may overlap inference for an earlier batch.
    for bi, pack in enumerate(packs):
        zs = pack["z"]["strings"]
        zb, _, zheader, _, _ = _warp_meta(zs, f"packs[{bi}].z.strings")
        if zb != sessions.batch_size:
            raise ValueError(f"packs[{bi}] batch size does not match decoder sessions")
        with torch.cuda.stream(ans_stream):
            zidx = eb._build_indexes(
                (zb, zc, zh, zw), device=zs.packed.device
            ).reshape(zb, -1).to(torch.int32).contiguous()
            zs.packed.record_stream(ans_stream)
            zs.max_rounds_u32.record_stream(ans_stream)
            zidx.record_stream(ans_stream)
            sessions.z[bi].launch(zs.packed, zs.max_rounds_u32, zheader, zidx)

    for bi, pack in enumerate(packs):
        zb = sessions.batch_size
        # finalize is deliberately on the default stream.  Record the produced
        # symbols there before transferring ownership to the inference stream.
        with torch.cuda.stream(ans_stream):
            zsym = sessions.z[bi].finalize().reshape(zb, zc, zh, zw)
            z_ready = torch.cuda.Event()
            z_ready.record(ans_stream)
            zsym.record_stream(trt_stream)

        with torch.cuda.stream(trt_stream):
            trt_stream.wait_event(z_ready)
            med = eb._extend_ndims(eb._get_medians().detach(), 2).expand(zb, zc, zh, zw)
            z_hat = eb.dequantize(zsym, med).to(engine.codec_input_dtype).contiguous()
            scales = engine._ensure_cuda_contiguous(engine.runners["h_z_s1"](engine._cast(z_hat, engine.h_z_s1_input_dtype)))
            means = engine._ensure_cuda_contiguous(engine.runners["h_z_s2"](engine._cast(z_hat, engine.h_z_s2_input_dtype)))
            dt = engine._repeat_dt(zb, z_hat.device, means.dtype)
            hats: List[torch.Tensor] = []

        y_strings = pack["y"]["strings"]
        for si in range(engine.num_slices):
            with torch.cuda.stream(trt_stream):
                support_slices = hats if engine.max_support_slices < 0 else hats[:engine.max_support_slices]
                query = torch.cat([scales, means] + support_slices, 1)
                info = engine.runners[f"dt_cross_attention_{si}"](engine._cast(query, engine.dt_ca_input_dtypes[si]), dt)
                support = torch.cat([query, info], 1)
                mu = engine.runners[f"cc_mean_{si}"](engine._cast(support, engine.cc_mean_input_dtypes[si]))[:, :, :yh, :yw].float()
                scale = engine.runners[f"cc_scale_{si}"](engine._cast(support, engine.cc_scale_input_dtypes[si]))[:, :, :yh, :yw].float()
                indexes = gc.build_indexes(scale).reshape(zb, -1).to(torch.int32).contiguous()
                indexes_ready = torch.cuda.Event()
                indexes_ready.record(trt_stream)
                indexes.record_stream(ans_stream)

            tight = y_strings[si]
            _, _, header, _, _ = _warp_meta(tight, f"packs[{bi}].y.strings[{si}]")
            with torch.cuda.stream(ans_stream):
                ans_stream.wait_event(indexes_ready)
                tight.packed.record_stream(ans_stream)
                tight.max_rounds_u32.record_stream(ans_stream)
                sessions.y[bi][si].launch(tight.packed, tight.max_rounds_u32, header, indexes)
                symbols = sessions.y[bi][si].finalize().reshape(zb, ycs, yh, yw)
                symbols_ready = torch.cuda.Event()
                symbols_ready.record(ans_stream)
                symbols.record_stream(trt_stream)

            with torch.cuda.stream(trt_stream):
                trt_stream.wait_event(symbols_ready)
                y_hat = gc.dequantize(symbols, mu)
                lrp_in = torch.cat([engine._cast(support, engine.lrp_input_dtypes[si]), engine._cast(y_hat, engine.lrp_input_dtypes[si])], 1)
                lrp = 0.5 * torch.tanh(engine.runners[f"lrp_transforms_{si}"](lrp_in))
                hats.append((y_hat + lrp.to(y_hat.dtype)).contiguous())

        with torch.cuda.stream(trt_stream):
            y_hat = torch.cat(hats, 1).to(engine.gs_input_dtype).contiguous()
            x_hat = engine.runners["gs"](y_hat)
            if not isinstance(x_hat, torch.Tensor):
                raise TypeError("gs runner must return torch.Tensor")
            outputs.append(x_hat.clamp_(0, 1).contiguous())

    # Returned tensors must be usable immediately from any caller stream.
    trt_stream.synchronize()
    return outputs
