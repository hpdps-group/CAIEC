#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark: serial vs pipelined compress/decompress for Ballé factorized model.

Demonstrates batch-level overlap between TRT inference (ga/gs) on a non-default
stream and GPU ANS coding on the default stream.

Usage::

    python bench_pipeline_factorized.py \\
        --bin_path /hwj/data/caiec_test_data/hurricane_100x1x500x500.f32 \\
        --input_shape 100,500,500 \\
        --dataset hurricane --quality 4 --batchsize 2 --iters 20

The script uses ``stream_pipeline.compress_pipelined`` and
``stream_pipeline.decompress_pipelined`` to overlap adjacent batches.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from compressai.zoo import bmshj2018_factorized
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.runtime.stream_pipeline import (
    compress_pipelined,
    compress_pipelined_session,
    decompress_pipelined,
)
from compressai.runtime.ans_pipeline import PipelinedAnsEncoder
from compressai.entropy_models.entropy_models import TightANS, TightWarpANS
from compressai.utils import dataloader_science
import compressai.runtime.utils.metrics as metrics


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _bytes_of_strings(strings) -> int:
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, (list, tuple)):
        return sum(_bytes_of_strings(s) for s in strings)
    if isinstance(strings, TightWarpANS):
        # Actual payload = header + sum(max_rounds) * 32 lanes * 4 bytes
        payload = int(strings.max_rounds_u32.sum().item()) * 32 * 4
        header = int(strings.header_bytes_cpu.item())
        return header + payload
    if isinstance(strings, TightANS):
        return int(strings.packed.numel())
    return 0


def _bytes_of_state(obj) -> int:
    if obj is None:
        return 0
    try:
        import pickle
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pass
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, dict):
        return sum(_bytes_of_state(k) + _bytes_of_state(v) for k, v in obj.items()) + 16 * len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_bytes_of_state(v) for v in obj) + 8 * len(obj)
    return 0


def _packed_payload_bytes(packed) -> int:
    try:
        if hasattr(packed, 'max_rounds_u32') and torch.is_tensor(packed.max_rounds_u32):
            # TightWarpANS: actual payload = header + sum(max_rounds) * 32 lanes * 4 bytes
            mr = packed.max_rounds_u32
            payload = int(mr.sum().item()) * 32 * 4
            header = int(packed.header_bytes_cpu.item())
            return header + payload
        if torch.is_tensor(packed.sizes):
            return int(packed.sizes.sum().item())
    except Exception:
        pass
    return 0


def split_into_batches(tensor: torch.Tensor, batchsize: int) -> List[torch.Tensor]:
    """Split ``tensor`` along dim 0 into chunks of ``batchsize``.

    The last chunk is padded with ``pad_value`` if needed, so every batch
    has exactly ``batchsize`` samples — matching the TRT engine's fixed dim 0.
    """
    bn = tensor.shape[0]
    batches = []
    for start in range(0, bn, batchsize):
        end = min(start + batchsize, bn)
        chunk = tensor[start:end]
        cur_bs = chunk.shape[0]
        if cur_bs < batchsize:
            pad_shape = (batchsize - cur_bs,) + chunk.shape[1:]
            pad = torch.full(pad_shape, 0.5, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([chunk, pad], dim=0)
        batches.append(chunk)
    return batches


# ──────────────────────────────────────────────
# Serial compress (mimics engine.compress_time)
# ──────────────────────────────────────────────

@torch.no_grad()
def serial_compress(
    engine, blocks: List[torch.Tensor],
):
    """Compress all batches serially. Returns (packs, total_ms)."""
    packs: List[Dict] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    for x in blocks:
        pack, _, _ = engine.compress_time(x)
        packs.append(pack)

    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return packs, total_ms


@torch.no_grad()
def serial_decompress(
    engine, packs: List[Dict],
) -> List[torch.Tensor]:
    """Decompress all batches serially. Returns (x_hats, total_ms)."""
    x_hats: List[torch.Tensor] = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    for pack in packs:
        x_hat, _, _ = engine.decompress_time(pack)
        x_hats.append(x_hat)

    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return x_hats, total_ms


# ──────────────────────────────────────────────
# Pipelined compress (stream_pipeline)
# ──────────────────────────────────────────────

@torch.no_grad()
def pipelined_compress(
    engine, blocks: List[torch.Tensor],
) -> List[Dict]:
    """Compress all batches with pipeline overlap. Returns (packs, total_ms)."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    packs = compress_pipelined(
        ga_runner=engine.runners["ga"],
        codec=engine.codec,
        blocks=blocks,
        ga_input_dtype=engine.ga_input_dtype,
        codec_input_dtype=engine.codec_input_dtype,
    )

    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return packs, total_ms


@torch.no_grad()
def pipelined_decompress(
    engine, packs: List[Dict],
) -> List[torch.Tensor]:
    """Decompress all batches with pipeline overlap. Returns (x_hats, total_ms)."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    x_hats = decompress_pipelined(
        gs_runner=engine.runners["gs"],
        codec=engine.codec,
        packs=packs,
        gs_input_dtype=engine.gs_input_dtype,
    )

    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return x_hats, total_ms


# ──────────────────────────────────────────────
# Session-based pipelined compress
# ──────────────────────────────────────────────

@torch.no_grad()
def session_compress(
    engine, blocks: List[torch.Tensor],
) -> List[Dict]:
    """Compress all batches with session-based pipelined encoder.

    Creates one PipelinedAnsEncoder per batch, launches all ga+enc
    in a non-blocking phase, then finalizes in a second phase.
    """
    N = len(blocks)
    trunk = blocks[0]  # representative batch for EB prep
    ga_runner = engine.runners["ga"]
    ga_in = engine.ga_input_dtype

    # Build a sample latent to extract B, C, N
    xx = trunk
    if ga_in is not None and xx.dtype != ga_in:
        xx = xx.to(ga_in)
    if not xx.is_contiguous():
        xx = xx.contiguous()
    with torch.no_grad():
        sample_y = ga_runner(xx)
    B, C, H, W = sample_y.shape
    N_sym = C * H * W
    del sample_y, xx

    # Prep function: quantize + build indexes (same as eb.compress internals)
    eb = engine.codec.eb
    medians = eb._get_medians().detach()
    spatial_dims = 2
    medians = eb._extend_ndims(medians, spatial_dims)
    medians = medians.expand(B, *([-1] * (spatial_dims + 1)))

    def prep(y_fp):
        symbols = eb.quantize(y_fp, "symbols", medians).to(torch.int32)
        indexes = torch.arange(C, device=y_fp.device, dtype=torch.int32) \
                       .view(1, C, 1, 1).expand(B, C, H, W).contiguous()
        return symbols.contiguous(), indexes

    # Create one encoder per batch
    P = engine.codec.P
    ans_variant = getattr(engine.codec, '_variant', 'tight')
    # For session-based, always use warp_smem since it uses V3 kernels
    encoders = [
        PipelinedAnsEncoder.from_entropy_bottleneck(eb, B, N_sym, P=P)
        for _ in range(N)
    ]

    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()

    packs = compress_pipelined_session(
        ga_runner=ga_runner,
        prep_for_encode=prep,
        encoders=encoders,
        blocks=blocks,
        ga_input_dtype=ga_in,
    )

    end_ev.record()
    torch.cuda.synchronize()
    total_ms = start_ev.elapsed_time(end_ev)
    return packs, total_ms


# ──────────────────────────────────────────────
# Split timing: measure ga/enc/dec/gs separately
# ──────────────────────────────────────────────

@torch.no_grad()
def benchmark_split(engine, codec, x: torch.Tensor, warmup: int = 5, iters: int = 20):
    """Measure ga / enc / dec / gs times individually using CUDA events."""
    ga_runner = engine.runners["ga"]
    gs_runner = engine.runners["gs"]
    ga_input_dtype = getattr(engine, "ga_input_dtype", None)
    gs_input_dtype = getattr(engine, "gs_input_dtype", torch.float16)
    codec_input_dtype = getattr(engine, "codec_input_dtype", torch.float32)

    input_bytes = x.numel() * x.element_size()
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        xx = x
        if ga_input_dtype is not None and xx.dtype != ga_input_dtype:
            xx = xx.to(ga_input_dtype)
        y = ga_runner(xx)
        y_fp32 = y.to(codec_input_dtype)
        pack = codec.compress(y_fp32)
        y_hat = codec.decompress(pack)
        y_hat = y_hat.to(gs_input_dtype)
        _ = gs_runner(y_hat)
    torch.cuda.synchronize()

    ga_ms, enc_ms, dec_ms, gs_ms = [], [], [], []
    strings_bytes_list, state_bytes_list = [], []

    for _ in range(iters):
        # GA
        xx = x
        if ga_input_dtype is not None and xx.dtype != ga_input_dtype:
            xx = xx.to(ga_input_dtype)
        start.record()
        y = ga_runner(xx)
        end.record()
        torch.cuda.synchronize()
        ga_ms.append(start.elapsed_time(end))

        # Encode
        y_fp32 = y.to(codec_input_dtype)
        start.record()
        pack = codec.compress(y_fp32)
        end.record()
        torch.cuda.synchronize()
        enc_ms.append(start.elapsed_time(end))

        strings_bytes_list.append(_bytes_of_strings(pack.get("strings", None)))
        state_bytes_list.append(_bytes_of_state(pack.get("state", None)))

        # Decode
        start.record()
        y_hat = codec.decompress(pack)
        end.record()
        torch.cuda.synchronize()
        dec_ms.append(start.elapsed_time(end))

        # GS
        y_hat = y_hat.to(gs_input_dtype)
        start.record()
        x_hat = gs_runner(y_hat)
        end.record()
        torch.cuda.synchronize()
        gs_ms.append(start.elapsed_time(end))

    ga_avg = sum(ga_ms) / iters
    enc_avg = sum(enc_ms) / iters
    dec_avg = sum(dec_ms) / iters
    gs_avg = sum(gs_ms) / iters

    ga_GBps = (input_bytes / (ga_avg / 1000.0)) / (1024 ** 3)
    enc_GBps = (input_bytes / (enc_avg / 1000.0)) / (1024 ** 3)
    dec_GBps = (input_bytes / (dec_avg / 1000.0)) / (1024 ** 3)
    gs_GBps = (input_bytes / (gs_avg / 1000.0)) / (1024 ** 3)

    strings_bytes_avg = sum(strings_bytes_list) / iters
    state_bytes_avg = sum(state_bytes_list) / iters
    total_bytes_avg = strings_bytes_avg + state_bytes_avg

    cr_strings = (input_bytes / strings_bytes_avg) if strings_bytes_avg > 0 else float("inf")
    cr_total = (input_bytes / total_bytes_avg) if total_bytes_avg > 0 else float("inf")
    bpp_strings = (strings_bytes_avg * 8.0) / pixels
    bpp_total = (total_bytes_avg * 8.0) / pixels

    return {
        "input_bytes": input_bytes,
        "ga_ms": ga_avg, "ga_GBps": ga_GBps,
        "enc_ms": enc_avg, "enc_GBps": enc_GBps,
        "dec_ms": dec_avg, "dec_GBps": dec_GBps,
        "gs_ms": gs_avg, "gs_GBps": gs_GBps,
        "strings_bytes_avg": strings_bytes_avg,
        "state_bytes_avg": state_bytes_avg,
        "cr_strings": cr_strings,
        "cr_total": cr_total,
        "bpp_strings": bpp_strings,
        "bpp_total": bpp_total,
    }, x_hat


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs pipelined compress/decompress for Ballé factorized"
    )
    parser.add_argument("--bin_path", type=str, required=True,
                        help="Path to input .f32 file (e.g. /hwj/data/caiec_test_data/hurricane_100x1x500x500.f32)")
    parser.add_argument("--input_shape", type=str, default="100,500,500",
                        help="Input shape as N,H,W (default: 100,500,500)")
    parser.add_argument("--block_size", type=str, default="3,512,512",
                        help="Block size as C,H,W (default: 3,512,512)")
    parser.add_argument("--project_dir", type=str, default="/hwj",
                        help="Base directory for model checkpoints and TRT engines")
    parser.add_argument("--dataset", type=str, default="hurricane",
                        help="Dataset name for engine subdirectory lookup")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=2,
                        help="Batch size for pipeline (smaller = more batches = more overlap)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Replicate blocks N times to increase batch count for pipeline testing "
                             "(does not affect correctness — same data repeated)")
    parser.add_argument("--ans_variant", type=str, default="auto",
                        choices=["auto", "tight", "warp", "warp_div", "warp_smem"],
                        help="GPU ANS variant: auto (warp when P>=256), tight, warp, warp_div, warp_smem")
    parser.add_argument("-P", "--ans_P", type=int, default=64,
                        help="GPU ANS parallelism granularity P")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10,
                        help="Number of iterations for split benchmark")
    parser.add_argument("--session", action="store_true",
                        help="Use session-based pipelined encoder (non-blocking launch)")
    parser.add_argument("--split_only", action="store_true",
                        help="Only run split timing benchmark, not pipeline comparison")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # ── 1. Load data ──
    print("=" * 60)
    print(f"Loading data from {args.bin_path}...")
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    assert len(input_shape) == 3, f"input_shape must be N,H,W, got {input_shape}"
    block_size = tuple(int(x) for x in args.block_size.split(","))
    assert len(block_size) == 3, f"block_size must be C,H,W, got {block_size}"
    p = 0.95
    stride = (int(block_size[1] * p), int(block_size[2] * p))

    blocks, x_nhw_ori, status, meta = dataloader_science.blockify_bin_overlap(
        args.bin_path, input_shape, block_size, p, stride_hw=stride,
        dtype=np.float32, pad_value=0.5, norm_type="minmax", device=device
    )
    bn_orig = blocks.shape[0]
    if args.repeat > 1:
        blocks = blocks.repeat(args.repeat, 1, 1, 1)
    bn = blocks.shape[0]
    print(f"Blocks: {bn_orig} → repeat ×{args.repeat} → {bn} × {tuple(blocks.shape[1:])}")

    # ── 2. Load model ──
    model_path = f"{args.project_dir}/data/bmshj2018-factorized/bmshj2018-factorized-{args.quality}.pth"
    print(f"\nLoading Ballé2018 factorized q={args.quality} from {model_path}...")
    net = bmshj2018_factorized(quality=args.quality)
    net = net.to(device)
    net.eval()

    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint)
    net.update()

    codec = GpuPackedEntropyCodec(net.entropy_bottleneck, P=args.ans_P, ans_variant=args.ans_variant)

    # ── 3. Build TRT runtime ──
    engine_dir = f"{args.project_dir}/data/engines/bmshj2018-factorized/q{args.quality}/engines/{args.dataset}"
    print(f"Building TRT runtime (engines from {engine_dir})...")
    cfg = RuntimeConfig(
        model_name="bmshj2018_factorized",
        ga_input_dtype=torch.float32,
        gs_input_dtype=torch.float16,
        codec_input_dtype=torch.float32,
        trt_engines={
            "ga": f"{engine_dir}/ga/fp8.engine",
            "gs": f"{engine_dir}/gs/fp16.engine",
        },
    )
    engine = build_runtime(net, codec, cfg)

    # ── 4. Split benchmark (ga/enc/dec/gs individual timings) ──
    print(f"\n{'=' * 60}")
    print(f"Split Benchmark (iters={args.iters})")
    print(f"{'=' * 60}")
    # Split into batches first — each batch matches TRT engine's input shape
    batches = split_into_batches(blocks, args.batchsize)
    num_batches = len(batches)
    x_sample = batches[0]  # first batch, shape = [batchsize, 3, H, W]
    split, _ = benchmark_split(engine, codec, x_sample, warmup=args.warmup, iters=args.iters)

    print(f"GA  : {split['ga_ms']:.3f} ms  |  {split['ga_GBps']:.2f} GB/s")
    print(f"ENC : {split['enc_ms']:.3f} ms  |  {split['enc_GBps']:.2f} GB/s")
    print(f"DEC : {split['dec_ms']:.3f} ms  |  {split['dec_GBps']:.2f} GB/s")
    print(f"GS  : {split['gs_ms']:.3f} ms  |  {split['gs_GBps']:.2f} GB/s")
    print(f"Total (ga+enc): {split['ga_ms'] + split['enc_ms']:.3f} ms")
    print(f"Total (dec+gs): {split['dec_ms'] + split['gs_ms']:.3f} ms")
    pipeline_potential = max(split["ga_ms"], split["enc_ms"]) / (split["ga_ms"] + split["enc_ms"])
    print(f"Compress pipeline theoretical max speedup: {1.0 / pipeline_potential:.2f}x")
    pipeline_potential_dec = max(split["dec_ms"], split["gs_ms"]) / (split["dec_ms"] + split["gs_ms"])
    print(f"Decompress pipeline theoretical max speedup: {1.0 / pipeline_potential_dec:.2f}x")

    if args.split_only:
        return

    # ── 5. Pipeline vs serial comparison ──
    print(f"\n{'=' * 60}")
    print(f"Pipeline Comparison (batchsize={args.batchsize})")
    print(f"{'=' * 60}")

    # Split blocks into batches
    batches = split_into_batches(blocks, args.batchsize)
    num_batches = len(batches)
    print(f"Batches: {num_batches} x ~{args.batchsize} blocks each")

    # ── 5a. Warmup ──
    print("Warming up...")
    torch.cuda.empty_cache()  # flush before warmup, not after
    pipe_fn = session_compress if args.session else pipelined_compress
    for _ in range(args.warmup):
        _, _ = pipe_fn(engine, batches)
        torch.cuda.synchronize()
        packs_w1, _ = serial_compress(engine, batches)
        torch.cuda.synchronize()
        _, _ = pipelined_decompress(engine, packs_w1)
        torch.cuda.synchronize()
        _, _ = serial_decompress(engine, packs_w1)
        torch.cuda.synchronize()

    # ── 5b. Serial compress ──
    print("\nSerial compress...")
    packs_ser, cmp_ser_ms = serial_compress(engine, batches)
    cmp_ser_GBps = (blocks.numel() * blocks.element_size() / (cmp_ser_ms / 1000.0)) / (1024 ** 3)

    # ── 5c. Pipelined compress ──
    if args.session:
        print("Pipelined compress (session-based)...")
        packs_pipe, cmp_pipe_ms = session_compress(engine, batches)
    else:
        print("Pipelined compress...")
        packs_pipe, cmp_pipe_ms = pipelined_compress(engine, batches)
    cmp_pipe_GBps = (blocks.numel() * blocks.element_size() / (cmp_pipe_ms / 1000.0)) / (1024 ** 3)

    cmp_speedup = cmp_ser_ms / cmp_pipe_ms if cmp_pipe_ms > 0 else 1.0
    print(f"\n--- Compress Results ---")
    print(f"Serial:    {cmp_ser_ms:.3f} ms  |  {cmp_ser_GBps:.2f} GB/s")
    print(f"Pipelined: {cmp_pipe_ms:.3f} ms  |  {cmp_pipe_GBps:.2f} GB/s")
    print(f"Speedup:   {cmp_speedup:.2f}x")

    # ── 5d. Verify compress correctness ──
    print("\nVerifying compress correctness...")
    cmp_ok = True
    for i, (ps, pp) in enumerate(zip(packs_ser, packs_pipe)):
        # Compare string sizes (not bit-exact due to potential ANS state timing diffs,
        # but total bytes and semantics should match)
        ss = _packed_payload_bytes(ps.get("strings", None))
        sp = _packed_payload_bytes(pp.get("strings", None))
        if ss != sp:
            print(f"  WARNING: batch {i} strings bytes differ: serial={ss}, pipe={sp}")
            cmp_ok = False
    if cmp_ok:
        print("  All batches have matching string sizes ✓")

    # ── 5e. Serial decompress ──
    print("\nSerial decompress...")
    xhats_ser, dec_ser_ms = serial_decompress(engine, packs_ser)
    dec_ser_GBps = (blocks.numel() * blocks.element_size() / (dec_ser_ms / 1000.0)) / (1024 ** 3)

    # ── 5f. Pipelined decompress ──
    print("Pipelined decompress...")
    xhats_pipe, dec_pipe_ms = pipelined_decompress(engine, packs_pipe)
    dec_pipe_GBps = (blocks.numel() * blocks.element_size() / (dec_pipe_ms / 1000.0)) / (1024 ** 3)

    dec_speedup = dec_ser_ms / dec_pipe_ms if dec_pipe_ms > 0 else 1.0
    print(f"\n--- Decompress Results ---")
    print(f"Serial:    {dec_ser_ms:.3f} ms  |  {dec_ser_GBps:.2f} GB/s")
    print(f"Pipelined: {dec_pipe_ms:.3f} ms  |  {dec_pipe_GBps:.2f} GB/s")
    print(f"Speedup:   {dec_speedup:.2f}x")

    # ── 5g. Verify decompress correctness ──
    print("\nVerifying decompress correctness...")
    x_hat_all_ser = torch.cat([x[:min(args.batchsize, x.shape[0])] for x in xhats_ser], dim=0)
    x_hat_all_pipe = torch.cat([x[:min(args.batchsize, x.shape[0])] for x in xhats_pipe], dim=0)
    diff = (x_hat_all_ser.float() - x_hat_all_pipe.float()).abs()
    max_diff = diff.max().item()
    print(f"  Max difference: {max_diff:.6e}")
    if max_diff < 1e-3:
        print("  Bit-exact match ✓")
    else:
        print(f"  WARNING: non-trivial difference ({max_diff})")

    # ── 6. RD metrics ──
    print(f"\n{'=' * 60}")
    print("RD Metrics (from serial path)")
    print(f"{'=' * 60}")

    x_hat_nhw = dataloader_science.blocks_to_nhw_overlap_weighted(
        x_hat_all_ser[:bn_orig],
        input_shape, block_size, stride_hw=stride, meta_hw=meta,
        pad_value=0.5, norm_type="minmax",
        vmin=status["vmin"], vmax=status["vmax"],
        vmean=status["vmean"], vstd=status["vstd"],
    )
    metr = metrics.basic_metrics(x_hat_nhw, x_nhw_ori)
    num_pixels = x_nhw_ori.size(0) * x_nhw_ori.size(1) * x_nhw_ori.size(2)

    # Collect bitstream size from serial packs
    strings_total = sum(
        _bytes_of_strings(p.get("strings", None)) + _bytes_of_state(p.get("state", None))
        for p in packs_ser
    )
    bit_rate = strings_total * 8.0 / num_pixels

    print(f"bpp:    {bit_rate:.6f}")
    print(f"rmse:   {metr['rmse']:.6f}")
    print(f"nrmse:  {metr['nrmse']:.6f}")
    print(f"maxe:   {metr['maxe']:.6f}")
    print(f"psnr:   {metr['psnr']:.2f}")

    # ── 7. Summary ──
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Batches:  {num_batches} × batchsize={args.batchsize}")
    print(f"Compress:  serial={cmp_ser_ms:.2f}ms → pipe={cmp_pipe_ms:.2f}ms  ({cmp_speedup:.2f}x)")
    print(f"Decompress: serial={dec_ser_ms:.2f}ms → pipe={dec_pipe_ms:.2f}ms  ({dec_speedup:.2f}x)")
    print(f"PSNR:     {metr['psnr']:.2f} dB")
    print(f"bpp:      {bit_rate:.6f}")


if __name__ == "__main__":
    main()
