#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_cnn_trt_fp16_gpu_packed.py

End-to-end TRT (fp16) benchmark for bmshj2018_factorized where the codec is a GPU PackedANS
EntropyBottleneck (your GPU rANS path), compatible with SimpleCnnAdapter:

  pack = codec.compress(y_fp32) -> {"strings": PackedANS, "state": {"size_hw": (H,W)}}
  y_hat = codec.decompress(pack["strings"], pack["state"]) -> Tensor

Important for TensorRT sync:
- Run the entire encode/decode pipeline on a NON-default CUDA stream.
  TensorRTModule uses torch.cuda.current_stream().cuda_stream, so we simply set a
  non-default current stream for the benchmark loop.
- Timing events are recorded on that same stream.

Bytes/CR/bpp:
- strings_bytes is computed from PackedANS payload: sum(packed.sizes)
- state_bytes uses pickle size of the small state dict (size_hw)
- Optionally include packed container overhead via --packed_overhead

Usage example:
  python runtime_cnn_trt_fp16_gpu_packed.py --P 64
  python runtime_cnn_trt_fp16_gpu_packed.py --P 128 --iters 100 --warmup 20
"""

import argparse
import pickle
import numpy as np
import torch

from compressai.zoo import bmshj2018_factorized
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig

import gc

# --------------------------
# Metrics
# --------------------------
def metrics(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    x_f = x.float()
    y_f = x_hat.float()

    diff = y_f - x_f
    mse = torch.mean(diff * diff)
    rmse = torch.sqrt(mse)

    maxe = torch.max(torch.abs(diff))

    range_ = torch.max(x_f) - torch.min(x_f)
    nrmse = rmse / range_

    psnr = 20.0 * torch.log10(range_ / (2.0 * rmse + eps))
    return rmse.item(), nrmse.item(), maxe.item(), psnr.item()


# --------------------------
# Bytes helpers
# --------------------------
def _bytes_of_state(obj):
    if obj is None:
        return 0
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pass

    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, int):
        return 8
    if isinstance(obj, float):
        return 8
    if isinstance(obj, dict):
        overhead = 16 * len(obj)
        return overhead + sum(_bytes_of_state(k) + _bytes_of_state(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        overhead = 8 * len(obj)
        return overhead + sum(_bytes_of_state(v) for v in obj)
    return 0


def packed_payload_bytes(packed) -> int:
    # PackedANS is typically a torch.classes object; treat it duck-typed.
    try:
        sizes = packed.sizes
        if torch.is_tensor(sizes):
            return int(sizes.sum().item())
    except Exception:
        pass
    return 0


def packed_container_overhead_bytes(packed) -> int:
    # Optional: sizes tensor bytes + stride estimate
    sizes_bytes = 0
    try:
        sizes = packed.sizes
        if torch.is_tensor(sizes):
            sizes_bytes = int(sizes.numel() * sizes.element_size())
    except Exception:
        sizes_bytes = 0
    stride_bytes = 8
    return sizes_bytes + stride_bytes


# --------------------------
# GPU Packed EB codec (SimpleCnnAdapter-compatible)
# --------------------------
class GpuPackedEntropyBottleneckCodec:
    """
    Uses net.entropy_bottleneck with your GPU rANS packed API.

    You said your working validation code uses:
      eb.use_gpu_ans = True
      eb.gpu_ans_parallelism = P
      packed = eb.compress(y)
      y_hat = eb.decompress(packed, size_hw)

    We keep that behavior here, but adapt to SimpleCnnAdapter contract.
    """
    def __init__(self, net, P: int = 64, packed_overhead: bool = False):
        self.net = net
        self.eb = net.entropy_bottleneck
        self.P = int(P)
        self.packed_overhead = bool(packed_overhead)

    def _apply_flags(self):
        # Prefer packed path if you expose it
        if hasattr(self.eb, "use_gpu_ans_packed"):
            self.eb.use_gpu_ans_packed = True

        # Your current validation uses use_gpu_ans=True; keep it.
        if hasattr(self.eb, "use_gpu_ans"):
            self.eb.use_gpu_ans = True

        if hasattr(self.eb, "gpu_ans_parallelism"):
            self.eb.gpu_ans_parallelism = int(self.P)

    @torch.no_grad()
    def compress(self, y: torch.Tensor):
        # TRT 的 ga_runner 在 bench_stream（当前 stream）上产出 y
        trt_stream = torch.cuda.current_stream(device=y.device)
        default_stream = torch.cuda.default_stream(device=y.device)

        # 1) default stream 等待 TRT stream：确保 y 写完
        default_stream.wait_stream(trt_stream)

        # 2) 在 default stream 上做 EB compress（如果 EB 内部强制 default stream，这也不坏）
        with torch.cuda.stream(default_stream):
            self.eb.use_gpu_ans = True
            self.eb.gpu_ans_parallelism = int(self.P)
            if not y.is_contiguous():
                y = y.contiguous()
            packed = self.eb.compress(y)
            state = {"size_hw": tuple(y.shape[-2:])}

        # 3) TRT stream 等待 default stream：确保 packed 写完
        trt_stream.wait_stream(default_stream)

        return {"strings": packed, "state": state}


    @torch.no_grad()
    def decompress(self, strings, state):
        dev = strings.arena.device if hasattr(strings, "arena") else None
        if dev is None:
            dev = torch.device("cuda")

        trt_stream = torch.cuda.current_stream(device=dev)
        default_stream = torch.cuda.default_stream(device=dev)

        # 1) default 等待 TRT：确保 strings/state 都准备好
        default_stream.wait_stream(trt_stream)

        with torch.cuda.stream(default_stream):
            self.eb.use_gpu_ans = True
            self.eb.gpu_ans_parallelism = int(self.P)
            size_hw = state["size_hw"]
            y_hat = self.eb.decompress(strings, size_hw)
            if not y_hat.is_contiguous():
                y_hat = y_hat.contiguous()

        # 2) TRT 等待 default：确保 y_hat 写完，给 gs 用
        trt_stream.wait_stream(default_stream)

        return y_hat


    def pack_bytes(self, pack: dict):
        strings = pack.get("strings", None)
        state = pack.get("state", None)

        state_bytes = _bytes_of_state(state)
        payload = packed_payload_bytes(strings)
        if payload > 0:
            if self.packed_overhead:
                payload += packed_container_overhead_bytes(strings)
            return float(payload), float(state_bytes)

        # Fallback (shouldn't happen for packed path)
        return 0.0, float(state_bytes)


# --------------------------
# Benchmarks (run on non-default stream)
# --------------------------
def benchmark_engine(engine, codec_for_bytes, x: torch.Tensor, warmup=10, iters=50, stream=None):
    assert x.is_cuda, "x must be on CUDA"
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W
    input_bytes = x.numel() * x.element_size()

    stream = stream if stream is not None else torch.cuda.Stream(device=x.device)

    # We record events on this stream.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            pack = engine.compress(x)
            _ = engine.decompress(pack)
        # Ensure warmup done before timing
        end.record()
    stream.synchronize()

    enc_times = []
    dec_times = []
    strings_bytes_list = []
    state_bytes_list = []
    rmse_list, nrmse_list, maxe_list, psnr_list = [], [], [], []

    with torch.cuda.stream(stream):
        for _ in range(iters):
            # Encode timing
            start.record()
            pack = engine.compress(x)
            end.record()
            # Sync only this stream to get accurate time
            end.synchronize()
            enc_times.append(start.elapsed_time(end))

            s_b, st_b = codec_for_bytes.pack_bytes(pack)
            strings_bytes_list.append(s_b)
            state_bytes_list.append(st_b)

            # Decode timing
            start.record()
            x_hat = engine.decompress(pack)
            end.record()
            end.synchronize()
            dec_times.append(start.elapsed_time(end))

            rmse, nrmse, maxe, psnr = metrics(x_hat, x)
            rmse_list.append(rmse)
            nrmse_list.append(nrmse)
            maxe_list.append(maxe)
            psnr_list.append(psnr)

    # Final sync
    stream.synchronize()

    enc_ms_avg = sum(enc_times) / len(enc_times)
    dec_ms_avg = sum(dec_times) / len(dec_times)

    strings_bytes_avg = sum(strings_bytes_list) / len(strings_bytes_list)
    state_bytes_avg = sum(state_bytes_list) / len(state_bytes_list)
    total_bytes_avg = strings_bytes_avg + state_bytes_avg

    enc_s = enc_ms_avg / 1000.0
    dec_s = dec_ms_avg / 1000.0
    enc_GBps = (input_bytes / enc_s) / (1024**3)
    dec_GBps = (input_bytes / dec_s) / (1024**3)

    cr_strings = (input_bytes / strings_bytes_avg) if strings_bytes_avg > 0 else float("inf")
    cr_total = (input_bytes / total_bytes_avg) if total_bytes_avg > 0 else float("inf")

    bpp_strings = (strings_bytes_avg * 8.0) / pixels
    bpp_total = (total_bytes_avg * 8.0) / pixels

    rmse_avg = sum(rmse_list) / len(rmse_list)
    nrmse_avg = sum(nrmse_list) / len(nrmse_list)
    maxe_avg = sum(maxe_list) / len(maxe_list)
    psnr_avg = sum(psnr_list) / len(psnr_list)

    return {
        "input_bytes": input_bytes,
        "enc_ms_avg": enc_ms_avg,
        "dec_ms_avg": dec_ms_avg,
        "enc_GBps": enc_GBps,
        "dec_GBps": dec_GBps,
        "strings_bytes_avg": strings_bytes_avg,
        "state_bytes_avg": state_bytes_avg,
        "cr_strings": cr_strings,
        "cr_total": cr_total,
        "bpp_strings": bpp_strings,
        "bpp_total": bpp_total,
        "rmse": rmse_avg,
        "nrmse": nrmse_avg,
        "maxe": maxe_avg,
        "psnr": psnr_avg,
    }


def benchmark_split(codec, codec_for_bytes, engine, x: torch.Tensor, warmup=10, iters=50, stream=None):
    """
    Split timing for:
      - ga inference
      - codec encode
      - codec decode
      - gs inference

    Uses engine.ga_runner / engine.gs_runner directly, but runs on the same non-default stream.
    """
    assert x.is_cuda

    ga_runner = engine.ga_runner
    gs_runner = engine.gs_runner
    ga_input_dtype = getattr(engine, "ga_input_dtype", None)
    gs_input_dtype = getattr(engine, "gs_input_dtype", torch.float16)
    codec_input_dtype = getattr(engine, "codec_input_dtype", torch.float32)

    input_bytes = x.numel() * x.element_size()
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W

    stream = stream if stream is not None else torch.cuda.Stream(device=x.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            xx = x
            if ga_input_dtype is not None and xx.dtype != ga_input_dtype:
                xx = xx.to(ga_input_dtype)
            y = ga_runner(xx)
            y_fp32 = y.to(codec_input_dtype)
            pack = codec.compress(y_fp32)
            y_hat = codec.decompress(pack["strings"], pack["state"])
            y_hat = y_hat.to(gs_input_dtype)
            _ = gs_runner(y_hat)
        end.record()
    stream.synchronize()
    
    input_bytes_codec = y.numel() * 4

    ga_ms, enc_ms, dec_ms, gs_ms = [], [], [], []
    strings_bytes_list, state_bytes_list = [], []

    with torch.cuda.stream(stream):
        for _ in range(iters):
            # GA
            xx = x
            if ga_input_dtype is not None and xx.dtype != ga_input_dtype:
                xx = xx.to(ga_input_dtype)
            start.record()
            y = ga_runner(xx)
            end.record()
            end.synchronize()
            ga_ms.append(start.elapsed_time(end))

            # ENC
            y_fp32 = y.to(codec_input_dtype)
            start.record()
            pack = codec.compress(y_fp32)
            end.record()
            end.synchronize()
            enc_ms.append(start.elapsed_time(end))

            s_b, st_b = codec_for_bytes.pack_bytes(pack)
            strings_bytes_list.append(s_b)
            state_bytes_list.append(st_b)

            # DEC
            start.record()
            y_hat = codec.decompress(pack["strings"], pack["state"])
            end.record()
            end.synchronize()
            dec_ms.append(start.elapsed_time(end))

            # GS
            y_hat = y_hat.to(gs_input_dtype)
            start.record()
            _ = gs_runner(y_hat)
            end.record()
            end.synchronize()
            gs_ms.append(start.elapsed_time(end))

    stream.synchronize()

    ga_avg = sum(ga_ms) / iters
    enc_avg = sum(enc_ms) / iters
    dec_avg = sum(dec_ms) / iters
    gs_avg = sum(gs_ms) / iters

    ga_GBps  = (input_bytes / (ga_avg/1000.0)) / (1024**3)
    enc_GBps = (input_bytes_codec / (enc_avg/1000.0)) / (1024**3)
    dec_GBps = (input_bytes_codec / (dec_avg/1000.0)) / (1024**3)
    gs_GBps  = (input_bytes / (gs_avg/1000.0)) / (1024**3)

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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, default="/hwj")
    parser.add_argument("--quality", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--npy_path", type=str, default="/hwj/project/aiz-accelerate/data/nyx-dark_matter_density.npy")
    parser.add_argument("--engine_dir", type=str, default="/hwj/project/aiz-accelerate/engine")
    parser.add_argument("--P", type=int, default=64, help="Stream parallelism P for GPU rANS")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--packed_overhead", action="store_true", help="count packed container overhead into strings bytes")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # 1) load net
    net = bmshj2018_factorized(quality=args.quality, pretrained=False)
    model_path = f"{args.project_dir}/data/model/bmshj2018-factorized-prior-{args.quality}.pth"
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.eval().to(device)

    # 2) build your packed EB codec
    codec_for_bytes = GpuPackedEntropyBottleneckCodec(net, P=args.P, packed_overhead=args.packed_overhead)
    codec = codec_for_bytes  # same object used by adapter
    codec.eb.use_gpu_ans = True
    codec.eb.gpu_ans_parallelism = int(args.P)

    # 3) TRT runtime (engines exist offline)
    config = RuntimeConfig(
        mode="trt",
        precision="fp16",
        trt_engines={
            "ga": f"{args.engine_dir}/ga_output_f8_q{args.quality}.trt",
            "gs": f"{args.engine_dir}/gs_output_f8_q{args.quality}.trt",
        },
        ga_input_dtype=torch.float32,
        gs_input_dtype=torch.float16,
    )
    engine = build_runtime(net, codec, config)

    # 4) load input
    arr = np.load(args.npy_path)
    x = torch.from_numpy(arr).float().to(device)
    print("Input:", x.shape, x.dtype, x.device)
    print(f"GPU Packed rANS P={args.P} | packed_overhead={args.packed_overhead}")

    # 5) run on a non-default stream (important for TRT sync)
    bench_stream = torch.cuda.Stream(device=device)
    print(f"Benchmark stream (non-default): {bench_stream}")

    stats = benchmark_engine(engine, codec_for_bytes, x, warmup=args.warmup, iters=args.iters, stream=bench_stream)
    split = benchmark_split(codec, codec_for_bytes, engine, x, warmup=args.warmup, iters=args.iters, stream=bench_stream)

    print("\n==== TRT Benchmark Results (avg over iters) ====")
    print(f"Input: {tuple(x.shape)}, dtype={x.dtype}, input_bytes={stats['input_bytes']:,}")
    print(f"Encode: {stats['enc_ms_avg']:.3f} ms  |  Throughput: {stats['enc_GBps']:.2f} GB/s")
    print(f"Decode: {stats['dec_ms_avg']:.3f} ms  |  Throughput: {stats['dec_GBps']:.2f} GB/s")

    print("\n==== Split Benchmark (avg over iters) ====")
    print(f"GA   : {split['ga_ms']:.3f} ms | {split['ga_GBps']:.2f} GB/s")
    print(f"ENC  : {split['enc_ms']:.3f} ms | {split['enc_GBps']:.2f} GB/s")
    print(f"DEC  : {split['dec_ms']:.3f} ms | {split['dec_GBps']:.2f} GB/s")
    print(f"GS   : {split['gs_ms']:.3f} ms | {split['gs_GBps']:.2f} GB/s")

    print("\n---- Bitstream / Ratio ----")
    print(f"strings bytes (avg): {stats['strings_bytes_avg']:.1f}   (Packed payload{' + overhead' if args.packed_overhead else ''})")
    print(f"state bytes (avg):   {stats['state_bytes_avg']:.1f}   (size_hw metadata)")
    print(f"CR (strings-only):   {stats['cr_strings']:.3f}")
    print(f"CR (strings+state):  {stats['cr_total']:.3f}")
    print(f"bpp (strings-only):  {stats['bpp_strings']:.4f}")
    print(f"bpp (strings+state): {stats['bpp_total']:.4f}")
    print(f"rmse={stats['rmse']:.6f} nrmse={stats['nrmse']:.6f} maxe={stats['maxe']:.6f} psnr={stats['psnr']:.2f}")


if __name__ == "__main__":
    main()
