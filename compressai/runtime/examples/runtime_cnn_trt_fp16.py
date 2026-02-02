import pickle
import numpy as np
import torch
import math

from compressai.zoo import bmshj2018_factorized
from compressai import gpu_codec
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig

def metrics(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
    """
    Return rmse, nrmse, maxe, psnr in a cuZFP-like style.
    Compute in fp32 for consistency.
    """
    x_f = x.float()
    y_f = x_hat.float()

    diff = y_f - x_f
    mse = torch.mean(diff * diff)
    rmse = torch.sqrt(mse)

    maxe = torch.max(torch.abs(diff))

    range = torch.max(x_f) - torch.min(x_f)
    nrmse = rmse / range

    psnr = 20.0 * torch.log10(range / (2.0 * rmse + eps))

    # return python floats
    return rmse.item(), nrmse.item(), maxe.item(), psnr.item()

def _bytes_of_strings(strings):
    total = 0
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, (list, tuple)):
        for s in strings:
            total += _bytes_of_strings(s)
        return total
    return 0


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


def benchmark_engine(engine, x: torch.Tensor, warmup=10, iters=50):
    assert x.is_cuda, "x must be on CUDA"
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W
    input_bytes = x.numel() * x.element_size()

    # Warmup
    for _ in range(warmup):
        pack = engine.compress(x)
        _ = engine.decompress(pack)
    torch.cuda.synchronize()

    enc_times = []
    dec_times = []
    strings_bytes_list = []
    state_bytes_list = []
    rmse_list, nrmse_list, maxe_list, psnr_list = [], [], [], []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        # Encode
        start.record()
        pack = engine.compress(x)
        end.record()
        torch.cuda.synchronize()
        enc_ms = start.elapsed_time(end)
        enc_times.append(enc_ms)

        strings = pack.get("strings", None)
        state = pack.get("state", None)
        strings_bytes_list.append(_bytes_of_strings(strings))
        state_bytes_list.append(_bytes_of_state(state))

        # Decode
        start.record()
        x_hat = engine.decompress(pack)
        end.record()
        torch.cuda.synchronize()
        dec_ms = start.elapsed_time(end)
        dec_times.append(dec_ms)
        
        # Metrics
        rmse, nrmse, maxe, psnr = metrics(x_hat, x)
        rmse_list.append(rmse)
        nrmse_list.append(nrmse)
        maxe_list.append(maxe)
        psnr_list.append(psnr)

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

def benchmark_split(net, codec, engine, x: torch.Tensor, warmup=10, iters=50):
    """
    Split timing for:
      - ga inference
      - codec encode
      - codec decode
      - gs inference

    Notes:
      - ga/gs runners come from engine (TRT runners inside build_runtime)
      - codec is the same object used by adapter
      - codec input is forced FP32
      - y_hat cast to match gs precision (from engine settings)
    """
    assert x.is_cuda

    # Pull runners + dtype rules from engine (CnnEngine)
    ga_runner = engine.ga_runner
    gs_runner = engine.gs_runner
    ga_input_dtype = getattr(engine, "ga_input_dtype", None)
    gs_input_dtype = getattr(engine, "gs_input_dtype", torch.float16)
    codec_input_dtype = getattr(engine, "codec_input_dtype", torch.float32)

    input_bytes = x.numel() * x.element_size()
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ---------------- warmup ----------------
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
    torch.cuda.synchronize()

    # ---------------- timed iters ----------------
    ga_ms, enc_ms, dec_ms, gs_ms = [], [], [], []
    strings_bytes_list, state_bytes_list = [], []

    for _ in range(iters):
        # ---- GA ----
        xx = x
        if ga_input_dtype is not None and xx.dtype != ga_input_dtype:
            xx = xx.to(ga_input_dtype)

        start.record()
        y = ga_runner(xx)
        end.record()
        torch.cuda.synchronize()
        ga_ms.append(start.elapsed_time(end))

        # ---- Codec Encode (y -> pack) ----
        y_fp32 = y.to(codec_input_dtype)

        start.record()
        pack = codec.compress(y_fp32)
        end.record()
        torch.cuda.synchronize()
        enc_ms.append(start.elapsed_time(end))

        strings_bytes_list.append(_bytes_of_strings(pack.get("strings", None)))
        state_bytes_list.append(_bytes_of_state(pack.get("state", None)))

        # ---- Codec Decode (pack -> y_hat) ----
        start.record()
        y_hat = codec.decompress(pack["strings"], pack["state"])
        end.record()
        torch.cuda.synchronize()
        dec_ms.append(start.elapsed_time(end))

        # ---- GS ----
        y_hat = y_hat.to(gs_input_dtype)

        start.record()
        x_hat = gs_runner(y_hat)
        end.record()
        torch.cuda.synchronize()
        gs_ms.append(start.elapsed_time(end))

    # averages
    ga_avg = sum(ga_ms) / iters
    enc_avg = sum(enc_ms) / iters
    dec_avg = sum(dec_ms) / iters
    gs_avg = sum(gs_ms) / iters

    # Throughput (GB/s) for each stage, using input_bytes as numerator
    # (same denominator definition as your end-to-end benchmark)
    ga_GBps  = (input_bytes / (ga_avg/1000.0)) / (1024**3)
    enc_GBps = (input_bytes / (enc_avg/1000.0)) / (1024**3)
    dec_GBps = (input_bytes / (dec_avg/1000.0)) / (1024**3)
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
    

# --------------------------
# 1) load net
project_dir = "/hwj"
quality = 1
device = "cuda:0"

net = bmshj2018_factorized(quality=quality, pretrained=False)
old_state_dict = torch.load(
    f"{project_dir}/data/model/bmshj2018-factorized-prior-{quality}.pth",
    map_location=device
)
net.load_state_dict(old_state_dict)
net.eval().to(device)

# 2) build codec
cfg = {
    "normalizer": {"name": "minmax", "params": {"scale_bits": 8}},
    "quantizer": {"name": "eb", "params": {"eb": 0.8}},
    "algo": {"name": "bitcomp", "params": {}},
}
codec = gpu_codec.build_gpu_codec(cfg)

# 3) build TRT runtime (engines exist offline)
engine_dir = "/hwj/project/aiz-accelerate/engine"
config = RuntimeConfig(
    mode="trt",
    precision="fp16",
    trt_engines={
        "ga": f"{engine_dir}/ga_output_f16_q{quality}.trt",
        "gs": f"{engine_dir}/gs_output_f16_q{quality}.trt",
    },
    ga_input_dtype=torch.float16,
    gs_input_dtype=torch.float16,
)

engine = build_runtime(net, codec, config)

# 4) load npy -> tensor
path = "/hwj/project/aiz-accelerate/data/nyx-dark_matter_density.npy"
arr = np.load(path)

x = torch.from_numpy(arr).float().to(device)  # keep fp32 input for fairness vs torch baseline
# If your TRT engine expects fp16 input, build_runtime/CnnEngine should cast via ga_input_dtype.
print("Input:", x.shape, x.dtype, x.device)

# 5) benchmark
stats = benchmark_engine(engine, x, warmup=10, iters=50)
split = benchmark_split(net, codec, engine, x, warmup=10, iters=50)

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
print(f"strings bytes (avg): {stats['strings_bytes_avg']:.1f}")
print(f"state bytes (pickle avg): {stats['state_bytes_avg']:.1f}")
print(f"CR (strings-only): {stats['cr_strings']:.3f}")
print(f"CR (strings+state): {stats['cr_total']:.3f}")
print(f"bpp (strings-only): {stats['bpp_strings']:.4f}")
print(f"bpp (strings+state): {stats['bpp_total']:.4f}")
print(f"rmse={stats['rmse']:.4f} nrmse={stats['nrmse']:.5f} maxe={stats['maxe']:.4f} psnr={stats['psnr']:.2f}")

