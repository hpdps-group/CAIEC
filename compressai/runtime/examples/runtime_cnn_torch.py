import time
import numpy as np
import torch
from compressai.runtime.adapters.cnn_simple import SimpleCnnAdapter
from compressai.runtime.engines.cnn_engine import CnnEngine
from compressai.zoo import bmshj2018_factorized
from compressai import gpu_codec

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

def make_torch_runner(module):
    module.eval().cuda()
    @torch.no_grad()
    def _run(x):
        return module(x)
    return _run


def _bytes_of_strings(strings):
    # strings is typically list[list[bytes]] or list[bytes]
    total = 0
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, (list, tuple)):
        for s in strings:
            total += _bytes_of_strings(s)
        return total
    # unknown type
    return 0


import pickle
import torch

def _bytes_of_state(obj):
    """
    Estimate bytes of `state`.
    Prefer pickle size (closer to actual metadata serialization),
    fallback to recursive approximation.
    """
    if obj is None:
        return 0

    # Fast path: try pickle (best approximation for python dict metadata)
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pass

    # Fallback: recursive estimate
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()

    if isinstance(obj, (bytes, bytearray)):
        return len(obj)

    if isinstance(obj, str):
        return len(obj.encode("utf-8"))

    if isinstance(obj, bool):
        return 1

    if isinstance(obj, int):
        # heuristic: 8 bytes for typical int
        return 8

    if isinstance(obj, float):
        return 8

    if isinstance(obj, dict):
        # Add a tiny overhead per entry to reflect structure
        overhead = 16 * len(obj)
        return overhead + sum(_bytes_of_state(k) + _bytes_of_state(v) for k, v in obj.items())

    if isinstance(obj, (list, tuple)):
        overhead = 8 * len(obj)
        return overhead + sum(_bytes_of_state(v) for v in obj)

    # Unknown python objects: ignore
    return 0



def benchmark_engine(engine: CnnEngine, x: torch.Tensor, warmup=10, iters=50):
    assert x.is_cuda, "x must be on CUDA"
    H, W = x.shape[-2], x.shape[-1]
    pixels = H * W

    # Use input bytes as "throughput numerator"
    input_bytes = x.numel() * x.element_size()

    # Warmup
    for _ in range(warmup):
        pack = engine.compress(x)
        _ = engine.decompress(pack)
    torch.cuda.synchronize()
    
    # Timers
    enc_times = []
    dec_times = []
    strings_bytes_list = []
    state_bytes_list = []
    rmse_list, nrmse_list, maxe_list, psnr_list = [], [], [], []

    # CUDA events for accurate GPU timing
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

        # Collect size stats
        strings = pack.get("strings", None)
        state = pack.get("state", None)
        s_bytes = _bytes_of_strings(strings)
        st_bytes = _bytes_of_state(state)
        strings_bytes_list.append(s_bytes)
        state_bytes_list.append(st_bytes)

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

    # Aggregate
    enc_ms_avg = sum(enc_times) / len(enc_times)
    dec_ms_avg = sum(dec_times) / len(dec_times)

    strings_bytes_avg = sum(strings_bytes_list) / len(strings_bytes_list)
    state_bytes_avg = sum(state_bytes_list) / len(state_bytes_list)

    total_bytes_avg = strings_bytes_avg + state_bytes_avg

    # Throughput (bytes / second)
    enc_s = enc_ms_avg / 1000.0
    dec_s = dec_ms_avg / 1000.0

    enc_GBps = (input_bytes / enc_s) / (1024**3)
    dec_GBps = (input_bytes / dec_s) / (1024**3)

    # Compression ratio
    # CR = original_bytes / compressed_bytes
    # Use strings-only as "real bitstream size" (typical)
    cr_strings = (input_bytes / strings_bytes_avg) if strings_bytes_avg > 0 else float("inf")
    cr_total = (input_bytes / total_bytes_avg) if total_bytes_avg > 0 else float("inf")

    # bpp (strings only / total)
    bpp_strings = (strings_bytes_avg * 8.0) / pixels
    bpp_total = (total_bytes_avg * 8.0) / pixels
    
    rmse_avg = sum(rmse_list) / len(rmse_list)
    nrmse_avg = sum(nrmse_list) / len(nrmse_list)
    maxe_avg = sum(maxe_list) / len(maxe_list)
    psnr_avg = sum(psnr_list) / len(psnr_list)

    return {
        "input_bytes": input_bytes,
        "pixels": pixels,
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


# --------------------------
# 1) load net
project_dir = '/hwj'
quality = 1
device = 'cuda:0'

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

adapter = SimpleCnnAdapter(net, codec)

ga_runner = make_torch_runner(adapter.subgraphs()["ga"])
gs_runner = make_torch_runner(adapter.subgraphs()["gs"])

# FP32 baseline (torch runners). For later TRT FP16/FP8, you’ll switch runners + gs_input_dtype.
engine = CnnEngine(
    adapter,
    ga_runner,
    gs_runner,
    codec_input_dtype=torch.float32,   # per your rule: codec input always FP32
    gs_input_dtype=torch.float32,      # gs is fp32 in this baseline
    ga_input_dtype=torch.float32,
)

# Test input
# N, C, H, W = 1, 3, 512, 512
# x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
path = "/hwj/project/aiz-accelerate/data/nyx-dark_matter_density.npy"

# 4) load npy -> tensor
arr = np.load(path)
x = torch.from_numpy(arr).float().to(device)
print(x.shape, x.dtype, x.device)

stats = benchmark_engine(engine, x, warmup=10, iters=50)

print("\n==== Benchmark Results (avg over iters) ====")
print(f"Input: {x.shape}, dtype={x.dtype}, input_bytes={stats['input_bytes']:,}")
print(f"Encode: {stats['enc_ms_avg']:.3f} ms  |  Throughput: {stats['enc_GBps']:.2f} GB/s")
print(f"Decode: {stats['dec_ms_avg']:.3f} ms  |  Throughput: {stats['dec_GBps']:.2f} GB/s")

print("\n---- Bitstream / Ratio ----")
print(f"strings bytes (avg): {stats['strings_bytes_avg']:.1f}")
print(f"state bytes (est avg): {stats['state_bytes_avg']:.1f}")
print(f"CR (strings-only): {stats['cr_strings']:.3f}")
print(f"CR (strings+state est): {stats['cr_total']:.3f}")
print(f"bpp (strings-only): {stats['bpp_strings']:.4f}")
print(f"bpp (strings+state est): {stats['bpp_total']:.4f}")
print(f"rmse={stats['rmse']:.4f} nrmse={stats['nrmse']:.5f} maxe={stats['maxe']:.4f} psnr={stats['psnr']:.2f}")

