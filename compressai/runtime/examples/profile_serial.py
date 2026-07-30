#!/usr/bin/env python3
"""Minimal script for nsys profiling of SERIAL compress (no pipeline)."""
import argparse
import torch
import numpy as np

from compressai.zoo import bmshj2018_factorized
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.utils import dataloader_science

parser = argparse.ArgumentParser()
parser.add_argument("--bin_path", type=str, required=True)
parser.add_argument("--input_shape", type=str, default="100,500,500")
parser.add_argument("--block_size", type=str, default="3,512,512")
parser.add_argument("--project_dir", type=str, default="/hwj")
parser.add_argument("--dataset", type=str, default="hurricane")
parser.add_argument("--quality", type=int, default=4)
parser.add_argument("--batchsize", type=int, default=7)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--ans_variant", type=str, default="warp_div",
                    choices=["auto", "tight", "warp", "warp_div", "warp_smem"])
parser.add_argument("-P", "--ans_P", type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda:0")
input_shape = tuple(int(x) for x in args.input_shape.split(","))
block_size = tuple(int(x) for x in args.block_size.split(","))
p = 0.95
stride = (int(block_size[1] * p), int(block_size[2] * p))

blocks, _, _, _ = dataloader_science.blockify_bin_overlap(
    args.bin_path, input_shape, block_size, p, stride_hw=stride,
    dtype=np.float32, pad_value=0.5, norm_type="minmax", device=device
)

if args.repeat > 1:
    blocks = blocks.repeat(args.repeat, 1, 1, 1)

net = bmshj2018_factorized(quality=args.quality).to(device).eval()
ckpt = torch.load(
    f"{args.project_dir}/data/bmshj2018-factorized/bmshj2018-factorized-{args.quality}.pth",
    map_location=device)
net.load_state_dict(ckpt)
net.update()
codec = GpuPackedEntropyCodec(net.entropy_bottleneck, P=args.ans_P, ans_variant=args.ans_variant)

engine_dir = f"{args.project_dir}/data/engines/bmshj2018-factorized/q{args.quality}/engines/{args.dataset}"
cfg = RuntimeConfig(
    model_name="bmshj2018_factorized",
    ga_input_dtype=torch.float32, gs_input_dtype=torch.float16, codec_input_dtype=torch.float32,
    trt_engines={"ga": f"{engine_dir}/ga/fp8.engine", "gs": f"{engine_dir}/gs/fp16.engine"},
)
engine = build_runtime(net, codec, cfg)

# Split into batches and pad last
bn = blocks.shape[0]
batches = []
for start in range(0, bn, args.batchsize):
    end = min(start + args.batchsize, bn)
    chunk = blocks[start:end]
    cur_bs = chunk.shape[0]
    if cur_bs < args.batchsize:
        pad = torch.full((args.batchsize - cur_bs,) + chunk.shape[1:], 0.5,
                         dtype=chunk.dtype, device=device)
        chunk = torch.cat([chunk, pad], dim=0)
    batches.append(chunk)

N_batches = len(batches)
print(f"Batches: {N_batches}, batchsize={args.batchsize}"
      + (f", repeat={args.repeat}" if args.repeat > 1 else ""))

# ═══════════════════════════════════════════════════════════════
# Serial compress — same as bench but NO compress_time (no intermediate syncs)
# ═══════════════════════════════════════════════════════════════

ga_runner = engine.runners["ga"]
ga_in = engine.ga_input_dtype
codec_in = engine.codec_input_dtype

def serial_compress_all(blocks):
    packs = []
    for x in blocks:
        # ga
        xx = x
        if ga_in is not None and xx.dtype != ga_in:
            xx = xx.to(ga_in)
        if not xx.is_contiguous():
            xx = xx.contiguous()
        y = ga_runner(xx)
        # codec
        y_fp32 = y.to(codec_in) if y.dtype != codec_in else y
        packs.append(codec.compress(y_fp32))
    return packs

# Warmup
torch.cuda.empty_cache()
for _ in range(5):
    _ = serial_compress_all(batches)
torch.cuda.synchronize()

packs = serial_compress_all(batches)
torch.cuda.synchronize()

print("Done. Packs:", len(packs))
for i, p in enumerate(packs):
    s = p["strings"]
    if hasattr(s, "max_rounds_u32"):
        sz = int(s.max_rounds_u32.sum().item()) * 32 * 4 + int(s.header_bytes_cpu.item())
    else:
        sz = int(s.sizes_u32.sum().item())
    print(f"  batch {i}: {sz} bytes")
