#!/usr/bin/env python3
"""Serial compress + decompress benchmark — batch-by-batch, aligned with notebook."""
import argparse
import torch, numpy as np

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
parser.add_argument("--repeat", type=int, default=1,
                    help="Replicate blocks N times to increase batch count")
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
net.load_state_dict(ckpt); net.update()
codec = GpuPackedEntropyCodec(net.entropy_bottleneck, P=args.ans_P, ans_variant=args.ans_variant)

engine_dir = f"{args.project_dir}/data/engines/bmshj2018-factorized/q{args.quality}/engines/{args.dataset}"
cfg = RuntimeConfig(
    model_name="bmshj2018_factorized",
    ga_input_dtype=torch.float32, gs_input_dtype=torch.float16, codec_input_dtype=torch.float32,
    trt_engines={"ga": f"{engine_dir}/ga/fp8.engine", "gs": f"{engine_dir}/gs/fp16.engine"},
)
engine = build_runtime(net, codec, cfg)

# Split into batches of batchsize, pad last if needed
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
total_bytes = bn * blocks[0].numel() * blocks[0].element_size()  # original data, not padded
print(f"Blocks: {bn}, batches: {N_batches} × ~{args.batchsize} blocks each")

# Warmup — run the full serial loop 5 times
print("Warmup...")
for _ in range(5):
    for b in batches:
        pack, _, _ = engine.compress_time(b)
        x_hat, _, _ = engine.decompress_time(pack)
    torch.cuda.synchronize()
torch.cuda.synchronize()

# Compress — 5 rounds of all batches
cmp_per_round = []
for r in range(1):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    packs = []
    for b in batches:
        pack, _, _ = engine.compress_time(b)
        packs.append(pack)
    e.record(); torch.cuda.synchronize()
    cmp_per_round.append(s.elapsed_time(e))

cmp_avg = sum(cmp_per_round) / len(cmp_per_round)
cmp_gbps = (total_bytes / (cmp_avg / 1000)) / (1024 ** 3)

# Decompress — 5 rounds of all batches
dec_per_round = []
for r in range(1):
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for pack in packs:
        _, _, _ = engine.decompress_time(pack)
    e.record(); torch.cuda.synchronize()
    dec_per_round.append(s.elapsed_time(e))

dec_avg = sum(dec_per_round) / len(dec_per_round)
dec_gbps = (total_bytes / (dec_avg / 1000)) / (1024 ** 3)

print(f"Compress:   {cmp_avg:.2f} ms  ({cmp_gbps:.2f} GB/s)")
print(f"Decompress: {dec_avg:.2f} ms  ({dec_gbps:.2f} GB/s)")
print(f"Per-batch:  ~{cmp_avg/N_batches:.2f} ms cmp, ~{dec_avg/N_batches:.2f} ms dec")
