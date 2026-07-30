#!/usr/bin/env python3
"""Minimal script for nsys profiling of the compress pipeline."""
import argparse
import torch
import numpy as np

from compressai.zoo import bmshj2018_factorized
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.runtime.stream_pipeline import (
    compress_pipelined,
    compress_pipelined_session,
)
from compressai.runtime.ans_pipeline import PipelinedAnsEncoder
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
                    help="Replicate blocks N times to increase batch count for profiling")
parser.add_argument("--ans_variant", type=str, default="warp_div",
                    choices=["auto", "tight", "warp", "warp_div", "warp_smem"])
parser.add_argument("-P", "--ans_P", type=int, default=64,
                    help="GPU ANS parallelism granularity P")
parser.add_argument("--session", action="store_true",
                    help="Use session-based pipelined encoder (non-blocking launch)")
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
# Session setup (if requested)
# ═══════════════════════════════════════════════════════════════
if args.session:
    ga_runner = engine.runners["ga"]
    ga_in = engine.ga_input_dtype
    eb = codec.eb

    # Sample one batch to get B, C, N
    xx = batches[0]
    if ga_in is not None and xx.dtype != ga_in:
        xx = xx.to(ga_in)
    with torch.no_grad():
        sample_y = ga_runner(xx if xx.is_contiguous() else xx.contiguous())
    B, C, H, W = sample_y.shape
    N_sym = C * H * W
    del sample_y, xx

    # Prep function — same as eb.compress internals
    medians = eb._get_medians().detach()
    medians = eb._extend_ndims(medians, 2)
    # medians is [C, 1, 1]; expand to [B, C, 1, 1]
    medians = medians.expand(B, *([-1] * (medians.dim())))

    def prep(y_fp):
        symbols = eb.quantize(y_fp, "symbols", medians).to(torch.int32)
        indexes = torch.arange(C, device=y_fp.device, dtype=torch.int32) \
                       .view(1, C, 1, 1).expand(B, C, H, W).contiguous()
        return symbols.contiguous(), indexes

    encoders = [
        PipelinedAnsEncoder.from_entropy_bottleneck(eb, B, N_sym, P=args.ans_P)
        for _ in range(N_batches)
    ]

    pipe_fn = lambda: compress_pipelined_session(
        ga_runner=ga_runner, prep_for_encode=prep, encoders=encoders,
        blocks=batches, ga_input_dtype=ga_in)
else:
    pipe_fn = lambda: compress_pipelined(
        ga_runner=engine.runners["ga"], codec=engine.codec, blocks=batches,
        ga_input_dtype=engine.ga_input_dtype,
        codec_input_dtype=engine.codec_input_dtype)

# ═══════════════════════════════════════════════════════════════
# Warmup + profile
# ═══════════════════════════════════════════════════════════════
torch.cuda.empty_cache()
for _ in range(5):
    _ = pipe_fn()
torch.cuda.synchronize()

packs = pipe_fn()
torch.cuda.synchronize()

print("Done. Packs:", len(packs))
for i, p in enumerate(packs):
    s = p["strings"]
    if hasattr(s, "max_rounds_u32"):
        sz = int(s.max_rounds_u32.sum().item()) * 32 * 4 + int(s.header_bytes_cpu.item())
    else:
        sz = int(s.sizes_u32.sum().item())
    print(f"  batch {i}: {sz} bytes")
