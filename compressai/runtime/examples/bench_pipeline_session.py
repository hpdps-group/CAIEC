#!/usr/bin/env python3
"""Pipelined session-based compress + decompress benchmark."""
import argparse
import torch, numpy as np

from compressai.zoo import bmshj2018_factorized
from compressai.runtime import build_runtime
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.runtime.stream_pipeline import (
    compress_pipelined_session,
    decompress_pipelined,
    decompress_pipelined_session,
)
from compressai.runtime.ans_pipeline import PipelinedAnsEncoder, PipelinedAnsDecoder
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
ga_runner = engine.runners["ga"]
gs_runner = engine.runners["gs"]
ga_in = engine.ga_input_dtype
gs_in = engine.gs_input_dtype
eb = codec.eb

# Split into batches of batchsize, pad last
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
total_bytes = bn * blocks[0].numel() * blocks[0].element_size()
print(f"Blocks: {bn}, batches: {N_batches} × ~{args.batchsize} blocks each")

# ── Session setup (one encoder per batch) ──
xx = batches[0]
if ga_in is not None and xx.dtype != ga_in:
    xx = xx.to(ga_in)
with torch.no_grad():
    sample_y = ga_runner(xx if xx.is_contiguous() else xx.contiguous())
B, C, H, W = sample_y.shape
N_sym = C * H * W

# Prep — same as eb.compress internals
medians = eb._get_medians().detach()
medians = eb._extend_ndims(medians, 2)
medians = medians.expand(B, *([-1] * (medians.dim())))

def prep(y_fp):
    symbols = eb.quantize(y_fp, "symbols", medians).to(torch.int32)
    indexes = torch.arange(C, device=y_fp.device, dtype=torch.int32) \
                   .view(1, C, 1, 1).expand(B, C, H, W).contiguous()
    return symbols.contiguous(), indexes

del sample_y, xx

# ── Warmup ──
print("Warmup...")
for _ in range(5):
    encoders = [PipelinedAnsEncoder.from_entropy_bottleneck(eb, B, N_sym, P=args.ans_P) for _ in range(N_batches)]
    packs = compress_pipelined_session(
        ga_runner=ga_runner, prep_for_encode=prep, encoders=encoders,
        blocks=batches, ga_input_dtype=ga_in)
    for p in packs:
        y_hat = codec.decompress(p)
        _ = gs_runner(y_hat.to(gs_in))
    torch.cuda.synchronize()
torch.cuda.synchronize()

# ── Compress (5 rounds) ──
cmp_per_round = []
for r in range(2):
    encoders = [PipelinedAnsEncoder.from_entropy_bottleneck(eb, B, N_sym, P=args.ans_P) for _ in range(N_batches)]
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    packs = compress_pipelined_session(
        ga_runner=ga_runner, prep_for_encode=prep, encoders=encoders,
        blocks=batches, ga_input_dtype=ga_in)
    e.record(); torch.cuda.synchronize()
    cmp_per_round.append(s.elapsed_time(e))

cmp_avg = sum(cmp_per_round) / len(cmp_per_round)
cmp_gbps = (total_bytes / (cmp_avg / 1000)) / (1024 ** 3)

# ── Decompress (session-based) ──
dec_per_round = []
for r in range(2):
    pack0_str = packs[0]["strings"]
    sz_hw = packs[0]["state"]["size_hw"]
    K_dec = pack0_str.max_rounds_u32.size(1)
    cl_dec = int(pack0_str.chunk_len_cpu.item())
    HW_dec = sz_hw[0] * sz_hw[1]
    N_dec = C * HW_dec

    medians = eb._get_medians().detach()
    medians = eb._extend_ndims(medians, 2)
    medians = medians.expand(B, *([-1] * (medians.dim())))
    def dequant(out_i32):
        y_q = out_i32.reshape(B, C, sz_hw[0], sz_hw[1])
        return y_q.to(torch.float32) + medians

    decoders = [PipelinedAnsDecoder.from_entropy_bottleneck(
        eb, B, N_dec, K_dec, cl_dec, HW_dec) for _ in range(N_batches)]

    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    x_hats = decompress_pipelined_session(
        gs_runner=gs_runner, dequantize_for_gs=dequant,
        decoders=decoders, packs=packs, gs_input_dtype=gs_in)
    e.record(); torch.cuda.synchronize()
    dec_per_round.append(s.elapsed_time(e))

dec_avg = sum(dec_per_round) / len(dec_per_round)
dec_gbps = (total_bytes / (dec_avg / 1000)) / (1024 ** 3)

print(f"Compress:   {cmp_avg:.2f} ms  ({cmp_gbps:.2f} GB/s)")
print(f"Decompress: {dec_avg:.2f} ms  ({dec_gbps:.2f} GB/s)")
print(f"Per-batch:  ~{cmp_avg/N_batches:.2f} ms cmp, ~{dec_avg/N_batches:.2f} ms dec")

# Verify correctness on last round
y0 = ga_runner(batches[0] if batches[0].is_contiguous() else batches[0].contiguous())
y_hat0 = codec.decompress(packs[0])
ok = torch.allclose(y0, y_hat0, atol=0.5)
print(f"Roundtrip OK: {ok}")
