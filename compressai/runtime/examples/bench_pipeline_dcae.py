#!/usr/bin/env python3
"""Benchmark serial and session-pipelined DCAE compression/decompression."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from compressai.models import DCAE
from compressai.runtime import build_runtime
from compressai.runtime.codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.runtime.config import RuntimeConfig
from compressai.runtime.dcae_stream_pipeline import (
    compress_dcae_pipelined_session,
    create_dcae_decode_sessions,
    create_dcae_encode_sessions,
    decompress_dcae_pipelined_session,
)
import compressai.runtime.utils.metrics as metrics
from compressai.utils import dataloader_science


BASE_COMPONENTS = ("ga", "gs", "ha", "h_z_s1", "h_z_s2")
DEFAULT_PRECISIONS = {
    "ga": "fp16", "gs": "fp16", "ha": "fp8",
    "h_z_s1": "fp8", "h_z_s2": "fp16",
    "dt_cross_attention": "fp16", "cc_mean": "fp8",
    "cc_scale": "fp8", "lrp_transforms": "fp8",
}
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "fp8": torch.float32}


def csv_tuple(value: str, length: int, name: str) -> Tuple[int, ...]:
    out = tuple(int(v) for v in value.split(","))
    if len(out) != length or any(v <= 0 for v in out):
        raise ValueError(f"{name} must contain {length} positive comma-separated integers")
    return out


def split_batches(x: torch.Tensor, batch_size: int, pad_value: float) -> Tuple[List[torch.Tensor], List[int]]:
    batches, valid = [], []
    for start in range(0, x.shape[0], batch_size):
        chunk = x[start:start + batch_size]
        valid.append(int(chunk.shape[0]))
        if chunk.shape[0] < batch_size:
            pad = torch.full((batch_size - chunk.shape[0], *chunk.shape[1:]), pad_value,
                             dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat((chunk, pad), 0)
        batches.append(chunk.contiguous())
    return batches, valid


def parse_precision_overrides(values: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--engine_precision expects COMPONENT=PRECISION, got {value!r}")
        component, precision = value.split("=", 1)
        if precision not in ("fp8", "fp16", "fp32"):
            raise ValueError(f"invalid precision {precision!r} for {component}")
        out[component] = precision
    return out


def engine_paths(engine_dir: Path, num_slices: int, overrides: Dict[str, str]) -> Dict[str, str]:
    components = list(BASE_COMPONENTS)
    for i in range(num_slices):
        components += [f"dt_cross_attention_{i}", f"cc_mean_{i}",
                       f"cc_scale_{i}", f"lrp_transforms_{i}"]
    unknown = set(overrides) - set(components)
    if unknown:
        raise ValueError(f"unknown engine components: {', '.join(sorted(unknown))}")
    paths = {}
    for component in components:
        family = component.rsplit("_", 1)[0] if component.rsplit("_", 1)[-1].isdigit() else component
        precision = overrides.get(component, DEFAULT_PRECISIONS[family])
        paths[component] = str(engine_dir / component / f"{precision}.engine")
    return paths


def timed_cuda(fn, *args):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args)
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def serial_compress(engine, batches):
    return [engine.compress(x) for x in batches]


def serial_decompress(engine, packs):
    return [engine.decompress(pack) for pack in packs]


def payload_bytes(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, (list, tuple)):
        return sum(payload_bytes(v) for v in value)
    if (hasattr(value, "max_rounds_u32")
            and torch.is_tensor(value.max_rounds_u32)):
        payload = int(value.max_rounds_u32.sum().item()) * 32 * 4
        header = int(value.header_bytes_cpu.reshape(-1)[0].item())
        return header + payload
    if hasattr(value, "packed") and torch.is_tensor(value.packed):
        return int(value.packed.numel())
    return 0


def state_bytes(value: Any) -> int:
    return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def byte_stats(packs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    stats = {"z": 0, "y": 0, "state": 0}
    for pack in packs:
        stats["z"] += payload_bytes(pack["z"]["strings"])
        stats["y"] += payload_bytes(pack["y"]["strings"])
        stats["state"] += state_bytes(pack["z"].get("state"))
        stats["state"] += state_bytes(pack["y"].get("state"))
    stats["total"] = stats["z"] + stats["y"] + stats["state"]
    return stats


def flatten_valid(outputs: Sequence[torch.Tensor], valid: Sequence[int]) -> torch.Tensor:
    return torch.cat([x[:n] for x, n in zip(outputs, valid)], 0)


def verify(name: str, reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (reference.float() - candidate.float()).abs()
    maximum = float(diff.max())
    mean = float(diff.mean())
    ok = torch.allclose(reference.float(), candidate.float(), atol=atol, rtol=rtol)
    print(f"  {name:<32} {'PASS' if ok else 'FAIL'}  max={maximum:.6g} mean={mean:.6g}")
    if not ok:
        raise RuntimeError(f"interoperability check failed: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark serial vs WarpANS session-pipelined DCAE")
    p.add_argument("--bin_path", required=True, help="Science float32 binary file")
    p.add_argument("--input_shape", default="100,500,500", help="N,H,W")
    p.add_argument("--block_size", default="3,512,512", help="C,H,W")
    p.add_argument("--checkpoint", default=None, help="DCAE checkpoint (default: /hwj/data/dcae/dcae-Q.pth)")
    p.add_argument("--quality", type=int, default=1)
    p.add_argument("--dataset", default="hurricane")
    p.add_argument("--engine_dir", default=None, help="Directory containing COMPONENT/PRECISION.engine")
    p.add_argument("--engine_precision", action="append", default=[], metavar="COMPONENT=PRECISION",
                   help="Override one engine, e.g. cc_mean_0=fp16 (repeatable)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batchsize", type=int, default=34)
    p.add_argument("--repeat", type=int, default=1,
                   help="Replicate blocks N times to increase the number of pipeline batches")
    p.add_argument("--pad_value", type=float, default=0.5)
    p.add_argument("--overlap", type=float, default=0.95)
    p.add_argument("-P", "--ans_P", type=int, default=256)
    p.add_argument("--ans_variant", choices=("warp_smem",), default="warp_smem",
                   help="Session API currently uses WarpANS V3 (warp_smem) kernels")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("this benchmark requires CUDA")
    if args.batchsize <= 0 or args.repeat <= 0 or args.ans_P <= 0 or args.warmup < 0:
        raise ValueError("batchsize/repeat/ans_P must be positive and warmup non-negative")
    input_shape = csv_tuple(args.input_shape, 3, "input_shape")
    block_size = csv_tuple(args.block_size, 3, "block_size")
    device = torch.device(args.device)
    stride = (int(block_size[1] * args.overlap), int(block_size[2] * args.overlap))
    if min(stride) <= 0:
        raise ValueError("overlap produces a non-positive stride")

    blocks, original, status, meta = dataloader_science.blockify_bin_overlap(
        args.bin_path, input_shape, block_size, args.overlap, stride_hw=stride,
        dtype=np.float32, pad_value=args.pad_value, norm_type="minmax", device=device)
    original_block_count = int(blocks.shape[0])
    if args.repeat > 1:
        blocks = blocks.repeat(args.repeat, 1, 1, 1)
    batches, valid = split_batches(blocks, args.batchsize, args.pad_value)
    if not batches:
        raise ValueError("input produced no blocks")

    checkpoint_path = Path(args.checkpoint or f"/hwj/data/dcae/dcae-{args.quality}.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("checkpoint must be a state_dict or contain a 'state_dict' mapping")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net = DCAE()
    net.load_state_dict(state)
    net.update()
    net.eval().to(device)

    codec = GpuPackedEntropyCodec(net.entropy_bottleneck, net.gaussian_conditional,
                                  P=args.ans_P, ans_variant=args.ans_variant)
    root = Path(args.engine_dir or f"/hwj/data/engines/dcae/q{args.quality}/engines/{args.dataset}")
    overrides = parse_precision_overrides(args.engine_precision)
    paths = engine_paths(root, int(net.num_slices), overrides)
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing TensorRT engines:\n  " + "\n  ".join(missing))
    precision = lambda component: overrides.get(component, DEFAULT_PRECISIONS[component])
    slice_dtypes = lambda family: [DTYPES[overrides.get(f"{family}_{i}", DEFAULT_PRECISIONS[family])]
                                   for i in range(int(net.num_slices))]
    cfg = RuntimeConfig(
        model_name="dcae", trt_engines=paths,
        ga_input_dtype=DTYPES[precision("ga")], gs_input_dtype=DTYPES[precision("gs")],
        ha_input_dtype=DTYPES[precision("ha")], codec_input_dtype=torch.float32,
        h_z_s1_input_dtype=DTYPES[precision("h_z_s1")],
        h_z_s2_input_dtype=DTYPES[precision("h_z_s2")],
        dt_ca_input_dtypes=slice_dtypes("dt_cross_attention"),
        cc_mean_input_dtypes=slice_dtypes("cc_mean"),
        cc_scale_input_dtypes=slice_dtypes("cc_scale"),
        lrp_input_dtypes=slice_dtypes("lrp_transforms"),
    )
    engine = build_runtime(net, codec, cfg)

    with torch.no_grad():
        sample_y = engine.runners["ga"](engine._cast(batches[0], engine.ga_input_dtype)).contiguous()
        sample_z = engine.runners["ha"](engine._cast(sample_y, engine.ha_input_dtype)).contiguous()
    y_shape, z_shape = tuple(sample_y.shape[1:]), tuple(sample_z.shape[1:])
    del sample_y, sample_z
    encode_sessions = create_dcae_encode_sessions(
        engine, len(batches), args.batchsize, z_shape, y_shape)

    for _ in range(args.warmup):
        warm_packs = serial_compress(engine, batches)
        serial_decompress(engine, warm_packs)
        warm_decode = create_dcae_decode_sessions(engine, warm_packs)
        decompress_dcae_pipelined_session(engine, warm_decode, warm_packs)

    serial_packs, serial_c_ms = timed_cuda(serial_compress, engine, batches)
    pipeline_packs, pipeline_c_ms = timed_cuda(
        compress_dcae_pipelined_session, engine, encode_sessions, batches)

    # Decoder creation depends on each pack's WarpANS metadata and is intentionally untimed.
    serial_pack_sessions = create_dcae_decode_sessions(engine, serial_packs)
    pipeline_pack_sessions = create_dcae_decode_sessions(engine, pipeline_packs)
    serial_serial, serial_d_ms = timed_cuda(serial_decompress, engine, serial_packs)
    pipe_serial, pipe_serial_d_ms = timed_cuda(
        decompress_dcae_pipelined_session, engine, serial_pack_sessions, serial_packs)
    serial_pipe, serial_pipe_d_ms = timed_cuda(serial_decompress, engine, pipeline_packs)
    pipe_pipe, pipe_d_ms = timed_cuda(
        decompress_dcae_pipelined_session, engine, pipeline_pack_sessions, pipeline_packs)

    reference = flatten_valid(serial_serial, valid)
    print("\nFour-way interoperability (decoded values, not encoded sizes):")
    verify("serial encode -> serial decode", reference, reference, args.atol, args.rtol)
    verify("serial encode -> pipeline decode", reference, flatten_valid(pipe_serial, valid), args.atol, args.rtol)
    verify("pipeline encode -> serial decode", reference, flatten_valid(serial_pipe, valid), args.atol, args.rtol)
    verify("pipeline encode -> pipeline decode", reference, flatten_valid(pipe_pipe, valid), args.atol, args.rtol)

    stats = byte_stats(serial_packs)
    reconstructed = dataloader_science.blocks_to_nhw_overlap_weighted(
        reference[:original_block_count], input_shape, block_size,
        stride_hw=stride, meta_hw=meta, pad_value=args.pad_value,
        norm_type="minmax", vmin=status["vmin"], vmax=status["vmax"],
        vmean=status["vmean"], vstd=status["vstd"])
    rd = metrics.basic_metrics(reconstructed, original)
    logical_pixels = int(original.numel()) * args.repeat
    bpp = stats["total"] * 8.0 / logical_pixels
    source_bytes = sum(batch.numel() * batch.element_size() for batch in batches)
    gbps = lambda ms: source_bytes / (ms / 1000.0) / 1024**3

    print("\nTiming (session creation excluded):")
    print(f"  compress serial={serial_c_ms:.3f} ms ({gbps(serial_c_ms):.3f} GiB/s), "
          f"pipeline={pipeline_c_ms:.3f} ms ({gbps(pipeline_c_ms):.3f} GiB/s), "
          f"speedup={serial_c_ms / pipeline_c_ms:.3f}x")
    print(f"  decompress serial/serial={serial_d_ms:.3f} ms, pipeline/serial-pack={pipe_serial_d_ms:.3f} ms")
    print(f"  decompress serial/pipeline-pack={serial_pipe_d_ms:.3f} ms, pipeline/pipeline-pack={pipe_d_ms:.3f} ms, "
          f"speedup={serial_d_ms / pipe_d_ms:.3f}x")
    print("\nBytes (serial pack):")
    print(f"  z={stats['z']:,} y={stats['y']:,} state={stats['state']:,} total={stats['total']:,}")
    print("\nRD metrics:")
    print(f"  bpp={bpp:.6f} rmse={rd['rmse']:.6g} nrmse={rd['nrmse']:.6g} "
          f"maxe={rd['maxe']:.6g} psnr={rd['psnr']:.3f}")
    print("\nSummary:")
    print(f"  blocks={original_block_count} x repeat={args.repeat} -> {blocks.shape[0]}, "
          f"batches={len(batches)}, batchsize={args.batchsize}, "
          f"y={y_shape}, z={z_shape}, WarpANS={args.ans_variant}/P{args.ans_P}")
    print(f"  compress speedup={serial_c_ms / pipeline_c_ms:.3f}x, "
          f"decompress speedup={serial_d_ms / pipe_d_ms:.3f}x, bpp={bpp:.6f}, psnr={rd['psnr']:.3f}")


if __name__ == "__main__":
    main()
