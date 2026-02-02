# compressai/runtime/utils/benchmark.py
from __future__ import annotations

from typing import Any, Dict, Optional
import torch

from .metrics import basic_metrics


@torch.no_grad()
def run_e2e(
    engine,
    codec: Any,
    x: torch.Tensor,
    *,
    warmup: int = 5,
    iters: int = 20,
    stream: Optional[torch.cuda.Stream] = None,
) -> Dict[str, float]:
    """
    End-to-end benchmark:
      pack = engine.compress(x)
      x_hat = engine.decompress(pack)

    - Runs on a (by default) non-default CUDA stream.
    - Times encode/decode separately using CUDA events.
    - Computes avg bytes/bpp if codec.pack_bytes exists.
    - Computes basic reconstruction metrics.
    """
    assert x.is_cuda, "x must be on CUDA"
    H, W = int(x.shape[-2]), int(x.shape[-1])
    pixels = H * W * int(x.shape[0])
    input_bytes = float(x.numel() * x.element_size())

    stream = stream if stream is not None else torch.cuda.Stream(device=x.device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    # with torch.cuda.stream(stream):
    for _ in range(warmup):
        pack = engine.compress(x)
        _ = engine.decompress(pack)
    end.record()
    stream.synchronize()

    enc_ms_list = []
    dec_ms_list = []
    strings_bytes_list = []
    state_bytes_list = []
    x_hat = None
    
    # with torch.cuda.stream(stream):
    for _ in range(iters):
        # Encode
        start.record()
        pack = engine.compress(x)
        end.record()
        end.synchronize()
        enc_ms_list.append(float(start.elapsed_time(end)))

        # # Bytes (optional)
        # if hasattr(codec, "pack_bytes"):
        #     b = codec.pack_bytes(pack)
        #     strings_bytes_list.append(float(b.get("strings_bytes", 0.0)))
        #     state_bytes_list.append(float(b.get("state_bytes", 0.0)))
        #     print(float(b.get("strings_bytes", 0.0)), float(b.get("state_bytes", 0.0)))

        # Decode
        start.record()
        x_hat = engine.decompress(pack)
        end.record()
        end.synchronize()
        dec_ms_list.append(float(start.elapsed_time(end)))

        # m = basic_metrics(x_hat, x)
        # for k in metric_acc:
        #     metric_acc[k] += float(m[k])

    stream.synchronize()
    m = basic_metrics(x_hat, x)
    
    # Bytes (optional)
    if hasattr(codec, "pack_bytes"):
        b = codec.pack_bytes(pack)
        strings_bytes_list.append(float(b.get("strings_bytes", 0.0)))
        state_bytes_list.append(float(b.get("state_bytes", 0.0)))

    enc_ms = sum(enc_ms_list) / len(enc_ms_list)
    dec_ms = sum(dec_ms_list) / len(dec_ms_list)

    out: Dict[str, float] = {
        "input_bytes": input_bytes,
        "enc_ms": float(enc_ms),
        "dec_ms": float(dec_ms),
        "enc_GBps": float((input_bytes / (enc_ms / 1000.0)) / (1024**3)),
        "dec_GBps": float((input_bytes / (dec_ms / 1000.0)) / (1024**3)),
    }

    if strings_bytes_list:
        print("strings_bytes_list:", strings_bytes_list)
        strings_bytes = sum(strings_bytes_list) / len(strings_bytes_list)
        state_bytes = sum(state_bytes_list) / len(state_bytes_list)
        total_bytes = strings_bytes + state_bytes

        out.update({
            "strings_bytes": float(strings_bytes),
            "state_bytes": float(state_bytes),
            "total_bytes": float(total_bytes),
            "bpp_strings": float((strings_bytes * 8.0) / pixels),
            "bpp_total": float((total_bytes * 8.0) / pixels),
            "cr_strings": float(input_bytes / strings_bytes) if strings_bytes > 0 else float("inf"),
            "cr_total": float(input_bytes / total_bytes) if total_bytes > 0 else float("inf"),
        })

    for k in m:
        out[k] = m[k]

    return out, x_hat, x
