# compressai/runtime/utils/onnx_export.py
from __future__ import annotations

from typing import Dict, Optional, Sequence
import torch
import torch.nn as nn


@torch.no_grad()
def export_onnx(
    module: nn.Module,
    example_input: torch.Tensor,
    out_path: str,
    opset: int = 17,
    dynamic_hw: bool = True,
):
    module.eval()

    # Export expects CPU? 其实支持 CUDA，但更稳是先在 CUDA 跑出 trace，再 export
    x = example_input
    if not x.is_contiguous():
        x = x.contiguous()

    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic_hw:
        # N,C 固定，H/W 动态
        dynamic_axes = {
            "input": {0: "N", 2: "H", 3: "W"},
            "output": {0: "N", 2: "Hout", 3: "Wout"},
        }

    torch.onnx.export(
        module,
        (x,),
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
