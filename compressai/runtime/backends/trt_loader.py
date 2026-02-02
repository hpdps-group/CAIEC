# compressai/runtime/backends/trt_loader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .trt_module import TRTModule


@dataclass(frozen=True)
class TRTEngineSpec:
    """
    Points to an existing TRT engine file produced offline.
    """
    engine_path: str
    input_name: Optional[str] = None
    output_name: Optional[str] = None

    def validate(self) -> None:
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(
                f"TRT engine not found: {self.engine_path}\n"
                f"Please run offline engine build first."
            )


class TRTRunnerFactory:
    """
    Runtime-only loader: no ONNX export, no quantization, no TRT build.
    """

    def __init__(self):
        pass

    def load(self, spec: TRTEngineSpec) -> TRTModule:
        spec.validate()
        return TRTModule(
            engine_path=spec.engine_path,
            input_name=spec.input_name,
            output_name=spec.output_name,
        )
