# compressai/runtime/engines/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch

from ..adapters.base import ModelAdapter

class CnnEngine(ABC):
    """
    Orchestrates:
      y = ga_runner(x)
      pack = codec.compress(y_cast_fp32)            # via adapter.compress_glue
      y_hat = codec.decompress(strings, state)      # via adapter.decompress_glue
      x_hat = gs_runner(y_hat_cast_to_gs_dtype)

    Dtype rules (per your corrected requirements):
      - compress: ALWAYS cast y -> FP32 before codec.compress (avoid lossless errors)
      - decompress: cast y_hat -> gs_input_dtype before gs_runner (match gs precision)
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        ga_runner,
        gs_runner,
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        # Optional: cast x into what ga_runner expects (e.g. TRT prefers FP16 input)
        ga_input_dtype: Optional[torch.dtype] = None,
    ):
        self.adapter = adapter
        self.ga_runner = ga_runner
        self.gs_runner = gs_runner

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype

    @staticmethod
    def _ensure_cuda_contiguous(x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("Runtime expects CUDA tensors for accelerated execution.")
        return x.contiguous() if not x.is_contiguous() else x
    
    @abstractmethod
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError