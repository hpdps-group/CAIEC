# compressai/gpu_codec/normalizers/minmax_norm.py
from __future__ import annotations

from typing import Tuple

import torch

from ..base import BaseNormalizer, NormMeta
from ..registry import register_normalizer


@register_normalizer("minmax")
class MinMaxNormalizer(BaseNormalizer):
    """
    Min-Max Normalizer
    """

    def __init__(self, scale_bits: int = 8, eps: float = 1e-8):
        self.scale_bits = int(scale_bits)
        self.eps = float(eps)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, NormMeta]:
        y_min = torch.amin(y)
        y_max = torch.amax(y)
        denom = (y_max - y_min).clamp_min(self.eps)
        y_norm = (y - y_min) / denom  # float, approx [0,1]
        y_scaled = torch.clip(y_norm * (2 ** self.scale_bits), 0, 32767)

        meta = NormMeta(params={
            "min": float(y_min.item()),
            "max": float(y_max.item()),
            "scale_bits": self.scale_bits,
        })
        return y_scaled, meta

    def inverse(self, y_norm: torch.Tensor, meta: NormMeta) -> torch.Tensor:
        y_min = float(meta.params["min"])
        y_max = float(meta.params["max"])
        scale_bits = float(meta.params["scale_bits"])
        y_norm = y_norm / (2 ** scale_bits)
        return y_norm * (y_max - y_min) + y_min
