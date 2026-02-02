# compressai/gpu_codec/quantizers/eb_quant.py
from __future__ import annotations

from typing import Tuple

import torch

from ..base import BaseQuantizer, QuantMeta
from ..registry import register_quantizer


@register_quantizer("eb")
class EBQuantizer(BaseQuantizer):
    """
    EB Quantizer
    """

    def __init__(self, eb: float = 0.8):
        self.eb = float(eb)

    def forward(self, y_norm: torch.Tensor) -> Tuple[torch.Tensor, QuantMeta]:
        scaled = y_norm / (2.0 * self.eb)

        # core rounding logic
        result = torch.floor(scaled + 0.5)
        mask = scaled < -0.5
        result = result - mask.to(result.dtype)

        y_q = result.to(torch.int16)

        meta = QuantMeta(params={"eb": self.eb})
        return y_q, meta

    def inverse(self, y_q: torch.Tensor, meta: QuantMeta) -> torch.Tensor:
        eb = float(meta.params["eb"])
        return y_q.to(torch.float32) * (2.0 * eb)
