# compressai/runtime/adapters/cnn_simple.py
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from .base import ModelAdapter, Pack
from ..codecs.base import Codec


class SimpleCnnAdapter(ModelAdapter):
    """
    Pure CNN adapter:
      y = net.g_a(x)
      pack = codec.compress(y_fp32) -> {"strings", "state"}
      y_hat = codec.decompress(pack)
      x_hat = net.g_s(y_hat)
    """

    def __init__(self, net: nn.Module, codec: Codec):
        self.net = net
        self.codec = codec

        if not hasattr(net, "g_a") or not hasattr(net, "g_s"):
            raise AttributeError("Expected net to have g_a and g_s submodules.")

        self.ga = net.g_a
        self.gs = net.g_s

    def subgraphs(self) -> Dict[str, nn.Module]:
        subs = {"ga": self.ga, "gs": self.gs}
        # hyperprior models expose h_a / h_s
        if hasattr(self.net, "h_a") and hasattr(self.net, "h_s"):
            subs["ha"] = self.net.h_a
            subs["hs"] = self.net.h_s
        return subs


    def compress_glue(self, y: torch.Tensor) -> Pack:
        pack = self.codec.compress(y)
        if not isinstance(pack, dict):
            raise TypeError("codec.compress(y) must return a dict-like pack.")
        if "strings" not in pack or "state" not in pack:
            raise KeyError('pack must contain keys: "strings" and "state".')
        return pack

    def decompress_glue(self, pack: Pack) -> torch.Tensor:
        y_hat = self.codec.decompress(pack)
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("codec.decompress(pack) must return a torch.Tensor.")
        return y_hat
