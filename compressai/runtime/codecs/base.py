# compressai/runtime/codecs/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

Pack = Dict[str, Any]  # {"strings": ..., "state": ...}

class Codec(ABC):
    @abstractmethod
    def compress(self, y_fp32: torch.Tensor, **kwargs) -> Pack:
        """Compress FP32 latent -> pack containing 'strings' and 'state'."""
        raise NotImplementedError

    @abstractmethod
    def decompress(self, pack: Pack, **kwargs) -> torch.Tensor:
        """Decompress pack -> latent tensor (typically FP32)."""
        raise NotImplementedError

    # Optional, for benchmark/UI
    def pack_bytes(self, pack: Pack) -> Dict[str, float]:
        """Return {'strings_bytes':..., 'state_bytes':...}."""
        return {"strings_bytes": 0.0, "state_bytes": 0.0}
