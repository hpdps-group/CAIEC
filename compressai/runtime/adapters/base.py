# compressai/runtime/adapters/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


Pack = Dict[str, Any]  # pack["strings"], pack["state"], ... (extensible)


class ModelAdapter(ABC):
    """
    Adapter hides model-structure differences and exposes:
      - subgraphs to be compiled/executed by a backend (TRT/Torch/etc.)
      - glue logic for codec compress/decompress (lossless/entropy/etc.)
    """

    @abstractmethod
    def subgraphs(self) -> Dict[str, nn.Module]:
        """Return subgraphs that can be accelerated by a backend."""
        raise NotImplementedError

    @abstractmethod
    def compress_glue(self, y: torch.Tensor) -> Pack:
        """Encode latent y to bitstream/state pack."""
        raise NotImplementedError

    @abstractmethod
    def decompress_glue(self, pack: Pack) -> torch.Tensor:
        """Decode pack to latent y_hat."""
        raise NotImplementedError
