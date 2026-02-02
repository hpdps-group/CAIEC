# compressai/gpu_codec/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class NormMeta:
    """meta data for Normalizer inverse"""
    params: Dict[str, Any]


@dataclass
class QuantMeta:
    """meta data for Quantizer inverse"""
    params: Dict[str, Any]


@dataclass
class AlgoMeta:
    """meta data for nvCOMP Algo decode"""
    params: Dict[str, Any]


class BaseNormalizer(ABC):
    """float -> float（e.g. minmax/log/uniform）"""

    name: str = "base"

    @abstractmethod
    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, NormMeta]:
        """y(float) -> y_norm(float), meta"""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y_norm: torch.Tensor, meta: NormMeta) -> torch.Tensor:
        """y_norm(float), meta -> y(float)"""
        raise NotImplementedError


class BaseQuantizer(ABC):
    """float -> int（e.g. eb/truncate）"""

    name: str = "base"

    @abstractmethod
    def forward(self, y_norm: torch.Tensor) -> Tuple[torch.Tensor, QuantMeta]:
        """y_norm(float) -> y_q(int tensor), meta"""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, y_q: torch.Tensor, meta: QuantMeta) -> torch.Tensor:
        """y_q(int tensor), meta -> y_norm(float)"""
        raise NotImplementedError


class BaseNvcompAlgo(ABC):
    """int tensor <-> bytes （e.g. nvcomp）"""

    name: str = "base"

    @abstractmethod
    def encode(self, y_q: torch.Tensor) -> Tuple[List[bytes], AlgoMeta]:
        """y_q(int tensor) -> list[bytes], meta"""
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        strings: List[bytes],
        meta: AlgoMeta,
    ) -> torch.Tensor:
        """list[bytes], shape, meta -> y_q(int tensor)"""
        raise NotImplementedError
