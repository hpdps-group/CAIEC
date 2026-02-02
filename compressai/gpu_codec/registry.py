# compressai/gpu_codec/registry.py
from __future__ import annotations

from typing import Dict, Type

from .base import BaseNormalizer, BaseQuantizer, BaseNvcompAlgo

NORMALIZERS: Dict[str, Type[BaseNormalizer]] = {}
QUANTIZERS: Dict[str, Type[BaseQuantizer]] = {}
ALGOS: Dict[str, Type[BaseNvcompAlgo]] = {}


def register_normalizer(name: str):
    def deco(cls: Type[BaseNormalizer]) -> Type[BaseNormalizer]:
        NORMALIZERS[name] = cls
        cls.name = name
        return cls
    return deco


def register_quantizer(name: str):
    def deco(cls: Type[BaseQuantizer]) -> Type[BaseQuantizer]:
        QUANTIZERS[name] = cls
        cls.name = name
        return cls
    return deco


def register_algo(name: str):
    def deco(cls: Type[BaseNvcompAlgo]) -> Type[BaseNvcompAlgo]:
        ALGOS[name] = cls
        cls.name = name
        return cls
    return deco
