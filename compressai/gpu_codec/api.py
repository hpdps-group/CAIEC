# compressai/gpu_codec/api.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import torch

from .base import (
    AlgoMeta,
    BaseNormalizer,
    BaseNvcompAlgo,
    BaseQuantizer,
    NormMeta,
    QuantMeta,
)
from .registry import ALGOS, NORMALIZERS, QUANTIZERS

# side-effect imports to register components
from . import normalizers  # noqa: F401
from . import quantizers   # noqa: F401
from . import compressors  # noqa: F401


@dataclass
class GPUCodecState:
    """
    Metadata for GPUCodec (serializable)

    - *_cfg: ctor params (hyper-params) to rebuild components
    - *_meta: runtime params from forward/encode for inverse/decode
    """
    norm_name: str
    quant_name: str
    algo_name: str

    norm_cfg: Dict[str, Any]
    quant_cfg: Dict[str, Any]
    algo_cfg: Dict[str, Any]

    norm_meta: NormMeta
    quant_meta: QuantMeta
    algo_meta: AlgoMeta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GPUCodecState":
        return GPUCodecState(
            norm_name=d["norm_name"],
            quant_name=d["quant_name"],
            algo_name=d["algo_name"],

            norm_cfg=d["norm_cfg"],
            quant_cfg=d["quant_cfg"],
            algo_cfg=d["algo_cfg"],

            norm_meta=NormMeta(params=d["norm_meta"]["params"]),
            quant_meta=QuantMeta(params=d["quant_meta"]["params"]),
            algo_meta=AlgoMeta(params=d["algo_meta"]["params"]),
        )


class GPUCodec:
    """
    Pipeline = Normalizer + Quantizer + nvCOMP Algo
    Align with CompressAI:
      compress(y)  -> {"strings": [...], "state": dict}
      decompress(strings, state) -> y_hat
    """

    def __init__(
        self,
        normalizer: BaseNormalizer,
        quantizer: BaseQuantizer,
        algo: BaseNvcompAlgo,
        norm_cfg: Dict[str, Any],
        quant_cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
    ):
        self.normalizer = normalizer
        self.quantizer = quantizer
        self.algo = algo

        # save ctor params for state
        self.norm_cfg = dict(norm_cfg)
        self.quant_cfg = dict(quant_cfg)
        self.algo_cfg = dict(algo_cfg)

    @torch.no_grad()
    def compress(self, y: torch.Tensor) -> Dict[str, Any]:
        assert y.is_cuda, "GPUCodec expects CUDA tensor"

        y_norm, norm_meta = self.normalizer.forward(y)
        y_q, quant_meta = self.quantizer.forward(y_norm)
        strings, algo_meta = self.algo.encode(y_q)

        state = GPUCodecState(
            norm_name=self.normalizer.name,
            quant_name=self.quantizer.name,
            algo_name=self.algo.name,

            norm_cfg=self.norm_cfg,
            quant_cfg=self.quant_cfg,
            algo_cfg=self.algo_cfg,

            norm_meta=norm_meta,
            quant_meta=quant_meta,
            algo_meta=algo_meta,
        )
        return {"strings": strings, "state": state.to_dict()}

    @torch.no_grad()
    def decompress(self, strings: List[bytes], state_dict: Dict[str, Any]) -> torch.Tensor:
        state = GPUCodecState.from_dict(state_dict)

        # rebuild components ONLY from ctor cfg (hyper-params)
        norm_cls = NORMALIZERS[state.norm_name]
        quant_cls = QUANTIZERS[state.quant_name]
        algo_cls = ALGOS[state.algo_name]

        normalizer: BaseNormalizer = norm_cls(**state.norm_cfg)
        quantizer: BaseQuantizer = quant_cls(**state.quant_cfg)
        algo: BaseNvcompAlgo = algo_cls(**state.algo_cfg)

        # restore using runtime meta
        y_q = algo.decode(strings, state.algo_meta)
        y_norm = quantizer.inverse(y_q, state.quant_meta)
        y_hat = normalizer.inverse(y_norm, state.norm_meta)
        
        return y_hat


def build_gpu_codec(cfg: Dict[str, Any]) -> GPUCodec:
    """
    cfg example:
    {
      "normalizer": {"name":"minmax", "params":{"scale_bits":8}},
      "quantizer": {"name":"eb", "params":{"eb":0.8}},
      "algo": {"name":"bitcomp", "params":{...}}
    }
    """
    ncfg = cfg["normalizer"]
    qcfg = cfg["quantizer"]
    acfg = cfg["algo"]

    norm_name = ncfg["name"]
    quant_name = qcfg["name"]
    algo_name = acfg["name"]

    norm_params = dict(ncfg.get("params", {}))
    quant_params = dict(qcfg.get("params", {}))
    algo_params = dict(acfg.get("params", {}))

    normalizer: BaseNormalizer = NORMALIZERS[norm_name](**norm_params)
    quantizer: BaseQuantizer = QUANTIZERS[quant_name](**quant_params)
    algo: BaseNvcompAlgo = ALGOS[algo_name](**algo_params)

    return GPUCodec(
        normalizer=normalizer,
        quantizer=quantizer,
        algo=algo,
        norm_cfg=norm_params,
        quant_cfg=quant_params,
        algo_cfg=algo_params,
    )
