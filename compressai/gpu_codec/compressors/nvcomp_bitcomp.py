# compressai/gpu_codec/compressors/nvcomp_bitcomp.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import nvidia.nvcomp as nvcomp

from ..base import AlgoMeta, BaseNvcompAlgo
from ..registry import register_algo


@register_algo("bitcomp")
class NvcompBitcompAlgo(BaseNvcompAlgo):
    """
    nvCOMP bitcomp compressor (batch mode)

    encode:
      y_q(int16 CUDA tensor) -> List[bytes] with ONE element

    decode:
      List[bytes] -> y_q(int16 CUDA tensor)
    """

    def __init__(
        self,
        algo_name: str = "bitcomp",
        algo_type: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.algo_name = algo_name
        self.algo_type = algo_type
        self.kwargs: Dict[str, Any] = dict(kwargs)

        if algo_type is None:
            self.codec = nvcomp.Codec(algorithm=algo_name, **kwargs)
        else:
            self.codec = nvcomp.Codec(
                algorithm=algo_name,
                algorithm_type=algo_type,
                **kwargs,
            )

    def encode(self, y_q: torch.Tensor) -> Tuple[List[bytes], AlgoMeta]:
        """
        Args:
            y_q: int16 CUDA tensor
        Returns:
            strings: [one_big_bytes]
            meta: AlgoMeta
        """
        assert y_q.is_cuda, "nvcomp bitcomp expects CUDA tensor"
        assert y_q.dtype == torch.int16, f"Expected int16, got {y_q.dtype}"

        # batch -> bytes view
        y_bytes = y_q.view(torch.uint8).contiguous()
        # arr = nvcomp.as_array(torch.utils.dlpack.to_dlpack(y_bytes))

        arr = nvcomp.as_array(y_bytes)
        # one-shot encode
        comp_arr = self.codec.encode(arr).cpu()

        # comp_arr -> torch.uint8 -> bytes
        # comp_torch = torch.utils.dlpack.from_dlpack(comp_arr.to_dlpack())
        # one_big_bytes = comp_torch[:valid_len].detach().cpu().numpy().tobytes()
        comp_np = np.asarray(comp_arr)
        one_big_bytes = comp_np.tobytes()

        strings: List[bytes] = [one_big_bytes]

        meta = AlgoMeta(params={
            "algo_name": self.algo_name,
            "algo_type": self.algo_type,
            "batch_mode": True,
            "orig_shape": list(y_q.shape),
            **self.kwargs,
        })
        return strings, meta

    def decode(
        self,
        strings: List[bytes],
        meta: AlgoMeta,
    ) -> torch.Tensor:
        """
        Args:
            strings: list[bytes]
        Returns:
            y_q: int16 CUDA tensor
        """
        assert isinstance(meta, AlgoMeta)
        assert len(strings) == 1, "batch_mode decode expects a single bitstream"
        
        b = strings[0]

        # bytes -> torch.uint8(CUDA) -> dlpack -> nvcomp arr
        # bt = torch.frombuffer(memoryview(b), dtype=torch.uint8)
        # bt = bt.to("cuda").contiguous()
        # arr = nvcomp.as_array(torch.utils.dlpack.to_dlpack(bt))
        arr = nvcomp.as_array(b).cuda()

        # one-shot decode
        decomp_arr = self.codec.decode(arr)

        # decomp_arr -> torch.uint8(CUDA) -> view int16 -> reshape
        y_hat_bytes = torch.utils.dlpack.from_dlpack(decomp_arr).clone()
        y_q = y_hat_bytes.view(torch.int16)
        
        orig_shape = meta.params.get("orig_shape", None)
        
        if orig_shape is not None:
            y_q = y_q.reshape(orig_shape)
        else:
            raise RuntimeError(
            "[NvcompBitcompAlgo] Missing `orig_shape` in AlgoMeta. "
            "Lossless nvCOMP algorithms require original tensor shape "
            "to restore decoded data."
        )

        return y_q
