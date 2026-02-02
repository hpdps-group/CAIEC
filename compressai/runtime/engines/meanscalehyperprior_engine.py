# compressai/runtime/engines/hyperprior_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.zoo import mbt2018_mean


class MeanScaleHyperpriorEngine(CnnEngine):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).
    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►──┤h_s├─┐
                  └───┘    │     └───┘     └───┘       codec      └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                     codec : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        codec = GpuPackedEntropyCodec
    """

    def __init__(
        self,
        net: mbt2018_mean,
        codec: GpuPackedEntropyCodec,
        runners: Dict[str, Any],
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        ga_input_dtype: Optional[torch.dtype] = None,
        ha_input_dtype: Optional[torch.dtype] = None,
        hs_input_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.codec = codec
        self.runners = runners  # expects keys: ga, gs, ha, hs

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype
        self.ha_input_dtype = ha_input_dtype
        self.hs_input_dtype = hs_input_dtype

        for k in ("ga", "gs", "ha", "hs"):
            if k not in self.runners:
                raise ValueError(f"HyperpriorEngine requires runner '{k}'")

    def _cast(self, x: torch.Tensor, dt: Optional[torch.dtype]) -> torch.Tensor:
        if dt is not None and x.dtype != dt:
            return x.to(dt)
        return x

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self._ensure_cuda_contiguous(x)

        # ga
        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)                    # y = self.g_a(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)

        # ha
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](y_ha)                # z = self.h_a(y)
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # codec(z) requires FP32
        if z.dtype != self.codec_input_dtype:
            z_fp32 = z.to(self.codec_input_dtype)
        else:
            z_fp32 = z
        z_pack = self.codec.compress(z_fp32)       # z_strings = self.entropy_bottleneck.compress(z)

        # z_hat
        z_hat = self.codec.decompress(z_pack)      # z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        gaussian_params = self.runners["hs"](z_hat_hs)  # gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # codec(y | params) requires FP32 y
        if y.dtype != self.codec_input_dtype:
            y_fp32 = y.to(self.codec_input_dtype)
        else:
            y_fp32 = y
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)
        y_pack = self.codec.compress(y_fp32, indexes, means=means_hat)  # y_strings = self.gaussian_conditional.compress(y, scales_hat, means=means_hat)

        return {"y": y_pack, "z": z_pack}

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        # z_hat
        z_hat = self.codec.decompress(pack["z"])     # z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        gaussian_params = self.runners["hs"](z_hat_hs)   # gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)

        # y_hat
        y_hat = self.codec.decompress(pack["y"], params=indexes, means=means_hat)
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("codec.decompress(y_pack) must return torch.Tensor")
        y_hat = self._ensure_cuda_contiguous(y_hat)

        # gs
        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)
        x_hat = self.runners["gs"](y_hat).clamp_(0, 1)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs runner must return torch.Tensor")
        return self._ensure_cuda_contiguous(x_hat)
    
    def compress_time(self, x: torch.Tensor) -> Tuple[Dict[str, Any], float, float]:
        """
        Returns:
        pack: {"y": y_pack, "z": z_pack}
        inference_ms: ga/ha/hs等推理+张量处理耗时（不含codec）
        codec_ms: codec compress/decompress累计耗时
        """
        inference_ms = 0.0
        codec_ms = 0.0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ---------- inference part (non-codec): ensure + ga + ha + cast(z_fp32) ----------
        start.record()

        x = self._ensure_cuda_contiguous(x)

        # ga
        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)  # y = self.g_a(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)

        # ha
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](y_ha)  # z = self.h_a(y)
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # codec(z) requires FP32 (cast计入 inference_ms，和你上面的计时方法一致)
        if z.dtype != self.codec_input_dtype:
            z_fp32 = z.to(self.codec_input_dtype)
        else:
            z_fp32 = z

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # ---------- codec part: z compress ----------
        start.record()
        z_pack = self.codec.compress(z_fp32)  # entropy_bottleneck.compress(z)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        # ---------- codec part: z decompress ----------
        start.record()
        z_hat = self.codec.decompress(z_pack)  # entropy_bottleneck.decompress(...)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # ---------- inference part (non-codec): hs + indexes + cast(y_fp32) ----------
        start.record()

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        gaussian_params = self.runners["hs"](z_hat_hs)  # gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # codec(y | params) requires FP32 y (cast计入 inference_ms)
        if y.dtype != self.codec_input_dtype:
            y_fp32 = y.to(self.codec_input_dtype)
        else:
            y_fp32 = y

        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # ---------- codec part: y compress ----------
        start.record()
        y_pack = self.codec.compress(y_fp32, indexes, means=means_hat)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        return {"y": y_pack, "z": z_pack}, inference_ms, codec_ms


    def decompress_time(self, pack: Dict[str, Any]) -> Tuple[torch.Tensor, float, float]:
        """
        Returns:
        x_hat
        inference_ms: hs/gs等推理+张量处理耗时（不含codec）
        codec_ms: codec decompress累计耗时（z + y）
        """
        inference_ms = 0.0
        codec_ms = 0.0

        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # ---------- codec part: z decompress ----------
        start.record()
        z_hat = self.codec.decompress(pack["z"])
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # ---------- inference part: hs + indexes ----------
        start.record()

        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        gaussian_params = self.runners["hs"](z_hat_hs)  # gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # ---------- codec part: y decompress ----------
        start.record()
        y_hat = self.codec.decompress(pack["y"], params=indexes, means=means_hat)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("codec.decompress(y_pack) must return torch.Tensor")
        y_hat = self._ensure_cuda_contiguous(y_hat)

        # ---------- inference part: gs ----------
        start.record()

        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)

        x_hat = self.runners["gs"](y_hat).clamp_(0, 1)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs runner must return torch.Tensor")

        x_hat = self._ensure_cuda_contiguous(x_hat)

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        return x_hat, inference_ms, codec_ms

