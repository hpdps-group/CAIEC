# compressai/runtime/engines/hyperprior_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.zoo import bmshj2018_hyperprior


class HyperpriorEngine(CnnEngine):
    """
    Hyperprior pipeline:
      y = ga(x)
      z = ha(y)
      z_pack = codec.compress(z_fp32)
      z_hat = codec.decompress(z_pack)
      params = hs(z_hat)
      y_pack = codec.compress(y_fp32, params=params)
      (return {"y": y_pack, "z": z_pack})

    Decompress:
      z_hat = codec.decompress(z_pack)
      params = hs(z_hat)
      y_hat = codec.decompress(y_pack, params=params)
      x_hat = gs(y_hat)
    """

    def __init__(
        self,
        net: bmshj2018_hyperprior,
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
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)

        # ha
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](torch.abs(y_ha))
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # codec(z) requires FP32
        if z.dtype != self.codec_input_dtype:
            z_fp32 = z.to(self.codec_input_dtype)
        else:
            z_fp32 = z
        
        # symbols = z_fp32.detach().cpu().numpy()
        # symbols.tofile("/hwj/project/caiec-script/latent_data/nyx/Z_90x128x8x8.f32")
        # return 0
        z_pack = self.codec.compress(z_fp32)

        # z_hat
        z_hat = self.codec.decompress(z_pack)
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        scales_hat = self.runners["hs"](z_hat_hs)

        # codec(y | params) requires FP32 y
        if y.dtype != self.codec_input_dtype:
            y_fp32 = y.to(self.codec_input_dtype)
        else:
            y_fp32 = y
        params = self.codec.gaussian_conditional.build_indexes(scales_hat)
        # symbols = y_fp32.detach().cpu().numpy()
        # symbols.tofile("/hwj/project/caiec-script/latent_data/nyx/Y_90x192x32x32.f32")
        y_pack = self.codec.compress(y_fp32, params=params)

        return {"y": y_pack, "z": z_pack}

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        # z_hat
        z_hat = self.codec.decompress(pack["z"])
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # hs -> params
        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        scales_hat = self.runners["hs"](z_hat_hs)
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)

        # y_hat
        y_hat = self.codec.decompress(pack["y"], params=indexes, dtype=z_hat.dtype)
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

        # ---------- inference part (non-codec) ----------
        start.record()

        x = self._ensure_cuda_contiguous(x)

        # ga
        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)

        # ha
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](torch.abs(y_ha))
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # codec(z) requires FP32 (cast is counted into inference_ms like in your factorized version)
        if z.dtype != self.codec_input_dtype:
            z_fp32 = z.to(self.codec_input_dtype)
        else:
            z_fp32 = z

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        start.record()
        z_pack = self.codec.compress(z_fp32)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        # ---------- codec part: z decompress ----------
        start.record()
        z_hat = self.codec.decompress(z_pack)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # ---------- inference part: hs + indexes + y_fp32 cast ----------
        start.record()

        z_hat_hs = self._cast(z_hat, self.hs_input_dtype)
        scales_hat = self.runners["hs"](z_hat_hs)

        # codec(y | params) requires FP32 y
        if y.dtype != self.codec_input_dtype:
            y_fp32 = y.to(self.codec_input_dtype)
        else:
            y_fp32 = y

        params = self.codec.gaussian_conditional.build_indexes(scales_hat)

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # ---------- codec part: y compress ----------
        start.record()
        y_pack = self.codec.compress(
            y_fp32, params=params
        )
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
        scales_hat = self.runners["hs"](z_hat_hs)
        indexes = self.codec.gaussian_conditional.build_indexes(scales_hat)
        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # ---------- codec part: y decompress ----------
        start.record()
        y_hat = self.codec.decompress(pack["y"], params=indexes, dtype=z_hat.dtype)
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
