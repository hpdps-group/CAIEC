# compressai/runtime/engines/cnn_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.zoo import bmshj2018_factorized

class FactorizedEngine(CnnEngine):
    def __init__(
        self,
        net: bmshj2018_factorized,
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
        self.net = net
        self.runners = runners  # expects keys: ga, gs, ha, hs

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype
        self.ha_input_dtype = ha_input_dtype
        self.hs_input_dtype = hs_input_dtype

        for k in ("ga", "gs"):
            if k not in self.runners:
                raise ValueError(f"FactorizedEngine requires runner '{k}'")


    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        # x = self._ensure_cuda_contiguous(x)

        # Optional: adapt x dtype for ga_runner backend
        if self.ga_input_dtype is not None and x.dtype != self.ga_input_dtype:
            x = x.to(self.ga_input_dtype)

        y = self.runners["ga"](x)
        # if not isinstance(y, torch.Tensor):
        #     raise TypeError("ga_runner must return a torch.Tensor.")
        # y = self._ensure_cuda_contiguous(y)

        # REQUIRED: codec.compress must see FP32 latent
        if y.dtype != self.codec_input_dtype:
            y = y.to(self.codec_input_dtype)

        pack = self.codec.compress(y)
        return pack

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        y_hat = self.codec.decompress(pack)
        # if not isinstance(y_hat, torch.Tensor):
        #     raise TypeError("adapter.decompress_glue(pack) must return a torch.Tensor.")
        # y_hat = self._ensure_cuda_contiguous(y_hat)

        # REQUIRED: match gs precision expectation
        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)

        x_hat = self.runners["gs"](y_hat)
        # if not isinstance(x_hat, torch.Tensor):
        #     raise TypeError("gs_runner must return a torch.Tensor.")
        # x_hat = self._ensure_cuda_contiguous(x_hat)
        return x_hat
    
    def compress_time(self, x: torch.Tensor) -> Tuple[Dict[str, Any], float, float]:
        inference_ms = 0.0
        codec_ms = 0.0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        if self.ga_input_dtype is not None and x.dtype != self.ga_input_dtype:
            x = x.to(self.ga_input_dtype)
            
        x = self._ensure_cuda_contiguous(x)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga_runner must return a torch.Tensor.")
        y = self._ensure_cuda_contiguous(y)

        if y.dtype != self.codec_input_dtype:
            y = y.to(self.codec_input_dtype)
            
        end.record()
        torch.cuda.synchronize()
        inference_ms = float(start.elapsed_time(end))
        
        start.record()
        pack = self.codec.compress(y)
        end.record()
        torch.cuda.synchronize()
        codec_ms = float(start.elapsed_time(end))
        
        return pack, inference_ms, codec_ms

    def decompress_time(self, pack: Dict[str, Any]) -> Tuple[torch.Tensor, float, float]:
        inference_ms = 0.0
        codec_ms = 0.0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        y_hat = self.codec.decompress(pack)
        end.record()
        torch.cuda.synchronize()
        codec_ms = float(start.elapsed_time(end))
        
        start.record()
        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("adapter.decompress_glue(pack) must return a torch.Tensor.")
        

        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)
            
        y_hat = self._ensure_cuda_contiguous(y_hat)
        x_hat = self.runners["gs"](y_hat)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs_runner must return a torch.Tensor.")
        x_hat = self._ensure_cuda_contiguous(x_hat)
        
        end.record()
        torch.cuda.synchronize()
        inference_ms = float(start.elapsed_time(end))

        return x_hat, inference_ms, codec_ms
    
    def compress_cpulossless(self, x: torch.Tensor) -> Dict[str, Any]:
        inference_ms = 0.0
        codec_ms = 0.0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        if self.ga_input_dtype is not None and x.dtype != self.ga_input_dtype:
            x = x.to(self.ga_input_dtype)

        x = self._ensure_cuda_contiguous(x)
        y = self.runners["ga"](x)

        if not isinstance(y, torch.Tensor):
            raise TypeError("ga_runner must return a torch.Tensor.")
        y = self._ensure_cuda_contiguous(y)

        # builtin entropy bottleneck expects float input
        if y.dtype != self.codec_input_dtype:
            y = y.to(self.codec_input_dtype)
            
        end.record()
        torch.cuda.synchronize()
        inference_ms = float(start.elapsed_time(end))
        
        start.record()
        strings = self.net.entropy_bottleneck.compress(y)
        end.record()
        torch.cuda.synchronize()
        codec_ms = float(start.elapsed_time(end))
        
        pack = {
            "strings": strings,
            "shape": tuple(y.shape[-2:]),   # H, W of latent
            "batch_size": int(y.shape[0]),
        }
        return pack, inference_ms, codec_ms

    def decompress_cpulossless(self, pack: Dict[str, Any]) -> torch.Tensor:
        strings = pack["strings"]
        shape = pack["shape"]
        
        inference_ms = 0.0
        codec_ms = 0.0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        y_hat = self.net.entropy_bottleneck.decompress(strings, shape)
        end.record()
        torch.cuda.synchronize()
        codec_ms = float(start.elapsed_time(end))
        
        start.record()

        if not isinstance(y_hat, torch.Tensor):
            raise TypeError("entropy_bottleneck.decompress must return a torch.Tensor.")

        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)

        y_hat = self._ensure_cuda_contiguous(y_hat)
        x_hat = self.runners["gs"](y_hat)

        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs_runner must return a torch.Tensor.")
        x_hat = self._ensure_cuda_contiguous(x_hat)
        end.record()
        torch.cuda.synchronize()
        inference_ms = float(start.elapsed_time(end))
        return x_hat, inference_ms, codec_ms

    
    
