# compressai/runtime/engines/hyperprior_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.models import TCM
from compressai.entropy_models import _gpu_ans_encode_with_indexes_tight, _gpu_ans_decode_with_indexes_tight


class TCMEngine(CnnEngine):
    def __init__(
        self,
        net: TCM,
        codec: GpuPackedEntropyCodec,
        runners: Dict[str, Any],
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        ga_input_dtype: Optional[torch.dtype] = None,
        ha_input_dtype: Optional[torch.dtype] = None,
        h_mean_s_input_dtype: Optional[torch.dtype] = None,
        h_scale_s_input_dtype: Optional[torch.dtype] = None,
        atten_mean_input_dtypes: Optional[List[torch.dtype]] = None,
        atten_scale_input_dtypes: Optional[List[torch.dtype]] = None,
        cc_mean_input_dtypes: Optional[List[torch.dtype]] = None,
        cc_scale_input_dtypes: Optional[List[torch.dtype]] = None,
        lrp_input_dtypes: Optional[List[torch.dtype]] = None,
        **kwargs,
    ):
        self.codec = codec
        self.runners = runners  # expects keys: ga, gs, ha, hs

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype
        self.ha_input_dtype = ha_input_dtype
        self.h_mean_s_input_dtype = h_mean_s_input_dtype
        self.h_scale_s_input_dtype = h_scale_s_input_dtype
        
        self.atten_mean_input_dtypes = atten_mean_input_dtypes
        self.atten_scale_input_dtypes = atten_scale_input_dtypes
        self.cc_mean_input_dtypes = cc_mean_input_dtypes
        self.cc_scale_input_dtypes = cc_scale_input_dtypes
        self.lrp_input_dtypes = lrp_input_dtypes
        
        self.num_slices = net.num_slices
        self.max_support_slices = net.max_support_slices
        
        # required base runners
        for k in ("ga", "ha", "h_mean_s", "h_scale_s", "gs"):
            if k not in self.runners:
                raise ValueError(f"TCMEngine requires runner '{k}'")

        # required per-slice runners
        for i in range(self.num_slices):
            for k in (f"atten_mean_{i}", f"atten_scale_{i}", f"cc_mean_{i}", f"cc_scale_{i}", f"lrp_transforms_{i}"):
                if k not in self.runners:
                    raise ValueError(f"TCMEngine requires runner '{k}'")

        # must have gaussian_conditional in codec for GC path
        if self.codec is None:
            raise ValueError("GpuPackedEntropyCodec is required for TCMEngine")

    def _cast(self, x: torch.Tensor, dt: Optional[torch.dtype]) -> torch.Tensor:
        if dt is not None and x.dtype != dt:
            return x.to(dt)
        return x

    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        x = self._ensure_cuda_contiguous(x)

        # -------------------------
        # ga
        # -------------------------
        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)
        y_shape = y.shape[2:]  # (H, W)

        # -------------------------
        # ha
        # -------------------------
        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](y_ha)
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        # -------------------------
        # z -> EB codec (FP32)
        # -------------------------
        z_fp32 = z.to(self.codec_input_dtype) if z.dtype != self.codec_input_dtype else z
        z_pack = self.codec.compress(z_fp32)

        # z_hat
        z_hat = self.codec.decompress(z_pack)
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # hs -> latent params
        # -------------------------
        z_hat_hs_scale = self._cast(z_hat, self.h_scale_s_input_dtype)
        z_hat_hs_mean  = self._cast(z_hat, self.h_mean_s_input_dtype)

        latent_scales = self.runners["h_scale_s"](z_hat_hs_scale)
        latent_means  = self.runners["h_mean_s"](z_hat_hs_mean)

        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("hs runners must return torch.Tensor")

        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means  = self._ensure_cuda_contiguous(latent_means)

        # -------------------------
        # prepare GC tables on GPU (int32, contiguous)
        # -------------------------
        # IMPORTANT: do NOT use .tolist() here (kills throughput)
        cdfs_i32       = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32  = self.codec.gaussian_conditional._cdf_length
        offsets_i32    = self.codec.gaussian_conditional._offset

        # make sure dtype/device/contig
        if cdfs_i32.dtype != torch.int32:      cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32: cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:   offsets_i32 = offsets_i32.to(torch.int32)

        cdfs_i32      = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32   = self._ensure_cuda_contiguous(offsets_i32)

        # -------------------------
        # slice loop (must be sequential due to dependency)
        # but entropy coding per slice is batch-parallel tight
        # -------------------------
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices: list[torch.Tensor] = []

        y_tight_list = []
        y_state_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # ---- mean path ----
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support_in = self._cast(mean_support, self.atten_mean_input_dtypes[slice_index])
            mean_support = self.runners[f"atten_mean_{slice_index}"](mean_support_in)
            mean_support_in = self._cast(mean_support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](mean_support_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            # ---- scale path ----
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support_in = self._cast(scale_support, self.atten_scale_input_dtypes[slice_index])
            scale_support = self.runners[f"atten_scale_{slice_index}"](scale_support_in)
            scale_support_in = self._cast(scale_support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](scale_support_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # ---- build indexes / quantize symbols (FP32 recommended) ----
            scale_fp32 = self._cast(scale, self.codec_input_dtype)
            mu_fp32    = self._cast(mu,    self.codec_input_dtype)

            indexes = self.codec.gaussian_conditional.build_indexes(scale_fp32)
            # quantize -> int symbols
            y_q_slice = self.codec.gaussian_conditional.quantize(y_slice, "symbols", mu_fp32)

            # tight encoder expects CUDA int32 contiguous
            if y_q_slice.dtype != torch.int32:
                y_q_slice_i32 = y_q_slice.to(torch.int32)
            else:
                y_q_slice_i32 = y_q_slice
            if indexes.dtype != torch.int32:
                indexes_i32 = indexes.to(torch.int32)
            else:
                indexes_i32 = indexes

            y_q_slice_i32 = self._ensure_cuda_contiguous(y_q_slice_i32)
            indexes_i32   = self._ensure_cuda_contiguous(indexes_i32)

            # ---- per-slice tight encode (batch-parallel) ----
            tight = _gpu_ans_encode_with_indexes_tight(
                y_q_slice_i32,          # CUDA int32 [B,Cs,H,W]
                indexes_i32,            # CUDA int32 [B,Cs,H,W]
                cdfs_i32, cdf_sizes_i32, offsets_i32,
                parallelism=self.codec.P,
            )

            y_tight_list.append(tight)
            y_state_list.append({"size_hw": y_shape})

            # ---- reconstruct y_hat_slice for next slice's context ----
            # dequantize: y_hat = y_q + mu
            y_hat_slice = y_q_slice_i32.to(mu_fp32.dtype) + mu_fp32

            # ---- lrp refinement (same as original) ----
            mean_support_lrp = self._cast(mean_support, self.lrp_input_dtypes[slice_index])
            y_hat_lrp        = self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index])
            lrp_support = torch.cat([mean_support_lrp, y_hat_lrp], dim=1)

            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

        # -------------------------
        # pack output
        # -------------------------
        y_pack = {
            "strings": y_tight_list,   # [TightANS, TightANS, ...] length = num_slices
            "state":   y_state_list,   # [{"size_hw": ...}, ...]
        }
        return {"y": y_pack, "z": z_pack}


    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        y_pack = pack["y"]
        if not isinstance(y_pack, dict) or "strings" not in y_pack:
            raise ValueError("pack['y'] must be a dict containing 'strings' list")

        y_strings = y_pack["strings"]
        if not isinstance(y_strings, (list, tuple)) or len(y_strings) != self.num_slices:
            raise ValueError(f"pack['y']['strings'] must be a list of length {self.num_slices}")

        # -------------------------
        # 1) z_hat
        # -------------------------
        z_hat = self.codec.decompress(pack["z"])
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # 2) latent params from z_hat
        # -------------------------
        z_hat_hs_scale = self._cast(z_hat, self.h_scale_s_input_dtype)
        z_hat_hs_mean  = self._cast(z_hat, self.h_mean_s_input_dtype)

        # ⚠️ 你 compress 里 runners 用的是 "h_scale_s"/"h_mean_s"
        latent_scales = self.runners["h_scale_s"](z_hat_hs_scale)
        latent_means  = self.runners["h_mean_s"](z_hat_hs_mean)

        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_scale_s/h_mean_s runners must return torch.Tensor")
        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means  = self._ensure_cuda_contiguous(latent_means)

        # # y spatial size：优先从 pack 里拿（更稳），否则按 z_hat * 4
        # if isinstance(pack["y"], dict) and "state" in pack["y"] and "size_hw" in pack["y"]["state"]:
        #     y_shape = tuple(pack["y"]["state"]["size_hw"])
        # else:
        y_shape = (int(z_hat.shape[-2] * 4), int(z_hat.shape[-1] * 4))

        # -------------------------
        # 3) slice-by-slice decode y
        # -------------------------
        y_hat_slices = []
        cdfs_i32       = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32  = self.codec.gaussian_conditional._cdf_length
        offsets_i32    = self.codec.gaussian_conditional._offset
        
        if cdfs_i32.dtype != torch.int32:      cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32: cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:   offsets_i32 = offsets_i32.to(torch.int32)

        cdfs_i32      = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32   = self._ensure_cuda_contiguous(offsets_i32)

        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # ---- mean path ----
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support_in = self._cast(mean_support, self.atten_mean_input_dtypes[slice_index])
            mean_support = self.runners[f"atten_mean_{slice_index}"](mean_support_in)

            mean_support_in = self._cast(mean_support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](mean_support_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean runner[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # ---- scale path ----
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support_in = self._cast(scale_support, self.atten_scale_input_dtypes[slice_index])
            scale_support = self.runners[f"atten_scale_{slice_index}"](scale_support_in)

            scale_support_in = self._cast(scale_support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](scale_support_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale runner[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            # ---- indexes ----
            scale_in = self._cast(scale, self.codec_input_dtype)  # FP32 recommended
            mu_in    = self._cast(mu,    self.codec_input_dtype)  # FP32 recommended
            indexes  = self.codec.gaussian_conditional.build_indexes(scale_in)

            # ---- decode this slice from the SAME y stream ----
            tight_i = y_strings[slice_index]
            
            y_q = _gpu_ans_decode_with_indexes_tight(
                tight_i, indexes, cdfs_i32, cdf_sizes_i32, offsets_i32
            )
            B = int(indexes.size(0))
            y_q = y_q.reshape(B, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.codec.gaussian_conditional.dequantize(y_q, mu_in)

            # ---- lrp refinement ----
            mean_support_lrp = self._cast(mean_support, self.lrp_input_dtypes[slice_index])
            y_hat_slice_lrp  = self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index])
            lrp_support = torch.cat([mean_support_lrp, y_hat_slice_lrp], dim=1)

            # 你 compress 里这里用的是 self.runners["lrp_transforms"][i]
            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms runner[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)

            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        # -------------------------
        # 4) gs
        # -------------------------
        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)

        x_hat = self.runners["gs"](y_hat).clamp_(0, 1)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs runner must return torch.Tensor")

        return self._ensure_cuda_contiguous(x_hat)
    
    def compress_time(self, x: torch.Tensor) -> Tuple[Dict[str, Any], float, float]:
        inference_ms = 0.0
        codec_ms = 0.0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # -------------------------
        # inference: ga + ha + prep
        # -------------------------
        start.record()

        x = self._ensure_cuda_contiguous(x)

        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)
        y_shape = y.shape[2:]

        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](y_ha)
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        z_fp32 = z.to(self.codec_input_dtype) if z.dtype != self.codec_input_dtype else z

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # -------------------------
        # codec: z EB compress + decompress
        # -------------------------
        start.record()
        z_pack = self.codec.compress(z_fp32)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        start.record()
        z_hat = self.codec.decompress(z_pack)
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # inference: hs + tables prep (non-tight)
        # -------------------------
        start.record()

        z_hat_hs_scale = self._cast(z_hat, self.h_scale_s_input_dtype)
        z_hat_hs_mean  = self._cast(z_hat, self.h_mean_s_input_dtype)

        latent_scales = self.runners["h_scale_s"](z_hat_hs_scale)
        latent_means  = self.runners["h_mean_s"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("hs runners must return torch.Tensor")

        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means  = self._ensure_cuda_contiguous(latent_means)

        cdfs_i32      = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32   = self.codec.gaussian_conditional._offset

        if cdfs_i32.dtype != torch.int32:      cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32: cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:   offsets_i32 = offsets_i32.to(torch.int32)

        cdfs_i32      = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32   = self._ensure_cuda_contiguous(offsets_i32)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices: list[torch.Tensor] = []

        y_tight_list = []
        y_state_list = []

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # -------------------------
        # slice loop:
        #   inference: mean/scale paths + build_indexes/quantize/lrp
        #   codec_ms: tight encode
        # -------------------------
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # ===== inference part (everything except tight) =====
            start.record()

            # mean path
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support_in = self._cast(mean_support, self.atten_mean_input_dtypes[slice_index])
            mean_support = self.runners[f"atten_mean_{slice_index}"](mean_support_in)
            mean_support_in = self._cast(mean_support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](mean_support_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            # scale path
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support_in = self._cast(scale_support, self.atten_scale_input_dtypes[slice_index])
            scale_support = self.runners[f"atten_scale_{slice_index}"](scale_support_in)
            scale_support_in = self._cast(scale_support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](scale_support_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # build indexes / quantize symbols
            scale_fp32 = self._cast(scale, self.codec_input_dtype)
            mu_fp32    = self._cast(mu,    self.codec_input_dtype)

            indexes = self.codec.gaussian_conditional.build_indexes(scale_fp32)
            y_q_slice = self.codec.gaussian_conditional.quantize(y_slice, "symbols", mu_fp32)

            y_q_slice_i32 = y_q_slice.to(torch.int32) if y_q_slice.dtype != torch.int32 else y_q_slice
            indexes_i32   = indexes.to(torch.int32)   if indexes.dtype   != torch.int32 else indexes

            y_q_slice_i32 = self._ensure_cuda_contiguous(y_q_slice_i32)
            indexes_i32   = self._ensure_cuda_contiguous(indexes_i32)

            end.record()
            torch.cuda.synchronize()
            inference_ms += float(start.elapsed_time(end))

            # ===== codec part: tight encode =====
            start.record()
            tight = _gpu_ans_encode_with_indexes_tight(
                y_q_slice_i32,
                indexes_i32,
                cdfs_i32, cdf_sizes_i32, offsets_i32,
                parallelism=self.codec.P,
            )
            end.record()
            torch.cuda.synchronize()
            codec_ms += float(start.elapsed_time(end))

            y_tight_list.append(tight)
            y_state_list.append({"size_hw": y_shape})

            # ===== inference part: reconstruct + lrp =====
            start.record()

            y_hat_slice = y_q_slice_i32.to(mu_fp32.dtype) + mu_fp32

            mean_support_lrp = self._cast(mean_support, self.lrp_input_dtypes[slice_index])
            y_hat_lrp        = self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index])
            lrp_support = torch.cat([mean_support_lrp, y_hat_lrp], dim=1)

            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

            end.record()
            torch.cuda.synchronize()
            inference_ms += float(start.elapsed_time(end))

        y_pack = {"strings": y_tight_list, "state": y_state_list}
        return {"y": y_pack, "z": z_pack}, inference_ms, codec_ms
    
    def decompress_time(self, pack: Dict[str, Any]) -> Tuple[torch.Tensor, float, float]:
        inference_ms = 0.0
        codec_ms = 0.0

        if "y" not in pack or "z" not in pack:
            raise KeyError("Hyperprior pack must contain keys: 'y' and 'z'")

        y_pack = pack["y"]
        if not isinstance(y_pack, dict) or "strings" not in y_pack:
            raise ValueError("pack['y'] must be a dict containing 'strings' list")

        y_strings = y_pack["strings"]
        if not isinstance(y_strings, (list, tuple)) or len(y_strings) != self.num_slices:
            raise ValueError(f"pack['y']['strings'] must be a list of length {self.num_slices}")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # -------------------------
        # codec: z EB decompress
        # -------------------------
        start.record()
        z_hat = self.codec.decompress(pack["z"])
        end.record()
        torch.cuda.synchronize()
        codec_ms += float(start.elapsed_time(end))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # inference: latent params + tables prep
        # -------------------------
        start.record()

        z_hat_hs_scale = self._cast(z_hat, self.h_scale_s_input_dtype)
        z_hat_hs_mean  = self._cast(z_hat, self.h_mean_s_input_dtype)

        latent_scales = self.runners["h_scale_s"](z_hat_hs_scale)
        latent_means  = self.runners["h_mean_s"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_scale_s/h_mean_s runners must return torch.Tensor")

        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means  = self._ensure_cuda_contiguous(latent_means)

        y_shape = (int(z_hat.shape[-2] * 4), int(z_hat.shape[-1] * 4))

        cdfs_i32      = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32   = self.codec.gaussian_conditional._offset

        if cdfs_i32.dtype != torch.int32:      cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32: cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:   offsets_i32 = offsets_i32.to(torch.int32)

        cdfs_i32      = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32   = self._ensure_cuda_contiguous(offsets_i32)

        y_hat_slices = []

        end.record()
        torch.cuda.synchronize()
        inference_ms += float(start.elapsed_time(end))

        # -------------------------
        # slice loop:
        #   inference: mean/scale paths + build_indexes + lrp + concat + gs
        #   codec_ms: tight decode
        # -------------------------
        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # ===== inference: everything except tight decode =====
            start.record()

            # mean path
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mean_support_in = self._cast(mean_support, self.atten_mean_input_dtypes[slice_index])
            mean_support = self.runners[f"atten_mean_{slice_index}"](mean_support_in)

            mean_support_in = self._cast(mean_support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](mean_support_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean runner[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # scale path
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale_support_in = self._cast(scale_support, self.atten_scale_input_dtypes[slice_index])
            scale_support = self.runners[f"atten_scale_{slice_index}"](scale_support_in)

            scale_support_in = self._cast(scale_support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](scale_support_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale runner[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            # indexes
            scale_in = self._cast(scale, self.codec_input_dtype)
            mu_in    = self._cast(mu,    self.codec_input_dtype)
            indexes  = self.codec.gaussian_conditional.build_indexes(scale_in)

            end.record()
            torch.cuda.synchronize()
            inference_ms += float(start.elapsed_time(end))

            # ===== codec: tight decode =====
            tight_i = y_strings[slice_index]
            start.record()
            y_q = _gpu_ans_decode_with_indexes_tight(
                tight_i, indexes, cdfs_i32, cdf_sizes_i32, offsets_i32
            )
            end.record()
            torch.cuda.synchronize()
            codec_ms += float(start.elapsed_time(end))

            # ===== inference: reshape + dequantize + lrp =====
            start.record()

            B = int(indexes.size(0))
            y_q = y_q.reshape(B, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.codec.gaussian_conditional.dequantize(y_q, mu_in)

            mean_support_lrp = self._cast(mean_support, self.lrp_input_dtypes[slice_index])
            y_hat_slice_lrp  = self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index])
            lrp_support = torch.cat([mean_support_lrp, y_hat_slice_lrp], dim=1)

            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms runner[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(y_hat_slice)

            end.record()
            torch.cuda.synchronize()
            inference_ms += float(start.elapsed_time(end))

        # finish (inference): concat + gs
        start.record()

        y_hat = torch.cat(y_hat_slices, dim=1)

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

        
