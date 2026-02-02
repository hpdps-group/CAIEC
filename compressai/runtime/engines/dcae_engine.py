# compressai/runtime/engines/dcae_engine.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import torch

from .base import CnnEngine
from ..codecs.compress_packed_gpu import GpuPackedEntropyCodec
from compressai.models import CompressionModel  # or from your dcae module import DCAE
from compressai.entropy_models import (
    _gpu_ans_encode_with_indexes_tight,
    _gpu_ans_decode_with_indexes_tight,
)


class DCAEEngine(CnnEngine):
    """
    Engine for DCAE-style hyperprior + slice-wise GaussianConditional.

    y: per-slice tight-ANS stream (aligned with TCMEngine)
    z: EB path via GpuPackedEntropyCodec
    """

    def __init__(
        self,
        net: CompressionModel,  # ideally DCAE
        codec: GpuPackedEntropyCodec,
        runners: Dict[str, Any],
        *,
        codec_input_dtype: torch.dtype = torch.float32,
        gs_input_dtype: torch.dtype = torch.float16,
        ga_input_dtype: Optional[torch.dtype] = None,
        ha_input_dtype: Optional[torch.dtype] = None,
        h_z_s1_input_dtype: Optional[torch.dtype] = None,
        h_z_s2_input_dtype: Optional[torch.dtype] = None,
        # per-slice
        dt_ca_input_dtypes: Optional[List[torch.dtype]] = None,
        cc_mean_input_dtypes: Optional[List[torch.dtype]] = None,
        cc_scale_input_dtypes: Optional[List[torch.dtype]] = None,
        lrp_input_dtypes: Optional[List[torch.dtype]] = None,
        **kwargs,
    ):
        self.codec = codec
        self.runners = runners

        self.codec_input_dtype = codec_input_dtype
        self.gs_input_dtype = gs_input_dtype
        self.ga_input_dtype = ga_input_dtype
        self.ha_input_dtype = ha_input_dtype
        self.h_z_s1_input_dtype = h_z_s1_input_dtype
        self.h_z_s2_input_dtype = h_z_s2_input_dtype

        self.dt_ca_input_dtypes = dt_ca_input_dtypes
        self.cc_mean_input_dtypes = cc_mean_input_dtypes
        self.cc_scale_input_dtypes = cc_scale_input_dtypes
        self.lrp_input_dtypes = lrp_input_dtypes

        # DCAE attributes we need
        if not hasattr(net, "num_slices"):
            raise ValueError("DCAEEngine expects net.num_slices")
        if not hasattr(net, "max_support_slices"):
            raise ValueError("DCAEEngine expects net.max_support_slices")

        self.num_slices = int(net.num_slices)
        self.max_support_slices = int(net.max_support_slices)

        # base runners required
        for k in ("ga", "ha", "h_z_s1", "h_z_s2", "gs"):
            if k not in self.runners:
                raise ValueError(f"DCAEEngine requires runner '{k}'")

        # per-slice runners required
        for i in range(self.num_slices):
            for k in (f"dt_cross_attention_{i}", f"cc_mean_{i}", f"cc_scale_{i}", f"lrp_transforms_{i}"):
                if k not in self.runners:
                    raise ValueError(f"DCAEEngine requires runner '{k}'")

        # dt buffer/weight: we assume net.dt exists and is a Parameter [dict_num, dict_dim]
        if not hasattr(net, "dt"):
            raise ValueError("DCAEEngine expects net.dt (dictionary parameter)")

        self._dt_param = net.dt  # keep reference; it will live on whatever device net lives on

        if self.codec is None:
            raise ValueError("GpuPackedEntropyCodec is required for DCAEEngine")

    def _cast(self, x: torch.Tensor, dt: Optional[torch.dtype]) -> torch.Tensor:
        if dt is not None and x.dtype != dt:
            return x.to(dt)
        return x

    def _repeat_dt(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # net.dt is [dict_num, dict_dim] -> repeat to [B, dict_num, dict_dim]
        dt = self._dt_param
        if not isinstance(dt, torch.Tensor):
            raise TypeError("net.dt must be a torch.Tensor/Parameter")
        if dt.device != device:
            dt = dt.to(device)
        if dt.dtype != dtype:
            dt = dt.to(dtype)
        dt = dt.unsqueeze(0).expand(batch, -1, -1).contiguous()
        return dt

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
        B = int(x.size(0))

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

        z_hat = self.codec.decompress(z_pack)
        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # hs -> latent params
        # -------------------------
        z_hat_hs_scale = self._cast(z_hat, self.h_z_s1_input_dtype)
        z_hat_hs_mean = self._cast(z_hat, self.h_z_s2_input_dtype)

        latent_scales = self.runners["h_z_s1"](z_hat_hs_scale)
        latent_means = self.runners["h_z_s2"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_z_s1/h_z_s2 runners must return torch.Tensor")

        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means = self._ensure_cuda_contiguous(latent_means)

        # -------------------------
        # prepare GC tables on GPU (int32, contiguous)
        # -------------------------
        cdfs_i32 = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32 = self.codec.gaussian_conditional._offset

        if cdfs_i32.dtype != torch.int32:
            cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32:
            cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:
            offsets_i32 = offsets_i32.to(torch.int32)

        cdfs_i32 = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32 = self._ensure_cuda_contiguous(offsets_i32)

        # -------------------------
        # dt (dictionary) for cross-attention
        # -------------------------
        # dtype choice: align with latent_means/scales dtype by default (safe),
        # but if you want strict control, wire a separate dt dtype argument.
        dt = self._repeat_dt(B, device=y.device, dtype=latent_means.dtype)

        # -------------------------
        # slice loop (sequential due to context dependency)
        # y: per-slice tight stream
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

            # query = [latent_scales, latent_means, support_slices]
            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            query_in = self._cast(query, self.dt_ca_input_dtypes[slice_index])

            # dict_info = dt_cross_attention(query, dt)
            dict_info = self.runners[f"dt_cross_attention_{slice_index}"](query_in, dt)
            if not isinstance(dict_info, torch.Tensor):
                raise TypeError(f"dt_ca[{slice_index}] must return torch.Tensor")

            # support = [query, dict_info]
            support = torch.cat([query] + [dict_info], dim=1)

            # mu, scale
            support_mu_in = self._cast(support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](support_mu_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            support_sc_in = self._cast(support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](support_sc_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            # build indexes / quantize symbols (FP32 recommended)
            scale_fp32 = self._cast(scale, self.codec_input_dtype)
            mu_fp32 = self._cast(mu, self.codec_input_dtype)

            indexes = self.codec.gaussian_conditional.build_indexes(scale_fp32)
            y_q_slice = self.codec.gaussian_conditional.quantize(y_slice, "symbols", mu_fp32)

            y_q_slice_i32 = self._ensure_cuda_contiguous(y_q_slice)
            indexes_i32 = self._ensure_cuda_contiguous(indexes)

            # per-slice tight encode
            tight = _gpu_ans_encode_with_indexes_tight(
                y_q_slice_i32, indexes_i32, cdfs_i32, cdf_sizes_i32, offsets_i32,
                parallelism=self.codec.P,
            )
            y_tight_list.append(tight)
            y_state_list.append({"size_hw": y_shape})

            # reconstruct y_hat_slice (for next slice context)
            y_hat_slice = y_q_slice_i32 + mu_fp32

            # lrp refinement: lrp_support = cat([support, y_hat_slice])
            lrp_support = torch.cat(
                [
                    self._cast(support, self.lrp_input_dtypes[slice_index]),
                    self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index]),
                ],
                dim=1,
            )
            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

        y_pack = {"strings": y_tight_list, "state": y_state_list}
        return {"y": y_pack, "z": z_pack}

    def decompress(self, pack: Dict[str, Any]) -> torch.Tensor:
        if "y" not in pack or "z" not in pack:
            raise KeyError("DCAE pack must contain keys: 'y' and 'z'")

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
        z_hat_hs_scale = self._cast(z_hat, self.h_z_s1_input_dtype)
        z_hat_hs_mean = self._cast(z_hat, self.h_z_s2_input_dtype)

        latent_scales = self.runners["h_z_s1"](z_hat_hs_scale)
        latent_means = self.runners["h_z_s2"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_z_s1/h_z_s2 runners must return torch.Tensor")
        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means = self._ensure_cuda_contiguous(latent_means)

        # y spatial size (same heuristic as TCMEngine)
        y_shape = (int(z_hat.shape[2] * 4), int(z_hat.shape[3] * 4))
        B = int(z_hat.size(0))

        # dt dictionary
        dt = self._repeat_dt(B, device=z_hat.device, dtype=latent_means.dtype)

        # prepare GC tables
        cdfs_i32 = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32 = self.codec.gaussian_conditional._offset

        cdfs_i32 = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32 = self._ensure_cuda_contiguous(offsets_i32)

        # -------------------------
        # 3) slice-by-slice decode y
        # -------------------------
        y_hat_slices: list[torch.Tensor] = []

        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            query_in = self._cast(query, self.dt_ca_input_dtypes[slice_index])

            dict_info = self.runners[f"dt_cross_attention_{slice_index}"](query_in, dt)
            if not isinstance(dict_info, torch.Tensor):
                raise TypeError(f"dt_cross_attention[{slice_index}] must return torch.Tensor")

            support = torch.cat([query] + [dict_info], dim=1)

            support_mu_in = self._cast(support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](support_mu_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            support_sc_in = self._cast(support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](support_sc_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scale_in = self._cast(scale, self.codec_input_dtype)
            mu_in = self._cast(mu, self.codec_input_dtype)
            indexes = self.codec.gaussian_conditional.build_indexes(scale_in)

            tight_i = y_strings[slice_index]
            y_q = _gpu_ans_decode_with_indexes_tight(
                tight_i, indexes, cdfs_i32, cdf_sizes_i32, offsets_i32
            )
            # reshape to [B, Cs, H, W]
            y_q = y_q.reshape(B, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.codec.gaussian_conditional.dequantize(y_q, mu_in)

            # lrp refinement
            lrp_support = torch.cat(
                [
                    self._cast(support, self.lrp_input_dtypes[slice_index]),
                    self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index]),
                ],
                dim=1,
            )
            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

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

        stream = torch.cuda.default_stream()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)

        # -------------------------
        # inference: ga + ha + z_fp32 cast
        # -------------------------
        s.record(stream)

        x = self._ensure_cuda_contiguous(x)

        x = self._cast(x, self.ga_input_dtype)
        y = self.runners["ga"](x)
        if not isinstance(y, torch.Tensor):
            raise TypeError("ga runner must return torch.Tensor")
        y = self._ensure_cuda_contiguous(y)
        y_shape = y.shape[2:]
        B = int(x.size(0))

        y_ha = self._cast(y, self.ha_input_dtype)
        z = self.runners["ha"](y_ha)
        if not isinstance(z, torch.Tensor):
            raise TypeError("ha runner must return torch.Tensor")
        z = self._ensure_cuda_contiguous(z)

        z_fp32 = z.to(self.codec_input_dtype) if z.dtype != self.codec_input_dtype else z

        e.record(stream)
        e.synchronize()
        inference_ms += float(s.elapsed_time(e))

        # -------------------------
        # codec: z EB compress + decompress
        # -------------------------
        s.record(stream)
        z_pack = self.codec.compress(z_fp32)
        z_hat = self.codec.decompress(z_pack)
        e.record(stream)
        e.synchronize()
        codec_ms += float(s.elapsed_time(e))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # inference: latent params + GC tables + dt + loop non-tight parts
        # -------------------------
        s.record(stream)

        z_hat_hs_scale = self._cast(z_hat, self.h_z_s1_input_dtype)
        z_hat_hs_mean  = self._cast(z_hat, self.h_z_s2_input_dtype)

        latent_scales = self.runners["h_z_s1"](z_hat_hs_scale)
        latent_means  = self.runners["h_z_s2"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_z_s1/h_z_s2 runners must return torch.Tensor")

        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means  = self._ensure_cuda_contiguous(latent_means)

        # GC tables (int32, contiguous)
        cdfs_i32      = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32   = self.codec.gaussian_conditional._offset

        if cdfs_i32.dtype != torch.int32:
            cdfs_i32 = cdfs_i32.to(torch.int32)
        if cdf_sizes_i32.dtype != torch.int32:
            cdf_sizes_i32 = cdf_sizes_i32.to(torch.int32)
        if offsets_i32.dtype != torch.int32:
            offsets_i32 = offsets_i32.to(torch.int32)
            
        cdfs_i32      = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32   = self._ensure_cuda_contiguous(offsets_i32)

        # dt dictionary for cross-attention
        dt = self._repeat_dt(B, device=y.device, dtype=latent_means.dtype)

        # slice prep
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices: list[torch.Tensor] = []
        y_tight_list = []
        y_state_list = []

        e.record(stream)
        e.synchronize()
        inference_ms += float(s.elapsed_time(e))

        # -------------------------
        # slice loop:
        #   inference: attention/cc/build_indexes/quantize/reconstruct/lrp
        #   codec_ms: tight encode
        # -------------------------
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # -------- inference (pre-tight) --------
            s.record(stream)

            query = torch.cat([latent_scales] + [latent_means] + support_slices, dim=1)
            query_in = self._cast(query, self.dt_ca_input_dtypes[slice_index])

            dict_info = self.runners[f"dt_cross_attention_{slice_index}"](query_in, dt)
            if not isinstance(dict_info, torch.Tensor):
                raise TypeError(f"dt_ca[{slice_index}] must return torch.Tensor")

            support = torch.cat([query] + [dict_info], dim=1)

            support_mu_in = self._cast(support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](support_mu_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            support_sc_in = self._cast(support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](support_sc_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scale_fp32 = self._cast(scale, self.codec_input_dtype)
            mu_fp32 = self._cast(mu, self.codec_input_dtype)

            indexes = self.codec.gaussian_conditional.build_indexes(scale_fp32)
            y_q_slice = self.codec.gaussian_conditional.quantize(y_slice, "symbols", mu_fp32)

            y_q_slice_i32 = self._ensure_cuda_contiguous(y_q_slice)
            indexes_i32 = self._ensure_cuda_contiguous(indexes)

            e.record(stream)
            e.synchronize()
            inference_ms += float(s.elapsed_time(e))

            # -------- codec (tight encode) --------
            s.record(stream)
            tight = _gpu_ans_encode_with_indexes_tight(
                y_q_slice_i32, indexes_i32, cdfs_i32, cdf_sizes_i32, offsets_i32,
                parallelism=self.codec.P,
            )
            e.record(stream)
            e.synchronize()
            codec_ms += float(s.elapsed_time(e))

            y_tight_list.append(tight)
            y_state_list.append({"size_hw": y_shape})

            # -------- inference (reconstruct + lrp) --------
            s.record(stream)

            y_hat_slice = y_q_slice_i32 + mu_fp32

            # lrp refinement: lrp_support = cat([support, y_hat_slice])
            lrp_support = torch.cat(
                [
                    self._cast(support, self.lrp_input_dtypes[slice_index]),
                    self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index]),
                ],
                dim=1,
            )
            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp.to(y_hat_slice.dtype)

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

            e.record(stream)
            e.synchronize()
            inference_ms += float(s.elapsed_time(e))

        y_pack = {"strings": y_tight_list, "state": y_state_list}
        return {"y": y_pack, "z": z_pack}, inference_ms, codec_ms

    def decompress_time(self, pack: Dict[str, Any]) -> Tuple[torch.Tensor, float, float]:
        inference_ms = 0.0
        codec_ms = 0.0
        
        if "y" not in pack or "z" not in pack:
            raise KeyError("DCAE pack must contain keys: 'y' and 'z'")

        y_pack = pack["y"]
        if not isinstance(y_pack, dict) or "strings" not in y_pack:
            raise ValueError("pack['y'] must be a dict containing 'strings' list")
        y_strings = y_pack["strings"]
        if not isinstance(y_strings, (list, tuple)) or len(y_strings) != self.num_slices:
            raise ValueError(f"pack['y']['strings'] must be a list of length {self.num_slices}")

        stream = torch.cuda.default_stream()
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)

        # -------------------------
        # codec: z EB decompress
        # -------------------------
        s.record(stream)
        z_hat = self.codec.decompress(pack["z"])
        e.record(stream)
        e.synchronize()
        codec_ms += float(s.elapsed_time(e))

        if not isinstance(z_hat, torch.Tensor):
            raise TypeError("codec.decompress(z_pack) must return torch.Tensor")
        z_hat = self._ensure_cuda_contiguous(z_hat)

        # -------------------------
        # inference: latent params + tables + dt + loop non-tight parts
        # -------------------------
        s.record(stream)

        z_hat_hs_scale = self._cast(z_hat, self.h_z_s1_input_dtype)
        z_hat_hs_mean = self._cast(z_hat, self.h_z_s2_input_dtype)

        latent_scales = self.runners["h_z_s1"](z_hat_hs_scale)
        latent_means = self.runners["h_z_s2"](z_hat_hs_mean)
        if not isinstance(latent_scales, torch.Tensor) or not isinstance(latent_means, torch.Tensor):
            raise TypeError("h_z_s1/h_z_s2 runners must return torch.Tensor")
        latent_scales = self._ensure_cuda_contiguous(latent_scales)
        latent_means = self._ensure_cuda_contiguous(latent_means)

        # y spatial size (same heuristic as TCMEngine)
        y_shape = (int(z_hat.shape[2] * 4), int(z_hat.shape[3] * 4))
        B = int(z_hat.size(0))

        # dt dictionary
        dt = self._repeat_dt(B, device=z_hat.device, dtype=latent_means.dtype)

        # prepare GC tables
        cdfs_i32 = self.codec.gaussian_conditional._quantized_cdf
        cdf_sizes_i32 = self.codec.gaussian_conditional._cdf_length
        offsets_i32 = self.codec.gaussian_conditional._offset
        
        cdfs_i32 = self._ensure_cuda_contiguous(cdfs_i32)
        cdf_sizes_i32 = self._ensure_cuda_contiguous(cdf_sizes_i32)
        offsets_i32 = self._ensure_cuda_contiguous(offsets_i32)

        y_hat_slices: list[torch.Tensor] = []

        e.record(stream)
        e.synchronize()
        inference_ms += float(s.elapsed_time(e))

        # -------------------------
        # slice loop:
        #   inference: attention/cc/build_indexes/dequantize/lrp
        #   codec_ms: tight decode
        # -------------------------
        for slice_index in range(self.num_slices):
            support_slices = (
                y_hat_slices if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # -------- inference (pre-tight) --------
            s.record(stream)

            query = torch.cat([latent_scales, latent_means] + support_slices, dim=1)
            query_in = self._cast(query, self.dt_ca_input_dtypes[slice_index])

            dict_info = self.runners[f"dt_cross_attention_{slice_index}"](query_in, dt)
            if not isinstance(dict_info, torch.Tensor):
                raise TypeError(f"dt_cross_attention[{slice_index}] must return torch.Tensor")

            support = torch.cat([query] + [dict_info], dim=1)

            support_mu_in = self._cast(support, self.cc_mean_input_dtypes[slice_index])
            mu = self.runners[f"cc_mean_{slice_index}"](support_mu_in)
            if not isinstance(mu, torch.Tensor):
                raise TypeError(f"cc_mean[{slice_index}] must return torch.Tensor")
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            support_sc_in = self._cast(support, self.cc_scale_input_dtypes[slice_index])
            scale = self.runners[f"cc_scale_{slice_index}"](support_sc_in)
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"cc_scale[{slice_index}] must return torch.Tensor")
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            scale_in = self._cast(scale, self.codec_input_dtype)
            mu_in = self._cast(mu, self.codec_input_dtype)
            indexes = self.codec.gaussian_conditional.build_indexes(scale_in)

            e.record(stream)
            e.synchronize()
            inference_ms += float(s.elapsed_time(e))

            # -------- codec (tight decode) --------
            tight_i = y_strings[slice_index]
            s.record(stream)
            y_q = _gpu_ans_decode_with_indexes_tight(
                tight_i, indexes, cdfs_i32, cdf_sizes_i32, offsets_i32
            )
            e.record(stream)
            e.synchronize()
            codec_ms += float(s.elapsed_time(e))

            # -------- inference (reshape + dequantize + lrp) --------
            s.record(stream)

            y_q = y_q.reshape(B, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.codec.gaussian_conditional.dequantize(y_q, mu_in)

            # lrp refinement
            lrp_support = torch.cat(
                [
                    self._cast(support, self.lrp_input_dtypes[slice_index]),
                    self._cast(y_hat_slice, self.lrp_input_dtypes[slice_index]),
                ],
                dim=1,
            )
            lrp = self.runners[f"lrp_transforms_{slice_index}"](lrp_support)
            if not isinstance(lrp, torch.Tensor):
                raise TypeError(f"lrp_transforms[{slice_index}] must return torch.Tensor")
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(self._ensure_cuda_contiguous(y_hat_slice))

            e.record(stream)
            e.synchronize()
            inference_ms += float(s.elapsed_time(e))

        # -------------------------
        # inference: concat + gs
        # -------------------------
        s.record(stream)
        y_hat = torch.cat(y_hat_slices, dim=1)

        if y_hat.dtype != self.gs_input_dtype:
            y_hat = y_hat.to(self.gs_input_dtype)
        x_hat = self.runners["gs"](y_hat).clamp_(0, 1)
        if not isinstance(x_hat, torch.Tensor):
            raise TypeError("gs runner must return torch.Tensor")

        x_hat = self._ensure_cuda_contiguous(x_hat)

        e.record(stream)
        e.synchronize()
        inference_ms += float(s.elapsed_time(e))

        return x_hat, inference_ms, codec_ms
