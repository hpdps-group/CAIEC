# compressai/runtime/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import torch

Precision = Literal["fp32", "fp16", "fp8"]

@dataclass(frozen=True)
class RuntimeConfig:
    model_name: str = "bmshj2018_factorized"
    trt_engines: Optional[Dict[str, str]] = None

    # Optional: whether to require engines exist (recommended True)
    require_engine: bool = True
    ga_input_dtype: torch.dtype = torch.float32
    gs_input_dtype: torch.dtype = torch.float32
    ha_input_dtype: torch.dtype = torch.float32
    hs_input_dtype: torch.dtype = torch.float32
    codec_input_dtype: torch.dtype = torch.float32
    
    # ---- TCM 新增 dtype 配置（默认 None，不影响旧模型）----
    h_mean_s_input_dtype: Optional[torch.dtype] = None
    h_scale_s_input_dtype: Optional[torch.dtype] = None

    atten_mean_input_dtypes: Optional[List[torch.dtype]] = None,
    atten_scale_input_dtypes: Optional[List[torch.dtype]] = None,
    cc_mean_input_dtypes: Optional[List[torch.dtype]] = None,
    cc_scale_input_dtypes: Optional[List[torch.dtype]] = None,
    lrp_input_dtypes: Optional[List[torch.dtype]] = None,
    
    # ---- DCAE 新增 dtype 配置（默认 None，不影响旧模型）----
    h_z_s1_input_dtype: Optional[torch.dtype] = None
    h_z_s2_input_dtype: Optional[torch.dtype] = None
    dt_ca_input_dtypes: Optional[List[torch.dtype]] = None,
