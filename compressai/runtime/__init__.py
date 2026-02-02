# compressai/runtime/__init__.py
from __future__ import annotations

import torch
from typing import Any

from .config import RuntimeConfig
from .adapters.cnn_simple import SimpleCnnAdapter
from .engines.factory import create_engine   

def build_runtime(net, codec: Any, config: RuntimeConfig):
    runners = {}
    
    if not config.trt_engines:
        raise ValueError("TRT mode requires config.trt_engines")

    from .backends.trt_loader import TRTRunnerFactory, TRTEngineSpec
    fac = TRTRunnerFactory()

    for name, engine_path in config.trt_engines.items():
        spec = TRTEngineSpec(engine_path=engine_path)
        mod = fac.load(spec)
        # runners[name] = lambda x, m=mod: m(x)
        runners[name] = lambda *args, m=mod: m(*args)

    engine = create_engine(
        config.model_name,
        net=net,
        codec=codec,
        runners=runners,

        codec_input_dtype=config.codec_input_dtype,
        gs_input_dtype=config.gs_input_dtype,
        ga_input_dtype=config.ga_input_dtype,
        ha_input_dtype=getattr(config, "ha_input_dtype", None),
        hs_input_dtype=getattr(config, "hs_input_dtype", None),

        # ---- TCM new ----
        h_mean_s_input_dtype=getattr(config, "h_mean_s_input_dtype", None),
        h_scale_s_input_dtype=getattr(config, "h_scale_s_input_dtype", None),

        atten_mean_input_dtypes=getattr(config, "atten_mean_input_dtypes", None),
        atten_scale_input_dtypes=getattr(config, "atten_scale_input_dtypes", None),
        cc_mean_input_dtypes=getattr(config, "cc_mean_input_dtypes", None),
        cc_scale_input_dtypes=getattr(config, "cc_scale_input_dtypes", None),
        lrp_input_dtypes=getattr(config, "lrp_input_dtypes", None),
        
        # ---- DCAE new ----
        h_z_s1_input_dtype=getattr(config, "h_z_s1_input_dtype", None),
        h_z_s2_input_dtype=getattr(config, "h_z_s2_input_dtype", None),
        dt_ca_input_dtypes=getattr(config, "dt_ca_input_dtypes", None),
    )
    return engine