# compressai/runtime/codecs/eb_packed_gpu.py
from __future__ import annotations

import pickle
from typing import Dict
import torch

from .base import Codec, Pack

def _packed_payload_bytes(packed) -> int:
    return int(packed.packed.numel())

def _bytes_of_state(obj) -> int:
    if obj is None:
        return 0
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0

class GpuPackedEntropyCodec(Codec):
    """
    GPU PackedANS codec for CompressAI EntropyModel.
    - Enforces FP32 input latent for lossless coding.
    - Handles stream synchronization internally (TRT stream <-> default stream),
      since EB implementations often assume default stream.
    """

    def __init__(self, entropy_bottleneck, gaussian_conditional=None, P: int = 64, error: float = None, latent_norm: bool = False):
        self.P = int(P)
        self.eb = entropy_bottleneck
        self.eb.use_gpu_ans = True
        self.eb.gpu_ans_parallelism = int(P)
        self.gaussian_conditional = gaussian_conditional
        if self.gaussian_conditional is not None:
            self.gaussian_conditional.use_gpu_ans = True
            self.gaussian_conditional.gpu_ans_parallelism = int(P)

    @torch.no_grad()
    def compress(self, y_fp32: torch.Tensor, params=None, means = None, error=None, min_val=None, max_val=None) -> Pack:
        assert y_fp32.dtype == torch.float32

        if params is None:
            # EB path (factorized 或 hyperprior 的 z)
            strings = self.eb.compress(y_fp32)
            return {"strings": strings, "state": {"size_hw": y_fp32.shape[-2:]}}
        else:
            # GC path (hyperprior 的 y)
            if self.gaussian_conditional is None:
                raise ValueError("gaussian_conditional is required when params is provided.")
            if means is not None:
                strings = self.gaussian_conditional.compress(y_fp32, params, means=means)
            else:
                strings = self.gaussian_conditional.compress(y_fp32, params)
            return {"strings": strings, "state": {"size_hw": y_fp32.shape[-2:]}}


    @torch.no_grad()
    def decompress(self, pack: Pack, params=None, dtype=torch.float, means = None) -> torch.Tensor:
        strings = pack["strings"]
        size_hw = pack["state"]["size_hw"]

        if params is None:
            # print("In codec.decompress   error:", error, "min_val:", min_val, "max_val:", max_val)
            y_hat = self.eb.decompress(strings, size_hw)
            return y_hat
        else:
            if self.gaussian_conditional is None:
                raise ValueError("gaussian_conditional is required when params is provided.")
            y_hat = self.gaussian_conditional.decompress(strings, params, dtype, means)
            return y_hat

    # def pack_bytes(self, pack: Pack) -> Dict[str, float]:
    #     payload = float(_packed_payload_bytes(pack.get("strings", None)))
    #     state_bytes = float(_bytes_of_state(pack.get("state", None)))
    #     return {"strings_bytes": payload, "state_bytes": state_bytes}
    def pack_bytes(self, pack) -> Dict[str, float]:
        """
        Support:
        - single Pack: {"strings": packed, "state": ...}
        - multi-pack dict: {"y": Pack, "z": Pack, ...}
        Returns total bytes and (optionally) per-stream breakdown.
        """
        # Case 1: single Pack
        if isinstance(pack, dict) and ("strings" in pack or "state" in pack):
            strings = pack.get("strings", None)
            payload = float(_packed_payload_bytes(strings)) if strings is not None else 0.0
            state_bytes = float(_bytes_of_state(pack.get("state", None)))
            return {"strings_bytes": payload, "state_bytes": state_bytes}

        # Case 2: multi-stream packs
        if isinstance(pack, dict):
            total_strings = 0.0
            total_state = 0.0
            out: Dict[str, float] = {}

            for k, v in pack.items():

                # --- NEW: handle TCM-style per-slice list-of-strings/state directly ---
                if isinstance(v, dict) and isinstance(v.get("strings", None), (list, tuple)):
                    strings_list = v.get("strings", [])
                    s = float(sum(_packed_payload_bytes(si) for si in strings_list))
                    t = float(_bytes_of_state(v.get("state", None)))
                else:
                    # original behavior
                    b = self.pack_bytes(v)  # recurse
                    s = float(b.get("strings_bytes", 0.0))
                    t = float(b.get("state_bytes", 0.0))

                total_strings += s
                total_state += t
                out[f"{k}_strings_bytes"] = s
                out[f"{k}_state_bytes"] = t

            out["strings_bytes"] = total_strings
            out["state_bytes"] = total_state
            return out

        # Unknown type
        return {"strings_bytes": 0.0, "state_bytes": 0.0}

