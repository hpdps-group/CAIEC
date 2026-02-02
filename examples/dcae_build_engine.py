import os
import subprocess
from typing import List, Optional

import numpy as np


# -----------------------------
# utils
# -----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def run_cmd(cmd: List[str]):
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def safe_name(comp: str) -> str:
    # filenames use underscore, keep consistent
    return comp.replace("/", "_").replace(".", "_")

def list_components(subonnx_dir: str) -> List[str]:
    """
    Discover components by scanning *_fp32.onnx in subonnx_dir.
    Component base name: "<comp>_fp32.onnx" -> "<comp>"
    """
    comps = set()
    for fn in os.listdir(subonnx_dir):
        if fn.endswith("_fp32.onnx"):
            comps.add(fn[:-len("_fp32.onnx")])
    return sorted(comps)

def modelopt_quant_fp8(fp32_onnx: str, out_fp8_qdq_onnx: str, calib_npy: str):
    if not calib_npy:
        raise ValueError("FP8 quant requires calib_npy")
    run_cmd([
        "python", "-m", "modelopt.onnx.quantization",
        f"--onnx={fp32_onnx}",
        "--quantize_mode=fp8",
        f"--calibration_data={calib_npy}",
        "--calibration_method=max",
        f"--output_path={out_fp8_qdq_onnx}",
    ])

def build_trt_engine(onnx_path: str, engine_path: str, precision: str):
    ensure_dir(os.path.dirname(engine_path))
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]

    if precision == "fp16":
        cmd += ["--fp16"]
    elif precision == "fp8":
        cmd += ["--stronglyTyped"]
    else:
        raise ValueError(precision)

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Failed to build {engine_path}: {e}")


# -----------------------------
# main (call this in your code)
# -----------------------------
def build_all_component_engines(
    *,
    model_tag: str,                 # e.g. "dcae-q1"
    subonnx_root: str,              # e.g. "/hwj/out_engines/subonnx"
    calib_root: str,                # e.g. "/hwj/out_engines/calib"
    engine_root: str,               # e.g. "/hwj/out_engines/engines"
    build_fp16: bool = True,
    build_fp8: bool = True,
    reuse_qdq: bool = True,
    skip_missing_calib_for_fp8: bool = False,
):
    """
    For every component under subonnx/<model_tag>:
      - fp16: build from <comp>_fp16.onnx -> engines/<model_tag>/<comp>/fp16.engine
      - fp8 : quantize <comp>_fp32.onnx using calib/<model_tag>/<comp>.npy -> <comp>_fp8_qdq.onnx
              then build -> engines/<model_tag>/<comp>/fp8.engine

    Notes:
      - trtexec shape is NOT specified (as you requested).
      - calib file name is assumed to be <comp>.npy under calib/<model_tag>.
    """
    subonnx_dir = os.path.join(subonnx_root, model_tag)
    calib_dir   = os.path.join(calib_root, model_tag)
    engine_dir  = os.path.join(engine_root, model_tag)

    if not os.path.isdir(subonnx_dir):
        raise FileNotFoundError(f"subonnx_dir not found: {subonnx_dir}")

    comps = list_components(subonnx_dir)
    if not comps:
        raise RuntimeError(f"No *_fp32.onnx found under: {subonnx_dir}")

    print(f"[Info] Found {len(comps)} components in: {subonnx_dir}")

    for comp in comps:
        fp32_onnx = os.path.join(subonnx_dir, f"{comp}_fp32.onnx")
        fp16_onnx = os.path.join(subonnx_dir, f"{comp}_fp16.onnx")

        calib_npy = os.path.join(calib_dir, f"{comp}_calib.npz")
        if not os.path.isfile(calib_npy):
            # fallback if caller passes dotted names etc.
            alt = os.path.join(calib_dir, f"{safe_name(comp)}.npz")
            if os.path.isfile(alt):
                calib_npy = alt

        # -------------------------
        # fp16
        # -------------------------
        if build_fp16:
            if not os.path.isfile(fp16_onnx):
                print(f"[Skip][FP16] {comp}: missing {fp16_onnx}")
            else:
                out_engine = os.path.join(engine_dir, comp, "fp16.engine")
                if os.path.isfile(out_engine):
                    print(f"[Reuse][FP16] {out_engine}")
                else:
                    print(f"\n[Build][FP16] {comp}")
                    build_trt_engine(fp16_onnx, out_engine, "fp16")

        # -------------------------
        # fp8
        # -------------------------
        if build_fp8:
            if not os.path.isfile(fp32_onnx):
                print(f"[Skip][FP8] {comp}: missing {fp32_onnx}")
                continue

            if not os.path.isfile(calib_npy):
                msg = f"[Missing][FP8] {comp}: calib not found: {calib_npy}"
                if skip_missing_calib_for_fp8:
                    print(msg + " (skip)")
                    continue
                raise FileNotFoundError(msg)

            # optional: sanity check calib load (helps debug early)
            _ = np.load(calib_npy)

            qdq_onnx = os.path.join(subonnx_dir, f"{comp}_fp8_qdq.onnx")
            if os.path.isfile(qdq_onnx) and reuse_qdq:
                print(f"[Reuse][FP8-QDQ] {qdq_onnx}")
            else:
                print(f"\n[Quant][FP8] {comp}")
                modelopt_quant_fp8(fp32_onnx, qdq_onnx, calib_npy)

            out_engine = os.path.join(engine_dir, comp, "fp8.engine")
            if os.path.isfile(out_engine):
                print(f"[Reuse][FP8] {out_engine}")
            else:
                print(f"[Build][FP8] {comp}")
                build_trt_engine(qdq_onnx, out_engine, "fp8")

    print("\nDone. Engines saved under:", engine_dir)


# -----------------------------
# example usage (edit paths)
# -----------------------------
build_all_component_engines(
    model_tag="dcae-q1",
    subonnx_root="/hwj/project/CompressAI-Science/examples/out_engines/subonnx",
    calib_root="/hwj/project/CompressAI-Science/examples/out_engines/calib",
    engine_root="/hwj/project/CompressAI-Science/examples/out_engines/engines",
    build_fp16=True,
    build_fp8=True,
    reuse_qdq=False,
    skip_missing_calib_for_fp8=False,
)
