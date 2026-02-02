import os
import json
import argparse
import subprocess
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnx
import torch

from onnxconverter_common import float16
from onnx import TensorProto
from onnx2torch import convert
from onnxsim import simplify

try:
    import onnxruntime as ort
except Exception:
    ort = None


# -----------------------------
# Helpers
# -----------------------------
def parse_shape(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.replace(" ", "").split(","))


def parse_component_config(s: str) -> Dict[str, str]:
    """
    "ga-fp8,gs-fp16,ha-fp8,hs-fp16" -> {"ga":"fp8","gs":"fp16","ha":"fp8","hs":"fp16"}
    """
    out = {}
    items = [x.strip() for x in s.split(",") if x.strip()]
    for it in items:
        comp, prec = it.split("-")
        comp = comp.lower()
        prec = prec.lower()
        if comp not in ("ga", "gs", "ha", "hs"):
            raise ValueError(f"Unknown component: {comp}")
        if prec not in ("fp16", "fp8"):
            raise ValueError(f"Unknown precision: {prec}")
        out[comp] = prec
    return out


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def run_cmd(cmd: List[str]):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_subgraph(full_onnx: str, out_onnx: str, inputs: List[str], outputs: List[str]):
    onnx.utils.extract_model(full_onnx, out_onnx, inputs, outputs)

def fix_onnx_input_shape_inplace(onnx_path: str, fixed_shape: Tuple[int, ...], input_index: int = 0):
    """
    Overwrite ONNX graph input dims with fixed dim_value.
    This is critical for TensorRT ConvTranspose which requires static channel dimension.
    """
    m = onnx.load(onnx_path)
    if len(m.graph.input) <= input_index:
        raise ValueError(f"{onnx_path}: no input at index {input_index}")

    inp = m.graph.input[input_index]
    dims = inp.type.tensor_type.shape.dim

    if len(dims) != len(fixed_shape):
        raise ValueError(f"{onnx_path}: rank mismatch, onnx rank={len(dims)} vs fixed={len(fixed_shape)}")

    for i, v in enumerate(fixed_shape):
        # clear dim_param and set dim_value
        dims[i].dim_param = ""
        dims[i].dim_value = int(v)

    onnx.save(m, onnx_path)
    print(f"[FixShape] {os.path.basename(onnx_path)} input={inp.name} -> {fixed_shape}")

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

def convert_fp32_onnx_to_fp16(fp32_onnx: str, fp16_onnx: str, keep_io_types: bool = False):
    """
    Generic FP32 ONNX -> FP16 ONNX conversion (no PyTorch model needed).
    keep_io_types:
      - False: model inputs/outputs become FP16 too (recommended for TRT fp16 engines)
      - True : keep inputs/outputs FP32, internal becomes FP16 (sometimes useful, but TRT boundary may stay FP32)
    """
    if os.path.isfile(fp16_onnx):
        print(f"[FP16] Reuse existing: {fp16_onnx}")
        return fp16_onnx

    print(f"[FP16] Converting FP32 ONNX -> FP16 ONNX: {fp16_onnx}")
    m = onnx.load(fp32_onnx)

    # Convert graph float tensors/initializers to float16
    m_fp16 = float16.convert_float_to_float16(
        m,
        keep_io_types=keep_io_types,
        # optional: disable shape inference if you hit issues
        # disable_shape_infer=True,
    )

    onnx.save(m_fp16, fp16_onnx)
    return fp16_onnx

def build_trt_engine(onnx_path: str, engine_path: str, precision: str, fixed_input_shape: Optional[Tuple[int, ...]] = None):
    ensure_dir(os.path.dirname(engine_path))
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]

    if precision == "fp16":
        cmd += ["--fp16"]
    elif precision == "fp8":
        cmd += ["--stronglyTyped"]
    else:
        raise ValueError(precision)
    
    # if fixed_input_shape is not None:
    #     m = onnx.load(onnx_path)
    #     in_name = m.graph.input[0].name
    #     shape_str = "x".join(str(x) for x in fixed_input_shape)
    #     cmd += [f"--shapes={in_name}:{shape_str}"]

    run_cmd(cmd)


def _infer_model_input_name(full_onnx_path: str) -> str:
    m = onnx.load(full_onnx_path)
    if len(m.graph.input) < 1:
        raise ValueError("ONNX model has no graph input")
    return m.graph.input[0].name


def _load_calib_x(calib_npy: str, input_shape: Optional[Tuple[int, ...]], max_samples: int) -> np.ndarray:
    x = np.load(calib_npy)
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    # If user gave input_shape, sanity check (optional)
    if input_shape is not None:
        if tuple(x.shape) != tuple(input_shape):
            # allow leading sample dim variants; but your case uses full batch already
            raise ValueError(f"calib_npy shape {x.shape} != --input_shape {input_shape}")

    # Limit samples if requested (assume batch dim 0)
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples].copy()

    return x


def _ort_session(onnx_path: str, prefer_cuda: bool = True):
    if ort is None:
        raise ImportError("onnxruntime is not installed. Please install onnxruntime-gpu or onnxruntime.")

    providers = []
    if prefer_cuda:
        # If CUDA EP is not available, ORT will throw; we fallback to CPU below.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    except Exception:
        # hard fallback to CPU
        return ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])


def generate_component_calib_from_full_onnx(
    full_onnx: str,
    model_input_name: str,
    tap_tensor_name: str,
    x_calib: np.ndarray,
    out_calib_npy: str,
    tmp_dir: str,
    *,
    prefer_cuda: bool = True,
):
    """
    Build a "tap" ONNX that outputs the internal tensor tap_tensor_name,
    run it with x_calib, and save the output as calib for that component.
    """
    ensure_dir(tmp_dir)
    tap_onnx = os.path.join(tmp_dir, f"tap_{_safe_filename(tap_tensor_name)}.onnx")

    # Build tap model: input is original model input, output is internal tensor
    if not os.path.isfile(tap_onnx):
        extract_subgraph(full_onnx, tap_onnx, [model_input_name], [tap_tensor_name])

    sess = _ort_session(tap_onnx, prefer_cuda=prefer_cuda)

    # ORT input name should match model_input_name; if not, use sess input name
    ort_in_name = sess.get_inputs()[0].name
    feed = {ort_in_name: x_calib}

    outs = sess.run(None, feed)
    if len(outs) != 1:
        raise RuntimeError(f"Tap model should produce 1 output, got {len(outs)}")
    arr = outs[0]
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    np.save(out_calib_npy, arr)
    return out_calib_npy, arr.shape


def _safe_filename(s: str) -> str:
    # make tensor name safe for filenames
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:180]


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_fp32", required=True, help="Full FP32 model ONNX path")
    ap.add_argument("--onnx_fp16", required=True, help="Full FP16 model ONNX path (if any component is fp16)")
    ap.add_argument("--input_shape", required=True, help="Input shape for model input, e.g. 512,3,128,128")
    ap.add_argument("--config", required=True, help="e.g. ga-fp8,gs-fp16,ha-fp8,hs-fp16")
    ap.add_argument("--boundaries", required=True, help="JSON file defining component input/output tensor names")
    ap.add_argument("--calib_npy", default="", help="Calibration npy for MODEL INPUT (x). Required if any FP8.")
    ap.add_argument("--out_dir", required=True, help="Output root dir for sub-onnx and engines")
    ap.add_argument("--model_tag", default="model", help="Tag for output folder naming")
    ap.add_argument("--max_calib_samples", type=int, default=256, help="Limit calibration batch dim to this number (0=no limit)")
    ap.add_argument("--prefer_cuda_ort", action="store_true", help="Prefer onnxruntime CUDA EP when generating internal calibs")
    args = ap.parse_args()

    input_shape = parse_shape(args.input_shape)
    cfg = parse_component_config(args.config)
    
    any_fp16 = any(v == "fp16" for v in cfg.values())
    full_fp16_onnx = ""

    full_fp16_onnx = args.onnx_fp16
    
    with open(args.boundaries, "r", encoding="utf-8") as f:
        boundaries = json.load(f)

    # output folders
    subonnx_dir = os.path.join(args.out_dir, "subonnx", args.model_tag)
    engine_dir  = os.path.join(args.out_dir, "engines", args.model_tag)
    calib_dir   = os.path.join(args.out_dir, "calib", args.model_tag)
    tap_dir     = os.path.join(args.out_dir, "tap_onnx", args.model_tag)
    ensure_dir(subonnx_dir)
    ensure_dir(engine_dir)
    ensure_dir(calib_dir)
    ensure_dir(tap_dir)

    # If any component uses fp8, we need base calib x
    any_fp8 = any(v == "fp8" for v in cfg.values())
    if any_fp8 and not args.calib_npy:
        raise ValueError("At least one component is fp8, so --calib_npy (MODEL INPUT x) is required.")

    x_calib = None
    if args.calib_npy:
        x_calib = _load_calib_x(args.calib_npy, input_shape, args.max_calib_samples)
    if x_calib is None and any_fp8:
        raise RuntimeError("Internal error: x_calib is None but use fp8 quant.")
    if x_calib is None:
        print("[Info] No fp8 calib provided; using random x to tap component shapes.")
        x_calib = np.random.randn(*input_shape).astype(np.float32)


    # Determine the full model input name (or allow boundaries override)
    model_input_name = boundaries.get("model_input", None) or _infer_model_input_name(args.onnx_fp32)

    # 1) extract component subgraphs from full onnx
    comp_fp16_onnx = {}
    comp_fp32_onnx = {}

    for comp in ("ga", "ha", "hs", "gs"):
        if comp not in boundaries:
            print(f"[Skip] boundaries missing component: {comp}")
            continue

        prec = cfg.get(comp, "fp16")

        if prec == "fp16":
            if not full_fp16_onnx:
                raise RuntimeError("Internal error: full_fp16_onnx is empty but fp16 component exists")
            src_full = full_fp16_onnx
            suffix = "fp16"
        else:
            src_full = args.onnx_fp32
            suffix = "fp32"

        comp_out = os.path.join(subonnx_dir, f"{comp}_{suffix}.onnx")
        extract_subgraph(
            src_full,
            comp_out,
            boundaries[comp]["inputs"],
            boundaries[comp]["outputs"],
        )

        if prec == "fp16":
            comp_fp16_onnx[comp] = comp_out
        else:
            comp_fp32_onnx[comp] = comp_out

        print(f"[OK] Extracted {comp} ({suffix}): {comp_out}")


    # 2) if gs/ha/hs need fp8, generate their calib by tapping full model once
    #    rule: component calib should match THAT COMPONENT input tensor.
    comp_calib_map: Dict[str, str] = {}
    # Will store fixed input shapes for each component (for static TRT build)
    comp_fixed_shape: Dict[str, Tuple[int, ...]] = {}
    comp_fixed_shape["ga"] = tuple(input_shape)  # ga input is model input


    # 2) if gs/ha/hs need fp8, generate their calib by tapping full model once
    #    ALSO: record ha/hs/gs input shapes (used later to build static TRT engines)
    for comp in ("ga", "ha", "hs", "gs"):
        # skip if component not defined in boundaries
        if comp != "ga" and comp not in boundaries:
            print(f"[Skip] boundaries missing component: {comp}")
            continue

        prec = cfg.get(comp, "fp16")

        if comp == "ga":
            # ga input is model input x; shape is known
            comp_fixed_shape["ga"] = tuple(input_shape)
            if prec == "fp8":
                # ga fp8 uses user provided x calib directly
                comp_calib_map["ga"] = args.calib_npy
            continue

        # For ha/hs/gs: tap its input tensor to obtain shape (always do it once)
        tap_tensor = boundaries[comp]["inputs"][0]
        out_calib = os.path.join(calib_dir, f"calib_{comp}.npy")

        # Ensure we have x_calib to run ORT once (fp8 must have it; fp16 we can still use it if provided)
        if x_calib is None:
            raise RuntimeError(
                f"Need x_calib to tap shapes for {comp}. "
                f"If you don't use any fp8 component, please create a random x_calib earlier."
            )

        # Always tap once to get shape (reuse existing npy if already generated)
        if os.path.isfile(out_calib):
            # reuse existing calib to avoid running ORT again
            arr = np.load(out_calib)
            comp_fixed_shape[comp] = tuple(arr.shape)
            print(f"[Shape] {comp} fixed input shape = {comp_fixed_shape[comp]} (reuse calib file)")
        else:
            # If this component is fp8, we need to save calib anyway
            # If fp16, we still save it as a convenient cache for shape (you can delete later if you want)
            print(f"[Tap] Tapping {comp} input tensor to get shape: {tap_tensor}")
            saved, shape = generate_component_calib_from_full_onnx(
                full_onnx=args.onnx_fp32,
                model_input_name=model_input_name,
                tap_tensor_name=tap_tensor,
                x_calib=x_calib,
                out_calib_npy=out_calib,    # cache
                tmp_dir=tap_dir,
                prefer_cuda=args.prefer_cuda_ort,
            )
            comp_fixed_shape[comp] = tuple(shape)
            print(f"[Shape] {comp} fixed input shape = {comp_fixed_shape[comp]} (saved: {saved})")

        # Only fp8 components need to be in comp_calib_map for quantization
        if prec == "fp8":
            comp_calib_map[comp] = out_calib

    for comp, onnx_path in comp_fp16_onnx.items():
        if comp in comp_fixed_shape:
            fix_onnx_input_shape_inplace(onnx_path, comp_fixed_shape[comp])

    for comp, onnx_path in comp_fp32_onnx.items():
        if comp in comp_fixed_shape:
            fix_onnx_input_shape_inplace(onnx_path, comp_fixed_shape[comp])

    # 3) per-component quantize + build engine
    for comp, fp32_path in comp_fp32_onnx.items():
        prec = cfg.get(comp, "fp16")  # default fp16 if not specified
        comp_out_engine = os.path.join(engine_dir, comp, f"{prec}.engine")
        print(f"\n[Engine] Building {comp} engine ({prec}): {comp_out_engine}")
        ensure_dir(os.path.dirname(comp_out_engine))
        if prec == "fp8":
            calib_for_comp = comp_calib_map.get(comp, "")
            if not calib_for_comp:
                raise ValueError(f"FP8 for {comp} requires calib, but it was not generated/found.")
            fp8_qdq_onnx = os.path.join(subonnx_dir, f"{comp}_fp8_qdq.onnx")
            if not os.path.isfile(fp8_qdq_onnx):
                print(f"[FP8] Generating FP8 QDQ ONNX for {comp}")
                modelopt_quant_fp8(fp32_path, fp8_qdq_onnx, calib_for_comp)
            if not os.path.isfile(comp_out_engine):
                build_trt_engine(fp8_qdq_onnx, comp_out_engine, "fp8", fixed_input_shape=comp_fixed_shape[comp])
                print(f"[OK] Engine built: {comp_out_engine}")
        else:
            raise ValueError(f"Unsupported precision: {prec}")
        
    for comp, fp16_path in comp_fp16_onnx.items():
        prec = cfg.get(comp, "fp16")  # default fp16 if not specified
        comp_out_engine = os.path.join(engine_dir, comp, f"{prec}.engine")
        print(f"\n[Engine] Building {comp} engine ({prec}): {comp_out_engine}")
        ensure_dir(os.path.dirname(comp_out_engine))
        if prec == "fp16":
            fp16_onnx = comp_fp16_onnx[comp]
            if not os.path.isfile(comp_out_engine):
                build_trt_engine(fp16_onnx, comp_out_engine, "fp16", fixed_input_shape=comp_fixed_shape[comp])
                print(f"[OK] FP16 Engine built: {comp_out_engine}")
        else:
            raise ValueError(f"Unsupported precision: {prec}")

    print("\nDone. Engines saved under:", engine_dir)
    if comp_calib_map:
        print("Calibs saved under:", calib_dir)


if __name__ == "__main__":
    main()
