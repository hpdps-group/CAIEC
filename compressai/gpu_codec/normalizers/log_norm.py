import numpy as np

def quantize_log(y_data, scale_bits=8):
    """对数量化"""
    epsilon = 1e-7
    y_sign = np.sign(y_data)
    y_log = np.log1p(np.abs(y_data))
    y_log_norm = y_log / np.max(y_log)
    y_int = np.clip(y_log_norm * (2 ** scale_bits), 0, 32767)

    ctx = {"max_log": np.max(y_log), "scale_bits": scale_bits}
    return (y_int * y_sign), ctx

def dequantize_log(y_fp, ctx):
    scale_bits = ctx["scale_bits"]
    max_log = ctx["max_log"]
    y_sign = np.sign(y_fp)
    y_abs = np.abs(y_fp) / (2 ** scale_bits)
    y_val = np.expm1(y_abs * max_log)
    return y_val * y_sign
