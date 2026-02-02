import numpy as np

def quantize_uniform(y_data, scale_bits=8, eb = 0.01):
    """标准化 + 均匀线性量化"""
    mean_val = np.mean(y_data)
    std_val = np.std(y_data)
    epsilon = 1e-7
    if std_val < epsilon:
        std_val = epsilon
    y_standardized = (y_data - mean_val) / (std_val + epsilon)
    y_int = np.clip(y_standardized * (2 ** scale_bits), -32768, 32767)
    ctx = {"mean": mean_val, "std": std_val, "scale_bits": scale_bits}
    return y_int, ctx

def dequantize_uniform(y_fp, ctx):
    mean_val = ctx["mean"]
    std_val = ctx["std"]
    scale_bits = ctx["scale_bits"]
    return y_fp / (2 ** scale_bits) * std_val + mean_val
