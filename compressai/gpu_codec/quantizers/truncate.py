import numpy as np

def quantize_truncate(y_fp):
    return y_fp.astype(np.int16)

def dequantize_truncate(y_int):
    return y_int.astype(np.float32)
