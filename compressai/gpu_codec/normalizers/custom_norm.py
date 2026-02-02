import numpy as np

def quantize_custom(y_data, **kwargs):
    """自定义量化方法模板"""
    # TODO: 根据研究想法定义新的量化策略
    raise NotImplementedError("请在此定义新的量化方法")

def dequantize_custom(y_int, ctx):
    """自定义反量化方法模板"""
    raise NotImplementedError("请在此定义对应的反量化方法")
