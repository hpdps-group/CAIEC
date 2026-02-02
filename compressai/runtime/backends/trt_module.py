# compressai/runtime/backends/trt_module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _trt_dtype_to_torch(dt: trt.DataType) -> torch.dtype:
    if dt == trt.float16:
        return torch.float16
    if dt == trt.float32:
        return torch.float32
    if dt == trt.int8:
        return torch.int8
    if dt == trt.int32:
        return torch.int32
    raise TypeError(f"Unsupported TRT dtype: {dt}")


@dataclass
class TRTModule:
    """
    Runtime-only TensorRT module loader.
    - Supports TRT 10.x (I/O tensors + execute_async_v3)
    - Backward compatible with TRT 8/9 (bindings + execute_async_v2) when available

    Assumes single input / single output for MVP. (Easy to extend later.)
    """
    engine_path: str
    input_name: Optional[str] = None
    output_name: Optional[str] = None

    def __post_init__(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine from: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        # Detect TRT API style
        self._use_io_tensors = hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "execute_async_v3")

        if self._use_io_tensors:
            # TRT 10.x style: discover IO tensor names
            self._init_io_tensors_trt10()
        else:
            # TRT 8/9 style: discover binding indices
            self._init_bindings_legacy()

    def _init_io_tensors_trt10(self):
        input_names = []
        output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(name)

        if len(input_names) < 1 or len(output_names) < 1:
            raise RuntimeError(
                f"Expected at least 1 input and 1 output tensor, got inputs={input_names}, outputs={output_names}"
            )

        # 支持多输入：保存引擎输入顺序
        self._input_names = input_names

        # 兼容旧字段：如果用户传了 input_name，仅用于单输入或作为第一个输入的覆盖
        if self.input_name is not None:
            if self.input_name not in input_names:
                raise ValueError(f"input_name={self.input_name} not found in engine inputs: {input_names}")
            # 如果你希望“把指定 input_name 放到第一个”，可以重排
            self._input_names = [self.input_name] + [n for n in input_names if n != self.input_name]

        # 输出仍按原逻辑：只取一个输出
        self._output_name = self.output_name or output_names[0]
        if self._output_name not in output_names:
            raise ValueError(f"output_name={self._output_name} not found in engine outputs: {output_names}")

        self._out_torch_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(self._output_name))


    def _init_bindings_legacy(self):
        if not hasattr(self.engine, "num_bindings"):
            raise RuntimeError(
                "TensorRT engine does not expose num_bindings; "
                "please use TRT 10 IO-tensor path or update TRTModule."
            )

        n = self.engine.num_bindings
        self._bindings: List[int] = [0] * n

        input_binding_idx = []
        output_binding_idx = []
        input_binding_names = []
        output_binding_names = []

        for i in range(n):
            if self.engine.binding_is_input(i):
                input_binding_idx.append(i)
                input_binding_names.append(self.engine.get_binding_name(i))
            else:
                output_binding_idx.append(i)
                output_binding_names.append(self.engine.get_binding_name(i))

        if len(input_binding_idx) < 1 or len(output_binding_idx) < 1:
            raise RuntimeError(
                f"Expected at least 1 input and 1 output binding, got inputs={input_binding_idx}, outputs={output_binding_idx}"
            )

        # 支持多输入：保存输入 binding index 的顺序
        self._in_idxs = input_binding_idx
        self._input_names = input_binding_names  # 仅用于对齐/调试

        # 兼容旧字段：如果指定 input_name，则把对应 binding 放到第一个
        if self.input_name is not None:
            if self.input_name not in input_binding_names:
                raise ValueError(f"input_name={self.input_name} not found in engine inputs: {input_binding_names}")
            first = input_binding_names.index(self.input_name)
            # 重排 idx 和 name
            self._in_idxs = [input_binding_idx[first]] + [idx for j, idx in enumerate(input_binding_idx) if j != first]
            self._input_names = [self.input_name] + [n for n in input_binding_names if n != self.input_name]

        # 输出仍按原逻辑：只取第一个输出
        self._out_idx = output_binding_idx[0]
        self._output_name = self.output_name or output_binding_names[0]
        self._out_torch_dtype = _trt_dtype_to_torch(self.engine.get_binding_dtype(self._out_idx))


    @torch.no_grad()
    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        # 支持 mod(x) / mod(x, y)
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            xs = [args[0]]
        else:
            xs = list(args)

        if not hasattr(self, "_input_names"):
            # legacy 路径一定会设置；保险
            raise RuntimeError("TRTModule input names are not initialized")

        if len(xs) != len(self._input_names):
            raise ValueError(f"Engine expects {len(self._input_names)} inputs, got {len(xs)}")

        for i, x in enumerate(xs):
            if not x.is_cuda:
                raise ValueError(f"TRTModule expects CUDA tensor input (arg {i})")
            if not x.is_contiguous():
                xs[i] = x.contiguous()

        if self._use_io_tensors:
            return self._run_trt10(xs)
        else:
            return self._run_legacy(xs)


    def _run_trt10(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # Set shapes for all inputs
        for name, x in zip(self._input_names, xs):
            shape = tuple(x.shape)
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(name, shape)
            else:
                self.context.set_tensor_shape(name, shape)

        # Allocate output
        # 用第一个输入的 device
        device = xs[0].device
        out_shape = tuple(self.context.get_tensor_shape(self._output_name))
        y = torch.empty(out_shape, device=device, dtype=self._out_torch_dtype)

        # Bind addresses
        for name, x in zip(self._input_names, xs):
            self.context.set_tensor_address(name, int(x.data_ptr()))
        self.context.set_tensor_address(self._output_name, int(y.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v3(stream_handle=stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")

        return y

    def _run_legacy(self, xs: List[torch.Tensor]) -> torch.Tensor:
        # Set binding shapes if dynamic
        if hasattr(self.context, "set_binding_shape"):
            for in_idx, x in zip(self._in_idxs, xs):
                self.context.set_binding_shape(in_idx, tuple(x.shape))

        out_shape = tuple(self.context.get_binding_shape(self._out_idx))
        y = torch.empty(out_shape, device=xs[0].device, dtype=self._out_torch_dtype)

        # Bind pointers
        for in_idx, x in zip(self._in_idxs, xs):
            self._bindings[in_idx] = int(x.data_ptr())
        self._bindings[self._out_idx] = int(y.data_ptr())

        stream = torch.cuda.current_stream().cuda_stream
        ok = self.context.execute_async_v2(bindings=self._bindings, stream_handle=stream)
        if not ok:
            raise RuntimeError("TRT execute_async_v2 failed")

        return y
