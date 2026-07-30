# compressai/runtime/ans_pipeline.py
"""
Session-based pipelined ANS encoder.

Wraps the C++ ``ans_create_session`` / ``ans_launch`` / ``ans_finalize``
API into a Python-friendly ``PipelinedAnsEncoder`` class.

Typical pipelined usage::

    encoder = PipelinedAnsEncoder(eb, B, N, P)
    encoder.launch(symbols, indexes)      # non-blocking (ANS stream)
    ...
    pack = encoder.finalize()             # sync + TightWarpANS

The ``launch()`` call submits all GPU kernels and returns immediately.
The ``finalize()`` call synchronises and collects the result.  Between
``launch()`` and ``finalize()`` the host is free to schedule inference
kernels on the TRT stream, enabling GPU-level overlap.
"""

from __future__ import annotations

from typing import Optional
import torch

import compressai.ans_gpu as ans_gpu
from compressai.entropy_models.entropy_models import TightWarpANS


class PipelinedAnsEncoder:
    """
    Pipelined WarpANS V3 encoder with pre-allocated buffers.

    Created once per (B, N, C, P) configuration.  Each call to ``launch()``
    submits kernels without GPU synchronisation; ``finalize()`` collects.
    """

    def __init__(
        self,
        cdfs: torch.Tensor,          # int32 [C, Lmax]
        cdf_sizes: torch.Tensor,     # int32 [C]
        offsets: torch.Tensor,       # int32 [C]
        B: int,
        N: int,
        P: int = 64,
        fast_idx_is_channel: bool = False,
    ):
        self._cdfs = cdfs.cuda().to(torch.int32).contiguous()
        self._cdf_sizes = cdf_sizes.cuda().to(torch.int32).contiguous()
        self._offsets = offsets.cuda().to(torch.int32).contiguous()
        self.B = B
        self.N = N
        self.P = P

        self._session = ans_gpu.ans_create_session(
            self._cdfs, self._cdf_sizes, self._offsets, B, N, P,
            fast_idx_is_channel,
        )

    @staticmethod
    def from_entropy_bottleneck(
        eb,      # EntropyBottleneck instance
        B: int,
        N: int,
        P: int = 64,
    ) -> "PipelinedAnsEncoder":
        """Convenience factory from an EntropyBottleneck instance."""
        return PipelinedAnsEncoder(
            cdfs=eb._quantized_cdf,
            cdf_sizes=eb._cdf_length,
            offsets=eb._offset,
            B=B, N=N, P=P, fast_idx_is_channel=True,
        )

    @staticmethod
    def from_gaussian_conditional(
        gc,      # GaussianConditional instance
        B: int,
        N: int,
        P: int = 64,
    ) -> "PipelinedAnsEncoder":
        """Convenience factory from a GaussianConditional instance."""
        return PipelinedAnsEncoder(
            cdfs=gc._quantized_cdf,
            cdf_sizes=gc._cdf_length,
            offsets=gc._offset,
            B=B, N=N, P=P, fast_idx_is_channel=False,
        )

    def launch(
        self,
        symbols: torch.Tensor,        # int32 [B, ...]
        indexes: torch.Tensor,        # int32 [B, ...]
    ) -> None:
        """
        Submit all encode kernels to the default (ANS) CUDA stream.

        This call is *non-blocking* — it returns immediately without any
        GPU synchronisation.  The host can proceed to launch inference
        kernels on other streams while the encoding runs on the GPU.

        Args:
            symbols: quantised symbols, int32 on CUDA, shape [B, ...]
            indexes: CDF indexes, int32 on CUDA, same shape as symbols
        """
        if not symbols.is_cuda or not indexes.is_cuda:
            raise ValueError("symbols and indexes must be CUDA tensors")
        if symbols.dtype != torch.int32 or indexes.dtype != torch.int32:
            raise TypeError("symbols and indexes must have dtype int32")
        if not symbols.is_contiguous() or not indexes.is_contiguous():
            raise ValueError("symbols and indexes must be contiguous")
        if symbols.shape != indexes.shape:
            raise ValueError("symbols and indexes must have the same shape")
        if symbols.numel() != self.B * self.N:
            raise ValueError(f"symbols and indexes must contain B*N={self.B * self.N} elements")
        if symbols.device != indexes.device or symbols.device != self._cdfs.device:
            raise ValueError("symbols, indexes, and session CDFs must be on the same device")

        B, N = self.B, self.N
        sym_bxn = symbols.reshape(B, N)
        idx_bxn = indexes.reshape(B, N)
        ans_gpu.ans_launch(self._session, sym_bxn, idx_bxn)

    def finalize(self) -> TightWarpANS:
        """
        Synchronise the ANS stream and collect the compressed result.

        Returns a ``TightWarpANS`` ready for storage or decompression.
        """
        packed_u8, max_rounds_u32, header_bytes_cpu, chunk_len_cpu, P_cpu = \
            ans_gpu.ans_finalize(self._session)
        return TightWarpANS(
            packed_u8, max_rounds_u32, header_bytes_cpu, chunk_len_cpu, P_cpu
        )


class PipelinedAnsDecoder:
    """
    Pipelined WarpANS V3 decoder with pre-allocated buffers.

    Created once per (B, N, C, K, chunk_len) configuration.  Each ``launch()``
    submits all decode kernels without GPU sync; ``finalize()`` collects the
    decoded int32 symbols.
    """

    def __init__(
        self,
        cdfs: torch.Tensor,          # int32 [C, Lmax]
        cdf_sizes: torch.Tensor,     # int32 [C]
        offsets: torch.Tensor,       # int32 [C]
        B: int,
        N: int,
        K: int,
        chunk_len: int,
        HW: int,
        fast_idx_is_channel: bool = False,
    ):
        self._cdfs = cdfs.cuda().to(torch.int32).contiguous()
        self._cdf_sizes = cdf_sizes.cuda().to(torch.int32).contiguous()
        self._offsets = offsets.cuda().to(torch.int32).contiguous()
        self.B = B
        self.N = N
        self.K = K
        self.chunk_len = chunk_len
        self.HW = HW

        self._session = ans_gpu.ans_decode_create_session(
            self._cdfs, self._cdf_sizes, self._offsets,
            B, N, K, chunk_len, HW, fast_idx_is_channel
        )

    @staticmethod
    def from_entropy_bottleneck(
        eb,                     # EntropyBottleneck instance
        B: int,
        N: int,
        K: int,
        chunk_len: int,
        HW: int,
    ) -> "PipelinedAnsDecoder":
        """Convenience factory from an EntropyBottleneck instance."""
        return PipelinedAnsDecoder(
            cdfs=eb._quantized_cdf,
            cdf_sizes=eb._cdf_length,
            offsets=eb._offset,
            B=B, N=N, K=K, chunk_len=chunk_len, HW=HW,
            fast_idx_is_channel=True,
        )

    @staticmethod
    def from_gaussian_conditional(
        gc,                     # GaussianConditional instance
        B: int,
        N: int,
        K: int,
        chunk_len: int,
        HW: int,
    ) -> "PipelinedAnsDecoder":
        """Convenience factory from a GaussianConditional instance."""
        return PipelinedAnsDecoder(
            cdfs=gc._quantized_cdf,
            cdf_sizes=gc._cdf_length,
            offsets=gc._offset,
            B=B, N=N, K=K, chunk_len=chunk_len, HW=HW,
            fast_idx_is_channel=False,
        )

    def launch(
        self,
        packed_u8: torch.Tensor,           # uint8 [total_bytes]  from TightWarpANS
        max_rounds_u32: torch.Tensor,      # uint32 [B, K]
        header_bytes: int,
        indexes_bxn: torch.Tensor,         # int32 [B, N]
    ) -> None:
        """
        Submit all decode kernels to the default (ANS) CUDA stream.
        Returns immediately — no GPU sync.

        Args:
            packed_u8: raw packed bytes from TightWarpANS.packed
            max_rounds_u32: max_rounds from TightWarpANS
            header_bytes: header size from TightWarpANS.header_bytes_cpu[0]
            indexes_bxn: CDF indexes, shape [B, N], int32, CUDA contiguous
        """
        if not packed_u8.is_cuda or not max_rounds_u32.is_cuda or not indexes_bxn.is_cuda:
            raise ValueError("packed_u8, max_rounds_u32, and indexes_bxn must be CUDA tensors")
        if packed_u8.dtype != torch.uint8:
            raise TypeError("packed_u8 must have dtype uint8")
        if max_rounds_u32.dtype != torch.uint32:
            raise TypeError("max_rounds_u32 must have dtype uint32")
        if indexes_bxn.dtype != torch.int32:
            raise TypeError("indexes_bxn must have dtype int32")
        if not all(t.is_contiguous() for t in (packed_u8, max_rounds_u32, indexes_bxn)):
            raise ValueError("decode inputs must be contiguous")
        if packed_u8.ndim != 1:
            raise ValueError("packed_u8 must be one-dimensional")
        if tuple(max_rounds_u32.shape) != (self.B, self.K):
            raise ValueError(f"max_rounds_u32 must have shape ({self.B}, {self.K})")
        if indexes_bxn.numel() != self.B * self.N:
            raise ValueError(f"indexes_bxn must contain B*N={self.B * self.N} elements")
        if indexes_bxn.ndim < 1 or indexes_bxn.shape[0] != self.B:
            raise ValueError(f"indexes_bxn must have leading dimension B={self.B}")
        devices = {packed_u8.device, max_rounds_u32.device, indexes_bxn.device, self._cdfs.device}
        if len(devices) != 1:
            raise ValueError("decode inputs and session CDFs must be on the same device")

        ans_gpu.ans_decode_launch(
            self._session, packed_u8, max_rounds_u32,
            int(header_bytes), indexes_bxn.reshape(self.B, self.N)
        )

    def finalize(self) -> torch.Tensor:
        """
        Synchronise the ANS stream and return decoded symbols.
        Returns int32 tensor of shape [B, N].
        """
        return ans_gpu.ans_decode_finalize(self._session)
