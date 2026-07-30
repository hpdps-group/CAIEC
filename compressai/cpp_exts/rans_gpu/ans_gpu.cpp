// compressai/cpp_exts/rans_gpu/ans_gpu.cpp
// Python bindings for GPU rANS entropy coding.
// Exposes legacy chunk-parallel API, WarpANS 32-lane interleaved API,
// and the pipelined session-based encoder.

#include <torch/extension.h>
#include <vector>

#include "ans_gpu_pipeline.h"

// ---- Legacy (chunk-parallel) API ----
std::vector<torch::Tensor> encode_with_indexes_tight_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_tight_cuda(
    torch::Tensor packed_u8,
    torch::Tensor sizes_u32,
    torch::Tensor header_bytes_cpu,
    torch::Tensor chunk_len_cpu,
    torch::Tensor P_cpu,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
);

// ---- WarpANS (32-lane interleaved) API ----
std::vector<torch::Tensor> encode_with_indexes_warp_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_warp_cuda(
    torch::Tensor packed_u8,
    torch::Tensor max_rounds_u32,      // [B,K] uint32 — max words per lane per chunk
    torch::Tensor header_bytes_cpu,    // CPU int64 [1]
    torch::Tensor chunk_len_cpu,       // CPU int32 [1]
    torch::Tensor P_cpu,               // CPU int32 [1]
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
);

// ---- WarpANS V2 (32-lane interleaved + fast division) API ----
// Only encode differs from V1. Decode reuses V1 implementation.
std::vector<torch::Tensor> encode_with_indexes_warp_v2_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

// ---- WarpANS V3 (V2 + shared memory CDF cache) API ----
std::vector<torch::Tensor> encode_with_indexes_warp_v3_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_warp_v3_cuda(
    torch::Tensor packed_u8,
    torch::Tensor max_rounds_u32,
    torch::Tensor header_bytes_cpu,
    torch::Tensor chunk_len_cpu,
    torch::Tensor P_cpu,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Legacy: chunk-parallel (one thread per chunk)
    m.def("encode_with_indexes_tight", &encode_with_indexes_tight_cuda,
          "ANS encode tight — chunk-parallel (1 thread/chunk)");
    m.def("decode_with_indexes_tight", &decode_with_indexes_tight_cuda,
          "ANS decode tight — chunk-parallel (1 thread/chunk)");

    // WarpANS V1: 32-lane interleaved
    m.def("encode_with_indexes_warp", &encode_with_indexes_warp_cuda,
          "ANS encode tight — WarpANS (32-lane interleaved, one warp/chunk)");
    m.def("decode_with_indexes_warp", &decode_with_indexes_warp_cuda,
          "ANS decode tight — WarpANS (32-lane interleaved, one warp/chunk)");

    // WarpANS V2: 32-lane interleaved + strength-reduced division
    m.def("encode_with_indexes_warp_v2", &encode_with_indexes_warp_v2_cuda,
          "ANS encode tight — WarpANS V2 (32-lane + fast division)");
    m.def("decode_with_indexes_warp_v2", &decode_with_indexes_warp_cuda,
          "ANS decode tight — WarpANS V2 (same as V1 decode)");

    // WarpANS V3: V2 + shared memory CDF cache
    m.def("encode_with_indexes_warp_v3", &encode_with_indexes_warp_v3_cuda,
          "ANS encode tight — WarpANS V3 (shared mem CDF + fast division)");
    m.def("decode_with_indexes_warp_v3", &decode_with_indexes_warp_v3_cuda,
          "ANS decode tight — WarpANS V3 (shared mem CDF)");

    // ---- Pipelined Session-based Encoder (V3 kernels) ----
    pybind11::class_<AnsEncodeSession>(m, "AnsEncodeSession")
        .def(py::init<>())
        .def_readwrite("B", &AnsEncodeSession::B)
        .def_readwrite("N", &AnsEncodeSession::N)
        .def_readwrite("C", &AnsEncodeSession::C);

    m.def("ans_create_session", &ans_encode_create_session,
          "Create a pre-allocated encoder session for pipelined use",
          pybind11::arg("cdfs"), pybind11::arg("cdf_sizes"), pybind11::arg("offsets"),
          pybind11::arg("B"), pybind11::arg("N"), pybind11::arg("P"),
          pybind11::arg("fast_idx_is_channel") = false);

    m.def("ans_launch", &ans_encode_launch,
          "Submit all encode kernels to ANS stream (non-blocking, no GPU sync)",
          pybind11::arg("session"), pybind11::arg("symbols"), pybind11::arg("indexes"));

    m.def("ans_finalize", &ans_encode_finalize,
          "Synchronize ANS stream and build result tensors",
          pybind11::arg("session"));

    // ---- Pipelined Session-based Decoder ----
    pybind11::class_<AnsDecodeSession>(m, "AnsDecodeSession")
        .def(pybind11::init<>());

    m.def("ans_decode_create_session", &ans_decode_create_session,
          "Create a pre-allocated decoder session for pipelined use",
          pybind11::arg("cdfs"), pybind11::arg("cdf_sizes"), pybind11::arg("offsets"),
          pybind11::arg("B"), pybind11::arg("N"), pybind11::arg("K"),
          pybind11::arg("chunk_len"), pybind11::arg("HW"),
          pybind11::arg("fast_idx_is_channel") = false);

    m.def("ans_decode_launch", &ans_decode_launch,
          "Submit all decode kernels to ANS stream (non-blocking, no GPU sync)",
          pybind11::arg("session"), pybind11::arg("packed_u8"),
          pybind11::arg("max_rounds_u32"), pybind11::arg("header_bytes"),
          pybind11::arg("indexes_bxn"));

    m.def("ans_decode_finalize", &ans_decode_finalize,
          "Synchronize ANS stream and return decoded symbols",
          pybind11::arg("session"));
}
