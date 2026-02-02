#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> encode_with_indexes_tight_cuda(
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int64_t P_in
);

torch::Tensor decode_with_indexes_tight_cuda(
    torch::Tensor packed_u8,       // CUDA uint8 [total]
    torch::Tensor sizes_u16,       // CUDA uint16 [B,K]   (NOTE: K may differ from P_cpu)
    torch::Tensor header_bytes_cpu,// CPU int64 [1]
    torch::Tensor chunk_len_cpu,   // CPU int32 [1]
    torch::Tensor P_cpu,           // CPU int32 [1]   (fast: Pch, generic: K)
    torch::Tensor indexes_bxn,     // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,        // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,     // CUDA int32 [M]
    torch::Tensor offsets_m        // CUDA int32 [M]
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_with_indexes_tight", &encode_with_indexes_tight_cuda, "ANS encode tight (GPU packed)");
    m.def("decode_with_indexes_tight", &decode_with_indexes_tight_cuda, "ANS decode tight (GPU packed)");
}
