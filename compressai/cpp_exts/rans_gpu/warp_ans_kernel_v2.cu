// compressai/cpp_exts/rans_gpu/warp_ans_kernel_v2.cu
// WarpANS V2: 32-lane warp-level interleaved rANS + strength-reduced division.
//
// Only defines the V2 encode kernel + host API. Decode + helper kernels
// are shared with V1 via forward declarations and the V2 host API calls
// V1's C++ decode function.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"        // EncPutBits, EncInit, EncFlush
#include "rans64_gpu_v2.cuh"     // Rans64EncPutV2
#include "rans64dec_gpu.cuh"
#include "warp_rans.cuh"

// ---- Shared declarations (defined in warp_ans_kernel.cu) ----
extern __global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, int N, int C, int HW,
    int32_t* __restrict__ flag);

extern __global__ void warp_pack_tight_payload_kernel(
    const uint8_t* __restrict__ temp_arena_u8,
    int64_t temp_stride, int64_t temp_header_padded,
    int cap_words_per_lane, int B, int K,
    const int32_t* __restrict__ lane_word_counts_flat,
    const int32_t* __restrict__ max_rounds_flat,
    const uint32_t* __restrict__ chunk_offsets_u32,
    uint8_t* __restrict__ packed_u8, int64_t header_bytes);

extern __global__ void warp_decode_chunks_kernel(
    const uint8_t* __restrict__ packed_u8, int64_t header_bytes,
    const uint32_t* __restrict__ chunk_offsets_u32,
    const int32_t* __restrict__ max_rounds_flat,
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,
    const int32_t* __restrict__ cdfs_mxl, int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int32_t* __restrict__ out_symbols_bxn,
    int fast_idx_is_channel, int HW);

extern __global__ void max_rounds_to_sizes_kernel(
    const int32_t* __restrict__ max_rounds, int n,
    uint32_t* __restrict__ sizes_u32, int multiplier);

extern __global__ void write_warp_tight_header_kernel(
    uint8_t* __restrict__ packed_u8, int N, int chunk_len, int K, int B,
    int flags, int warp_lanes, int C, int HW, int64_t header_bytes,
    const int32_t* __restrict__ max_rounds_flat, int n_max_rounds);

// ---- V2 encode kernel (only difference from V1) ----

constexpr int kWarpLanes = 32;
constexpr int kPrecision = 16;
constexpr uint16_t kBypassPrecision = 4;
constexpr uint16_t kMaxBypassVal = (1 << kBypassPrecision) - 1;

static inline int ceil_div_int_v2(int a, int b) { return (a + b - 1) / b; }

__global__ void warp_encode_chunks_kernel_v2(
    const int32_t* __restrict__ symbols_bxn,
    const int32_t* __restrict__ indexes_bxn,
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl, int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int K, int chunk_len, int HW,
    uint8_t* __restrict__ arena_u8,
    int64_t stride, int64_t header_bytes_padded,
    int cap_words_per_lane,
    int32_t* __restrict__ lane_word_counts_flat,
    int32_t* __restrict__ max_rounds_flat,
    const uint64_t* __restrict__ magic_table,
    int fast_idx_is_channel
) {
    int b = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    if (b >= B || chunk_id >= K) return;

    int lane = (int)threadIdx.x;
    int start = chunk_id * chunk_len;
    if (start >= N) {
        lane_word_counts_flat[b * K * kWarpLanes + chunk_id * kWarpLanes + lane] = 0;
        if (lane == 0) max_rounds_flat[b * K + chunk_id] = 0;
        return;
    }
    int end = start + chunk_len;
    if (end > N) end = N;

    int cap_bytes_per_lane = cap_words_per_lane * 4;
    uint8_t* stream_base = arena_u8 + (int64_t)b * stride;
    uint8_t* payload_base = stream_base + header_bytes_padded;
    uint8_t* chunk_slot = payload_base + (int64_t)chunk_id * kWarpLanes * cap_bytes_per_lane;
    uint8_t* sub_slot = chunk_slot + (int64_t)lane * cap_bytes_per_lane;

    uint32_t* base_u32 = reinterpret_cast<uint32_t*>(sub_slot);
    uint32_t* ptr = base_u32 + cap_words_per_lane;

    Rans64State r;
    Rans64EncInit(&r);

    const int32_t* sym = symbols_bxn + (int64_t)b * N;
    const int32_t* idx = indexes_bxn + (int64_t)b * N;

    int32_t last_cdf_idx = -1;
    const int32_t* last_cdf = nullptr;
    int32_t last_cdf_size = 0;
    int32_t last_offsetv = 0;

    int first = start + lane;
    int last  = first + ((end - 1 - first) / kWarpLanes) * kWarpLanes;
    int used_words = 0;

    if (first < end) {
        for (int i = last; i >= first; i -= kWarpLanes) {
            int32_t cdf_idx = fast_idx_is_channel ? (i / HW) : idx[i];
            const int32_t* cdf; int32_t cdf_size, offsetv;
            if (cdf_idx == last_cdf_idx) {
                cdf = last_cdf; cdf_size = last_cdf_size; offsetv = last_offsetv;
            } else {
                cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
                cdf_size = cdf_sizes_m[cdf_idx];
                offsetv = offsets_m[cdf_idx];
                last_cdf_idx = cdf_idx; last_cdf = cdf;
                last_cdf_size = cdf_size; last_offsetv = offsetv;
            }

            int32_t max_value = cdf_size - 2;
            int32_t value = sym[i] - offsetv;
            uint32_t raw_val = 0;
            if (value < 0) {
                raw_val = (uint32_t)(-2 * value - 1); value = max_value;
            } else if (value >= max_value) {
                raw_val = (uint32_t)(2 * (value - max_value)); value = max_value;
            }

            if (value == max_value) {
                int32_t n_bypass = 0;
                if (raw_val) {
                    int msb = 31 - __clz(raw_val);
                    n_bypass = (msb / kBypassPrecision) + 1;
                }
                for (int32_t j = n_bypass - 1; j >= 0; --j) {
                    uint32_t v = (raw_val >> (j * kBypassPrecision)) & kMaxBypassVal;
                    Rans64EncPutBits(&r, &ptr, v, kBypassPrecision);
                }
                int32_t val = n_bypass; int32_t n_full = 0;
                while (val >= (int32_t)kMaxBypassVal) { val -= kMaxBypassVal; n_full++; }
                Rans64EncPutBits(&r, &ptr, (uint32_t)val, kBypassPrecision);
                for (int k2 = 0; k2 < n_full; ++k2) {
                    Rans64EncPutBits(&r, &ptr, (uint32_t)kMaxBypassVal, kBypassPrecision);
                }
            }

            uint32_t start_c = (uint32_t)cdf[value];
            uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
            // V2: fast division via __umul64hi
            Rans64EncPutV2(&r, &ptr, start_c, freq, kPrecision, magic_table);
        }
        Rans64EncFlush(&r, &ptr);
        used_words = (int)(base_u32 + cap_words_per_lane - ptr);
    }

    int k_idx = b * K * kWarpLanes + chunk_id * kWarpLanes + lane;
    lane_word_counts_flat[k_idx] = used_words;
    int max_r = warp_reduce_max(used_words);
    if (lane == 0) max_rounds_flat[b * K + chunk_id] = max_r;
}

// ---- V2 Host API ----

std::vector<torch::Tensor> encode_with_indexes_warp_v2_cuda(
    torch::Tensor symbols_bxn, torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl, torch::Tensor cdf_sizes_m, torch::Tensor offsets_m,
    int64_t P_in
) {
    TORCH_CHECK(symbols_bxn.is_cuda() && indexes_bxn.is_cuda(), "CUDA required");

    const int B = (int)symbols_bxn.size(0);
    const int N = (int)symbols_bxn.size(1);
    const int Lmax = (int)cdfs_mxl.size(1);
    const int C = (int)cdfs_mxl.size(0);

    auto dev = symbols_bxn.device();
    auto stream = at::cuda::getDefaultCUDAStream();
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);
    auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(dev);

    int fast_idx_is_channel = 0, HW = 0;
    if (N % C == 0) {
        HW = N / C;
        auto flag = torch::ones({1}, opts_i32);
        int t = 256, blk = (C + t - 1) / t;
        check_idx_is_channel_kernel<<<blk, t, 0, stream>>>(
            indexes_bxn.data_ptr<int32_t>(), N, C, HW, flag.data_ptr<int32_t>());
        fast_idx_is_channel = flag.cpu().item<int32_t>();
    }

    int Pch = (int)P_in; if (Pch < 1) Pch = 1; if (Pch > C) Pch = C;
    int K = ceil_div_int_v2(C, Pch);
    int chunk_len = Pch * HW;
    int cap_words_per_lane = (chunk_len * 12 + 8) / kWarpLanes + 8;
    int cap_bytes_per_lane = cap_words_per_lane * 4;

    int64_t temp_header_padded = (((int64_t)4 * ((int64_t)K * kWarpLanes + 2)) + 15) & ~((int64_t)15);
    int64_t arena_stride = temp_header_padded + (int64_t)K * kWarpLanes * cap_bytes_per_lane;

    auto temp_arena_u8 = torch::empty({(long long)(arena_stride * B)}, opts_u8);
    auto lane_word_counts_flat = torch::zeros({B * K * kWarpLanes}, opts_i32);
    auto max_rounds_flat = torch::zeros({B * K}, opts_i32);

    // Magic table — built once, cached across calls
    static std::vector<uint64_t> magic_vec_v2 = build_magic_table_v2();
    static auto magic_table = torch::from_blob(
        (void*)magic_vec_v2.data(), {(long long)magic_vec_v2.size()},
        torch::TensorOptions().dtype(torch::kUInt64)
    ).to(dev);

    // (1) V2 Encode
    warp_encode_chunks_kernel_v2<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        symbols_bxn.data_ptr<int32_t>(), indexes_bxn.data_ptr<int32_t>(),
        B, N, cdfs_mxl.data_ptr<int32_t>(), Lmax,
        cdf_sizes_m.data_ptr<int32_t>(), offsets_m.data_ptr<int32_t>(),
        K, chunk_len, HW, temp_arena_u8.data_ptr<uint8_t>(),
        arena_stride, temp_header_padded, cap_words_per_lane,
        lane_word_counts_flat.data_ptr<int32_t>(),
        max_rounds_flat.data_ptr<int32_t>(),
        magic_table.data_ptr<uint64_t>(),
        fast_idx_is_channel);

    // (2) sizes = max_rounds * 32
    auto sizes_u32_flat = torch::empty({B * K}, opts_u32);
    { int n = B * K, t = 256; max_rounds_to_sizes_kernel<<<(n+t-1)/t, t, 0, stream>>>(
        max_rounds_flat.data_ptr<int32_t>(), n, sizes_u32_flat.data_ptr<uint32_t>(), kWarpLanes); }

    // (3) Exclusive scan
    auto chunk_offsets_u32 = torch::empty({B * K}, opts_u32);
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (const uint32_t*)nullptr, (uint32_t*)nullptr, B*K, stream.stream());
    auto scan_temp = torch::empty({(long long)scan_bytes}, opts_u8);
    cub::DeviceScan::ExclusiveSum(scan_temp.data_ptr(), scan_bytes,
        sizes_u32_flat.data_ptr<uint32_t>(), chunk_offsets_u32.data_ptr<uint32_t>(), B*K, stream.stream());

    // (4) Header + total
    int64_t last_off = chunk_offsets_u32[B*K-1].to(torch::kInt64).cpu().item<int64_t>();
    int64_t last_sz  = sizes_u32_flat[B*K-1].to(torch::kInt64).cpu().item<int64_t>();
    int64_t total_payload_words = last_off + last_sz;
    int64_t header_raw = 32 + 4LL * (B * K);
    int64_t header_bytes = (header_raw + 15) & ~((int64_t)15);
    auto packed_u8 = torch::empty({(long long)(header_bytes + total_payload_words * 4)}, opts_u8);

    // (5) Write header
    int flags = (fast_idx_is_channel ? 1 : 0) | 2;
    write_warp_tight_header_kernel<<<1, 1, 0, stream>>>(
        packed_u8.data_ptr<uint8_t>(), N, chunk_len, K, B,
        flags, kWarpLanes, fast_idx_is_channel ? C : 0, fast_idx_is_channel ? HW : 0,
        header_bytes, max_rounds_flat.data_ptr<int32_t>(), B * K);

    // (6) Pack (V1 helper kernel — compatible binary format)
    warp_pack_tight_payload_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        temp_arena_u8.data_ptr<uint8_t>(), arena_stride, temp_header_padded,
        cap_words_per_lane, B, K,
        lane_word_counts_flat.data_ptr<int32_t>(),
        max_rounds_flat.data_ptr<int32_t>(),
        chunk_offsets_u32.data_ptr<uint32_t>(),
        packed_u8.data_ptr<uint8_t>(), header_bytes);

    auto header_bytes_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    header_bytes_cpu[0] = header_bytes;
    auto chunk_len_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    chunk_len_cpu[0] = chunk_len;
    auto P_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    P_cpu[0] = Pch;

    return {packed_u8, max_rounds_flat.view({B, K}).to(torch::kUInt32), header_bytes_cpu, chunk_len_cpu, P_cpu};
}
