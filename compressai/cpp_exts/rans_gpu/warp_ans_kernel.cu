// compressai/cpp_exts/rans_gpu/warp_ans_kernel.cu
// WarpANS: 32-lane warp-level interleaved rANS encode/decode kernels.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"
#include "rans64dec_gpu.cuh"
#include "warp_rans.cuh"

// Forward declaration
extern __global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, int N, int C, int HW,
    int32_t* __restrict__ flag);

constexpr int kWarpLanes = 32;
constexpr int kPrecision = 16;
constexpr uint16_t kBypassPrecision = 4;
constexpr uint16_t kMaxBypassVal = (1 << kBypassPrecision) - 1;

__device__ __forceinline__ int32_t warp_cdf_find_symbol(
    const int32_t* __restrict__ cdf, int32_t cdf_size, uint32_t cum_freq
) {
    int32_t lo = 0;
    int32_t hi = cdf_size - 2;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if ((uint32_t)cdf[mid + 1] > cum_freq) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

static inline int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

// ======================================================================
// WARP ENCODE KERNEL
// Grid: dim3(B, K), BlockDim: 32 (one warp)
// ======================================================================
__global__ void warp_encode_chunks_kernel(
    const int32_t* __restrict__ symbols_bxn,   // [B,N]
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int K, int chunk_len, int HW,
    uint8_t* __restrict__ arena_u8,
    int64_t stride, int64_t header_bytes_padded,
    int cap_words_per_lane,
    int32_t* __restrict__ lane_word_counts_flat, // [B*K*32]
    int32_t* __restrict__ max_rounds_flat,       // [B*K]
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

    // Per-lane sub-slot
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

    // Strided reverse encode
    int first = start + lane;
    int last  = first + ((end - 1 - first) / kWarpLanes) * kWarpLanes;
    int used_words = 0;

    if (first < end) {
        for (int i = last; i >= first; i -= kWarpLanes) {
            int32_t cdf_idx = fast_idx_is_channel ? (i / HW) : idx[i];

            const int32_t* cdf;
            int32_t cdf_size;
            int32_t offsetv;

            if (cdf_idx == last_cdf_idx) {
                cdf = last_cdf;
                cdf_size = last_cdf_size;
                offsetv = last_offsetv;
            } else {
                cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
                cdf_size = cdf_sizes_m[cdf_idx];
                offsetv = offsets_m[cdf_idx];
                last_cdf_idx = cdf_idx;
                last_cdf = cdf;
                last_cdf_size = cdf_size;
                last_offsetv = offsetv;
            }

            int32_t max_value = cdf_size - 2;
            int32_t value = sym[i] - offsetv;

            uint32_t raw_val = 0;
            if (value < 0) {
                raw_val = (uint32_t)(-2 * value - 1);
                value = max_value;
            } else if (value >= max_value) {
                raw_val = (uint32_t)(2 * (value - max_value));
                value = max_value;
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

                int32_t val = n_bypass;
                int32_t n_full = 0;
                while (val >= (int32_t)kMaxBypassVal) { val -= kMaxBypassVal; n_full++; }
                Rans64EncPutBits(&r, &ptr, (uint32_t)val, kBypassPrecision);
                for (int k2 = 0; k2 < n_full; ++k2) {
                    Rans64EncPutBits(&r, &ptr, (uint32_t)kMaxBypassVal, kBypassPrecision);
                }
            }

            uint32_t start_c = (uint32_t)cdf[value];
            uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
            Rans64EncPut(&r, &ptr, start_c, freq, kPrecision);
        }

        Rans64EncFlush(&r, &ptr);
        used_words = (int)(base_u32 + cap_words_per_lane - ptr);
    }

    int k_idx = b * K * kWarpLanes + chunk_id * kWarpLanes + lane;
    lane_word_counts_flat[k_idx] = used_words;

    int max_r = warp_reduce_max(used_words);
    if (lane == 0) max_rounds_flat[b * K + chunk_id] = max_r;
}

// ======================================================================
// WARP PACK KERNEL
// Grid: dim3(B, K), BlockDim: 32
// ======================================================================
__global__ void warp_pack_tight_payload_kernel(
    const uint8_t* __restrict__ temp_arena_u8,
    int64_t temp_stride, int64_t temp_header_padded,
    int cap_words_per_lane,
    int B, int K,
    const int32_t* __restrict__ lane_word_counts_flat,   // [B*K*32]
    const int32_t* __restrict__ max_rounds_flat,          // [B*K]
    const uint32_t* __restrict__ chunk_offsets_u32,       // [B*K] in words
    uint8_t* __restrict__ packed_u8,
    int64_t header_bytes
) {
    int b = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    if (b >= B || chunk_id >= K) return;

    int lane = (int)threadIdx.x;
    int k = b * K + chunk_id;
    int max_rounds = max_rounds_flat[k];
    if (max_rounds == 0) return;

    int k_idx = b * K * kWarpLanes + chunk_id * kWarpLanes + lane;
    int used_words = lane_word_counts_flat[k_idx];

    int cap_bytes_per_lane = cap_words_per_lane * 4;
    uint8_t* stream_base = (uint8_t*)temp_arena_u8 + (int64_t)b * temp_stride;
    uint8_t* payload_base = stream_base + temp_header_padded;
    uint8_t* chunk_slot = payload_base + (int64_t)chunk_id * kWarpLanes * cap_bytes_per_lane;
    uint8_t* sub_slot = chunk_slot + (int64_t)lane * cap_bytes_per_lane;
    uint32_t* lane_words = (uint32_t*)(sub_slot + cap_bytes_per_lane - (int64_t)used_words * 4);

    uint32_t* chunk_out = (uint32_t*)(packed_u8 + header_bytes) + chunk_offsets_u32[k];

    for (int r = 0; r < max_rounds; r++) {
        chunk_out[r * kWarpLanes + lane] = (r < used_words) ? lane_words[r] : 0u;
    }
}

// ======================================================================
// WARP DECODE KERNEL
// Grid: dim3(B, K), BlockDim: 32
// ======================================================================
__global__ void warp_decode_chunks_kernel(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint32_t* __restrict__ chunk_offsets_u32,       // [B*K] in words
    const int32_t* __restrict__ max_rounds_flat,            // [B*K]
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,
    const int32_t* __restrict__ cdfs_mxl,
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int32_t* __restrict__ out_symbols_bxn,
    int fast_idx_is_channel,
    int HW
) {
    int b = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    if (b >= B || chunk_id >= K) return;

    int lane = (int)threadIdx.x;
    int k = b * K + chunk_id;

    int max_rounds = max_rounds_flat[k];
    if (max_rounds == 0) return;

    int start = chunk_id * chunk_len;
    if (start >= N) return;
    int end = start + chunk_len;
    if (end > N) end = N;

    const uint32_t* payload = (const uint32_t*)(packed_u8 + header_bytes);
    const uint32_t* chunk_base = payload + chunk_offsets_u32[k];

    int lane_start = start + lane;
    int my_count = 0;
    if (lane_start < end) {
        my_count = (end - 1 - lane_start) / kWarpLanes + 1;
    }
    if (my_count <= 0) return;

    uint64_t r;
    uint64_t w0 = (uint64_t)chunk_base[0 * kWarpLanes + lane];
    uint64_t w1 = (uint64_t)chunk_base[1 * kWarpLanes + lane];
    r = w0 | (w1 << 32);
    int word_idx = 2;

    const int32_t* idx = indexes_bxn + (int64_t)b * N;
    int32_t* out = out_symbols_bxn + (int64_t)b * N;

    int32_t last_cdf_idx = -1;
    const int32_t* last_cdf = nullptr;
    int32_t last_cdf_size = 0;
    int32_t last_max_value = 0;
    int32_t last_offsetv = 0;

    for (int s = 0; s < my_count; s++) {
        int i = start + lane + s * kWarpLanes;

        int32_t cdf_idx = fast_idx_is_channel ? (i / HW) : idx[i];

        const int32_t* cdf;
        int32_t cdf_size, max_value, offsetv;

        if (cdf_idx == last_cdf_idx) {
            cdf = last_cdf;
            cdf_size = last_cdf_size;
            max_value = last_max_value;
            offsetv = last_offsetv;
        } else {
            cdf = cdfs_mxl + (int64_t)cdf_idx * Lmax;
            cdf_size = cdf_sizes_m[cdf_idx];
            max_value = cdf_size - 2;
            offsetv = offsets_m[cdf_idx];
            last_cdf_idx = cdf_idx;
            last_cdf = cdf;
            last_cdf_size = cdf_size;
            last_max_value = max_value;
            last_offsetv = offsetv;
        }

        uint32_t cum_freq = (uint32_t)(r & ((1u << kPrecision) - 1));
        int32_t symbol = warp_cdf_find_symbol(cdf, cdf_size, cum_freq);

        uint32_t start_c = (uint32_t)cdf[symbol];
        uint32_t freq = (uint32_t)(cdf[symbol + 1] - cdf[symbol]);
        WarpRans64DecAdvance(&r, chunk_base, lane, &word_idx, start_c, freq, kPrecision);

        int32_t value = symbol;

        if (value == max_value) {
            int32_t val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrecision);
            int32_t n_bypass = val;
            while (val == (int32_t)kMaxBypassVal) {
                val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrecision);
                n_bypass += val;
            }

            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrecision);
                raw_val |= (val << (j * kBypassPrecision));
            }

            value = raw_val >> 1;
            if (raw_val & 1) value = -value - 1;
            else value += max_value;
        }

        out[i] = value + offsetv;
    }
}

// ======================================================================
// Helper kernels
// ======================================================================
__global__ void max_rounds_to_sizes_kernel(
    const int32_t* __restrict__ max_rounds, int n,
    uint32_t* __restrict__ sizes_u32, int multiplier
) {
    int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sizes_u32[i] = (uint32_t)(max_rounds[i] * multiplier);
}

__global__ void write_warp_tight_header_kernel(
    uint8_t* __restrict__ packed_u8,
    int N, int chunk_len, int K, int B,
    int flags, int warp_lanes,
    int C, int HW,
    int64_t header_bytes,
    const int32_t* __restrict__ max_rounds_flat,
    int n_max_rounds
) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x != 0) return;

    uint32_t* h = (uint32_t*)packed_u8;
    h[0] = (uint32_t)N;
    h[1] = (uint32_t)chunk_len;
    h[2] = (uint32_t)(((uint32_t)K & 0xFFFFu) | (((uint32_t)B & 0xFFFFu) << 16));
    h[3] = (uint32_t)(((uint32_t)flags & 0xFFFFu) | (((uint32_t)warp_lanes & 0xFFFFu) << 16));
    h[4] = (uint32_t)C;
    h[5] = (uint32_t)HW;
    h[6] = 0;
    h[7] = 0;

    int32_t* mr_out = (int32_t*)(packed_u8 + 32);
    for (int i = 0; i < n_max_rounds; i++) {
        mr_out[i] = max_rounds_flat[i];
    }
}

// ======================================================================
// Host-callable API
// ======================================================================
#include <vector>

std::vector<torch::Tensor> encode_with_indexes_warp_cuda(
    torch::Tensor symbols_bxn, torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl, torch::Tensor cdf_sizes_m, torch::Tensor offsets_m,
    int64_t P_in
) {
    TORCH_CHECK(symbols_bxn.is_cuda() && indexes_bxn.is_cuda(), "CUDA required");
    TORCH_CHECK(cdfs_mxl.is_cuda() && cdf_sizes_m.is_cuda() && offsets_m.is_cuda(), "CUDA required");

    const int B = (int)symbols_bxn.size(0);
    const int N = (int)symbols_bxn.size(1);
    const int Lmax = (int)cdfs_mxl.size(1);
    const int C = (int)cdfs_mxl.size(0);

    auto dev = symbols_bxn.device();
    auto stream = at::cuda::getDefaultCUDAStream();
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);

    int fast_idx_is_channel = 0, HW = 0;
    if (N % C == 0) {
        HW = N / C;
        auto flag = torch::ones({1}, opts_i32);
        int t = 256, blk = (C + t - 1) / t;
        check_idx_is_channel_kernel<<<blk, t, 0, stream>>>(
            indexes_bxn.data_ptr<int32_t>(), N, C, HW, flag.data_ptr<int32_t>());
        fast_idx_is_channel = flag.cpu().item<int32_t>();
    }

    int Pch = (int)P_in;
    if (Pch < 1) Pch = 1;
    if (Pch > C) Pch = C;
    int K = ceil_div_int(C, Pch);
    int chunk_len = Pch * HW;
    int cap_words_per_lane = (chunk_len * 12 + 8) / kWarpLanes + 8;
    int cap_bytes_per_lane = cap_words_per_lane * 4;

    int64_t temp_header_padded = (((int64_t)4 * ((int64_t)K * kWarpLanes + 2)) + 15) & ~((int64_t)15);
    int64_t arena_stride = temp_header_padded + (int64_t)K * kWarpLanes * cap_bytes_per_lane;

    auto temp_arena_u8 = torch::empty({(long long)(arena_stride * B)}, opts_u8);
    auto lane_word_counts_flat = torch::zeros({B * K * kWarpLanes}, opts_i32);
    auto max_rounds_flat = torch::zeros({B * K}, opts_i32);

    // (1) Encode
    warp_encode_chunks_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        symbols_bxn.data_ptr<int32_t>(), indexes_bxn.data_ptr<int32_t>(),
        B, N, cdfs_mxl.data_ptr<int32_t>(), Lmax,
        cdf_sizes_m.data_ptr<int32_t>(), offsets_m.data_ptr<int32_t>(),
        K, chunk_len, HW, temp_arena_u8.data_ptr<uint8_t>(),
        arena_stride, temp_header_padded, cap_words_per_lane,
        lane_word_counts_flat.data_ptr<int32_t>(),
        max_rounds_flat.data_ptr<int32_t>(), fast_idx_is_channel);

    // (2) sizes = max_rounds * 32 (GPU kernel)
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

    // (6) Pack
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

torch::Tensor decode_with_indexes_warp_cuda(
    torch::Tensor packed_u8, torch::Tensor max_rounds_u32,
    torch::Tensor header_bytes_cpu, torch::Tensor chunk_len_cpu,
    torch::Tensor P_cpu, torch::Tensor indexes_bxn,
    torch::Tensor cdfs_mxl, torch::Tensor cdf_sizes_m, torch::Tensor offsets_m
) {
    TORCH_CHECK(packed_u8.is_cuda() && max_rounds_u32.is_cuda(), "CUDA required");

    int B = (int)indexes_bxn.size(0);
    int N = (int)indexes_bxn.size(1);
    int Lmax = (int)cdfs_mxl.size(1);
    int C = (int)cdfs_mxl.size(0);

    int64_t header_bytes = header_bytes_cpu[0].item<int64_t>();
    int chunk_len = chunk_len_cpu[0].item<int32_t>();
    (void)P_cpu;

    TORCH_CHECK(max_rounds_u32.size(0) == B, "max_rounds dim0 mismatch");
    int K = (int)max_rounds_u32.size(1);

    auto dev = packed_u8.device();
    auto stream = at::cuda::getDefaultCUDAStream();
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

    int fast_idx_is_channel = 0, HW = 0;
    if (C > 0 && (N % C == 0)) {
        HW = N / C;
        auto flag = torch::ones({1}, opts_i32);
        int t = 256, blk = (C + t - 1) / t;
        check_idx_is_channel_kernel<<<blk, t, 0, stream>>>(
            indexes_bxn.data_ptr<int32_t>(), N, C, HW, flag.data_ptr<int32_t>());
        fast_idx_is_channel = flag.cpu().item<int32_t>();
    }

    auto max_rounds_i32 = max_rounds_u32.to(torch::kInt32).contiguous().view({B * K});
    auto sizes_u32_flat = torch::empty({B * K}, opts_u32);
    { int n = B * K, t = 256; max_rounds_to_sizes_kernel<<<(n+t-1)/t, t, 0, stream>>>(
        max_rounds_i32.data_ptr<int32_t>(), n, sizes_u32_flat.data_ptr<uint32_t>(), kWarpLanes); }

    auto chunk_offsets_u32 = torch::empty({B * K}, opts_u32);
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (const uint32_t*)nullptr, (uint32_t*)nullptr, B*K, stream.stream());
    auto scan_temp = torch::empty({(long long)scan_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    cub::DeviceScan::ExclusiveSum(scan_temp.data_ptr(), scan_bytes,
        sizes_u32_flat.data_ptr<uint32_t>(), chunk_offsets_u32.data_ptr<uint32_t>(), B*K, stream.stream());

    auto out = torch::empty({B, N}, opts_i32);
    warp_decode_chunks_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        packed_u8.data_ptr<uint8_t>(), header_bytes,
        chunk_offsets_u32.data_ptr<uint32_t>(),
        max_rounds_i32.data_ptr<int32_t>(),
        B, K, N, chunk_len,
        indexes_bxn.data_ptr<int32_t>(),
        cdfs_mxl.data_ptr<int32_t>(), Lmax,
        cdf_sizes_m.data_ptr<int32_t>(), offsets_m.data_ptr<int32_t>(),
        out.data_ptr<int32_t>(), fast_idx_is_channel, HW);

    return out;
}
