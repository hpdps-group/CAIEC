// compressai/cpp_exts/rans_gpu/warp_ans_kernel_v3.cu
// WarpANS V3: V2 + shared memory CDF cache.
//
// Key change from V2: CDF tables loaded into __shared__ memory before
// the per-lane encode/decode loop. This eliminates global memory CDF
// reads (300+ cycles → ~20 cycles) for the hottest path.
//
// Magic table stays in global memory for now (co-location with CDF in
// shared memory planned for future refinement).
//
// Output path (pack/sizes/scan/header) reuses V1 helpers.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"
#include "rans64_gpu_v2.cuh"
#include "rans64dec_gpu.cuh"
#include "warp_rans.cuh"

extern __global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, int N, int C, int HW,
    int32_t* __restrict__ flag);

extern __global__ void warp_pack_tight_payload_kernel(
    const uint8_t*, int64_t, int64_t, int, int, int,
    const int32_t*, const int32_t*, const uint32_t*, uint8_t*, int64_t);

extern __global__ void max_rounds_to_sizes_kernel(
    const int32_t*, int, uint32_t*, int);

extern __global__ void write_warp_tight_header_kernel(
    uint8_t*, int, int, int, int, int, int, int, int, int64_t, const int32_t*, int);

constexpr int kWarpLanes = 32;
constexpr int kPrecision = 16;
constexpr uint16_t kBypassPrec = 4;
constexpr uint16_t kMaxBypass = (1 << kBypassPrec) - 1;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ======================================================================
// V3 ENCODE KERNEL — shared memory CDF cache
// Grid: dim3(B, K), BlockDim: 32
// ======================================================================
__global__ void warp_encode_chunks_kernel_v3(
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
    // ---- Dynamic shared memory ----
    extern __shared__ int32_t smem[];
    int Pch = chunk_len / HW;

    // Layout: cdfs[Pch][Lmax] | cdf_sizes[Pch] | offsets[Pch]
    int32_t* cdf_cache       = smem;
    int32_t* cdf_sizes_cache = smem + Pch * Lmax;
    int32_t* offsets_cache   = cdf_sizes_cache + Pch;

    int b = (int)blockIdx.x;
    int chunk_id = (int)blockIdx.y;
    if (b >= B || chunk_id >= K) return;

    int lane = (int)threadIdx.x;
    int start = chunk_id * chunk_len;
    if (start >= N) {
        int k_idx = b * K * kWarpLanes + chunk_id * kWarpLanes + lane;
        lane_word_counts_flat[k_idx] = 0;
        if (lane == 0) max_rounds_flat[b * K + chunk_id] = 0;
        return;
    }
    int end = start + chunk_len;
    if (end > N) end = N;

    // ---- Phase 1: cooperative load CDFs into shared memory ----
    int total_entries = Pch * Lmax;
    for (int idx = lane; idx < total_entries; idx += kWarpLanes) {
        int ch = idx / Lmax;
        int entry = idx % Lmax;
        int global_ch = chunk_id * Pch + ch;
        cdf_cache[idx] = cdfs_mxl[global_ch * Lmax + entry];
    }
    for (int ch = lane; ch < Pch; ch += kWarpLanes) {
        int global_ch = chunk_id * Pch + ch;
        cdf_sizes_cache[ch] = cdf_sizes_m[global_ch];
        offsets_cache[ch]   = offsets_m[global_ch];
    }
    __syncwarp();

    // ---- Phase 2: per-lane strided reverse encode ----
    int cap_bytes_per_lane = cap_words_per_lane * 4;
    uint8_t* stream_base  = arena_u8 + (int64_t)b * stride;
    uint8_t* payload_base = stream_base + header_bytes_padded;
    uint8_t* chunk_slot   = payload_base + (int64_t)chunk_id * kWarpLanes * cap_bytes_per_lane;
    uint8_t* sub_slot     = chunk_slot + (int64_t)lane * cap_bytes_per_lane;

    uint32_t* base_u32 = reinterpret_cast<uint32_t*>(sub_slot);
    uint32_t* ptr = base_u32 + cap_words_per_lane;

    Rans64State r;
    Rans64EncInit(&r);

    const int32_t* sym = symbols_bxn + (int64_t)b * N;
    const int32_t* idx = indexes_bxn + (int64_t)b * N;

    int32_t last_local_idx = -1;
    const int32_t* last_cdf = nullptr;
    int32_t last_cdf_size = 0, last_offsetv = 0;

    int first = start + lane;
    int last  = first + ((end - 1 - first) / kWarpLanes) * kWarpLanes;
    int used_words = 0;

    if (first < end) {
        for (int i = last; i >= first; i -= kWarpLanes) {
            int32_t cdf_idx  = fast_idx_is_channel ? (i / HW) : idx[i];
            int32_t local_idx = cdf_idx - chunk_id * Pch;  // remap to [0, Pch)

            const int32_t* cdf;
            int32_t cdf_size, offsetv;

            if (local_idx == last_local_idx) {
                cdf = last_cdf; cdf_size = last_cdf_size; offsetv = last_offsetv;
            } else {
                cdf = cdf_cache + (int64_t)local_idx * Lmax;    // ← SHARED MEMORY
                cdf_size = cdf_sizes_cache[local_idx];
                offsetv  = offsets_cache[local_idx];
                last_local_idx = local_idx; last_cdf = cdf;
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
                if (raw_val) { int msb = 31 - __clz(raw_val); n_bypass = (msb / kBypassPrec) + 1; }
                for (int32_t j = n_bypass - 1; j >= 0; --j) {
                    uint32_t v = (raw_val >> (j * kBypassPrec)) & kMaxBypass;
                    Rans64EncPutBits(&r, &ptr, v, kBypassPrec);
                }
                int32_t val = n_bypass; int32_t n_full = 0;
                while (val >= (int32_t)kMaxBypass) { val -= kMaxBypass; n_full++; }
                Rans64EncPutBits(&r, &ptr, (uint32_t)val, kBypassPrec);
                for (int k2 = 0; k2 < n_full; ++k2)
                    Rans64EncPutBits(&r, &ptr, (uint32_t)kMaxBypass, kBypassPrec);
            }

            uint32_t start_c = (uint32_t)cdf[value];
            uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
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

// ======================================================================
// V3 DECODE KERNEL — shared memory CDF cache
// Grid: dim3(B, K), BlockDim: 32
// ======================================================================
__global__ void warp_decode_chunks_kernel_v3(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint32_t* __restrict__ chunk_offsets_u32,
    const int32_t* __restrict__ max_rounds_flat,
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,
    const int32_t* __restrict__ cdfs_mxl, int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int32_t* __restrict__ out_symbols_bxn,
    int fast_idx_is_channel, int HW
) {
    extern __shared__ int32_t smem[];
    int Pch = chunk_len / HW;

    int32_t* cdf_cache       = smem;
    int32_t* cdf_sizes_cache = smem + Pch * Lmax;
    int32_t* offsets_cache   = cdf_sizes_cache + Pch;

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

    // ---- Phase 1: cooperative load CDFs into shared memory ----
    int total_entries = Pch * Lmax;
    for (int idx = lane; idx < total_entries; idx += kWarpLanes) {
        int ch = idx / Lmax;
        int entry = idx % Lmax;
        int global_ch = chunk_id * Pch + ch;
        cdf_cache[idx] = cdfs_mxl[global_ch * Lmax + entry];
    }
    for (int ch = lane; ch < Pch; ch += kWarpLanes) {
        int global_ch = chunk_id * Pch + ch;
        cdf_sizes_cache[ch] = cdf_sizes_m[global_ch];
        offsets_cache[ch]   = offsets_m[global_ch];
    }
    __syncwarp();

    // ---- Phase 2: per-lane strided forward decode ----
    const uint32_t* payload = (const uint32_t*)(packed_u8 + header_bytes);
    const uint32_t* chunk_base = payload + chunk_offsets_u32[k];

    int lane_start = start + lane;
    int my_count = 0;
    if (lane_start < end) my_count = (end - 1 - lane_start) / kWarpLanes + 1;
    if (my_count <= 0) return;

    uint64_t r;
    r = (uint64_t)chunk_base[0*kWarpLanes + lane]
      | ((uint64_t)chunk_base[1*kWarpLanes + lane] << 32);
    int word_idx = 2;

    const int32_t* idx = indexes_bxn + (int64_t)b * N;
    int32_t* out = out_symbols_bxn + (int64_t)b * N;

    int32_t last_local_idx = -1;
    const int32_t* last_cdf = nullptr;
    int32_t last_cdf_size = 0, last_max_value = 0, last_offsetv = 0;

    for (int s = 0; s < my_count; s++) {
        int i = start + lane + s * kWarpLanes;
        int32_t cdf_idx  = fast_idx_is_channel ? (i / HW) : idx[i];
        int32_t local_idx = cdf_idx - chunk_id * Pch;

        const int32_t* cdf;
        int32_t cdf_size, max_value, offsetv;
        if (local_idx == last_local_idx) {
            cdf = last_cdf; cdf_size = last_cdf_size;
            max_value = last_max_value; offsetv = last_offsetv;
        } else {
            cdf = cdf_cache + (int64_t)local_idx * Lmax;        // ← SHARED MEMORY
            cdf_size = cdf_sizes_cache[local_idx];
            max_value = cdf_size - 2;
            offsetv  = offsets_cache[local_idx];
            last_local_idx = local_idx; last_cdf = cdf;
            last_cdf_size = cdf_size; last_max_value = max_value; last_offsetv = offsetv;
        }

        uint32_t cum_freq = (uint32_t)(r & ((1u << kPrecision) - 1));
        int32_t symbol = 0;
        {
            int32_t lo = 0, hi = cdf_size - 2;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if ((uint32_t)cdf[mid + 1] > cum_freq) hi = mid; else lo = mid + 1;
            }
            symbol = lo;
        }

        uint32_t start_c = (uint32_t)cdf[symbol];
        uint32_t freq = (uint32_t)(cdf[symbol + 1] - cdf[symbol]);
        WarpRans64DecAdvance(&r, chunk_base, lane, &word_idx, start_c, freq, kPrecision);

        int32_t value = symbol;
        if (value == max_value) {
            int32_t val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrec);
            int32_t n_bypass = val;
            while (val == (int32_t)kMaxBypass) {
                val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrec);
                n_bypass += val;
            }
            int32_t raw_val = 0;
            for (int j = 0; j < n_bypass; ++j) {
                val = (int32_t)WarpRans64DecGetBits(&r, chunk_base, lane, &word_idx, kBypassPrec);
                raw_val |= (val << (j * kBypassPrec));
            }
            value = raw_val >> 1;
            if (raw_val & 1) value = -value - 1; else value += max_value;
        }
        out[i] = value + offsetv;
    }
}

// ======================================================================
// V3 HOST API
// ======================================================================

std::vector<torch::Tensor> encode_with_indexes_warp_v3_cuda(
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

    // Fast-path detection — done in kernel, no CPU sync.
    // EB/GC always produce channel-based indexes when N % C == 0.
    int HW = 0, fast_idx_is_channel = 1;
    if (N % C == 0) {
        HW = N / C;
    } else {
        fast_idx_is_channel = 0;
    }

    int Pch = (int)P_in; if (Pch < 1) Pch = 1; if (Pch > C) Pch = C;

    // Clamp Pch to fit shared memory budget (37 KB for CDF + sizes + offsets)
    int maxPch = (37 * 1024) / (Lmax * 4 + 8);
    if (Pch > maxPch) Pch = maxPch;
    if (Pch < 1) Pch = 1;

    int K = ceil_div(C, Pch);
    int chunk_len = Pch * HW;
    int cap_words_per_lane = (chunk_len * 12 + 8) / kWarpLanes + 8;
    int cap_bytes_per_lane = cap_words_per_lane * 4;

    // ── Pre-computed sizes (no GPU sync needed) ──
    int64_t temp_header_padded = (((int64_t)4 * ((int64_t)K * kWarpLanes + 2)) + 15) & ~((int64_t)15);
    int64_t arena_stride = temp_header_padded + (int64_t)K * kWarpLanes * cap_bytes_per_lane;
    int64_t header_raw = 32 + 4LL * (B * K);
    int64_t header_bytes = (header_raw + 15) & ~((int64_t)15);

    // ── Buffer cache: arena, lane_counts, max_rounds, sizes, offsets only ──
    struct BufferCache {
        int64_t key_B = -1, key_stride = -1;
        torch::Tensor arena, lane_counts, max_rounds, sizes, offsets;
    };
    static BufferCache g_cache;
    bool reuse = (g_cache.key_B == (int64_t)B && g_cache.key_stride == arena_stride);

    torch::Tensor temp_arena_u8, lane_word_counts_flat, max_rounds_flat,
                  sizes_u32_flat, chunk_offsets_u32;

    temp_arena_u8      = reuse ? g_cache.arena       : torch::empty({(long long)(arena_stride * B)}, opts_u8);
    lane_word_counts_flat = reuse ? g_cache.lane_counts : torch::empty({B * K * kWarpLanes}, opts_i32);
    max_rounds_flat     = reuse ? g_cache.max_rounds  : torch::empty({B * K}, opts_i32);
    sizes_u32_flat      = reuse ? g_cache.sizes       : torch::empty({B * K}, opts_u32);
    chunk_offsets_u32   = reuse ? g_cache.offsets     : torch::empty({B * K}, opts_u32);

    if (!reuse) {
        g_cache.key_B = B;       g_cache.key_stride = arena_stride;
        g_cache.arena = temp_arena_u8;   g_cache.lane_counts = lane_word_counts_flat;
        g_cache.max_rounds = max_rounds_flat;  g_cache.sizes = sizes_u32_flat;
        g_cache.offsets = chunk_offsets_u32;
    }

    // Magic table — built once, cached across calls
    static std::vector<uint64_t> magic_vec = build_magic_table_v2();
    static auto magic_table = torch::from_blob(
        (void*)magic_vec.data(), {(long long)magic_vec.size()},
        torch::TensorOptions().dtype(torch::kUInt64)
    ).to(dev);

    // Shared memory bytes for encode kernel
    int smem_bytes = Pch * Lmax * 4 + Pch * 4 + Pch * 4;

    // (1) V3 Encode — shared memory CDF cache
    warp_encode_chunks_kernel_v3<<<dim3(B, K, 1), kWarpLanes, smem_bytes, stream>>>(
        symbols_bxn.data_ptr<int32_t>(), indexes_bxn.data_ptr<int32_t>(),
        B, N, cdfs_mxl.data_ptr<int32_t>(), Lmax,
        cdf_sizes_m.data_ptr<int32_t>(), offsets_m.data_ptr<int32_t>(),
        K, chunk_len, HW, temp_arena_u8.data_ptr<uint8_t>(),
        arena_stride, temp_header_padded, cap_words_per_lane,
        lane_word_counts_flat.data_ptr<int32_t>(),
        max_rounds_flat.data_ptr<int32_t>(),
        magic_table.data_ptr<uint64_t>(),
        fast_idx_is_channel);

    // (2) sizes kernel
    { int n = B * K, t = 256; max_rounds_to_sizes_kernel<<<(n+t-1)/t, t, 0, stream>>>(
        max_rounds_flat.data_ptr<int32_t>(), n, sizes_u32_flat.data_ptr<uint32_t>(), kWarpLanes); }

    // (3) scan — scan_bytes fixed for given B*K, allocate fresh each time
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr, B*K, stream.stream());
    auto scan_temp = torch::empty({(long long)(scan_bytes > 0 ? scan_bytes : 1)}, opts_u8);
    cub::DeviceScan::ExclusiveSum(scan_temp.data_ptr(), scan_bytes,
        sizes_u32_flat.data_ptr<uint32_t>(),
        chunk_offsets_u32.data_ptr<uint32_t>(), B*K, stream.stream());

    // (4) Allocate packed at exact size (short GPU sync for size query)
    int64_t last_off = chunk_offsets_u32[B*K-1].to(torch::kInt64).cpu().item<int64_t>();
    int64_t last_sz  = sizes_u32_flat[B*K-1].to(torch::kInt64).cpu().item<int64_t>();
    int64_t total_payload_words = last_off + last_sz;
    auto packed_u8 = torch::empty({(long long)(header_bytes + total_payload_words * 4)}, opts_u8);

    int flags = (fast_idx_is_channel ? 1 : 0) | 2;
    write_warp_tight_header_kernel<<<1, 1, 0, stream>>>(
        packed_u8.data_ptr<uint8_t>(), N, chunk_len, K, B,
        flags, kWarpLanes, fast_idx_is_channel ? C : 0, fast_idx_is_channel ? HW : 0,
        header_bytes, max_rounds_flat.data_ptr<int32_t>(), B * K);

    warp_pack_tight_payload_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        temp_arena_u8.data_ptr<uint8_t>(), arena_stride, temp_header_padded,
        cap_words_per_lane, B, K,
        lane_word_counts_flat.data_ptr<int32_t>(),
        max_rounds_flat.data_ptr<int32_t>(),
        chunk_offsets_u32.data_ptr<uint32_t>(),
        packed_u8.data_ptr<uint8_t>(), header_bytes);

    // CPU metadata (trivially small, no GPU sync)
    auto header_bytes_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    header_bytes_cpu[0] = header_bytes;
    auto chunk_len_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    chunk_len_cpu[0] = chunk_len;
    auto P_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    P_cpu[0] = Pch;

    return {packed_u8, max_rounds_flat.view({B, K}).to(torch::kUInt32), header_bytes_cpu, chunk_len_cpu, P_cpu};
}

// V3 decode host API — uses V3 decode kernel with shared memory CDF cache
torch::Tensor decode_with_indexes_warp_v3_cuda(
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
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr, B*K, stream.stream());
    auto scan_temp = torch::empty({(long long)scan_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    cub::DeviceScan::ExclusiveSum(scan_temp.data_ptr(), scan_bytes,
        sizes_u32_flat.data_ptr<uint32_t>(),
        chunk_offsets_u32.data_ptr<uint32_t>(), B*K, stream.stream());

    // Shared memory budget
    int Pch = chunk_len / HW;
    int smem_bytes = Pch * Lmax * 4 + Pch * 4 + Pch * 4;

    auto out = torch::empty({B, N}, opts_i32);
    warp_decode_chunks_kernel_v3<<<dim3(B, K, 1), kWarpLanes, smem_bytes, stream>>>(
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
