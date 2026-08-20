// compressai/cpp_exts/rans_gpu/ans_gpu_pipeline.cu
// Pipelined WarpANS V3 encoder — Session-based implementation.

#include "ans_gpu_pipeline.h"

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"
#include "rans64_gpu_v2.cuh"
#include "rans64dec_gpu.cuh"
#include "warp_rans.cuh"
#include <cuda_runtime.h>

#include <cmath>

extern __global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, int N, int C, int HW,
    int32_t* __restrict__ flag);

extern __global__ void warp_encode_chunks_kernel(
    const int32_t* __restrict__ symbols_bxn,
    const int32_t* __restrict__ indexes_bxn,
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int K, int chunk_len, int HW,
    uint8_t* __restrict__ arena_u8,
    int64_t stride, int64_t header_bytes_padded,
    int cap_words_per_lane,
    int32_t* __restrict__ lane_word_counts_flat,
    int32_t* __restrict__ max_rounds_flat,
    int fast_idx_is_channel);

extern __global__ void warp_encode_chunks_kernel_v3(
    const int32_t* __restrict__ symbols_bxn,
    const int32_t* __restrict__ indexes_bxn,
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl, int Lmax, int C,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int compact_cdf_entries,
    int K, int chunk_len, int HW,
    uint8_t* __restrict__ arena_u8,
    int64_t stride, int64_t header_bytes_padded,
    int cap_words_per_lane,
    int32_t* __restrict__ lane_word_counts_flat,
    int32_t* __restrict__ max_rounds_flat,
    const uint64_t* __restrict__ magic_table,
    int fast_idx_is_channel);

extern __global__ void max_rounds_to_sizes_kernel(
    const int32_t*, int, uint32_t*, int);

extern __global__ void write_warp_tight_header_kernel(
    uint8_t*, int, int, int, int, int, int, int, int, int64_t, const int32_t*, int);

extern __global__ void warp_pack_tight_payload_kernel(
    const uint8_t*, int64_t, int64_t, int, int, int,
    const int32_t*, const int32_t*, const uint32_t*, uint8_t*, int64_t);

extern __global__ void warp_decode_chunks_kernel(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint32_t* __restrict__ chunk_offsets_u32,
    const int32_t* __restrict__ max_rounds_flat,
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,
    const int32_t* __restrict__ cdfs_mxl,
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int32_t* __restrict__ out_symbols_bxn,
    int fast_idx_is_channel,
    int HW);

extern __global__ void warp_decode_chunks_kernel_v3(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint32_t* __restrict__ chunk_offsets_u32,
    const int32_t* __restrict__ max_rounds_flat,
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,
    const int32_t* __restrict__ cdfs_mxl, int Lmax, int C,
    const int32_t* __restrict__ cdf_sizes_m,
    const int32_t* __restrict__ offsets_m,
    int compact_cdf_entries,
    int32_t* __restrict__ out_symbols_bxn,
    int fast_idx_is_channel, int HW);

constexpr int kWarpLanes = 32;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Host magic values are shared; each session owns a device-local copy.
static const std::vector<uint64_t> g_magic_vec = build_magic_table_v2();

// ======================================================================
// SESSION INIT
// ======================================================================

AnsEncodeSession ans_encode_create_session(
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int B, int N, int64_t P_in, bool fast_idx_is_channel)
{
    TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs must be CUDA");
    int C = (int)cdfs_mxl.size(0);
    int Lmax = (int)cdfs_mxl.size(1);

    auto dev = cdfs_mxl.device();
    c10::cuda::CUDAGuard device_guard(dev);
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);

    AnsEncodeSession s;
    s.B = B;
    s.N = N;
    s.C = C;
    s.Lmax = Lmax;

    s.fast_idx_is_channel = fast_idx_is_channel ? 1 : 0;
    TORCH_CHECK(N % C == 0, "session requires N divisible by C");
    s.HW = N / C;

    // ---- Compute Pch / K / chunk_len / capacity ----
    s.Pch = (int)P_in; if (s.Pch < 1) s.Pch = 1; if (s.Pch > C) s.Pch = C;
    int maxPch = (37 * 1024) / (Lmax * 4 + 8);
    if (s.Pch > maxPch) s.Pch = maxPch;
    if (s.Pch < 1) s.Pch = 1;

    s.K = ceil_div(C, s.Pch);
    s.chunk_len = s.Pch * s.HW;
    s.cap_words_per_lane = (s.chunk_len * 12 + 8) / kWarpLanes + 8;
    s.cap_bytes_per_lane = s.cap_words_per_lane * 4;
    s.smem_bytes = s.Pch * Lmax * 4 + s.Pch * 4 + s.Pch * 4;

    // ---- Buffer sizes ----
    s.temp_header_padded = (((int64_t)4 * ((int64_t)s.K * kWarpLanes + 2)) + 15) & ~((int64_t)15);
    s.arena_stride = s.temp_header_padded + (int64_t)s.K * kWarpLanes * s.cap_bytes_per_lane;
    int64_t header_raw = 32 + 4LL * (B * s.K);
    s.header_bytes = (header_raw + 15) & ~((int64_t)15);

    // max_packed_bytes = upper bound.
    // Each symbol ≤1 word in rANS. Flush adds 2 words/lane.
    s.max_packed_bytes = s.header_bytes
        + (int64_t)B * N * 4                    // symbol words (upper bound)
        + (int64_t)s.K * B * kWarpLanes * 2 * 4; // flush overhead

    // ---- Allocate buffers ----
    s.arena_u8      = torch::empty({(long long)(s.arena_stride * B)}, opts_u8);
    s.lane_counts   = torch::empty({B * s.K * kWarpLanes}, opts_i32);
    s.max_rounds    = torch::empty({B * s.K}, opts_i32);
    s.sizes_u32     = torch::empty({B * s.K}, opts_u32);
    s.offsets_u32   = torch::empty({B * s.K}, opts_u32);

    // scan_temp size
    size_t scan_bytes = 0;
    auto init_stream = at::cuda::getDefaultCUDAStream();
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr, B * s.K,
        init_stream.stream());
    s.scan_temp     = torch::empty({(long long)(scan_bytes > 0 ? scan_bytes : 1)}, opts_u8);

    s.packed_u8     = torch::empty({s.max_packed_bytes}, opts_u8);

    // ---- Device-local magic table owned by this session ----
    s.magic_table = torch::from_blob(
        const_cast<uint64_t*>(g_magic_vec.data()),
        {(long long)g_magic_vec.size()},
        torch::TensorOptions().dtype(torch::kUInt64)
    ).to(dev);

    // ---- CDF references (keep alive) ----
    s.cdfs       = cdfs_mxl;
    s.cdf_sizes  = cdf_sizes_m;
    s.offsets    = offsets_m;

    return s;
}

// ======================================================================
// ASYNC LAUNCH — submits all kernels, returns immediately.  No GPU sync.
// ======================================================================

void ans_encode_launch(
    AnsEncodeSession& s,
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn)
{
    c10::cuda::CUDAGuard device_guard(s.cdfs.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    int B = s.B, N = s.N, K = s.K, Lmax = s.Lmax;
    int C = (int)s.cdfs.size(0);
    int chunk_len = s.chunk_len, HW = s.HW;
    int cap_words_per_lane = s.cap_words_per_lane;
    int fast_idx = s.fast_idx_is_channel;

    // (1) Encode kernel → arena, lane_counts, max_rounds. The V3 cache is
    // valid only for channel-major indexes; generic indexes require global CDFs.
    if (fast_idx) {
        warp_encode_chunks_kernel_v3<<<dim3(B, K, 1), kWarpLanes, s.smem_bytes, stream>>>(
            symbols_bxn.data_ptr<int32_t>(), indexes_bxn.data_ptr<int32_t>(), B, N,
            s.cdfs.data_ptr<int32_t>(), Lmax, C, s.cdf_sizes.data_ptr<int32_t>(),
            s.offsets.data_ptr<int32_t>(), 0, K, chunk_len, HW,
            s.arena_u8.data_ptr<uint8_t>(), s.arena_stride, s.temp_header_padded,
            cap_words_per_lane, s.lane_counts.data_ptr<int32_t>(),
            s.max_rounds.data_ptr<int32_t>(), s.magic_table.data_ptr<uint64_t>(),
            fast_idx);
    } else {
        warp_encode_chunks_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
            symbols_bxn.data_ptr<int32_t>(), indexes_bxn.data_ptr<int32_t>(), B, N,
            s.cdfs.data_ptr<int32_t>(), Lmax, s.cdf_sizes.data_ptr<int32_t>(),
            s.offsets.data_ptr<int32_t>(), K, chunk_len, HW,
            s.arena_u8.data_ptr<uint8_t>(), s.arena_stride, s.temp_header_padded,
            cap_words_per_lane, s.lane_counts.data_ptr<int32_t>(),
            s.max_rounds.data_ptr<int32_t>(), fast_idx);
    }

    // (2) max_rounds → sizes
    { int n = B * K, t = 256;
      max_rounds_to_sizes_kernel<<<(n+t-1)/t, t, 0, stream>>>(
        s.max_rounds.data_ptr<int32_t>(), n,
        s.sizes_u32.data_ptr<uint32_t>(), kWarpLanes); }

    // (3) cub::DeviceScan → offsets
    size_t scan_bytes = s.scan_temp.numel();
    cub::DeviceScan::ExclusiveSum(
        s.scan_temp.data_ptr(), scan_bytes,
        s.sizes_u32.data_ptr<uint32_t>(),
        s.offsets_u32.data_ptr<uint32_t>(), B * K,
        stream.stream());

    // (4) Header
    int flags = (fast_idx ? 1 : 0) | 2;
    int64_t hdr = s.header_bytes;
    write_warp_tight_header_kernel<<<1, 1, 0, stream>>>(
        s.packed_u8.data_ptr<uint8_t>(),
        N, chunk_len, K, B,
        flags, kWarpLanes,
        fast_idx ? s.C : 0, fast_idx ? HW : 0,
        hdr, s.max_rounds.data_ptr<int32_t>(), B * K);

    // (5) Pack payload
    warp_pack_tight_payload_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
        s.arena_u8.data_ptr<uint8_t>(),
        s.arena_stride, s.temp_header_padded,
        cap_words_per_lane, B, K,
        s.lane_counts.data_ptr<int32_t>(),
        s.max_rounds.data_ptr<int32_t>(),
        s.offsets_u32.data_ptr<uint32_t>(),
        s.packed_u8.data_ptr<uint8_t>(), hdr);
}

// ======================================================================
// FINALIZE — GPU sync + build result tensors
// ======================================================================

std::vector<torch::Tensor> ans_encode_finalize(AnsEncodeSession& s) {
    c10::cuda::CUDAGuard device_guard(s.cdfs.device());
    // Sync: ensure all kernels on default stream are done
    auto stream = at::cuda::getDefaultCUDAStream();
    auto err = cudaStreamSynchronize(stream.stream());
    TORCH_CHECK(err == cudaSuccess, "cudaStreamSynchronize failed: ", cudaGetErrorString(err));

    // Read actual packed size from scan result
    int last_idx = s.B * s.K - 1;
    int64_t last_off = s.offsets_u32[last_idx].item<int64_t>();
    int64_t last_sz  = s.sizes_u32[last_idx].item<int64_t>();
    int64_t actual_payload_bytes = (last_off + last_sz) * 4;
    int64_t actual_total = s.header_bytes + actual_payload_bytes;

    TORCH_CHECK(actual_total <= s.max_packed_bytes,
        "Packed buffer overflow: actual=", actual_total,
        " max=", s.max_packed_bytes);

    // Slice the pre-allocated buffer to exact size
    auto packed_out = s.packed_u8.narrow(0, 0, actual_total).clone();

    // max_rounds as uint32 [B, K]
    auto max_rounds_out = s.max_rounds.view({s.B, s.K}).to(torch::kUInt32);

    // CPU metadata
    auto header_bytes_cpu = torch::empty({1},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    header_bytes_cpu[0] = s.header_bytes;
    auto chunk_len_cpu = torch::empty({1},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    chunk_len_cpu[0] = s.chunk_len;
    auto P_cpu = torch::empty({1},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    P_cpu[0] = s.Pch;

    return {packed_out, max_rounds_out, header_bytes_cpu, chunk_len_cpu, P_cpu};
}

// ======================================================================
// DECODE SESSION INIT
// ======================================================================

AnsDecodeSession ans_decode_create_session(
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int B, int N, int K, int chunk_len, int HW, bool fast_idx_is_channel)
{
    TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs must be CUDA");
    int C = (int)cdfs_mxl.size(0);
    int Lmax = (int)cdfs_mxl.size(1);

    auto dev = cdfs_mxl.device();
    c10::cuda::CUDAGuard device_guard(dev);
    auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);

    AnsDecodeSession s;
    s.B = B; s.N = N; s.C = C; s.Lmax = Lmax;
    s.HW = HW; s.K = K; s.chunk_len = chunk_len;
    s.fast_idx_is_channel = fast_idx_is_channel ? 1 : 0;

    int Pch = chunk_len / HW;
    s.smem_bytes = Pch * Lmax * 4 + Pch * 4 + Pch * 4;

    // Pre-allocate buffers
    s.max_rounds_i32 = torch::empty({B * K}, opts_i32);
    s.sizes_u32      = torch::empty({B * K}, opts_u32);
    s.offsets_u32    = torch::empty({B * K}, opts_u32);
    s.output         = torch::empty({B, N}, opts_i32);

    // scan_temp size
    size_t scan_bytes = 0;
    auto init_stream = at::cuda::getDefaultCUDAStream();
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes,
        (const uint32_t*)nullptr, (uint32_t*)nullptr, B * K,
        init_stream.stream());
    s.scan_temp      = torch::empty({(long long)(scan_bytes > 0 ? scan_bytes : 1)}, opts_u8);

    // CDF references
    s.cdfs       = cdfs_mxl;
    s.cdf_sizes  = cdf_sizes_m;
    s.offsets    = offsets_m;

    return s;
}

// ======================================================================
// DECODE ASYNC LAUNCH
// ======================================================================

void ans_decode_launch(
    AnsDecodeSession& s,
    torch::Tensor packed_u8,
    torch::Tensor max_rounds_u32,
    int64_t header_bytes,
    torch::Tensor indexes_bxn)
{
    c10::cuda::CUDAGuard device_guard(s.cdfs.device());
    auto stream = at::cuda::getDefaultCUDAStream();

    int B = s.B, N = s.N, K = s.K, Lmax = s.Lmax;
    int C = (int)s.cdfs.size(0);
    int chunk_len = s.chunk_len, HW = s.HW;

    int fast_idx = s.fast_idx_is_channel;

    // Convert max_rounds to int32 (non-blocking, on ANS stream)
    s.max_rounds_i32 = max_rounds_u32.to(torch::kInt32).contiguous().view({B * K});

    // (1) max_rounds → sizes
    { int n = B * K, t = 256;
      max_rounds_to_sizes_kernel<<<(n+t-1)/t, t, 0, stream>>>(
        s.max_rounds_i32.data_ptr<int32_t>(), n,
        s.sizes_u32.data_ptr<uint32_t>(), kWarpLanes); }

    // (2) cub::DeviceScan → offsets
    size_t scan_bytes = s.scan_temp.numel();
    cub::DeviceScan::ExclusiveSum(
        s.scan_temp.data_ptr(), scan_bytes,
        s.sizes_u32.data_ptr<uint32_t>(),
        s.offsets_u32.data_ptr<uint32_t>(), B * K,
        stream.stream());

    // (3) Decode with the same CDF-access mode used by the encoder.
    if (fast_idx) {
        warp_decode_chunks_kernel_v3<<<dim3(B, K, 1), kWarpLanes, s.smem_bytes, stream>>>(
            packed_u8.data_ptr<uint8_t>(), header_bytes,
            s.offsets_u32.data_ptr<uint32_t>(), s.max_rounds_i32.data_ptr<int32_t>(),
            B, K, N, chunk_len, indexes_bxn.data_ptr<int32_t>(),
            s.cdfs.data_ptr<int32_t>(), Lmax, C, s.cdf_sizes.data_ptr<int32_t>(),
            s.offsets.data_ptr<int32_t>(), 0,
            s.output.data_ptr<int32_t>(), fast_idx, HW);
    } else {
        warp_decode_chunks_kernel<<<dim3(B, K, 1), kWarpLanes, 0, stream>>>(
            packed_u8.data_ptr<uint8_t>(), header_bytes,
            s.offsets_u32.data_ptr<uint32_t>(), s.max_rounds_i32.data_ptr<int32_t>(),
            B, K, N, chunk_len, indexes_bxn.data_ptr<int32_t>(),
            s.cdfs.data_ptr<int32_t>(), Lmax, s.cdf_sizes.data_ptr<int32_t>(),
            s.offsets.data_ptr<int32_t>(), s.output.data_ptr<int32_t>(), fast_idx, HW);
    }
}

// ======================================================================
// DECODE FINALIZE
// ======================================================================

torch::Tensor ans_decode_finalize(AnsDecodeSession& s) {
    c10::cuda::CUDAGuard device_guard(s.cdfs.device());
    auto stream = at::cuda::getDefaultCUDAStream();
    auto err = cudaStreamSynchronize(stream.stream());
    TORCH_CHECK(err == cudaSuccess, "cudaStreamSynchronize failed: ", cudaGetErrorString(err));
    return s.output.clone();
}
