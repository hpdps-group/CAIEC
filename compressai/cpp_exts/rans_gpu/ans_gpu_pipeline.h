// compressai/cpp_exts/rans_gpu/ans_gpu_pipeline.h
// Pipelined WarpANS V3 encoder — Session-based API.
//
// The encoder is split into three phases:
//   1. ans_encode_create_session() — one-time init, pre-allocates all GPU buffers
//   2. ans_encode_launch()         — submits kernels, RETURNS IMMEDIATELY (no sync)
//   3. ans_encode_finalize()       — GPU sync + build result tensors
//
// When used in a pipeline:
//   session = create_session(...)    // once
//   for each batch i:
//       launch(session, sym[i], idx[i])   // non-blocking
//       ga(x[i+1])                        // runs on TRT stream, overlaps with encode
//       pack[i] = finalize(session)       // wait for encode, collect result
//
// This eliminates the GPU→CPU sync from the hot path, enabling real overlap.

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>

struct AnsEncodeSession {
    // ---- Fixed parameters (computed once) ----
    int B = 0;
    int N = 0;
    int C = 0;
    int Lmax = 0;
    int HW = 0;
    int Pch = 0;
    int K = 0;
    int chunk_len = 0;
    int cap_words_per_lane = 0;
    int cap_bytes_per_lane = 0;
    int smem_bytes = 0;
    int fast_idx_is_channel = 0;
    int64_t temp_header_padded = 0;
    int64_t arena_stride = 0;
    int64_t header_bytes = 0;
    int64_t max_packed_bytes = 0;

    // ---- Pre-allocated GPU buffers ----
    torch::Tensor arena_u8;         // uint8  [B * arena_stride]
    torch::Tensor lane_counts;      // int32  [B * K * kWarpLanes]
    torch::Tensor max_rounds;       // int32  [B * K]
    torch::Tensor sizes_u32;        // uint32 [B * K]
    torch::Tensor offsets_u32;      // uint32 [B * K]
    torch::Tensor scan_temp;        // uint8  [scan_bytes]
    torch::Tensor packed_u8;        // uint8  [max_packed_bytes] — upper bound

    // ---- Shared across all sessions ----
    torch::Tensor magic_table;      // uint64 [65536]
    torch::Tensor cdfs;             // int32  [C, Lmax] (reference)
    torch::Tensor cdf_sizes;        // int32  [C] (reference)
    torch::Tensor offsets;          // int32  [C] (reference)
};

// ======================================================================
// Public API (implemented in ans_gpu_pipeline.cu)
// ======================================================================

AnsEncodeSession ans_encode_create_session(
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int B, int N, int64_t P_in, bool fast_idx_is_channel);

void ans_encode_launch(
    AnsEncodeSession& session,
    torch::Tensor symbols_bxn,
    torch::Tensor indexes_bxn);

std::vector<torch::Tensor> ans_encode_finalize(AnsEncodeSession& session);

// ======================================================================
// Decode Session — mirror of encode session
// ======================================================================

struct AnsDecodeSession {
    // ---- Fixed parameters (computed once) ----
    int B = 0;
    int N = 0;
    int C = 0;
    int Lmax = 0;
    int HW = 0;
    int K = 0;
    int chunk_len = 0;
    int smem_bytes = 0;
    int fast_idx_is_channel = 0;

    // ---- Pre-allocated GPU buffers ----
    torch::Tensor max_rounds_i32;    // int32  [B * K]
    torch::Tensor sizes_u32;         // uint32 [B * K]
    torch::Tensor offsets_u32;       // uint32 [B * K]
    torch::Tensor scan_temp;         // uint8  [scan_bytes]
    torch::Tensor output;            // int32  [B, N]

    // ---- CDF references ----
    torch::Tensor cdfs;              // int32  [C, Lmax]
    torch::Tensor cdf_sizes;         // int32  [C]
    torch::Tensor offsets;           // int32  [C]
};

AnsDecodeSession ans_decode_create_session(
    torch::Tensor cdfs_mxl,
    torch::Tensor cdf_sizes_m,
    torch::Tensor offsets_m,
    int B, int N, int K, int chunk_len, int HW, bool fast_idx_is_channel);

void ans_decode_launch(
    AnsDecodeSession& session,
    torch::Tensor packed_u8,
    torch::Tensor max_rounds_u32,
    int64_t header_bytes,
    torch::Tensor indexes_bxn);

torch::Tensor ans_decode_finalize(AnsDecodeSession& session);

// Compute total packed bytes from the output tensor dimensions.
inline int64_t total_packed_bytes(const AnsEncodeSession& sess) {
    int64_t actual_payload = 0;
    if (sess.offsets_u32.numel() > 0 && sess.sizes_u32.numel() > 0) {
        auto last_off = sess.offsets_u32[sess.B * sess.K - 1].item<int64_t>();
        auto last_sz  = sess.sizes_u32[sess.B * sess.K - 1].item<int64_t>();
        actual_payload = (last_off + last_sz) * 4;
    }
    return sess.header_bytes + actual_payload;
}
