// compressai/cpp_exts/rans_gpu/warp_rans.cuh
// Warp-level interleaved rANS primitives for 32-lane parallel entropy coding.
//
// Key insight: 32 lanes × 4 bytes/word × 1 word/round = 128 bytes/round,
// which matches the L1 cache line size exactly — guaranteeing coalesced access.
//
// Bitstream layout per chunk:
//   [max_rounds: u32] [round_0: 32×u32] [round_1: 32×u32] ... [round_{R-1}: 32×u32]
// Each chunk is also 128-byte aligned (max_rounds guaranteed even via flush).
//
// Decoding reads directly from interleaved global memory — no de-interleave
// buffer needed. Each lane maintains its own word_idx as a virtual read pointer.

#pragma once
#include <stdint.h>

// ------------------------------------------------------------
// Interleaved payload read
// Lane L's word at round r is at: chunk_base[r * 32 + L]
// ------------------------------------------------------------
__device__ __forceinline__ uint32_t warp_rans_read_word(
    const uint32_t* __restrict__ chunk_base, int lane, int word_idx
) {
    return chunk_base[word_idx * 32 + lane];
}

// ------------------------------------------------------------
// Interleaved version of Rans64DecAdvance
// Reads from interleaved chunk payload instead of contiguous buffer.
// ------------------------------------------------------------
__device__ __forceinline__ void WarpRans64DecAdvance(
    uint64_t* r, const uint32_t* __restrict__ chunk_base,
    int lane, int* word_idx,
    uint32_t start, uint32_t freq, uint32_t scale_bits)
{
    uint64_t mask = (1ull << scale_bits) - 1;
    uint64_t x = *r;
    x = (uint64_t)freq * (x >> scale_bits) + (x & mask) - (uint64_t)start;

    if (x < (1ull << 31)) {
        x = (x << 32) | (uint64_t)chunk_base[(*word_idx) * 32 + lane];
        (*word_idx)++;
    }
    *r = x;
}

// ------------------------------------------------------------
// Interleaved version of Rans64DecGetBits
// Reads from interleaved chunk payload instead of contiguous buffer.
// ------------------------------------------------------------
__device__ __forceinline__ uint32_t WarpRans64DecGetBits(
    uint64_t* r, const uint32_t* __restrict__ chunk_base,
    int lane, int* word_idx, uint32_t n_bits)
{
    uint64_t x = *r;
    uint32_t val = (uint32_t)(x & ((1u << n_bits) - 1));
    x = x >> n_bits;
    if (x < (1ull << 31)) {
        x = (x << 32) | (uint64_t)chunk_base[(*word_idx) * 32 + lane];
        (*word_idx)++;
    }
    *r = x;
    return val;
}

// ------------------------------------------------------------
// Warp-shuffle butterfly reduction: max of 32 lane values.
// 5 shuffle steps → ~5 cycles total.
// ------------------------------------------------------------
__device__ __forceinline__ int warp_reduce_max(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, val, offset);
        if (other > val) val = other;
    }
    return val;
}

// ------------------------------------------------------------
// Warp-shuffle broadcast: lane 0's value to all lanes.
// ------------------------------------------------------------
__device__ __forceinline__ int warp_broadcast(int val) {
    return __shfl_sync(0xffffffff, val, 0);
}
