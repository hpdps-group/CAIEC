// compressai/cpp_exts/rans_gpu/rans64_gpu_v2.cuh
// V2 rANS encode primitives: strength-reduced division via __umul64hi.
//
// Replaces 64-bit integer division (x / freq, x % freq) with:
//   magic = floor(2^64 / freq)               (precomputed, one per possible freq)
//   q     = __umul64hi(x, magic)             (~4 cycles)
//   r     = x - q * freq                      (~4 cycles)
//   if (r >= freq) { q++; r -= freq; }        (correction, rarely taken)
//
// __umul64hi(x, floor(2^64/freq)) yields floor(x/freq) or floor(x/freq)-1.
// The correction branch handles the -1 case, occurring for ~50% of x values
// (the actual rate depends on freq).
//
// Expected speedup: 3-5x for the division step (30-60 cycles → ~12 cycles).

#pragma once
#include <stdint.h>

// ------------------------------------------------------------
// Lookup magic number: magic[freq] = floor(2^64 / freq)
// ------------------------------------------------------------
__device__ __forceinline__ uint64_t rans_v2_magic_lookup(
    const uint64_t* __restrict__ magic_table, uint32_t freq)
{
    return magic_table[freq];
}

// ------------------------------------------------------------
// V2 Rans64EncPut — fast division version
//
// Equivalent to V1:
//   *r = ((x / freq) << scale_bits) + (x % freq) + start;
// but uses __umul64hi instead of 64-bit / and %.
// ------------------------------------------------------------
__device__ __forceinline__ void Rans64EncPutV2(
    uint64_t* r, uint32_t** pptr,
    uint32_t start, uint32_t freq, uint32_t scale_bits,
    const uint64_t* __restrict__ magic_table)
{
    uint64_t x = *r;
    uint64_t x_max = ((0x80000000ULL >> scale_bits) << 32) * (uint64_t)freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t)x;
        x >>= 32;
    }

    // --- Strength-reduced division ---
    uint64_t q, r32;
    if (freq <= 1) {
        q = (freq == 1) ? x : 0;
        r32 = (freq == 1) ? 0 : x;
    } else {
        uint64_t magic = magic_table[freq];
        q = __umul64hi(x, magic);         // approx floor(x / freq), may be off by -1
        r32 = x - q * (uint64_t)freq;
        // Correction: q may underestimate by 1, then r32 >= freq
        if (r32 >= freq) {
            q++;
            r32 -= freq;
        }
    }

    *r = (q << scale_bits) + ((uint32_t)r32 + start);
}

// ------------------------------------------------------------
// Host-side magic table builder
// magic[freq] = floor(2^64 / freq)  for freq ∈ [2, 65535]
// ------------------------------------------------------------
#include <cstdint>
#include <vector>

inline std::vector<uint64_t> build_magic_table_v2() {
    std::vector<uint64_t> table(65536, 0);
    for (uint64_t f = 2; f <= 65535; f++) {
        table[f] = 0xFFFFFFFFFFFFFFFFULL / f;  // floor(2^64 / f)
    }
    // freq = 0, 1: not used, stay 0
    return table;
}
