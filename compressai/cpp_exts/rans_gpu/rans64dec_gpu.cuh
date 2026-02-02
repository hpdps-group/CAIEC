// compressai/cpp_exts/rans_gpu/rans64dec_gpu.cuh
#pragma once
#include <stdint.h>

#define RANS64_L (1ull << 31)
typedef uint64_t Rans64State;

__device__ __forceinline__ void Rans64DecInit(Rans64State* r, const uint32_t** pptr) {
    uint64_t x;
    x  = (uint64_t)((*pptr)[0]) << 0;
    x |= (uint64_t)((*pptr)[1]) << 32;
    *pptr += 2;
    *r = x;
}

__device__ __forceinline__ uint32_t Rans64DecGet(Rans64State* r, uint32_t scale_bits) {
    return (uint32_t)(*r & ((1u << scale_bits) - 1));
}

__device__ __forceinline__ void Rans64DecAdvance(
    Rans64State* r, const uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {

    uint64_t mask = (1ull << scale_bits) - 1;
    uint64_t x = *r;
    x = (uint64_t)freq * (x >> scale_bits) + (x & mask) - start;

    if (x < RANS64_L) {
        x = (x << 32) | (uint64_t)(**pptr);
        *pptr += 1;
    }
    *r = x;
}

// ---- match your CPU rans_cpp Rans64DecGetBits exactly ----
__device__ __forceinline__ uint32_t Rans64DecGetBits(
    Rans64State* r, const uint32_t** pptr, uint32_t n_bits) {

    uint64_t x = *r;
    uint32_t val = (uint32_t)(x & ((1u << n_bits) - 1));

    x = x >> n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | (uint64_t)(**pptr);
        *pptr += 1;
    }
    *r = x;
    return val;
}
