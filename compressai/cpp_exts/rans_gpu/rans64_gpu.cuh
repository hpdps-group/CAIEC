// compressai/cpp_exts/rans_gpu/rans64_gpu.cuh
#pragma once
#include <stdint.h>

#define RANS64_L (1ull << 31)

typedef uint64_t Rans64State;

// CUDA 下的 64x64 -> high64
__device__ __forceinline__ uint64_t Rans64MulHi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#else
  // host fallback (should not be used in device code path)
    return (uint64_t)(((__uint128_t)a * b) >> 64);
#endif
}

__device__ __forceinline__ void Rans64EncInit(Rans64State* r) { *r = RANS64_L; }

__device__ __forceinline__ void Rans64EncPut(
    Rans64State* r, uint32_t** pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {

    uint64_t x = *r;
    uint64_t x_max = ((RANS64_L >> scale_bits) << 32) * (uint64_t)freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t)x;
        x >>= 32;
    }
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

__device__ __forceinline__ void Rans64EncFlush(Rans64State* r, uint32_t** pptr) {
    uint64_t x = *r;
    *pptr -= 2;
    (*pptr)[0] = (uint32_t)(x >> 0);
    (*pptr)[1] = (uint32_t)(x >> 32);
}

// --------- These two helpers replicate your rans_cpp "bypass bits" ----------
__device__ __forceinline__ void Rans64EncPutBits(
    Rans64State* r, uint32_t** pptr, uint32_t val, uint32_t nbits) {

    // matches: assert(nbits <= 16); assert(val < (1u << nbits));
    uint64_t x = *r;
    uint32_t freq = 1u << (16 - nbits);
    uint64_t x_max = ((RANS64_L >> 16) << 32) * (uint64_t)freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t)x;
        x >>= 32;
    }
    *r = (x << nbits) | (uint64_t)val;
}
