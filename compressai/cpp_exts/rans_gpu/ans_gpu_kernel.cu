// compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdint.h>

#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>

#include "rans64_gpu.cuh"
#include "rans64dec_gpu.cuh"

constexpr int precision = 16;
constexpr uint16_t bypass_precision = 4;
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;
constexpr int P_MAX = 512;  // max K in generic mode (and max K overall)

__host__ __device__ __forceinline__ int64_t align16_i64(int64_t x) {
  return (x + 15) & ~((int64_t)15);
}

__device__ __forceinline__ int32_t cdf_find_symbol_bs(
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

static inline int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

// ------------------------------------------------------------
// Fast-path check: idx == channel id (for batch 0) and constant inside channel
// Condition intended: N == C*HW, and idx0[ch*HW + t] == ch
// ------------------------------------------------------------
__global__ void check_idx_is_channel_kernel(
    const int32_t* __restrict__ indexes_bxn, // [B,N]
    int N, int C, int HW,
    int32_t* __restrict__ flag // [1], init to 1
) {
  int ch = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= C) return;
  if (*flag == 0) return;

  const int32_t* idx0 = indexes_bxn; // batch 0
  int base = ch * HW;
  if (base >= N) { atomicExch(flag, 0); return; }

  int32_t v0 = idx0[base];
  if (v0 != ch) { atomicExch(flag, 0); return; }

  int mid  = base + (HW >> 1);
  int last = base + HW - 1;

  if (mid < N)  { if (idx0[mid]  != v0) { atomicExch(flag, 0); return; } }
  if (last < N) { if (idx0[last] != v0) { atomicExch(flag, 0); return; } }
}

// ------------------------------------------------------------
// Stage1 temp-slot encoder
// Grid: (B, K)  one thread per (b, chunk_id)
// If fast_idx_is_channel: cdf_idx = i/HW (no idx loads)
// else: cdf_idx = idx[i]
// ------------------------------------------------------------
__global__ void encode_chunks_into_arena_kernel(
    const int32_t* __restrict__ symbols_bxn,   // [B,N]
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    int B, int N,
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int K,                 // number of chunks
    int chunk_len,         // points per chunk
    int HW,                // only valid if fast_idx_is_channel==1
    uint8_t* __restrict__ arena_u8,            // [B*stride]
    int64_t stride,
    int64_t header_bytes_padded,
    int cap_words_chunk,
    int32_t* __restrict__ used_words_flat,     // [B*K]
    int fast_idx_is_channel                    // 0/1
) {
  // int b = (int)blockIdx.x;
  // int chunk_id = (int)blockIdx.y;
  // if (b >= B || chunk_id >= K) return;
  // if (threadIdx.x != 0) return;

  // int start = chunk_id * chunk_len;
  // if (start >= N) {
  //   used_words_flat[b * K + chunk_id] = 0;
  //   return;
  // }
  // int end = start + chunk_len;
  // if (end > N) end = N;
  int b = (int)blockIdx.x;
  int chunk_id = (int)threadIdx.x + (int)blockIdx.y * (int)blockDim.x;
  if (b >= B || chunk_id >= K) return;

  int start = chunk_id * chunk_len;
  if (start >= N) { used_words_flat[b*K + chunk_id] = 0; return; }
  int end = min(start + chunk_len, N);



  const int32_t* sym = symbols_bxn + (int64_t)b * N;
  const int32_t* idx = indexes_bxn + (int64_t)b * N;

  uint8_t* stream_base = arena_u8 + (int64_t)b * stride;
  uint8_t* payload_base = stream_base + header_bytes_padded;

  int cap_bytes_chunk = cap_words_chunk * 4;
  uint8_t* slot = payload_base + (int64_t)chunk_id * cap_bytes_chunk;

  uint32_t* base_u32 = reinterpret_cast<uint32_t*>(slot);
  uint32_t* ptr = base_u32 + cap_words_chunk;

  Rans64State r;
  Rans64EncInit(&r);

  int32_t last_cdf_idx = -1;
  const int32_t* last_cdf = nullptr;
  int32_t last_cdf_size = 0;
  int32_t last_offsetv = 0;

  for (int i = end - 1; i >= start; --i) {
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
        n_bypass = (msb / bypass_precision) + 1;
      }

      for (int32_t j = n_bypass - 1; j >= 0; --j) {
        uint32_t v = (raw_val >> (j * bypass_precision)) & max_bypass_val;
        Rans64EncPutBits(&r, &ptr, v, bypass_precision);
      }

      int32_t val = n_bypass;
      int32_t n_full = 0;
      while (val >= (int32_t)max_bypass_val) { val -= max_bypass_val; n_full++; }
      Rans64EncPutBits(&r, &ptr, (uint32_t)val, bypass_precision);
      for (int k2 = 0; k2 < n_full; ++k2) {
        Rans64EncPutBits(&r, &ptr, (uint32_t)max_bypass_val, bypass_precision);
      }
    }

    uint32_t start_c = (uint32_t)cdf[value];
    uint32_t freq = (uint32_t)(cdf[value + 1] - cdf[value]);
    Rans64EncPut(&r, &ptr, start_c, freq, precision);
  }

  Rans64EncFlush(&r, &ptr);

  int used_words = (int)(base_u32 + cap_words_chunk - ptr);
  used_words_flat[b * K + chunk_id] = used_words;
}

__global__ void used_words_to_sizes_u32_kernel(
    const int32_t* __restrict__ used_words_flat, // [B*K]
    uint32_t* __restrict__ sizes_u32,            // [B*K]
    int n
) {
  int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t bytes = (uint32_t)used_words_flat[i] * 4u;
  sizes_u32[i] = bytes;  // 不再有 65535 限制
}


// ------------------------------------------------------------
// Tight header layout (little-endian):
//   u32 N
//   u32 chunk_len
//   u16 Pch              (0 if generic mode)
//   u16 K                (#chunks)
//   u16 B
//   u16 flags            (bit0 = fast_idx_is_channel)
//   u32 C                (#cdfs, 0 if generic)
//   u32 HW               (N/C, 0 if generic)
//   u32 sizes[B*K]
//   pad to 16
// ------------------------------------------------------------
__global__ void write_tight_global_header_kernel(
    uint8_t* __restrict__ packed_u8,
    int N, int chunk_len,
    int Pch, int K, int B,
    int flags,
    int C, int HW,
    const uint32_t* __restrict__ sizes_u32, // [B*K]
    int n_sizes
) {
  if (blockIdx.x != 0) return;

  if (threadIdx.x == 0) {
    uint32_t* h32 = reinterpret_cast<uint32_t*>(packed_u8);
    h32[0] = (uint32_t)N;
    h32[1] = (uint32_t)chunk_len;

    uint16_t* h16 = reinterpret_cast<uint16_t*>(packed_u8 + 8);
    h16[0] = (uint16_t)Pch;
    h16[1] = (uint16_t)K;
    h16[2] = (uint16_t)B;
    h16[3] = (uint16_t)flags;

    uint32_t* h32b = reinterpret_cast<uint32_t*>(packed_u8 + 16);
    h32b[0] = (uint32_t)C;
    h32b[1] = (uint32_t)HW;
  }

  // sizes start at packed_u8 + 24
  uint32_t* out_sizes = reinterpret_cast<uint32_t*>(packed_u8 + 24);
  for (int i = threadIdx.x; i < n_sizes; i += blockDim.x) {
    out_sizes[i] = sizes_u32[i];
  }
}

// Pack payload from temp slots into tight payload area
__global__ void pack_tight_payload_kernel(
    const uint8_t* __restrict__ temp_arena_u8,
    int64_t temp_stride,
    int64_t temp_header_padded,
    int cap_bytes_chunk,
    int B, int K,
    const uint32_t* __restrict__ sizes_u32,        // [B*K]
    const uint32_t* __restrict__ chunk_offsets_u32, // [B*K]
    uint8_t* __restrict__ packed_u8,
    int64_t header_bytes
) {
  int b = (int)blockIdx.x;
  int chunk_id = (int)blockIdx.y;
  if (b >= B || chunk_id >= K) return;

  int k = b * K + chunk_id;
  uint32_t used = sizes_u32[k];
  if (used == 0) return;

  const uint8_t* stream_base = temp_arena_u8 + (int64_t)b * temp_stride;
  const uint8_t* payload_base = stream_base + temp_header_padded;
  const uint8_t* slot = payload_base + (int64_t)chunk_id * cap_bytes_chunk;
  // const uint8_t* src = slot + (cap_bytes_chunk - (int)used);
  const uint8_t* src = slot + (int64_t)cap_bytes_chunk - (int64_t)used;

  uint8_t* out_payload = packed_u8 + header_bytes;
  uint8_t* dst = out_payload + (int64_t)chunk_offsets_u32[k];

  for (int i = threadIdx.x; i < (int)used; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// ------------------------------------------------------------
// Tight decode kernel: one block per stream, one thread per chunk
// Grid: (B), blockDim.x >= K
// ------------------------------------------------------------
__global__ void decode_streams_tight_kernel(
    const uint8_t* __restrict__ packed_u8,
    int64_t header_bytes,
    const uint32_t* __restrict__ sizes_u32_flat,        // [B*K]
    const uint32_t* __restrict__ chunk_offsets_u32_flat, // [B*K]
    int B, int K, int N, int chunk_len,
    const int32_t* __restrict__ indexes_bxn,   // [B,N]
    const int32_t* __restrict__ cdfs_mxl,      // [M,Lmax]
    int Lmax,
    const int32_t* __restrict__ cdf_sizes_m,   // [M]
    const int32_t* __restrict__ offsets_m,     // [M]
    int32_t* __restrict__ out_symbols_bxn,     // [B,N]
    int fast_idx_is_channel,
    int HW
) {
  int b = (int)blockIdx.x;
  if (b >= B) return;

  int chunk_id = (int)threadIdx.x;
  if (chunk_id >= K) return;

  int start = chunk_id * chunk_len;
  if (start >= N) return;
  int end = start + chunk_len;
  if (end > N) end = N;

  int k = b * K + chunk_id;
  uint32_t used = sizes_u32_flat[k];
  if (used == 0) return;

  const uint8_t* payload = packed_u8 + header_bytes;
  const uint8_t* chunk_bs = payload + (int64_t)chunk_offsets_u32_flat[k];
  const uint32_t* p = reinterpret_cast<const uint32_t*>(chunk_bs);

  const int32_t* idx = indexes_bxn + (int64_t)b * N;
  int32_t* out = out_symbols_bxn + (int64_t)b * N;

  Rans64State r;
  Rans64DecInit(&r, &p);

  int32_t last_cdf_idx = -1;
  const int32_t* last_cdf = nullptr;
  int32_t last_cdf_size = 0;
  int32_t last_max_value = 0;
  int32_t last_offsetv = 0;

  for (int i = start; i < end; ++i) {
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

    uint32_t cum_freq = Rans64DecGet(&r, precision);
    int32_t s = cdf_find_symbol_bs(cdf, cdf_size, cum_freq);

    uint32_t start_c = (uint32_t)cdf[s];
    uint32_t freq = (uint32_t)(cdf[s + 1] - cdf[s]);
    Rans64DecAdvance(&r, &p, start_c, freq, precision);

    int32_t value = s;

    if (value == max_value) {
      int32_t val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
      int32_t n_bypass = val;
      while (val == (int32_t)max_bypass_val) {
        val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = (int32_t)Rans64DecGetBits(&r, &p, bypass_precision);
        raw_val |= (val << (j * bypass_precision));
      }

      value = raw_val >> 1;
      if (raw_val & 1) value = -value - 1;
      else value += max_value;
    }

    out[i] = value + offsetv;
  }
}

// ------------------------------------------------------------
// Public APIs (tight)
// ------------------------------------------------------------
std::vector<torch::Tensor> encode_with_indexes_tight_cuda(
    torch::Tensor symbols_bxn,   // CUDA int32 [B,N]
    torch::Tensor indexes_bxn,   // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,      // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,   // CUDA int32 [M]
    torch::Tensor offsets_m,     // CUDA int32 [M]
    int64_t P_in
) {
  TORCH_CHECK(symbols_bxn.is_cuda(), "symbols_bxn must be CUDA");
  TORCH_CHECK(indexes_bxn.is_cuda(), "indexes_bxn must be CUDA");
  TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs_mxl must be CUDA");
  TORCH_CHECK(cdf_sizes_m.is_cuda(), "cdf_sizes_m must be CUDA");
  TORCH_CHECK(offsets_m.is_cuda(), "offsets_m must be CUDA");

  const int B = (int)symbols_bxn.size(0);
  const int N = (int)symbols_bxn.size(1);
  const int Lmax = (int)cdfs_mxl.size(1);
  const int C = (int)cdfs_mxl.size(0);

  TORCH_CHECK(B > 0 && N > 0, "Invalid shape");
  TORCH_CHECK(C > 0, "Invalid CDF count");

  auto dev = symbols_bxn.device();
  auto opts_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  // auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(dev);
  auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);


  // ----------------------------------------------------------
  // Decide fast-path: idx == channel (only if N==C*HW)
  // ----------------------------------------------------------
  int fast_idx_is_channel = 0;
  int HW =  0;
  if (N % C == 0) {
    HW = N / C;
    auto flag = torch::ones({1}, opts_i32);
    int threads = 256;
    int blocks = (C + threads - 1) / threads;
    check_idx_is_channel_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        indexes_bxn.data_ptr<int32_t>(), N, C, HW, flag.data_ptr<int32_t>());
    fast_idx_is_channel = flag.cpu().item<int32_t>(); // 1 int sync
  }

  // ----------------------------------------------------------
  // Chunking policy
  // fast mode: P_in means Pch (channels per chunk), K=ceil(C/Pch), chunk_len=Pch*HW
  // generic : P_in means K (num chunks), chunk_len=ceil(N/K), Pch=0
  // ----------------------------------------------------------
  int Pch = 0;
  int K = 0;
  int chunk_len = 0;
  int flags = 0;

  flags |= 1;
  Pch = (int)P_in;
  if (Pch < 1) Pch = 1;
  if (Pch > C) Pch = C;
  K = ceil_div_int(C, Pch);
  chunk_len = Pch * HW;
  
  // temp slot capacity estimate
  const int cap_words_chunk = chunk_len * 12 + 8;
  const int cap_bytes_chunk = cap_words_chunk * 4;

  // temp arena layout: header padding + K slots
  const int64_t temp_header_raw = (int64_t)4 * (K + 2);
  const int64_t temp_header_padded = align16_i64(temp_header_raw);
  const int64_t temp_stride = temp_header_padded + (int64_t)K * (int64_t)cap_bytes_chunk;

  auto temp_arena_u8 = torch::empty({(long long)(temp_stride * (int64_t)B)}, opts_u8);
  auto used_words_flat = torch::zeros({B * K}, opts_i32);

  // int threads = 256;                   
  // int blocksY = (K + threads - 1) / threads;
  int threads = 1;
  while (threads < K) threads <<= 1;
  threads = min(threads, 256);  // 或 512
  int blocksY = (K + threads - 1) / threads;
  dim3 grid(B, blocksY, 1);
  encode_chunks_into_arena_kernel<<<grid, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      symbols_bxn.data_ptr<int32_t>(),
      indexes_bxn.data_ptr<int32_t>(),
      B, N,
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      K,
      chunk_len,
      HW,
      temp_arena_u8.data_ptr<uint8_t>(),
      temp_stride,
      temp_header_padded,
      cap_words_chunk,
      used_words_flat.data_ptr<int32_t>(),
      fast_idx_is_channel);

  // used_words -> sizes_u16
  // auto sizes_u16_flat = torch::empty({B * K}, opts_u16);
  // auto overflow_flag = torch::zeros({1}, opts_i32);

  // {
  //   int n = B * K;
  //   int threads = 256;
  //   int blocks = (n + threads - 1) / threads;
  //   used_words_to_sizes_u16_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
  //       used_words_flat.data_ptr<int32_t>(),
  //       sizes_u16_flat.data_ptr<uint16_t>(),
  //       overflow_flag.data_ptr<int32_t>(),
  //       n);
  // }

  // int32_t overflow = overflow_flag.cpu().item<int32_t>();
  // TORCH_CHECK(overflow == 0, "Chunk compressed size overflow (>65535 bytes). Use smaller chunks or switch sizes to uint32.");

  auto sizes_u32_flat = torch::empty({B * K}, opts_u32);
  {
    int n = B * K;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    used_words_to_sizes_u32_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        used_words_flat.data_ptr<int32_t>(),
        sizes_u32_flat.data_ptr<uint32_t>(),
        n);
  }

  auto chunk_offsets_u32_flat = torch::empty({B * K}, opts_u32);

  auto stream = at::cuda::getDefaultCUDAStream();

  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, scan_temp_bytes,
      (const uint32_t*)nullptr,
      (uint32_t*)nullptr,
      B * K,
      stream.stream());

  auto scan_temp = torch::empty(
      {(long long)scan_temp_bytes},
      torch::TensorOptions().dtype(torch::kUInt8).device(dev));

  cub::DeviceScan::ExclusiveSum(
      scan_temp.data_ptr(),
      scan_temp_bytes,
      sizes_u32_flat.data_ptr<uint32_t>(),
      chunk_offsets_u32_flat.data_ptr<uint32_t>(),
      B * K,
      stream.stream());

  // uint32_t last_off = chunk_offsets_u32_flat.index({B * K - 1}).cpu().item<uint32_t >();
  // uint32_t last_sz  = sizes_u32_flat.index({B * K - 1}).cpu().item<uint32_t>();
  // uint32_t total_payload_bytes = last_off + last_sz;
  int64_t last_off = chunk_offsets_u32_flat[B*K - 1].to(torch::kInt64).cpu().item<int64_t>();
  int64_t last_sz  = sizes_u32_flat[B*K - 1].to(torch::kInt64).cpu().item<int64_t>();
  int64_t total_payload_bytes = last_off + last_sz;

  // header bytes:
  // fixed header = 24 bytes, then sizes (2*(B*K)), then pad to 16
  int64_t header_raw = 24 + (int64_t)4 * (int64_t)(B * K);
  int64_t header_bytes = align16_i64(header_raw);

  int64_t total_bytes = header_bytes + (int64_t)total_payload_bytes;
  auto packed_u8 = torch::empty({(long long)total_bytes}, opts_u8);

  // write header (one block)
  write_tight_global_header_kernel<<<1, 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      packed_u8.data_ptr<uint8_t>(),
      N, chunk_len,
      Pch, K, B,
      flags,
      fast_idx_is_channel ? C : 0,
      fast_idx_is_channel ? HW : 0,
      sizes_u32_flat.data_ptr<uint32_t>(),
      B * K);

  // pack payload
  pack_tight_payload_kernel<<<dim3(B, K, 1), 256, 0, at::cuda::getDefaultCUDAStream()>>>(
      temp_arena_u8.data_ptr<uint8_t>(),
      temp_stride,
      temp_header_padded,
      cap_bytes_chunk,
      B, K,
      sizes_u32_flat.data_ptr<uint32_t>(),
      chunk_offsets_u32_flat.data_ptr<uint32_t>(),
      packed_u8.data_ptr<uint8_t>(),
      header_bytes);

  // sizes as [B, K]
  auto sizes_u32 = sizes_u32_flat.view({B, K});

  // return cpu meta (keep signature compatible: header_bytes, chunk_len, P_cpu=Pch or K? -> here keep P_cpu = P_in_effective (Pch in fast, K in generic))
  auto header_bytes_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  header_bytes_cpu[0] = header_bytes;

  auto chunk_len_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  chunk_len_cpu[0] = chunk_len;

  auto P_cpu = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  // IMPORTANT: for your desired fast mode semantics, P_cpu should be Pch.
  // For generic mode (no fast idx), keep it as K to preserve old meaning.
  P_cpu[0] = fast_idx_is_channel ? Pch : K;

  return {packed_u8, sizes_u32, header_bytes_cpu, chunk_len_cpu, P_cpu};
}

torch::Tensor decode_with_indexes_tight_cuda(
    torch::Tensor packed_u8,       // CUDA uint8 [total]
    torch::Tensor sizes_u32,       // CUDA uint32 [B,K]   (NOTE: K may differ from P_cpu)
    torch::Tensor header_bytes_cpu,// CPU int64 [1]
    torch::Tensor chunk_len_cpu,   // CPU int32 [1]
    torch::Tensor P_cpu,           // CPU int32 [1]   (fast: Pch, generic: K)
    torch::Tensor indexes_bxn,     // CUDA int32 [B,N]
    torch::Tensor cdfs_mxl,        // CUDA int32 [M,Lmax]
    torch::Tensor cdf_sizes_m,     // CUDA int32 [M]
    torch::Tensor offsets_m        // CUDA int32 [M]
) {
  TORCH_CHECK(packed_u8.is_cuda(), "packed_u8 must be CUDA");
  TORCH_CHECK(sizes_u32.is_cuda(), "sizes_u32 must be CUDA");
  TORCH_CHECK(indexes_bxn.is_cuda(), "indexes_bxn must be CUDA");
  TORCH_CHECK(cdfs_mxl.is_cuda(), "cdfs_mxl must be CUDA");
  TORCH_CHECK(cdf_sizes_m.is_cuda(), "cdf_sizes_m must be CUDA");
  TORCH_CHECK(offsets_m.is_cuda(), "offsets_m must be CUDA");

  int B = (int)indexes_bxn.size(0);
  int N = (int)indexes_bxn.size(1);
  int Lmax = (int)cdfs_mxl.size(1);
  int C = (int)cdfs_mxl.size(0);

  int64_t header_bytes = header_bytes_cpu[0].item<int64_t>();
  int chunk_len = chunk_len_cpu[0].item<int32_t>();
  (void)P_cpu; // we do NOT trust P_cpu for K anymore

  TORCH_CHECK(sizes_u32.size(0) == B, "sizes_u32 first dim must be B");
  int K = (int)sizes_u32.size(1);     // CHANGED: K comes from sizes shape
  TORCH_CHECK(K >= 1 && K <= P_MAX, "Invalid K inferred from sizes_u32");

  auto dev = packed_u8.device();
  auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(dev);
  auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);


  // Decide fast path again (safe for general use)
  int fast_idx_is_channel = 0;
  int HW = 0;
  if (C > 0 && (N % C == 0)) {
    HW = N / C;
    auto flag = torch::ones({1}, opts_i32);
    int threads = 256;
    int blocks = (C + threads - 1) / threads;
    check_idx_is_channel_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        indexes_bxn.data_ptr<int32_t>(), N, C, HW, flag.data_ptr<int32_t>());
    fast_idx_is_channel = flag.cpu().item<int32_t>();
  }

  // // sizes_u16 -> sizes_i32_flat
  auto sizes_u32_flat = sizes_u32.contiguous().view({B * K});

  // chunk_offsets = exclusive scan(sizes_i32) length B*K
  auto chunk_offsets_u32_flat = torch::empty({B * K}, opts_u32);
  auto stream = at::cuda::getDefaultCUDAStream();

  size_t scan_temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, scan_temp_bytes,
      (const uint32_t*)nullptr,
      (uint32_t*)nullptr,
      B * K,
      stream.stream());

  auto scan_temp = torch::empty({(long long)scan_temp_bytes},
                                torch::TensorOptions().dtype(torch::kUInt8).device(dev));

  cub::DeviceScan::ExclusiveSum(
      scan_temp.data_ptr(),
      scan_temp_bytes,
      sizes_u32_flat.data_ptr<uint32_t>(),
      chunk_offsets_u32_flat.data_ptr<uint32_t>(),
      B * K,
      stream.stream());

  auto out = torch::empty({B, N}, opts_i32);

  // threads >= K (cap 512)
  int threads = 1;
  while (threads < K) threads <<= 1;
  if (threads > 512) threads = 512;

  decode_streams_tight_kernel<<<B, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      packed_u8.data_ptr<uint8_t>(),
      header_bytes,
      sizes_u32_flat.data_ptr<uint32_t>(),
      chunk_offsets_u32_flat.data_ptr<uint32_t>(),
      B, K, N, chunk_len,
      indexes_bxn.data_ptr<int32_t>(),
      cdfs_mxl.data_ptr<int32_t>(),
      Lmax,
      cdf_sizes_m.data_ptr<int32_t>(),
      offsets_m.data_ptr<int32_t>(),
      out.data_ptr<int32_t>(),
      fast_idx_is_channel,
      HW);

  return out;
}