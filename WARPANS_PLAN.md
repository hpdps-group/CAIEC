# WarpANS：32路Warp级交错并行rANS 实现计划

---

## 1. 问题与洞察

### 1.1 当前实现的瓶颈

当前 CAIEC 的 GPU rANS 采用 **chunk-parallel** 策略：每个chunk由一个线程独立编码/解码，chunk间并行。

```
编码: thread k → chunk k → 逆序串行编码 → 1个rANS状态 → flush
解码: thread k → chunk k → 正序串行解码 → 1个rANS状态
```

**核心瓶颈**：chunk内存是完全串行的——每个符号经历过 `Rans64EncPut`（含64位除法、CDF全局内存随机访问、旁路编码），单线程延迟成为上限。

### 1.2 WarpANS 的核心洞察

**一个warp（32线程）处理一个chunk，将chunk内的符号以跨步（stride=32）分配给32个lane，每个lane维护独立的rANS状态。** 32路交错天然匹配GPU的128字节L1缓存行——每个round恰好一个合并访存。

---

## 2. 算法设计

### 2.1 符号分配（跨步分解）

chunk覆盖符号 `[start, end)`，共 `chunk_len` 个符号：

```
编码（逆序跨步）:
  lane L 处理: end-1-L, end-1-L-32, end-1-L-64, ...

解码（正序跨步）:
  lane L 处理: start+L, start+L+32, start+L+64, ...
```

**示例**：chunk_len=320, start=0, end=320

```
编码: lane 0 ── 319, 287, 255, ..., 31   ← 10个符号
      lane 1 ── 318, 286, 254, ..., 30
      ...
      lane 31 ─ 288, 256, 224, ..., 0

解码: lane 0 ── 0, 32, 64, ..., 288      ← 10个符号  
      lane 1 ── 1, 33, 65, ..., 289
      ...
      lane 31 ─ 31, 63, 95, ..., 319
```

### 2.2 编码流程

```
每个warp处理一个chunk: dim3(B, K) 个block，每个block 32线程

Phase 1 — 独立编码:
  lane L 在私有的sub-slot中逆序跨步编码符号
  产出 word_count[L] 个uint32
  rANS flush 固定产出2 word → word_count[L] 必然是偶数

Phase 2 — Warp reduce 求 max_rounds:
  max_rounds = max(word_count[0], ..., word_count[31])
  通过 warp shuffle 的蝴蝶归约（5步）完成

Phase 3 — 交错写入:
  for round in 0..max_rounds-1:
    lane L: output[round*32 + L] = (round < word_count[L]) ? my_word[round] : 0
  总共写入 max_rounds × 32 个word
```

### 2.3 解码流程

```
Phase 1 — 设置:
  从chunk header读 max_rounds
  每个lane计算自己的符号数: my_symbols = (chunk_len + 31 - lane) / 32
  初始化rANS状态: 从 payload[0*32+L] 和 payload[1*32+L] 读前2个word

Phase 2 — 独立解码:
  lane L 从交错payload中正序跨步读取word
  维护 word_idx 模拟连续读取（见§3.1）
  解码 my_symbols 个符号，跨步写入输出

关键: 解码过程中 word_idx 从 0 递增到 word_count[L], 
      绝不会超过 max_rounds（由填充保证）
      绝不会错误地读到0填充word（到达符号数就停止）
```

### 2.4 比特流格式

#### Chunk格式

```
┌──────────────────┬──────────────────────────────────────────────┐
│ max_rounds (u32) │ 交错payload                                  │
│ 4 bytes          │ max_rounds × 32 × 4 bytes                    │
├──────────────────┼──────────────────────────────────────────────┤
│                  │ round 0: [L0_w0][L1_w0]...[L31_w0] (128B)   │
│                  │ round 1: [L0_w1][L1_w1]...[L31_w1] (128B)   │
│                  │ ...                                          │
│                  │ round R-1: [L0_w{R-1}]...[L31_w{R-1}] (128B) │
└──────────────────┴──────────────────────────────────────────────┘

每个chunk大小 = 4 + max_rounds × 128 字节
每个chunk天然128字节对齐（header 4B + padding 124B, 或 header放在紧致header区域不在此处）
```

#### 全局Tight Header扩展

```
当前header (28 bytes):
  u32 N, u32 chunk_len, u16 Pch, u16 K, u16 B, u16 flags, u32 C, u32 HW

扩展header (32 bytes, 16对齐):
  u32 N, u32 chunk_len, u16 Pch, u16 K, u16 B, u16 flags, u16 warp_lanes, u32 C, u32 HW
                                                    ↑ 新增: 1=legacy, 32=WarpANS

当 warp_lanes=32:
  sizes字段: u32 max_rounds[B*K]    （每个chunk 1个u32，而非原来的32个u32）
```

#### 完整比特流布局

```
[全局Tight Header: 32B(padded to 16对齐)]
[max_rounds 数组: B*K × u32]
[padding to 128B 对齐]
[chunk_0 payload: max_rounds[0] × 128 bytes]  ← 128B对齐
[chunk_1 payload: max_rounds[1] × 128 bytes]  ← 128B对齐
...
```

---

## 3. 关键实现细节

### 3.1 交错直接读取的解码原语

解码时不需要先解交错到连续缓冲区，而是**直接从交错的全局内存中读取word**：

```cpp
// 从交错的chunk payload中读取第word_idx个word（lane L的视角）
__device__ __forceinline__ uint32_t warp_rans_read_word(
    const uint32_t* chunk_base, int lane, int word_idx
) {
    return chunk_base[word_idx * 32 + lane];
}

// 修改后的 Rans64DecAdvance（交错版本）
__device__ __forceinline__ void WarpRans64DecAdvance(
    Rans64State* r, const uint32_t* chunk_base, int lane,
    int* word_idx, uint32_t start, uint32_t freq, uint32_t scale_bits
) {
    uint64_t mask = (1ull << scale_bits) - 1;
    uint64_t x = *r;
    x = (uint64_t)freq * (x >> scale_bits) + (x & mask) - start;
    
    if (x < RANS64_L) {
        x = (x << 32) | (uint64_t)chunk_base[(*word_idx) * 32 + lane];
        (*word_idx)++;
    }
    *r = x;
}

// 修改后的 Rans64DecGetBits（交错版本）
__device__ __forceinline__ uint32_t WarpRans64DecGetBits(
    Rans64State* r, const uint32_t* chunk_base, int lane,
    int* word_idx, uint32_t n_bits
) {
    uint64_t x = *r;
    uint32_t val = (uint32_t)(x & ((1u << n_bits) - 1));
    x = x >> n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | (uint64_t)chunk_base[(*word_idx) * 32 + lane];
        (*word_idx)++;
    }
    *r = x;
    return val;
}
```

**访存模式分析**：

```
每个 round：32个lane各读 chunk_base[r*32 + lane]
→ 地址连续: [r*128, r*128+4, r*128+8, ..., r*128+124]
→ 恰好 128 字节 = 1个L1缓存行
→ 单次合并访存（coalesced），100% 缓存行利用率
```

对比原始单线程per-chunk：4/128 ≈ 3% 缓存行利用率。**访存效率提升32倍。**

### 3.2 8字节对齐保证

rANS flush固定产出2个word，正常编码过程中 `Rans64EncPut` 每次可能产出0或1个word。编码结束后检查word_count奇偶性：

```cpp
// 编码完成后
int used_words = base_u32 + cap - ptr;
if (used_words % 2 != 0) {
    // 补一个0 word确保8字节对齐
    *(--ptr) = 0;
    used_words++;
}
```

由于`max_rounds`是各lane word_count的最大值，且每个lane word_count都是偶数，因此：

```
chunk_payload_bytes = max_rounds × 32 × 4 = max_rounds × 128
→ 天然 128 字节对齐
→ 所有chunk从 128 字节对齐的地址开始
→ 合并访存始终命中对齐的缓存行
```

### 3.3 max_rounds的warp reduce

```cpp
// 每个lane已知自己的word_count
int my_words = used_words;
// Butterfly reduce: 5次shuffle得到max
for (int offset = 16; offset > 0; offset >>= 1) {
    int other = __shfl_xor_sync(0xffffffff, my_words, offset);
    my_words = max(my_words, other);
}
int max_rounds = __shfl_sync(0xffffffff, my_words, 0);
```

5条shuffle指令，每条~1 cycle，总共约5 cycles完成32个值的归约。

---

## 4. 内核设计

### 4.1 编码内核：`warp_encode_chunks_kernel`

```
Grid:      dim3(B, K)         每个(B, chunk)对应一个block
BlockDim:  32                 恰好1个warp
SharedMem: 0 bytes            (本阶段不做CDF缓存)

伪代码:
  1. 计算 chunk 范围: start, end
  2. 定位 sub-slot: arena[b][chunk_id * cap_words_per_lane + lane * cap_words_per_lane]
  3. 初始化 Rans64State
  4. for i = end-1-lane downto start step 32:    // 逆序跨步
       编码符号 symbols[b][i]（与原始逻辑一致）
  5. 8字节对齐: if word_count % 2 != 0: 写1个0
  6. warp reduce: max_rounds = max(word_counts)
  7. 交错写入: for r in 0..max_rounds-1:
        output[r*32+lane] = (r < word_count) ? sub_slot[cap-r-1+r] : 0
  8. 写 max_rounds 到 chunk_header
```

### 4.2 解码内核：`warp_decode_chunks_kernel`

```
Grid:      dim3(B, K)
BlockDim:  32
SharedMem: 0 bytes

伪代码:
  1. 从chunk_header读 max_rounds
  2. 计算 my_symbols = (chunk_len + 31 - lane) / 32   // 跨步分配
  3. 定位 chunk_base = payload + chunk_offsets[k]
  4. 从交错的第0和第1个word初始化 Rans64State
  5. word_idx = 2
  6. for s in 0..my_symbols-1:                        // 正序跨步解码
       cdf_idx = fast_idx_is_channel ? (i/HW) : idx[i]
       cum_freq = Rans64DecGet(&r, 16)
       symbol = cdf_binary_search(cdf, cum_freq)
       Rans64DecAdvance_interleaved(&r, chunk_base, lane, &word_idx, ...)
       output[b][start + lane + s*32] = symbol
       // 旁路解码同理使用交错版本的Rans64DecGetBits
```

### 4.3 Host端调度变化

```
当前: 编码 = check_idx + encode_chunks + sizes + scan + header + pack  (6 kernels)
WarpANS: 编码 = check_idx + warp_encode_chunks + max_rounds_scan + header + pack  (5 kernels)
         解码 = scan + warp_decode_chunks  (2 kernels)

关键变化: 
  - sizes kernel 被消除（max_rounds在warp_encode_chunks内部归约得出）
  - encode和interleave在同一个kernel内完成，无需额外的pack kernel
```

---

## 5. P参数的语义与推荐配置

### 5.1 语义定义

```
原始:  parallelism = Pch     → 每个chunk Pch个通道
       K = ceil(C / Pch)     → K个chunk
       chunk_len = Pch × HW  → 每个chunk中1个线程处理的符号数

WarpANS: parallelism = Pch   → 每个chunk Pch个通道（不变）
         K = ceil(C / Pch)   → K个chunk（不变）
         chunk_len = Pch × HW → 每个chunk中32个lane分摊的符号数
         每个lane ≈ chunk_len / 32 个符号
```

### 5.2 推荐配置

WarpANS的优势在于**减少K（chunk数）而不牺牲并行度**：

```
原始模式: P=64  → K=4  → 4个并行线程  → chunk_len=262144
WarpANS:  P=128 → K=2  → 2×32=64个线程 → chunk_len=524288
```

| 模式 | Pch | K | 总并行度 | 每chunk刷新开销 | 总刷新开销 |
|---|---|---|---|---|---|
| Legacy | 16 | 16 | 16线程 | 8B | 128B |
| Legacy | 64 | 4 | 4线程 | 8B | 32B |
| Legacy | 256 | 1 | 1线程 | 8B | 8B |
| **WarpANS** | 128 | 2 | 64线程 | 32×8=256B+填充 | ~512B |
| **WarpANS** | 256 | 1 | 32线程 | 32×8=256B+填充 | ~256B |
| **WarpANS** | 512 | 1 | 32线程 | 32×8=256B+填充 | ~256B |

WarpANS在Pch=256时达到最优：K=1（1个chunk），32路并行，总开销仅256字节。

---

## 6. 开销分析

### 6.1 每个chunk的存储开销

| 项目 | 大小 | 备注 |
|---|---|---|
| max_rounds | 4 bytes | u32，替代原始per-chunk size字段 |
| payload填充 | ≤128 bytes | 32 lane × 4B × 最多1个round的差异 |
| 总开销/chunk | ≤132 bytes | |

### 6.2 与原始实现对比

对于典型场景（C=256, HW=4096, N=C*HW=1,048,576）：

| 模式 | Pch | K | 每chunk开销 | 总开销 | 占压缩比例 |
|---|---|---|---|---|---|
| Legacy | 64 | 4 | size字段: 4B | 16B | ~0.005% |
| Legacy | 16 | 16 | size字段: 4B | 64B | ~0.02% |
| WarpANS | 256 | 1 | max_rounds+填充: ~132B | 132B | ~0.04% |
| WarpANS | 512 | 1 | max_rounds+填充: ~132B | 132B | ~0.04% |

结论：WarpANS的总存储开销与原始P=16模式相当（~132B vs 64B），远小于1%，可忽略。同时WarpANS P=256提供32路并行 vs 原始P=16的16线程并行。

---

## 7. 实现步骤

### Phase 1: warp_rans.cuh — 交错编解码原语（3天）

**文件**: `compressai/cpp_exts/rans_gpu/warp_rans.cuh`

```cpp
#pragma once
#include "rans64_gpu.cuh"
#include "rans64dec_gpu.cuh"

constexpr int WARP_LANES = 32;

// 交错直接读取
__device__ __forceinline__ uint32_t warp_rans_read_word(
    const uint32_t* chunk_base, int lane, int word_idx);

// 交错版本的编解码原语
__device__ __forceinline__ void WarpRans64DecAdvance(
    Rans64State* r, const uint32_t* chunk_base, int lane,
    int* word_idx, uint32_t start, uint32_t freq, uint32_t scale_bits);

__device__ __forceinline__ uint32_t WarpRans64DecGetBits(
    Rans64State* r, const uint32_t* chunk_base, int lane,
    int* word_idx, uint32_t n_bits);

// Warp reduce
__device__ __forceinline__ int warp_reduce_max(int val);
```

### Phase 2: warp_ans_kernel.cu — Warp编解码内核（4天）

**文件**: `compressai/cpp_exts/rans_gpu/warp_ans_kernel.cu`

包含：
- `warp_encode_chunks_kernel` — 编码+交错合并
- `warp_decode_chunks_kernel` — 交错直接解码

### Phase 3: ans_gpu_kernel.cu 扩展（2天）

修改 `encode_with_indexes_tight_cuda` 和 `decode_with_indexes_tight_cuda`：
- 新增 `warp_lanes` 参数（1=legacy, 32=WarpANS）
- warp_lanes=32时调用WarpANS内核
- header格式扩展（新增 `u16 warp_lanes` 字段）
- chunk_offsets 计算适配 `max_rounds × 128` 

### Phase 4: Python绑定与接口（2天）

- 修改 `compressai/cpp_exts/rans_gpu/ans_gpu.cpp`，新增 `warp_lanes` 参数
- 修改 `compressai/entropy_models/entropy_models.py` 的 `TightANS` 和包装函数
- 修改 `compressai/runtime/codecs/compress_packed_gpu.py`，P≥256时自动启用 warp_lanes=32

### Phase 5: 正确性验证（2天）

| 测试 | 方法 |
|---|---|
| Bit-exact 往返 | 随机符号，(warp=32,P=256) 与 (warp=1,P=256) 分别编解码，验证 decode(encode(x))==x |
| 边缘chunk大小 | chunk_len=32, 31, 33（边界情况） |
| 溢出符号 | 显式测试bypass路径（超出CDF范围的符号值） |
| 大batch | B=16, P=256 |
| 端到端 | 集成到TCM/DCAE engine，验证PSNR/bpp与原始一致 |

### Phase 6: 性能Profiling（2天）

| 指标 | 工具 |
|---|---|
| 端到端编码/解码延迟 | cudaEvent |
| SM占用率 | Nsight Compute |
| L1缓存命中率 | Nsight Compute (`l1tex__t_hit`) |
| 合并访存效率 | Nsight Compute (`l1tex__t_requests`, `l1tex__t_sectors`) |
| 与legacy对比 | P=16(legacy) vs P=256(WarpANS) vs P=512(WarpANS) |

---

## 8. 文件清单

| 操作 | 文件 | 描述 |
|---|---|---|
| **新增** | `warp_rans.cuh` | 交错读写原语、WarpRans64Dec*系列函数 |
| **新增** | `warp_ans_kernel.cu` | warp_encode_chunks_kernel, warp_decode_chunks_kernel |
| **修改** | `ans_gpu.cpp` | 新增warp_lanes绑定参数 |
| **修改** | `ans_gpu_kernel.cu` | header扩展(u16 warp_lanes), 调度分支, chunk_offsets适配 |
| **修改** | `entropy_models.py` | TightANS扩展warp_lanes, API透传 |
| **修改** | `compress_packed_gpu.py` | 自动warp选择 |

---

## 9. 时间线

```
第1-3天    warp_rans.cuh 原语实现
第4-7天    warp_ans_kernel.cu 编解码内核
第8-9天    ans_gpu_kernel.cu 集成 + Header格式扩展
第10-11天  Python绑定 + TightANS适配
第12-13天  正确性验证 + Bug修复
第14-15天  性能Profiling + 调优
          ─────────────────────────
          总计: 15个工作日（约3周）
```
