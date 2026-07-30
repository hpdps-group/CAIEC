# 面向 GPU 加速非对称数字系统的并行优化研究计划

## 投稿目标：PPoPP 2027（Principles and Practice of Parallel Programming）

---

## 1. 问题陈述与研究动机

### 1.1 背景

非对称数字系统（Asymmetric Numeral Systems, ANS）已成为AI图像压缩和科学数据压缩中的事实标准熵编码算法，在压缩率和单遍编码方面均优于传统的 Huffman 编码和算术编码。然而，ANS 存在根本性的**顺序依赖**——每个符号的编码状态依赖于前一个符号的状态——这给 GPU 加速带来了核心挑战。

当前最先进的 GPU ANS 实现（CAIEC, VLDB 2026）通过**将符号流划分为独立chunk**来实现并行化，每个chunk由单个GPU线程编码。我们称这种方法为**chunk并行rANS**。该方法面临根本性的权衡：更多chunk提高并行度但降低压缩率（每个chunk需要独立的rANS状态刷新，约增加8字节开销）。这形成了一个制约吞吐量和压缩率的 Pareto 前沿。

### 1.2 当前实现的五大瓶颈

| 瓶颈编号 | 瓶颈名称 | 根源 | 量化影响 |
|---|---|---|---|
| **B1** | Chunk内串行瓶颈 | 每个chunk仅一个线程处理，chunk内完全串行rANS | 每SM吞吐量上限约50 Msymbols/s |
| **B2** | 64位整数除法 | `Rans64EncPut` 每符号执行 `x/freq` 和 `x%freq`，GPU无原生64位除法指令 | 每符号20-40周期仅用于除法 |
| **B3** | 全局内存CDF随机访问 | CDF表存储在全局内存中，每符号至少一次随机访问 | 每次CDF查找未命中200-800周期 |
| **B4** | 内核启动开销膨胀 | 每次编码/解码启动5+个内核 + 2次CUB扫描 | 每次压缩调用约50-100μs启动开销 |
| **B5** | CPU-GPU同步 | fast-path检测标志每次调用时执行 `cudaMemcpy` | 每次调用约5-10μs同步开销 |

### 1.3 研究问题

- **RQ1**：能否通过 warp 级 SIMD 分解，在单个 rANS 流内部挖掘细粒度并行性，打破chunk内串行瓶颈？
- **RQ2**：能否通过面向rANS频率域特化的预计算模逆，消除整数除法开销？
- **RQ3**：能否通过持久内核（persistent kernel）设计配合共享内存CDF缓存，消除内核启动开销并减少全局内存访问？
- **RQ4**：GPU加速熵编码的吞吐量-压缩率理论 Pareto 前沿是什么？我们离它有多近？

---

## 2. 优化方案设计

### 2.1 O1：Warp级交错并行 rANS（WarpANS）

**核心思想**：将一个chunk的符号流分解到32个 warp lane 上，每个lane维护独立的rANS状态，编码后通过 warp shuffle 指令合并状态。

#### 2.1.1 编码算法

```
规格：32-lane warp，处理一个含 N 个符号的chunk
┌──────────────────────────────────────────────────────┐
│ 第1步：跨步分配（Strided Distribution）              │
│   lane k 处理符号 {k, k+32, k+64, ...}              │
│   按逆序处理（满足rANS后进先出的要求）               │
│   每个lane维护独立的 Rans64State r[k]                │
├──────────────────────────────────────────────────────┤
│ 第2步：逐lane独立rANS编码                            │
│   for s in reversed(lane_symbols):                   │
│     Rans64EncPut(&r[k], &ptr[k], start, freq, 16)    │
│   Rans64EncFlush(&r[k], &ptr[k])                     │
│   每个lane写入独立的输出缓冲区                       │
├──────────────────────────────────────────────────────┤
│ 第3步：Warp shuffle交错合并                          │
│   产出交错比特流：                                   │
│   lane0_word0, lane1_word0, ..., lane31_word0,       │
│   lane0_word1, lane1_word1, ...                      │
│                                                       │
│   mask = __activemask()                               │
│   for round in 0..max_words_per_lane:                 │
│     word = (round < my_cnt) ? my_buf[round] : 0       │
│     for offset in [1, 2, 4, 8, 16]:                  │
│       word = __shfl_xor_sync(mask, word, offset)      │
│     写入交错输出                                      │
└──────────────────────────────────────────────────────┘
```

#### 2.1.2 解码算法

```
┌──────────────────────────────────────────────────────┐
│ 第1步：Warp shuffle 交错解合并（逆转编码的合并过程） │
│   从交错比特流中恢复各lane的独立流                   │
├──────────────────────────────────────────────────────┤
│ 第2步：逐lane独立rANS解码                            │
│   Rans64DecInit(&r[k], &ptr[k])                      │
│   for i in 0..symbols_per_lane:                      │
│     cum_freq = Rans64DecGet(&r[k], 16)               │
│     s = binary_search_cdf(cdf, cum_freq)              │
│     Rans64DecAdvance(&r[k], &ptr[k], ...)             │
│     output[k + 32*i] = s  // 跨步写入                │
└──────────────────────────────────────────────────────┘
```

#### 2.1.3 压缩率分析

每个 lane 独立刷新 rANS 状态，增加 8 字节开销。总开销：
- Chunk并行（当前）：`8 × K` 字节（K = chunk数）
- WarpANS：`8 × K × 32` 字节（每chunk内32个lane）

但 WarpANS 允许使用**更大的chunk**（更少的K）而不牺牲并行度，可能实现**更低**的总开销：

- Chunk并行 P=16：K=16，开销=128字节
- WarpANS P=128：K=2，开销=2×32×8=512字节
- Chunk并行 P=128：K=2，开销=16字节

核心贡献之一是**分析该权衡并找到吞吐量-压缩率Pareto前沿上的最优点**。

#### 2.1.4 理论分析

我们计划给出以下理论界限：
- **定理1（并行度下界）**：将N个符号划分为K个独立编码分区的任何GPU ANS算法，必须产生至少 `K × 8` 字节的状态刷新开销
- **定理2（WarpANS开销）**：WarpANS使用W个warp lane时，开销为 `K × W × 8` 字节。对于目标并行度 `L = K × W`，使给定吞吐量下开销最小化的最优K-W分解在 `W = min(32, L_opt)` 处达到
- **定理3（加速比上界）**：WarpANS相比单线程每chunk的加速比为 `S = W / (1 + αW/N)`，其中α为warp shuffle每次归并成本，N为每chunk符号数

---

### 2.2 O2：基于预计算模逆的强度削弱

**核心思想**：用预计算的定点倒数乘法替代 `Rans64EncPut` 中的64位整数除法。

#### 2.2.1 数学推导

rANS编码的核心运算是：
```
x_new = ((x / freq) << scale_bits) | ((x % freq) + start)
```

对于 `scale_bits = 16`，`freq ∈ [1, 2^16)`。对每个可能的 `freq` 预计算：
```
magic[freq] = floor(2^k / freq)    // k选取得当以避免溢出
shift[freq] = k
```

则除法转化为：
```
quotient  = __umul64hi(x, magic[freq]) >> shift[freq]
remainder = x - quotient * freq
x_new     = (quotient << 16) | (remainder + start)
```

在 H100（SM90）上 `__umul64hi` 约需 4 个周期，而软件模拟的64位除法需 30-60 个周期，该操作可获得 **5-15倍加速**。

#### 2.2.2 内存开销分析

对 `freq ∈ [1, 65536]`，存储 `magic` 和 `shift`：
- 朴素方案：`65536 × 8 × 2 = 1 MB` 每CDF表
- 量化方案（仅存储CDF中出现的值）：通常 < 100 个不同频率值/表
- 共享内存方案：每CDF表约 800 字节（适合缓存）

我们将设计**跨CDF表频率去重的压缩magic表**。

#### 2.2.3 对解码的影响

类似优化可应用于 `Rans64DecAdvance` 中的乘法：
```cpp
x = (uint64_t)freq * (x >> scale_bits) + (x & mask) - start;
```
该操作已使用32位乘法（`freq ≤ 2^16`），GPU原生支持，开销较小。主要收益在编码侧。

---

### 2.3 O3：持久协作内核 + 共享内存CDF缓存

**核心思想**：用单次持久内核启动替代多内核流水线，将CDF表缓存于共享内存中，线程块协作处理。

#### 2.3.1 内核架构

```
┌──────────────────────────────────────────────────────────┐
│ 持久协作内核（一次启动，处理完所有工作后退出）            │
│                                                           │
│ __shared__ int32_t  cdf_cache[M][Lmax];     // M≤64个CDF │
│ __shared__ uint64_t magic_cache[M][MAX_FREQ]; // 快速除法 │
│ __shared__ uint32_t size_reduction[BLOCK_SIZE]; // 归约  │
│                                                           │
│ while (work_remaining):  // 原子计数器获取工作            │
│   ┌────────────────────────────────────────────────┐     │
│   │ 阶段1: 协作加载CDF                              │     │
│   │   每个warp加载一行CDF至共享内存                  │     │
│   │   Warp 0加载cdf[0], Warp 1加载cdf[1], ...      │     │
│   │   __syncwarp() 确保加载完成                     │     │
│   ├────────────────────────────────────────────────┤     │
│   │ 阶段2: 逐warp执行WarpANS编码 (见§2.1)           │     │
│   │   每个warp处理一个chunk                         │     │
│   │   CDF查表命中共享内存（~20周期 vs ~300周期）     │     │
│   ├────────────────────────────────────────────────┤     │
│   │ 阶段3: 协作大小归约                              │     │
│   │   Block级归约计算sizes[]前缀和                  │     │
│   │   无需CUB内核启动                               │     │
│   ├────────────────────────────────────────────────┤     │
│   │ 阶段4: 协作打包                                  │     │
│   │   Block级前缀和计算偏移量                       │     │
│   │   协作拷贝至紧凑输出缓冲区                       │     │
│   └────────────────────────────────────────────────┘     │
│   __syncthreads();  barriers between phases              │
└──────────────────────────────────────────────────────────┘
```

#### 2.3.2 共享内存CDF布局优化

为最小化 SM90 上的 bank conflict（32 banks, 4 bytes each）：
```
// 交错布局，确保warp各lane访问无冲突
// Lane i 访问 cdf[ch][i]
// 不带填充: bank = (ch * Lmax + i) % 32
// 带填充:   bank = (ch * (Lmax + PAD) + i) % 32

__shared__ int32_t cdf_cache[M][Lmax + 1]; // PAD=1 大幅减少冲突
```

#### 2.3.3 收益分析

| 优化点 | 效果 |
|---|---|
| 消除7次内核启动 | 每次压缩调用减少约50-100μs延迟 |
| 消除临时全局内存arena | 减少约2倍内存流量 |
| 共享内存CDF命中 | ~20周期 vs ~200-800周期（全局内存） |
| 协作前缀和 | 避免CUB库开销和流同步 |
| 协作打包 | 避免额外内存分配和释放 |

---

### 2.4 O4：面向多切片模型的CUDA图特化

**核心思想**：对于TCM/DCAE等多切片模型（5个slice），将整个多切片编解码过程录制为CUDA Graph，摊销内核启动开销。

#### 2.4.1 Graph设计

```
CUDA Graph（每(B, C, H, W)配置录制一次）：
  ┌──────────────────┐
  │ Slice 0 WarpANS编码│──┐
  ├──────────────────┤  │
  │ Slice 1 WarpANS编码│──┤  各slice若独立可overlap
  ├──────────────────┤  │
  │ Slice 2 WarpANS编码│──┤
  ├──────────────────┤  │
  │ Slice 3 WarpANS编码│──┤
  ├──────────────────┤  │
  │ Slice 4 WarpANS编码│──┘
  ├──────────────────┤
  │ GPU Direct 合并   │  ← 原地拼接各slice输出
  └──────────────────┘
```

#### 2.4.2 端到端Graph

对于配合TRT推理的完整流水线：
```
[TRT ga] → [TRT ha] → [z编码] → [TRT hs] → 
  [WarpANS Slice 0] → [LRP 0] →
  [WarpANS Slice 1] → [LRP 1] →
  ...
  [WarpANS Slice 4] → [LRP 4] → [TRT gs]
```

录制为单个CUDA Graph后，**消除全部内核启动和CPU-GPU同步开销**，将端到端延迟降至最低。

---

### 2.5 O5：异步多流重叠管线（可选/扩展目标）

**核心思想**：通过多CUDA流将编码和打包阶段流水线化。

```
流0: [enc chunk 0-7]  [enc chunk 8-15]  [enc chunk 16-23] ...
流1:                    [pack chunk 0-7] [pack chunk 8-15]  ...
流2: [scan batch 0]    [scan batch 1]    [scan batch 2]     ...
```

将计算密集（编码）与访存密集（打包）阶段重叠，隐藏延迟。

---

## 3. 理论贡献

### 3.1 吞吐量-压缩率Pareto最优性

- **定理1（刷新开销下界）**：任何将N个符号划分为K个独立编码分区的GPU ANS算法，必须产生至少 `K × 8` 字节的状态刷新开销

- **定理2（WarpANS开销界）**：WarpANS使用W个warp lane时，刷新开销为 `K × W × 8` 字节。在总并行度 `L = K × W` 约束下，使开销最小化的最优K-W分解满足 `W = min(32, argmin(·))` ，并给出闭式解

- **定理3（加速比界）**：WarpANS相比单线程每chunk的加速比为 `S = W / (1 + αW/N)`，其中α为warp shuffle归并每字成本，N为每chunk符号数。当 `N >> αW` 时，加速比趋近于理想的 `W`

### 3.2 CDF缓存最优策略

- **驱逐策略**：基于CDF访问频率（由符号分布导出）的LRU近似
- **缺失率上界**：对于M个CDF表（每个Lmax entries），在共享内存容量为S时，缺失率上界为 `O(M·Lmax / S)`

---

## 4. 评估方案

### 4.1 硬件平台

| 平台 | GPU | 计算能力 | 用途 |
|---|---|---|---|
| H100 | NVIDIA H100 80GB | SM90 | 主要评估平台 |
| A100 | NVIDIA A100 80GB | SM80 | 通用性验证 |
| RTX 4090 | ADA 24GB | SM89 | 消费级GPU基线 |

### 4.2 基线方法

| 基线 | 描述 | 代码来源 |
|---|---|---|
| **CAIEC (VLDB'26)** | 当前chunk并行GPU rANS | 本仓库 |
| **CPU rANS (ryg)** | Fabian Giesen的优化标量实现 | `third_party/ryg_rans/` |
| **CPU SSE rANS** | SSE4.1 向量化实现（4路SIMD） | `rans_word_sse41.h` |
| **nvCOMP Bitcomp** | NVIDIA官方bitcomp算法 | `gpu_codec/compressors/` |
| **cuSZ-Hi** | SOTA GPU科学压缩器 | 外部引用 |
| **cuZFP** | GPU ZFP压缩器 | 外部引用 |

### 4.3 评估数据集

| 数据集 | 领域 | 维度 | 大小范围 |
|---|---|---|---|
| NYX | 宇宙学模拟 | 512×512×512 | ~512 MB |
| CESM | 气候模拟 | 1800×3600 | ~25 MB |
| Hurricane | 气象模拟 | 500×500×500 | ~500 MB |
| COVID | 医学影像 | 512×512 | ~0.25 MB |
| STEM | 显微成像 | 多种 | 10-100 MB |
| Tomobank | X射线断层 | 多种 | 50-500 MB |

### 4.4 评价指标

| 指标 | 定义 | 优先级 |
|---|---|---|
| **编码吞吐量** | GB/s（原始输入 / 编码时间） | 首要 |
| **解码吞吐量** | GB/s（原始输入 / 解码时间） | 首要 |
| **压缩率** | 原始大小 / 压缩后大小 | 首要 |
| **率失真** | PSNR vs. bpp | 次要 |
| **SM占用率** | 实际占用率 / 理论最大值 | 分析 |
| **L1/L2缓存命中率** | 通过Nsight Compute采集 | 分析 |
| **HBM带宽利用率** | 占峰值带宽百分比 | 分析 |
| **SM利用率** | 活跃周期比例 | 分析 |

### 4.5 消融实验设计

| 实验编号 | 启用的优化 | 目的 |
|---|---|---|
| E0（基线） | 原始CAIEC | 建立基线 |
| E1 | + 除法强度削弱（O2） | 隔离指令级收益 |
| E2 | + 共享内存CDF缓存（O3部分） | 隔离存储层次收益 |
| E3 | + 持久内核（O3完整） | 隔离内核启动开销 |
| E4 | + WarpANS（O1） | 隔离warp级并行收益 |
| E5 | + CUDA Graph（O4） | 隔离系统级收益 |
| E6（完整） | O1+O2+O3+O4 | 综合效果 |

### 4.6 参数扫描

- `Pch ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}`
- `Warp lanes ∈ {1, 2, 4, 8, 16, 32}`（WarpANS模式）
- `CDF 缓存大小 ∈ {0, 16, 32, 64, 128}`（共享内存分配）
- `Batch size ∈ {1, 2, 4, 8, 16}`

---

## 5. 实施计划

### 5.1 第一阶段：微基准测试与Profiling（第1-2周）

**目标**：为所有五个瓶颈建立精确的定量基线。

**任务清单**：
- [ ] 在H100上使用 Nsight Compute 对当前实现进行profiling
- [ ] 精确测量 `Rans64EncPut` 的每条指令周期数（除法 vs 乘法对比）
- [ ] 通过 `lts__t_sectors_srcunit_*` 等硬件计数器测量CDF缓存缺失率
- [ ] 通过CUPTI测量内核启动开销
- [ ] 分析SM占用率和warp停顿原因（stall reasons）
- [ ] 生成rANS编解码的Roofline模型

**交付物**：详细profiling报告，包含按瓶颈分解的周期占比。

---

### 5.2 第二阶段：除法强度削弱（第3-4周）

**目标**：实现并验证O2（预计算模逆）。

**任务清单**：
- [ ] 对 `freq ∈ [1, 65536]`, `precision=16` 推导magic数公式
- [ ] 实现 `Rans64EncPutFast`（使用 `__umul64hi`）
- [ ] 与参考rANS实现进行 bit-exact 正确性验证
- [ ] 单线程隔离场景下profiling加速比
- [ ] 实现频率去重压缩magic表

**交付物**：`rans64_gpu_v2.cuh`，含经过验证的快速除法原语。

---

### 5.3 第三阶段：Warp级并行rANS（第5-8周）

**目标**：设计、实现并验证O1（WarpANS）。

**任务清单**：
- [ ] 设计warp交错/解交错合并算法
- [ ] 在 `ans_gpu_kernel_warp.cu` 中原型单warp编码器
- [ ] 原型单warp解码器
- [ ] 正确性测试：随机符号下 `decode(encode(x)) == x` 的bit-exact验证
- [ ] 压缩率分析：与chunk并行基线的开销对比
- [ ] 吞吐量分析：不同Pch和warp配置的参数扫描
- [ ] 使用 `__shfl_xor_sync` 实现高效warp shuffle合并
- [ ] 处理变长输出（各lane产生的word数可能不同）
- [ ] 性能调优：占用率优化，寄存器压力降低

**交付物**：`warp_ans.cuh` + `warp_ans_kernel.cu`，含经过验证的编解码器。

---

### 5.4 第四阶段：持久协作内核（第9-11周）

**目标**：实现O3（持久内核 + 共享内存CDF缓存）。

**任务清单**：
- [ ] 设计持久内核grid launch配置
- [ ] 实现协作式CDF加载至共享内存
- [ ] 实现协作式size归约（block级，不依赖CUB）
- [ ] 实现协作式前缀和计算偏移量
- [ ] 实现协作式紧凑打包输出
- [ ] 将WarpANS集成至持久内核
- [ ] 调优共享内存CDF布局以最小化bank conflict
- [ ] 处理边界情况：CDF表超出共享内存容量时的回退策略

**交付物**：`persistent_ans_kernel.cu`，集成WarpANS和CDF缓存。

---

### 5.5 第五阶段：CUDA Graph集成（第12-13周）

**目标**：实现O4（面向多切片模型的CUDA Graph）。

**任务清单**：
- [ ] 将单slice编码录制为CUDA Graph节点
- [ ] 将多slice编码录制为CUDA Graph
- [ ] 将端到端流水线（TRT推理 + ANS + LRP）录制为CUDA Graph
- [ ] 通过graph update或shape-tolerant capture处理动态shape
- [ ] 测量graph launch开销 vs 传统内核启动开销

**交付物**：`graph_ans_runner.py` + C++ graph capture辅助代码。

---

### 5.6 第六阶段：全系统集成与基准测试（第14-16周）

**目标**：将所有优化集成至CAIEC框架并运行完整基准测试。

**任务清单**：
- [ ] 将优化后的内核集成至 `compressai/cpp_exts/rans_gpu/`
- [ ] 更新新内核API的Python绑定
- [ ] 更新 `GpuPackedEntropyCodec` 和引擎实现
- [ ] 在所有数据集和基线上运行完整基准测试套件
- [ ] 完成所有配置的消融实验
- [ ] 生成图表和表格

**交付物**：经过基准测试的集成系统。

---

### 5.7 第七阶段：论文撰写（第17-20周）

**任务清单**：
- [ ] 撰写摘要和引言
- [ ] 撰写背景（rANS、GPU架构、相关工作）
- [ ] 撰写算法描述（含伪代码）
- [ ] 撰写理论分析（开销下界、加速比上界）
- [ ] 生成出版级图表
- [ ] 撰写评估章节
- [ ] 内部审阅和修改
- [ ] Artifact Evaluation准备

---

## 6. 时间线总览

```
第1-2周   ████ Profiling与瓶颈分析
第3-4周   ████ 除法强度削弱（O2）
第5-8周   ████████ Warp级并行ANS（O1）← 关键路径
第9-11周  ██████ 持久协作内核（O3）
第12-13周 ████ CUDA Graph（O4）
第14-16周 ██████ 全系统集成与基准测试
第17-20周 ████████ 论文撰写
         ─────────────────────────────────────────
         总计：20周（约5个月）
```

### 6.1 关键里程碑

| 周次 | 里程碑 | 检查点 |
|---|---|---|
| 2 | Profiling报告完成 | 瓶颈周期分解 |
| 4 | 快速除法验证通过 | bit-exact正确，加速比确认 |
| 8 | WarpANS编解码器工作 | 正确性+压缩率验证通过 |
| 11 | 持久内核集成 | 全流水线单内核运行 |
| 13 | CUDA Graph工作 | 端到端graph录制成功 |
| 16 | 完整基准测试完成 | 所有消融数据收集完毕 |
| 20 | 论文初稿完成 | 内部审阅通过 |

**目标投稿**：PPoPP 2027（约2026年11月截止）

---

## 7. 风险分析与应对策略

| 风险 | 可能性 | 影响 | 应对策略 |
|---|---|---|---|
| WarpANS刷新开销过高 | 中 | 高 | 混合模式：大chunk用WarpANS，小chunk用chunk并行 |
| 共享内存无法容纳所有CDF | 中 | 中 | 多趟策略；将低频CDF淘汰至全局内存 |
| 强度削弱后仍有性能瓶颈 | 低 | 高 | 探索32位rANS变体（适用于小字母表场景） |
| 32-lane warp占用率过低 | 中 | 中 | 寄存器优化；探索warp-group（4个连续warp）协作模式 |
| CUDA Graph与动态TRT shape不兼容 | 中 | 低 | 使用graph update；shape变化时回退至非graph路径 |
| 与竞品工作重叠 | 中 | 中 | 持续监控文献；通过warp级贡献差异化 |

### 7.1 回退策略

如果WarpANS的刷新开销被证明过高，有两种回退选择：

**回退A：N流交错（N < 32）**
在warp内使用更少的交错流（如4或8条），减少刷新开销同时保持部分warp内并行度。

**回退B：推测性解码**
在解码端使用warp级推测：32个lane同时推测下一个符号，一个lane正确（通过warp vote确定），通过shuffle收集结果。增加工作量但消除解码的顺序依赖。

---

## 8. 相关工作

### 8.1 ANS基础
- Duda (2009)：原始rANS/rABS提出
- Giesen (2014)：`ryg_rans`——优化标量CPU实现
- Giesen (2015)：SSE4.1向量化rANS（`rans_word_sse41.h`），4路SIMD

### 8.2 GPU熵编码
- **CAIEC** (Huang et al., VLDB 2026)：Chunk并行GPU rANS用于AI科学数据压缩——**我们的直接基线**
- Funasaka et al. (2018)：GPU Huffman编码，使用并行前缀和
- Yamamoto et al. (2019)：GPU区间编码，使用chunk级并行

### 8.3 GPU压缩系统
- **cuSZ** (Tian et al., 2020)：GPU加速的SZ压缩器
- **cuZFP** (2021)：GPU ZFP编解码器
- **nvCOMP** (NVIDIA, 2022)：GPU压缩库（bitcomp, LZ4, Snappy等）

### 8.4 GPU并行原语
- **CUB**：协作原语（scan, reduce, sort）
- **Cooperative Groups** (NVIDIA, CUDA 9+)：灵活线程分组
- **CUDA Graphs** (NVIDIA, CUDA 10+)：内核图录制，降低延迟
- **Warp Shuffle** (NVIDIA, CUDA 3.0+)：warp内寄存器交换

### 8.5 本工作的差异化

据我们所知，**尚无先前的工���**同时做到：
1. 对rANS编解码提出warp级交错并行算法
2. 系统分析GPU ANS的吞吐量-压缩率理论Pareto前沿
3. 设计面向熵编码的持久协作GPU内核
4. 将CUDA Graph应用于端到端AI压缩流水线

---

## 9. 预期贡献

1. **算法贡献**：WarpANS——首个warp级并行rANS算法，打破chunk内串行瓶颈，同时保持最优压缩效率
2. **理论贡献**：GPU ANS吞吐量-压缩率Pareto前沿的形式化刻画，含开销下界和加速比上界
3. **系统贡献**：面向熵编码的持久协作GPU内核设计，通过共享内存CDF缓存消除内核启动开销并减少全局内存访问
4. **实验贡献**：在6+科学数据集、3种GPU架构上的全面评估，演示相比SOTA GPU ANS实现的3-5倍加速
5. **开源贡献**：完整实现集成至CAIEC框架，提供可复现的基准测试

---

## 10. 附录：关键代码位置

| 组件 | 文件 | 行数 |
|---|---|---|
| GPU ANS编码内核 | `compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu` | 1-195 |
| GPU ANS解码内核 | `compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu` | 295-394 |
| 编码公开API | `compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu` | 399-604 |
| 解码公开API | `compressai/cpp_exts/rans_gpu/ans_gpu_kernel.cu` | 606-704 |
| Python绑定 | `compressai/cpp_exts/rans_gpu/ans_gpu.cpp` | 1-28 |
| rANS64编码原语 | `compressai/cpp_exts/rans_gpu/rans64_gpu.cuh` | 1-55 |
| rANS64解码原语 | `compressai/cpp_exts/rans_gpu/rans64dec_gpu.cuh` | 1-48 |
| Python TightANS封装 | `compressai/entropy_models/entropy_models.py` | 83-161 |
| GPU ANS在compress()中 | `compressai/entropy_models/entropy_models.py` | 350-426 |
| GPU ANS在decompress()中 | `compressai/entropy_models/entropy_models.py` | 428-455 |
| GpuPackedEntropyCodec | `compressai/runtime/codecs/compress_packed_gpu.py` | 21-121 |
| TCM引擎 | `compressai/runtime/engines/tcm_engine.py` | 1-710 |
| DCAE引擎 | `compressai/runtime/engines/dcae_engine.py` | 1-1053 |
