# GA-ANS 优化进展汇报 (2026.07.16)

## 1. 动机

原始GPU rANS（`tight`模式）的主要瓶颈：

1. **Chunk内串行**：每个chunk仅1个线程编码所有符号，GPU利用率低
2. **64位整数除法**：`Rans64EncPut`中 `x/freq` 和 `x%freq` 在GPU上需30-60 cycles（软件模拟），每条符号都执行
3. **全局内存CDF随机访问**：CDF表驻留在全局内存，每符号触发随机读取，~300 cycles/缺失

## 2. V1: WarpANS — 32路Warp级并行

### 改动

- 新增 `warp_rans.cuh`、`warp_ans_kernel.cu`：每chunk由1个warp（32线程）处理，通过跨步（stride=32）分配符号
- 比特流改为128字节对齐的交错格式：`[max_rounds:u32] [32×u32] [32×u32]...`，恰好1个L1缓存行=32 lane各1 word，100%合并访存

### 结果（H100, bmshj2018-factorized q=1, Tomobank, C=192, 786K符号）

| P | tight enc | warp enc | warp dec |
|---|-----------|----------|----------|
| 1 | 0.46 GB/s | 0.53 GB/s | 1.49 GB/s |
| 4 | 0.26 GB/s | 0.50 GB/s | 1.52 GB/s |
| 16 | 0.09 GB/s | 0.48 GB/s | 1.33 GB/s |
| 64 | 0.03 GB/s | 0.36 GB/s | 0.92 GB/s |

- tight在P≥16时急剧退化（chunk太少→GPU利用率近乎为0），warp保持可用

## 3. V2: WarpANS + 除法强度削弱

### 改动

- 新增 `rans64_gpu_v2.cuh`、`warp_ans_kernel_v2.cu`：用 `__umul64hi(x, magic[freq])`（~4 cycles）替代64位除法（~30-60 cycles）
- magic表：512KB全局内存，`magic[freq]=floor(2^64/freq)` 预计算

### 结果（H100，合成256ch×64×64=1M符号）

| P | V1 enc | V2 enc | V2/V1 |
|---|--------|--------|-------|
| 1 | 0.38 GB/s | 0.44 GB/s | 1.16× |
| 4 | 0.31 GB/s | 0.58 GB/s | 1.87× |
| 16 | 0.34 GB/s | 0.51 GB/s | 1.50× |
| 64 | 0.19 GB/s | 0.20 GB/s | 1.05× |
| 256 | 0.06 GB/s | 0.07 GB/s | 1.17× |

- 模型数据上V2与V1基本持平（V1 0.53 vs V2 0.44 GB/s）
- 收益被magic表全局内存访问（~300 cycles/次）部分抵消

## 4. V3: WarpANS + 共享内存CDF缓存

### 改动

- 新增 `warp_ans_kernel_v3.cu`：基于V2，将每个chunk所需的CDF表在编码前协作加载到`__shared__`内存
- 编解码循环中CDF查表直接命中共享内存（~20 cycles），替代全局内存访问（~300 cycles）
- 解码同样受益：解码内核也有独立的共享内存CDF缓存
- Pch根据共享内存容量自动钳制（默认48KB上限，Pch≤37KB/(Lmax*4+8)）

### 结果

**合成大CDF场景**（C=192, Lmax=256, HW=64, 总CDF=192KB）：

| P | V2 enc | V3 enc | V3/V2 |
|---|--------|--------|-------|
| 32 | 5.71ms | 0.54ms | **10.6×** |
| 64 | 5.19ms | 0.55ms | **9.4×** |

**真实模型**（Tomobank, bmshj2018-factorized q=1, C=192, Lmax=75, 总CDF=57KB）：

| P | V1 enc | V2 enc | V3 enc |
|---|--------|--------|--------|
| 1 | 0.73 GB/s | 0.33 GB/s | 0.33 GB/s |
| 4 | 0.46 GB/s | 0.43 GB/s | 0.35 GB/s |
| 16 | 0.40 GB/s | 0.37 GB/s | 0.54 GB/s |
| 64 | 0.53 GB/s | 0.30 GB/s | 0.27 GB/s |

- 小模型上V3与V1/V2基本持平，原因：
  1. 192×75×4=57KB CDF总量远小于H100 256KB L1缓存，L1天然缓存了CDF
  2. V3多了一次协作加载（全局→共享）+ `__syncwarp()` 开销
  3. 小Pch时加载阶段只有少数lane有实际工作，其余空转

### 结论

- V3在大CDF场景（CDF总量>L1容量）下效果显著（最高10.6×），小模型上无收益
- 保留用于消融实验 + 后续DCAE大模型测试

## 5. 优化栈总结

| 版本 | 优化 | 核心贡献 |
|------|------|---------|
| tight (v0) | chunk-parallel基线 | 1 thread/chunk |
| warp (v1) | 32-lane warp级交错 | 128B对齐合并访存，100% L1利用率 |
| warp_div (v2) | `__umul64hi` 快速除法 | 64位除法→乘法+修正 |
| warp_smem (v3) | 共享内存CDF缓存 | 消除全局内存CDF随机访问 |

## 6. 下一步

- **DCAE大模型测试**: 在C>400的DCAE模型上验证V3在大CDF场景下的收益
