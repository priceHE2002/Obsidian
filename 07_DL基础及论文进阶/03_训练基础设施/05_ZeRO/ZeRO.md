---
tags:
  - 论文
  - 训练基础设施
  - 分布式训练
  - 显存优化
  - DeepSpeed
created: 2026-06-30
paper_title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
paper_authors: "Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He"
paper_year: 2020
paper_venue: "SC 2020"
paper_citations: "~3,000+"
paper_url: "https://arxiv.org/abs/1910.02054"
github: "https://github.com/microsoft/DeepSpeed"
---

# ZeRO

**ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**
*Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He | Microsoft Research | SC 2020 | arXiv: 1910.02054*

> 将优化器状态、梯度和参数以三阶段渐进式分片到各数据并行（DP）进程上，使显存消耗从 $O(4\Psi + 12\Psi)$ 降至 $O(4\Psi)$，且不增加任何通信量（只改变通信模式）。万亿参数模型的直接使能技术——DeepSpeed 框架的学术基石。

---

## 一、Background / Core Idea

### 1.1 问题：显存墙——数据并行的隐形成本

在大模型训练中，**显存的分配并不均匀**。以 Adam 优化器、FP16 混合精度训练为基准，训练时的显存消耗分为三部分：

$$\text{训练显存} = \underbrace{\Psi \cdot K}_{\text{模型参数}} + \underbrace{2\Psi \cdot 4}_{\text{优化器状态}} + \underbrace{\Psi}_{\text{梯度}} + \underbrace{\text{激活值}}_{\text{可优化}}$$

其中 $\Psi$ 为参数数量（以 FP16 为单位），$K$ 为数据类型的字节数（FP16=2, FP32=4）。

| 组件 | 数据类型 | 存储开销（以 $\Psi$ 为单位） | 占 7.5B 模型总显存 |
|:----|:--------|:-------------------------:|:-----------------:|
| FP16 模型参数 | FP16 | $2\Psi$ | ~15 GB |
| **FP32 优化器状态**（Adam momentum + variance） | FP32 | **$8\Psi$** | **~60 GB** |
| FP16 梯度 | FP16 | $2\Psi$ | ~15 GB |
| FP32 梯度副本 | FP32 | $4\Psi$ | ~30 GB |
| **总计** | — | **$16\Psi$** | **~120 GB** |

对于 GPT-3 175B 模型，$16\Psi = 16 \times 175 \text{GB} = 2.8\text{TB}$ 的显存需求——即使使用 80 张 A100 80GB 也无法容纳。

### 1.2 核心洞察：显存冗余是数据并行的固有缺陷

论文的根本性洞察是：**在标准数据并行中，所有 GPU 都存储了完全相同的优化器状态和梯度信息**。

$$\text{DP 中每个 GPU: } \underbrace{\Psi \cdot K}_{\text{参数}} + \underbrace{2\Psi \cdot 4}_{\text{优化器状态}} + \underbrace{\Psi}_{\text{梯度}}$$

因为数据并行的梯度同步（all-reduce）已经确保了所有 GPU 拥有相同的梯度值，所以**理论上每个设备只需要存储 $1/N_d$ 的优化器状态和梯度**——这正是 ZeRO 的核心思想。

### 1.3 ZeRO 三阶段（与 DP 对比）

| 阶段 | 分片对象 | 单设备显存（7.5B） | 显存减少比例 |
|:----|:--------|:-----------------:|:----------:|
| **ZeRO-1 (OS)** | 优化器状态 | $2\Psi + \frac{8\Psi}{N_d} + \Psi$ | 4× |
| **ZeRO-2 (OS+G)** | 优化器状态 + 梯度 | $2\Psi + \frac{8\Psi+\Psi}{N_d}$ | 8× |
| **ZeRO-3 (OS+G+P)** | 优化器状态 + 梯度 + 参数 | $\frac{2\Psi+8\Psi+\Psi}{N_d}$ = $\frac{16\Psi}{N_d}$ | **$N_d$ 倍** |

其中 $N_d$ 为数据并行度。当 $N_d=64$ 时，ZeRO-3 将每设备显存降为 **$16\Psi/64 = 0.25\Psi$**——GPT-3 175B 的 2.8TB 需求降至约 44GB/设备。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 ZeRO-1：优化器状态分片（Optimizer State Partitioning）

优化器状态是训练中最大的显存消耗者（80%）。ZeRO-1 将其按数据并行进程均匀分片：

$$\text{进程 } i \text{ 持有: } \text{Adam states}_i = \{m_t^{(j)}, v_t^{(j)} \mid j \in \text{partition}_i\}$$

**通信流程**（梯度计算完成后）：

1. 每个 DP 进程获得完整梯度（来自 all-reduce）
2. 各进程只更新自己分片内的优化器状态
3. 更新对应分片内的参数：$\theta_i^{(t+1)} = \text{Adam}(\theta_i^{(t)}, g_i)$
4. 下一轮前向开始时，各进程 broadcast 自己更新的参数分片给所有进程

**通信量分析**：

| 操作 | ZeRO-1 通信量 | 标准 DP 通信量 | 差量 |
|:----|:------------:|:-------------:|:----|
| 梯度同步 | $2\Psi$（all-reduce） | $2\Psi$（all-reduce） | 相同 |
| 参数 broadcast | $\Psi \cdot (N_d - 1)/N_d$ | 0（全参数持有） | **新增** |
| 总计 | $2\Psi + \frac{N_d-1}{N_d}\Psi$ | $2\Psi$ | 略增 |

**ZeRO-1 的通信量几乎与 DP 相同**（$\approx 3\Psi$ vs $2\Psi$），但显存减少了 4 倍。这是 ZeRO 优雅的核心所在——**它不降低通信量，而是改变通信模式**。

### 2.2 ZeRO-2：梯度分片（Gradient Partitioning）

ZeRO-2 在 ZeRO-1 之上进一步减少梯度的存储：

**标准 DP 梯度流**：每个进程存储完整梯度 $G \in \mathbb{R}^\Psi$，通过 all-reduce 同步。
**ZeRO-2 梯度流**：

1. 反向传播后，每个进程只保留自己分区内的梯度：$G_i \in \mathbb{R}^{\Psi/N_d}$
2. 使用 reduce-scatter 而非 all-reduce 来聚合梯度（reduce-scatter 在聚合后按分区分布）
3. 各进程利用本地梯度更新本地参数分片
4. 与 ZeRO-1 一致：更新后 broadcast 参数

**通信量分析**：reduce-scatter 的通信量与 all-reduce 相同（$2\Psi$），所以 ZeRO-2 的通信总量与 DP 完全相同（$2\Psi$）。

### 2.3 ZeRO-3：参数分片（Parameter Partitioning）

ZeRO-3 将模型参数也分片，是最激进的阶段：

$$\text{每个 DP 进程:  参数 } \frac{\Psi}{N_d} + \text{优化器状态 } \frac{8\Psi}{N_d} + \text{梯度 } \frac{\Psi}{N_d} = \frac{16\Psi}{N_d}$$

**前向通信**：在计算第 $l$ 层之前，从持有者获取完整的参数分片（all-gather）。
**反向通信**：在计算第 $l$ 层的梯度后，通过 reduce-scatter 聚合梯度到对应分区。

**每次 Transformer 层需要两次 all-gather + 一次 reduce-scatter**——通信量比 ZeRO-2 增大至 $3\Psi$（原始 DP 的 1.5 倍）。这是 ZeRO-3 以轻微通信增加换取最大显存节省的根本原因。

### 2.4 三阶段显存节省对比

| 模型大小 | 标准 DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|:--------|:------:|:------:|:------:|:------:|
| GPT-3 175B (FP16, Nd=64) | ~2.8TB | ~700GB | ~395GB | **~44GB** |
| 7.5B (FP16, Nd=8) | ~120GB | ~30GB | ~17GB | **~15GB** |
| 1.5B (FP16, Nd=4) | ~24GB | ~6GB | ~3.5GB | **~1.5GB** |

ZeRO-3 使 GPT-3 175B 的显存需求从 2.8TB 降至 44GB——正好在一块 A100 80GB 上运行（**跨 64 张 GPU 扩展**）。

---

## 三、Experiments and Key Findings

### 3.1 吞吐量比较

论文在 DGX-2 集群（16× V100 32GB）上对 10B-170B 参数的 GPT-2 模型进行了吞吐测试：

| 模型规模 | 单卡显存需求 | 标准 DP | ZeRO-2 | ZeRO-3 | ZeRO-3 的加速 |
|:--------|:----------:|:------:|:------:|:------:|:------------:|
| 10B | ~160GB | ❌ OOM | 45 TFlops/GPU | 40 TFlops/GPU | 可训练 |
| 30B | ~480GB | ❌ OOM | 42 TFlops/GPU | 38 TFlops/GPU | 可训练 |
| 100B | ~1.6TB | ❌ OOM | 35 TFlops/GPU | 32 TFlops/GPU | 可训练 |
| 170B | ~2.7TB | ❌ OOM | 28 TFlops/GPU | 25 TFlops/GPU | 可训练 |

**标准 DP 在超过 1.5B（单卡）时直接 OOM。** ZeRO 是唯一使 10B+ 参数训练在有限 GPU 上可行的方案。

### 3.2 ZeRO-2 与标准 DP 的吞吐对比（1.5B）

| 批量大小 | 标准 DP (s/iter) | ZeRO-2 (s/iter) | ZeRO-2 相对速度 |
|:-------:|:---------------:|:---------------:|:--------------:|
| 16 | 0.78 | **0.71** | **+9.8%** |
| 32 | 1.31 | **1.04** | **+25.4%** |
| 64 | 2.35 | **1.66** | **+41.6%** |

**反直觉的发现**：ZeRO-2 不仅节省显存，而且**比标准 DP 更快**。原因：reduce-scatter 操作比 all-reduce 的通信模式更高效，且显存的释放允许更大的 batch size。

### 3.3 ZeRO-3 与模型并行的对比（40B 模型）

| 配置 | GPU 数 | 速度 (s/iter) | 可训模型 |
|:----|:-----:|:------------:|:--------:|
| Model Parallelism (TP-16) | 16 | 1.82 | ~10B |
| ZeRO-3 (DP-16) | 16 | **1.21** | **~40B** |
| ZeRO-3 (DP-32) | 32 | 0.70 | ~40B |
| ZeRO-3 (DP-64) | 64 | 0.38 | ~40B |

**ZeRO-3 在 40B 模型上比 TP-16 快 1.5 倍**，且扩展到 64 GPU 时线性加速（0.38s 即 4.8× 相对于 1.82s）。

### 3.4 带宽敏感性

| 网络带宽 | 参数 all-gather 时间 | ZeRO-3 时间 | 效率对比 DP |
|:--------|:-------------------:|:----------:|:----------:|
| NVLink (600 GB/s) | 0.03 ms | 1.21 s | 99.5% |
| InfiniBand (200 Gbps) | 0.82 ms | 1.24 s | 97.3% |
| 以太网 (25 Gbps) | 6.60 ms | 1.52 s | 80.1% |

**ZeRO-3 的性能效率对网络带宽高度敏感**——在理想 NVLink 下几乎无开销，但在以太网下损失约 20%。

---

## 四、Limitations and Challenges

1. **ZeRO-3 的通信开销随模型增大不成比例**：大型模型中参数 all-gather 的通信量随 $N_d$ 增大而增长，在 1000+ GPU 集群上通信占比如不忽略
2. **仅解决数据并行的显存问题**：ZeRO 是 DP 维度的优化，当模型大到一个 DP 进程也无法容纳 $\Psi/N_d$ 参数时，ZeRO-3 不够——需要结合 [[Megatron-LM]] 的 TP
3. **训练-推理不对称**：ZeRO-3 在推理时需要 gather 完整参数，导致推理效率下降。需要通过参数 offload 或模型合并来弥补
4. **网格搜索空间扩大**：结合 TP 后，ZeRO-3 的通信模式与 TP 的通信模式可能互相干扰（参数 all-gather + 张量并行的 all-reduce 共同占用带宽）
5. **梯度累积的挑战**：当 micro-batch 数量很大时，ZeRO-3 的 reduce-scatter 需要等待所有 micro-batch 的反向完成——引入同步屏障
6. **显存节省在极大规模时非线性**：当 $N_d$ 非常大（>256）时，$\Psi/N_d$ 的显存节省收益递减（因为激活值成为新的瓶颈）

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 ZeRO 的关系 |
|:--------|:----:|:--------------|
| [[Mixed Precision Training]] (Micikevicius et al.) | 2018 | 混合精度训练是 ZeRO 优化的前提——FP16 参数 + FP32 优化器状态 |
| [[Megatron-LM]] (Shoeybi et al.) | 2019 | ZeRO（DP 内分片）+ Megatron-TP（节点内并行）互补——DeepSpeed-Megatron 融合框架 |
| [[3D Parallelism]] (Narayanan et al.) | 2021 | ZeRO-3 作为 DP 维度的显存分片，融入 PTD-P 三维框架 |
| **ZeRO-Offload** (Ren et al.) | 2021 | 将优化器状态和梯度卸载到 CPU 内存，ZeRO 的显存扩展 |
| **ZeRO-Infinity** (Rasley et al.) | 2023 | ZeRO-3 + NVMe offload，百亿亿次参数模型的训练 |
| **FSDP** (PyTorch) | 2022 | ZeRO-3 的 PyTorch 原生实现，统一接口 `torch.distributed.fsdp` |
| **DeepSpeed ZeRO++** (Rajbhandari et al.) | 2023 | 量化通信 + 分层分片，进一步减少 ZeRO-3 的通信开销 |

**影响评估**：ZeRO 是**分布式训练领域引用量最高的工作之一**。它将数据并行的显存优雅降到 $O(1/N_d)$，使数十亿到数万亿参数模型的训练变得实际。FSDP（PyTorch 原生 ZeRO-3 实现）使 ZeRO 成为 HuggingFace Transformers、Meta LLaMA 训练的事实标准。DeepSpeed ZeRO 是 Microsoft 对大规模分布式训练最关键的技术贡献。

---

## 六、Implications for You / Hardware Compatibility

### ZeRO 各阶段在不同 GPU 上的适用性

| GPU 配置 | ZeRO-1 可训模型 | ZeRO-2 可训模型 | ZeRO-3 可训模型 | ZeRO-3 + Offload |
|:---------|:--------------:|:--------------:|:--------------:|:----------------:|
| RTX 3060 (12GB) | ~1.5B | ~2.5B | ~7B (bs=1) | ~13B (慢) |
| RTX 4060 (12GB) | ~1.5B | ~2.5B | ~7B (bs=1) | ~13B (慢) |
| RTX 3090 (24GB) | ~5B | ~7B | ~13B (bs=1) | ~30B (慢) |
| RTX 4090 (24GB) | ~5B | ~7B | ~13B (bs=1+检查点) | ~30B (慢) |
| A100 80GB (单卡) | ~15B | ~30B | ~70B (bs=1+检查点) | ~175B |
| 8× A100 (80GB) | ~120B | ~240B | ~530B (bs=1+检查点) | ~1T |

### 对大规模训练的指导

- **ZeRO 各阶段的选择原则**：优先用 ZeRO-2（效率最高），若显存不够升 ZeRO-3，若仍不够加 CPU offload（ZeRO-Offload）
- **ZeRO-3 + gradient checkpointing 是最低配置**：2 个正交优化叠加在消费级 GPU 上实现 7B 模型训练
- **ZeRO-3 与 TP 的配合**：跨节点使用 ZeRO-3（DP 维度），节点内使用 TP（[[Megatron-LM]]），典型配置为 ZeRO-3(64) × TP(8)
- **通信开销与模型大小的关系**：小模型（<1B）ZeRO-3 不如 ZeRO-2 高效，因为参数 all-gather 的通信时间相对前向计算不可忽略
- **FP32 权重精度不受影响**：ZeRO 的显存节省来自分片而不是量化，数学上与标准 DP 完全等价

### 硬件兼容性总结

- ✅ ZeRO-2：全平台兼容，所有使用 all-reduce 的分布式训练
- ✅ ZeRO-3：NVLink/InfiniBand 集群（最优）；单机多卡（良好）
- ⚠️ ZeRO-3 + CPU Offload：消费级 GPU 可运行，但训练速度慢 3-5 倍
- ❌ ZeRO-3 在以太网互联集群：通信开销使训练效率 < 50%
- ⚠️ ZeRO-3 + QLoRA：两者正交但需要注意参数 all-gather 与量化参数的交互

## PDF

[[ZeRO 原文.pdf]]
