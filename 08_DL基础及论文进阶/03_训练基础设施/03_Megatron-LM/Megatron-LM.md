---
tags:
  - 论文
  - 训练基础设施
  - 分布式训练
  - 张量并行
  - 模型并行
created: 2026-06-30
paper_title: "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
paper_authors: "Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro"
paper_year: 2019
paper_venue: "arXiv preprint"
paper_citations: "~2,500+"
paper_url: "https://arxiv.org/abs/1909.08053"
github: "https://github.com/NVIDIA/Megatron-LM"
---

# Megatron-LM

**Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
*Mohammad Shoeybi, Mostofa Patwary, Raul Puri et al. | NVIDIA | 2019 | arXiv: 1909.08053*

> 从 Transformer 的数学结构出发，将每个多头注意力的头（head）和 FFN 的列/行切分到不同 GPU 上，在 32 张 V100 上实现 8.3B 参数的端到端训练。张量并行（Tensor Parallelism, TP）的奠基工作，[[3D Parallelism]] 最关键的技术组件。

---

## 一、Background / Core Idea

### 1.1 问题：数据并行不能无限扩展

当模型参数超过单张 GPU 显存时，数据并行（Data Parallelism, DP）不再可行：

- **DP 的限制**：每个 GPU 持有完整模型副本。对于 8.3B 参数的模型，纯 FP16 的模型副本为 $8.3 \times 10^9 \times 2 = 16.6$ GB。加上梯度（16.6 GB）和优化器状态（Adam: $2 \times 2 = 4$ 倍参数 = 66.4 GB），单卡需求远超 V100 32GB
- **Pipeline 并行的前身**：当时的模型并行（如 Mesh-TensorFlow）将层切分到不同设备，但存在严重的计算空闲（bubble）
- **通信成本**：传统模型并行的通信量级等于所有训练批次中每个样本的完整隐藏状态，在 $n$ 设备间造成严重带宽竞争

### 1.2 核心洞察：Transformer 的结构可分解性

论文的核心洞察在于：**Transformer 层内天然存在可并行的计算结构**：

1. **Multi-Head Attention（MHA）**：每个 head 的计算是独立的 $Q_i, K_i, V_i$ 投影和后续的 attention 计算
2. **Feed-Forward Network（FFN）**：使用 GeLU 激活的两层 MLP，其第一层的列可以被独立计算

$$\text{MHA: } \text{Attention}(QKV) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

$$\text{FFN: } \text{FFN}(x) = \text{GeLU}(xW_1)W_2$$

如果我们将 $W_Q, W_K, W_V$ 沿列分割、将 $W_O$ 沿行分割，且 $W_1$ 沿列分割、$W_2$ 沿行分割——则**中间每个单独的输出计算不需要跨设备通信**。

### 1.3 与其他并行策略的对比

| 策略 | 切分维度 | 通信粒度 | 通信量 | 设备间的依赖 |
|:----|:--------|:--------|:-----|:-----------|
| 数据并行 (DP) | 样本 | 梯度 all-reduce | $O(\frac{N}{P})$ 参数 | 梯度同步 |
| 流水线并行 (PP) | 层 | 激活值 p2p | $O(h \cdot s)$ 每 batch | **空闲时间** |
| **张量并行 (TP)** | 层内参数 | **all-reduce 跨 GPU** | **$O(4h)$ 每层** | **极小** |

---

## 二、Method / Architecture / Technical Contribution

### 2.1 Transformer 层的张量切分方案

#### Self-Attention 的列切分

论文将注意力层的 QKV 投影矩阵沿列（输出维度）均匀切分到 $t$ 个设备：

$$Q_i = XW_{qi}, \quad K_i = XW_{ki}, \quad V_i = XW_{vi}, \quad i \in [1, t]$$

每个设备 $i$ 持有 $\frac{h}{t}$ 个 attention head，独立计算局部 attention 输出。设备间通过 all-gather 收集头输出后拼合。

**输出投影的逆切分**：$W_O$ 沿行切分，每设备持有一半的列。计算局部输出后，通过 all-reduce 求和得到最终输出。

#### FFN 的列行双切分

GeLU FFN 的切分更为精巧。设 $W_1 \in \mathbb{R}^{h \times 4h}$，$W_2 \in \mathbb{R}^{4h \times h}$：

1. **$W_1$ 列切分**：将 $4h$ 的中间维度均匀分到 $t$ 设备，每设备持有 $W_{1i} \in \mathbb{R}^{h \times 4h/t}$
2. **$W_2$ 行切分**：输出投影 $W_2$ 沿行切分，每设备持有 $W_{2i} \in \mathbb{R}^{4h/t \times h}$

前向计算中：
$$\text{Step 1: } H_i = \text{GeLU}(XW_{1i}) \quad \text{(每设备独立计算)}$$
$$\text{Step 2: } Y = \sum_{i=1}^t H_iW_{2i} \quad \text{(all-reduce 合并)}$$

这种设计确保**中间激活值 $H_i$ 始终在设备本地**，无额外通信。

### 2.2 通信模式分析

论文设计了两种计算/通信交错的模式：

| 操作 | 通信类型 | 数据量 | 通信次数 |
|:----|:--------|:------|:--------|
| Attention 前向 | all-gather | $2bsh$ | 2 次（forward + backward） |
| Attention 输出投影 | all-reduce | $2bsh$ | 2 次 |
| FFN $W_1$ 前向 | 无 | — | 0 |
| FFN $W_2$ 后 all-reduce | all-reduce | $2bsh$ | 2 次 |

**每 Transformer 层的总通信量 = $4bsh$**（在每设备上）。

比较：数据并行（DP）的通信量为每步全模型梯度 $O(P)$，而 TP 的通信量为 $O(bsh)$ ——当模型远超单卡时，TP 的通信量**与模型大小无关**（仅与隐藏维度和序列长度相关），这是 TP 在超大模型训练中胜出的核心原因。

### 2.3 损失函数计算的分片

为避免交叉熵损失的完整 softmax 在单设备计算后引起通信，论文将损失函数也分片到 $t$ 个设备：

$$\text{局部 logits: } \text{logits}_i = \text{embedding}_i \cdot h$$

然后跨设备 all-gather logits，每设备独立计算 $-\log p$ 并用 all-reduce 求和。这确保了 loss 计算的每一步也是设备本地的，无单点瓶颈。

### 2.4 激活值重计算（与 [[Gradient Checkpointing]] 集成）

论文应用激活值检查点以减少张量并行的显存需求：

- 每个 Transformer 层前向时只保存输入激活值
- 反向传播时重算注意力和 FFN 的中间输出
- 张量并行下，重算在所有设备上同时进行，不增加墙钟时间

对 8.3B 模型，检查点将每设备的激活值显存从 ~32GB 降至 ~12GB。

---

## 三、Experiments and Key Findings

### 3.1 扩展性：从 1.2B 到 8.3B 参数

| 模型 | 参数 | 层数 | 隐藏维度 | Attention 头数 | GPU 数 | 并行策略 | 训练吞吐 (tokens/s) | TFLOPS/GPU |
|:---|:---:|:----:|:--------:|:------------:|:-----:|:--------|:-----------------:|:----------:|
| 1.2B DP | 1.2B | 32 | 2048 | 32 | 32 V100 32GB | DP + 仅 FP16 | — | — |
| 1.2B TP | 1.2B | 32 | 2048 | 32 | 32 | DP(4) × TP(8) | — | — |
| **8.3B** | **8.3B** | 72 | 3072 | 48 | 32 V100 32GB | DP(4) × TP(8) | — | — |
| 8.3B (重算) | 8.3B | 72 | 3072 | 48 | 32 | DP(4) × TP(8) | 14.3k (重算) | 31.4 |
| 8.3B (无重算) | 8.3B | 72 | 3072 | 48 | 64 | DP(4) × TP(16) | 16.8k (无重算) | 29.6 |

**关键发现**：8.3B 模型在 32 GPU 上以 31.4 TFLOPS/GPU（约 32% 的 V100 峰值）运行。即使重算激活值，吞吐仍远高于纯数据并行方案（数据并行在 8.3B 模型上根本无法运行）。

### 3.2 并行规模扩展（PPL 和 GLUE）

| 模型 | DP | TP | 训练方法 | 困惑度 (WikiText-2) | 语言建模 PPL |
|:---|:-:|:-:|:--------|:-----------------:|:-----------:|
| 1.2B | 8 | 4 | Adam + FP16 | 18.6 | — |
| 5.0B | 4 | 8 | Adam + FP16 | **16.8** | — |
| 8.3B | 4 | 8 | Adam + FP16 + 重算 | **15.8** | — |

**8.3B 模型在所有规模下都是质量最优的**——更大的模型始终获得更低的困惑度。结果验证了缩放法则（Scaling Law）在模型并行下的适用性。

### 3.3 通信和计算重叠

| GPU 数量 | TP 通信开销 (μs) | 与计算可重叠比例 | 有效效率损失 |
|:-------:|:---------------:|:--------------:|:----------:|
| 2 (PCIe) | 45 | 55% | ~3% |
| 4 (NVLink) | 22 | 75% | ~2% |
| 8 (DGX-2, NVSwitch) | 12 | **90%** | **<1%** |

**NVIDIA DGX-2（NVSwitch）**的全连接拓扑使得 TP 通信几乎完全被计算掩盖。这是 V100 时代 NVSwitch 相比于 PCIe 环状拓扑的关键优势。

---

## 四、Limitations and Challenges

1. **设备间通信带宽敏感**：TP 要求张量并行组内的设备间延迟极低（理想为 NVLink/NVSwitch）。跨节点 TP（跨机器）由于网络延迟会导致显著的通信开销
2. **并行粒度受限于头数**：attention head 数量决定了 TP 的最大并行度。对于 32 头的模型，TP 最多使用 32 个设备——超过此数需要重新切分注意力头
3. **仅为 Transformer 设计**：论文的张量切分方案高度针对 Transformer 的 MHA 和 FFN 结构。对 CNN、RNN 或 MoE 模型不直接适用，缺乏通用性
4. **未提供完整的并行框架**：仅解决了单节点内的张量并行。缺乏跨节点的流水线并行支持——需要 [[3D Parallelism]] 的后续工作来补全
5. **激活值显存仍是大问题**：虽然使用了检查点，但每个设备仍需存储部分激活值。序列长度增加时（如 2048→8192），显存需求线性增长
6. **TP 组内 GPU 数量必须一致**：无法灵活调整，对于不同大小的模型需要重新配置集群拓扑。DGX 节点只能提供固定的 8-GPU 互连拓扑

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Megatron-LM 的关系 |
|:--------|:----:|:--------------------|
| [[3D Parallelism]] (Narayanan et al.) | 2021 | 在 TP 基础上增加流水线并行和数据并行，形成 PTD-P 三维混合并行 |
| **Megatron v2/v3** (NVIDIA) | 2021-2022 | 序列并行 + 分布式优化器，TP 扩展到更大模型 |
| [[ZeRO]] (Rajbhandari et al.) | 2020 | 参数分片与 TP 互补——TP 只在单节点内，ZeRO 跨节点扩展 |
| **DeepSpeed-Megatron 融合** | 2021 | Microsoft 将 DeepSpeed 的 ZeRO-3 与 NVIDIA Megatron TP 整合 |
| **Colossal-AI** (HPC-AI Tech) | 2022 | 自动化的张量并行搜索扩展 Megatron-TP 到任意形状模型 |
| **NVIDIA NeMo** | 2021 | Megatron-LM 作为核心引擎的 NLP 框架 |
| **GPT-3** (Brown et al.) | 2020 | 同期工作，175B 参数训练使用了类似的模型并行技术 |

**影响评估**：Megatron-LM 的定义级贡献在于**将 Transformer 层内并行从思想变为工程实践**。其张量切分策略被几乎所有后续大规模训练框架采纳，是 PTD-P 三维并行不可或缺的技术组件。从 GPT-NeoX 到 Llama 3，TP 是唯一可在大规模 Transformer 训练中跨节点内 GPU 扩展的方案。

---

## 六、Implications for You / Hardware Compatibility

### 张量并行在不同硬件配置下的可行性

| GPU 配置 | GPU 间互连 | TP 可用性 | 推荐 TP 度 | 可训练最大模型 |
|:--------|:----------|:---------|:---------:|:-------------:|
| 单卡 RTX 3090 | — | ❌ 不支持 | 1 | ~7B (QLoRA) |
| 双卡 RTX 4090 | PCIe 4.0 x16 | ⚠️ 勉强 | 2 | ~13B |
| 4× A100 (单节点) | NVLink 600GB/s | ✅ | 4 | ~70B |
| 8× A100 (DGX) | NVSwitch 600GB/s | ✅ | 8 | ~175B |
| 跨节点 16× A100 | InfiniBand 200GB/s | ⚠️ 不推荐跨节点 TP | 每节点 8，跨节点 DP | ~530B |
| 8× H100 (DGX) | NVSwitch 900GB/s | ✅ | 8 | ~1T |

### 对大规模训练的指导

- **TP 是"最后一公里"局部优化**：TP 组大小不应跨节点，TP 内通信依赖 NVLink。跨节点通信留给 DP/PP
- **TP 总是在 DP + PP 之上使用**：合理的配置是 DP(8) × PP(4) × TP(8) ——DP 跨节点、PP 跨节点内、TP 节点内
- **TP 度必须整除 attention head 数**：典型 Llama 65B 的 64 头 → 可用 TP=1/2/4/8/16/32/64
- **与激活值检查点配合**：TP 下检查点的重算在所有 GPU 上同时进行，不增加 wall time，所以总是启用
- **7B 以下不需要 TP**：单卡可放下 7B 参数时，直接用 DP 更简单

### 硬件兼容性总结

- ✅ 单节点内 TP（NVLink DGX 节点）：A100/H100 DGX，推荐 TP=8
- ⚠️ TP=2 在 PCIe 互联 GPU 上：通信开销显著，仅当必须时使用
- ❌ 跨节点 TP：不推荐（通信延迟过高，计算效率 < 50%）
- ❌ 消费级 GPU TP：RTX 系列无 NVLink，TP 通信开销不可接受

## PDF

[[Megatron-LM 原文.pdf]]
