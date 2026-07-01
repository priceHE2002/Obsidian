---
tags:
  - 论文
  - 训练基础设施
  - 分布式训练
  - 3D并行
  - 流水线并行
created: 2026-06-30
paper_title: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"
paper_authors: "Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia"
paper_year: 2021
paper_venue: "SC 2021 (ACM Gordon Bell Prize Finalist)"
paper_citations: "~1,500+"
paper_url: "https://arxiv.org/abs/2104.04473"
github: "https://github.com/NVIDIA/Megatron-LM"
---

# 3D Parallelism

**Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**
*Deepak Narayanan, Mohammad Shoeybi, Jared Casper et al. | NVIDIA & Microsoft & Stanford | SC 2021 | arXiv: 2104.04473*

> 将数据并行（DP）、流水线并行（PP）和张量并行（TP）统一为三维混合并行策略（PTD-P），在 512-3072 张 A100 GPU 上实现 530B 到 1T 参数的有效训练。这是 NeMo-Megatron 框架的核心并行方案，定义了现代大模型训练的基础架构蓝图。

---

## 一、Background / Core Idea

### 1.1 问题：单一并行策略的三重局限

在大模型训练中，单一并行策略存在不可兼顾的三方面约束：

1. **数据并行（DP）**：当模型 + 优化器状态 > 单卡显存时，DP 直接不可行
2. **张量并行（TP）**：组内通信依赖 NVSwitch 的极低延迟，跨节点 TP 效率不可接受
3. **流水线并行（PP）**：简单层间切分产生大量空闲 bubble，硬件利用率低

三者各有短板，但三者恰好互补。问题在于：**如何形式化地找到最优组合？**

### 1.2 核心洞察：三维正交化

论文的核心洞察在于三个并行维度的**正交性**：

| 并行维度 | 切分对象 | 通信模式 | 通信范围 | 最优粒度限制 |
|:--------|:--------|:--------|:--------|:-----------|
| DP | 训练样本 | gradient all-reduce | **跨节点**（InfiniBand） | 总 GPU 数 |
| TP | 层内参数 | all-gather / reduce-scatter | **节点内**（NVSwitch） | $\le$ 节点 GPU 数 |
| PP | 层 | point-to-point 激活值 | **节点间**（IB/NVLink） | 不超过总层数 |

**DP 善跨节点、TP 善节点内、PP 善跨节点但会产生 bubble**——三者组合后可 100% 利用集群的带宽层次结构。

### 1.3 集群拓扑与 DP/TP/PP 的映射

一个典型的 A100 DGX 集群拓扑：

```
节点 0 (8×A100 NVSwitch)      节点 1 (8×A100 NVSwitch)      节点 N-1
    ↓      ↑                        ↓      ↑                    ↓  ↑
    └──────┴─────── InfiniBand ─────┴──────┴──────── InfiniBand ──┘  └
```

论文提出的映射规则极其简洁：

- **TP = 节点级并行**：TP 组大小 = 8（一个 DGX 节点内的 8 张 GPU）
- **PP = 跨节点组内并行**：4-8 个 TP 组通过 IB 组成 PP 流水线
- **DP = 跨流水线副本并行**：多个 PP 流水线副本之间数据并行

```
              ┌─────────────────────────────────┐
              │        流水线副本 0 (DP=0)        │
              │  Node 0 (TP=8) → Node 1 (TP=8)  │
              │              ↓                   │
              │        流水线副本 1 (DP=1)        │
              │  Node 2 (TP=8) → Node 3 (TP=8)  │
              └─────────────────────────────────┘
```

---

## 二、Method / Architecture / Technical Contribution

### 2.1 1F1B 流水线调度（One-Forward-One-Backward）

论文最大的技术贡献是 **1F1B 调度**——相比传统 GPipe 调度，它大幅降低了流水线 bubble 比例。

**GPipe 调度的 bubble**：
```
GPU0: [F0.F1.F2.F3][B0.B1.B2.B3]                    ← 50% idle
GPU1: ....[F0.F1.F2.F3][B0.B1.B2.B3]                ← 50% idle
GPU2: ........[F0.F1.F2.F3][B0.B1.B2.B3]            ← 50% idle
GPU3: ............[F0.F1.F2.F3][B0.B1.B2.B3]        ← 50% idle
```

Bubble ratio: $\frac{p-1}{m+p-1}$，其中 $p$ 是流水线深度，$m$ 是 micro-batches 数。当 $m$ 较大时 bubble 较小，但需要更多显存。

**1F1B 调度**：
```
GPU0: [F0][F1][F2][F3][B0][B1][B2][B3]             ← 几乎无 idle
GPU1: ....[F0][F1][F2][B0][F3][B1][B2][B3]         
GPU2: ........[F0][F1][B0][F2][B1][F3][B2][B3]     
GPU3: ............[F0][B0][F1][B1][F2][B2][F3][B3]
```

**1F1B 的关键机制**：每个设备的计算局部执行"前向→反向交替"模式，一旦收到前向结果立即开始前向，收到反向激活立即开始反向。**减少空闲的代价是增加了显存压力**——因为多个 micro-batch 的激活值需要同时保留。

### 2.2 显存分析的统一框架

论文提出了一种统一框架来分析 DP、PP、TP 在不同配置下的显存需求：

$$\text{每 GPU 显存} = \underbrace{\text{模型参数}}_{\text{TP 分担}} + \underbrace{\text{优化器状态}}_{\text{DP + TP 分担}} + \underbrace{\text{激活值}}_{\text{PP + TP 分担 + 检查点}}$$

对于 Transformer 模型，每层激活值大小为：

$$A_{\text{per\_layer}} = s \times b \times h \times (34 + 5 \times \frac{s}{h \times t})$$

其中 $s$ = 序列长度，$b$ = micro-batch size，$h$ = 隐藏维度，$t$ = TP 并行度。

关键参数分析（530B 模型，A100 80GB）：

| 配置 | 模型参数 | 优化器状态 | 激活值 | 总计 | 是否可训练 |
|:----|:-------:|:---------:|:-----:|:---:|:---------:|
| DP(3072) | 1TB | 8TB | 4.5TB | 13.5TB | ❌ 每卡 > 80GB |
| TP(8)+DP(384) | 66GB | 20GB | 32GB | 118GB | ⚠️ 勉强 |
| **TP(8)+PP(4)+DP(96)** | 66GB | 10GB | 8GB | **84GB** | **✅ 接近 80GB** |
| TP(8)+PP(4)+DP(96)+检查点 | 66GB | 10GB | 4GB | **80GB** | **✅ 刚好** |

### 2.3 Interleaved 1F1B 调度（v2 版本的优化）

论文在后续实验中引入交错 1F1B（Interleaved 1F1B），将每个设备分配到多个不相邻的流水线阶段：

```
GPU0: [F0][F4][F1][F5][B0][B4][B1][B5]
GPU1: [F2][F6][F3][F7][B2][B6][B3][B7]
```

Bubble ratio 从 $\frac{p-1}{m+p-1}$ 降至 $\frac{p-1}{2m+p-1}$——近似减半。代价是增加了约 2 倍的 p2p 通信量。

### 2.4 序列并行（Sequence Parallelism）

为解决长序列训练的内存瓶颈，论文将 **LayerNorm + Dropout 的计算**沿序列维度切分到 TP 组的所有 GPU：

$$\text{LN 前: } \text{all-reduce}_{seq} \to \text{局部 LN} \to \text{局部 Dropout} \to \text{all-gather}_{seq}$$

通常 LayerNorm 和 Dropout 的激活值占 Transformer 层总激活值的约 60%，序列并行可将这部分显存再减少 TP 倍。

---

## 三、Experiments and Key Findings

### 3.1 530B 参数模型（A100 集群）

| 配置 | GPU 数 | DP | TP | PP | 激活重算 | TFLOPs/GPU | 效率 |
|:----|:-----:|:--:|:--:|:--:|:-------:|:---------:|:----:|
| 530B 大 batch | 1024 | 32 | 4 | 8 | 无 | **138** | **52%** |
| 530B 小 batch | 1024 | 64 | 2 | 8 | 有 | 112 | 42% |
| **1T 参数弱扩展** | 3072 | 96 | 8 | 4 | 有 | **141** | **53%** |

**530B 模型在 1024 A100 GPU 上以 52% 的 MFU（Model FLOPS Utilization）运行**——这在当时（2021 年）是行业最高记录。

### 3.2 流水线并行 vs 激活重算的显存效果

| 配置 | 激活值显存/GPU | 总显存/GPU | 空闲 bubble 比例 | 吞吐 (tokens/s) |
|:----|:------------:|:---------:|:---------------:|:--------------:|
| 无 PP, 无重算 | 96GB | 160GB | 0% | ❌ 超出显存 |
| PP=4, 无重算 | 32GB | 96GB | ~10% | 254k |
| PP=4, 有重算 | 12GB | 76GB | ~10% | 234k |
| PP=8, 有重算 | 8GB | 72GB | ~20% | 197k |

**PP=4 + 激活重算**在显存效率和吞吐之间达成最佳平衡。

### 3.3 微批量大小对效率的影响

| Micro-batch / GPU | 总 batch size | 吞吐 (tokens/s) | MFU |
|:----------------:|:------------:|:--------------:|:---:|
| 1 | 256 | 108k | 44% |
| 2 | 512 | 168k | 48% |
| 4 | 1024 | 197k | 50% |
| 8 | 2048 | 234k | 52% |
| 16 | 4096 | 258k | 52% |

**关键结论**：微批量大小超过 4 后 MFU 饱和。在总 batch size 固定的前提下，应选择不损失 MFU 的最小 micro-batch 以节省显存。

### 3.4 端到端训练 BLEU（机器翻译任务）

| 模型 | 参数 | 并行策略 | BLEU (En→De) | BLEU (En→Fr) |
|:----|:---:|:--------|:-----------:|:-----------:|
| 标准 Transformer Big | ~0.2B | DP(32) | 28.6 | 41.0 |
| Megatron-LM 2.5B | 2.5B | DP(8) × TP(4) | 29.2 | 41.4 |
| Megatron-LM 8.3B | 8.3B | DP(4) × TP(8) | 29.4 | 41.6 |
| Megatron-Turing NLG 530B | 530B | DP(32)×PP(8)×TP(4) | **30.5** | **42.8** |

**模型越大 BLEU 越高**——即使在机器翻译这种相对低维的任务上，530B 参数仍比 0.2B 高出近 2 个 BLEU 点。

---

## 四、Limitations and Challenges

1. **Bubble 比例的理论下界**：即使使用 1F1B 调度，流水线并行仍会引入 $\frac{p-1}{m+p-1}$ 的空闲比例。当流水线深度 $p$ 增大时（如 PP=32），bubble 比例不可忽略
2. **TP 组大小受限于节点 GPU 数**：DGX A100 最多 8 卡 NVSwitch，TP 最大为 8。——超大规模下 TP=8 不足以显存容纳，需要 PP 分担
3. **PP 调度引入通信-计算耦合**：1F1B 的交替模式使调度器实现复杂，且对网络延迟敏感（延迟增加直接扩大 bubble）
4. **隐式假定同构集群**：论文假设集群中所有节点相同。异构集群（如 A100 80G + A100 40G 混合）的配置未讨论
5. **序列并行仅覆盖 LayerNorm + Dropout**：注意力层自身的 O$(s^2)$ 激活值并未被序列并行覆盖，长序列仍受限
6. **配置搜索空间爆炸**：三个并行维度的选择组合呈 $O(N_{dp} \times N_{tp} \times N_{pp})$，每次集群重配置需要重新搜索最优参数

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 3D Parallelism 的关系 |
|:--------|:----:|:-----------------------|
| [[Megatron-LM]] (Shoeybi et al.) | 2019 | TP 的基础工作，3D Parallelism 将其从单节点扩展到跨节点 |
| [[ZeRO]] (Rajbhandari et al.) | 2020 | 参数分片的跨节点方案，与 PTD-P 互补（ZeRO-3+TP+PP 组合） |
| **DeepSpeed ZeRO-3 + Megatron 融合** | 2021 | Microsoft 将两个框架的并行策略统一在同一训练流水线中 |
| **NVIDIA NeMo Megatron** | 2021 | 3D Parallelism 的工程实现框架，支持全自动并行配置 |
| **T5-PaLM** (Chowdhery et al.) | 2022 | 540B PaLM 训练采用类似的 DP+TP+PP 策略，确认 3D Parallelism 的通用性 |
| **FSDP** (Zhao et al.) | 2023 | ZeRO-3 的 PyTorch 原生实现，DP 维度的显存分片 |
| **Colossal-AI** | 2022 | 自动化的 3D 并行配置搜索，将 PTD-P 的配置工程自动化 |

**影响评估**：3D Parallelism 是**大语言模型训练的终极并行配方**。从 2021 年到今天，所有超过 100B 参数的模型训练（GPT-4、PaLM、Llama 3、Claude）都使用了此论文定义的三维混合并行模式。PTD-P（Pipeline-Tensor-Data Parallelism）已成为工业界的行业标准。

---

## 六、Implications for You / Hardware Compatibility

### 不同 GPU 规模下的 PTD-P 配置建议

| 集群规模 | GPU 类型 | 推荐配置 | 可训最大模型 | 预期 MFU |
|:---------|:--------|:---------|:-----------:|:--------:|
| 单节点 (8 GPU) | A100 80G | DP(8) | ~7B (全参) | ~55% |
| 单节点 (8 GPU) | H100 80G | TP(8) | ~13B (全参) | ~52% |
| 4 节点 (32 GPU) | A100 80G | DP(4) × PP(2) × TP(8) | ~175B | ~48% |
| 16 节点 (128 GPU) | A100 80G | DP(4) × PP(4) × TP(8) | ~530B | ~52% |
| 64 节点 (512 GPU) | A100 80G | DP(8) × PP(8) × TP(8) | ~1T | ~50% |
| 128 节点 (1024 GPU) | H100 | DP(16) × PP(8) × TP(8) | ~3T | ~53% |

### 对大规模训练的指导

- **三层级映射是固定规则**：TP=节点内互联（NVSwitch/NVLink），PP=跨节点组（IB），DP=跨流水线副本
- **1F1B 调度的显存-效率权衡**：增加 micro-batch 数量可减少 bubble，但增加激活值显存。应按 1F1B 调度找到每 GPU 可容纳的最大 micro-batch 数
- **优先调整 PP 深度**：在三个维度中，PP 的 bubble 效率损失最可预测（公式已知）。TP 的通信开销最无弹性。因此调整策略为：TP 固定为节点 GPU 数 → PP 调整为使显存恰好可容纳 → DP 填满剩余 GPU
- **激活重算总是开启**：3D Parallelism 中激活值检查点的 1.2-1.3× 计算开销远小于 PP bubble 增加一档的损失

### 硬件兼容性总结

- ✅ 3D Parallelism（PTD-P）：A100/H100 DGX 集群（最优）
- ✅ PTD-P（dp+pp+tp)：4+ 节点 DGX 集群
- ⚠️ PTD-P 在 NVLink 互联的消费者 GPU（RTX 3090 SLI）上：TP 通信成为瓶颈
- ❌ 单卡或小集群（<4 GPU）：不需要 PTD-P，DP 或简单的 PP 即可

## PDF

[[3D Parallelism 原文.pdf]]
