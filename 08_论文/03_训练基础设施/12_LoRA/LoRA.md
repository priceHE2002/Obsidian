---
tags:
  - 论文
  - 参数高效微调
  - 低秩分解
  - PEFT
created: 2026-06-30
paper_title: "LoRA: Low-Rank Adaptation of Large Language Models"
paper_authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen"
paper_year: 2021
paper_venue: "ICLR 2022"
paper_citations: "~28,000+"
paper_url: "https://arxiv.org/abs/2106.09685"
github: "https://github.com/microsoft/LoRA"
---

# LoRA

**LoRA: Low-Rank Adaptation of Large Language Models**
*Edward J. Hu, Yelong Shen, Phillip Wallis et al. | Microsoft Research | ICLR 2022 | arXiv: 2106.09685*

> 冻结预训练权重，在旁路注入可训练的低秩分解矩阵（$B \cdot A$, $r \ll d$），使微调参数量减少 10,000 倍且推理零延迟。对 VLA 研究而言，LoRA（尤其是 4-bit QLoRA）是让 7B 模型在 16GB 消费级 GPU 上可微调的核心使能技术。

---

## 一、Background / Core Idea

### 1.1 问题：大模型微调的存储与部署困境

GPT-3 175B 规模下，全量微调存在根本性挑战：
- **存储成本**：每个下游任务需要存储完整的 175B 参数（约 350GB 的 fp16 checkpoints）。100 个独立微调副本需要 35TB
- **GPU 显存**：全量微调需要存储优化器状态（Adam: 参数的 momentum + variance 共 $2 \times$ 参数量），GPT-3 175B 训练时 VRAM 需求高达 1.2TB
- **部署切换**：切换下游任务需要加载整个模型，无法实现热切换

### 1.2 核心洞察：预训练模型的低内在维度

论文的核心理论基础来自 **Aghajanyan et al. (2020) 的"Intrinsic Dimensionality"**：

> 预训练模型在过参量化空间中实际嵌入在**低维子空间**中。即使将参数更新随机投影到一个极低维的子空间（如 $d=200$），模型仍能高效学习。

LoRA 论文进一步提出**更强的假设**：权重更新矩阵 $\Delta W$ 本身也具有**低"内在秩"**（low intrinsic rank），即：

$$\Delta W = B \cdot A, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d,k)$$

### 1.3 与已有 PEFT 方法的对比

| 方法 | 推理延迟 | 序列长度影响 | 质量 vs 全量微调 |
|------|:-:|:-:|:-:|
| **Adapter Layers** (Houlsby 2019) | **有**（顺序执行） | 无 | 接近 |
| **Prefix Tuning** (Li & Liang 2021) | 无 | **有**（占用输入序列） | 较差（非单调） |
| **Prompt Tuning** (Lester 2021) | 无 | 有 | 较差 |
| **LoRA (本文)** | **零延迟** | **无** | **持平或更优** |

Adapter 层虽然参数量少（<1%），但在小批量推理场景下引入显著延迟——在 GPT-2 Medium 上单卡单样本推理，Adapter 导致 20-30% 的延迟增加（瓶颈维度过大时更严重）。Prefix Tuning 的困难在于：性能随可训练参数数目的变化**非单调**，超过特定阈值后反而下降。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 低秩参数化更新（Low-Rank-Parametrized Update Matrices）

**数学形式：**

给定预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，其更新约束为低秩分解：

$$h = W_0 x + \Delta W x = W_0 x + BA x$$

其中：
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$
- $W_0$ **冻结**，不接收梯度更新
- $A$ 和 $B$ 是唯一可训练的参数

**初始化策略：**

| 矩阵 | 初始化方法 | 理由 |
|------|-----------|------|
| $A$ | 随机高斯 $\mathcal{N}(0, \sigma^2)$ | 打破对称性，实现梯度传播 |
| $B$ | **零初始化** | 训练开始时 $\Delta W = BA = 0$，不影响原模型输出 |

这与 Adapter 层的初始化策略完全不同——Adapter 通常使用近零初始化（如 $\mathcal{N}(0, 0.01)$），而 LoRA 在开始时严格为零更新，避免破坏预训练权重的初始输出分布。

### 2.2 缩放因子 $\frac{\alpha}{r}$

LoRA 的关键实践技巧：

$$\text{output} = W_0 x + \frac{\alpha}{r} \cdot BA x$$

其中 $\alpha$ 是与 $r$ 相关的常数。论文指出：**调 $\alpha$ 在 Adam 优化下大致等价于调学习率**。实践中，通常将 $\alpha$ 设置为第一次尝试的 $r$ 值（如 $r=8, \alpha=8$），后续改变 $r$ 时不再重新调参。这一缩放灵感来自 **Yang & Hu (2021) 的 µP (Maximal Update Parameterization)**。

### 2.3 推理时合并（Zero Additional Latency）

训练完成后，可直接计算合并权重：

$$\tilde{W} = W_0 + \frac{\alpha}{r} \cdot BA$$

推理时 $h = \tilde{W}x$，与全量微调模型的计算图完全相同——无额外深度、无额外激活值计算。切换任务时只需 $\tilde{W} \to \tilde{W} - B_1A_1 + B_2A_2$，存储仅需基模型 (350GB) + 多个 LoRA 模块 (每个 35MB)。

### 2.4 对 Transformer 各权重矩阵的应用

论文系统研究了 LoRA 应应用于哪些注意力权重矩阵。GPT-3 175B 的参数预算为 18M 可训练参数：

| 权重类型 | WikiSQL 准确率 | MultiNLI 准确率 | 最佳组合 |
|:-:|:-:|:-:|:-:|
| $W_q$ (r=8) | 70.4 | 91.0 | — |
| $W_k$ (r=8) | 70.0 | 90.8 | — |
| $W_v$ (r=8) | 73.0 | 91.0 | — |
| $W_o$ (r=8) | 73.2 | 91.3 | — |
| $W_q, W_k$ (r=4) | 71.4 | 91.3 | — |
| $W_q, W_v$ (r=4) | **73.7** | **91.3** | ✅ **最佳** |
| $W_q, W_k, W_v, W_o$ (r=2) | 73.7 | **91.7** | 质量略优 |

**关键发现**：适应 $W_q$ 和 $W_v$ 的 LoRA 组合最优。仅适应 $W_q$ 会损失性能。**宁可减小 $r$ 也要覆盖更多权重类型**——说明 $\Delta W$ 的信息分布在多个投影矩阵中。

### 2.5 Rank 的实证研究

在 GPT-3 175B 上测试 $r=1,2,4,8,64$ 的效果：

| 权重配置 | $r=1$ | $r=2$ | $r=4$ | $r=8$ | $r=64$ |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 仅 $W_q$ (WikiSQL) | 68.8 | 69.6 | 70.5 | 70.4 | 70.0 |
| $W_q, W_v$ (WikiSQL) | 73.4 | 73.3 | 73.7 | 73.8 | 73.5 |
| $W_q, W_k, W_v, W_o$ (WikiSQL) | 74.1 | 73.7 | 74.0 | 74.0 | 73.9 |

**惊人发现：$r=1$ 足以完成任务**（对 $W_q, W_v$ 组合）。这意味着 $\Delta W$ 的"有效秩"低至 1。

**子空间分析**（Grassmann 距离）：论文通过奇异值分解分析 $A_{r=8}$ 和 $A_{r=64}$ 的子空间相似度：

$$\phi(A_{r=8}, A_{r=64}, i, j) = \frac{\|U_{A_{r=8}}^{i\top} U_{A_{r=64}}^{j}\|_F^2}{\min(i,j)} \in [0,1]$$

结论：**$r=8$ 和 $r=64$ 学习到的子空间在前 1 个奇异方向上高度重叠**（相似度 > 0.5），其他方向主要为训练中积累的随机噪声。这正是极低 $r$ 仍有效的深层原因。

### 2.6 $\Delta W$ 与 $W$ 的关系

论文通过计算 Frobenius 范数探究 $\Delta W$ 是否放大 $W$ 中已有的特征方向：

| 比较对象 | $r=4$ $\|U^\top W V^\top\|_F$ | $r=64$ $\|U^\top W V^\top\|_F$ |
|:-:|:-:|:-:|
| $\Delta W$ 方向 | **0.32** | 1.90 |
| $W$ 的 top-r 方向 | 21.67 | 37.71 |
| 随机矩阵方向 | 0.02 | 0.33 |
| $\|W\|_F = 61.95$ | $\|\Delta W\|_F = 6.91$ | $\|\Delta W\|_F = 3.57$ |

**三个结论：**
1. $\Delta W$ 与 $W$ 的相关性远大于随机矩阵 → $\Delta W$ 放大了 $W$ 中已有的特征
2. $\Delta W$ 重复的不是 $W$ 的 top 方向 → 它放大了 $W$ 中**未强调**的方向
3. **放大因子巨大**：$r=4$ 时放大因子 $\approx 6.91/0.32 \approx 21.6$ 倍

这意味着 LoRA 通过学习任务特定的"弱特征放大"来实现高效适应。

---

## 三、Experiments and Key Findings

### 3.1 RoBERTa Base/Large 在 GLUE 上的表现

| 模型与方法 | 可训练参数 | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B | **平均** |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| RoBERTa base (全量微调) | 125.0M | 87.6 | 94.8 | 90.2 | 63.6 | 92.8 | 91.9 | 78.7 | 91.2 | 86.4 |
| RoBERTa base (LoRA) | **0.3M** | **87.5** | **95.1** | 89.7 | 63.4 | 93.3 | **90.8** | **86.6** | **91.5** | **87.2** |
| RoBERTa large (全量微调) | 355.0M | 90.2 | 96.4 | 90.9 | 68.0 | 94.7 | 92.2 | 86.6 | 92.4 | 88.9 |
| RoBERTa large (LoRA) | **0.8M** | **90.6** | 96.2 | **90.9** | **68.2** | **94.9** | 91.6 | **87.4** | **92.6** | **89.0** |

**LoRA 超越全量微调**（尤其在 RTE 和 STS-B 上），作者归因于低秩约束的正则化效应。

### 3.2 GPT-3 175B 上的大规模验证

| 方法 | 可训练参数 | WikiSQL | MNLI-m | SAMSum (R1/R2/RL) |
|------|:-:|:-:|:-:|:-:|
| 全量微调 | 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| BitFit | 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| PrefixEmbed (256+8) | 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| PrefixLayer (8+8) | 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| AdapterH (r=4) | 7.1M | 71.9 | 89.8 | 53.0/28.9/44.8 |
| AdapterH (r=8) | 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| **LoRA (r_v=2)** | **4.7M** | **73.4** | **91.7** | **53.8/29.8/45.9** |
| LoRA ($W_q,W_k,W_v,W_o$, r=2) | 37.7M | 74.0 | 91.6 | 53.4/29.2/45.1 |

**LoRA 用 0.003% 的参数超越全量微调**。

### 3.3 低数据场景

| 方法 | MNLI-100 | MNLI-1k | MNLI-10k | MNLI-full |
|:-:|:-:|:-:|:-:|:-:|
| 全量微调 | 60.2 | 85.8 | 88.9 | 89.5 |
| PrefixEmbed | 37.6 | 75.2 | 79.5 | 88.6 |
| PrefixLayer | 48.3 | 82.5 | 85.9 | 89.6 |
| **LoRA** | **63.8** | **85.6** | **89.2** | **91.7** |

LoRA 在极低数据场景（100 样本）下表现最佳，而 Prefix Tuning 甚至不及随机（37.6% vs 33.3%）。

### 3.4 推理延迟分析（LoRA vs Adapter）

| 条件 | Fine-Tune/LoRA | AdapterL | AdapterH |
|:-:|:-:|:-:|:-:|
| Batch=32, Seq=512 | 1449.4±0.8 ms | 1482.0 (+2.2%) | 1492.2 (+3.0%) |
| Batch=16, Seq=256 | 338.0±0.6 ms | 354.8 (+5.0%) | 366.3 (+8.4%) |
| **Batch=1, Seq=128** | **19.8±2.7 ms** | **23.9 (+20.7%)** | **25.8 (+30.3%)** |

在线小批量推理场景下，Adapter 层的顺序执行导致不可忽略的延迟增加。LoRA 与全量微调零差异。

---

## 四、Limitations and Challenges

1. **Rank $r$ 的手工选择**：$r=1$ 对 GPT-3 有效，但 GPT-2 Medium 的最佳 $r$ 在 4-16 之间。不同模型、不同任务的最佳 $r$ 不同，缺乏理论指导
2. **批量混合多任务困难**：不同 LoRA 模块的 $B, A$ 无法在一个 batch 中混合处理（除非不合并权重，但这样失去零延迟优势）
3. **低数据过拟合风险**：低秩约束在某些场景下不够强，LoRA 在小数据上可能过拟合（虽比 Prefix Tuning 好）
4. **未覆盖所有层**：论文仅应用 LoRA 于自注意力权重，MLP 层和 LayerNorm 偏置未被系统研究
5. **增量更新的性质**：LoRA 的线性设计本质上假设更新是加性的——对需要参数剧变的任务（如语言迁移）不适应
6. **SVD 子空间分析局限**：子空间相似度度量 $\phi$ 对维度敏感，高维低相关性不等于无贡献

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 LoRA 的关系 |
|---------|:----:|---------------|
| **QLoRA** (Dettmers et al.) | 2023 | LoRA + 4-bit NormalFloat 量化 + 双量化，65B 模型微调仅需 48GB |
| **DoRA** (Liu et al.) | 2024 | 权重分解：$W = m \cdot \frac{V}{\|V\|_c}$，LoRA 应用于 $V$ 分量 |
| **AdapterFusion** (Pfeiffer et al.) | 2021 | 组合多个 Adapter/LoRA 的知识 |
| **VeRA** (Kopiczko et al.) | 2023 | 共享随机矩阵 + 可训练缩放向量，进一步减少参数 |
| **PiSSA** (Meng et al.) | 2024 | 用主奇异分量初始化 LoRA，加速收敛 |
| **LoRA-FA** (Zhang et al.) | 2023 | 冻结 $A$ 仅训练 $B$，进一步减少梯度计算 |
| **Delta-LoRA** (Zi et al.) | 2023 | 在 LoRA 权重更新中加入动量项 |
| **HuggingFace PEFT** | 2023 | LoRA 作为核心实现的统一 PEFT 框架 |

**影响评估**：LoRA 已成为大模型微调的**事实标准**。2023-2024 年几乎所有开源 LLM 微调都基于 LoRA 或其变体。从 Llama 2 到 CodeLlama、从 Mistral 到 Gemma，LoRA 是支撑参数高效微调的基石技术。

---

## 六、Implications for You / Hardware Compatibility

### GPU 显存需求（LoRA 微调 7B 模型）

| 配置 | 训练显存 | 推理显存 | 可使用 GPU |
|------|:-:|:-:|:--|
| bf16 LoRA (r=32, QKV) | ~14-22GB | ~14GB | ✅ RTX 3090/4090 (24GB) |
| 4-bit QLoRA (r=32) | ~10-16GB | ~6-8GB | ✅ RTX 4060 (16GB) / **RTX 3060 (12GB)** |
| bf16 LoRA (7B, 仅 QV, r=8) | ~14GB | ~14GB | ❌ RTX 4060 Ti (16GB, 勉强) |
| 4-bit QLoRA (13B, r=32) | ~20-24GB | ~10GB | ✅ A100 (40GB) / RTX 4090 (24GB) |
| 4-bit QLoRA (70B, r=16) | ~48GB | ~35GB | ⚠️ 仅 A100 (80GB) |

### 对 VLA 研究的指导

- **OpenVLA 微调标配 = LoRA (rank=32)**：7B VLA 模型在消费 GPU 上可微调。使用 4-bit QLoRA + gradient checkpointing 可在 16GB GPU 上运行（batch_size=1）
- **理解 LoRA 的选择**：LoRA 应用于 QKV + FFN 投影矩阵（通常约 60-80 个 LoRA 模块），而非 RMSNorm（已足够稳定）
- **推理效率**：合并后的模型与原始 OpenVLA 推理代码完全兼容，无额外延迟
- **α/r 缩放的重要性**：典型设置为 $\alpha=32$, $r=32$（ratio=1），但 $\alpha$ 可能需要根据特定 VLA 任务微调
- **多任务部署**：可为不同机器人形态（Aloha、Franka、UR5）训练独立 LoRA 模块，推理时热切换

### 硬件兼容性总结
- ✅ bf16 LoRA 微调 7B：RTX 3090/4090 (24GB)
- ✅ 4-bit QLoRA 微调 7B：RTX 3060/4060 (12-16GB)
- ⚠️ 4-bit QLoRA 微调 13B：RTX 4090 (24GB, 需要 gradient checkpointing)
- ❌ 全量微调 7B：仅 A100 (80GB)

## PDF

[[LoRA 原文.pdf]]
