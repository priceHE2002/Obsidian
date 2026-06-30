---
tags:
  - 论文
  - 参数高效微调
  - 低秩分解
created: 2026-06-30
paper_title: "LoRA: Low-Rank Adaptation of Large Language Models"
paper_authors: "Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen"
paper_year: 2021
paper_venue: "ICLR 2022"
paper_citations: "~25,000+"
paper_url: "https://arxiv.org/abs/2106.09685"
---

# LoRA

**LoRA: Low-Rank Adaptation of Large Language Models**
*Microsoft | ICLR 2022 | arXiv: 2106.09685*

> 提出 Low-Rank Adaptation——不修改原始模型参数，在旁路加入可训练的低秩矩阵（$A \times B$, $r \ll d$），将微调的参数量减少 10,000 倍，同时保持全量微调的性能。对于 VLA 研究者来说，这是让 7B 模型在消费级 GPU 上可微调的关键技术。

---

## 一、研究背景与动机

大语言模型的规模持续增长（GPT-3 175B, LLaMA 65B），但全量微调（full fine-tuning）的代价高到不可接受——每个下游任务都要存储一个完整的模型副本。Adapter、Prefix Tuning 等参数高效微调（PEFT）方法虽然减少了参数量，但引入了推理延迟。

核心观察：预训练模型具有**低内在维度**（intrinsic dimension）——即模型在预训练中学到的特征空间实际上是低秩的。这意味着权重更新的有效自由度远小于参数量。

## 二、核心方法

**核心假设：** 预训练权重的更新矩阵 $\Delta W$ 具有低"内在秩"——可以用两个低秩矩阵的乘积近似。

**公式：**

$$h = W_0 \cdot x + \Delta W \cdot x = W_0 \cdot x + B \cdot A \cdot x$$

其中 $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$, $r \ll \min(d_{in}, d_{out})$

| 组件 | 初始化 | 作用 |
|------|--------|------|
| $W_0$ | 预训练权重（冻结） | 保留原始知识 |
| $A$ | 高斯初始化 $N(0, \sigma^2)$ | 降维投影 |
| $B$ | 零初始化 | 升维投影 |

**初始化策略：** $A$ 用高斯初始化，$B$ 用零初始化。训练开始时 $\Delta W = 0$，不影响原模型输出。这避免了训练初期破坏预训练权重。

**推理时合并：** $\tilde{W} = W_0 + B \cdot A$，合并后零额外推理延迟。

**在 GPT-3 175B 上的关键结果：**

| 方法 | 可训练参数 | 参数减少倍数 | 性能 |
|------|-----------|-------------|------|
| 全量微调 | 175B | 1x | 基准 |
| LoRA (r=4) | 2.4M | ~73,000x | 持平 |
| LoRA (r=8) | 4.7M | ~37,000x | 持平 |
| LoRA (r=64) | 33M | ~5,300x | 持平 |

一个预训练模型 + 多个 LoRA 模块 = 多任务部署，每个模块只需几 MB。

## 三、关键实验与发现

- **GPT-3 175B**：LoRA (r=8) 性能与全量微调持平，参数量减少 37,000 倍
- **RoBERTa**：在 GLUE 基准上，LoRA 超过全量微调（作为正则化）
- **DeBERTa**：在 XL Sum 摘要任务上，LoRA 与全量微调持平
- **低秩假设验证**：对 $\Delta W$ 做 SVD 分解，发现其奇异值分布高度衰减，验证了"低内在秩"假设
- **最佳 rank 选择**：通常 r=8 到 64 之间，r 越大量化越接近全量微调但参数更多

## 四、局限性与后续影响

**局限：**
- rank r 需要手工选择（通常 r=8~64）
- 对某些分布偏移剧烈的任务，LoRA 可能不如全量微调灵活
- 在只有少量训练数据的场景下，LoRA 可能过拟合（因为低秩约束不够强）

**后续影响：**
- **QLoRA (Dettmers et al., 2023)**：LoRA + 4-bit NormalFloat 量化 → 65B 模型微调仅需 48GB（单卡 A100）
- **AdapterFusion, AdapterBridge**：LoRA 思想的后续拓展
- 事实上成为**大模型微调的事实标准**——HuggingFace PEFT 库的核心实现

## 五、VLA/机器人研究中的角色

LoRA 是让 VLA 模型在消费级 GPU 上可微调的关键技术：

- **OpenVLA 的微调 = LoRA (rank=32)**：在消费级 GPU 上微调 7B VLA 模型的核心技术。4-bit QLoRA 微调在 16GB 显卡上可行（12-16GB, batch_size=1）
- **VLA-Adapter** 与 LoRA 互补——LoRA 微调 VLM 内部，VLA-Adapter 在外部加 Bridge
- **FLOWER 和 X-VLA** 的 prompt tuning 可以看作 LoRA 思想的扩展
- 应用场景：用 LoRA 微调 OpenVLA 适配特定机器人形态或特定任务数据集

## 六、对你的启示

- **LoRA 是 VLA 微调的必备技能**：有 16GB GPU 的话，4-bit QLoRA (rank=32) 微调 7B OpenVLA 是可行的（batch_size=1）
- **理解低秩假设**：LoRA 成功的前提是预训练模型的"低内在秩"——这对模型理解能力没有根本损失
- **QLoRA 是实际选择**：纯 bf16 LoRA 微调 7B ~ 14-22GB，4-bit QLoRA 可降到 10-16GB，正好适配 16GB GPU
- **LoRA 模块的复用性**：为不同任务（桌面操作、导航等）训练不同的 LoRA 模块，部署时灵活切换
- **关注后续工作**：DoRA (Weight-Decomposed Low-Rank Adaptation, 2024) 和 FourierFT 是 LoRA 的改进变体

## PDF

[[LoRA.pdf]]
