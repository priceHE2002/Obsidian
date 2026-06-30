---
tags:
  - 论文
  - Scaling Law
  - 模型规模
  - 涌现
created: 2026-06-30
paper_title: "Scaling Laws for Neural Language Models"
paper_authors: "Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei"
paper_year: 2020
paper_venue: "arXiv 2001.08361"
paper_citations: "~8,000+"
paper_url: "https://arxiv.org/abs/2001.08361"
---

# Scaling Laws

**Scaling Laws for Neural Language Models**
*OpenAI | arXiv 2001.08361*

> 建立了神经网络性能与模型规模、数据量、计算量之间的幂律关系。核心发现：性能提升不是线性的，而是遵循幂律——每翻倍模型规模，性能提升有固定比例。解释了为什么 VLA 从 5B 扩到 55B 不只是"更好一点"，而是涌现出新能力。

---

## 一、研究背景与动机

在 2020 年之前，深度学习的"越大越好"还是一种经验感受而非科学规律。研究者们直观地知道更大的模型表现更好，但没有清晰的数学模型来描述规模与性能之间的关系。

关键问题：
- 给定固定的计算预算，应该增大模型还是增大数据？
- 模型性能是否有上限？还是可以无限提升？
- 不同规模的模型之间是否存在可预测的关系？

Scaling Laws 论文通过在 8 个数量级的计算量范围内进行系统实验（从 768-parameter LSTM 到 1.5B-parameter Transformer），给出了这些问题的首次定量回答。

## 二、核心方法

### 三个核心 Scaling Law

论文建立了三个核心的幂律关系：

#### 1. 模型规模 (N) 的 Scaling Law

$$ L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N} $$

其中 $L$ 是交叉熵损失，$N$ 是参数量（不含 embedding），$N_c \approx 8.8 \times 10^{13}$，$\alpha_N \approx 0.076$。

**解读**：模型参数每翻倍（$N \to 2N$），损失减少 $1 - 2^{-\alpha_N} \approx 5.1\%$。

#### 2. 数据量 (D) 的 Scaling Law

$$ L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D} $$

其中 $D_c \approx 5.4 \times 10^{13}$，$\alpha_D \approx 0.095$。

#### 3. 计算量 (C) 的 Scaling Law

$$ L(C_{\min}) = \left(\frac{C^{\min}_c}{C_{\min}}\right)^{\alpha_{C_{\min}}} $$

其中 $C^{\min}_c \approx 10^9$ PF-days，$\alpha_{C_{\min}} \approx 0.050$。

### 关键发现

| 发现 | 含义 | 后续修正 |
|------|------|---------|
| 模型规模比数据量更重要 | 有限预算下优先增大模型 | Chinchilla (2022) 修正为 1:1 |
| 幂律关系跨架构成立 | Transformer 和 LSTM 遵循相同规律 | — |
| 迁移学习也服从 scaling law | 下游任务性能同样可预测 | — |
| 最佳 batch size 随 loss 降低而增长 | 大模型需要更大 batch | — |

### 计算最优分配

论文推导了给定计算预算 $C$ 时的最优参数分配：

| 预算 | 最优参数量 | 最优数据量 | 比喻 |
|------|:-:|:-:|------|
| 小 (10²² FLOPs) | ~1B | ~150B tokens | Tiny 模型 |
| 中 (10²³ FLOPs) | ~10B | ~1.5T tokens | GPT-3 级别 |
| 大 (10²⁴ FLOPs) | ~100B | ~15T tokens | Llama 3 级别 |

**但论文当时的结论偏向了"优先增大模型"**，这一结论被 DeepMind 的 Chinchilla (2022) 修正——最优应该是模型和数据同比例增长。

### 涌现现象的框架

论文的理论框架解释了涌现现象：当参数量超过某个阈值时，模型在特定任务上的表现会出现"质的飞跃"。这是因为：

1. 某些能力要求模型容量超过特定下限
2. 模型学习能力的增长不是平滑的——某些任务的评估指标存在"触发的阈值"
3. 多任务训练中，大模型有机会学到小模型"忽略"的规律

## 三、关键实验与发现

### 跨规模预测

论文的核心实验方法：在小规模模型上拟合幂律，然后预测大规模模型的表现。例如，用 10⁸ 参数模型的实验结果预测 10⁹ 参数模型的表现，误差在 1-2% 以内。

```
Loss = 5.3 × N^(-0.076) + 1.7   (拟合结果随数据集不同)
```

### 重要观察

1. **Overfitting 完全由数据量决定**：只要 tokens/parameter > ~10²，模型就不会过拟合
2. **学习率调度对性能影响较小**只要总计算量固定：不同学习率策略最终收敛到相同的 loss
3. **模型深度比宽度更重要**（在一定程度上）：在同样参数量下，更深的模型略优
4. **大规模模型可以用小规模模型进行预实验**：这是论文最具实用价值的方法论贡献

### 对训练实践的影响

论文发现的最佳 batch size 随 loss 降低而增长：

$$ B_{\text{opt}} \propto \frac{1}{L} $$

这意味着训练后期需要使用更大的 batch 来维持计算效率。

## 四、局限性与后续影响

### 局限性

1. **"优先增大模型"结论被 Chinchilla 修正**：论文认为在固定计算预算下应偏向增大模型而非数据；Chinchilla 证明两者应等比放大
2. **实验架构有限**：主要基于 LSTM 和早期 Transformer（如 GPT-1 规模的模型），现代架构（MoE、GQA）可能有不同的 scaling 特性
3. **未考虑训练数据质量**：现实中的数据质量和多样性对性能有巨大影响，但论文使用统一的 WebText 数据
4. **仅研究自回归语言建模**：多模态、指令微调等更复杂的训练目标的 scaling 行为不同

### 后续影响

- **Chinchilla (2022)**：修正了最优数据-模型比例
- **PaLM、GPT-4、Llama 3**：将这些规律应用到万亿参数级别验证
- **涌现能力的理论化**：为后来 LLM 的涌现能力研究提供基础框架
- **Deepmind 的 Scaling Laws for Robot Learning (2024)**：将 scaling law 扩展到机器人学习领域

## 五、VLA/机器人研究中的角色

Scaling Laws 对 VLA 研究的指导作用贯穿始终：

- **RT-2 5B → 55B 的能力涌现**：RT-2 论文报告 55B 版本能完成符号理解、关系推理等任务，而 5B 版本完全做不到。这正是 scaling law 预言的"阈值现象"——超过某个规模后出现质的飞跃
- **OpenVLA 选择 7B 而非 1B**：Scaling law 直接解释了为什么 VLA 需要 7B 参数骨干——更大的模型在 OOD 泛化上优势显著
- **Open X-Embodiment** 的关键发现与 scaling law 一致：他们测试了从 35M 到 55B 的模型，发现"模型容量是决定 VLM 泛化性能的最关键因素"
- **Scaling Laws for Robot Learning (DeepMind, 2024)**：发现机器人的 scaling law 与 NLP 相似——模型大小、数据量、和环境多样性之间存在幂律关系

### 对 VLA 架构选择的直接影响

| 决策 | Scaling Law 的贡献 |
|------|-------------------|
| 为什么选 7B 骨干 | 1B 模型在复杂操控任务上性能不足 |
| 为什么需要大训练集 | 数据量减少直接导致 OOD 性能下降 |
| 为什么涌现 VLA | 规模到达阈值后，模型能学会"组合泛化" |
| 为什么 VLA 不用更小模型 | 小模型在机器人任务上存在能力天花板 |

## 六、对你的启示

1. **别在小于 3B 的模型上期待 VLA 的涌现能力**：如果你的实验目标包括"模型能否学会从未见过的物体操作"，7B 起步
2. **16GB GPU 可以做的事情**：在 7B 模型上做 LoRA 微调（不是 full-finetune）是可行的。能研究的是"给定固定骨干，VLA 如何更好地利用已有能力"
3. **数据量是关键瓶颈**：对个人研究者，scaling law 的核心启示是 **"数据质量比模型修改更重要"**——不要在改架构上花时间，在数据清洗和增强上投入
4. **用小模型做快速实验，用大模型做最终验证**：Scaling law 证明了在小规模上验证的结果可以外推到大规模
5. **理解 scaling law 帮助你避免"1B 模型跑通了但 7B 跑不通"的陷阱**：超参需要在不同规模上重新调优

## PDF

[[Scaling Laws 原文.pdf]]
