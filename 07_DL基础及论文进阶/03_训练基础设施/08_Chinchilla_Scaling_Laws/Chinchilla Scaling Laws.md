---
tags:
  - 论文
  - 训练基础设施
  - 缩放定律
  - 计算优化
  - DeepMind
created: 2026-06-30
paper_title: "Training Compute-Optimal Large Language Models"
paper_authors: "Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, Laurent Sifre, Laurent"
paper_year: 2022
paper_venue: "NeurIPS 2022"
paper_citations: "~8,000+"
paper_url: "https://arxiv.org/abs/2203.15556"
github: ""
---

# Chinchilla Scaling Laws

**Training Compute-Optimal Large Language Models**
*Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch et al. | DeepMind | NeurIPS 2022 | arXiv: 2203.15556*

> 推翻 OpenAI Kaplan 等人（2020）的"模型规模唯一论"，证明在固定算力预算下，模型参数和训练数据量应**等比缩放**——多数模型被严重过训练（undertrained）。Chinchilla 70B 在 1.4T tokens 上训练，以同算力预算超越 Gopher 280B。这一发现从根本上改变了整个行业的 LLM 训练策略。

---

## 一、Background / Core Idea

### 1.1 问题：Kaplan Scaling Laws 主导下的行业实践

2020 年，OpenAI 的 Kaplan et al. 提出 Scaling Laws，核心结论：

$$L(N) \propto N^{-\alpha_N}, \quad L(D) \propto D^{-\alpha_D}$$

- 增大模型参数 $N$ 带来的 loss 下降大于增大数据 $D$
- **模型规模越大越好，数据相对不重要**

这一结论直接推动 GPT-3 (175B/300B tokens)、Gopher (280B/300B tokens) 等"大参数、小数据"的模型——参数量和训练数据量比例约为 **1:1.7**（GPT-3）。

### 1.2 核心洞察：Kaplan 定律的假设缺陷

Chinchilla 论文通过系统的计算-最优分析，指出 Kaplan 定律的三个问题：

1. **Kaplan 的测试 range 太窄**：仅测试了 $N \in [7\ 10^7]$ 到 $[1.5 \times 10^9]$ 参数量范围，数据量 $D$ 固定少量
2. **过小计算预算下的外推**：Kaplan 在小模型上推导的规律外推到大规模时失效
3. **固定 $\text{FLOPs}$ 假设模糊**：未明确定义"最优"是指给定 FLOPs 预算下 loss 最小化

**Chinchilla 重定义问题**：

> 给定固定计算预算 $C$（FLOPs），$\min_{N,D} L(N, D) \quad \text{s.t.} \quad \text{FLOPs}(N, D) = C$

### 1.3 核心矛盾：Token 不足的行业现实

| 模型 | 参数量 | 训练 tokens | Kaplan 建议 | Chinchilla 建议 |
|:----|:-----:|:-----------:|:-----------:|:---------------:|
| GPT-3 | 175B | 300B | 合理 | **严重不足（需 3.3T）** |
| Gopher | 280B | 300B | 模型偏大 | **严重不足（需 5.2T）** |
| OPT | 175B | 300B | 合理 | **严重不足（需 3.3T）** |
| **Chinchilla** | **70B** | **1.4T** | 数据过量 | **计算最优** |
| LLaMA 1 | 65B | 1.4T | — | **接近最优** |

---

## 二、Method / Architecture / Technical Contribution

### 2.1 三种分析方法（Three Approaches）

论文使用三种独立方法推导最优参数/数据比例，互相验证：

#### Approach 1：固定模型规模，变化数据量

在三组模型大小（70M, 300M, 1.5B, 6B）上训练不同 token 数，拟合得到：

$$L(N, D) = \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}} + 1.69$$

**关键推导**：给定 FLOPs 预算 $C$，最优参数分配：

$$N_{\text{opt}}(C) \propto C^{0.46}, \quad D_{\text{opt}}(C) \propto C^{0.54}$$

#### Approach 2：固定 FLOPs 预算，变化参数量

对不同的 FLOPs 预算（$10^{17}$ 到 $10^{21}$），固定计算量并搜索最优参数：

| FLOPs 预算 | $\log C$ | 最优参数量 $N_{\text{opt}}$ | 所需 tokens $D_{\text{opt}}$ | 比例 $D/N$ |
|:----------:|:--------:|:--------------------------:|:---------------------------:|:----------:|
| $10^{17}$ | 17.0 | 19.1M | 20.9B | ~1094 |
| $10^{18}$ | 18.0 | 52.7M | 75.7B | ~1436 |
| $10^{19}$ | 19.0 | 145.7M | 274.1B | ~1880 |
| $10^{20}$ | 20.0 | 402.1M | 993.2B | ~2470 |
| **$10^{21}$** | **21.0** | **1.1B** | **3.6T** | **~3273** |
| **$10^{24}$ (GPT-3 级别)** | **24.0** | **24.8B** | **171.6T** | **~6920** |

#### Approach 3：直接拟合参数—数据联合 loss

通过 IsoFLOPs 曲线（固定 FLOPs 下 loss vs. 参数量的 U 型曲线）直接寻找最优：

```
Loss (IsoFLOPs 曲线示意):
            ┌─────────── U 型曲线 ───────────┐
低 FLOPs  ──┤     ●（最优参数-数据组合）        ├── 低 loss
            │   ●         ●                   │
            │  ●              ●               │
高 FLOPs  ──┤ ●                   ●           ├── 高 loss
            └───────────────────────────────────┘
                  参数量 N →
```

### 2.2 核心结果：Chinchilla 定律

三种独立方法得到**高度一致的结论**：

$$\boxed{N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}}$$

> **对于计算最优训练，模型参数每增加 1 倍，训练数据也需增加 1 倍。**
> 即：训练 tokens 数 ≈ **20 × 模型参数量**

### 2.3 Chinchilla 70B 模型

基于上述定律，DeepMind 在 280B Gopher 相同的 FLOPs 预算下，训练了 Chinchilla 70B：

| 属性 | Gopher | **Chinchilla** | 变化 |
|:----|:-----:|:--------------:|:----:|
| 参数量 | 280B | **70B** | **4× 缩小** |
| 训练 tokens | 300B | **1.4T** | **4.7× 放大** |
| 总 FLOPs | $5.76 \times 10^{23}$ | $5.76 \times 10^{23}$ | **相同** |
| 推理成本 | 1× | **~1/4** | 更便宜 |
| 微调成本 | 1× | **~1/4** | 更方便 |

---

## 三、Experiments and Key Findings

### 3.1 Chinchilla vs Gopher 全面对比

| 评测基准 | Gopher (280B) | **Chinchilla (70B)** | 改善 |
|:---------|:------------:|:--------------------:|:----:|
| MMLU (0-shot) | 60.0% | **67.6%** | +7.6% |
| MMLU (5-shot) | 63.6% | **67.5%** | +3.9% |
| LAMBADA (zero-shot) | 73.4% | **76.3%** | +2.9% |
| LAMBADA (5-shot) | 74.5% | **78.9%** | +4.4% |
| RACE-h (5-shot) | 47.4% | **50.1%** | +2.7% |
| RACE-m (5-shot) | 76.0% | **79.2%** | +3.2% |
| PIQA (0-shot) | 81.1% | **81.8%** | +0.7% |
| HellaSwag (0-shot) | 79.0% | **80.8%** | +1.8% |
| **MMLU 平均** | 60.0% | **67.5%** | **+7.5%** |

**Chinchilla 70B 在几乎所有基准上超越 Gopher 280B**，且推理代价降低 4 倍。

### 3.2 对已有模型的评估

论文用 Chinchilla 定律重新评估已有模型的训练充分度：

| 模型 | 参数量 | 训练 tokens | 理论上所需 tokens | 欠训练程度 |
|:----|:-----:|:-----------:|:----------------:|:----------:|
| GPT-3 | 175B | 300B | 3.5T | **91% 欠训练** |
| Gopher | 280B | 300B | 5.6T | **95% 欠训练** |
| OPT | 175B | 180B | 3.5T | **95% 欠训练** |
| Jurassic-1 | 178B | 300B | 3.6T | **92% 欠训练** |
| **Chinchilla** | **70B** | **1.4T** | **1.4T** | **✅ 计算最优** |

### 3.3 Chinchilla 的其他关键发现

1. **小模型 + 大数据 > 大模型 + 小数据**：相同 FLOPs 下，Chinchilla 70B 全面优于 Gopher 280B
2. **少量数据过拟合的代价**：在 300B tokens 上训练 280B Gopher 导致严重过拟合（Gopher 训练 loss 高于 Chinchilla 验证 loss）
3. **推理收益**：小模型在同等服务质量下推理更便宜——Chinchilla 的推理成本仅为 Gopher 的 1/4

---

## 四、Limitations and Challenges

1. **仅考虑预训练 FLOPs**：Chinchilla 定律只优化预训练计算效率，忽略了微调、推理、数据收集的完整成本
2. **训练数据的质量未建模**：Gemini、Llama 3 等后续实践表明，数据质量至少和数据量同等重要。Chinchilla 定律假设训练数据无限且同质
3. **与 Kaplan 定律的矛盾**：两种 Scaling Law 各自建立在不同的实验设置上，缺乏统一的解释框架
4. **$N_{\text{opt}} \propto C^{0.5}$ 的前提假设**：该结论高度依赖模型架构（Transformer+Adam），对其他架构（如 Mamba、Mixture-of-Experts）可能不同
5. **未考虑退化/涌现等非线性现象**：过小的模型（<1B）即使数据最优，也无法涌现某些能力
6. **下游任务感知缺失**：最优的预训练配置不一定是下游任务性能最优——例如更大的模型在少量样本学习上可能更好

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Chinchilla 的关系 |
|---------|:----:|---------------------|
| **LLaMA 1** (Meta, Touvron et al.) | 2023 | 直接受 Chinchilla 定律指导：7B/2T, 13B/2T, 65B/1.4T |
| **LLaMA 2** (Meta) | 2023 | 进一步增加 tokens（7B/2T → 改进数据质量） |
| **LLaMA 3** (Meta) | 2024 | 15T tokens 训练 8B/70B/405B，远超 Chinchilla 建议 |
| **GPT-4** (OpenAI) | 2023 | 据传使用"小模型 + 海量数据"策略（受 Chinchilla 影响） |
| **Gemma** (Google) | 2024 | 7B/2T, 2B/2T——完全按照 Chinchilla 最优比例 |
| **Mistral 7B** (Mistral AI) | 2023 | 7B/~8T——大幅超过 Chinchilla 建议 |
| **Qwen 72B** (Alibaba) | 2023 | 72B/3T——接近但略少于最优（需 1.4T） |
| **Scaling Data-Constrained LLMs** (Muennighoff et al.) | 2023 | 研究数据受限时的 Scaling Law，数据重复的收益 |
| **Scaling Laws for MoE** (Clark et al.) | 2022 | 将 Chinchilla 定律扩展到混合专家模型 |

**影响评估**：Chinchilla 定律是 LLM 训练策略领域**最被引用的工作之一**。在 Chinchilla 之前，行业的隐含假设是"越大越好"；Chinchilla 之后，"数据量"被正式提升到与"模型参数"同等重要的地位。Llama 1/2、Gemma、Falcon、Qwen 等几乎所有开源模型都直接采用 Chinchilla 建议的比例。

---

## 六、Implications for You / Hardware Compatibility

### 计算资源估算（基于 Chinchilla 定律）

| 参数量 | 所需 tokens（20×） | 总 FLOPs | 推荐硬件 | 训练时间估计 |
|:-----:|:----------------:|:--------:|:--------|:-----------:|
| 1B | 20B | $1.2 \times 10^{20}$ | 8× A100 (40GB) | ~1 天 |
| 7B | 140B | $2.8 \times 10^{21}$ | 64× A100 (80GB) | ~7 天 |
| 13B | 260B | $1.0 \times 10^{22}$ | 128× A100 (80GB) | ~10 天 |
| 70B | 1.4T | $2.9 \times 10^{23}$ | 512× A100 (80GB) | ~20 天 |
| 405B | 8.1T | $1.0 \times 10^{25}$ | 16,384× H100 | ~35 天 |

### 对训练策略的指导

- **不要在大模型上过度投资数据**：如果只有 500B tokens 数据，不应训练 70B 模型（只需 ~25B 参数就足够）
- **数据质量优先于重复**：当数据量不足 20× 参数时优先提升数据质量（去重、过滤），而非重复使用
- **MoE 的 Scaling Law 不同**：对于 MoE 模型，有效参数量不等于激活参数量，Chinchilla 公式需调整
- **推理成本是隐藏变量**：如果服务的推理负载很高，用小模型训练更多数据比大模型更经济

### 对资源受限场景的启示

| 约束场景 | 策略 | 与 Chinchilla 的关系 |
|:---------|:----|:--------------------|
| GPU 少但数据多 | 小模型 + 大量数据（完全遵循 Chinchilla） | ✅ 推荐 |
| GPU 多但数据少 | 中等模型 + 数据增强 + 重复使用 | ⚠️ 违背 Chinchilla，但别无选择 |
| 快速原型 | 极小模型（<1B）+ 少量数据做缩放预测 | ✅ 遵循 Chinchilla 推导 FLOPs |
| 极致质量导向 | 大模型 + 数据质量过滤 + 有监督微调 | ⚠️ Chinchilla 不覆盖下游 |
| 推理成本敏感 | 优选 缩小模型 + 等比放大数据 | ✅ Chinchilla 暗示推理收益 |

### 硬件兼容性总结
- ✅ 小参数 Chinchilla 最优模型（<7B）：消费级 GPU（3090/4090）可训练
- ⚠️ 标准 Chinchilla 最优模型（70B）：仅 A100/H100 集群（64+ GPU）
- ❌ 超大规模 Chinchilla 最优（>400B）：仅超算中心/H100 集群（1000+ GPU）
- ✅ Chinchilla 推理更友好：同等质量下 70B 推理优于 280B，更适合消费级部署

---

## PDF

[[Chinchilla Scaling Laws 原文.pdf]]
