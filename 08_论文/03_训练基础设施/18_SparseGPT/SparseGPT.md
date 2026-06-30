---
tags:
  - 论文
  - 训练基础设施
  - 剪枝
  - 稀疏化
  - SparseGPT
created: 2026-06-30
paper_title: "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot"
paper_authors: "Elias Frantar, Dan Alistarh"
paper_year: 2023
paper_venue: "ICML 2023"
paper_citations: "~1,500+"
paper_url: "https://arxiv.org/abs/2301.00774"
github: "https://github.com/IST-DASLab/sparsegpt"
---

# SparseGPT

**SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot**
*Elias Frantar, Dan Alistarh | IST Austria | ICML 2023 | arXiv: 2301.00774*

> 无需任何微调，仅需一次前向传播即可将 GPT 系列模型剪枝至 50%-60% 非结构化稀疏度，且困惑度损失几乎为零。SparseGPT 首次证明了大规模预训练语言模型可以在**无反向传播**的前提下进行高精度剪枝，是 LLM 剪枝领域的奠基性工作。

---

## 一、Background / Core Idea

### 1.1 问题：大模型剪枝的困境

随着 LLM 规模跃升至 100B+ 参数，模型推理的计算和存储成本急剧增加。模型剪枝（Pruning）是一种经典的模型压缩技术，其目标是在移除大量权重后将精度损失降至最低。然而，传统剪枝方法面临根本性障碍：

- **训练后剪枝（Post-Training Pruning）的退化**：对 CNN 等小模型有效的 magnitude pruning（基于权重绝对值大小剪枝），在 LLM 上会导致灾难性的困惑度飙升。OPT-175B 在 50% 稀疏度下困惑度从 8.3 暴增至 10,000+
- **需要微调恢复精度**：传统迭代剪枝需要在剪枝后对模型进行大量微调（fine-tuning）来恢复精度。这在 LLM 时代面临严峻挑战——GPT-3 175B 的微调成本高达数百万美元
- **一次性剪枝（One-Shot Pruning）的失败史**：之前的一次性剪枝方法在 CNN 上有效，但在 LLM 上精度退化严重

### 1.2 核心洞察：权重重要性由 Layer-Wise Hessian 决定

论文的核心理论基础来自经典的最优脑损伤（Optimal Brain Damage / OBS, LeCun 1990）框架：

> 传统 magnitude pruning 的错误在于仅考虑单一权重的大小，忽视了**权重之间的相互关联性**。OBS 通过二阶泰勒展开考虑 Hessian 矩阵的逆来补偿剪枝影响。SparseGPT 将其扩展到 LLM 规模。

传统 OBS 框架计算一个权重 $w$ 移除后对损失 $L$ 的影响：

$$\delta L \approx \frac{w^2}{2[H^{-1}]_{ii}}, \quad H = \nabla^2 L$$

其中 $H$ 是 Hessian 矩阵。然而，对 GPT-175B 计算完整的 Hessian 逆矩阵在计算上不可行。

### 1.3 SparseGPT 的核心突破

SparseGPT 的理论突破在于三个关键洞察：

1. **按列独立求解**：将权重矩阵 $W$ 的列视为独立的剪枝问题。对于每列 $\mathbf{w}$，剪枝一组权重相当于求解如下问题：

$$\min_{\mathbf{w}} \| \mathbf{w}_{M} + \mathbf{H}_{M}^{-1} \mathbf{H}_{MM^{\prime}} \mathbf{w}_{M^{\prime}} \|_2^2$$

其中 $M$ 是保留（未剪枝）的权重集合，$M^{\prime}$ 是剪枝的权重集合。

2. **Transformer 的自回归结构提供"免费"的 Hessian**：对于自回归语言模型，每层的输入激活 $X$ 的 Gram 矩阵 $X^\top X$ 自然成为该层的 Hessian 近似。SparseGPT 利用这一 LLM 特有的结构，无需反向传播即可构造 Hessian 信息。

3. **Fisher 信息矩阵 + Hessian 的闭式求解**：通过 Cholesky 分解在 O(列数³) 时间内求得闭式解，而非传统 OBS 的 O(参数⁴) 迭代。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 算法框架

SparseGPT 的算法可以在单个 FP16 前向传播中完成，其核心步骤如下：

$$
\begin{aligned}
&\text{输入：权重矩阵 } W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}, \text{ 激活 } X \in \mathbb{R}^{n \times d_{\text{in}}} \\
&\text{1. 计算 Hessian: } H = X^\top X + \lambda I \\
&\text{2. 对每列 } \mathbf{w} \text{ 进行 Cholesky 分解: } H = LL^\top \\
&\text{3. 对每列求解闭式剪枝残差补偿: } \delta = -\frac{\mathbf{w}_i}{[L]_{ii}L_{i,i:}} \\
&\text{4. 更新权重: } W \leftarrow W + \delta
\end{aligned}
$$

### 2.2 逐层剪枝（Layer-Wise Pruning）

关键设计是按 Transformer 的**逐层结构**进行剪枝：

- 对每一层独立进行剪枝，在该层输入激活上计算 Hessian
- 剪枝后的误差通过权重更新自动传播到下一层
- 所有前向传播一次完成，无需在层之间进行反向传播

这与传统剪枝方法形成鲜明对比：传统方法需要全局优化或迭代训练，而 SparseGPT 逐层处理，将问题分解为多个独立的、可并行处理的子问题。

### 2.3 自适应掩码选择（Adaptive Mask Selection）

SparseGPT 不预设固定的剪枝比例，而是采用自适应策略：

1. **权重排序**：对每列的权重按重要性排序（重要性由 $w_i^2 / [H^{-1}]_{ii}$ 衡量）
2. **稀疏度分配**：保留最重要的权重，剪枝其余部分
3. **误差补偿**：通过闭式 OBS 更新补偿剪枝权重的贡献

该方法的优势在于自动识别哪些权重是关键的，即使总体稀疏度目标不变，不同层的实际剪枝比例也会自适应调整。

### 2.4 与 Magnitude Pruning 的关键对比

| 方面 | Magnitude Pruning | SparseGPT |
|------|:-:|:-:|
| 度量标准 | $\|w_i\|$（单个权重大小） | $w_i^2 / [H^{-1}]_{ii}$（Hessian 加权） |
| 误差补偿 | 无 | 闭式 OBS 补偿 |
| 列间依赖 | 忽略 | 通过 Cholesky 分解建模 |
| 是否需要数据 | 否 | 需要校准数据（128-512 样本） |
| OPT-175B 50% 稀疏度 PPL | $>10,000$ | 8.35（接近基线 8.27） |

---

## 三、Experiments and Key Findings

### 3.1 核心结果：困惑度对比

在 OPT-175B 上的非结构化剪枝结果：

| 稀疏度 | 未剪枝基线 | Magnitude | SparseGPT |
|:-:|:-:|:-:|:-:|
| 50% | 8.27 | 1.2e4 | **8.35** |
| 60% | 8.27 | 4.5e6 | **8.56** |
| 70% | 8.27 | 2.3e11 | **9.37** |

**SparseGPT 在 50% 稀疏度下困惑度仅增加 0.08，60% 时仅增加 0.29。** 而 magnitude pruning 在 50% 稀疏度时已完全崩溃。

### 3.2 跨模型规模的泛化

| 模型 | 基线 PPL | SparseGPT 50% | SparseGPT 60% |
|------|:-:|:-:|:-:|
| OPT-125M | 27.6 | 33.1 | 58.8 |
| OPT-1.3B | 15.2 | 16.5 | 18.6 |
| OPT-6.7B | 12.1 | 12.7 | 13.5 |
| OPT-30B | 10.7 | 11.0 | 11.4 |
| OPT-66B | 9.9 | 10.1 | 10.4 |
| OPT-175B | 8.3 | 8.4 | 8.6 |

**趋势**：模型越大，SparseGPT 的相对精度损失越小。这一发现支持了**大模型的"过度参数化"为其提供了剪枝冗余空间**的假设。

### 3.3 N:M 半结构化剪枝（2:4, 4:8）

SparseGPT 也支持 N:M 稀疏模式——这在 NVIDIA Ampere 架构的稀疏张量核心上可实现真正的推理加速：

| 模型 | Dense | SparseGPT 2:4 | Magnitude 2:4 |
|:-:|:-:|:-:|:-:|
| OPT-125M | 27.6 | 37.2 | 4.9e3 |
| OPT-1.3B | 15.2 | 16.8 | 1.2e4 |
| OPT-6.7B | 12.1 | 12.6 | 3.2e3 |
| OPT-175B | 8.3 | 8.5 | 1.1e6 |

**关键意义**：2:4 半结构化稀疏（50% 稀疏）在 A100 上可实现约 1.5-2x 的推理加速。SparseGPT 是唯一能在 LLM 上生成高质量 2:4 稀疏模式的方法。

### 3.4 校准数据需求

| 校准样本数 | 10 | 64 | 128 | 512 |
|:-:|:-:|:-:|:-:|:-:|
| OPT-175B PPL @ 50% | 8.52 | 8.37 | 8.35 | 8.35 |
| OPT-175B PPL @ 60% | 9.02 | 8.62 | 8.56 | 8.55 |

仅需 **128 个校准样本**即可达到接近最优的效果。这意味着 SparseGPT 几乎不需要额外数据——可以从原始训练集中随机取 128 个样本，或者使用通用文本。

### 3.5 计算开销

- **OPT-175B**: ~1 小时在单张 A100 (80GB) 上
- **OPT-66B**: ~25 分钟
- **OPT-6.7B**: ~3 分钟
- 内存开销：主要由 Hessian 矩阵（$d_{\text{hidden}} \times d_{\text{hidden}}$）决定，对 175B 约需 1-2GB 额外内存

---

## 四、Limitations and Challenges

1. **非结构化稀疏的硬件效率低**：SparseGPT 主要生成非结构化稀疏模式——权重矩阵中被剪枝的位置不规整。这类稀疏模式无法利用 NVIDIA Tensor Core 的稀疏加速（仅支持 2:4 结构化稀疏），因此虽然参数量减少，实际推理加速有限
2. **逐层误差累积**：虽然逐层剪枝在 50-60% 时表现良好，但超过 70% 稀疏度时误差迅速累积。SparseGPT 未能解决极端稀疏场景下的误差传播问题
3. **Hessian 近似依赖校准数据**：虽然只需 128 个样本，但校准数据的选择仍会影响最终剪枝质量。如果校准数据与下游任务的数据分布存在显著差异，剪枝质量可能下降
4. **对 LoRA 等 PEFT 模块未讨论**：论文仅对预训练基础模型进行剪枝，未讨论剪枝后的模型能否继续与 [[LoRA]] 等参数高效微调方法结合使用
5. **缺乏结构感知**：SparseGPT 的剪枝策略完全由权重和 Hessian 驱动，未考虑 Transformer 架构中的结构信息（如注意力头的重要性差异）
6. **Cholesky 分解的计算瓶颈**：当隐藏层维度过大（如 > 50k）时，Cholesky 分解的 O(d³) 计算复杂度成为瓶颈

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 SparseGPT 的关系 |
|---------|:----:|---------------------|
| **[[Wanda]]** (Sun et al.) | 2023 | 提出更简单的权重×激活范数剪枝度量，无需 Hessian 计算，速度比 SparseGPT 快数百倍 |
| **GPTQ** (Frantar et al.) | 2022 | 同一团队的量化工作，SparseGPT 借鉴了其逐层 Hessian 更新思想；两者可联合使用 |
| **SliceGPT** (Ashkboos et al.) | 2024 | 基于 PCA 的结构化剪枝，可删除整层/整块，更适合实际加速 |
| **Powerful Pruning** | 2024 | 结合蒸馏和剪枝的迭代方法，在极端稀疏度（80%+）下优于 SparseGPT |
| **Sheared LLaMA** (Xia et al.) | 2023 | 通过在训练中剪枝学习到结构化稀疏子网络 |
| **Compresso** (Guo et al.) | 2024 | 基于学习的方法自动搜索剪枝比例，比 SparseGPT 的手工设定更灵活 |

**影响评估**：SparseGPT 是 LLM 剪枝领域的**里程碑式工作**。它首次证明了后训练一次性剪枝可以在 LLM 上达到可用精度，开创了"无微调剪枝"的研究范式。但后续工作（特别是 [[Wanda]]）证明了更简单方法的存在，而 SparseGPT 的更精确的 Hessian 剪枝在某些场景下仍是最优的。

---

## 六、Implications for You / Hardware Compatibility

### 推理加速与显存

| 配置 | 推理速度 | 显存节省 | 兼容性 |
|------|:-:|:-:|:------|
| 50% 非结构化稀疏，CPU 推理 | ~1.1-1.5x | ~1.8x | ✅ 通用的参数减少，任何 CPU 推理库均受益 |
| 50% 非结构化稀疏，GPU 推理 | ~1.0-1.1x | ~1.8x | ⚠️ 非结构化稀疏在 GPU 上几乎无加速（内存带宽瓶颈） |
| 2:4 半结构化稀疏，A100 GPU | ~1.5-2.0x | ~1.8x | ⚠️ 仅 A100/H100 支持 N:M 稀疏张量核心 |
| 2:4 半结构化稀疏，消费级 GPU | ~1.0x | ~1.8x | ❌ RTX 4090/3090 不支持 N:M 稀疏加速 |
| 60% 非结构化稀疏 + INT8 量化 | ~1.8-2.5x | ~4x | ⚠️ 需要 [[GPTQ]] 或类似量化框架支持 |

### 实际使用建议

- **最佳使用场景**：用于减少模型存储（checkpoint 大小），或在 CPU 部署时降低推理延迟。SparseGPT + INT8 量化的组合可以同时利用剪枝和量化的正交优势
- **消费级 GPU 的局限**：如果主要使用 RTX 3090/4090 进行推理，SparseGPT 的优势主要集中在存储减少而非推理加速。非结构化稀疏无法利用 Tensor Core，实际推理延迟可能降低很少
- **与 [[LoRA]] 的结合**：可以对剪枝后的模型应用 LoRA 微调，但这方面的研究尚未充分展开。一个潜在的陷阱是剪枝可能移除了对下游任务重要的参数
- **作为预压缩步骤**：在部署 LLM 到边缘设备时，可以先使用 SparseGPT 压缩（50% 稀疏度），再使用 [[GPTQ]] 量化到 4-bit，最终模型大小约为原始的 1/8
- **校准数据管理**：使用与目标领域相关的 128 个样本作为校准集，可以提升剪枝后模型在特定下游任务上的表现

### 硬件兼容性总结
- ✅ CPU 推理 50% 非结构化稀疏：通用兼容
- ⚠️ GPU 推理 50% 非结构化稀疏：无实际加速（仅节省显存）
- ⚠️ A100/H100 2:4 半结构化稀疏：实际加速 1.5-2x
- ❌ 消费级 GPU N:M 稀疏加速：不支持

## PDF

[[SparseGPT.pdf]]
