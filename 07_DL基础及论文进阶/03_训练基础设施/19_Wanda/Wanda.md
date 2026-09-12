---
tags:
  - 论文
  - 训练基础设施
  - 剪枝
  - Wanda
  - 非结构化剪枝
created: 2026-06-30
paper_title: "A Simple and Effective Pruning Approach for Large Language Models"
paper_authors: "Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter"
paper_year: 2023
paper_venue: "arXiv preprint"
paper_citations: "~800+"
paper_url: "https://arxiv.org/abs/2306.11695"
github: "https://github.com/locuslab/wanda"
---

# Wanda

**A Simple and Effective Pruning Approach for Large Language Models**
*Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter | CMU & Meta AI | arXiv: 2306.11695*

> 将 SparseGPT 的 OBS 剪枝公式在"所有权重交互被忽略"这个极端简化下进行推导，恰好得到"权重 × 输入激活列范数"的剪枝度量——Wanda。无需 Hessian、无需迭代、比 SparseGPT 快数百倍，同时保持了相近的剪枝质量。

---

## 一、Background / Core Idea

### 1.1 问题：剪枝方法的复杂度与可扩展性

[[SparseGPT]] 展示了 LLM 可以在无需微调的情况下通过一次性剪枝达到高精度，但其算法相对复杂：需要逐层计算 Hessian 矩阵的 Cholesky 分解、跟踪权重更新等。这引出一个自然的问题：

> SparseGPT 的精度提升究竟来自其精妙的 OBS 误差补偿机制，还是仅仅来自更好的**重要性度量**？

### 1.2 核心洞察：OBS 框架在"无补偿"假设下退化为简单度量

Wanda 的核心思想从 SparseGPT 的 OBS 推导出发，但做一个**极端的近似假设**——不考虑权重移除后的补偿（即假设 $H$ 是对角矩阵）：

$$
\begin{aligned}
\text{SparseGPT:} \quad & \text{Importance}(w) = \frac{w^2}{[H^{-1}]_{ii}} \\
\text{Wanda:} \quad & \text{Importance}(w_{ij}) = |w_{ij}| \cdot \|x_j\|_2
\end{aligned}
$$

其中 $x_j$ 是第 $j$ 个输入特征的列向量（在 batch 维度上）。这一度量有以下解释：

1. **权重绝对值 $|w|$**：衡量权重本身的大小（与 magnitude pruning 相同）
2. **输入激活的列范数 $\|x\|_2$**：衡量该特征在正向传播中的"重要性"——如果一个特征在激活中范数很小（即该通道几乎从不激活），那么对应的权重自然不太重要

### 1.3 与 SparseGPT 的关系

Wanda 可以理解为 SparseGPT 在以下两种简化下的极限情况：

| 方面 | [[SparseGPT]] | **Wanda** |
|------|:-:|:-:|
| 重要性度量 | $w^2 / [H^{-1}]_{ii}$ | $\|w\| \cdot \|x\|_2$ |
| 权重补偿 | 闭式 OBS 补偿更新 | **无补偿** |
| 层内依赖建模 | 完整的 Cholesky 分解 | 忽略 |
| 计算复杂度 (d 列, n 样本) | O(d³ + nd²) | O(nd) |
| OPT-175B 运行时间 | ~1 小时 | ~5 秒 |

### 1.4 Wanda 度量的直觉理解

Wanda 度量可以拆解为两个正交的因子：

- **权重大小 ($|w|$)**：大型权重与输出有更大的相关性，移除后对输出影响更大
- **激活范数 ($\|x\|_2$)**：如果某输入通道的激活值在整个 batch 上的 L2 范数很小，说明该通道在模型推理中**很少"说话"**——修剪其权重的影响较小

两者的乘积形成了一种"输入感知的权重大小"：它不仅问"这个权重有多大"，还问"这个权重接收的信号有多强"。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 算法步骤

Wanda 的算法极其简洁：

$$
\begin{aligned}
&\text{输入：权重矩阵 } W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}, \text{ 激活 } X \in \mathbb{R}^{n \times d_{\text{in}}} \\
&\text{1. 对每列计算 L2 范数: } \|x_j\|_2 = \sqrt{\sum_{i=1}^n X_{ij}^2} \\
&\text{2. 对每列计算 Wanda 分数: } s_{ij} = |W_{ij}| \cdot \|x_j\|_2 \\
&\text{3. 对每列保留分数最大的 } k \text{ 个权重，其余剪枝}
\end{aligned}
$$

**关键**：剪枝是在每列（per-channel）粒度进行的——对每个输出神经元（列），保留该列内 Wanda 分数最大的 top-k 权重，剪枝其余的。这与按全局剪枝（全矩阵保留 top-k%）的策略有本质区别，因为全局剪枝可能导致某些列被完全消除。

### 2.2 逐列剪枝 vs 全局剪枝

Wanda 采用**逐列剪枝**（column-wise / per-output pruning）：

- 对权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 的每一列（对应一个输出神经元），分别进行剪枝
- 每个输出神经元保留固定比例的权重
- 保证了**每个输出神经元输入权重的稀疏度一致**

这与 SparseGPT 的列级 OBS 推导一致，但极大地简化了计算。

### 2.3 N:M 半结构化稀疏的自适应扩展

Wanda 将其剪枝策略自然地扩展到 N:M 稀疏模式：

- 对每 $M$ 个连续的权重，仅保留 Wanda 分数最高的 $N$ 个
- 在 OPT-175B 上实现了接近 SparseGPT 的 2:4 剪枝质量

### 2.4 激活范数计算的技巧

实际实现中，Wanda 在校准集上收集激活值的列范数：

$$
\|x_j\|_2 = \sqrt{\sum_{\text{sample}=1}^n \sum_{\text{token}=1}^t (x_{\text{sample,token},j})^2}
$$

其中 $n$ 是校准样本数（通常 128），$t$ 是每样本的 token 数。这一统计量可以在一次前向传播中作为一个 hook 轻松收集，无需额外计算开销。

---

## 三、Experiments and Key Findings

### 3.1 与 SparseGPT 的精度对比（困惑度）

| 模型 | Dense | SparseGPT 50% | Wanda 50% | SparseGPT 60% | Wanda 60% |
|------|:-:|:-:|:-:|:-:|:-:|
| OPT-125M | 27.6 | 33.1 | 33.9 | 58.8 | 71.9 |
| OPT-1.3B | 15.2 | 16.5 | 16.7 | 18.6 | 19.5 |
| OPT-6.7B | 12.1 | 12.7 | 12.8 | 13.5 | 14.1 |
| OPT-30B | 10.7 | 11.0 | 11.1 | 11.4 | 11.7 |
| OPT-66B | 9.9 | 10.1 | 10.2 | 10.4 | 10.7 |
| OPT-175B | 8.3 | 8.4 | 8.4 | 8.6 | 8.8 |

**核心发现**：Wanda 在 50% 稀疏度下与 SparseGPT 几乎无法区分（差距 < 0.1 PPL），在 60% 时差距稍大但仍可接受（~0.2-0.5 PPL）。考虑到 Wanda 比 SparseGPT 快 **700 倍**（OPT-175B: 5 秒 vs 1 小时），这一精度-速度权衡极具吸引力。

### 3.2 与 Magnitude Pruning 的对比

| 方法 | OPT-6.7B @ 50% | OPT-6.7B @ 60% |
|------|:-:|:-:|
| Dense (PPL) | 12.1 | 12.1 |
| Magnitude | 2.0e3 | 5.2e5 |
| Magnitude (每列) | 26.5 | 223.1 |
| **Wanda** | **12.8** | **14.1** |

**关键**：即使是列级 magnitude（每列保留最大权重），在 60% 时 PPL 也暴增至 223.1。Wanda 的激活范数信息带来了 **15x+ 的精度优势**。

### 3.3 LLaMA 系列上的结果

| 模型 | Dense | Magnitude 50% | Wanda 50% | SparseGPT 50% |
|------|:-:|:-:|:-:|:-:|
| LLaMA-7B | 5.68 | 3.4e3 | **6.01** | 5.99 |
| LLaMA-13B | 5.09 | 5.4e2 | **5.36** | 5.35 |
| LLaMA-30B | 4.10 | 4.6e2 | **4.51** | 4.51 |
| LLaMA-65B | 3.53 | 9.6e2 | **3.90** | 3.88 |

**发现**：LLaMA 系列对剪枝的鲁棒性优于 OPT 系列（LLaMA 在 Wanda 50% 下 PPL 从 5.68 升至 6.01，而 OPT-125M 从 27.6 升至 33.9）。这暗示 LLaMA 的训练方式（数据质量、训练时长等）产生了更具冗余性的权重分布。

### 3.4 半结构化 N:M 稀疏

| 模型 | SparseGPT 2:4 | Wanda 2:4 | SparseGPT 4:8 | Wanda 4:8 |
|------|:-:|:-:|:-:|:-:|
| LLaMA-7B | 7.23 | 7.34 | 7.93 | 8.14 |
| LLaMA-13B | 6.13 | 6.16 | 6.48 | 6.54 |
| LLaMA-30B | 4.99 | 5.08 | 5.24 | 5.40 |
| LLaMA-65B | 4.13 | 4.31 | 4.64 | 4.78 |

在 2:4 模式下，Wanda 与 SparseGPT 的差距在 0.1-0.2 PPL 范围内；在更严格的 4:8 模式下差距稍增大。

### 3.5 零样本下游任务

| 模型 | 方法 | PIQA | HellaSwag | ARC-e | ARC-c | BoolQ | 平均 |
|------|------|:-:|:-:|:-:|:-:|:-:|:-:|
| LLaMA-7B | Dense | 78.1 | 73.0 | 52.4 | 41.4 | 73.3 | **63.6** |
| LLaMA-7B | Wanda 50% | 75.9 | 68.6 | 51.4 | 36.9 | 66.5 | **59.9** |
| LLaMA-7B | SparseGPT 50% | 76.5 | 68.2 | 50.8 | 37.1 | 66.1 | **59.7** |
| LLaMA-7B | Magnitude 50% | 60.2 | 35.4 | 30.4 | 22.3 | 52.7 | **40.2** |

在零样本任务上，Wanda 和 SparseGPT 几乎相同，而 magnitude pruning 的准确率大幅下降（差距高达 20 个百分点）。

---

## 四、Limitations and Challenges

1. **非结构化稀疏的硬件效率瓶颈**：与 [[SparseGPT]] 相同，Wanda 主要生成非结构化稀疏模式，在 GPU 推理时几乎无法实现实际加速。仅在 2:4 模式下可受益于 N:M 稀疏核心
2. **60%+ 稀疏度时的精度退化**：当目标稀疏度超过 60% 时，Wanda 与 SparseGPT 的差距开始显著扩大。在 70% 时 Wanda 的 PPL 已经明显劣于 SparseGPT，说明在极端稀疏场景下 OBS 的误差补偿仍是必要的
3. **"无补偿"假设的固有限制**：Wanda 完全忽略权重移除后的残差补偿，这在高稀疏度场景下意味着移除的权重的贡献被彻底丢弃，没有通过调整剩余权重来弥补
4. **仅依赖校准集的正向传播**：校准集的质量和数量直接决定激活范数的准确性。如果校准集与推理数据分布存在偏移，Wanda 度量的有效性可能会下降
5. **缺乏结构化约束**：Wanda 不考虑 Transformer 的结构化信息（头注意力、FFN 层内维度分组等），可能导致某些功能性单元被破坏
6. **无稀疏度自适应分配**：Wanda 对所有层和所有输出神经元施加相同的稀疏度，未考虑不同层对剪枝的敏感度差异（而 SparseGPT 在 A100 上可通过 OBS 分配自适应学习）

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 Wanda 的关系 |
|---------|:----:|----------------|
| **[[SparseGPT]]** (Frantar & Alistarh) | 2023 | Wanda 的起点；Wanda 证明了 SparseGPT 的精度优势主要来自激活感知的度量而非 OBS 补偿 |
| **SliceGPT** (Ashkboos et al.) | 2024 | 利用 PCA 对 LLM 进行结构化剪枝；与 Wanda 的非结构化路线互补 |
| **Sparsegpt + Wanda 联合** | 2023 | 社区实践表明：在 <60% 稀疏度时用 Wanda（更快），在 >60% 或高精度需求场景用 SparseGPT |
| **Powerful Pruning** (Sun et al.) | 2024 | 提出"重参化"剪枝——将权重参数化为幅度和方向，结合 Wanda 度量扩展 |
| **Sheared LLaMA** (Xia et al.) | 2023 | 从 7B 剪枝出 2.7B 规整稀疏子网络，剪枝方法结合了类似 Wanda 的度量 |

**影响评估**：Wanda 以其**极致的简洁性**在 LLM 剪枝社区获得了广泛关注。它证明了顶尖的剪枝效果不一定需要复杂算法——对绝大多数实际场景（50-60% 稀疏度），Wanda 提供了与 [[SparseGPT]] 几乎相同的质量，但运行时间从小时级降至秒级。这一工作极大地降低了 LLM 剪枝的准入门槛，使剪枝从专门工具包的"高级功能"变成了普通研究者也能轻松尝试的常规操作。

---

## 六、Implications for You / Hardware Compatibility

### 计算资源需求比较

| 方法 | 70B 模型运行时间 | 所需 GPU 显存 | 是否需要多 GPU |
|------|:-:|:-:|:-:|
| [[SparseGPT]] | ~25 分钟 | ~2-4GB 额外空间 | ✅ 单卡 A100 |
| **Wanda** | **~1 分钟** | **~0.5-1GB 额外空间** | ✅ 单卡任意 GPU |
| Magnitude Pruning | ~1 秒 | 0GB | ✅ 无需 GPU |

### 实际使用建议

- **首选的快速剪枝方案**：对 50-60% 稀疏度的场景，Wanda 是精度-时间最优权衡。能在几分钟内在单张消费级 GPU 上对 7B-13B 模型完成剪枝
- **与 [[LoRA]] 结合**：可以先用 Wanda 对模型剪枝到 50% 稀疏度（减少存储和推理计算），再用 LoRA 微调下游任务。这种"剪枝 + 微调"组合在边缘设备部署场景中极具价值
- **对校准数据敏感**：建议使用与下游任务分布相近的校准数据（128 样本足矣）。通用文本（如 C4 数据集）也是可接受的默认选择
- **剪枝 + 量化配合**：Wanda 剪枝 + [[GPTQ]] 量化是正交的压缩策略，可以叠加使用。先剪枝再量化的组合可以压缩单个权重从 FP16 到 4-bit，但从 `2×` 个参数角度来看，剪枝在移除参数的同时破坏模型结构，与量化的精度损失可能存在非加性交互
- **RTX 4090 推理注意**：Wanda 的非结构化剪枝在消费级 GPU 推理时几乎无加速效果。如果想要实际的推理加速，应转向 2:4 结构化稀疏（需 A100/H100）或使用 [[SliceGPT]] 进行结构化剪枝

### 硬件兼容性总结
- ✅ 单卡任意 GPU（甚至 CPU）执行 Wanda 剪枝：计算极轻量
- ⚠️ 非结构化稀疏在 NVIDIA GPU 推理：仅节省显存，无延迟收益
- ⚠️ A100/H100 上的 2:4 稀疏推理：实际加速 1.5-2x
- ❌ 消费级 GPU 上的推理加速：不支持
- ✅ Wanda 剪枝 + [[LoRA]] 微调：社区验证的高效方案

## PDF

[[Wanda 原文.pdf]]
