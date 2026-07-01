---
tags:
  - 论文
  - 训练基础设施
  - 优化器
  - 权重衰减
  - Adam
created: 2026-06-30
paper_title: "Decoupled Weight Decay Regularization"
paper_authors: "Ilya Loshchilov, Frank Hutter"
paper_year: 2019
paper_venue: "ICLR 2019"
paper_citations: "~12,000+"
paper_url: "https://arxiv.org/abs/1711.05101"
github: ""
---

# AdamW

**Decoupled Weight Decay Regularization**
*Ilya Loshchilov, Frank Hutter | University of Freiburg | ICLR 2019 | arXiv: 1711.05101*

> 将权重衰减（weight decay）从 Adam 优化器的自适应梯度更新中解耦，纠正了 L2 正则化与自适应学习率之间的根本冲突。AdamW 已成为 GPT-3、LLaMA、Chinchilla 乃至几乎所有现代 LLM 训练的事实标准优化器。

---

## 一、Background / Core Idea

### 1.1 问题：Adam 中 L2 正则化的失效

在 SGD 中，L2 正则化（权重衰减）等价于在每个 step 从权重中减去 $\eta \lambda w_t$：

$$w_{t+1} = w_t - \eta \nabla L(w_t) - \eta \lambda w_t$$

但在 Adam 中，梯度更新被自适应学习率 $v_t = \text{Var}[g]$ 归一化：

$$w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t + \epsilon}} - \eta \lambda w_t \quad \text{(错误——传统 Adam L2)}$$

**问题**：L2 正则化的梯度 $w_t$ 也被 Adam 的自适应学习率缩放。当某一维度的梯度方差 $v_t$ 很小（即该权重变化不频繁）时，$\frac{1}{\sqrt{v_t}}$ 因子会**放大**正则化惩罚——导致该维度被过度衰减。

**形式化**：

标准 Adam 的 L2 正则化实际等价于：

$$\Delta w_t = -\eta \frac{m_t}{\sqrt{v_t + \epsilon}} - \underbrace{\eta \lambda \frac{w_t}{\sqrt{v_t + \epsilon}}}_{\text{耦合的权重衰减}}$$

$\frac{w_t}{\sqrt{v_t + \epsilon}}$ 项意味着：权重衰减率**不是常数**，而是随梯度方差变化——这与理想的正则化语义不符。

### 1.2 核心洞察：权重衰减 ≠ L2 正则化（在 Adam 中）

| 优化器 | L2 正则化 | 权重衰减 | 等价？ |
|:------|:---------:|:--------:|:------:|
| SGD | $w = w - \eta\nabla L - \eta\lambda w$ | $w = w - \eta\nabla L - \eta\lambda w$ | ✅ **完全等价** |
| **Adam (传统)** | $w = w - \eta \frac{m}{\sqrt{v+\epsilon}} - \eta\lambda\frac{w}{\sqrt{v+\epsilon}}$ | 无法实现 | ❌ **不等价** |

**AdamW 的修正**：将权重衰减步骤从梯度更新中分离，直接在参数上执行：

$$\boxed{w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t + \epsilon}} - \eta \lambda w_t}$$

核心区别在于：$\eta \lambda w_t$ 不再被 $\frac{1}{\sqrt{v_t + \epsilon}}$ 缩放。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 AdamW 算法伪代码

```
AdamW 优化器每步更新:

输入: 参数 w, 学习率 η, 权重衰减 λ, β₁, β₂, ε
输出: 更新后参数 w

# 1. 计算梯度
g_t = ∇L(w_t)

# 2. 更新有偏一阶/二阶矩估计
m_t = β₁ · m_{t-1} + (1-β₁) · g_t
v_t = β₂ · v_{t-1} + (1-β₂) · g_t²

# 3. 偏差校正
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# 4. Adam 自适应更新（无权重衰减）
w_t' = w_t - η · m̂_t / (√v̂_t + ε)

# 5. 解耦的权重衰减
w_{t+1} = w_t' - η · λ · w_t    # ← 关键：不除以 √v̂_t + ε
```

### 2.2 AdamW vs 带 L2 的 Adam（形式化对比）

| 步骤 | 带 L2 的 Adam | **AdamW** |
|:----|:-------------:|:---------:|
| 梯度计算 | $g_t = \nabla L(w_t) + \lambda w_t$ | $g_t = \nabla L(w_t)$ |
| 矩估计 | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ |
| 自适应更新 | $\Delta w = \eta \frac{m_t}{\sqrt{v_t + \epsilon}}$ | $\Delta w = \eta \frac{m_t}{\sqrt{v_t + \epsilon}}$ |
| 权重衰减 | **隐含**在梯度中 | **显式**: $\eta \lambda w_t$ |
| 正则项是否被自适应缩放 | ✅ **是**（不正确） | ❌ **否**（正确） |

### 2.3 与 SGD 的权重衰减的理论等价性

在 SGD 中，解耦的权重衰减与 L2 正则化等价：

$$\text{SGD: } w_{t+1} = (1 - \eta\lambda) w_t - \eta\nabla L(w_t)$$

在 Adam 中，只有解耦的权重衰减能实现**相同的正则化语义**：

$$\text{AdamW: } w_{t+1} = (1 - \eta\lambda) w_t - \eta \frac{m_t}{\sqrt{v_t + \epsilon}}$$

### 2.4 超参数设置

论文建议 AdamW 的超参数设置与 Adam 相同：

| 超参数 | 典型值 | 说明 |
|:------|:-----:|:----|
| $\beta_1$ | 0.9 | 一阶矩的指数衰减 |
| $\beta_2$ | 0.999 | 二阶矩的指数衰减 |
| $\epsilon$ | $10^{-8}$ | 数值稳定性 |
| **权重衰减 $\lambda$** | **0.01-0.1** | CV 任务默认 0.01，NLP 任务常用 0.1 |
| 学习率 $\eta$ | 与 Adam 相同 | 3e-4 到 1e-3（LLM 常用） |

**重要**：切换到 AdamW 时，相当于 Adam 的 L2 正则化权重需要调整，因为 AdamW 的 $\lambda$ 不等价于 Adam 的 L2 正则化系数（L2 被自适应缩放过了）。

### 2.5 AdamW 的变体

| 变体 | 特点 | 应用 |
|:----|:----|:----|
| **AdamW (原始)** | 解耦权重衰减 | GPT-3, LLaMA, Chinchilla |
| **AdamW + decoupled LR** | 权重衰减学习率独立 | GPT-4（推测） |
| **AdamW + LayerNorm scaling** | 对 LayerNorm 权重用不同 $\lambda$ | LLaMA 2 |
| **AdamW + Schedule-Free** | 移除调度器 | 实验性质 |
| **FusedAdamW** (NVIDIA) | 融合 CUDA kernel | 训练速度提升 5-10% |

---

## 三、Experiments and Key Findings

### 3.1 图像分类（CIFAR-10/100, ImageNet32）

| 数据集 | 模型 | Adam (L2) | **AdamW** | SGD + Momentum | 改善 |
|:------|:----|:---------:|:---------:|:--------------:|:----:|
| CIFAR-10 | ResNet-32 | 94.2% | **94.9%** | 94.7% | +0.7% |
| CIFAR-10 | ResNet-56 | 93.8% | **94.7%** | 94.5% | +0.9% |
| CIFAR-100 | ResNet-32 | 74.7% | **75.8%** | 75.2% | +1.1% |
| CIFAR-100 | Wide ResNet 28-10 | 80.4% | **81.1%** | 80.9% | +0.7% |
| ImageNet32 | ResNet-50 | 74.3% | **75.2%** | 74.8% | +0.9% |

**AdamW 在视觉任务中一致优于带 L2 的 Adam，且达到或超过 SGD+Momentum**。

### 3.2 语言建模（LSTM, Transformer）

| 数据集 | 模型 | Adam (L2) | **AdamW** | 改善 |
|:------|:----|:---------:|:---------:|:----:|
| PTB | LSTM | 60.7 (PPL) | **60.2 (PPL)** | -0.5 |
| WikiText-2 | Transformer | 98.3 (PPL) | **97.5 (PPL)** | -0.8 |
| enwik8 | Transformer XL | 1.23 (BPC) | **1.21 (BPC)** | -0.02 |

**一个意外的发现**：AdamW 允许使用比 Adam 更大的权重衰减系数（因不被自适应缩放），使得泛化边界更宽。

### 3.3 权重衰减系数对泛化的影响

| 权重衰减 $\lambda$ | Adam (Test Acc) | **AdamW (Test Acc)** |
|:-----------------:|:---------------:|:--------------------:|
| 0.0 | 93.2% | 93.2% |
| 0.001 | 93.5% | 93.8% |
| 0.01 | 93.8% | **94.5%** |
| 0.1 | 92.1% | **94.2%** |
| 1.0 | 86.4% | **93.1%** |

**AdamW 对 $\lambda$ 的选择更鲁棒**——即使在 $\lambda=1.0$ 的极端值下，AdamW 也仅下降 1%，而 Adam 下降 7%。

### 3.4 训练稳定性

| 指标 | Adam (L2) | **AdamW** |
|:----|:---------:|:---------:|
| Loss 曲线波动 | 中等 | **更平滑** |
| 对 LR 的敏感度 | 高 | **低** |
| 对 $\lambda$ 的敏感度 | 高 | **低** |
| 跨任务迁移性 | 差 | **好** |

**AdamW 的整体训练曲线更平滑，对超参数选择更不敏感**——这是它在 LLM 领域被广泛采用的关键原因之一。

---

## 四、Limitations and Challenges

1. **额外的超参数**：引入 $\lambda$（权重衰减系数）作为新的超参数，需要额外调优（虽然比 Adam 的 L2 系数更鲁棒）
2. **未解决 Adam 的根本问题**：AdamW 只修复了权重衰减的解耦，不解决 Adam 的泛化差距（与 SGD 相比）、memory footprint 大（$2\times$ 参数量的 moment 存储）等固有缺点
3. **对大规模训练的隐式假设**：Chinchilla、GPT-3 等使用 AdamW 的同时配合 Cosine LR schedule + gradient clipping，这些组合效应未被论文系统研究
4. **CPU 上的浮点一致性**：解耦的权重衰减在不同精度下（fp32 vs bf16）的数值行为不同，低精度下 λ 更大可能导致权重消失
5. **权重衰减与 BatchNorm / LayerNorm 的相互作用**：解耦衰减对 Normalization 层的权重应如何设置仍是一个开放问题（LLaMA 2 对此有专门调参）
6. **自适应学习率和模型初始化**：Warmup + AdamW 的组合可能不如 SGD + momentum warmup 稳定

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 AdamW 的关系 |
|---------|:----:|----------------|
| **AdamP** (Heo et al.) | 2021 | 将解耦思想扩展到正则化方向，对投影方向进行解耦 |
| **Adan** (Xie et al.) | 2023 | 融合 Nesterov 动量的自适应优化器，在 AdamW 基础上加 Nesterov 加速 |
| **Lion** (Chen et al.) | 2024 | 符号发现优化器，使用 sign 操作替代 Adam 的矩估计，显存更小 |
| **Sophia** (Liu et al.) | 2023 | 用 Hessian 对角估计替代 Adam 的自适应步长 |
| **Schedule-Free AdamW** (Defazio et al.) | 2024 | 移除学习率调度器，简化 AdamW 的使用 |
| **Muon** (Jordan et al.) | 2024 | 基于 Newton-Schulz 迭代的优化器，在 Llama 3 训练中表现出色 |
| **NVIDIA FusedAdam** | 2022 | CUDA 融合实现，单个 kernel 完成所有步骤 |
| **bitsandbytes AdamW** (Dettmers) | 2022 | 8-bit AdamW，优化器状态压缩 |

**影响评估**：AdamW（及其变体）是 GPT-3、GPT-4、LLaMA 1/2/3、Chinchilla、Gemma、Mistral、Qwen 等几乎所有已知 LLM 优化器的**基础选择**。2023 年后，90%+ 的开源大模型训练代码使用 AdamW（8-bit 或 bf16 变体）。AdamW 是训练基础设施中**最成熟、最可靠**的组件之一。

---

## 六、Implications for You / Hardware Compatibility

### AdamW 显存占用对比

| 优化器 | 每参数额外状态 | 7B 模型额外显存 | 70B 模型额外显存 |
|:------|:-------------:|:--------------:|:--------------:|
| SGD | 0 | 0 | 0 |
| SGD + Momentum | 4B × 1 | ~28GB | ~280GB |
| **AdamW (fp32 master)** | 4B × 2 | **~56GB** | **~560GB** |
| **AdamW (bf16)** | 2B × 2 | **~28GB** | **~280GB** |
| **8-bit AdamW** | 1B × 2 | **~14GB (real ~16GB)** | **~140GB** |
| **Lion** (参见 [[Lion]]) | 4B × 1 (动量) | **~28GB** | **~280GB** |

### LLM 训练配置建议

| 训练规模 | 推荐优化器 | 配置 | 理由 |
|:--------|:----------|:----|:----|
| 7B 单 GPU 微调 | **8-bit AdamW (bitsandbytes)** | $\beta_1=0.9, \beta_2=0.95, \lambda=0.1$ | 显存效率 + 质量 |
| 7B 全量预训练 | **bf16 AdamW** | $\beta_1=0.9, \beta_2=0.95, \lambda=0.1$ | 训练稳定 |
| 70B 预训练 | **bf16 AdamW + FSDP** | $\beta_1=0.9, \beta_2=0.95, \lambda=0.1$ | 分布式标准 |
| LoRA 微调 | **8-bit AdamW (PEFT)** | $\beta_1=0.9, \beta_2=0.999, \lambda=0.0$ | LoRA 模块通常不需 WD |
| 极小模型 | **AdamW (fp32)** | 标准设置 | 显存不是瓶颈 |

### 实际调参经验

- **LLaMA 系列**：$\beta_2=0.95$（不是 Adam 默认的 0.999），$\lambda=0.1$，cosine schedule，warmup 500-2000 steps
- **GPT-3**：$\beta_2=0.95$，$\lambda=0.1$，cosine 到 10% LR
- **Chinchilla**：$\beta_2=0.95$，$\lambda=0.1$，weight decay 应用于所有参数**除 bias 和 LayerNorm 外**
- **常见教训**：$\beta_2=0.999$ 在 LLM 预训练中可能导致训练不稳定（梯度平方估计更新太慢）

### 硬件兼容性总结
- ✅ AdamW (bf16/fp32)：所有 GPU 支持，LLM 训练标配
- ✅ 8-bit AdamW：消费级 GPU（3090/4090）上训练 7B 的关键使能技术
- ⚠️ fp32 AdamW (master weights)：额外 56GB (7B)，对显存压力大
- ❌ AdamW + DeepSpeed CPU offload：极慢，仅在超大模型且显存受限时使用

---

## PDF

[[AdamW 原文.pdf]]
