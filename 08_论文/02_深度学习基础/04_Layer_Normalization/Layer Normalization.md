---
tags:
  - 论文
  - 归一化
  - 训练稳定性
  - 架构组件
created: 2026-06-30
paper_title: "Layer Normalization"
paper_authors: "Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton"
paper_year: 2016
paper_venue: "arXiv"
paper_citations: "~20,000+"
paper_url: "https://arxiv.org/abs/1607.06450"
---

# Layer Normalization

**Layer Normalization**
*University of Toronto | arXiv 1607.06450*

> Transformer 架构中不可或缺的组件（每个子层后都有 LayerNorm）。与 BatchNorm 不同，LayerNorm 在每个样本的特征维度上做归一化，不依赖 batch 统计量，因此天然适合序列建模和 RNN/Transformer。

---

## 一、研究背景与动机

在深度学习训练中，一种常见策略是对每层输入进行归一化，使分布稳定，从而加速训练、缓解梯度消失/爆炸问题。Batch Normalization (Ioffe & Szegedy, 2015) 在 CNN 上取得了巨大成功，但它有根本性局限：

1. **依赖 batch 大小**：小 batch（<8）时统计量噪声大，训练不稳定
2. **训练-推理行为不一致**：推理时使用全局 running mean/var，而非当前 batch 统计量
3. **不适用于 RNN**：RNN 在不同时间步处理不同长度的序列，不同时间步的统计量不同，无法合理计算 batch 归一化

Layer Normalization 的提出正是为了解决这些问题——通过**在每个样本内部沿特征维度做归一化**，彻底消除对 batch 的依赖。

## 二、核心方法

### 2.1 基本公式

$$
\mu^l = \frac{1}{H} \sum_{i=1}^{H} a_i^l
$$
$$
\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_i^l - \mu^l)^2}
$$
$$
h^l = f\left( \frac{g}{\sigma^l} \odot (a^l - \mu^l) + b \right)
$$

其中 $H$ 是特征维度（hidden size），$a^l$ 是第 $l$ 层的输入，$g$ 和 $b$ 是可学习的 scale (γ) 和 shift (β) 参数。

### 2.2 与 Batch Normalization 的对比

| 特性 | Batch Normalization | Layer Normalization |
|------|--------------------|--------------------|
| 归一化维度 | batch (N) | feature (H) |
| 依赖 batch | 是（需要统计量） | 否 |
| 训练/推理行为 | 不一致（running stats） | 一致 |
| RNN 适用性 | 差 | 好 |
| CNN 适用性 | 好 | 一般 |
| 小 batch 稳定性 | 差 | 好 |
| 额外存储 | running mean/var | 无 |

### 2.3 RNN 中的 Layer Normalization

在 RNN 中，每个时间步的归一化独立计算：

$$
h_t = f(W_h h_{t-1} + W_x x_t)
$$
$$
h_t = LN(h_t) = \frac{g}{\sigma_t} \odot (h_t - \mu_t) + b
$$

其中 $\mu_t$ 和 $\sigma_t$ 在隐藏向量的维度上计算，每个时间步独立。

## 三、关键实验与发现

### 3.1 RNN 实验结果

| 实验设置 | 无归一化 | BatchNorm (T=1) | BatchNorm (T=full) | LayerNorm |
|---------|---------|------------------|--------------------|----------|
| 单步 RNN (6层) | 5-6 steps | 失败 | - | 4 steps |
| 排序任务 (字符级) | 收敛慢 | - | - | 收敛快 2x |
| IWSLT2014 英德翻译 | 16 PPL | - | - | 14.5 PPL |

### 3.2 关键发现

1. **LayerNorm 使 RNN 训练更稳定**：允许训练更深的 RNN（6层 vs 无归一化不稳定）
2. **与 BatchNorm 不同，LN 不依赖 batch**：在 batch size=1 时仍然有效
3. **LN 与 Dropout 互补**：LN 不引入dropout-like的随机性，与正则化方法正交
4. **LN 的作用不仅限于 RNN**：在 MLP 和 CNN 上也有一定效果

## 四、局限性与后续影响

**局限**：
1. **CNN 上不如 BatchNorm**：在计算机视觉任务上，LayerNorm 通常不如 BatchNorm 效果好（因为 CNN 特征的通道维度有统计意义）
2. **计算开销**：虽然不大，但与不归一化相比增加了前向/反向计算量
3. **理论上不如 BatchNorm 理解充分**：BatchNorm 的连接损失曲面（smoothing the optimization landscape）理论比 LN 更清晰

**后续影响**：
- **RMSNorm (2019)**：LayerNorm 的简化变体，去掉了 mean 的平移操作，仅做 RMS 缩放，计算量节省 50%
- LayerNorm 成为 Transformer 的标准组件（Attention Is All You Need）
- **Pre-LN** 替代 Post-LN：将 LayerNorm 放在子层之前而非之后，使训练更稳定
- Adaptive LayerNorm (AdaLN)：在 DiT（Diffusion Transformer）中使用，按时间步条件化

## 五、VLA/机器人研究中的角色

Layer Normalization 是所有 VLA 模型中的基础组件：

1. **所有 Transformer 骨干**：GPT、Llama、PaliGemma、Qwen 等每个 Transformer 子层后都有 LayerNorm（或其变体 RMSNorm）
2. **π0 的 DiT Action Expert**：使用 Adaptive LayerNorm (AdaLN)，输入时间步 t 和条件信号，这是 DiT 架构的核心创新
3. **FLOWER**：关键创新之一是将 AdaLN 的 scale 和 shift 参数作为条件化接口（而不是直接拼接 CNN 特征），显著提升了策略性能
4. **扩散策略**：Denoising Transformer 使用 LayerNorm 进行时间步条件化（通过 AdaLN）

LayerNorm 从 Transformer 中的"默认组件"变成了 VLA 中的"条件化接口"——FLOWER 和 DiT 的创新正体现在如何利用 LayerNorm 的参数化能力。

## 六、对你的启示

1. **不要忽略最简单的组件**：LayerNorm 看似只是一个归一化技巧，但在 FLOWER 中成为了核心创新点。理解 LN 的参数化机制（γ, β 的可学习性）是理解 AdaLN 条件化的基础
2. **Pre-LN vs Post-LN 的选择**：现代 Transformer 几乎全部使用 Pre-LN（GPT-2 开始），这使得训练更深（100层+）成为可能。在搭建 VLA 模型时，默认使用 Pre-LN
3. **RMSNorm 是更高效的选择**：Llama 系列使用 RMSNorm（移除了 mean 平移），在保证训练稳定的同时节省计算量
4. **代码实践**：
   - PyTorch 中直接使用 `nn.LayerNorm(hidden_size)`
   - 理解 `elementwise_affine` 参数（是否使用可学习的 γ, β）
   - 尝试在简化 Transformer 中将 Post-LN 改为 Pre-LN 观察训练稳定性变化

## PDF

[[1607.06450_Layer_Normalization.pdf]]
