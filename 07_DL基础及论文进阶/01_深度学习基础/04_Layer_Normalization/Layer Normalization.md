---
tags:
  - 论文
  - 归一化技术
  - Transformer架构
  - VLA基础设施
  - RNN
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
*Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton | University of Toronto | arXiv 1607.06450*

> Layer Normalization 是所有基于 Transformer 的模型的默认归一化方法。与 [[Batch Normalization]] 不同，LN 沿特征维度对每个独立样本进行归一化，因此既不依赖 batch size，也在训练和推理时行为完全一致。这一特性对于处理变长序列的 RNN 和处理数十亿 token 的 Transformer 至关重要。其现代变体 [[RMSNorm]]（用于 Llama 系列）去除了均值中心化步骤，获得了约 10% 的速度提升。

---

## 一、研究背景与核心思想

Layer Normalization 的提出直接源于 [[Batch Normalization]] 在序列模型中遇到的三个根本性局限。LN 的核心洞察极其简洁优雅——不跨样本归一化，而是对每个训练样本独立地在特征维度上进行归一化。这一简单的轴选择变化带来了根本性的能力差异。

### 1.1 Batch Normalization 在序列模型中的局限

[[Batch Normalization]] 彻底革新了 CNN 训练，但对于 RNN 和 Transformer，它存在三个致命局限：

**1. 对 batch size 的依赖**：BN 的统计量（$\mu_\mathcal{B}$、$\sigma^2_\mathcal{B}$）在 batch 维度上计算。当 batch size 较小时（这在长序列模型中非常常见），这些估计会变得嘈杂。在在线学习或超大规模分布式模型中，mini-batch 可能仅包含 1-2 个样本。

**2. 训练-推理不一致**：BN 在推理时使用 running 统计量，导致训练和推理对应两个不同的计算图，这增加了部署和调试的复杂性。

**3. RNN 完全不兼容**：在 RNN 中，同一个单元在不同时间步上展开。BN 需要为每个时间步维护单独的 running 统计量，而测试序列如果长于任何训练序列，将遇到没有统计量的"未见时间步"。正如论文所述："如何将 [BN] 应用于循环神经网络并不显而易见。"

具体而言，RNN 在时间步 $t$ 的隐藏状态为：
$$h_t = f(W_h h_{t-1} + W_x x_t)$$

如果尝试在每个时间步对 $W_h h_{t-1}$ 应用 BN，就需要为每个 $t$ 分别估计统计量。一个长于所有训练序列的测试序列，其较晚的时间步将无法获得统计量。

### 1.2 核心洞察：逐样本、逐层归一化

LN 的关键洞察极其简洁优雅：**不在 batch 上做归一化，而是对每个训练样本独立地在特征维度上做归一化**。归一化统计量仅取决于当前输入在当前时间步的值，与其他样本或其他时间步完全无关。

这意味着：
- **无 batch 依赖**：batch size = 1 也能正常工作
- **训练 = 推理**：两阶段计算完全相同
- **时间步独立**：每个 RNN 时间步独立完成归一化
- **变长序列支持**：序列中每个位置的统计量都由自身特征计算，不存在"未见时间步"问题

### 1.3 为什么 Transformer 需要 LN

当 [[Attention Is All You Need]] 在 2017 年提出 Transformer 时，设计者面临一个选择：RNN 中已经证明 BN 不适用，但深层 Transformer 同样需要归一化来稳定训练。Transformer 的独有挑战包括：

1. **多头注意力输出的尺度不一致**：多个注意力头的输出拼接后，不同头的激活值量级可能差异巨大，需要归一化来统一尺度
2. **残差连接中的梯度累积**：深度 Transformer（12 层以上）中，未归一化的残差流会导致激活值逐层增长，最终数值溢出
3. **变长序列批次**：Transformer 的 padding 操作使得 batch 维度上包含无效的填充 token，BN 无法正确处理

LN 完美回答了所有这些需求：它不关心序列长度、不依赖 batch 统计量、处理每个 token 独立的特征分布。这就是为什么 LN（及其变体 RMSNorm）成为了 Transformer 架构的标配。

## 二、方法/架构/技术贡献

LN 的计算与 BN 在代数形式上高度相似，区别仅在于归一化轴的选择。论文还提供了 LN 在 RNN 和 LSTM 中的完整公式，并分析了 LN 的不变性属性。

### 2.1 Layer Normalization 变换

对于一个具有 $H$ 个隐藏单元（特征维度）的层，设求和输入为 $a^l$：

$$\mu^l = \frac{1}{H} \sum_{i=1}^{H} a_i^l$$

$$\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_i^l - \mu^l)^2}$$

$$h^l = f\left(\frac{g}{\sigma^l} \odot (a^l - \mu^l) + b\right)$$

其中：
- $g$（gain）和 $b$（bias）是可学习参数，类比 BN 中的 $\gamma$ 和 $\beta$
- $\odot$ 表示逐元素乘法
- $f(\cdot)$ 是非线性激活函数（在归一化之后应用）

**与 BN 的关键区别**：归一化是对单个训练样本的整个隐藏层向量（全部 $H$ 个单元）进行的，而不是对 batch 中所有样本的某一个特征进行。这一区别带来了完全不同的行为特性。

### 2.2 与 Batch Normalization 的详细对比

| 属性 | Batch Normalization | Layer Normalization |
|---|---|---|
| **归一化轴** | Batch (N) | 特征 (H) |
| **统计量计算范围** | N × H × W（跨样本） | H（单样本内） |
| **是否依赖 batch size** | 是——小 batch 显著影响性能 | 否——batch size = 1 也可用 |
| **训练 vs 推理** | 不同（running stats） | 完全相同 |
| **RNN 适用性** | 差（时间步统计量问题） | 天然适用（每步独立） |
| **CNN 效果** | 优秀 | 差（见 3.3 节） |
| **每层参数量** | $2 \times C$（逐通道 $\gamma, \beta$） | $2 \times H$（逐神经元 $\gamma, \beta$） |
| **额外存储** | running mean/var（2 个 buffer） | 无 |
| **小 batch 稳定性** | 不稳定（$m < 8$） | 稳定 |
| **变长序列支持** | 破坏（未见时间步） | 天然支持 |

### 2.3 LN 在 RNN 中的应用

论文的主要贡献之一就是让归一化在 RNN 中可行。LN-RNN 的公式如下：

**标准 RNN 单元**：
$$a_t = W_{hh} h_{t-1} + W_{xh} x_t$$

**Layer Normalized RNN 单元**：
$$\mu_t = \frac{1}{H} \sum_{i=1}^{H} a_t^{(i)}, \quad \sigma_t = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_t^{(i)} - \mu_t)^2}$$
$$h_t = f\left(\frac{g}{\sigma_t} \odot (a_t - \mu_t) + b\right)$$

每个时间步的 $\mu_t$ 和 $\sigma_t$ 从当前的 $a_t$ 独立计算——没有统计量跨时间步共享。这阻止了通常在 RNN 中出现的梯度爆炸/消失，因为 $\sigma_t$ 会在每个时间步自适应 $a_t$ 的尺度。

**LN 应用于 LSTM**：论文提供了具体的 LN-LSTM 公式。以标准 LSTM 为例：

标准 LSTM：
$$\begin{pmatrix} f_t \\ i_t \\ o_t \\ g_t \end{pmatrix} = W_h h_{t-1} + W_x x_t + b$$
$$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t)$$
$$h_t = \sigma(o_t) \odot \tanh(c_t)$$

应用 LN（分别对循环权重和输入权重做归一化）：
$$\begin{pmatrix} f_t \\ i_t \\ o_t \\ g_t \end{pmatrix} = \text{LN}(W_h h_{t-1}; \alpha_1, \beta_1) + \text{LN}(W_x x_t; \alpha_2, \beta_2) + b$$
$$h_t = \sigma(o_t) \odot \tanh(\text{LN}(c_t; \alpha_3, \beta_3))$$

LN 被应用于**求和输入之后、非线性激活之前**，与 BN 的放置位置类似。LN 应用于 GRU 的模式也同理：每个线性投影在求和之前独立进行归一化。

### 2.4 LN 的不变性属性（理论贡献）

论文对 LN 的不变性进行了严谨的理论分析：

| 方法 | 权重重新缩放 | 权重重新中心化 | 权向量重新缩放 | 数据集重新缩放 | 数据集重新中心化 | 单样本重新缩放 |
|---|---|---|---|---|---|---|
| BatchNorm | 不变 | 否 | 不变 | 不变 | 不变 | 否 |
| WeightNorm | 不变 | 否 | 不变 | 否 | 否 | 否 |
| **LayerNorm** | **不变** | **不变** | 否 | 不变 | 否 | **不变** |

LN 对**整个权重矩阵的缩放**以及**所有权重的偏移**均具不变性。具体来说，如果 $W' = \delta W + \mathbf{1} \gamma^\top$，LN 模型的输出保持不变。这是因为 LN 的 $\mu$ 和 $\sigma$ 统计量涉及该层的所有神经元，所有权重同时被一个常向量偏移会被归一化操作吸收。

最重要的是，LN 对**单个训练样本的缩放具有不变性**——归一化的标量仅取决于当前输入。这一性质使 LN 对变长序列和不同量级的输入具有鲁棒性。

## 三、实验与关键发现

论文通过多组实验系统验证了 LN 的效果，覆盖了图像-文本排序、阅读理解、MNIST 分类和 CNN 等多个任务。论文最令人印象深刻的是对 CNN 效果不佳的诚实评估——这在学术论文中并不多见。

### 3.1 图像-文本排序（Order-Embeddings）

第一个实验在 GRU-based 的 order-embedding 模型上测试 LN 的效果，任务为跨模态检索（MS COCO 数据集）：

| 模型 | Caption R@1 | Caption R@5 | Caption R@10 | Image R@1 | Image R@5 | Image R@10 |
|---|---|---|---|---|---|---|
| OE（基线） | 46.6 | 79.3 | 89.1 | 37.8 | 73.6 | 85.7 |
| OE + LN | **48.5** | **80.6** | **89.8** | **38.9** | **74.3** | **86.3** |

- LN 在所有召回指标上均有提升
- LN 在**基线 60% 的训练时间内**完成收敛
- 这达到了当时 RNN 嵌入模型的最先进水平

### 3.2 Attentive Reader（问答任务）

阅读理解任务（CNN corpus）直接比较了 LN 与循环 BN（recurrent BN）：

| 模型 | 验证错误率 |
|---|---|
| LSTM（基线） | 最高 |
| BN-LSTM（Cooijmans et al.） | 较低 |
| BN-everywhere | 与 BN-LSTM 相近 |
| **LN-LSTM** | **最低** |

LN-LSTM 的表现优于基线和专门设计的循环 BN。验证曲线显示 LN 不仅收敛更快，最终错误率也更低。

### 3.3 排列不变 MNIST（前馈网络）

该实验在前馈 MLP 上测试 LN，并在不同 batch size 下与 BN 进行对比：

| 条件（batch size） | 基线 | BN | LN |
|---|---|---|---|
| **大 batch（128）** | NLL 高 | NLL 较低 | **NLL 最低** |
| **小 batch（4）** | 无法收敛 | BN 不稳定（高方差） | **最稳定，收敛最快** |

小 batch 实验具有关键意义：在 batch size = 4 时，BN 的方差估计极度嘈杂，导致训练不稳定。而 LN **完全不受 batch size 影响**。

### 3.4 CNN 评估（诚实的评估）

论文测试了 LN 在 CNN 上的表现，并诚实地报告了结果：**LN 在 CNN 上的表现不如 BN**。CIFAR-10 上的测试错误率：

| 方法 | 测试错误率 |
|---|---|
| 基线（无归一化） | 8.96% |
| BatchNorm | **8.25%** |
| LayerNorm | 10.49% |

LN 甚至**损害**了 CNN 的表现——比基线还差 1.53%。论文解释道："在全连接层中，所有隐藏单元对最终预测的贡献趋于一致……然而对于卷积神经网络，这一假设不再成立。"具体而言，边界神经元（靠近图像边缘）的统计量与内部神经元差异巨大，将它们放在一起归一化损害了性能。

### 3.5 其他实验结果汇总

| 任务 | 模型 | LN 的影响 |
|---|---|---|
| Skip-thought vectors | 句子编码器 RNN | 提升所有下游任务（MR: +2.2%, CR: +0.8%, SUBJ: +0.8%） |
| 手写体生成 | 500 步长序列 RNN | LN 将 NLL 从约 0 降至 -700（相比基线） |
| DRAW（MNIST 生成） | 循环注意力模型 | LN 收敛快 2 倍，最终 NLL 更优（82.09 vs 82.36 nats） |

## 四、局限性与挑战

LN 并非万能。它在 CNN 上的糟糕表现是一个显著的短板，而其理论理解的完善程度也落后于 BN。

### 4.1 CNN 上的糟糕表现

如 3.4 节所示，LN 在 CNN 上的表现劣于 BN。根本原因在于 CNN 特征的空间统计特性不一致——图像边缘附近的边缘检测器与图像中心区域的检测器具有完全不同的统计量。将它们一起归一化会破坏这种有用的多样性。

这导致了一个现在已经广为人知的经验法则：**CNN 优先使用 BN，序列模型优先使用 LN**。

### 4.2 理论理解落后于 BN

BN 平滑损失景观的效应已经有了严谨的分析（Santurkar et al., 2018）。对于 LN，理论上的理解相对不够完备。论文在补充材料中提供了 Fisher 信息矩阵分析，但与训练动态的联系不那么直接。

### 4.3 计算开销

LN 为每层增加了两个计算操作：
1. 在特征维度上计算 $\mu$ 和 $\sigma$
2. 应用仿射变换（$g \odot \hat{x} + b$）

虽然开销相对适中，但这也成为了 [[RMSNorm]] 的提出动机——RMSNorm 移除了均值 $\mu$ 的计算，在大型 Transformer 上节省约 10% 的计算量。

## 五、与后续工作的关系/对领域的影响

LN 在深度学习历史中占据了一个极其特殊的位置：它是 Transformer 架构正常工作的基石，Pre-LN vs Post-LN 的设计决策影响了几乎所有后续大语言模型的架构。从 LN 到 RMSNorm 再到 AdaLN 的演化链，直接构成了 VLA 领域的归一化技术基础设施。

### 5.1 Pre-LN vs Post-LN：Transformer 架构的关键争论

原始 Transformer（[[Attention Is All You Need]], 2017）使用了 **Post-LN**：LN 应用在残差连接之后：
$$\text{Output} = \text{LN}(x + \text{Sublayer}(x))$$

后续工作发现 **Pre-LN**（在子层之前应用 LN）提供了更稳定的训练：
$$\text{Output} = x + \text{Sublayer}(\text{LN}(x))$$

| 属性 | Post-LN | Pre-LN |
|---|---|---|
| 梯度信号 | 残差路径上有 LN，阻碍干净梯度流 | 残差路径干净（无 LN），梯度自由流动 |
| 是否需 Warmup | 是（需要仔细的 warmup 策略） | 一般 warmup 即可 |
| 深层模型稳定性（< 100 层） | 容易发散 | 稳定 |
| 用于 | 原始 Transformer（2017） | GPT-2 之后的所有现代 LLM |
| 理论分析 | 困难（LN 与残差交互复杂） | 容易（LN 在每个 block 之前） |

所有现代 VLA 架构（[[Llama 2]]、Gemma、Qwen、PaliGemma）都使用 Pre-LN。

### 5.2 完整演化：LN → RMSNorm → AdaLN

| 方法 | 提出年份 | 与 LN 的关系 | 关键创新 | 代表使用 |
|---|---|---|---|---|
| **LayerNorm** | 2016 | 原始方法 | 均值+方差归一化 | 原始 Transformer |
| **[[RMSNorm]]** | 2019 | 去均值化 | 仅 RMS 缩放，省去均值计算约 10% | Llama 系列 |
| **AdaLN** | 2023 | 条件化 $\gamma, \beta$ | $\gamma$ 和 $\beta$ 由条件输入（timestep、class）预测 | [[DiT]], pi-zero |

在三者之中：
- **LayerNorm** 是理论基础，定义了"归一化 → 仿射变换"的模式
- **[[RMSNorm]]** 是最实用的工程改进，Llama 3 70B 使用 RMSNorm 相比 LN 节省了数十亿次浮点运算
- **AdaLN** 是最具创新性的架构扩展，将归一化参数变成了**条件信息的注入接口**

在 VLA 中的应用：
- **pi-zero（2024）**：使用 AdaLN 向 DiT 动作专家注入 timestep 条件
- **FLOWER（2024）**：将 AdaLN 的 $\gamma, \beta$ 参数作为核心条件接口——这是 FLOWER 的关键架构创新之一
- **OpenVLA**：Llama 2 骨干网络使用 RMSNorm

### 5.3 与 LN 相关的论文双链

本笔记与其他论文的关联：

| 相关论文 | 关系 | 链接 |
|---|---|---|
| [[Attention Is All You Need]] | 原始 Transformer 使用 Post-LN | 开创性工作 |
| [[Batch Normalization]] | LN 的前身，为 LN 提供问题背景 | 直接对比对象 |
| [[RMSNorm]] | LN 的工程简化版，用于 Llama 系列 | 直系后代 |
| [[Group Normalization]] | 同期工作，解决 BN 小 batch 问题 | 互补方案 |
| [[ResNet]] | 残差连接与 LN 共同构成现代 Transformer block | 架构组件 |

## 六、对你的启示/硬件兼容性

针对 VLA 研究者和 RTX 3090 24GB / RTX 4070 Ti Super 16GB 的实际条件，以下是在归一化选择上的实用建议。

### 6.1 实践建议

1. **✅ Transformer 默认归一化 = RMSNorm**：在任何新的基于 Transformer 的 VLA 模型中，使用 RMSNorm（无均值中心化）直接替换 LN，可节省计算量且不损失质量

2. **✅ Pre-LN 是默认配置**：始终在 Transformer block 中使用 Pre-LN（子层前归一化）。这是所有 GPT-2 之后架构的标准做法。具体位置：`x = x + Sublayer(LN(x))`

3. **⚠️ 注意 $\epsilon$ 参数**：LN 中的 $\epsilon$ 防止除零。在 fp16/bf16 训练时，将 $\epsilon$ 从默认的 $10^{-6}$ 提升到 $10^{-5}$ 以获得更好的数值稳定性

4. **❌ LN 并非万能**：对于基于 CNN 的视觉编码器，batch >= 16 时使用 BN，batch < 16 时使用 GN。LN 是为序列模型和 Transformer 设计的

### 6.2 PyTorch 实现

```python
# 标准 LayerNorm
ln = nn.LayerNorm(hidden_size)  # elementwise_affine=True 默认
ln_no_bias = nn.LayerNorm(hidden_size, bias=False)  # 仅 scale 不 shift

# 无可学习参数的固定 LN
ln_fixed = nn.LayerNorm(hidden_size, elementwise_affine=False)

# RMSNorm（手动实现，约 10% 更快）
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# 在 Transformer block 中使用 Pre-LN
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(...)

    def forward(self, x):
        # Pre-LN 模式：LN 在子层前
        x = x + self.attn(self.norm1(x))  # 注意这里
        x = x + self.ffn(self.norm2(x))   # 注意这里
        return x
```

### 6.3 为什么理解 LN 对 VLA 创新至关重要

LN 看似只是一个"归一化技巧"，但它已经成为一个创新面：

- **FLOWER 的核心贡献**正是利用 AdaLN 的 $\gamma, \beta$ 参数作为条件接口——将归一化参数重新定义为信息载体
- **DiT 的创新**在于将 LN 参数以扩散 timestep 为条件——打开了"归一化参数 = 条件信号"的设计空间
- **理解 $\gamma$ 和 $\beta$** 作为可学习的、能够携带条件信息的参数，是阅读现代 VLA 论文的必备知识

### 6.4 硬件兼容性

| 硬件 | LN/RMSNorm 表现 | 显存影响 |
|---|---|---|
| RTX 3090 24GB | ✅ 无任何问题，是 LN 的理想运行环境 | LN 本身无额外显存开销 |
| RTX 4070 Ti Super 16GB | ✅ 同样无问题 | 16GB 可运行 7B 模型（fp16）的 LN 计算 |
| fp16/bf16 混合精度 | ✅ 将 eps 设为 1e-5 即可 | 无额外精度损失 |
| 量化推理（INT8/INT4） | ⚠️ LN 的数值范围在量化时需注意 | 建议使用 nn.LayerNorm 的量化版本 |

### 6.5 归一化技术选择总决策树

```
你的模型是什么类型？
├── CNN-based 视觉编码器
│   ├── batch size >= 16 → ✅ BatchNorm
│   ├── batch size < 16  → ✅ GroupNorm
│   └── 视频/3D 大输入    → ✅ GroupNorm（强制）
├── Transformer（任何变体）
│   ├── 标准 Transformer   → ✅ RMSNorm（替代 LN）
│   ├── Diffusion Transformer → ✅ AdaLN（条件归一化）
│   └── 混合模型（CNN+Transformer）→ 视觉部分用 GN/GN，Transformer 部分用 RMSNorm
└── RNN/LSTM（极少见）
    └── ✅ LayerNorm（原始方法）
```

## PDF

[[Layer Normalization 原文.pdf]]
