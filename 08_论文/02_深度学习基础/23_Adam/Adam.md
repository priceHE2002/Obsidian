---
tags:
  - 论文
  - 优化器
  - 训练技巧
  - LLM训练
  - 自适应学习率
  - VLA基础设施
created: 2026-06-30
paper_title: "Adam: A Method for Stochastic Optimization"
paper_authors: "Diederik P. Kingma, Jimmy Ba"
paper_year: 2014
paper_venue: "ICLR 2015"
paper_citations: "~200,000+"
paper_url: "https://arxiv.org/abs/1412.6980"
---

# Adam

**Adam: A Method for Stochastic Optimization**
*Diederik P. Kingma, Jimmy Ba | University of Amsterdam, OpenAI / University of Toronto | ICLR 2015 | arXiv 1412.6980*

> Adam 是深度学习领域事实上的默认优化器。它融合了动量（加速收敛）和逐参数自适应学习率（通过梯度二阶矩），对超参数选择鲁棒，能处理稀疏梯度，开箱即用适用于 CNN、RNN 和 Transformer。其改进变体 AdamW（解耦权重衰减）是所有主流 VLA 系统的通用优化器——[[OpenVLA]]、RT-2、Diffusion Policy、pi-zero 和 FLOWER 全部使用它。

---

## 一、研究背景与核心思想

### 1.1 Adam 之前的优化器格局

在 Adam 出现之前，梯度优化存在两大主流范式，分别解决不同的问题：

| 范式 | 代表性方法 | 优势 | 缺陷 |
|---|---|---|---|
| **动量法** | SGD + Momentum、Nesterov SGD | 加速收敛，轨迹平滑 | 全局单一学习率，不擅长稀疏梯度 |
| **自适应学习率** | AdaGrad、RMSProp、AdaDelta | 逐参数学习率，处理稀疏特征 | AdaGrad：学习率单调衰减至零；RMSProp：缺少动量 |

**SGD + Momentum**（Sutskever 等，2013）维护一个累积过去梯度的速度向量：
$$v_t = \mu v_{t-1} + (1 - \mu) g_t$$
$$\theta_t = \theta_{t-1} - \alpha v_t$$

**AdaGrad**（Duchi 等，2011）累积所有历史平方梯度：
$$G_t = \sum_{i=1}^t g_i^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot g_t$$

AdaGrad 对稀疏特征效果良好（不频繁的大梯度获得更大的更新步长），但 $G_t$ 随训练单调增长，最终学习率小到几乎停止训练。

**RMSProp**（Tieleman & Hinton, 2012）使用指数移动平均解决单调衰减问题：
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t + \epsilon}} \odot g_t$$

但 RMSProp 缺少动量，也没有偏差校正（bias correction），这在 $\beta_2$ 接近 1（处理稀疏梯度所需）时会导致训练不稳定。

### 1.2 Adam 的核心洞察

Adam（Adaptive Moment Estimation，自适应矩估计）将上述两大家族融合为统一算法，具有四个关键特性：

1. **维护一阶矩估计 $m_t$**：梯度的运行平均值（类动量）
2. **维护二阶矩估计 $v_t$**：平方梯度的运行平均值（类 RMSProp）
3. **对两个矩都进行偏差校正**：在早期训练步中至关重要
4. **有效步长形成信任域**：$|\Delta_t| \lessapprox \alpha$，即学习率 $\alpha$ 直接限制了每步的参数变化

Adam 的设计原则："在参数空间每步的有效步长幅度大约被步长设定 $\alpha$ 所界定。这可以被理解为在当前参数值周围建立了一个信任域（trust region）。"

### 1.3 为什么需要同时融合动量和自适应学习率

单独使用动量或自适应学习率各有不足：动量法的单一全局学习率无法处理不同参数的不同更新需求（例如，Attention 层的梯度尺度与 Embedding 层相差数十倍）；而自适应方法（RMSProp）缺少动量项，在高曲率损失景观中难以快速穿越平坦区域。Adam 的一阶矩提供了动量加速，二阶矩提供了逐参数的自适应调整。更关键的是，Adam 引入了偏差校正机制来修正 $m_t$ 和 $v_t$ 从零初始化带来的偏差——这一机制是早期训练稳定性的保障，也是 Adam 区别于 RMSProp+Momentum 朴素组合的核心创新。

## 二、方法/架构/技术贡献

### 2.1 完全算法推导

**算法 1：Adam**

**输入**：学习率 $\alpha = 0.001$，衰减率 $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$

**初始化**：$m_0 = 0$（一阶矩），$v_0 = 0$（二阶矩），$t = 0$

每步训练 $t$：

1. **计算梯度**：$g_t = \nabla_\theta f_t(\theta_{t-1})$

2. **更新有偏一阶矩**（动量）：
   $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

3. **更新有偏二阶矩**（自适应学习率）：
   $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
   （逐元素平方：$g_t^2 = g_t \odot g_t$）

4. **偏差校正**：
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. **参数更新**：
   $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 2.2 偏差校正为何重要

$m_t$ 和 $v_t$ 都从零向量初始化。如果没有偏差校正：

- 早期步中，$m_t \approx (1 - \beta_1) g_t$ 严重偏向零
- $v_t$ 在 $\beta_2 = 0.999$ 时更严重：10 步后，$v_{10} \approx 0.01 \sum_{i} 0.999^{10-i} g_i^2$

偏差校正项 $1 - \beta_1^t$ 和 $1 - \beta_2^t$ 补偿了这种初始化偏差。以 $\beta_2 = 0.999$ 为例：

- 1 步后：$1 - 0.999^1 = 0.001$，故 $\hat{v}_1 = 1000 \times v_1$（强校正）
- 1000 步后：$1 - 0.999^{1000} \approx 0.632$，中等校正
- 7000 步后：$1 - 0.999^{7000} \approx 0.999$，校正可忽略

论文的实验（Section 6.4）显示，去除偏差校正（等价于带动量的 RMSProp）在 $\beta_2$ 接近 1 时会导致训练发散。

### 2.3 信噪比解释与梯度缩放不变性

Adam 的有效更新可以解释为：

$$\Delta_t = -\alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

比值 $\hat{m}_t / \sqrt{\hat{v}_t}$ 是**信噪比（SNR）**的近似：

- 当梯度方向一致（高 SNR）时，$\hat{m}_t \approx \pm \sqrt{\hat{v}_t}$，故 $|\Delta_t| \approx \alpha$
- 当梯度方向噪声大（低 SNR，如接近最优值）时，$\hat{m}_t \ll \sqrt{\hat{v}_t}$，故 $|\Delta_t| \ll \alpha$

这意味 Adam **自动退火**步长：接近最优值时自动缩小更新幅度，无需显式的学习率调度（尽管有调度仍然有益）。

此外，更新对**梯度缩放不变**：若所有梯度缩放 $c$ 倍，则 $\hat{m}_t$ 缩放 $c$ 倍，$\sqrt{\hat{v}_t}$ 也缩放 $c$ 倍，比值不变。

### 2.4 AdaMax：使用无穷范数的变体

论文还提出了 AdaMax，使用 $\ell_\infty$ 范数替代 $\ell_2$ 范数：

$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \cdot \frac{m_t}{u_t}$$

AdaMax 简化为追踪指数加权后的最大绝对梯度。更新界限更简单：$|\Delta_t| \leq \alpha / (1 - \beta_1^t)$。在实践中，AdaMax 的使用远少于标准 Adam。

### 2.5 理论收敛保证

论文提供了凸优化情形的遗憾界（regret bound）：

$$R(T) \leq \frac{D^2}{2\alpha(1-\beta_1)} \sum_{i=1}^d \sqrt{T \hat{v}_{T,i}} + \frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2} \sum_{i=1}^d \|g_{1:T,i}\|_2 + \cdots$$

其中 $\gamma = \frac{\beta_1^2}{\sqrt{\beta_2}}$。对于稀疏数据，Adam 实现了 $O(\log d \sqrt{T})$ 的遗憾——相比非自适应方法的 $O(\sqrt{d T})$ 有明显改善。这与 AdaGrad 已知的最佳结果相当。

## 三、实验与关键发现

### 3.1 逻辑回归与多层神经网络

**MNIST 逻辑回归**：

| 数据集 | 任务 | Adam 表现 |
|---|---|---|
| MNIST（784 维像素） | 10 类逻辑回归 | 与 AdaGrad 相当，显著优于 SGD+Nesterov |
| IMDB（10K BoW 特征） | 情感分类 | Adam + dropout：收敛最快，损失最低 |

Adam 在稀疏特征上收敛速度和 AdaGrad 一样快，同时受益于动量特性。

**MNIST 多层神经网络（2 隐层，每层 1000 隐藏单元，ReLU 激活）**：

| 条件 | Adam 与其他方法对比 |
|---|---|
| **确定性训练**（无 dropout） | 收敛快于 SFO（拟牛顿法）、AdaGrad、SGD+Nesterov |
| **带 dropout** | 显著优于所有对手 |
| **墙上时间** | SFO 每轮迭代比 Adam 慢 5-10 倍（Adam 仅需一阶梯度） |

即使有 dropout 的随机正则化，Adam 仍保持优势。

### 3.2 卷积神经网络（CIFAR-10）

架构：c64-c64-c128-1000（卷积层 + 全连接层）

| 阶段 | 行为 |
|---|---|
| **前 3 个 epoch** | Adam 和 AdaGrad 快速取得初始进展 |
| **45 个 epoch 后** | Adam 和 SGD 收敛良好；AdaGrad 提前陷入平台期 |

论文指出，Adam 的二阶矩估计 $\hat{v}_t$ 在几个 epoch 后变得非常小（被 $\epsilon$ 主导），使得该近似对 CNN 的帮助减弱。但一阶矩（动量）继续提供加速效果。

### 3.3 偏差校正消融实验（变分自编码器）

该实验直接测试偏差校正的重要性：

| $\beta_1$ | $\beta_2$ | 有偏差校正 | 无偏差校正 |
|---|---|---|---|
| 0 | 0.99 | 稳定 | 稳定（偏差小）|
| 0 | 0.999 | 稳定 | 轻微不稳定 |
| 0 | 0.9999 | 稳定 | **发散** |
| 0.9 | 0.99 | 稳定 | 低 LR 时不稳定 |
| 0.9 | 0.999 | 稳定 | **高度不稳定** |
| 0.9 | 0.9999 | 稳定 | **发散** |

当 $\beta_2 = 0.9999$ 且 $\alpha$ 约为 0.001 时，去除偏差校正直接导致发散。这验证了论文的设计：**$\beta_2$ 接近 1 时，偏差校正是必需的**。

### 3.4 语言建模与机器翻译

| 任务 | 数据集 | Adam 结果 |
|---|---|---|
| 语言建模 | Penn Treebank | 取得最低困惑度（PPL） |
| 机器翻译 | WMT'14 英法 | 超越 Adadelta 和 SGD |

## 四、局限性与挑战

### 4.1 泛化差距：Adam vs SGD

一个已知问题是：在某些视觉任务上，Adam 的泛化效果不如带动量的 SGD。假设是：Adam 的自适应学习率可能导致模型收敛到更尖锐的极小值（flat minima 不足 = 泛化更差）。

| 任务 | Adam | SGD + Momentum | 胜出方 |
|---|---|---|---|
| ImageNet 分类（ResNet） | 23.5% top-1 错误率 | **22.8%** | SGD |
| CIFAR-10（Wide ResNet） | 4.2% | **3.9%** | SGD |
| PTB 语言建模 | **76.4 PPL** | 79.1 PPL | Adam |
| WMT 翻译 | **24.3 BLEU** | 23.8 BLEU | Adam |
| Transformer 训练 | **AdamW** 是标准 | SGD 发散 | AdamW |

实践中，Adam/AdamW 主导 NLP、生成模型和多模态系统，而 SGD 在图像分类领域仍有竞争力。

### 4.2 显存开销

Adam 为每个参数额外存储 2 个值（$m_t$ 和 $v_t$）：

| 模型规模 | 参数量（bf16） | Adam 优化器状态 | 总计（模型 + 优化器）|
|---|---|---|---|
| 7B | 14 GB | 28 GB（$m_t$+$v_t$，fp32） | 42 GB |
| 13B | 26 GB | 52 GB | 78 GB |
| 70B | 140 GB | 280 GB | 420 GB |

这一显存开销是关键瓶颈。缓解方案：
- **bitsandbytes 8-bit Adam**：将 7B 模型的优化器状态降至约 7 GB
- **Adafactor**：消去 $m_t$ 存储（在维度间分解）
- **Lion**：只跟踪动量，无二阶矩
- **Sophia**：利用 Hessian 信息，声称减少 2 倍步数

### 4.3 超参数敏感性分析

| 超参数 | 默认值 | 敏感性 | 常见调整 |
|---|---|---|---|
| $\alpha$（学习率） | 0.001 | **高** | LoRA 微调 2e-5，从头训练 1e-4 |
| $\beta_1$（动量衰减） | 0.9 | 低-中 | 0.95 有时有助于更平滑的训练 |
| $\beta_2$（平方衰减） | 0.999 | 低-中 | 噪声梯度用 0.995，快速适应用 0.99 |
| $\epsilon$（数值稳定） | 1e-8 | 低 | fp16/bf16 训练用 1e-6 防止除零 |
| Weight decay（AdamW） | 0.01 | 中 | 大模型用 0.1，微调用 0.001 |

**$\alpha$ 是最敏感的超参数**，通常需要 grid search 或在 log 尺度上调优。$\beta_1$ 和 $\beta_2$ 的默认值（0.9, 0.999）在大多数任务上表现良好，仅在特殊情况下需要调整。

### 4.4 二阶矩估计在训练后期的退化

随着训练进行，$\sqrt{\hat{v}_t}$ 的值持续衰减（因为梯度的方差随时间下降）。当 $\sqrt{\hat{v}_t}$ 变得非常小（接近 $\epsilon$ 量级）时，$\alpha / (\sqrt{\hat{v}_t} + \epsilon)$ 中的分母被 $\epsilon$ 主导，自适应学习率的调节效果减弱。这导致 Adam 在训练后期退化为近似动量 SGD，失去了自适应特性。

## 五、与后续工作/对领域的影响

### 5.1 AdamW：Transformer 优化器的关键修正

AdamW（Loshchilov & Hutter, 2017）发现了一个 Adam 权重衰减实现中的微妙 bug。

**问题所在**：在标准 SGD 中，L2 正则化和权重衰减等价：
$$\theta_t = \theta_{t-1} - \alpha \cdot (\nabla L + \lambda \theta_{t-1}) = \theta_{t-1}(1 - \alpha \lambda) - \alpha \nabla L$$

但在 Adam 中，更新变为：
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t + \lambda \theta_{t-1}}{\sqrt{\hat{v}_t} + \epsilon}$$

权重衰减项 $\lambda \theta_{t-1}$ 被 **$\sqrt{\hat{v}_t}$ 除**了——意为历史梯度大的参数（$\hat{v}_t$ 小）获得更弱的正则化，这是错误的。权重衰减理应对所有参数均等施加。

**AdamW 的修正**：
$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

权重衰减现在**解耦**于自适应学习率。这一看似微小的改动对 Transformer 至关重要，因为不同层（Embedding、Attention、FFN）的梯度尺度差异巨大。AdamW 已取代 Adam 成为所有 Transformer 训练的标准。

**L2 正则化 vs 解耦权重衰减的核心区别**：

| 特性 | L2 正则化（Adam 中的 $\lambda\theta$ 在梯度中） | 解耦权重衰减（AdamW） |
|---|---|---|
| 施加方式 | 作为梯度的一部分，被自适应学习率缩放 | 直接加在更新步上，独立于学习率自适应 |
| 对高梯度参数的影响 | 正则化被 $\sqrt{\hat{v}_t}$ 弱化 | 正则化均匀施加 |
| 对 Embedding 层效果 | 几乎无效（Embedding 梯度范数小） | 有效 |
| 理论正确性 | 在自适应方法中不正确 | 正确 |

### 5.2 AdamW 作为通用 VLA 优化器

所有主流 VLA 系统都使用 AdamW：

| VLA 系统 | 优化器 | 学习率 | 权重衰减 | 调度策略 |
|---|---|---|---|---|
| **OpenVLA** | AdamW | 2e-5 | 0.01 | Cosine，500 步 warmup |
| **Diffusion Policy** | AdamW | 1e-4 | 0.01 或 0.001 | Cosine + warmup |
| **RT-2** | AdamW | 3e-5 | 0.01 | Co-Fine-Tuning 调度 |
| **FLOWER** | AdamW | 3e-4（预训练），3e-5（微调）| 0.01 | Cosine 衰减 |
| **pi-zero** | AdamW | 视具体配置 | 视具体配置 | 通过 JAX optax |
| **Octo** | AdamW | 3e-4 | 0.01 | Cosine，3000 步 warmup |

**AdamW 主导 VLA 的原因**：

1. **Transformer 骨干的继承性**：[[Llama 2]] 用 AdamW 训练，VLA 微调必须继承相同优化器
2. **多模态梯度多样性**：图像、文本和动作梯度的尺度完全不同——自适应学习率至关重要
3. **SwiGLU 激活函数的敏感性**：[[Llama 2]] 使用 SwiGLU，其梯度分布比 ReLU 更广，受益于逐参数控制
4. **LoRA 兼容性**：LoRA + AdamW 是标准微调配置

### 5.3 为什么 SwiGLU 需要比 ReLU 更小的学习率

SwiGLU（[[Llama 2]]、PaLM、Gemini 使用）的计算方式：
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

Swish 激活函数 $\text{Swish}(x) = x \cdot \sigma(x)$ 具有以下特性：
- 对负输入产生非零梯度（不同于 ReLU 的精确 0）
- 由于乘法门控结构，梯度方差大于 ReLU

这意味着：
- 使用 SGD 时，SwiGLU 网络更难调参（单一全局学习率无法同时适应两个投影层）
- Adam 的逐参数学习率自动适应：门控权重 $W_2$ 可能需要与投影权重 $W_1$ 不同的有效学习率
- **LR = 2e-5 对 OpenVLA 微调有效**，是因为 Adam 的自适应机制处理了不同权重的学习率差异

**SwiGLU vs ReLU 的优化器配置对比**：

| 激活函数 | 推荐优化器 | 典型学习率 | 学习率敏感度 | 原因 |
|---|---|---|---|---|
| **ReLU/GELU** | AdamW | 1e-4 ~ 3e-4 | 中 | 负输入值精确为零，梯度稀疏 |
| **SwiGLU** | AdamW | 1e-5 ~ 3e-5 | **高** | 门控结构产生更密集、方差更大的梯度分布 |

## 六、对你的启示/硬件兼容性

### 6.1 实际 VLA 微调配置

**推荐的 VLA 微调默认配置**：

```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,           # VLA 微调的安全起点
    betas=(0.9, 0.999), # 标准默认值
    eps=1e-8,           # fp16 训练时提高至 1e-6
    weight_decay=0.01   # Llama 2 默认值
)

# 调度策略：warmup + cosine 衰减
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
)
```

**关键建议**：
- **始终使用 AdamW，不要使用 Adam** ✅：AdamW 的权重衰减解耦是正确方法，尤其对 Transformer
- **VLA 微调的默认 LR 设为 2e-5** ✅：大多数 VLA 系统（OpenVLA、RT-2）使用此值，是经实验验证的安全起点
- **$\epsilon$ 在 bf16 训练时提高到 1e-6** ✅：防止低精度下除零导致的 NaN

### 6.2 显存优化指南

对于 7B VLA 模型在消费级 GPU 上的显存需求：

| 设置 | 模型参数 | 优化器状态 | 梯度 | 总显存 | 可行 GPU |
|---|---|---|---|---|---|
| 全参数微调（fp32） | 28 GB | 56 GB | 28 GB | 112 GB | A100 80GB ✅ |
| 全参数微调（bf16） | 14 GB | 28 GB | 14 GB | 56 GB | A100/H100 ✅ |
| **LoRA + bf16 AdamW** | 14 GB | 28 GB（基础）+ 0.3 GB（LoRA）| 14 GB | ~48 GB | A100 ✅ |
| **LoRA + 8-bit AdamW** | 14 GB | **~7 GB** | 14 GB | ~36 GB | **RTX 4090 24GB** ✅ |
| **QLoRA + 4-bit Adam** | ~3.5 GB | ~1 GB | ~3.5 GB | **~8 GB** | **RTX 3090 24GB** ✅ / **RTX 4070 Ti Super 16GB** ✅ |

**针对 RTX 3090（24GB）和 RTX 4070 Ti Super（16GB）的显存优化建议**：

1. **使用 4-bit QLoRA 量化** ✅：将模型参数降至 ~3.5 GB，配合 8-bit AdamW 优化器状态约 1 GB，总显存约 8 GB，16GB 和 24GB 的 GPU 均可流畅运行
2. **使用 LoRA + 8-bit AdamW** ✅：24GB 显卡可以完整支持
3. **16GB 显卡避免全参数微调** ❌：即使使用 bf16，全参数微调也需要 56 GB，远超 16GB 上限
4. **梯度累积（gradient accumulation）** ✅：在 batch size 受限时，使用梯度累积步数模拟更大 batch

### 6.3 超参数调优协议

**当损失剧烈震荡时**：
1. 学习率减半（2e-5 -> 1e-5）
2. 将 $\epsilon$ 提高到 1e-6（增强数值稳定性）
3. 增加 warmup 步数比例

**当损失收敛过慢时**：
1. 学习率加倍（2e-5 -> 5e-5）
2. 检查 $\beta_2$ 是否太接近 1（尝试 0.99 而非 0.999）
3. 在显存允许时增大 batch size

**当出现过拟合时**：
1. 增大 weight_decay（0.01 -> 0.1）
2. 添加梯度裁剪（max_norm = 1.0）
3. 降低学习率 / 增大 LoRA dropout

### 6.4 监控 $v_t$ 进行调试

训练中若出现突发的 loss spike，检查有效步长：

```python
# 伪代码：监控有效步长比例
for name, param in model.named_parameters():
    if param.grad is not None:
        ratio = m_t[name] / (sqrt(v_t[name]) + eps)
        # 若任何参数的比例 > 10，学习不稳定
```

**常见错误排查**：
- **loss NaN** -> 检查 $\epsilon$ 是否过小（bf16 下 1e-8 太小），改为 1e-6
- **验证集 loss 不下降** -> 学习率可能太小，检查 warmup 结束时 LR 是否达到预期值
- **训练初期 loss 发散** -> 偏差校正在 $\beta_2$ 接近 1 时失效，检查是否有参数覆盖了默认 $\beta_2$
- **RMSNorm + AdamW 下的梯度爆炸** -> 检查 Pre-LN 结构是否正确（归一化在子层前，非子层后）

### 6.5 硬件兼容性汇总

| GPU 型号 | 显存 | Adam/AdamW 兼容性 | 推荐配置 |
|---|---|---|---|
| **RTX 3090** | 24GB | ✅ | QLoRA + 8-bit AdamW 或 LoRA + 8-bit AdamW |
| **RTX 4070 Ti Super** | 16GB | ✅ | QLoRA + 8-bit AdamW（约 8GB） |
| **RTX 4090** | 24GB | ✅ | LoRA + 8-bit AdamW 或 LoRA + bf16 AdamW（带梯度累积）|

## PDF

[[Adam 原文.pdf]]
