---
tags:
  - 论文
  - 优化器
  - 训练技巧
  - 基础组件
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
*University of Amsterdam / University of Toronto | ICLR 2015 | arXiv: 1412.6980*

> 几乎所有深度学习模型（包括 VLA）的默认优化器。Adam 结合了动量（加速收敛 + 越过局部最优）和自适应学习率（每个参数有自己的学习率），并加入了偏差校正。从 CNNs 到 Transformers 到 VLA，几乎所有 SOTA 模型都在用 Adam 或其变体 AdamW。

---

## 一、研究背景与动机

在 Adam 之前，深度学习优化主要有两个流派：

1. **SGD + Momentum**：利用梯度的历史信息（一阶矩）加速收敛，但所有参数使用相同的学习率
2. **AdaGrad / RMSProp**：为每个参数自适应学习率（二阶矩），但缺乏动量机制

两者各有利弊：

| 方法 | 优势 | 劣势 |
|------|------|------|
| SGD + Momentum | 收敛稳定，泛化性好 | 需要手动调学习率，对稀疏梯度处理差 |
| AdaGrad | 自适应学习率，适合稀疏梯度 | 学习率单调递减，可能提前停止 |
| RMSProp | 滑动窗口自适应学习率 | 缺乏动量，收敛慢 |

**Adam (Adaptive Moment Estimation)** 的目标是：将两者的优势结合——既有 Momentum 的加速能力，又有 RMSProp 的自适应学习率，同时解决初期偏差问题。

## 二、核心方法

### 算法推导

Adam 维护两个状态变量：

#### 一阶矩估计（动量项）

$$ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $$

其中 $g_t = \nabla_\theta L_t(\theta_{t-1})$ 是当前步的梯度。$m_t$ 是梯度的指数移动平均，相当于 Momentum。

#### 二阶矩估计（自适应学习率项）

$$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $$

$v_t$ 是梯度平方的指数移动平均，相当于 RMSProp。学习了每个参数的学习率缩放。

#### 偏差校正

由于 $m_t$ 和 $v_t$ 初始化为零向量，在训练早期它们会偏向零。偏差校正补偿这一偏差：

$$ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $$

$$ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $$

注意 $\beta_1^t$ 随着 $t$ 增加迅速衰减到零，所以校正主要在早期步数发挥作用。

#### 参数更新

$$ \theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} $$

### 默认超参数

| 参数 | 默认值 | 作用 |
|------|:------:|------|
| $\alpha$ | 0.001 | 学习率（通常需要针对任务调优） |
| $\beta_1$ | 0.9 | 一阶矩的衰减率（动量系数） |
| $\beta_2$ | 0.999 | 二阶矩的衰减率 |
| $\epsilon$ | $10^{-8}$ | 防止除以零的数值稳定项 |

### AdamW 改进

AdamW (Loschilov & Hutter, 2017) 将权重衰减（weight decay）从梯度更新中解耦出来：

- **标准 Adam + 权重衰减**：$L_{\text{reg}} = L + \frac{\lambda}{2}||\theta||^2$（权重衰减耦合在梯度中）
- **AdamW**：$\theta_t = \theta_{t-1} - \alpha \cdot (\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_{t-1})$（权重衰减独立于自适应学习率）

**为什么这很重要？** 在 Transformer 中，AdamW 的解耦权重衰减显著提升了训练稳定性和最终性能——RMSNorm 的 scale 参数 $\gamma$ 和 SwiGLU 的权重矩阵尤其受益于解耦的正则化。

## 三、关键实验与发现

### 原始论文实验

| 任务 | 数据集 | Adam vs Baseline |
|------|--------|:-:|
| MNIST 分类 | MNIST | Adam 收敛更快，最终精度更高 |
| CIFAR-10 分类 | CIFAR-10 | Adam 超过 SGD 收敛速度 2-3× |
| 语言建模 | PTB | Adam 困惑度最低 |
| 机器翻译 | WMT'14 En-Fr | Adam 超过 Adadelta 和 SGD |
| 图像 captioning | MS COCO | Adam 训练更稳定 |

### 核心发现

1. **训练初期收敛速度极快**：偏差校正使得前几步的更新量合理有效
2. **超参数鲁棒性强**：$\alpha=0.001, \beta_1=0.9, \beta_2=0.999$ 在多数任务上有效
3. **适合非稳态目标**：在 RL 和在线学习中表现优异
4. **对梯度噪声不敏感**：在存在噪声估计的情况下（如小 batch），Adam 比 SGD 稳定得多

### 分析：何时 Adam 优于 SGD？

| 场景 | Adam 优势 | SGD 优势 |
|------|-----------|----------|
| 稀疏梯度 | √ 自适应学习率很好处理 | × |
| 大噪声梯度 | √ 动量 + 自适应平滑噪声 | × |
| CV (ImageNet) | × | √ 更好的泛化性 |
| NLP / LLM | √ 自适应学习率对嵌入层至关重要 | × |
| Transformer 训练 | √ (AdamW 是标准选择) | × |

## 四、局限性与后续影响

### 局限性

1. **在某些 CV 任务上泛化性不如 SGD + momentum**：特别是 ImageNet 分类
2. **存储开销大**：需要额外存储 $m_t$ 和 $v_t$（2 × 参数量）。对于 7B 模型，在 bf16 下需要 28GB 额外显存
3. **$\beta_2$ 对长尾 loss 敏感**：在稀疏梯度场景下，$\beta_2$ 太大可能导致高方差估计
4. **权重衰减实现有缺陷**（原始 Adam）：这是 AdamW 出现的原因
5. **学习率仍然需要调优**：虽然比 SGD 省事，但 $\alpha$ 仍然关键

### 后续影响

Adam 是引用量最高的深度学习论文之一（~200,000+），直接衍生出：
- **AdamW**：LLM/VLA 的默认优化器
- **AdaBelief**：关注二阶矩的离差
- **Lion**：Google 提出的更省显存的动量方法
- **Sophia**：使用梯度曲率信息的高效优化器

## 五、VLA/机器人研究中的角色

AdamW 是 VLA 训练的"统一优化器"：

| VLA 系统 | 优化器 | 学习率 | 细节 |
|----------|--------|:------:|------|
| **OpenVLA** | AdamW | 2e-5 | 复用 Prismatic VLM 超参，全部参数 |
| **Diffusion Policy** | AdamW | 1e-4 | 使用余弦学习率调度 + warmup |
| **RT-2** | AdamW | 3e-5 | Co-Fine-Tuning，冻结部分 ViT 参数 |
| **FLOWER** | AdamW | 3e-4 | 预训练；微调降至 3e-5 |
| **π0** | AdamW | 各种 schedule | 通过 JAX optax 实现 |

### 为何 VLA 普遍使用 AdamW？

1. **Transformer 骨干需要 AdamW**：Llama 2 模型在 AdamW 下训练，VLA 微调需继承相同优化器状态
2. **多模态训练的不稳定性**：图像特征 + 文本特征 + 动作特征的梯度尺度差异巨大，自适应学习率至关重要
3. **LoRA 微调与 AdamW 兼容**：LoRA + AdamW 是 VLA 微调的标准配置
4. **SwiGLU 对学习率敏感**：SwiGLU 激活函数的梯度分布比 ReLU 宽，需要 Adam 的自适应学习率来控制步长
5. **VLA 训练需要大量超参搜索**：但学习率可以锁定在 2e-5 ~ 3e-5 的范围

### 显存优化注意事项

对于 7B 模型的 VLA 训练：
- 标准 AdamW：模型参数 (14GB) + 优化器状态 (28GB) + 梯度 (14GB) = **~56GB 显存**
- 使用 bitsandbytes 8-bit Adam：优化器状态降至 **~7GB**（可以用 24GB 显卡微调）
- LoRA + 4-bit Adam：降至 **~8GB 总显存**（16GB GPU 足够）

## 六、对你的启示

1. **在 VLA 微调中永远选择 AdamW**：不使用 SGD 或原始 Adam。权重衰减的解耦对 Transformer 的 RMSNorm 和 embedding 层至关重要
2. **学习率是关键超参**：对于 VLA 微调，**2e-5 是一个安全的起点**。如果 loss 震荡，降到 1e-5；如果收敛太慢，升到 5e-5
3. **16GB GPU 的优化器选择**：使用 bitsandbytes 的 8-bit AdamW。在 LoRA 微调中，优化器状态占用的显存从 28GB 降到 < 5GB
4. **权重衰减设置**：VLA 微调时推荐 weight_decay = 0.01（这是 Llama 2 使用的设置），但如果 LoRA 的 rank 很小可以降到 0.001
5. **预热 + 余弦衰减是最佳实践**：warmup 5% steps + cosine annealing 是 VLA 微调的经验法则
6. **理解 $v_t$ 的作用有助于调试**：如果 loss 突然炸了，通常是 $v_t$ 中的某些参数对应梯度过大，可以临时增大 $\epsilon$ 到 $10^{-6}$ 或减小 $\beta_2$ 到 0.995

## PDF

[[Adam 原文.pdf]]
