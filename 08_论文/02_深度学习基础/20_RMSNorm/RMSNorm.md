---
tags:
  - 论文
  - 归一化技术
  - 高效架构
  - Transformer组件
  - Llama
  - VLA基础设施
created: 2026-06-30
paper_title: "Root Mean Square Layer Normalization"
paper_authors: "Biao Zhang, Rico Sennrich"
paper_year: 2019
paper_venue: "NeurIPS 2019"
paper_citations: "~2,500+"
paper_url: "https://arxiv.org/abs/1910.07467"
github: "https://github.com/bzhangGo/rmsnorm"
---

# RMSNorm

**Root Mean Square Layer Normalization**
*Biao Zhang, Rico Sennrich | University of Edinburgh, University of Zurich | NeurIPS 2019 | arXiv 1910.07467*

> RMSNorm 是对 Layer Normalization 的简化，去掉了均值中心化步骤，仅通过均方根统计量进行归一化。相比 LayerNorm 节省约 7-15% 的计算量，同时保持等价甚至更优的效果。它已成为所有主流开源 LLM（[[Llama 2]]、Llama 3、Mistral、Qwen、Gemma）的默认归一化方案，也是 [[OpenVLA]] 骨干网络的标准组件，对现代 VLA 研究具有基础性意义。

---

## 一、研究背景与核心思想

### 1.1 LayerNorm 的计算开销问题

[[Layer Normalization]] 通过均值和标准差归一化激活值，其完整计算包括两个统计量的求取：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$$

计算 $\mu$ 需要一次遍历输入，计算 $\sigma$ 需要第二次遍历来求 $(x_i - \mu)^2$。对于 32-80 层的深度 Transformer，这种两遍计算的累积开销不可忽视。更重要的是，该方法需要同时维护两个统计量——均值和标准差——而 RMSNorm 的核心洞察是：**均值中心化（re-centering）在 Transformer 架构中可能是不必要的**。

### 1.2 理论动机：均值中心化为何可舍去

论文从三个角度论证了 re-centering 操作对训练成功的贡献极小：

1. **残差连接提供隐式中心化**：在 Transformer 块 $x^{(l+1)} = x^{(l)} + F(x^{(l)})$ 中，残差连接自然使激活值的均值 $\mu$ 随时间推移趋近于零，无需显式减去均值。残差路径的累积效应天然抑制了激活值的偏移。

2. **权重衰减控制偏置**：对权重的 L2 正则化（weight decay）在不直接约束激活值的情况下，间接将激活值的均值推向零。这是因为权重衰减惩罚权重矩阵的范数，而大权重的特征是梯度更新方向上存在系统性偏移。

3. **$\beta$ 偏置参数的梯度极小**：实证发现 $\partial \mathcal{L}/\partial \beta$ 在多数更新步中接近于零。这意味着 LayerNorm 中可学习的偏置参数 $\beta$ 几乎没有贡献额外的表示能力，移除它不会影响模型性能。

### 1.3 与 [[Weight Normalization]] 的联系

论文指出了 RMSNorm 与 [[Weight Normalization]]（Salimans & Kingma, 2016）之间的理论联系。Weight Normalization 通过向量范数解耦权重的方向和大小，而 RMSNorm 在激活层面实现了类似效果。关键区别在于：

- **Weight Normalization**：通过重参数化权重向量来控制梯度的范数，需要修改网络结构
- **RMSNorm**：在激活值层面进行归一化，无需改变底层网络结构，可以直接替换 LayerNorm

这种设计使 RMSNorm 成为 LayerNorm 的即插即用替代方案——无需修改任何层结构，只需替换归一化函数本身。

## 二、方法/架构/技术贡献

### 2.1 RMSNorm 变换的定义

RMSNorm 去除了均值中心化步骤，仅通过均方根统计量进行归一化：

$$\overline{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

其中 RMS 统计量定义为：

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

与 LayerNorm 的关键差异：
- **无 $\mu$ 计算**：不需要计算均值，也不需要计算 $(x_i - \mu)^2$ 的偏差
- **无 $\beta$ 参数**：仅保留可学习的增益参数 $\gamma$（每特征维度一个），去掉了偏置参数 $\beta$
- **单遍计算**：仅需一次遍历即可完成归一化

完整的 RMSNorm 表达式为：

$$y = f\left(\frac{Wx}{\text{RMS}(a)} \odot g + b\right)$$

其中 $a = Wx$ 是线性变换后的求和输入，$g$ 是增益向量（初始化为 1），$b$ 是偏置（初始化为 0），$f$ 是非线性激活函数。

### 2.2 RMSNorm 与 LayerNorm 的全面对比

| 属性 | LayerNorm | RMSNorm |
|---|---|---|
| **均值计算** | $\mu = \frac{1}{d}\sum x_i$——需要一次遍历 | 不需要 |
| **方差计算** | $\sigma^2 = \frac{1}{d}\sum (x_i - \mu)^2$——需要第二次遍历 | 不需要 |
| **RMS 计算** | 不使用 | $\text{RMS} = \sqrt{\frac{1}{d}\sum x_i^2}$——单遍完成 |
| **均值中心化** | $x - \mu$ | 无 |
| **缩放因子** | $\sigma$（标准差） | $\text{RMS}$（均方根） |
| **可学习增益** | $\gamma$ | $\gamma$（相同） |
| **可学习偏置** | $\beta$ | 无 |
| **计算开销** | 约为 RMSNorm 的 2 倍 | 低 7-64%（取决于架构） |

**关于 RMS 与 std 的重要差异**：对于零均值信号，$\text{RMS}(x) = \text{std}(x)$。对于非零均值信号，$\text{RMS}(x)^2 = \text{std}(x)^2 + \mu^2$，这意味着 RMSNorm **间接惩罚了偏移的均值**——即使不显式减去均值，RMS 统计量也隐式包含了均值大小的信息。

### 2.3 不变性分析

论文提供了严格的理论分析，对比了 RMSNorm 与其他归一化方法的不变性特性：

| 方法 | 权重矩阵重缩放 | 权重矩阵重中心化 | 权重向量重缩放 | 数据集重缩放 | 数据集重中心化 | 单样本重缩放 |
|---|---|---|---|---|---|---|
| **BatchNorm** | 不变 | 否 | 不变 | 不变 | 不变 | 否 |
| **WeightNorm** | 不变 | 否 | 不变 | 否 | 否 | 否 |
| **LayerNorm** | 不变 | 不变 | 否 | 不变 | 否 | 不变 |
| **RMSNorm** | **不变** | **否** | 否 | 不变 | 否 | 不变 |
| **pRMSNorm** | **不变** | **否** | 否 | 不变 | 否 | 不变 |

RMSNorm 对**权重矩阵重缩放**不变（若 $W' = \delta W$，输出不变），但对**权重矩阵重中心化**没有不变性（与 LayerNorm 不同）。这是因为 RMS 具有线性特性 $\text{RMS}(\delta a) = \delta \cdot \text{RMS}(a)$，但不具备中心化特性。

### 2.4 梯度分析与隐式学习率自适应

论文推导了 RMSNorm 的梯度公式。对于输出 $\overline{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma$：

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \overline{x}_i} \cdot \frac{\gamma_i}{\text{RMS}(x)} - \frac{\gamma_i \cdot x_i}{d \cdot \text{RMS}(x)^3} \sum_{j} \frac{\partial \mathcal{L}}{\partial \overline{x}_j} \cdot \overline{x}_j$$

$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial v} \odot \frac{Wx}{\text{RMS}(a)}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial v}$$

**关键发现**：权重矩阵的梯度中包含一个项 $\mathbf{R} = \frac{1}{\text{RMS}(a)}(\mathbf{I} - \frac{(Wx)(Wx)^T}{d \cdot \text{RMS}(a)^2})$，该矩阵与输入和权重矩阵的缩放**负相关**。这充当了**隐式学习率适配器**：当梯度范数过大时自动缩小更新步长，避免权重矩阵范数失控，从而提高收敛稳定性。

### 2.5 部分 RMSNorm（pRMSNorm）变体

由于同一层中的神经元常具有独立同分布（i.i.d.）结构，RMS 可以从神经元的子集来估计。pRMSNorm 从求和输入的前 $p\%$ 部分估计 RMS：

$$\text{RMS}(a) = \sqrt{\frac{1}{k} \sum_{i=1}^{k} a_i^2}, \quad k = \lceil d \cdot p \rceil$$

当 $p = 6.25\%$ 时，pRMSNorm 仅使用 1/16 的神经元计算 RMS，但仍保持竞争力——这得益于 RMS 的线性特性仍然成立。该变体进一步减少了计算量，但后续实践表明其带来的 1-3% 额外加速不值得引入的复杂度。

## 三、实验与关键发现

### 3.1 机器翻译任务（WMT14 英德互译）

**基于 GRU 的 RNNSearch 模型**：

| 模型 | newstest2014 BLEU | newstest2017 BLEU | 每 1K 步耗时 | 相对 LN 加速比 |
|---|---|---|---|---|
| Baseline | 21.7 | 23.4 | 399s | - |
| LayerNorm | 22.6 | 23.6 | 665s | baseline |
| **RMSNorm** | **22.4** | **23.7** | **501s** | **24.7%** |
| pRMSNorm | 22.6 | 23.1 | 493s | 25.9% |

RMSNorm 在 BLEU 分数上持平或略优于 LayerNorm，同时训练速度快约 25%。RNN 上加速比尤为显著，因为 TensorFlow 中 LayerNorm 对循环架构的实现效率较低。

**Transformer 模型（6 层 base 配置）**：

| 模型 | newstest2014 BLEU | newstest2017 BLEU | 每 1K 步耗时 | 相对 LN 加速比 |
|---|---|---|---|---|
| Baseline | 发散 | 发散 | 210s | - |
| LayerNorm | 26.6 | 27.7 | 248s | baseline |
| **RMSNorm** | **26.8** | **27.7** | **231s** | **6.9%** |
| pRMSNorm | 26.5 | 27.8 | 225s | 9.3% |

Transformer 上的加速比降至 6.9%，因为归一化层在总计算量中的占比远小于 Multi-Head Attention 和 FFN。但重要的是，无归一化的 Baseline 完全发散，说明归一化对 Transformer 训练是必要的。

### 3.2 阅读理解与图像-文本检索任务

**CNN 语料库阅读理解（Attentive Reader, LSTM 架构）**：

| 模型 | 验证错误率 | 每 0.1K 步耗时 | 相对 LN 加速比 |
|---|---|---|---|
| Baseline | 最高 | 315s | - |
| BatchNorm-LSTM | 较低 | 345s | - |
| LayerNorm | ~0.83 | 392s | baseline |
| **RMSNorm** | **~0.83** | **333s** | **15.1%** |
| pRMSNorm | ~0.84 | 330s | 15.8% |

**图像-文本检索（Order-Embeddings 模型）**：

| 模型 | Caption R@1 | Caption R@5 | Caption R@10 | Image R@1 | 每 0.1K 步耗时 | 加速比 |
|---|---|---|---|---|---|---|
| Baseline | 45.8 | 79.7 | 88.8 | 37.6 | 2.11s | - |
| LayerNorm | 47.9 | 79.5 | 89.2 | 38.4 | 12.02s | baseline |
| **RMSNorm** | **48.7** | **79.7** | **89.5** | **39.0** | **7.12s** | **40.8%** |
| pRMSNorm | 46.8 | 79.8 | 90.3 | 39.0 | 4.34s | 63.9% |

在基于 Theano 的实现中，RMSNorm 的加速比达到 40.8%，因为 Theano 对 LayerNorm 在 GRU 单元中的实现特别慢。pRMSNorm 更是达到了 63.9% 的加速。

### 3.3 CIFAR-10 图像分类（卷积网络）

| 模型 | 测试错误率 | 每轮耗时 | 相对 LN 加速比 |
|---|---|---|---|
| Baseline | 8.96% | 21s | - |
| BatchNorm | **8.25%** | 38s | - |
| LayerNorm | 10.49% | 39s | baseline |
| **RMSNorm** | **8.83%** | **31s** | **20.5%** |
| pRMSNorm | 10.37% | 30s | 23.1% |

RMSNorm 在 CNN 上显著优于 LayerNorm（8.83% vs 10.49%），并快 20.5%。论文指出："虽然 LayerNorm 相比 Baseline 缩短了收敛时间，但在测试集上未能泛化。"这表明**去掉均值中心化实际上有助于 CNN 的泛化**。BatchNorm 仍然是 CNN 的最优选择，但 RMSNorm 在 LayerNorm 不适用的情况下提供了更好的替代。

### 3.4 隐层表示的均值/标准差分析

论文分析了 RNNSearch 模型中隐层表示的分布，以解释为何 RMSNorm 尽管不归一化均值仍然有效：

| 模型 | 均值（全部） | 标准差（全部） | 均值（位置 1） | 标准差（位置 1） |
|---|---|---|---|---|
| Baseline | -1.60 | 3.04 | -2.60 | 7.35 |
| LayerNorm | -0.51 | 1.51 | -0.43 | 1.19 |
| **RMSNorm** | **-0.73** | **1.50** | **-0.40** | **1.27** |

Baseline 在不同时间步上的均值和标准差变化剧烈（位置 1 均值 -2.60，全部均值 -1.60）。LayerNorm 和 RMSNorm 都能稳定标准差（1.50-1.51），而且出乎意料的是，RMSNorm **虽然没有显式归一化均值，却也稳定了均值**。这从实验上支持了论文的核心假说：re-centering 是不必要的。

## 四、局限性与挑战

### 4.1 理论层面的局限

1. **缺乏严格的理论证明**：论文对 re-centering 可舍去性的论证主要基于实证而非严格数学证明。后续工作（如 Pre-LN 结构分析）才逐步对这一现象进行了更深入的理论刻画。

2. **小规模实验验证**：原始论文仅在 6 层 Transformer（base 配置）上进行测试。RMSNorm 在数百亿参数模型上的表现（后来被 Llama 系列证实）未在原始论文中验证，其在大规模场景下的行为存在不确定性。

3. **任务多样性有限**：主要实验集中在 NLP（翻译、阅读理解、检索）和一个小型 CNN 实验，未在强化学习、语音识别、多模态等任务上测试。特别是在需要精细的数值稳定性的场景下，RMSNorm 的表现尚未被探索。

4. **$\epsilon$ 敏感性未分析**：数值稳定参数 $\epsilon$ 在与 RMS 和 std 交互时表现不同，但论文未分析这一差异的影响。

### 4.2 实际部署的权衡

| 考量维度 | RMSNorm 优势 | RMSNorm 劣势 |
|---|---|---|
| **计算效率** | 快 7-64%（取决于架构） | — |
| **实现简洁性** | 无需均值和方差计算 | — |
| **理论保证** | 具备重缩放不变性 | 缺乏重中心化不变性 |
| **实证支持** | 在测试任务上持平或超越 LN | 未在极深模型上验证（原始论文） |
| **数值稳定性** | 避免 $x-\mu$ 的精度损失，bf16 下更稳定 | 在极高噪声场景下尚不明确 |

## 五、与后续工作/对领域的影响

### 5.1 Llama 系列的默认归一化方案

RMSNorm 被 Meta 的 Llama 系列采用后，成为开放权重 LLM 的事实标准归一化方案：

| 模型 | 归一化方式 | 参数量 | 来源 |
|---|---|---|---|
| **Llama 1** | RMSNorm | 7B-65B | Meta（2023）|
| **Llama 2** | RMSNorm | 7B-70B | Meta（2023）|
| **Llama 3** | RMSNorm | 8B-405B | Meta（2024）|
| **Mistral 7B** | RMSNorm | 7B | Mistral AI |
| **Qwen 2.5** | RMSNorm | 0.5B-72B | Alibaba |
| **Gemma** | RMSNorm | 2B-7B | Google |
| **OLMo** | RMSNorm | 1B-7B | AI2 |

Llama 3 405B 模型包含 118 层 Transformer，每层 2 个 RMSNorm，共计 236 个 RMSNorm 实例——这充分验证了 RMSNorm 在大规模训练中的稳定性。

### 5.2 RMSNorm 在 VLA 系统中的核心地位

所有使用现代 LLM 骨干网络的 VLA 系统都继承了 RMSNorm：

| VLA 系统 | 骨干网络 | 归一化方式 | 说明 |
|---|---|---|---|
| **OpenVLA** | Llama 2 7B | RMSNorm | Pre-LN，每 Transformer 块 2 个 RMSNorm |
| **pi-zero** | PaliGemma | RMSNorm（通过 Gemma）| Gemma 所有层使用 RMSNorm |
| **RT-2** | PaLM-E | RMSNorm（通过 PaLM）| PaLM 论文明确指定 RMSNorm |
| **EmbodiedGPT** | LLaMA-Adapter | RMSNorm | Adapter 保留骨干网络归一化 |
| **RoboFlamingo** | MPT/OpenFlamingo | RMSNorm（通过 MPT）| MPT 系列使用 RMSNorm |

在 **OpenVLA** 中，32 层 Llama 2 的每层结构为：

```
x → RMSNorm → Attention → residual + x → RMSNorm → FFN（SwiGLU）→ residual + x
```

每层 2 个 RMSNorm 操作，单次前向传播共 **64 次 RMSNorm 应用**。

### 5.3 归一化技术的演化链：LN → RMSNorm → AdaLN

```
LayerNorm（2016）
  μ + σ 双统计量归一化，可学习 γ + β
  ⊥
  ├─> RMSNorm（2019）                         ──> Llama 系列、OpenVLA
  │    仅 RMS 归一化，仅可学习 γ
  │    比 LN 快 7-15%
  │    ⊥
  │    └─> AdaLN（DiT, 2023）                ──> pi-zero、FLOWER
  │         γ 和 β 由条件信号（t、类别标签）预测
  │         当归一化同时作为条件注入接口时使用
  │         ⊥
  │         └─> Cross-Attention LN（FLOWER, 2024）
  │              AdaLN 参数由 t 和视觉特征共同调节
  │
  ├─> Pre-LN（2018）                           ──> 所有现代 Transformer
  │    将 LN 从子层后移到子层前
  │    解决了 Post-LN 的梯度爆炸问题
  │
  └─> Sandwhich-LN（Llama 3 405B 训练）        ──> 大规模训练优化
      在 Attention 和 FFN 后额外加一个 LN
      防止输出 logits 随深度增长而发散
```

**每步进化的核心动机**：

1. **BN → LN**：解决 BN 对 batch size 敏感、不适用于 RNN/Transformer 的问题，消除对 batch 维度的依赖
2. **LN → RMSNorm**：消除不必要的均值中心化计算，降低计算开销，同时保持等价效果
3. **RMSNorm → AdaLN**：当归一化参数需要由条件信号（扩散时间步、类别标签）动态生成时，将 $\gamma$、$\beta$ 变为条件依赖的函数

## 六、对你的启示/硬件兼容性

### 6.1 实践建议

1. **所有新项目都用 RMSNorm 替换 LN** ✅：在 PyTorch 中 RMSNorm 是 LayerNorm 的零成本替代。在 Transformer 中，没有任何场景下 LN 的效果稳定且显著优于 RMSNorm，值得额外付出 7-15% 的计算代价。

2. **不要使用 pRMSNorm** ⚠️：pRMSNorm 带来的 1-3% 额外加速不值得增加的实现复杂度和轻微的质量下降。现代 GPU 对规整计算比对部分计算更友好。

3. **LoRA 微调时可以冻结 RMSNorm 的 $\gamma$ 参数** ✅：在 LoRA 微调时，冻结 $\gamma$ 参数不会造成显著的质量损失，可节省约 0.1-0.2% 的可训练参数量。

4. **bf16/fp16 训练时 RMSNorm 数值更稳定** ✅：RMSNorm 避免了 $x - \mu$ 操作，该操作在 $\mu$ 较大时会损失低精度表示下的有效数值范围。

5. **注意 $\epsilon$ 参数的设置** ⚠️：RMSNorm 的 $\epsilon$ 直接加在 RMS 分母上（而非 LayerNorm 的方差上）。在 bf16 训练下推荐使用 $\epsilon = 1 \times 10^{-5}$ 或 $1 \times 10^{-6}$（与 Llama 官方实现一致），而非默认的 $10^{-8}$。

### 6.2 PyTorch 实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**重要说明**：
- Llama 官方实现使用 $\epsilon = 1 \times 10^{-5}$（比默认的 $10^{-8}$ 更大），这在 bf16 精度下更稳定
- RMSNorm 遵循 Pre-LN 结构：放置在子层**之前**（Pre-LN），而非子层之后
- 非线性激活函数**不在** RMSNorm 之后应用，RMSNorm 的输出直接送入 Attention 或 FFN

### 6.3 计算开销细分解

以 Llama 2 7B 模型（hidden_size=4096，32 层）为例：

| 操作 | 每 token 每层的 FLOPs | 占总计算比例 |
|---|---|---|
| RMSNorm（x2） | ~16K | <0.01% |
| Attention（QKV + 输出投影） | ~16M | ~20% |
| FFN（SwiGLU + 门控） | ~64M | ~80% |

RMSNorm 占总计算量的比例 <0.01%。论文中声称的 7-15% 加速比是**相对于 LayerNorm 本身的比较**，而非相对于完整 Transformer 模型。在 RNN 为主的实现中，归一化层占计算比重更大，加速效果也更显著。

**最终结论**：在现代 Transformer 中，RMSNorm 相比 LN 的主要优势不是速度，而是**简洁性和数值稳定性**。质量方面两者等价。

### 6.4 硬件兼容性评估

| GPU 型号 | 显存 | RMSNorm 兼容性 | 说明 |
|---|---|---|---|
| **RTX 3090** | 24GB | ✅ | RMSNorm 无额外显存开销，完全兼容 |
| **RTX 4070 Ti Super** | 16GB | ✅ | <0.01% 计算占比，不影响训练速度 |
| **RTX 4090** | 24GB | ✅ | 所有 VLA 微调脚本（OpenVLA、FLOWER）默认使用 RMSNorm |

在 24GB 或 16GB 显存的消费级 GPU 上训练/微调 VLA 模型时，RMSNorm 不会成为任何瓶颈。

## PDF

[[RMSNorm 原文.pdf]]
