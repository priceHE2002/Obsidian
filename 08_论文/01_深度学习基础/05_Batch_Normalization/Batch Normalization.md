---
tags:
  - 论文
  - 归一化技术
  - CNN架构
  - 训练加速
  - VLA基础设施
created: 2026-06-30
paper_title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
paper_authors: "Sergey Ioffe, Christian Szegedy"
paper_year: 2015
paper_venue: "ICML 2015"
paper_citations: "~60,000+"
paper_url: "https://arxiv.org/abs/1502.03167"
---

# Batch Normalization

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**
*Sergey Ioffe, Christian Szegedy | Google Inc. | ICML 2015 | arXiv 1502.03167*

> Batch Normalization 是深度 CNN 训练中最重要的稳定性技术。通过在每个 mini-batch 上将层输入归一化为零均值和单位方差，它允许使用 5-30 倍更高的学习率，降低对初始化的敏感性，提供正则化效应，并使得使用饱和非线性（sigmoid）训练深层网络成为可能。其关键局限——对 batch 统计量的依赖——使其不适用于 RNN 和 Transformer，进而催生了 LN、GN、RMSNorm 等一系列归一化变体。

---

## 一、研究背景与核心思想

Batch Normalization（BN）的诞生源于深度学习训练中的一个根本性问题：随着训练推进，浅层参数的更新会改变深层网络所接收到的激活分布，迫使深层网络持续适应不断变化的输入分布，从而严重拖慢训练进程。Ioffe 和 Szegedy 将这一现象命名为**内部协变量偏移（Internal Covariate Shift，ICS）**，并提出 BN 作为高效的解决方案。

### 1.1 内部协变量偏移（Internal Covariate Shift）问题

在 Batch Normalization 提出之前，训练深度神经网络面临一个根本性障碍：随着训练进行，浅层网络的参数不断更新，导致深层网络看到的激活值分布持续变化。Ioffe & Szegedy 将这一现象定义为**内部协变量偏移（Internal Covariate Shift，ICS）**。

数学上，考虑一层网络的计算 $x = Wu + b$，其中 $u$ 是前一层的输出。当底层网络的参数在训练过程中变化时，$u$（进而 $x$）的分布也随之变化。深层网络必须持续适应不断变化的输入分布，这带来了以下问题：

| 后果 | 解释 | 严重程度 |
|---|---|---|
| 训练缓慢 | 必须使用很小的学习率以避免激活分布剧烈震荡 | 高 |
| Sigmoid/Tanh 饱和 | 当 $\|x\|$ 很大时，梯度 $g'(x) \to 0$，导致梯度消失 | 致命 |
| 对初始化敏感 | 糟糕的初始化将激活值推入饱和区域 | 高 |
| 需要 Dropout 辅助正则化 | 梯度噪声大，需要额外正则化手段 | 中等 |

论文的论述非常直白："我们将深度网络内部节点分布在训练过程中发生的变化称为**内部协变量偏移**。消除它有望实现更快的训练。"

### 1.2 完整白化为什么不可行

解决 ICS 的理想方案是对每一层的输入进行白化（whitening）：计算协方差矩阵 $\text{Cov}[x] = \mathbb{E}_{x \in X}[xx^T] - \mathbb{E}[x]\mathbb{E}[x]^T$，然后使用其逆平方根 $\text{Cov}[x]^{-1/2}$ 对输入进行变换，得到 $\text{Cov}[x]^{-1/2}(x - \mathbb{E}[x])$。然而这一方案在实践中完全不现实：

1. **计算成本**：在每次更新时计算每层输入的协方差矩阵及其逆平方根，计算量极其巨大
2. **梯度复杂度**：通过白化变换进行反向传播需要计算 $\frac{\partial \text{Norm}(x, X)}{\partial x}$ 和 $\frac{\partial \text{Norm}(x, X)}{\partial X}$ 的雅可比矩阵——后者依赖于整个训练集
3. **不可微分性**：完整的白化操作难以以兼容 SGD 的方式进行微分

BN 做出了两个关键的简化：
- **逐维度归一化**替代完整白化：独立地将每个标量特征归一化为零均值和单位方差
- **Mini-batch 估计**替代全数据集统计：仅从当前 mini-batch 计算所需的统计量

### 1.3 核心洞察：可微分的可学习归一化

BN 最精妙的设计在于它**不是简单地做归一化**，而是引入了一对可学习参数 $\gamma$（缩放）和 $\beta$（偏移），使得归一化操作本身成为网络的一部分：

$$y_i = \gamma \cdot \hat{x}_i + \beta$$

这一设计的关键洞察：如果归一化移除了有用的信息（例如把所有 sigmoid 的输入都推到线性区域），网络可以通过学习 $\gamma = \sqrt{\text{Var}[x]}$、$\beta = \mathbb{E}[x]$ 来"撤销"归一化，恢复恒等变换。这意味着 BN 层**永远不会损害网络的表达能力**——最坏情况下它等价于不做任何操作。

此外，BN 将归一化参数也纳入反向传播，使得 $\gamma$ 和 $\beta$ 可以通过梯度下降与其他参数一起联合学习。这一点在后来的 [[Layer Normalization]] 和 [[RMSNorm]] 中得到了完全继承。

## 二、方法/架构/技术贡献

Batch Normalization 的核心包含四个递进操作：计算 mini-batch 均值、计算 mini-batch 方差、执行归一化、以及可学习的缩放与偏移。论文还完整推导了 BN 层的反向传播公式，并区分了训练阶段与推理阶段的不同行为模式。

### 2.1 Batch Normalization 变换

对于一个大小为 $m$ 的 mini-batch $\mathcal{B} = \{x_{1 \dots m}\}$，BN 变换由四个顺序操作组成：

**步骤一：计算 mini-batch 均值**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**步骤二：计算 mini-batch 方差**
$$\sigma^2_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**步骤三：归一化**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$
其中 $\epsilon$ 是为数值稳定性而添加的小常数（默认 $10^{-5}$），防止除以零。

**步骤四：缩放与偏移（可学习）**
$$y_i = \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta}(x_i)$$

可学习参数 $\gamma$ 和 $\beta$ 的维度与输入 $x$ 的特征维度一致。在 CNN 中，$\gamma^{(k)}$ 和 $\beta^{(k)}$ 是**逐通道**的——每个特征图（feature map）共享一对参数，从而保持卷积的平移等变性（translation equivariance）。

### 2.2 反向传播：梯度如何流过 BN 层

BN 层必须可微方能纳入端到端的 BP 训练。论文完整推导了链式法则：

$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \frac{-1}{2} (\sigma^2_\mathcal{B} + \epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial \mu_\mathcal{B}} = \left( \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \right) + \frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_\mathcal{B})}{m}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} + \frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \ell}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$

关键观察：梯度 $\partial \ell / \partial x_i$ 依赖于 mini-batch 中**所有**样本（通过 $\mu_\mathcal{B}$ 和 $\sigma^2_\mathcal{B}$），不仅仅是 $x_i$ 自身。这种批依赖性是 BN 行为的基础，但也同时是限制其适用性的根本原因。

### 2.3 训练与推理的行为差异

| 阶段 | 使用的统计量 | 关键特性 |
|---|---|---|
| **训练** | 当前 mini-batch 的 $\mu_\mathcal{B}$、$\sigma^2_\mathcal{B}$ | 随机性——归一化值随 batch 不同而变化 |
| **推理** | 训练时累积的 running mean $\mathbb{E}[x]$、running variance $\text{Var}[x]$ | 确定性——固定的线性变换 |

训练过程中，BN 维护**滑动平均（running averages）**（通常 momentum 取 0.9 或 0.99）：

$$\mathbb{E}[x] \leftarrow (1 - \text{momentum}) \cdot \mathbb{E}[x] + \text{momentum} \cdot \mu_\mathcal{B}$$
$$\text{Var}[x] \leftarrow (1 - \text{momentum}) \cdot \text{Var}[x] + \text{momentum} \cdot \sigma^2_\mathcal{B}$$

推理时使用固定统计量：

$$\text{BN}(x) = \gamma \cdot \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} + \beta$$

这种**训练-推理不一致性**是一个根本性局限。相比之下，[[Layer Normalization]] 和 [[RMSNorm]] 在训练和推理时行为完全相同。

### 2.4 BN 为什么允许使用 5-30 倍更高的学习率

论文通过实验证明了 BN 能够支持极高学习率，但最初并未给出完整的理论解释。后续工作（Santurkar et al., 2018, "How Does Batch Normalization Help Optimization?"）揭示了背后的机制：

1. **平滑损失景观（Loss Landscape Smoothing）**：BN 使损失函数的 Lipschitz 常数变小，意味着梯度变化不再剧烈
2. **防止参数变化放大**：BN 阻止了"参数的小变化被放大为激活值和梯度的次优大变化"这一链条
3. **防止陷入非线性激活的饱和区域**：归一化确保了激活值总保持在非饱和区域

实际效果：在 ImageNet 上，使用 BN 后学习率可以从 0.0015（Inception 基线）提升到 0.045（BN-x30），提升了 30 倍。

### 2.5 BN 的正则化效应

BN 具有微妙的**正则化效应**：由于每个 mini-batch 的 $\mu_\mathcal{B}$ 和 $\sigma^2_\mathcal{B}$ 存在采样噪声，归一化后的输出 $\hat{x}_i$ 也带有随机性。这相当于向每个隐藏层激活引入了轻微噪声，迫使网络不过度依赖特定激活模式——与 Dropout 的机制类似。

实验表明，使用 BN 后可以将 Dropout 比率从 0.5 降低到 0.2 甚至完全移除，同时保持甚至提升泛化性能。然而这一效应在 batch size 增大时会减弱（统计量估计更准确，噪声更小）。

## 三、实验与关键发现

论文通过三组核心实验系统验证了 BN 的效果：MNIST 上的 sigmoid 网络验证 ICS 假设，ImageNet 上的 Inception 网络展示实际收益，以及集成模型的 SOTA 结果。实验数据极为扎实，是 BN 被广泛采纳的关键。

### 3.1 MNIST Sigmoid 网络——ICS 假设的直接验证

第一个实验在 MNIST 上训练一个 3 层全连接网络（每层 100 个隐藏单元），使用 **sigmoid** 激活函数。这是对内部协变量偏移假设的直接验证：

| 指标 | 无 BN | 有 BN |
|---|---|---|
| 50K 步后的测试准确率 | 较低 | 更高（用相似步数达到更高精度） |
| 激活值分布稳定性 | 均值/方差随训练显著漂移 | 分布保持稳定 |
| Sigmoid 饱和程度 | 激活值逐渐漂移至饱和区 | 激活值始终维持在零均值附近 |

论文图 1(c) 展示了典型 sigmoid 的输入分布演化：没有 BN 时，{15%, 50%, 85%} 分位数发生剧烈漂移；而有 BN 时分布保持稳定。这直接验证了内部协变量偏移假说，也是论文最令人信服的诊断实验。

### 3.2 ImageNet 分类——Inception 架构

核心实验使用改进的 Inception 网络。论文测试了多种配置：

| 模型 | 学习率 | 达到 72.2% 精度的步数 | 最高精度 | 备注 |
|---|---|---|---|---|
| **Inception（基线）** | 0.0015 | 31.0M 步 | 72.2% | 基线模型 |
| **BN-Baseline** | 0.0015 | 13.3M 步 | 72.7% | 同学习率，BN 单独将步数减半 |
| **BN-x5** | 0.0075 | 2.1M 步 | 73.0% | 5 倍学习率，14 倍加速 |
| **BN-x30** | 0.045 | 2.7M 步 | 74.8% | 30 倍学习率，最高精度最高 |
| **BN-x5-Sigmoid** | 0.0075 | — | 69.8% | 用 sigmoid 替代 ReLU，仍可训练 |

关键发现：
- **BN-Baseline**（与 Inception 同学习率）仅需基线 43% 的训练步数，证明 BN 本身即可加速训练
- **BN-x5** 仅用 2.1M 步达到相同精度，加速约 14 倍
- **BN-x30** 达到最高精度 74.8%，高学习率可能帮助逃离尖锐局部极小值
- **BN-x5-Sigmoid** 达到 69.8%——此前训练深层 sigmoid 网络被认为几乎不可能

### 3.3 集成模型与 SOTA 结果

6 个 BN-Inception 模型的集成取得了以下成果：
- **4.9% top-5 验证错误率**（单裁剪：7.82%）
- **4.82% top-5 测试错误率**（100K 图像测试集）
- 这一结果**超越了当时的人类水平**

该集成特别值得注意的是，BN 的正则化效应使得每个独立模型更具多样性，从而带来更高的集成收益。

### 3.4 其他关键发现

1. **Dropout 去除**：BN 允许将 Dropout 比率从 0.5 降至 0.2 或完全移除
2. **Sigmoid 网络的复兴**：BN 使得基于 sigmoid 的深层网络可以训练——此前学界认为必须使用 ReLU
3. **BN 的放置位置**：论文实验验证，将 BN 放在非线性激活之前（$z = g(\text{BN}(Wu))$）效果优于归一化层输入 $u$

## 四、局限性与挑战

尽管 BN 取得了巨大成功，它存在若干致命缺陷——尤其是对 batch size 的依赖和训练-推理行为不一致——这些缺陷在后来的 Transformer 时代使其被迅速取代。

### 4.1 根本性局限

| 局限性 | 原因 | 严重程度 | 缓解方案 |
|---|---|---|---|
| 小 batch 不稳定 | 当 $m < 8$ 时 $\mu$ 和 $\sigma^2$ 估计噪声大 | 致命（大模型常见） | [[Group Normalization]] (Wu & He, 2018) |
| 训练/推理不一致 | 两阶段使用不同统计量 | 中等 | [[Layer Normalization]] 消除了此问题 |
| RNN 不兼容 | 不同时间步需要分开的统计量 | 根本性 | [[Layer Normalization]]、[[RMSNorm]] |
| Transformer 不兼容 | 变长序列破坏 batch 统计量 | 根本性 | Pre-LN、[[RMSNorm]] |

### 4.2 小 Batch 问题

当 batch size 很小时（例如 2 或 4，常见于分割任务或视频任务的大输入），BN 的统计量变得极度嘈杂。方差估计本身具有高方差，导致训练不稳定。这一问题在训练 Diffusion Policy 的视觉编码器时特别突出——EMA 更新与 BN 的 running stats 不兼容，导致 running stats 过时。

### 4.3 分布式训练中的 BN

标准 BN 不会跨 GPU 同步统计量——每个 GPU 只在其本地 mini-batch 上计算统计量。这意味着 BN 层的有效 batch size 仅为每 GPU batch size。PyTorch 提供了 `SyncBatchNorm`（`torch.nn.SyncBatchNorm`）来跨所有 GPU 进行 all-gather 统计量同步，但这需要额外的通信开销。在数据并行（Data Parallel）训练中，使用 SyncBN 通常能使精度提升 0.5-1.0%，但需要确保每个 GPU 的 batch size 仍然足够大。

## 五、与后续工作的关系/对领域的影响

BN 不仅自身成为深度学习基础设施，更触发了一整条归一化技术的演进链。从 BN 到 LN 到 RMSNorm 再到 AdaLN，每一次演进都在特定维度上改进了前代方法的不足。

### 5.1 归一化技术演进链

BN 的成功引发了一场归一化方法的爆发：

| 方法 | 归一化轴 | 提出年份 | 适用架构 | 关键贡献 |
|---|---|---|---|---|
| **Batch Normalization** | (N, H, W) 跨样本 | 2015 | CNN | 首次提出 ICS 和批归一化 |
| **[[Layer Normalization]]** | (C, H, W) 单样本内 | 2016 | RNN, Transformer | 消除 batch 依赖，统一训练推理 |
| **Instance Normalization** | (H, W) 单样本单通道 | 2016 | 风格迁移 | 逐实例/逐通道归一化 |
| **[[Group Normalization]]** | (H, W) 按组 | 2018 | 小 batch CNN | 在 batch size=2 时表现优于 BN |
| **[[RMSNorm]]** | 特征维 (仅 RMS) | 2019 | Transformer | 去除均值中心化，提速约 10% |
| **AdaLN** | 特征维（条件化） | 2023 | DiT | $\gamma, \beta$ 由条件信号预测 |

各方法沿不同轴计算统计量：
- **BN**：在 (N, H, W) 上归一化，逐通道 (C) ——跨样本和空间维度
- **LN**：在 (C, H, W) 上归一化，逐样本 (N) ——单样本内的特征维度
- **GN**：在 (H, W) 上归一化，逐 (N, G 组通道) ——介于 BN 和 LN 之间
- **IN**：在 (H, W) 上归一化，逐 (N, C) ——逐通道

### 5.2 BN 之后的理论理解演变

有趣的是，BN 最初以"减少内部协变量偏移"为动机，但 Santurkar et al.（2018）的后续研究显示，ICS 的减少可能**不是** BN 起作用的主要机制。实际上，BN 平滑了优化景观——使损失函数更加 Lipschitz 连续，梯度行为更加稳定。这一更深入的理论理解影响了所有后续归一化方法的设计。

### 5.3 BN 在 VLA 时代的应用版图

**BN 仍占主导的领域**：基于 CNN 的视觉编码器仍然是 BN 的堡垒。在 VLA 系统中：

- **ResNet 骨干网络**（广泛用于机器人学习管线）：默认使用 BN，ResNet-50 在每层卷积后都有 BN（共 54 个 BN 层）
- **Diffusion Policy** 论文明确指出 EMA 更新与 BN 的 running stats 不兼容，因此在其视觉编码器中将 BN 切换为 GroupNorm
- **ACT（Action Chunking with Transformers）** 中的 ResNet 编码器仍然使用 BN

**BN 已被取代的领域**：所有基于 Transformer 的 VLA 组件都使用 [[Layer Normalization]] 或 [[RMSNorm]]：

- **OpenVLA**：Llama 2 骨干网络完全使用 RMSNorm
- **RT-2**：PaLM-E 使用 RMSNorm 变体
- **pi-zero**：DiT 使用 AdaLN（以 timestep 为条件）
- **FLOWER**：使用 AdaLN 作为条件接口

## 六、对你的启示/硬件兼容性

针对 VLA 研究者以及 RTX 3090 24GB / RTX 4070 Ti Super 16GB 的实际硬件条件，以下是在归一化选择上的实用指南。

### 6.1 归一化选择速查表

| 场景 | 默认选择 | 理由 | GPU 适配 |
|---|---|---|---|
| CNN 编码器，batch size >= 16 | ✅ BatchNorm | 成熟高效，CUDA 深度融合 | 16GB 可跑 batch=32（ResNet-50） |
| CNN 编码器，batch size < 16 | ✅ GroupNorm | BN 统计噪声大，GN 与 batch 无关 | 16GB 可跑 batch=4（GN 不降质） |
| 任何 Transformer | ✅ RMSNorm（或 LayerNorm） | 序列长度无关，训练=推理 | 兼容所有精度（fp16/bf16） |
| Diffusion UNet | ✅ GroupNorm | DDPM/DDIM 实现的标准配置 | 16GB 可跑 DiT-B/2 |
| 大模型微调（8+ GPUs） | ⚠️ SyncBatchNorm | 跨 GPU 统计同步提高精度 | 需 NCCL 通信开销 |
| Batch size=1 的在线学习 | ❌ BN → 必须用 LN/GN | BN 的 batch 统计量无意义 | 明确排除 BN |

### 6.2 PyTorch 实现要点与陷阱

```python
# 正确做法：BN 尊重 train/eval 模式
model.train()  # 使用 mini-batch 统计量
output = model(x)

model.eval()   # 使用 running 统计量
output = model(x)

# 常见错误：torch.no_grad() 不会阻止 BN 统计量更新！
with torch.no_grad():
    model.train()  # BN 仍然会更新 running stats！
    output = model(x)
```

关键实现细节：
- `nn.BatchNorm2d(num_features, affine=True)`：默认学习 $\gamma$ 和 $\beta$
- `track_running_stats=True`：默认维护 running mean/var
- `momentum=0.1`：默认 EMA 系数，通常保持默认即可
- **微调陷阱**：如果预训练 BN 统计量来自不同领域（如 ImageNet 自然图像 → 机器人仿真图像），running stats 可能过时。解决：微调初期使用较小的 momentum（如 0.01）让 BN 快速适应新分布
- **混合精度训练**：在使用自动混合精度（AMP）时，BN 在 fp32 下计算更稳定。PyTorch 的 AMP 自动将 BN 层保留在 fp32

### 6.3 为什么理解 BN 对 VLA 研究至关重要

1. **特征提取迁移**：使用预训练 ResNet 骨干时，BN 层加载了 ImageNet 的 running 统计量。当领域漂移发生时（如机器人摄像头图像 vs. ImageNet 照片），需要适应 BN 统计量。**4096 的 Effective batch size 不是总能保证的**——在 16GB 显存下，可能需要使用梯度累积来模拟大 batch

2. **归一化设计模式**：BN 建立的"归一化 → 仿射变换"基本设计模式，在 LN、GN、RMSNorm 中完全复用。理解 BN 的参数维度（$2 \times C$ 的 $\gamma, \beta$）等于理解了所有归一化方法的参数设计

3. **损失景观直觉**：BN 教会学界**归一化能平滑优化景观**。这一洞察是现代 VLA 训练中梯度裁剪（gradient clipping）、学习率预热（learning rate warmup）和精心放置归一化层的根本原因

### 6.4 硬件兼容性具体建议

| 硬件 | BN Batch Size 建议 | 替代方案 |
|---|---|---|
| RTX 3090 24GB | ResNet-50 可用 batch=64 | 若需 batch < 16 则换 GN |
| RTX 4070 Ti Super 16GB | ResNet-50 可用 batch=32 | 16GB 下视频模型大输入时必用 GN |
| 单卡微调 | 若 batch 受限 | 优先考虑 GN/LN 而非 BN |
| 多卡 DDP | 每卡 batch >= 8 时 BN 可用 | 若需精度则用 SyncBN |

## PDF

[[Batch Normalization 原文.pdf]]
