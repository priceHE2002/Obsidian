---
tags:
  - 论文
  - 扩散模型
  - 生成模型
  - 概率模型
created: 2026-06-30
paper_title: "Denoising Diffusion Probabilistic Models"
paper_authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel"
paper_year: 2020
paper_venue: "NeurIPS 2020"
paper_citations: "~25,000+"
paper_url: "https://arxiv.org/abs/2006.11239"
github: "https://github.com/hojonathanho/diffusion"
---

# DDPM

**Denoising Diffusion Probabilistic Models**
*Jonathan Ho, Ajay Jain, Pieter Abbeel / UC Berkeley | NeurIPS 2020 | arXiv: 2006.11239*

> **Pitch**: 扩散模型的基础论文，让扩散模型从理论可行变为工程可用。定义了前向加噪过程（马尔可夫链逐步破坏数据直至纯噪声）和反向去噪过程（学习神经网络从噪声恢复数据）。**关键贡献**：发现噪声预测网络 $\varepsilon_\theta$ 等价于学习 score function（数据分布对数密度的梯度），简化损失函数使训练稳定，生成质量与 GAN 相媲美。VLA 动作生成的理论基石——[[Diffusion Policy]] 直接基于此，[[Flow Matching]] 是其确定化演进。

---

## 一、Background / Core Idea

### 1.1 生成模型的困境（2019年前）

生成模型长期被 GAN（生成对抗网络）主导。GAN 通过生成器-判别器的对抗训练生成高质量样本，但存在根本问题：

- **训练不稳定**：判别器-生成器的极小极大博弈难以平衡，容易出现模式坍塌（mode collapse）
- **模式覆盖不足**：GAN 倾向于只学习数据分布的少数模式
- **缺乏理论优雅性**：没有显式的似然评估

VAE 和正常化流（flows）虽然训练稳定，但生成质量远不如 GAN。早期扩散模型（Sohl-Dickstein et al., 2015）从非平衡热力学出发提出了理论框架，但生成质量远不如 GAN，论文专注于理论分析而非工程实践。

### 1.2 扩散模型的直觉

DDPM 的灵感来自非平衡热力学（nonequilibrium thermodynamics）：**如果逐步向数据添加噪声直至完全破坏信号，那么学习逆向过程（去噪）即可生成数据。** 关键在于：当噪声增量足够小时，逆向过程可以参数化为高斯分布——这使得用神经网络学习变得可行。

### 1.3 DDPM 的核心贡献

DDPM 的贡献不仅是理论框架的延续，更关键的是证明了扩散模型在**工程上可行且优秀**：
1. 通过精心设计的噪声调度（方差表）和参数化方案
2. 简化损失函数使训练稳定且高效
3. 生成质量与 GAN 相媲美，同时克服 GAN 的训练不稳定
4. 生成多样性显著高于 GAN

---

## 二、Method / Architecture / Technical Contribution

### 2.1 前向过程（Forward Process / Diffusion Process）

定义马尔可夫链 $q$，逐步向数据 $x_0 \sim q(x_0)$ 添加高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

其中 $\beta_1, ..., \beta_T$ 是方差调度（variance schedule）。$T=1000$（DDPM 的默认值）。

通过重参数化技巧（reparameterization trick），可以在任意时刻 $t$ 直接采样 $x_t$ 而不必循环 $t$ 步：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。

当 $T$ 足够大且 $\beta_t$ 设计适当时，$x_T$ 近似标准正态分布 $\mathcal{N}(0, I)$。

### 2.2 反向过程（Reverse Process）

学习一个神经网络来逆转前向过程——从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 逐步去噪得到数据 $x_0$：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} | x_t)$$

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

通常固定 $\Sigma_\theta(x_t, t) = \sigma_t^2 \cdot I$（其中 $\sigma_t^2 = \beta_t$ 或 $\tilde{\beta}_t$），只有 $\mu_\theta$ 由神经网络参数化。

### 2.3 训练目标：重参数化和简化损失

标准的变分下界损失为：

$$L = \mathbb{E}_q\left[ -\log p_\theta(x_0) \right] \leq \mathbb{E}_q\left[ D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1} D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t)) - \log p_\theta(x_0|x_1) \right]$$

**关键推导**：$q(x_{t-1}|x_t, x_0)$ 有闭合形式（因为前向过程是高斯马尔可夫链）：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \cdot I)$$

其中 $\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$。

DDPM 的核心贡献是**重参数化 $\mu_\theta$ 为噪声预测网络** $\varepsilon_\theta$：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_\theta(x_t, t)\right)$$

代入 KL 散度后，损失简化为：

$$L_{simple}(\theta) = \mathbb{E}_{t, x_0, \varepsilon}\left[ ||\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \varepsilon, t)||^2 \right]$$

这就是 DDPM 实际的训练损失——**极其简单**：让网络预测添加的噪声 $\varepsilon$。

### 2.4 Score Function 的联系

DDPM 的一个重要理论贡献是揭示了扩散模型与 score matching 之间的联系：

$$\varepsilon_\theta(x_t, t) \approx -\sigma_t \cdot \nabla_x \log p(x_t)$$

其中 $\nabla_x \log p(x_t)$ 是数据分布的 **score function**（对数密度的梯度）。这意味着：

- 扩散模型**隐式地学习数据分布的几何结构**
- 训练过程等价于在不同噪声水平上做 denoising score matching
- 采样过程等价于 annealed Langevin dynamics

这一联系解释了为什么扩散模型的训练非常稳定——**score matching 目标函数是凸的**（在函数空间中），没有 GAN 的对抗不稳定。

### 2.5 噪声调度（Noise Schedule）

**线性调度（DDPM 默认）：**

$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$$

其中 $\beta_1 = 10^{-4}$，$\beta_T = 0.02$。

线性调度在高分辨率图像上表现好，但在低分辨率（如 CIFAR-10）上对数似然不够优。

**余弦调度（Improved DDPM 改进）：**

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

余弦调度在 $t$ 中等时的噪声添加更慢，避免了高噪声水平的信息"浪费"。

### 2.6 骨干网络：U-Net

DDPM 使用 **U-Net** 作为去噪网络：

- **下采样-上采样对称结构**：四次下采样（分辨率依次减半），对应四次上采样
- **残差块（ResNet blocks）**：每个分辨率层有多个卷积残差块
- **自注意力层**：在 $16 \times 16$ 分辨率上插入自注意力（spatial self-attention）
- **时间步嵌入**：类似 Transformer 的 sinusoidal positional encoding，将 $t$ 映射到 embedding（128/256 维），然后通过加法或 FiLM-style 调制注入到每个残差块
- **Group Normalization**：U-Net 使用 group norm 而非 batch norm

### 2.7 采样过程（Sampling Algorithm）

```
Algorithm 2: Sampling
x_T ~ N(0, I)
for t = T, ..., 1 do
    z ~ N(0, I) if t > 1 else z = 0
    x_{t-1} = μ_θ(x_t, t) + σ_t · z
end for
return x_0
```

每一步采样：$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right) + \sigma_t \cdot z$

需要 $T=1000$ 步才能生成一个样本——这是 DDPM 的主要瓶颈。

### 2.8 DDIM（Denoising Diffusion Implicit Models）

后续工作 DDIM（不是 DDPM 的一部分，但密切相关）将反向过程改为确定性的（去掉 $z$ 项），使采样步数从 1000 压缩到 50-100 步：

- 将扩散模型重新解释为**隐式模型**（implicit probabilistic model）
- 反向过程是确定性的：$x_{t-1} = f_\theta(x_t, t)$
- 可跳步采样（skipping steps）：直接对非相邻时间步做一步预测
- 步数从 1000 降至 50 步时 FID 仅小幅下降（CIFAR-10: 3.17 → 4.00）
- **使扩散模型的实际部署成为可能**

---

## 三、Experiments and Key Findings

### 3.1 无条件图像生成结果

| 数据集 | 指标 | DDPM 结果 | 当时 SOTA |
|--------|------|-----------|-----------|
| CIFAR-10 | FID | **3.17** | 当时无条件生成 SOTA |
| CIFAR-10 | Inception Score | **9.46** | 与 GAN 相当 |
| CelebA-HQ 256×256 | FID | 显著优于 ProgressiveGAN |
| LSUN Bedroom 256×256 | FID | 与 ProgressiveGAN 相当 |

这些结果首次证明扩散模型可以产生与 GAN 相媲美（甚至更好）的高质量样本。

### 3.2 简化损失的有效性

**核心消融发现：** $L_{simple}$ 比原始变分下界损失 $L$ 效果好得多。因为：
- $L_{simple}$ 对所有 $t$ 施加相同的权重
- 原始 $L$ 对小 $t$（低噪声水平）的权重过大，模型过度关注低噪声水平（简单去噪），忽视了高噪声水平的语义理解
- $L_{simple}$ 的均匀重加权让模型在所有噪声水平上学习良好

### 3.3 生成多样性

GAN 的 mode collapse 一直是核心问题。DDPM 展示了显著更高的生成多样性：
- Inception Score 的分布更一致
- 覆盖了数据分布的更多模式
- 没有观察到 GAN 常见的模式坍塌迹象

### 3.4 训练稳定性

DDPM 训练极其稳定：
- 没有 GAN 的对抗训练
- 损失稳定下降，没有振荡
- 不需要特殊技巧（如梯度惩罚、谱归一化）来维持训练
- 单调性：loss 越低生成质量越高（与 GAN 不同，GAN 的 loss 与生成质量没有单调关系）

### 3.5 采样时间与质量的权衡

DDPM 需要 1000 步采样，每步一次前向传播——这意味着生成单张图像需要 1000×U-Net 前向传播。推理速度是 DDPM 的主要瓶颈。

---

## 四、Limitations and Challenges

1. **推理速度极慢**：T=1000 步逐步去噪。即使 DDIM 加速到 50 步，相比 GAN（单次前馈）仍慢 50×。这是扩散模型在机器人实时控制中的核心挑战。

2. **U-Net 架构的局限性**：U-Net 作为卷积骨干，**无法很好地利用 scaling law**（增加参数不一定提升性能）。[[DiT]] 的出现正是因为发现了这一点。

3. **路径效率低**：DDPM 的采样路径是随机布朗桥——弯曲、低效。[[Flow Matching]] 提出确定性直线路径来解决这一问题。

4. **对数似然不如 VAE**：DDPM 虽然样本质量好，但 log-likelihood 低于基于似然的模型（如 VAE）。这在需要密度估计的场合是劣势。

5. **条件生成需要额外机制**：单纯 DDPM 无法做条件生成，需要 classifier guidance 或 classifier-free guidance 等额外机制。

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关联 | 时间 |
|---------|------|------|
| **DDIM** | 确定性采样，步数 1000→50-100 | 2020 |
| **Improved DDPM** | 余弦调度 + 学习方差，优化对数似然 | 2021 |
| **Diffusion Models Beat GANs** | classifer guidance + U-Net 架构改进 | 2021 |
| **Stable Diffusion** | 隐空间扩散，提升效率和分辨率 | 2022 |
| **Diffusion Policy** | **将 DDPM 用于机器人动作生成** | 2022-2023 |
| **[[Flow Matching]]** | 确定性直线流，DDPM 的确定化演进 | 2022 |
| **[[DiT]]** | ViT 替代 U-Net，提升可扩展性 | 2022 |

**在 VLA/机器人中的角色（最重要）：**

- **[[Diffusion Policy]] 直接基于 DDPM**：将机器人动作建模为条件去噪扩散过程 $p(a | O_t)$，在观测条件下对动作序列去噪。这是 DDPM 在机器人领域最直接的应用。
- **训练稳定性是 Diffusion Policy 成功的核心原因**：IBC（Implicit Behavior Cloning）需要 MCMC 推理，训练不稳定且收敛困难。Diffusion Policy 使用 DDPM 的噪声预测目标训练极稳定，且在多模态动作分布上天然表现好。
- **[[Octo]]** 使用 DDPM 作为动作解码头。
- **[[Flow Matching]]（[[π0]]、[[GR00T N1]]）** 是 DDPM 的确定化改进——从随机布朗桥变为最优传输直线，推理步数从 50-100 降至 5-20。

### 为什么 Diffusion Policy 训练稳定（深层原因）

Implicit policy（如 IBC）需要**配分函数（partition function）**——预测能量函数然后做 MCMC 推理，这在连续动作空间中极不稳定。Diffusion Policy 不需要显式建模概率分布的归一化常数——它通过预测噪声隐式学习数据分布，等价于 score matching 视角下的梯度场，**天然避开了配分函数的计算问题**。

---

## 六、Implications for You / Hardware Compatibility

| 维度 | 评价 |
|------|------|
| 推理硬件 | ✅ 16GB GPU 可运行 DDIM 50 步采样。DDPM 1000 步约需 20-30 秒/U-Net|
| 训练硬件 | ✅ CIFAR-10 单卡 24h。ImageNet 约需 8×A100。大部分实验室可负担 |
| 代码复杂度 | ✅ 简单——DDPM 的实现约 150 行 PyTorch。开源实现丰富 |
| 对 VLA 的意义 | ✅ **必须理解**——[[Diffusion Policy]] 的理论基础 |
| 实时性 | ⚠️ 1000 步采样远达不到实时。DDIM 50 步约 1-2 秒 | 1-2 |

**核心启示：**
1. **彻底理解 DDPM 范式**：前向破坏 + 反向生成、噪声预测目标、score function 解释——这是全部生成式机器人策略的理论起点
2. **DDPM 本身已不主流**：不必在纯 DDPM 上花过多时间。理解原理后直接学 [[Flow Matching]]
3. **Diffusion Policy 是 DDPM 的最重要应用**：它启发了整个"扩散生成式行为克隆"方向
4. **score function 的优化稳定性是机器人扩散策略方法的核心优势**

---

## PDF

[[DDPM 原文.pdf]]
