---
tags:
  - 论文
  - 生成模型
  - 变分推断
  - VAE
  - 深度学习基础
created: 2026-07-02
paper_title: "Auto-Encoding Variational Bayes"
paper_authors: "Diederik P. Kingma, Max Welling"
paper_year: 2013
paper_venue: "ICLR 2014"
paper_citations: "~55,000+"
paper_url: "https://arxiv.org/abs/1312.6114"
github: "https://github.com/DPKingma/Auto-Encoding-Variational-Bayes"
---

# VAE

**Auto-Encoding Variational Bayes**
*Diederik P. Kingma, Max Welling / Universiteit van Amsterdam | ICLR 2014 | arXiv: 1312.6114*

> **Pitch**: 深度学习时代最具影响力的生成模型之一。VAE 的核心创新是**重参数化技巧（reparameterization trick）**——用确定性的 `z = μ + σ ⊙ ε` 替代从 `q(z|x)` 中随机采样，使随机梯度可以穿过采样操作反向传播。第二个关键突破是**自编码贝叶斯（AEVB）算法**：用神经网络（encoder）同时拟合潜变量的近似后验，端到端地用 SGD 优化变分下界（ELBO），无需 MCMC 或逐样本推断。VAE 的思想——潜变量模型 + 重参数化 + 神经 encoder/decoder——被 [[CVAE]]（条件 VAE，用于 ACT 的动作生成）、VQ-VAE（离散潜变量）、β-VAE（解耦表征学习）大量继承和发展。

---

## 一、Background / Core Idea

### 1.1 变分推断的困境

在 VAE 出现之前，训练具有连续潜变量的概率生成模型面临两大困境：

- **难处理的推断（Intractable Inference）**：给定观测 $x$，潜变量的真实后验 $p_\theta(z|x)$ 通常无法解析计算（涉及 $p_\theta(x) = \int p_\theta(z)p_\theta(x|z)dz$，这个积分在大模型上不可解），因此 EM 算法无法使用，MCMC 采样又太慢
- **难以规模化（Scalability）**：传统的变分推断（mean-field VI）需要对每个数据点单独优化变分参数，无法用 minibatch SGD 处理大规模数据集

VAE 解决了这两个问题的底层方法——不是写一个新的变分推断算法，而是发现：**如果能让采样操作对参数可微，就能用反向传播训练整个模型**。

### 1.2 核心思路：从"推断"到"学习推断"

VAE 的哲学是将统计推断问题转化为优化问题：

1. 引入**识别模型（recognition model）** $q_\phi(z|x)$——也叫 encoder/probabilistic encoder——用神经网络直接从 $x$ 预测近似的后验分布参数（$\mu, \sigma$）
2. 用**重参数化技巧**使得采样 $z \sim q_\phi(z|x)$ 对 $\phi$ 可微
3. 联合优化 encoder 参数 $\phi$ 和生成模型（decoder）参数 $\theta$，最大化变分下界（ELBO）

因此 VAE = **变分推断 + 神经网络 + 重参数化**。与传统变分推断（每个数据点独立优化变分参数）不同，VAE 用同一个神经网络（encoder/识别模型）对所有数据点进行**摊销推断（amortized inference）**——一次向前传播即可推断任何 $x$ 的潜变量分布。

### 1.3 三个核心贡献

| 贡献 | 说明 |
|------|------|
| **SGVB 估计器** | 用重参数化技巧构造 ELBO 的低方差、可微 Monte Carlo 估计器 |
| **AEVB 算法** | 用神经网络实现 encoder（识别模型），SGD 联合训练 encoder 和 decoder |
| **重参数化技巧** | $z = g_\phi(\epsilon, x)$ 替代随机采样，解决反向传播不可导问题 |

### 1.4 与自编码器的关系

VAE 得名于它的损失函数结构天然对应自编码器：

- **KL 散度项**：$-D_{KL}(q_\phi(z|x) \parallel p_\theta(z))$ 起**正则化**作用——让潜变量分布接近先验
- **重构项**：$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ 对应自编码器的**重构误差**——鼓励解码器准确重构输入

这个结构带来了一个关键优势：不需要像去噪自编码器或稀疏自编码器那样人为添加正则化超参数——正则化强度由贝叶斯框架**自动确定**。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 生成模型（Generative Model / Decoder）

VAE 假设数据由潜变量 $z$ 经过两个步骤生成：

1. $z \sim p_\theta(z)$——从先验采样潜变量
2. $x \sim p_\theta(x|z)$——从潜变量生成数据

标准实现中，先验取标准正态分布 $p_\theta(z) = \mathcal{N}(0, I)$（不含可学习参数），似然 $p_\theta(x|z)$ 用神经网络（decoder）参数化：

- **二值数据（如 MNIST）**：Bernoulli 解码器，输出 pixel-wise 概率
- **连续数据**：Gaussian 解码器，输出 $\mu$ 和 $\log\sigma^2$

### 2.2 变分下界（ELBO）

对每个数据点 $x^{(i)}$，log 似然可分解为：

$$\log p_\theta(x^{(i)}) = D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z|x^{(i)})) + \mathcal{L}(\theta, \phi; x^{(i)})$$

由于 KL 散度 $\ge 0$，$\mathcal{L}$ 是 log 似然的**下界**：

$$\mathcal{L}(\theta, \phi; x^{(i)}) = \mathbb{E}_{q_\phi(z|x^{(i)})}[-\log q_\phi(z|x) + \log p_\theta(x, z)]$$

等价地分解为：

$$\mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) + \mathbb{E}_{q_\phi(z|x^{(i)})}[\log p_\theta(x^{(i)}|z)]$$

| 项 | 含义 | 直觉 |
|---|---|---|
| $-D_{KL}(q_\phi \parallel p_\theta)$ | 正则化：拉近近似后验与先验 | "把潜变量压回标准正态" |
| $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$ | 重构：最大化 $x$ 在潜变量下的似然 | "让 decoder 能还原输入" |

### 2.3 重参数化技巧（The Reparameterization Trick）

这是 VAE 最核心的技术贡献。问题在于：若要优化 $\mathcal{L}$ 对 $\phi$ 的梯度，需计算 $\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$，而 $q_\phi$ 本身包含 $\phi$——不能简单交换积分和微分。

**解决方案**：将随机变量 $z \sim q_\phi(z|x)$ 表示为确定性变换：

$$z = g_\phi(\epsilon, x), \quad \epsilon \sim p(\epsilon)$$

其中 $p(\epsilon)$ 是**独立于 $\phi$** 的噪声分布。

对于高斯后验 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$：

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这样梯度可以穿过采样操作直接作用于 $\mu_\phi$ 和 $\sigma_\phi$：

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \epsilon)]$$

**三种可重参数化的分布族**：
1. **逆 CDF 可解**：如指数分布、Cauchy、Logistic
2. **位置-尺度族**：Gaussian、Laplace、Student's t
3. **组合变换**：Log-Normal（正态取指数）、Gamma（指数变量之和）

### 2.4 SGVB 估计器

基于重参数化，构造两类 SGVB 估计器：

**通用形式 $\mathcal{L}^A$**（不需要解析 KL）：

$$\tilde{\mathcal{L}}^A(\theta, \phi; x^{(i)}) = \frac{1}{L}\sum_{l=1}^{L}\left[\log p_\theta(x^{(i)}, z^{(i,l)}) - \log q_\phi(z^{(i,l)}|x^{(i)})\right]$$

其中 $z^{(i,l)} = g_\phi(\epsilon^{(i,l)}, x^{(i)})$，$\epsilon^{(l)} \sim p(\epsilon)$。

**解析 KL 形式 $\mathcal{L}^B$**（方差更低，更常用）：

$$\tilde{\mathcal{L}}^B(\theta, \phi; x^{(i)}) = -D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) + \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x^{(i)}|z^{(i,l)})$$

对于 Gaussian 先验 + Gaussian 后验，KL 项有解析解：

$$-D_{KL}(q_\phi(z|x^{(i)}) \parallel p_\theta(z)) = \frac{1}{2}\sum_{j=1}^{J}\left[1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right]$$

论文发现：**$L=1$ 时效果已足够好**——只要 minibatch 大小 $M=100$ 即可，因为梯度噪声在 minibatch 维度上被平均了。

### 2.5 AEVB 算法

```
Algorithm: AEVB (Auto-Encoding Variational Bayes)
--------------------------------------------------
θ, φ ← Initialize parameters
repeat
    X^M ← Random minibatch of M datapoints
    ε  ← Random samples from N(0, I)
    g  ← ∇_{θ,φ} L̃(θ, φ; X^M, ε)    (SGVB estimator)
    θ, φ ← Update using Adagrad / Adam with gradients g
until convergence of (θ, φ)
```

### 2.6 架构设计

**Encoder（识别模型）**：
- 输入：$x$
- 输出：$\mu_\phi(x)$ 和 $\log\sigma_\phi^2(x)$（对角协方差）
- 结构：MLP with hidden layers，tanh 激活

**Decoder（生成模型）**：
- 输入：$z \sim q_\phi(z|x)$
- 输出：Bernoulli（二值数据）或 Gaussian（连续数据）
- 对于 Bernoulli 解码器：$\log p(x|z) = \sum_i [x_i \log y_i + (1-x_i)\log(1-y_i)]$，其中 $y = \sigma(\text{MLP}(z))$

**注意**：VAE 使用对角协方差的 Gaussian 后验是一个**简化假设**，不是方法局限。原则上 $q_\phi(z|x)$ 可以取任意分布形式，只要可重参数化即可。论文也给出了 full VB 版本的推导（同时对 $\theta$ 和 $z$ 做变分推断），但实验和主文本聚焦于 MAP 估计 + 对角高斯近似。

---

## 三、Experiments and Key Findings

### 3.1 训练效率：AEVB vs Wake-Sleep

论文在 MNIST 和 Frey Face 数据集上对比了 AEVB 和 Wake-Sleep 算法：

| 对比维度 | AEVB | Wake-Sleep |
|---------|------|-----------|
| 收敛速度 | **显著更快** | 较慢 |
| ELBO 最终值 | **更高** | 较低 |
| 优化目标 | 单个（ELBO 单调增大） | 两个（wake/sleep 阶段目标不一致） |
| 过拟合 | 无（ELBO 天然正则化） | — |

Wake-Sleep 的根本缺陷：它交替优化两个**不互相对应的**目标函数（wake phase 优化生成模型，sleep phase 优化识别模型），两者之和不构成任何似然的下界——无法保证收敛到好的解。

### 3.2 潜变量维度的影响

论文测试了 $N_z = 3, 5, 10, 20, 200$ 的情况，得出两个关键发现：
- **多余潜变量不导致过拟合**——ELBO 中的 KL 正则化自然地迫使无用维度趋近先验 $\mathcal{N}(0, I)$，即"自动剪枝"
- 更高 latent dim 通常带来更好的 ELBO，符合直觉——更多容量虽然带来了更多要学习的参数，但也让模型能编码更多信息

### 3.3 与 MCEM 的对比

在 $N_{train}=1000$ 和 $N_{train}=50000$ 两种数据量下，VAE 都显著优于基于 HMC 的 Monte Carlo EM（MCEM 是 offline 算法，无法用于大规模数据）。

### 3.4 潜在空间的语义结构

论文展示了 2D 潜在空间的流形可视化：在学到的 latent space 中，相近的 $z$ 生成相似的图像，潜在空间呈现平滑、连续的语义结构——这是 VAE 作为**表征学习**工具的重要性质，也是后来 β-VAE 探索解耦表征的起点。

---

## 四、Limitations and Challenges

1. **生成图像模糊**：VAE 的 Gaussian 解码器假设 $p(x|z)$ 是各向同性的（对角协方差），这导致生成的图像细节模糊——因为模型被鼓励输出"安全的"平均值而非有细节纹理的具体样本。这也是 GAN 能在图像质量上反超 VAE 的原因。

2. **后验坍缩（Posterior Collapse）**：当 decoder 能力过强时，可能直接忽略 $z$（$q_\phi(z|x) \rightarrow p(z) = \mathcal{N}(0, I)$），导致 KL 项趋近 0、潜变量失去信息量。这在后来的 VAE 训练中成为重大挑战，催生了 KL annealing、free bits 等技巧。

3. **高斯先验的限制**：标准正态先验 $p(z) = \mathcal{N}(0, I)$ 表达能力有限。Normalizing Flows（Rezende & Mohamed 2015）、VQ-VAE（离散潜变量）等方法部分解决了这个问题。

4. **连续潜变量的限制**：原文只能处理连续潜变量（需要重参数化）。离散潜变量（如 VQ-VAE、DALL-E 的 codebook）是后来的重要方向。Wake-Sleep 反而在离散潜变量上有优势。

5. **ELBO 只是下界**：优化 ELBO 不保证接近真实 log 似然——ELBO 和真实似然之间的 gap $= D_{KL}(q_\phi \parallel p_\theta)$ 可能很大。这催生了 IWAE（Importance Weighted AE）等紧致下界的改进。

6. **似然非直接优化目标**：VAE 的损失函数是 ELBO，而不是直接的样本质量指标（如 FID）。这意味着更低的 ELBO 不一定对应更逼真的生成样本——这为后来 GAN 在图像质量上的统治留下了空间。

---

## 五、Relationship with Subsequent Work / Impact on the Field

### 5.1 直接衍生工作

| 工作 | 关联 | 时间 |
|------|------|------|
| **CVAE（Conditional VAE）** | 引入条件信息 $c$：$p_\theta(x|z,c)$，$q_\phi(z|x,c)$ | 2015 |
| **β-VAE** | 在 KL 项加权重 $\beta > 1$，促进解耦表征学习 | 2017 |
| **IWAE** | 重要性加权估计器，收紧 ELBO，提供更紧致的似然下界 | 2016 |
| **VQ-VAE** | 将潜变量替换为离散 codebook，解决模糊问题 | 2017 |
| **Normalizing Flows** | 用可逆变换序列增强后验的表达能力 | 2015 |
| **NVAE** | 层次化 VAE + 深度卷积 + 残差结构，大幅提升生成质量 | 2021 |

### 5.2 在机器人/VLA 中的角色

VAE 在机器人学中的最主要应用是通过 **CVAE（条件 VAE）**实现多模态动作生成：

- **ACT（Action Chunking Transformer）**：使用 CVAE 作为动作生成头。encoder 输入观测+未来动作，学习潜变量 $z$；推理时从先验采样 $z$，decoder 生成 50 步动作 chunk。CVAE 的潜变量捕获了动作的多模态分布——同一观测下可能的多个动作策略。

- **为什么用 CVAE 而不是 Diffusion Policy？** CVAE 的优势在于单步推理（encoder→sample $z$→decoder），速度快。Diffusion Policy 需要多步去噪但能更好地覆盖多模态分布。它们在不同场景各有优势。

### 5.3 在深度学习历史上的位置

```
                                                                         ┌──→ Conditioned VAE → ACT
                                                                         │
VAE (2013) ─────→ 生成模型成为深度学习核心方向 ──→ GAN (2014) ──→ DDPM (2020) ──→ Flow Matching (2022)
  │                    │
  ├──→ 变分推断+神经网络  │
  ├──→ 重参数化技巧      │
  ├──→ ELBO 优化范式     │
  └──→ 摊销推断          │
                         └──→ Normalizing Flows → VQ-VAE → NVAE → Stable Diffusion 3（VAE作为潜空间压缩）
```

**重参数化技巧**的影响远超生成模型——它成为任何需要"通过采样操作反向传播"的场景的标准工具，包括强化学习（stochastic policy gradient 的改造）、贝叶斯神经网络、以及 [[Flow Matching]] 的采样过程。

---

## 六、Hardware Compatibility and Learning Suggestions

| 维度 | 评价 |
|------|------|
| 训练硬件 | ✅ 极低——MNIST 级 VAE 在 CPU 上即可训练（论文用 CPU 跑，约 20-40 分钟/百万样本）。ImageNet 级约需 1-2 GPU |
| 推理硬件 | ✅ 单次前向传播，极快——比扩散模型快 100-1000 倍 |
| 代码复杂度 | ✅ 简单——核心实现约 80 行 PyTorch，是理解生成模型的入门首选 |
| 对 VLA 的意义 | ✅ ACT 使用 CVAE 做动作生成——理解 VAE 是理解 ACT 的前提 |
| 与扩散模型对比 | VAE 快速但质量较粗糙；扩散模型慢但高质量——两者互补，各有用武之地 |

**核心启示：**

1. **重参数化技巧是所有"可微采样"的基础**——这个思想贯穿了扩散模型的 denoising 过程、flow matching 的路径构造、以及 RL 中的 stochastic policy
2. **ELBO 是理解所有生成模型损失函数的基础语言**——DDPM 的变分下界、Flow Matching 的条件流损失、都能追溯到 VAE 的 ELBO 框架
3. **VAE 本身在图像生成中已被扩散模型超越**——不必深入学 VAE 的工程细节，重点理解"重参数化"+"变分下界"+"摊销推断"三个核心概念
4. **如果要深入 VLA，重点看 CVAE 如何用于动作生成**——ACT 是最好的例子

---

## PDF

[[VAE 原文.pdf]]
