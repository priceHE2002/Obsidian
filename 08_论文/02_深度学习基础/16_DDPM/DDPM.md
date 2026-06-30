---
tags:
  - 论文
  - 扩散模型
  - 生成模型
created: 2026-06-30
paper_title: "Denoising Diffusion Probabilistic Models"
paper_authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel"
paper_year: 2020
paper_venue: "NeurIPS 2020"
paper_citations: "~25,000+"
paper_url: "https://arxiv.org/abs/2006.11239"
---

# DDPM

**Denoising Diffusion Probabilistic Models**
*UC Berkeley | NeurIPS 2020 | arXiv: 2006.11239*

> 扩散模型的基础论文，让扩散模型从理论可行变为工程可用。定义了前向加噪过程（逐步破坏数据）和反向去噪过程（从噪声恢复数据），并通过学习 score function（数据分布的梯度）来生成高质量样本。VLA 动作生成的理论基石——Diffusion Policy 直接基于此，Flow Matching 是其变体。

---

## 一、研究背景与动机

生成模型长期以来被 GAN（生成对抗网络）主导，但 GAN 存在训练不稳定、模式坍塌等问题。VAE 和 flowed 模型虽然训练稳定，但生成质量有限。早期的扩散模型（Sohl-Dickstein et al., 2015）从非平衡热力学现象出发提出了理论框架，但生成质量远不如 GAN。

DDPM 的贡献在于：证明了扩散模型在**工程上可行**——通过精心设计的噪声调度、参数化和训练目标，扩散模型可以生成**与 GAN 相媲美甚至更优**的高质量图像，同时克服 GAN 的训练不稳定问题。

## 二、核心方法

DDPM 定义了两个相互耦合的马尔可夫链：

**前向过程（逐步加噪）：** 每一步向数据添加小量高斯噪声，T 步后数据接近纯噪声。

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot I)$$

通过重参数化技巧，可以在任意时刻 t 一步采样到加噪结果：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

**反向过程（去噪生成）：** 学习一个神经网络逐步去除噪声，从纯噪声恢复出数据。

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \cdot I)$$

**训练目标：** 简化后的损失函数——让网络预测被添加的噪声 $\varepsilon$：

$$L = \mathbb{E}\left[ ||\varepsilon - \varepsilon_\theta(x_t, t)||^2 \right]$$

| 组件 | 实现方式 |
|------|---------|
| 骨干网络 | U-Net（带自注意力和时间步嵌入） |
| 噪声调度 | 线性调度（$\beta_1=10^{-4}$, $\beta_T=0.02$）|
| 改进调度 | 余弦噪声调度（更适合低分辨率） |
| 加速采样 | DDIM（50-100 步替换 1000 步） |

**关键洞察：** 噪声预测网络 $\varepsilon_\theta$ 实际上在学习 score function——数据分布的对数密度的梯度：

$$\varepsilon_\theta(x_t, t) \approx -\sigma_t \cdot \nabla_x \log p(x_t)$$

这意味着扩散模型在**隐式地学习数据分布的几何结构**。

## 三、关键实验与发现

- **CIFAR-10 FID 3.17**：当时无条件图像生成的 SOTA
- **LSUN 和 CelebA-HQ**：验证了在高分辨率人脸和场景图上的可扩展性
- **消融实验**：验证了简化损失 $L_{simple}$ 优于原始变分下界损失
- **生成多样性**：相比于 GAN，扩散模型的生成多样性显著更高（Inception Score 更一致）

## 四、局限性与后续影响

**局限：**
- 推理速度慢：需要 T=1000 步逐步去噪，即使 DDIM 加速也需要 50-100 步
- U-Net 骨干对超分辨率和条件生成的支持有限
- 采样计算量远大于 GAN（单次前馈）

**后续影响：**
- DDIM (2020) —— 将反向过程改为确定性的隐式模型，将步数压缩到 50-100
- Improved DDPM (2021) —— 余弦调度 + 学习方差，提升对数似然
- Diffusion Models Beat GANs (2021) —— 基于分类器引导（classifier guidance）实现条件生成
- Stable Diffusion (2022) —— 在隐空间做扩散，大幅提升效率和分辨率
- Flow Matching (2022) —— 将随机扩散改为确定性直线流，DDPM 的理论继承者

## 五、VLA/机器人研究中的角色

DDPM 是 VLA 动作生成的理论奠基者：

- **Diffusion Policy** 直接基于 DDPM：将机器人动作建模为条件去噪扩散过程 $p(a | O_t)$，在观测条件下对动作序列去噪
- **Octo** 使用 DDPM 作为动作解码头
- **Flow Matching ($\pi_0$, GR00T N1)** 是 DDPM 的确定化改进——从随机布朗桥变为最优传输直线
- DDPM 的理论分析（score function = 数据分布梯度）解释了为什么扩散策略训练稳定而隐式策略（IBC）不稳定

## 六、对你的启示

- **彻底理解 DDPM 范式**：前向破坏 + 反向生成、噪声预测目标、score function 解释——这是生成式机器人策略的理论起点
- **Diffusion Policy 的起点**：DDPM 是 Diffusion Policy 的直系理论基础
- **DDIM 加速很重要**：1000 步不可能用于机器人，DDIM 将推理降到 10-50 步才使扩散策略可用
- 不必在纯 DDPM 上花过多时间——**Flow Matching** 是实际工程选型的主流
- 推理可在 16GB GPU 上运行（DDIM 10-50 步去噪）

## PDF

[[DDPM.pdf]]
