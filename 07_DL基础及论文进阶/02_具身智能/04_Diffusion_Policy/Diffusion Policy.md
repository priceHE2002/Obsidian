---
tags:
  - 论文
  - 动作生成
  - Diffusion
  - 模仿学习
created: 2026-06-30
paper_title: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
paper_authors: "Cheng Chi, Zhenjia Xu et al. (Columbia + TRI + MIT)"
paper_year: 2023
paper_venue: "IJRR 2024 (extended version of RSS 2023)"
paper_citations: "~800+"
paper_url: "https://arxiv.org/abs/2303.04137"
---

# Diffusion Policy

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
*Columbia + TRI + MIT | IJRR 2024 | arXiv 2303.04137*

> 这篇论文本身不是 VLA 论文，但它定义的"如何用扩散模型生成动作"的方法，成为了几乎所有 VLA（π0、GR00T、Octo）动作头的理论基础。**理解 VLA 必须理解这篇。**

---

## 一、为什么需要新的策略表示？

### 1.1 模仿学习中的动作生成困境

模仿学习最简单的形式就是"从观测到动作的监督回归"。但在实践中，这种简单的回归会遇到三个根本性挑战：

**挑战 1：多模态动作分布**

人类在做同一个任务时，往往有多种"正确"的动作方式。比如推一个 T 型块进入目标位置：
- 可以从左边绕过去
- 可以从右边绕过去
- 可以直推

但如果你训练一个确定性回归模型（MSE loss），它只会学到**所有可能动作的均值**——这个"均值动作"可能根本不在任何一条有效轨迹上。

**挑战 2：高维输出空间**

单步动作的维度（6-7 维）其实不高。但如果我们要预测**整个动作序列**（而不是单步），输出维度会迅速膨胀。例如：14 维动作 × 16 步 = 224 维。传统的回归模型在高维输出空间上表现很差。

**挑战 3：训练稳定性**

隐式策略（Implicit Policy / Energy-Based Model）理论上可以解决以上两个问题，但在实践中极难训练。原因是 EBM 需要**负采样**来估计一个无法解析计算的归一化常数，而负采样的不准确性导致训练不稳定。

### 1.2 现有方法的局限

| 方法 | 多模态建模 | 高维输出 | 训练稳定性 |
|------|----------|---------|----------|
| 显式策略 (GMM) | ⚠️ 需预设模态数 | ❌ | ✅ |
| 分类策略 (BeT) | ⚠️ k-means 离散化 | ❌ | ✅ |
| 隐式策略 (IBC) | ✅ | ❌ | ❌ 很不稳定 |
| **扩散策略** | ✅ | ✅ | ✅ |

---

## 二、方法：把扩散模型放在动作空间上

### 2.1 核心公式

Diffusion Policy 将机器人策略建模为一个**条件去噪扩散过程**：

**前向过程（加噪）**：
$$x^k = \sqrt{1-\beta_k} x^{k-1} + \sqrt{\beta_k} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

第 k 步的动作变体 = 原始动作 + 逐步增加的噪声。

**反向过程（去噪/推理）**：
$$x^{k-1} = \alpha(x^k - \gamma \varepsilon_\theta(O_t, x^k, k) + \mathcal{N}(0, \sigma^2 I))$$

其中 $\varepsilon_\theta$ 是噪声预测网络，$O_t$ 是观测条件，$k$ 是去噪步数。

**核心直觉**：噪声预测网络 $\varepsilon_\theta$ 实际上在学习**动作分布的梯度场**（score function）：

$$\varepsilon_\theta(O_t, a, k) \approx -\nabla_a \log p(a | O_t)$$

推理过程就是沿着这个梯度场做随机梯度下降，直到找到最可能的动作。

### 2.2 训练

训练极其简单——标准的 denoising score matching：

$$\mathcal{L} = \text{MSE}(\epsilon_k, \varepsilon_\theta(O_t, A_t^0 + \epsilon_k, k))$$

1. 从数据集中随机抽取一个干净的样本 $A_t^0$
2. 随机选一个去噪步数 $k$，加上对应强度的噪声 $\epsilon_k$
3. 让网络预测这个噪声
4. 用 MSE 算 loss

**没有对抗训练、没有负采样、没有复杂的优化技巧。**

### 2.3 为什么扩散策略训练稳定？（vs IBC）

这是论文中非常精彩的理论分析。

隐式策略（IBC）用 EBM 表示动作分布：

$$p_\theta(a|o) = \frac{e^{-E_\theta(o,a)}}{Z(o,\theta)}$$

麻烦在 $Z(o,\theta)$ ——这是一个对 $a$ 的积分，在连续高维空间中无法解析计算。IBC 使用 InfoNCE 风格的损失来近似：

$$\mathcal{L}_{\text{InfoNCE}} = -\log\left(\frac{e^{-E_\theta(o,a)}}{e^{-E_\theta(o,a)} + \sum_{j=1}^{N_{\text{neg}}} e^{-E_\theta(o,\tilde{a}_j)}}\right)$$

**问题在于**：负样本 $\tilde{a}_j$ 的数量和质量直接决定了 $Z(o,\theta)$ 估计的准确度。不够好的负采样 → 坏的归一化常数估计 → 不稳定的训练。

**扩散策略绕过了这个问题的核心**：它直接学习 score function（梯度），而不是能量函数本身：

$$\nabla_a \log p(a|o) = -\nabla_a E_\theta(a, o) - \underbrace{\nabla_a \log Z(o, \theta)}_{=0} \approx -\varepsilon_\theta(a, o)$$

$Z(o,\theta)$ 的梯度恒为零（它不依赖于 a），因此 score function 的估计**完全独立于归一化常数**。这就是扩散策略训练极度稳定的数学根源。

---

## 三、三个关键设计决策

### 3.1 网络架构：CNN vs Transformer

**CNN-based Diffusion Policy**

采用 1D 时间卷积网络（基于 Janner et al. 的 Diffuser 架构修改）。用 FiLM（Feature-wise Linear Modulation）条件化：
- 观测特征 $O_t$ 通过 FiLM 调制每个卷积层的通道
- 去噪步数 $k$ 也通过 FiLM 注入

优点：开箱即用、超参鲁棒、推荐作为首试方案。
缺点：在高频动作变化时性能不佳（CNN 的时间平滑性 inductive bias 在需要快速切换方向时反而成问题）。

**Transformer-based Diffusion Policy**

观测 embedding 通过多头交叉注意力注入到每个 Transformer decoder block。动作 tokens 使用因果注意力（每个动作 token 只能看到之前的时间步）。

优点：擅长高频动作变化和复杂任务。
缺点：对超参数更敏感（Transformer 训练的固有问题）。

**建议**：新任务先用 CNN 版本，如果时间精细度不够再切换到 Transformer。

### 3.2 视觉编码器

设计要点：
1. **空间 softmax 池化**替代全局平均池化 → 保留空间信息（对机器人控制至关重要）
2. **GroupNorm 替代 BatchNorm** → 与 EMA（指数移动平均，DDPM 常用的训练技巧）兼容
3. **端到端训练**视觉编码器（从零开始或微调预训练权重）
4. 不同相机视角使用**独立的编码器**，每个时间步的图像独立编码后拼接

消融实验的结论：微调预训练视觉编码器（特别是 CLIP 训练的 ViT-B/16）+ 小的学习率（比策略网络小 10 倍）效果最好。但不同架构之间的差距并不大。

### 3.3 动作时序设计（最重要！）

**Closed-loop Action Sequence Prediction:**

```
预测 Tp 步 → 执行 Ta 步（Ta < Tp）→ 重新预测
```

三个关键参数：
- $T_o$：观测历史长度（输入多少帧图像）
- $T_p$：动作预测长度（预测多少步）
- $T_a$：动作执行长度（在执行多少步之后重规划）

**$T_a$ 的权衡**（这是整篇论文最实用的发现之一）：

| $T_a$ | 效果 |
|-------|------|
| 太小（如 1）| 反应快但时序一致性弱，动作 jitter 多 |
| 太大（如 $T_p$）| 时序一致但不响应环境变化，像开环控制 |
| **适中（如 8）** | **反应性和时序一致性的最优平衡** |

实验发现 $T_a = 8$ 在大多数任务上最优。

**Receding Horizon Control**：为进一步平滑动作，下一个推理周期可以"热启动"——用上一个周期的预测序列作为初始猜测，只需要对最新几步做去噪。

### 3.4 其他设计

- **噪声调度**：Square Cosine Schedule (iDDPM) 在所有任务上效果最好
- **DDIM 加速**：训练时 100 步去噪，推理时仅需 10 步。在 Nvidia 3080 上实现 **0.1s** 推理延迟
- **位置控制 vs 速度控制**：Diffusion Policy 在位置控制下比速度控制表现**更好**——这与几乎所有先前方法（BC-RNN、BET 等通常用速度控制）相反。原因是被扩散策略很好地处理了位置控制中的多模态性（而其他方法被多模态性困扰）

---

## 四、实验结果

### 4.1 整体性能

覆盖 15 个任务、4 个基准（Robomimic, Push-T, Block Push, Franka Kitchen）。**所有基准上一致超过 SOTA，平均提升 46.9%。**

### 4.2 多模态行为（最有说服力的定性结果）

在 Push-T 任务中：在同样的初始状态，扩散策略有 ~50% 概率选择从左边绕、~50% 概率选择从右边绕——**它学到了两种模式，每次 rollout 只执行其中一种，不混在一起。**

| 方法 | 行为 |
|------|------|
| LSTM-GMM | 偏向一边（没有真正学到多模态）|
| IBC | 同样偏向一边 |
| BET | 在两种模式间来回跳（缺乏时序一致性）|
| **扩散策略** | **学两种模式，选一种并坚持** ✅ |

### 4.3 训练稳定性（vs IBC）

IBC 在训练中 loss 曲线剧烈振荡，评估成功率也大幅波动——你不知道哪个 checkpoint 最好。而扩散策略的 loss 平滑下降，所有 checkpoint 的表现都很稳定。

**这在实践中意味着什么？** 用 IBC 你可能需要评估几十个 checkpoint 才能挑出最好的（在真机上这意味着几十次耗时的测试）。扩散策略你取最后一个 checkpoint 就行。

---

## 五、与 VLA 的关系

扩散策略不是 VLA 论文，但它是 VLA 动作头的基础：

- [[π0]] 的 **Flow Matching** 就是扩散模型的一个变体——它学习一个从噪声分布到数据分布的最优传输路径，而非一步一步去噪
- [[GR00T N1]] 的 DiT 动作头直接使用 Flow Matching
- [[Octo]] 使用 DDPM 作为动作解码头
- 把扩散动作生成和 VLM 语义理解结合起来，就是现代 VLA 的标准设计

## 六、硬件兼容性

✅ **4070 Ti Super 16GB 完美支持**
- CNN-based: 约 8-10 GB (batch_size=8)
- Transformer-based: 约 10-14 GB (batch_size=8)
- 降低 batch_size + 增加 gradient accumulation 可以稳定运行

## 七、对你的启示

1. **扩散策略是你最应该掌握的算法**——它是当前几乎所有 SOTA 机器人策略的基础
2. **位置控制 + 扩散**是一个强大的组合——它让你享受位置控制的精度，同时不被多模态性困扰
3. **训练稳定性是 diffusion 被低估的优势**——你可以花更多时间做研究，更少时间调参
4. **$T_a$（动作执行长度）是一个关键的 knobs**——在你自己训练时，花点时间扫这个参数

## PDF

[[Diffusion Policy 原文.pdf]]
