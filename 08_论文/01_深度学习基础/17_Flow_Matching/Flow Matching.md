---
tags:
  - 论文
  - 生成模型
  - Flow Matching
  - 连续正常化流
  - 最优传输
created: 2026-06-30
paper_title: "Flow Matching for Generative Modeling"
paper_authors: "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le"
paper_year: 2022
paper_venue: "ICLR 2023"
paper_citations: "~3,000+"
paper_url: "https://arxiv.org/abs/2210.02747"
github: "https://github.com/facebookresearch/flow_matching"
---

# Flow Matching

**Flow Matching for Generative Modeling**
*Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le / Meta AI (FAIR) + Weizmann Institute of Science | ICLR 2023 | arXiv: 2210.02747*

> **Pitch**: 提出 Flow Matching——一种比 [[DDPM]] 更简洁、更高效的生成建模框架。核心思想：学习一个确定性的速度场 $v_t(x)$，让噪声沿着最优传输（OT）直线直接流向数据，而非 DDPM 的随机蜿蜒路径。Conditional Flow Matching（CFM）证明：条件速度场的训练等价于边际速度场——无需显式计算不可行的边际路径。推理步数从 DDPM 的 1000 降至 5-20，是 $\pi_0$、[[GR00T N1]]、[[FLOWER]] 等新一代 VLA 动作生成的核心技术。

---

## 一、Background / Core Idea

### 1.1 DDPM 的根本局限

DDPM 虽然生成了高质量样本，但其随机扩散过程有根本弱点：
- **需要 1000 步迭代去噪**——推理极慢（机器人 50Hz 控制完全不可用）
- **采样路径是随机布朗桥**——弯曲、低效，"多余的弯路"浪费计算
- **每一步都引入随机性**——即使 DDIM 减轻了这个问题

核心问题：**为什么非要用一条随机曲线从噪声走到数据？能不能走直线？**

### 1.2 Flow Matching 的回答

Flow Matching 的回答是：**是的，可以直接走直线。** 如果噪声到数据的最优路径是直线（根据最优传输理论），那么模型只需要学一个速度场，沿着这个场从噪声走几大步就到了数据。

**关键洞察：**
- 不再需要扩散过程的 SDE 框架
- 直接学习一个确定性的向量场（vector field）$v_t(x)$
- 沿着这个场做 ODE 积分（而非 SDE 采样）
- 路径可以是最优传输直线（而不用跟随扩散路径）

### 1.3 与 DDPM 的根本区别

| 维度 | DDPM | Flow Matching |
|------|------|---------------|
| 路径类型 | 随机（SDE 布朗桥） | 确定性（ODE 流） |
| 训练目标 | 预测噪声 $\varepsilon$ | 预测速度 $v_t$ |
| 路径形状 | 弯曲（VP/VE 扩散路径） | 直线（OT 路径） |
| 推理步数 | 1000（DDIM: 50-100） | **5-20** |
| 框架复杂度 | 需理解 SDE、马尔可夫链、KL 散度 | 只需理解 ODE + 速度场 |
| 训练稳定性 | 稳定 | 更稳定（路径更简单） |

---

## 二、Method / Architecture / Technical Contribution

### 2.1 连续正常化流（CNF）框架

定义一个时变向量场 $v_t(x): [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$，通过 ODE 定义概率密度的流动：

$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x)), \quad \phi_0(x) = x$$

概率密度的变化由 push-forward 方程描述：$p_t = [\phi_t]_* p_0$，其中 $p_0$ 是简单先验（标准正态分布），$p_1$ 近似目标数据分布。

**核心挑战**：直接训练 CNF 需要 ODE 模拟（昂贵的序贯过程）或涉及不可计算积分/有偏梯度。

### 2.2 Flow Matching（FM）目标

给定目标概率路径 $p_t(x)$ 和对应的向量场 $u_t(x)$（$u_t$ 生成 $p_t$）：

$$L_{FM}(\theta) = \mathbb{E}_{t, p_t(x)}\left[ ||v_\theta(x,t) - u_t(x)||^2 \right]$$

问题：$p_t$ 和 $u_t$ 都是未知的。

### 2.3 Conditional Flow Matching（CFM）——核心理论贡献

CFM 的关键创新：**不需要知道边际概率路径和向量场**，只需定义**每个样本的条件路径**——这是可行的，因为条件路径只依赖于单个数据点。

**条件概率路径：** 对每个数据点 $x_1$，定义 $p_t(x|x_1)$ 满足 $p_0(x|x_1) = \mathcal{N}(x|0, I)$（噪声）、$p_1(x|x_1) \approx \mathcal{N}(x|x_1, \sigma_{min}^2 I)$（集中在数据点周围）。

**边际路径：** $p_t(x) = \int p_t(x|x_1) q(x_1) dx_1$

**边际向量场：** $u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1)q(x_1)}{p_t(x)} dx_1$

**CFM 损失（可直接计算）：**

$$L_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)}\left[ ||v_\theta(x,t) - u_t(x|x_1)||^2 \right]$$

**与 FM 的关系：** $\nabla_\theta L_{FM}(\theta) = \nabla_\theta L_{CFM}(\theta)$——**CFM 和 FM 的梯度等价**（定理 2）。这意味着训练 CFM 等于训练 FM。

证明概要：利用条件概率路径的边际化公式和向量场的边际化公式，通过交换积分和微分（Fubini 定理），可以证明 CFM 和 FM 只差一个与 $\theta$ 无关的常数项。

### 2.4 高斯条件路径族

论文考虑一般的条件高斯路径：

$$p_t(x|x_1) = \mathcal{N}(x|\mu_t(x_1), \sigma_t(x_1)^2 I)$$

其对应的流映射（flow map）：$\psi_t(x) = \sigma_t(x_1) \cdot x + \mu_t(x_1)$

对应的条件向量场（定理 3）：

$$u_t(x|x_1) = \frac{\sigma'_t(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu'_t(x_1)$$

### 2.5 扩散路径 vs 最优传输路径

**扩散路径（VP/VE）：**

VP 路径：$\mu_t(x_1) = \alpha_{1-t}x_1$, $\sigma_t(x_1) = \sqrt{1 - \alpha^2_{1-t}}$

VE 路径：$\mu_t(x_1) = x_1$, $\sigma_t(x_1) = \sigma_{1-t}$

扩散路径的特点：**弯曲、初始和结束阶段变化快、中间变化慢**，导致 ODE 求解困难。

**最优传输（OT）路径：**

$$\mu_t(x_1) = t \cdot x_1, \quad \sigma_t(x) = 1 - (1 - \sigma_{min})t$$

对应的向量场：

$$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{min})x}{1 - (1 - \sigma_{min})t}$$

当 $\sigma_{min} \to 0$ 时，CFM 损失简化为：

$$L_{CFM}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)}\left[ ||v_\theta(\psi_t(x_0)) - (x_1 - x_0)||^2 \right]$$

其中 $\psi_t(x_0) = (1-t) \cdot x_0 + t \cdot x_1$。

**OT 路径比扩散路径简单得多**——向量场方向在时间上恒定（图 2 直观可见），模型不需要学习"绕路"。扩散路径的 particles 可能"overshoot"最终样本，而 OT 路径保证直线。

### 2.6 训练与采样

**训练流程：**
1. 采样 $t \sim U[0,1]$
2. 采样 $x_1 \sim q(x_1)$（真实数据）
3. 采样 $x_0 \sim \mathcal{N}(0, I)$（随机噪声）
4. 计算 $x_t = (1-t)x_0 + t x_1$（OT 路径上的插值点）
5. 计算损失：$||v_\theta(x_t, t) - (x_1 - x_0)||^2$

**采样流程：**
1. 采样 $x_0 \sim \mathcal{N}(0, I)$
2. ODE 积分：$\frac{d}{dt} x_t = v_\theta(x_t, t)$，$t \in [0, 1]$
3. 积分可以用任何 ODE solver（Euler、Midpoint、RK4、dopri5）
4. 通常 5-20 步 Euler/Midpoint 就足够

### 2.7 与 DDPM 的深层联系

DDPM 的损失 $\mathbb{E}[||\varepsilon - \varepsilon_\theta||^2]$ 在连续极限下可以重参数化为速度预测形式。Flow Matching 不是对 DDPM 的简单改进，而是将整个范式从随机过程完全重写为确定性传输。CFM 框架包含了扩散路径作为特殊实例（当 $\mu_t, \sigma_t$ 设置为 VP/VE 时）。

### 2.8 Code 级实现要点

```python
# 训练循环核心（PyTorch 伪代码）
t = torch.rand(batch_size)  # U[0,1]
x_1 = data  # [B, C, H, W]
x_0 = torch.randn_like(x_1)  # 噪声

# OT 路径上的插值点
x_t = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1

# 目标速度
target = x_1 - x_0

# 预测速度
pred = model(x_t, t)

# CFM 损失
loss = F.mse_loss(pred, target)
```

---

## 三、Experiments and Key Findings

### 3.1 密度建模与样本质量：ImageNet

使用相同的 U-Net 架构（来自 Dhariwal & Nichol, 2021）在不同损失下比较：

| 方法 | CIFAR-10 NLL↓ | FID↓ | NFE↓ |
|------|--------------|------|------|
| DDPM | 3.12 | 7.48 | 274 |
| Score Matching | 3.16 | 19.94 | 242 |
| ScoreFlow | 3.09 | 20.78 | 428 |
| **FM w/ Diffusion** | **3.10** | **8.06** | **183** |
| **FM w/ OT** | **2.99** | **6.35** | **142** |

FM w/ OT 在所有指标上最优——**更低的 NLL、更低的 FID、更少的 NFE**。即使 FM 使用扩散路径（FM w/ Diffusion），也比 DDPM 更稳定、收敛更快。

| 方法 | ImageNet 32 NLL↓ | FID↓ | NFE↓ |
|------|-----------------|------|------|
| FM w/ OT | **3.53** | **5.02** | **122** |
| FM w/ Diffusion | 3.54 | 6.37 | 193 |
| DDPM | 3.54 | 6.99 | 262 |

| 方法 | ImageNet 64 NLL↓ | FID↓ | NFE↓ |
|------|-----------------|------|------|
| FM w/ OT | **3.31** | **14.45** | **138** |
| FM w/ Diffusion | 3.33 | 16.88 | 187 |
| DDPM | 3.32 | 17.36 | 264 |

### 3.2 训练收敛速度

FM w/ OT 收敛最快。在 ImageNet 64×64 训练曲线中，FM-OT 的 FID 下降速度显著快于 FM-Diff 和 SM-Diff，且最终 FID 更低。

### 3.3 采样效率

**固定步数求解器效果：**

在 ImageNet 32×32 上，使用固定步 Euler/Midpoint/RK4 求解器：
- FM w/ OT：10 步即可达到 FID ~20（可用质量），20 步达到 FID ~10
- FM w/ Diffusion：需要约 2× 的步数达到相同 FID
- SM w/ Diffusion：需要约 3× 的步数

**OT 路径的优势**：在低 NFE 下，FM-OT 始终产生最佳 FID。4 步即可产生可辨认图像（图 16, 17）。

### 3.4 采样路径可视化

扩散路径的采样可视化（图 6）：前 80% 的时间看起来都是噪声，只在最后时刻"突然"出现图像。OT 路径的噪声减少几乎是线性的，图像结构逐步浮现。

### 3.5 超分辨率条件生成

64×64 → 256×256 图像超分辨率：

| 方法 | FID↓ | IS↑ |
|------|------|-----|
| Reference | 1.9 | 240.8 |
| Regression | 15.2 | 121.1 |
| SR3 | 5.2 | 180.1 |
| **FM w/ OT** | **3.4** | **200.8** |

FM-OT 在 FID 和 IS 上均大幅超越 SR3（当时 SOTA 超分辨率方法）。

---

## 四、Limitations and Challenges

1. **条件 OT 不保证边际 OT**：虽然条件向量场是最优传输路径，但**边际向量场不一定是最优传输**（marginalization 可能破坏 OT 性质）。[[Rectified Flow]] 后续解决了这一问题。

2. **原始论文验证范围有限**：主要关注无条件图像生成和超分辨率，未在视频、3D 等模态上验证。不过后续工作已证明可扩展到各种模态。

3. **CFM 理论假设**：定理 2 假设 $p_t(x) > 0$（对所有 $x,t$），这在有限时间路径末期（$t \to 1$）可能不严格成立（当 $\sigma_{min} \to 0$ 时）。

4. **U-Net 仍然是骨干**：虽然 Flow Matching 改进了目标函数，但报告的实验仍使用 U-Net 架构（如 [[DiT]] 的工作展示了 ViT 在扩散/流匹配中更好）。

5. **与其他工作的并发性**：Rectified Flow（Liu et al., 2022）和 Stochastic Interpolants（Albergo & Vanden-Eijnden, 2022）同时提出了类似方法，Flow Matching 并非唯一贡献者。

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关联 | 时间 |
|---------|------|------|
| **Rectified Flow** | 保证边际路径严格直线，"straighten"路径 | 2022 |
| **Stable Diffusion 3 (MMDiT)** | 采用 Rectified Flow 替代原始扩散 | 2024 |
| **[[DiT]]** | ViT 替代 U-Net 作为 Flow Matching 骨干 | 2022 |
| **[[π0]]** | **Action Expert 使用 Flow Matching 生成 50 步动作序列** | 2024 |
| **[[GR00T N1]]** | **DiT 动作头（Flow Matching Transformer）~50Hz 动作生成** | 2025 |
| **[[FLOWER]]** | **使用 Rectified Flow（Flow Matching 变体）** | 2024 |
| **[[Cosmos Policy]]** | Video DiT + Flow Matching | 2025 |

### 为什么 Flow Matching 特别适合机器人动作生成

1. **连续动作空间**天然适合连续流建模——机器人的末端执行器位姿、关节角度等动作变量是连续的
2. **高频控制需要快速推理**——10 步（vs DDPM 的 100 步）使高频（50Hz）控制成为可能
3. **多模态动作分布可被天然建模**——Flow Matching 在 t=0 注入不同噪声 → 不同的动作样本，自然支持多模态行为
4. **确定性路径更可预测**——ODE 采样无随机性，在安全关键的机器人控制中是优势

### Flow Matching vs Diffusion Policy

| 维度 | Diffusion Policy (DDPM) | Flow Matching |
|------|------------------------|---------------|
| 动作生成步数 | 50-100 | 5-20 |
| 路径 | 随机（SDE） | 确定性（ODE） |
| 实时性 | 勉强（100ms） | 好（20ms） |
| 多模态建模 | 好 | 好 |
| 采用模型 | Octo | [[π0]], [[GR00T N1]], [[FLOWER]] |

---

## 六、Implications for You / Hardware Compatibility

| 维度 | 评价 |
|------|------|
| 训练硬件 | ✅ 与 DDPM 同等硬件需求（CIFAR: 单卡, ImageNet: 8×A100） |
| 推理硬件 | ✅ 极轻量——OT 路径 10 步推理所需计算量远小于 DDPM。16GB GPU 轻松运行 |
| 代码复杂度 | ✅ 比 DDPM 更简单（无 SDE/马尔可夫链框架），约 100 行 PyTorch 核心 |
| 对 VLA 的意义 | ✅ **必须理解**——[[π0]] 和 [[GR00T N1]] 的核心理论基础 |
| 实时控制 | ✅ 10 步推理可在 16GB GPU 上 < 50ms 完成，基本达到控制实时性要求 |

**核心启示：**
1. **Flow Matching 是当前动作生成的主流选型**——DDPM 在新工作中已被逐步取代。**新项目直接从 Flow Matching 开始**
2. **必须理解 CFM 的核心逻辑**：条件 vs 边际向量场的等价性，这是理解 [[π0]] Action Expert 动作头的前提
3. **OT 路径的简洁性是工程优势**：直线路径意味着更少的去噪步数（10 步），适合机器人的高频控制
4. **[[Rectified Flow]] 是进一步演进**——保证边际路径严格直线，被 [[FLOWER]]、Stable Diffusion 3 采用

---

## PDF

[[Flow Matching 原文.pdf]]
