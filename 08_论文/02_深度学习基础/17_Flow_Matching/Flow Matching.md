---
tags:
  - 论文
  - 生成模型
  - Flow Matching
created: 2026-06-30
paper_title: "Flow Matching for Generative Modeling"
paper_authors: "Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le"
paper_year: 2022
paper_venue: "ICLR 2023"
paper_citations: "~3,000+"
paper_url: "https://arxiv.org/abs/2210.02747"
---

# Flow Matching

**Flow Matching for Generative Modeling**
*Meta AI / FAIR + Weizmann Institute | ICLR 2023 | arXiv: 2210.02747*

> 提出 Flow Matching——一种比 DDPM 更简洁、更高效的生成建模框架。核心思想：学习一个确定性的速度场 $v_t(x)$，让噪声沿着最优传输（OT）直线直接流向数据，而非 DDPM 的随机蜿蜒路径。$\pi_0$ 的核心技术基石。

---

## 一、研究背景与动机

DDPM 虽然生成了高质量样本，但其随机扩散过程需要 1000 步迭代去噪，推理速度极慢。核心问题在于：DDPM 的路径是**随机的布朗桥**（Brownian Bridge）——噪声和数据的连线是一条随机曲线，需要很多步骤才能走完。

Flow Matching 的回答是：**为什么不用一条直线？** 如果噪声到数据的最优路径是直线（按照最优传输理论），那么模型只需要学一个恒定的速度场，沿着这个场从噪声走几步就到了数据。

## 二、核心方法

**连续归一化流（CNF）视角：** 定义一个时变向量场 $v_t(x)$，描述概率质量从先验 $p_0$（噪声）到目标 $p_1$（数据）的流动：

$$\frac{dx}{dt} = v_t(x_t)$$

Flow Matching 直接学习 $v_t$，而不需要通过似然最大化间接学习。

**Conditional Flow Matching (CFM)：** 核心技巧——不直接学习边际向量场（计算困难），而是学习条件向量场，两者的梯度等价：

$$L_{CFM} = \mathbb{E}\left[ ||v_\theta(t, x_t) - (x_1 - x_0)||^2 \right]$$

**最优传输 (OT) 路径：**

$$x_t = (1-t) \cdot x_0 + t \cdot x_1$$

| 特性 | DDPM | Flow Matching |
|------|------|---------------|
| 路径类型 | 随机布朗桥 | 确定性直线 (OT) |
| 训练目标 | 预测噪声 $\varepsilon$ | 预测速度 $(x_1 - x_0)$ |
| 推理步数 | 1000（DDIM: 50-100） | 5-20 |
| 路径曲率 | 随机弯曲 | 直线（最短路径） |

**与 DDPM 的深层联系：**

DDPM 的训练目标 $\mathbb{E}[||\varepsilon - \varepsilon_\theta||^2]$ 在连续极限下可以通过 reparameterization 变为速度预测形式。Flow Matching 直接将这一范式从随机过程简化为确定性传输。

## 三、关键实验与发现

- **ImageNet 64x64**：以更少的推理步数达到与 DDPM 相当的 FID
- **收敛速度**：Flow Matching 训练收敛更快——路径是直的，模型不需要"学习绕路"
- **CFM 的有效性**：条件 Flow Matching 与全局 Flow Matching 梯度等价，这是整个方法的理论基石
- **OT 路径的优越性**：最优传输路径比简单的线性插值（VP / VE 路径）生成质量更优

## 四、局限性与后续影响

**局限：**
- 原始论文侧重于无条件生成和图像生成
- 条件 Flow Matching（CFM）的实现有一些技术细节需要处理（如条件分布的建模）
- 对高维数据的 OT 配对策略值得进一步研究

**后续影响：**
- Rectified Flow (Liu et al., 2022) —— 保证路径是严格直线，进一步优化
- Stable Diffusion 3 (2024) —— 采用 Rectified Flow 替代原始扩散过程
- 在机器人领域，Flow Matching 正在取代 DDPM 成为动作生成的标准框架

## 五、VLA/机器人研究中的角色

Flow Matching 是新一代 VLA 动作生成的核心技术：

- **$\pi_0$ 的整个动作生成系统建立在 Flow Matching 上**——Action Expert 通过 Flow Matching 在 10 步去噪内生成 50 步连续动作序列
- **GR00T N1 的 DiT 动作头**使用 Flow Matching（NVIDIA 称之为 "Flow Matching Transformer"）
- **FLOWER** 使用 Rectified Flow（Flow Matching 的变体，保证路径是严格直线）
- Flow Matching 特别适合机器人：
  - 连续动作空间天然适合连续流建模
  - 50Hz 高频控制需要快速推理（10 步 vs DDPM 的 100 步）
  - 多模态动作分布可被 Flow Matching 天然建模

## 六、对你的启示

- **Flow Matching 是当前动作生成的主流选型**——DDPM 已经逐步被取代
- **必须理解 CFM 的核心逻辑**：条件 vs 边际向量场的等价性，这是理解 $\pi_0$ 动作头的前提
- **OT 路径的简洁性**是工程优势：直线路径意味着更少的去噪步数（10 步），适合机器人的高频控制
- 推理极高效（10 步去噪），16GB GPU 可轻松运行
- 从学习角度看：Flow Matching 的损失函数（预测速度）比 DDPM（预测噪声）在直觉上更清晰——直接学"从噪声到数据的直路"

## PDF

[[Flow Matching.pdf]]
