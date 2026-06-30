---
tags:
  - 论文
  - VLA
  - 视频基础模型
  - ICLR2026
  - NVIDIA
created: 2026-06-30
paper_title: "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning"
paper_authors: "Moo Jin Kim et al. (NVIDIA + Stanford)"
paper_year: 2026
paper_venue: "ICLR 2026"
paper_url: "https://arxiv.org/abs/2601.16163"
github: "https://github.com/NVlabs/cosmos-policy"
---

# Cosmos Policy

**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**
*NVIDIA Cosmos Lab + Stanford | ICLR 2026 | arXiv 2601.16163*

> **视频生成模型本身就懂物理——物体怎么运动、碰撞后怎么反弹、力怎么传递。为什么不直接把视频模型微调成机器人策略，而是从零开始训练一个 VLM+VLA？** Cosmos Policy 证明了：一个未修改架构的视频扩散模型，微调后就是最强的 VLA 策略之一。LIBERO 98.5%，真实世界 ALOHA 93.6%。

---

## 一、核心洞察：视频模型里有物理知识

### 1.1 VLM vs 视频模型的"知识"差异

- **VLM（如图文模型）**：知道"苹果是什么"，但不知道"球扔到墙上会反弹"
- **视频模型**：看过上亿小时的视频——物体怎么运动、水怎么流动、手怎么抓东西。这种**时空先验（spatiotemporal priors）**对机器人控制价值极高

### 1.2 Cosmos Policy 的赌注

> **如果视频模型已经理解了物理世界的大部分规律，用它做机器人策略只需要"教会它这个具体机器人怎么动"，而不需要"教它什么是物理"。**

---

## 二、方法：Latent Frame Injection

### 2.1 核心技巧

Cosmos Policy 基于 **Cosmos-Predict2**（一个 2B 参数的 latent video diffusion model）。关键创新是 **Latent Frame Injection**——把机器人相关的模态（动作、本体感觉、未来图像、价值估计）编码为"latent frames"，插入到视频扩散序列中：

```
视频帧 latent 1 → 视频帧 latent 2 → [动作 latent] → [本体感觉 latent] → [未来帧 latent] → ...
```

这种设计的精妙之处在于：**不需要任何架构修改。** Diffusion Transformer 本来就懂得怎么去噪 latent frames——现在只是多了一些"特殊"的 frame。

### 2.2 一个模型，三种能力

训练后，同一个模型可以：

1. **作为策略（Policy）**：给定当前帧 → 去噪出动作 latent → 解码为动作
2. **作为世界模型（World Model）**：给定当前帧 + 动作 → 去噪出未来帧
3. **作为价值函数（Value Function）**：给定轨迹 → 估计期望累积奖励（用于 planning）

价值函数是"副产品"——只需要在 latent 序列中加入一个特殊的 value latent，用实际累计奖励做监督。

### 2.3 双模式部署

| 模式 | 做法 | 性能 | 适用 |
|------|------|------|------|
| **Direct Policy** | 并行解码动作（不用 planning）| 快 | 简单任务 |
| **Model-based Planning** | 采样 N 个候选动作 → 世界模型展开未来 → 价值函数选最优 | 慢但强（+12.5%）| 困难任务 |

**Planning 模式不需要额外模型**——世界模型和价值函数是同一个架构的自然输出。

---

## 三、实验结果

### 3.1 基准对比

| 基准 | Cosmos Policy | 之前 SOTA |
|------|-------------|----------|
| LIBERO (仿真单臂) | **98.5%** | CogVLA 97.4% |
| RoboCasa (厨房仿真) | **67.1%** (仅 50 demos) | GR00T-N1.5+ HAMLET 66.4% (300 demos) |
| ALOHA (真实双臂) | **93.6%** | π0.5 88.6% |

### 3.2 数据效率

RoboCasa 只用 50 条演示就超过了 GR00T-N1.5 用 300 条演示的效果。视频模型的物理先验在这里发挥了巨大作用——不需要那么多数据来"教它物理"。

### 3.3 Model-based Planning 的价值

在真实世界 ALOHA 困难任务上：
- Direct Policy: 基线
- Model-based Planning (best-of-N): **+12.5 percentage points**

Planning 通过"试想多种可能的动作序列，选最优"显著提升了性能——这只有在模型有世界模型能力时才可能。

---

## 四、训练细节

| 基准 | GPU 配置 | 时间 |
|------|---------|------|
| LIBERO (~2000 demos) | 64×H100 | ~48h |
| RoboCasa (~1200 demos) | 32×H100 | ~48h |
| ALOHA (~185 demos) | 8×H100 | ~48h |

训练成本不低，但对于 98.5% 的 LIBERO 和 93.6% 的 ALOHA 来说，这个投资回报是可接受的。

---

## 五、与 VLA、WAM 的关系

Cosmos Policy 既是 VLA（它直接从观测到动作），又是 WAM（它有世界模型和 planning 能力）。它代表了 VLA 和 WAM 的**融合**——不是选边站，而是"为什么不能两者兼有？"

| | RT-2 | π0 | Motus | Cosmos Policy |
|---|---|---|---|---|
| VLM 骨干 | ✅ | ✅ | ✅ | ❌（视频骨干代替）|
| 世界模型 | ❌ | ❌ | ✅ | ✅ |
| Planning | ❌ | ❌ | ❌ | ✅ |
| 视频预训练 | ❌ | ❌ | ✅ | ✅（核心）|

---

## 六、硬件适配

❌ **4070 Ti Super 16GB**：训练需要 8×H100，不适合。但推理（2B 模型 + 量化）可能可行。

**启示**：视频模型的物理先验是极其宝贵的资源。你可以关注更小的视频模型（或蒸馏版本）来实验这个方向。

## PDF

[[Cosmos_Policy.pdf]]
