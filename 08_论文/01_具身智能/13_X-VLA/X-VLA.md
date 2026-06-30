---
tags:
  - 论文
  - VLA
  - 跨形态
  - ICLR2026
  - Soft-Prompt
created: 2026-06-30
paper_title: "X-VLA: Soft-Prompted Transformer for Scalable Cross-Embodiment VLA"
paper_authors: "Jinliang Zheng et al. (清华AIR + 上海AI Lab + 北大)"
paper_year: 2025
paper_venue: "ICLR 2026"
paper_url: "https://arxiv.org/abs/2503.10631"
github: "https://github.com/2toinf/X-VLA"
---

# X-VLA

**X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model**
*清华AIR + 上海AI Lab + 北大 | ICLR 2026 | arXiv 2503.10631*

> **跨形态 VLA 最优雅的解决方案：每个数据源加几个可学习的 embedding 向量（仅 0.04% 额外参数），0.9B 参数横扫 LIBERO (97-98%)。** IROS 2025 世界冠军方案。证明了 Soft Prompt——这个 NLP 的简单技术——在机器人上的巨大威力。

---

## 一、核心问题：跨形态的异构性

不同机器人有不同的：
- 动作空间（不同维度、绝对值 vs 增量、关节 vs 末端执行器）
- 相机数量和位姿
- 控制频率
- 运动学特性（每个关节的力矩和速度不同）

之前的解决方案各有限制：
- **每个机器人一个头** → 参数膨胀，不能泛化到新形态
- **统一动作空间** → 丢失了形态特有的信息
- **忽略差异** → 次优性能

### 1.1 Soft Prompt 的灵感

X-VLA 的想法来自 NLP 中的 **Soft Prompt Tuning**：给每个下游任务加几个可学习的 token embedding，模型的其他部分保持冻住。

在 VLA 的上下文中：**给每个数据源分配一组 learnable soft prompt vectors。** 这些 vector 自动学会了编码该数据源的"身份"——坐标系方向、控制模式、视角偏差等。

---

## 二、架构详解

### 2.1 输入流

X-VLA 使用标准 Transformer encoder（**没有使用 DiT 的复杂去噪设计**）：

```
输入序列: [Prompt_Tokens, Image_Tokens, Text_Tokens, Proprio_Tokens, Noise_Action_Tokens]
```

- **Prompt Tokens**：每个训练数据源有 $K$ 个 learnable embedding（$K$ 很小，论文中用 16-64）
- **Image Tokens**：多视角图像通过预训练 VLM (Florence-Large) 编码
- **Text Tokens**：语言指令 tokens
- **Proprioceptive Tokens**：本体感觉（关节角、末端位姿）通过 MLP 嵌入
- **Noise Action Tokens**：Flow Matching 的噪声输入

### 2.2 Soft Prompt 的工作机制

训练时，每个样本的 prompt tokens 来自其对应的数据源。**同一个机器人的不同数据集可以有不同的 prompt（因为"数据集上下文"可能不同）。**

推理时，对于新机器人：
1. Prompt Warm-up：冻住模型，只用少量数据学习新的 prompt tokens
2. Joint Fine-tuning：解冻部分模型参数，联合微调

这使得新机器人的适配极其高效——只需学习几个 prompt token。

### 2.3 Flow Matching 动作生成

X-VLA 使用 Flow Matching（与 [[π0]] 和 [[FLOWER]] 类似），但 transformer 架构更简洁——没有 DiT 的 AdaLN, 直接用标准 Transformer decoder blocks。

---

## 三、训练管道

### Phase I — 预训练

- 数据：290K 条轨迹，7 个平台，5 种机器人类型
- 任务：联合训练让模型学会"形态无关"的通用策略
- Prompt tokens 同时学习各自数据源的特征

### Phase II — 适配（两步走）

**Step 1: Prompt Warm-up**
- 冻住全部 Transformer 参数
- 只为新机器人学习新的 soft prompt tokens
- 数据需求：极少量（论文实验中用了几十到几百条轨迹）

**Step 2: Joint Fine-tuning**
- 解冻部分层 + prompt tokens 联合微调
- 进一步提升目标机器人上的性能

---

## 四、实验结果

### 4.1 核心基准

| 基准 | X-VLA (0.9B) | 对比模型 |
|------|------------|---------|
| LIBERO Spatial | **98%** | OpenVLA-OFT (7B): 97.6% |
| LIBERO Object | **98%** | OpenVLA-OFT (7B): 98.4% |
| LIBERO Goal | **97%** | OpenVLA-OFT (7B): 97.9% |
| LIBERO Long | **97%** | OpenVLA-OFT (7B): 94.5% |
| SimplerEnv WidowX | **96%** | GR00T-N1: -- |
| VLABench | **51.1%** | GR00T-N1: 39.7% |
| RoboTwin-2.0 Easy | **70%** | π0: 60.2% |
| RoboTwin-2.0 Hard | **39%** | π0: 39.8% |
| 现实世界 Bridge v2 | **82%** | -- |
| 现实世界叠衣服 | **~100%** | -- |

### 4.2 Scaling Law

X-VLA 在没有饱和迹象的情况下展现了清晰的 Scaling Law：
- 模型从 0.3B → 0.9B → 持续提升
- 数据从 100K → 290K → 持续提升
- 数据源从 3 → 7 → 持续提升

**"越大越好"在 X-VLA 上仍然成立**——这意味着 Soft Prompt 架构有巨大的扩展潜力。

### 4.3 IROS 2025 冠军

X-VLA 在 AGIBOT World Challenge (Manipulation track) 获得第一名。

---

## 五、Soft Prompt vs 其他跨形态方案

| 方案 | 额外参数 | 适配效率 | 泛化能力 |
|------|---------|---------|---------|
| 每种形态一个策略头 | 多 | 需要从头训练头 | 不能泛化到新形态 |
| 统一动作空间 | 零 | 直接可用 | 中等（丢失形态特征）|
| **Soft Prompt** | ~0.04% | Prompt warm-up（几十条轨迹）| **强（保留全部形态信息）** |

---

## 六、对你的启示

1. **Soft Prompt 是一个被严重低估的跨形态技术**——极其简单但极其有效
2. **不需要 DiT 也能达到 SOTA**——X-VLA 用标准 Transformer + Flow Matching，架构复杂度远低于 π0/GR00T
3. **0.9B 参数就能打败 7B**——架构设计（Soft Prompt + 跨形态预训练）比规模更重要
4. 如果你在设计自己的 VLA 训练框架，Soft Prompt 是一个**零成本增加泛化能力**的技巧

## 七、硬件适配

⚠️ **4070 Ti Super 16GB**：推理可能可行（0.9B 参数），微调取决于具体实现。Prompt Warm-up 只需要极少显存（仅学习 prompt tokens 和部分层）。

## PDF

[[X-VLA.pdf]]
