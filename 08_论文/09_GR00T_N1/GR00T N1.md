---
tags:
  - 论文
  - VLA
  - NVIDIA
  - 人形机器人
  - DiT
created: 2026-06-30
paper_title: "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots"
paper_authors: "NVIDIA GEAR Lab"
paper_year: 2025
paper_venue: "arXiv"
paper_citations: "~150+"
paper_url: "https://arxiv.org/abs/2503.14732"
---

# GR00T N1

**NVIDIA Isaac GR00T N1 - An Open Foundation Model for Generalist Humanoid Robots**
*NVIDIA GEAR Lab | 2025 | arXiv 2503.14732*

> NVIDIA 正式入场 VLA。GR00T N1 的架构与 π0 类似（双系统 VLM + DiT），但定制了更强的 NVIDIA 生态（DreamGen 合成数据、Isaac Sim 仿真、Jetson 部署）。3B 参数，Apache 2.0 开源。对人形机器人的专注使其在 VLA 生态中占据独特生态位。

---

## 一、为什么 NVIDIA 要做自己的 VLA？

### 1.1 生态战略

GR00T 不是孤立的产品，它是 NVIDIA 机器人全栈战略的一部分：

```
Isaac Sim (仿真) → DreamGen (数据生成) → GR00T N1 (模型) → Jetson (部署)
```

相比于 Physical Intelligence（π0）专注于模型本身，NVIDIA 的优势在于**全链路闭环**：
- 在 Isaac Sim 中仿真训练
- 用 DreamGen（视频世界模型）生成合成训练数据
- GR00T N1 模型训练和微调
- 在 Jetson 边缘设备上部署

### 1.2 人形机器人的特殊性

GR00T 专门面向**人形机器人**。与机械臂不同，人形机器人：
- 有更多的自由度（通常 20+ 个关节）
- 需要全身协调（走路 + 拿东西同时进行）
- 支持双手操作
- 动作空间维度更高，更复杂

---

## 二、架构详解

### 2.1 双系统设计

GR00T 也采用了双系统架构，与 [[π0]] 的思路类似但也有区别：

**System 2 (VLM, "慢系统")**
- 骨干：Cosmos-Reason-2B（NVIDIA 自研 VLM，基于 Eagle 2.5 或 Cosmos）
- 输入：相机图像 + 语言指令
- 输出：对场景的语义理解 embedding
- 运行频率：低频（~1-5Hz）

**System 1 (DiT, "快系统")**
- 架构：Diffusion Transformer (DiT)
- 输入：VLM 的语义 embedding + 本体感觉
- 输出：连续关节动作序列（通过 Flow Matching）
- 运行频率：高频（~50Hz）

### 2.2 与 π0 的核心区别

| | π0 | GR00T N1 |
|---|---|---|
| **VLM 骨干** | PaliGemma (Google) | Cosmos-Reason (NVIDIA 自研) |
| **动作生成** | Flow Matching | Flow Matching |
| **Action Expert** | DiT (300M) | DiT |
| **频率** | 50Hz | ~22-50Hz（取决于去噪步数）|
| **开源协议** | 代码/权重开源 | Apache 2.0 完全开源 |
| **数据格式** | 自研格式 | **LeRobot 兼容** ← 关键！|
| **主要目标机器人** | 机械臂 + 灵巧手 | **人形机器人** + 机械臂 |
| **合成数据** | 不支持 | **DreamGen** 视频世界模型 |

### 2.3 LeRobot 兼容——生态系统整合

GR00T 的数据格式兼容 [[LeRobot]] 的 LeRobotDataset 格式。这意味着：
- 可以直接用 LeRobot 的数据加载器
- 可以复用 HuggingFace Hub 上已有的机器人数据集
- 与 LeRobot 生态（SmolVLA, ACT, Diffusion Policy）互操作

这是一个明智的战略选择——NVIDIA 没有重复造轮子，而是选择了与社区标准对齐。

---

## 三、DreamGen——用世界模型造数据

DreamGen 是 GR00T 的配套数据生成工具，基于 **Cosmos-Predict2**（NVIDIA 的视频世界模型）。

工作流程：
1. **微调世界模型**：用真实的机器人操作视频微调 Cosmos-Predict2
2. **生成合成视频**：给定初始帧，让世界模型生成"如果机器人这样做会看到什么"
3. **提取动作**：IDM (Inverse Dynamics Model) 从合成视频中反推应该执行的动作
4. **用合成数据微调 GR00T**：将生成的 (视频帧, 动作) 对加入训练集

这意味着你可以**不需要遥操作就能扩展训练数据**。这对数据稀缺的机器人领域来说是颠覆性的。

---

## 四、支持的机器人形态

GR00T 通过 `EmbodimentTag` 机制支持多种机器人：

| 形态标签 | 机器人 | 动作空间 |
|---------|--------|---------|
| `GR1` | Fourier GR1 人形 | 关节空间 |
| `OXE_DROID` | DROID 单臂 | 末端执行器增量 |
| `AGIBOT_GENIE1` | 人形+夹爪 | 关节+夹爪 |
| `NEW_EMBODIMENT` | 自定义 | 从零微调 |

还支持 Franka Panda、SO-100/101、WidowX、RoboCasa 仿真等。

---

## 五、版本演化

| 版本 | 发布时间 | 参数 | 核心变化 |
|------|---------|------|---------|
| **N1** | 2025.3 | 2B | SigLip2 + T5, 首个开源版本 |
| **N1.5** | 2025.6 | 3B | Eagle 2.5 视觉, FLARE, DreamGen |
| **N1.6** | 最新 | 3B | Cosmos-Reason-2B VLM, 双倍 DiT 层(32), 双手+移动操作 |

N1.6 在 RTX 4090 上的推理速度：~44ms 端到端（4 步去噪，单视图）= ~22.8Hz。

---

## 六、与 π0 的选择

| 你关心什么？ | 选 π0 | 选 GR00T |
|------------|------|---------|
| 性能（SimplerEnv）| ✅ SOTA (~68%) | ⚠️ ~60% |
| 社区活跃度 | ✅ 更大 | ⚠️ 较小 |
| 开源完整度 | ✅ 完全开源 | ✅ Apache 2.0 |
| 如果你做人形机器人 | ⚠️ 有但非重点 | ✅ 核心设计目标 |
| 如果你用 LeRobot | ⚠️ 社区适配 | ✅ 原生支持 |
| 如果你需要合成数据 | ❌ 不支持 | ✅ DreamGen |
| 如果你用 NVIDIA 硬件栈 | ⚠️ | ✅ 原生优化 |
| 如果你用 16GB 显卡 | ❌ | ❌（两者都需要 24GB+）|

---

## 七、硬件约束

⚠️ **4070 Ti Super 16GB 有挑战**
- 微调建议 H100/L40/RTX 4090
- LoRA + gradient accumulation 可能在 16GB 上勉强运行（未经充分测试）
- 推理可能可行但速度会受影响

---

## 八、关键启示

1. **VLA 市场正在形成"NVIDIA vs 开源社区"的双轨格局**——类似 GPU 的 CUDA vs 开放标准
2. **LeRobot 正在成为数据格式的事实标准**——NVIDIA 选择兼容它就是一个信号
3. **合成数据 + 世界模型是 VLA 训练的下一步**——DreamGen 的方向值得关注
4. **人形机器人对 VLA 有独特的需求**——如果你不做人形，GR00T 的很多设计对你来说可能是 overkill

## PDF

[[GR00T N1 原文.pdf]]
