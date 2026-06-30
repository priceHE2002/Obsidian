---
tags:
  - 论文
  - VLA
  - 开源
  - 里程碑
  - LoRA
  - 量化
created: 2026-06-30
paper_title: "OpenVLA: An Open-Source Vision-Language-Action Model"
paper_authors: "Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti et al. (Stanford + UC Berkeley + TRI + Google DeepMind)"
paper_year: 2024
paper_venue: "CoRL 2024"
paper_citations: "~400+"
paper_url: "https://arxiv.org/abs/2406.09246"
---

# OpenVLA

**OpenVLA: An Open-Source Vision-Language-Action Model**
*Stanford + UC Berkeley + TRI + Google DeepMind | CoRL 2024 | arXiv 2406.09246*

> **OpenVLA 之于机器人，就像 Llama 之于 NLP。** 这是第一个真正开源的大规模 VLA 模型——权重、代码、训练配方全部公开，让社区可以自由下载、微调、改进。7B 参数击败了闭源的 55B RT-2-X，而且是第一个支持在消费级 GPU 上微调和推理的 VLA。

---

## 一、为什么 OpenVLA 是分水岭？

### 1.1 之前的局面

在 OpenVLA 之前：
- **RT-2** 很强，但闭源——你看不到代码，不知道具体怎么训练的，没法微调
- **RT-2-X** 在 OXE 上训练，更强，但同样闭源
- **Octo** 开源但性能有限（93M，没有真正的 VLM 推理能力）
- 如果你想研究 VLA，你只能在 RT-2 的论文上做"思想实验"

### 1.2 OpenVLA 改变了什么

OpenVLA 发布后，短短几个月内：
- 你可以下载它的权重，在你的机器人上跑零样本推理
- 你可以用 LoRA 在 24GB 显卡上微调它
- 你可以用 4-bit 量化在 7GB 显存上推理
- 你可以完整复现它的训练过程（如果你有足够的 GPU）
- 数十篇后续论文直接基于 OpenVLA 做改进

**这是社区从"仰望闭源巨头"转向"围绕开源标准迭代"的转折点。**

---

## 二、架构设计

### 2.1 三件套

OpenVLA = **视觉编码器 + 投影器 + 语言模型**

```
输入图片 (224×224)
    ↙           ↘
DINOv2        SigLIP
(空间特征)    (语义特征)
    ↘           ↙
  通道拼接
      ↓
  MLP 投影器 (2层)
      ↓
  Llama 2 7B
      ↓
  动作 tokens
```

### 2.2 视觉编码器——为什么用双编码器？

这是 OpenVLA 最关键的架构决策之一。

**DINOv2 (自监督 ViT)**：擅长捕获细粒度空间特征——物体的精确位置、形状、姿态。对机器人控制至关重要。

**SigLIP (语言监督 ViT)**：擅长语义理解——这是什么物体、它的类别和属性。对于理解"拿起红瓶子"中的"红瓶子"至关重要。

两者 features 按通道拼接后送入投影器。这个设计的动机来自 Prismatic VLM 的实验发现：双编码器融合在需要空间推理的任务上显著优于单一编码器。

### 2.3 语言模型

选择 **Llama 2 7B** 而非更大的模型（如 Llama 2 13B 或 70B），原因是：
- 7B 在性能和可用性之间有最好的平衡
- 可以在单张消费级 GPU 上做推理（bf16 需 15GB）
- 可以用 4-bit 量化 + LoRA 微调

### 2.4 动作离散化——和 RT-2 有什么不同？

与 RT-2 类似：连续动作 → 256 个离散 bin。但有一处重要改进：

**RT-2 的做法**：在整数据集的 min-max 范围上均匀分 bin。
**OpenVLA 的做法**：在整数据集的 **1st-99th 百分位**上分 bin。

为什么这个改进重要？因为 min-max 范围会被极端的异常值（如偶尔出现的很大位移）拉大，导致大多数"正常"动作被分到少数几个 bin 里，损失精细度。使用百分位可以忽略异常值，让 bin 的粒度更均匀。

**Token 分配**：Llama tokenizer 只有 ~100 个保留特殊 token，不够 256 个。OpenVLA 的做法和 RT-2 一样——直接**覆写词汇表末尾 256 个最不常用的 token**。

---

## 三、设计决策的"试错日记"

论文的一个宝贵之处在于它详细记录了开发过程中的试错经验。这在大多数 ML 论文中是看不到的。

### 3.1 VLM 骨干的选择（IDEFICS vs LLaVA vs Prismatic）

| 骨干 | 单物体任务 | 多物体语言 grounding | 选择？|
|------|----------|-------------------|------|
| IDEFICS-1 | OK | 差 | ❌ |
| LLaVA | OK | 好（+35% vs IDEFICS）| ⚠️ |
| **Prismatic** | **好** | **更好（+10% vs LLaVA）** | ✅ |

Prismatic 的优势在于双视觉编码器（DINOv2 + SigLIP）带来的空间推理能力。

### 3.2 图像分辨率：224 vs 384

**224×224 赢了。** 虽然 384×384 在 VLM 基准上更好，但在 VLA 上完全没有性能差异，而训练时间却是 224 的 3 倍。原因可能是机器人控制任务不需要那么细的视觉粒度——224 已经足够分辨物体和估计姿态。

### 3.3 视觉编码器：冻结 vs 微调

这是反直觉的发现：尽管在 VLM 训练中冻结视觉编码器通常更好，但 **VLA 训练必须微调视觉编码器**。

原因：预训练的视觉特征在全局语义上很强（"这是一个杯子"），但缺少精确的空间信息（"杯子的边缘在图像坐标 (342, 218)"）。机器人控制需要后者。

### 3.4 训练 Epoch 数

**27 个 epochs。** 这在 VLM 训练中是极其异常的——典型 VLM 训练只有 1-2 个 epoch。但在 VLA 训练中，性能持续提升直到 action token 准确率超过 95%。这说明 VLA 需要比 VLM 更多的"过遍历"来充分吸收机器人控制信号。

### 3.5 学习率

直接复用 Prismatic VLM 的学习率 **2e-5（固定），不需要 warmup**。额外惊喜——VLM 的学习率对 VLA 也是最优的。

---

## 四、实验结果

### 4.1 零样本泛化：7B 击败 55B

在 29 个评估任务上（WidowX + Google Robot 两个平台），OpenVLA 的性能比 RT-2-X（55B）高出 **16.5%** 绝对成功率。

**这是怎么做到的？** 论文分析了几个可能的原因：

1. **更大的训练数据**：970k vs 350k 轨迹
2. **更仔细的数据清洗**：发现并移除了 Bridge 数据中 18% 的全零动作帧（这些实际上是数据采集的 bug）
3. **双视觉编码器**：DINOv2 提供了更好的空间特征
4. **更长的训练**：27 epochs（RT-2 的原始训练配置更接近 1-2 epochs）

### 4.2 微调 vs 从零训练的专用策略

在 7 个不同的微调任务上（Franka 机器人），微调后的 OpenVLA 一致优于：
- Octo（微调）
- Diffusion Policy（从零训练）
- 尤其是在**多任务 + 多物体 + 语言 grounding**的场景中，VLA 的优势最明显（高出 20.4%）

这说明 VLA（有互联网预训练）的**语义理解优势在实际微调中确实转化为性能优势。**

---

## 五、消费级适配——这才是杀手锏

### 5.1 LoRA 微调

OpenVLA 支持 **LoRA (Low-Rank Adaptation)** 微调。这意味着大部分参数被冻住，只有 rank decomposition 矩阵被训练。

- LoRA rank=32 约等于全量微调的性能
- 训练参数减少到 1.4%（~100M 可训练参数）
- 显存需求大幅降低

### 5.2 4-bit 量化

通过 NF4 (NormalFloat4) 量化：
- bf16 推理：15GB → 仅需 **7GB**
- 推理速度：约 3Hz（RTX A5000 16GB）→ 约 6Hz（RTX 4090 24GB）
- **量化推理的性能损失几乎为零**

### 5.3 在你的硬件上

| 场景 | 4070 Ti Super (16GB) 可行性 |
|------|---------------------------|
| 4-bit 推理 | ✅ 完美 (~7GB) |
| bf16 推理 | ⚠️ 勉强 (~15GB, 需要关掉其他占用) |
| 4-bit QLoRA 微调 | ⚠️ 可行但紧 (12-16GB, batch_size=1) |
| 全量微调 | ❌ 不可行 (需要 ~80GB) |

---

## 六、OpenVLA 的生态影响

### 6.1 直接衍生的改进

- **OpenVLA-OFT** (2025)：正交微调 (Orthogonal Fine-Tuning)，推理速度提升 25-50×
- **SimpleVLA-RL** (ICLR 2026)：PPO/GRPO 的 RL 微调，17% → 92% 成功率
- **RL4VLA**：对比 PPO/GRPO/DPO 对 OpenVLA 的效果
- **PAIR-VLA**：视觉鲁棒的 RL 微调
- **MolmoAct** 使用 OpenVLA 作为关键基线

### 6.2 技术债务

但 OpenVLA 也有遗留问题：
- Llama 2 (2023) 已显老旧，社区在迁移到 Llama 3 / Qwen2.5
- 离散 256-bin 的动作 token——虽方便但本质上是一个粗糙的量化方案
- 6Hz 推理对很多动态任务仍不够
- 只支持单臂 7DoF → 不支持双手/移动/灵巧手

---

## 七、最重要的启示

1. **"开源"本身就是一种研究贡献**：OpenVLA 成为数十篇论文的基础设施，远远超出了它自己论文的直接引用
2. **数据清洗比大多数架构创新重要得多**：移除 18% 的全零动作帧可能是 OpenVLA 性能提升最被低估的因素
3. **VLA 和 VLM 的训练动态不同**：VLM 的经验不能直接搬过来——27 epochs vs 1-2 epochs 就是一个典型例子
4. **双视觉编码器（空间+语义）是 VLA 的正确方向**——被后续几乎所有 VLA 采用

## PDF

[[OpenVLA 原文.pdf]]
