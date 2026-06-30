---
tags:
  - 论文
  - VLA
  - 开源
  - 通用策略
created: 2026-06-30
paper_title: "Octo: An Open-Source Generalist Robot Policy"
paper_authors: "Octo Model Team (UC Berkeley + Stanford + CMU + Google DeepMind)"
paper_year: 2024
paper_venue: "RSS 2024"
paper_citations: "~300+"
paper_url: "https://arxiv.org/abs/2405.12213"
---

# Octo

**Octo: An Open-Source Generalist Robot Policy**
*UC Berkeley + Stanford + CMU + Google DeepMind | RSS 2024 | arXiv 2405.12213*

> 它是第一个真正意义上的开源通用机器人策略。虽然性能已落后，但 Octo 的模块化架构设计哲学和"任意 token 输入 → 任意 token 输出"的灵活范式，深刻影响了后续的所有开源 VLA 项目。

---

## 一、背景与定位

### 1.1 Octo 之前的开源生态

2024 年初，开源机器人学习社区面临一个困境：

- **有强大的闭源模型**（RT-2, RT-2-X），但无法使用和复现
- **有开源的具体任务策略**（ACT, Diffusion Policy），但它们都是从零训练的，缺乏泛化能力
- **有开源的大数据集**（Open X-Embodiment），但没有在其上预训练的、可直接使用的开源策略

Octo 的定位就是**填补这个空白**：一个在 OXE 上预训练的、开源的、可以控制多种机器人的通用策略。

### 1.2 核心理念

Octo 的设计哲学是 **"一个 transformer，任意 token 进，任意 token 出"**。不预设特定的传感器配置、动作空间或任务定义方式——所有这些都通过"token"的抽象来统一处理。

---

## 二、架构设计：灵活性的代价

### 2.1 Token 化方案

Octo 将所有输入-输出都统一为 token 序列：

**输入 tokens（按顺序拼接）：**

1. **任务 token（1 个）**: 语言指令（通过预训练的 T5-base 编码器转为单个 embedding）**或** 目标图像（通过浅层 CNN 转为一个 token）
2. **观测 tokens（多个）**: 每个相机视角的每帧图像 + 本体感觉（关节角度等）
   - 图像 → 预训练的 ResNet 或 ViT → embedding
   - 本体感觉 → MLP → embedding
3. **读入 tokens（可选的）**: 动作预测的"前缀"，初始化为可学习的 embedding

**输出 tokens：**
- 动作 tokens → 线性投影头 → 具体的关节位置或末端执行器位移

### 2.2 Transformer 骨干

使用标准 decoder-only Transformer（类似 GPT 架构）。

关键设计：**参数数量** —— Octo-Base (93M) 和 Octo-Small (27M)。这两个尺寸都是有意选得很小的——目的是在消费级硬件上可微调和推理。但这也成了 Octo 最大的性能瓶颈。

### 2.3 模块化注意力设计

Octo 的注意力机制对不同 token 类型做了不同的处理。这增加了灵活性，但也增加了实现复杂度：

- **观测 tokens**：全自注意力 → 所有观测之间可以互相关注
- **任务 token**：交叉注意力到观测 tokens → 任务定义可以"查询"观测信息
- **动作 tokens**：因果自注意力 → 动作序列生成是自回归的（每步只能看之前的步）

### 2.4 为什么 Octo 不是真正的 VLA？

Octo 常被归类为 VLA，但这不准确。关键区别：

| | Octo | VLA（如 RT-2, OpenVLA）|
|---|---|---|
| 语言模型 | T5-base（仅作为文本编码器）| 集成在骨干中（LLaMA, PaLI-X）|
| 视觉编码 | ResNet/ViT（从零训练或预训练）| VLM 的视觉编码器（互联网预训练）|
| 知识来源 | 仅来自 OXE 数据 | OXE 数据 + 互联网规模预训练 |
| 语义推理 | 有限（T5 embedding 是固定向量）| 强（LLM 可以推理）|

> Octo 更像是一个"跨形态通用策略"（Generalist Robot Policy, GRP），而不是 VLA。它的语言理解仅限于 T5 的文本 embedding，没有 LLM 级的推理能力。

---

## 三、训练细节

### 3.1 数据

在 800k OXE 轨迹上预训练。关键的数据工程贡献是**数据混合权重 (mixture weights)**：

- 对质量高、多样性好的数据集加高权重
- 对质量低、重复性高的数据集降权重或移除
- 这套权重方案后来被 [[OpenVLA]] 直接复用

### 3.2 训练配置

- Batch size: 256（分布式训练）
- 训练步数: 300K
- 优化器: AdamW
- 微分率: 根据参数规模在 Octo-Base 和 Octo-Small 之间略有不同

### 3.3 微调

Octo 最实用的特性是**模块化微调**：
- 只需替换输入头（新的相机配置）或输出头（新的动作空间）
- 核心 Transformer 参数可以被冻住（或微调）
- 在消费级 GPU 上几小时内可完成

---

## 四、实验结果

### 4.1 零样本泛化（Out-of-the-box）

Octo 在 9 种不同的机器人平台上进行了零样本评估。它能直接（不经微调）控制：
- WidowX (Bridge V2)
- UR5
- RT-1 Google Robot
- Berkeley Bimanual
- CMU Baking Robot
- Stanford Coffee Robot
- 等等

**但性能有限**——大部分任务的成功率在 10-40% 之间，远不如后来的 OpenVLA。

### 4.2 微调后的表现

微调能显著提升性能。在 WidowX 操作任务上，微调后的 Octo 接近从零训练的专用策略（如 ACT/Diffusion Policy），但所需的微调数据远少于从零训练。

### 4.3 关键局限

1. **语言指令遵循能力弱**：在多物体场景中，Octo 经常抓错物体
2. **空间推理几乎为零**：如"把物体放到左边"这类空间关系任务几乎完全失败
3. **对未见环境的泛化差**：见过的新背景经常导致崩溃
4. **速度慢**：自回归动作 token 生成比 Diffusion Policy 的并行去噪慢

---

## 五、Octo 的历史定位与遗产

### 5.1 Octo 被超越的原因

到 2024 年中，Octo 迅速被 OpenVLA 超越。核心原因：

1. **架构不是 VLA**：没有利用 VLM 的互联网预训练知识
2. **模型太小**：93M vs 7B，参数差距近 100 倍
3. **数据不足**：800k vs OpenVLA 的 970k（但差距不大）
4. **动作生成方式落后**：自回归 token 生成 vs 扩散/流匹配

### 5.2 Octo 的持久贡献

虽然性能已落后，但 Octo 留下了重要的遗产：

1. **模块化设计的哲学**被 [[OpenVLA]] 和 [[π0]] 的代码库继承
2. **数据混合权重** 成为 OXE 数据使用的标准参考
3. **"任意 token → 任意 token"的灵活性范式** 影响了 LeRobot 的设计
4. **开源精神** 推动了整个社区——证明了开源通用策略是可行的，为 [[OpenVLA]] 的发布铺平了道路

---

## 六、对现在的你还有用吗？

**学习价值 > 使用价值。**
- Octo 的架构设计论文是理解"如何设计灵活的多模态 transformer 策略"的好教材
- 但作为训练起点，你应该直接用 [[OpenVLA]] 或 [[SmolVLA]]
- Octo 简化的代码库（相比 OpenVLA 的 7B VLM）可能更容易理解核心概念

## PDF

[[Octo 原文.pdf]]
