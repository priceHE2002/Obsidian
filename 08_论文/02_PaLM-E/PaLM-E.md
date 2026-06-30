---
tags:
  - 论文
  - VLA
  - LLM
  - embodied
  - 多模态
created: 2026-06-30
paper_title: "PaLM-E: An Embodied Multimodal Language Model"
paper_authors: "Danny Driess, Fei Xia et al. (Google + TU Berlin)"
paper_year: 2023
paper_venue: "ICML 2023"
paper_citations: "~900+"
paper_url: "https://arxiv.org/abs/2303.03378"
---

# PaLM-E

**PaLM-E: An Embodied Multimodal Language Model**
*Google + TU Berlin | ICML 2023 | arXiv 2303.03378*

> 把 LLM 直接接上机器人传感器，让语言模型"看见"和"触碰"物理世界。这是 RT-2 的直接前身，也是"端到端 VLA"思想的最早实现之一。

---

## 一、研究背景与核心问题

### 1.1 当时（2023 年初）的局面

2023 年初，大语言模型正处于爆发期。GPT-4、PaLM、Chinchilla 展现了惊人的"零样本"能力——它们能写代码、解数学题、翻译、写作，但这些能力**仅限于文本世界**。

同时，机器人和 embodied AI 领域有一个顽固的问题：**grounding**。机器人不能只"理解"文字，还需要知道"橘子"在物理空间中是什么、在哪里、怎么抓。传统的做法是用单独的感知模块（目标检测、6D 姿态估计）+ 单独的规划模块（运动规划、抓取规划），每个模块都需要大量的人工工程。

### 1.2 PaLM-E 的核心赌注

PaLM-E 的核心假设是：**如果把机器人传感器数据当作"另一种语言的 token"直接喂给 LLM，LLM 就能学会在物理世界中进行推理和规划。** 不需要额外的感知模块，不需要任务特定的架构——一个足够大的语言模型就够了。

作者提出了 **"multimodal sentences"** 的概念：一个输入序列可以同时包含文本 token 和来自各种传感器模态的"嵌入"。LLM 不关心中间那些嵌入来自哪里——是文字、图片还是关节角度——它只做一件事：预测下一个 token。

```
Input:  "Human: Bring me the rice chips from the drawer."
        [相机图片嵌入]
        "Robot: 1. Go to the drawers. 2. Open top drawer."
        [相机图片嵌入]
        "I see [物体检测结果]. 3. Pick the green rice chip bag..."
```

---

## 二、方法细节

### 2.1 架构

PaLM-E 的架构可以概括为：**编码器-解码器 LLM + 多模态输入投影**

**输入侧——多模态 token 的生成：**

不同类型的感知数据通过不同的编码器转换为 embedding 向量，然后直接插入到 LLM 的输入序列中。

- **图像**: 通过 ViT (Vision Transformer) 编码为 patch embedding 序列
- **场景状态向量**: 物体的 3D 位置、机器人关节角度等连续向量，通过一个小的 MLP 投影到 LLM 的 embedding 空间
- **3D 场景表示**: 用 Neural Radiance Fields (NeRF) 的特征或 Object Scene Representation Transformer (OSRT) 的 token
- **文本**: 正常 token 化

所有这些 embedding 被拼接成一个 "多模态句子"，送入 PaLM 的 decoder-only Transformer。

**一个关键细节——"嵌入插入"的位置：**

不同类型的输入嵌入被插入到序列的不同位置，来提供不同的"上下文"：
- 场景状态（物体的 3D 形状和位姿）被放在任务描述之前——这是"世界是什么样"
- 相机图像被放在具体动作步骤附近——这是"我现在看到了什么"

**输出侧——所有任务都是"下一个 token 预测"：**

PaLM-E 不区分"语言任务"和"机器人任务"。它统一用标准的 next-token prediction 损失：

$$
\mathcal{L} = -\sum_{t} \log p_\theta(x_t | x_{<t})
$$

对于机器人控制，输出可以是：
- **高层规划**: 自然语言步骤 "First grasp yellow block and place it on the table, then grasp blue block"
- **低层动作序列**: 离散化的关节目标/末端执行器位移

### 2.2 训练策略与数据

PaLM-E 采用"多任务联合训练"：

1. **互联网预训练**: PaLM 基础模型（540B 参数的 decoder-only Transformer）
2. **多模态微调**: 同时训练以下所有数据类型：
   - 标准 VQA 和图像描述（保持视觉语言能力）
   - 语言推理（保持纯语言能力）
   - 机器人操作数据（来自 Google Robot 的真实操作数据，一个办公室厨房环境，13 个机器人 17 个月收集）
   - 机器人规划数据（TAMP —— Task and Motion Planning）
   - 场景理解与问答

这种联合训练的巧妙之处在于：**训练机器人控制数据时，模型被迫学习感知和物理推理；训练 VQA 数据时，模型又保持了高层语义理解的能力。两者相互增强。**

### 2.3 "正迁移"和"负迁移"的控制

PaLM-E 的一个重要贡献是**量化了多任务训练中的正面和负面迁移**：

- **正迁移（Positive Transfer）**: 在 VQA 任务上，PaLM-E（带有机器人训练）实际上比纯 PaLM 表现更好。这表明学习物理交互知识对视觉理解有帮助。论文在 OK-VQA benchmark 上报告了这一发现。
- **负迁移（Catastrophic Forgetting）**: 论文发现了一个关键方法：在训练混合中保持足够比例的语言-only 数据，可以有效防止遗忘。具体做法是在每个 batch 中混合多种类型的数据。

---

## 三、关键实验与发现

### 3.1 机器人规划（Tabletop Manipulation）

PaLM-E 在桌面操作的任务规划上展现了惊人的零样本能力。给定一张场景图片和一条指令（如"把蓝色积木放到绿色积木上"），PaLM-E 能直接输出一个多步计划。

传统方法（TAMP）需要精确的物体模型和状态估计；而 PaLM-E 完全从像素到计划端到端完成。

在实验中：
- PaLM-E-562B（ViT-4B 视觉编码器 + PaLM-540B）达到了最高的规划成功率
- 即使在训练中从未见过的物体组合上，也能产生合理的规划

### 3.2 移动操作（Mobile Manipulation）

这是最具野心的实验。让一个移动操作机器人在真实的办公室厨房中执行一系列任务：
- "去抽屉那里"
- "打开顶层抽屉"
- "看到绿色薯片袋，拿起来放在台面上"

PaLM-E 接收来自机器人传感器的实时图像，产生下一步动作。整个过程完全端到端，没有人工设计的状态机或物体检测器。

### 3.3 模型大小的关键作用

论文做了仔细的模型大小消融实验，发现了一个清晰趋势：

- PaLM-E-8B 已经能在简单规划任务上工作
- PaLM-E-62B 明显更好
- PaLM-E-540B 达到了最佳性能

> **规模本身就是一种能力。** 更大的模型不仅"更准确"，还展现出新的行为方式——例如对未见过的物体组合做出合理推理。

### 3.4 "可问性"——LLM 可用于查询物理世界

PaLM-E 的一个独特之处在于，它不仅能执行任务，还能**回答关于场景的问题**：

- "这里有什么水果？" → 🍎🍌🍇
- "我可能最常在星期几去这栋楼？" → "星期天"（看到教堂的图片）
- "描述这张图片" → 生成准确的场景描述

这为后来 VLA 的"语义泛化"（如 [[RT-2]] 的"拿起最小的物体"）铺平了道路。

---

## 四、与 RT-2 的关系

PaLM-E 是 RT-2 的**直接前身**。两者的关系和区别：

| | PaLM-E | RT-2 |
|---|---|---|
| **语言模型** | PaLM (decoder-only) | PaLI-X 或 PaLM-E |
| **视觉编码** | ViT → embedding 插入 LLM | ViT → token 插入 VLM |
| **输出** | 语言（规划）+ 行动 token | 纯行动 token |
| **训练** | 多任务联合训练 | 联合微调（co-fine-tuning）|
| **主要使用方式** | 高层规划 | 低层控制 |
| **开源** | ❌ | ❌ |

两者的核心区别在于：PaLM-E 更偏"高层规划器"（输出自然语言步骤，然后由其他控制器执行），RT-2 则更进一步——直接输出机器人动作 token，**省掉中间所有的规划模块**，真正实现了"从像素到动作"的端到端。

---

## 五、对后续工作的影响

PaLM-E 的影响力可以从几个维度来看：

1. **"多模态句子"的设计范式**被后续几乎所有 VLA 继承：[[RT-2]]、[[OpenVLA]]、[[π0]] 都采用了"视觉 token + 文本 token → LLM → 输出"的架构
2. **"联合训练保留能力"** 的发现被 RT-2 的 "co-fine-tuning" 直接延续
3. **"规模带来能力"** 的结论为后续更大规模 VLA 的实验提供了理论依据
4. 作者列表中的 Danny Driess、Pete Florence、Karol Hausman、Sergey Levine、Chelsea Finn 等人后来都成为了 VLA 领域的核心人物，参与了 RT-2、OpenVLA、π0 等工作

---

## 六、对你的启示

1. **不要让你的模型只学机器人数据。** PaLM-E 证明了保持通用视觉语言能力对机器人控制有益
2. **"多模态句子"是一个极其通用的设计模式**——你可以把任何传感器数据（力觉、触觉、音频）插入 LLM
3. **大模型的能力不只在"准确率"**——而是在于它对未见情况的创造性应对
4. 如果你想在 16GB 显卡上做类似的事情，现在的 [[SmolVLA]] 和 BitVLA 已经在朝着这个方向努力

## PDF

[[PaLM-E 原文.pdf]]
