---
tags:
  - 论文
  - 自监督学习
  - 视觉特征
  - ViT
  - 蒸馏
created: 2026-06-30
paper_title: "DINOv2: Learning Robust Visual Features without Supervision"
paper_authors: "Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski"
paper_year: 2023
paper_venue: "TMLR 2024"
paper_citations: "~5,000+"
paper_url: "https://arxiv.org/abs/2304.07193"
---

# DINOv2

**DINOv2: Learning Robust Visual Features without Supervision**
*Meta AI | TMLR 2024 | arXiv: 2304.07193*

> Meta AI 推出的自监督视觉基础模型，在 142M 精选图像上训练，生成的特征同时适用于图像级别和像素级别任务，且无需任何微调。OpenVLA 的视觉编码器之一——专门负责空间特征的提取。

---

## 一、研究背景与动机

自监督视觉预训练在 DINOv2 之前已经取得了显著进展，但存在一个问题：**现有的自监督方法（DINO、iBOT、MAE 等）虽然各自在某些任务上表现良好，但没有一个方法能同时适用于所有类型的视觉任务**——从图像级别（分类、检索）到像素级别（分割、深度估计）。与此同时，强监督方法（如 CLIP、SigLIP）需要大量标注数据，且特征的空间理解能力相对有限。

核心问题：**能否训练一个"视觉通用特征提取器"，其生成的特征在图像级和像素级任务上同时表现出色，且不需要任何微调？**

DINOv2 的答案是：将多种自监督目标（对比学习 + masked image modeling + 正则化）结合起来，在更大、更干净的数据集上训练更大的模型，并通过知识蒸馏产出不同规模的变体。

## 二、核心方法

**训练目标：三种损失的组合：**

$$
\mathcal{L} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + \mathcal{L}_{\text{KoLeo}}
$$

| 损失 | 来源 | 作用 | 说明 |
|------|------|------|------|
| $\mathcal{L}_{\text{DINO}}$ | DINO (2021) | 对比学习：教师-学生框架，跨视角一致性 | 确保不同裁剪视角下的特征一致性 |
| $\mathcal{L}_{\text{iBOT}}$ | iBOT (2021) | MIM：masked image modeling + 教师伪标签 | 弥补对比学习的局部细节缺失 |
| $\mathcal{L}_{\text{KoLeo}}$ | KoLeo (2018) | 批次内特征多样性正则化 | 防止特征坍塌，保证多样性 |

**关键技术元素：**

| 技术 | 实现 |
|------|------|
| **教师-学生框架** | EMA 更新的教师网络（不计算梯度），学生网络从教师学习 |
| **自注意力蒸馏** | 学生的自注意力图对齐教师的注意力图（来自 DINO） |
| **MIM 训练** | 随机 mask 部分 patch，用教师输出作为重建目标的伪标签（来自 iBOT） |
| **FlashAttention** | 集成 FlashAttention 加速训练 |
| **数据工程** | 142M 精选图像（Curated Data），从 1.2B 数据中筛选出的高质量子集 |

**训练设置：**

- 基础模型 ViT-g/14（1.1B 参数，Giant 版本）在 142M 精选图像上预训练
- 通过**知识蒸馏**从 ViT-g 蒸馏出 ViT-S/B/L 版本，保持高质量
- 使用大批量训练（批次大小 8192）和混合精度训练

## 三、关键实验与发现

1. **零样本特征质量惊人**：无需微调，直接使用 DINOv2 特征的线性分类器接近有监督 SOTA——ViT-g 在 ImageNet 上 linear probe 达到 86.6%（接近有监督 88.6%）。

2. **密集预测的突破**：DINOv2 特征在深度估计和语义分割任务上**无需微调即可使用**——这是此前自监督方法从未实现的。注意力图天然具有空间结构，可以直接用于像素级任务。

3. **蒸馏的有效性**：ViT-L（307M）蒸馏版在大多数任务上性能接近 ViT-g（1.1B）的 95%，但计算量仅约 1/4，证明了蒸馏的高效性。

4. **数据质量 > 数据数量**：142M 精选数据 vs 原始 1.2B 候选数据——精挑细选的 142M 数据在多项任务上明显优于全部使用 1.2B 数据。

5. **多任务统一**：DINOv2 是第一个在图像分类、语义分割、深度估计、检索等任务上同时达到 SOTA 或接近 SOTA 的自监督学习方法。此前的方法基本只能专精于某一类任务。

6. **对比 DINOv1 的显著提升**：DINOv2 在像素级任务上的提升远超图像级任务——深度估计提升 +15%，语义分割提升 +10%，这得益于 iBOT 的 MIM 训练弥补了 DINO 缺乏局部细节的缺陷。

## 四、局限性与后续影响

**局限性：**
- **预训练计算成本极高**：ViT-g 训练需要大量 GPU（~2000 GPU-days），大多数实验室无法复现
- **特征虽强但仍是"冻结的"**：在 VLA 等下游任务中，DINOv2 特征仍需微调以获得最佳的机器人控制性能
- **缺少语言对齐**：DINOv2 是纯视觉模型，不包含语言理解能力，需要配合语言模型使用
- **特征维度固定**：输出特征维度不可调整（依赖 ViT 配置），在某些场景下可能过杀或不足

**后续影响：**
- DINOv2 成为自监督视觉特征提取的 SOTA 基线
- 被 OpenVLA 等 VLA 系统采用，证明了双视觉编码器（DINOv2 + SigLIP）设计的有效性
- "空间特征 + 语义特征" 的双编码器范式被 FLOWER 等后续工作继承和验证
- 大规模数据蒸馏的训练范式影响了后续视觉基础模型的训练策略

## 五、VLA/机器人研究中的角色

DINOv2 对 VLA 的影响是**直接且核心**的：

- **OpenVLA 的双视觉编码器中 DINOv2 负责空间特征**——物体精确位置、形状、姿态的提取
- DINOv2 提供空间理解能力，SigLIP 提供语义理解能力，**两者互补构成完整的视觉表征**
- **"空间特征 + 语义特征"的双编码器设计**被 FLOWER 等后续 VLA 继承和验证——这是一个被证明有效的设计范式
- DINOv2 的高质量特征使得 7B 参数的 OpenVLA 在多项任务上超越 55B 参数的 RT-2-X，说明**视觉编码器的质量对 VLA 整体性能有决定性影响**
- 在机器人操控中，DINOv2 的 patch-level 特征可以提供精确的像素-语义对应关系，对抓取点预测和物体姿态估计至关重要

## 六、对你的启示

1. **组合优于单一**：DINOv2 成功的关键是将三种自监督损失的优点结合在一起——对比学习（全局语义）+ MIM（局部细节）+ 多样性正则化（防止坍塌）。这不是简单的"越复杂越好"，而是**针对不同层面的弱点进行互补设计**。

2. **视觉编码器是 VLA 的基石**：OpenVLA 使用 DINOv2 后在 7B 规模超越 RT-2-X 的 55B 规模，说明 **VLA 的性能上限高度依赖视觉编码器的质量**。在 VLA/具身智能项目中，视觉backbone 的选择可能是最重要的架构决策之一。

3. **空间 vs 语义是互补维度**：DINOv2（空间）和 SigLIP（语义）的双编码器设计揭示了一个通用原则——理解视觉场景需要同时捕获"什么东西"（语义）和"在哪里"（空间），单一编码器很难做到两方面都完美。

4. **大规模蒸馏是实用策略**：用超大模型（1.1B ViT-g）蒸馏出小模型（ViT-S/B/L）在保持质量的同时大幅降低成本，这种策略在自己的实验中也很适用——先用大模型探索上限，再蒸馏到适合部署的规模。

## PDF

[[DINOv2.pdf]]
