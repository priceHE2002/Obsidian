---
tags:
  - 论文
  - 多模态
  - 视觉-语言
  - 对比学习
  - 零样本
created: 2026-06-30
paper_title: "Learning Transferable Visual Models From Natural Language Supervision"
paper_authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever"
paper_year: 2021
paper_venue: "ICML 2021"
paper_citations: "~30,000+"
paper_url: "https://arxiv.org/abs/2103.00020"
---

# CLIP

**Learning Transferable Visual Models From Natural Language Supervision**
*OpenAI | ICML 2021 | arXiv: 2103.00020*

> 连接视觉和语言的关键桥梁。在 400M 图文对上用对比学习训练，使模型学会"把图像和对应文本映射到同一个 embedding 空间"。CLIP 的视觉编码器成为 VLA 视觉 backbone 的事实标准之一。

---

## 一、研究背景与动机

传统视觉模型受限于**固定类别标签范式**——训练时只能预测预定义类列表中的类别，模型输出 $P(y|x)$ 的条件概率分布。这不仅弱化了模型的泛化能力（模型只学会区分 1000 个 ImageNet 类，不知道"猫"的存在），而且每个新任务都需要重新标注数据。

核心洞察：**自然语言包含远比固定类别标签更丰富的信息**。如果能从互联网上数亿图文对中学习视觉概念和自然语言描述之间的关联，模型就能在训练后理解开放世界中的视觉概念。

CLIP 的目标：**设计一个可以使用自然语言作为监督信号的视觉训练框架，让模型具备零样本分类能力和强大的泛化能力。**

## 二、核心方法

**对比学习框架：**

CLIP 的核心是一个双塔架构（Dual-Encoder），将图像和文本映射到共享的 embedding 空间：

1. **图像编码器**：ResNet（改进版）或 ViT
2. **文本编码器**：Transformer（类似 GPT-2 的架构，63M 参数）

训练时，对每个 batch 的 $N$ 个图文对，构建 $N\times N$ 的相似度矩阵（正对角线是匹配对）。对比损失目标：最大化 $N$ 个正对的 cosine 相似度，最小化 $N^2 - N$ 个负对的相似度：

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^N \Bigg[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_j, T_i)/\tau)}\Bigg]
$$

这是对称版的 InfoNCE loss：图像→文本方向 + 文本→图像方向。

**核心设计选择：**

| 设计 | 选择 | 理由 |
|------|------|------|
| 对比学习而非生成式 | 预测式（captioning）太慢且低频信息干扰较多 | 对比式更高效，专攻图文对齐 |
| 大 batch size | 32768 | 提供足够多的负样本 |
| 温度系数 $\tau$ | 可学习的标量 | 调整对比损失的 sharpness |
| 文本编码器 | Transformer 而非 BoW/LSTM | 捕捉复杂语义关系 |
| Prompt ensembling | 80 种 text prompt 模板 | 提升零样本分类鲁棒性 |

**WIT 数据集（WebImageText）：** 从互联网收集的 400M 图文对，包含约 500,000 个查询词条。训练 32 epochs 相当于 12.8B 样本曝光。

## 三、关键实验与发现

1. **零样本图像分类的突破**：CLIP 在 ImageNet 上零样本达到 76.2%（ViT-L/14），与 ResNet-50 有监督训练（76.3%）几乎持平——**CLIP 从未见过任何 ImageNet 训练样本**，仅通过 "a photo of a {class}" 的文本 prompt 进行分类。

2. **惊人的分布偏移鲁棒性**：在规范情况下 CLIP 与 ResNet-101 表现相当，但在自然分布偏移测试集（ImageNet-A、ImageNet-R、ImageNet-Sketch 等）上 CLIP 大幅领先。例如 ImageNet-A（自然对抗样本）上 CLIP 76% vs ResNet-101 2%——**零样本模型比有监督模型对分布偏移更鲁棒**。

3. **Few-shot 能力**：零样本 CLIP 与 16-shot BiT-M 性能相当，少量样本微调后超过 SOTA。

4. **ViT-L/14 规模最优**：更大的视觉编码器在所有指标上表现更好，ViT-L/14 显著优于 ResNet-50 版本。

5. **训练效率**：对比学习比预测式训练（captioning）在效率上有约 4-10 倍的优势（以收敛到同等零样本性能所需计算量为基准）。

## 四、局限性与后续影响

**局限性：**
- **细粒度理解不足**：仅做全局图文匹配，缺少细粒度的空间/位置理解能力——能区分"狗和猫"，但不知道"狗在哪"
- **专业领域表现差**：医学影像、遥感、卫星图像等专业领域零样本表现显著下降
- **抽象/组合理解弱**：复杂组合概念（"蓝色的球在红色盒子右边"）理解能力有限
- **训练成本高**：ViT-L 的训练需要 256 块 GPU，一般的实验室无法复现
- **数据污染问题**：WIT 数据集与下游测试集可能存在重叠

**后续影响：**
- CLIP 奠定了视觉-语言多模态学习的主流范式
- 启发了 SigLIP（Sigmoid loss 变体，更高效）、BLIP（Bootstrapping）、ALIGN（更大数据量）等改进
- 推动了文生图模型（DALL·E、Stable Diffusion）的发展
- 开启了大模型零样本迁移的研究方向

## 五、VLA/机器人研究中的角色

CLIP 对 VLA 的影响是**核心且直接**的：

- **SigLIP**（CLIP 的 Sigmoid loss 改进版）是 OpenVLA 和 π0 等主流 VLA 的视觉编码器核心组件
- **"视觉和语言共享 embedding 空间"** 是 VLA 多模态融合的基础——让模型理解"拿起杯子"这个文本指令对应什么样的视觉场景
- CLIP 预训练权重被广泛用于初始化 VLA 的视觉编码器，减少了机器人数据微调的需求
- RT-2 的 VLM 骨干 PaLM-E / PaLI-X 也受益于 CLIP 启发的视觉-语言对齐训练
- 零样本的多模态对齐能力对于在开放世界环境中工作的机器人至关重要

## 六、对你的启示

1. **对比学习是高效的表示学习框架**：CLIP 证明了对比学习（不生成、只看匹配/不匹配）是连接不同模态的最经济有效的方式。在机器人领域，对比学习同样是连接视觉、语言和动作的重要工具。

2. **不要低估分布偏移**：有监督学习对分布偏移极其脆弱，这是 CLIP 最重要的发现之一。在机器人应用中（训练环境 vs 真实环境总是不同的），CLIP 式的开放世界理解具有天然优势。

3. **数据质量和覆盖比模型大小更重要**：CLIP 成功的核心是 400M 图文对的数据规模和质量。在机器人的场景中，数据覆盖（各种场景、光照、视角）比模型复杂度更能决定泛化能力。

4. **零样本能力打开了新范式**：CLIP 证明了不做任务特定微调也能零样本泛化。对机器人而言，这意味着一次预训练可以直接应用于多种新场景，大幅降低了部署成本。

## PDF

[[Learning Transferable Visual Models From Natural Language Supervision.pdf]]
