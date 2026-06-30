---
tags:
  - 论文
  - Transformer
  - 视觉模型
  - 图像分类
created: 2026-06-30
paper_title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
paper_authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby"
paper_year: 2020
paper_venue: "ICLR 2021"
paper_citations: "~60,000+"
paper_url: "https://arxiv.org/abs/2010.11929"
---

# ViT

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
*Google Research | ICLR 2021 | arXiv: 2010.11929*

> 将 Transformer 直接应用于图像——不做任何 CNN 特定的归纳偏置。将图像切成 16×16 patches，当作"视觉单词"喂给 Transformer，在大数据预训练下超越 CNN。这篇文章证明了"CNN 不是做视觉的必要条件"，直接开启了 Transformer 在 CV 的时代。

---

## 一、研究背景与动机

在 ViT 之前，CV 领域完全由 CNN 主导。CNN 拥有两个关键归纳偏置——**平移等变性**（translation equivariance）和**局部性**（locality）。这些归纳偏置在小数据上效果极好，但也限制了 CNN 对长距离依赖关系的建模能力。

与此同时，NLP 领域的 Transformer 已经展现出强大的序列建模能力，且在足够的数据下，**没有归纳偏置的模型可以比带有强偏置的模型学得更好**。BERT 和 GPT 的成功说明：如果数据足够大，模型可以自行学到数据中的结构，不需要手动设计偏置。

核心问题：**如果直接将 Transformer 应用于图像，不做任何 CNN 的归纳偏置，需要多少数据才能超越 CNN？**

## 二、核心方法

ViT 的设计极为简洁——尽量"不做 CV 特定设计"，直接复用 NLP Transformer。

**整体流程：**

1. **图像 Patch 化**：输入 $224\times 224\times 3$ 图像 → 分成 $(224/16)^2 = 196$ 个 patch，每个 patch 展平为 $16\times 16\times 3 = 768$ 维向量
2. **线性投影**：每个 patch 向量通过可学习线性投影变为 token 嵌入
3. **添加位置编码**：可学习的 1D 位置编码（每个 patch 位置一个可学习向量）
4. **[class] token**：额外添加一个特殊的 class token（借鉴 BERT 的 [CLS]），其最终输出对应的向量用于分类
5. **Transformer Encoder**：标准的 Multi-Head Self-Attention + MLP + LayerNorm + Residual Connection
6. **分类头**：[class] token 的最终表示通过 MLP 输出类别概率

**架构配置：**

| 模型 | Patch Size | Layers | Hidden Dim | Heads | Params |
|------|-----------|--------|-----------|-------|--------|
| ViT-B/16 | 16×16 | 12 | 768 | 12 | 86M |
| ViT-L/16 | 16×16 | 24 | 1024 | 16 | 307M |
| ViT-H/14 | 14×14 | 32 | 1280 | 16 | 632M |

**关键结构：**

$$
\begin{aligned}
\mathbf{z}_0 &= [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}} \\
\mathbf{z}'_\ell &= \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1} \\
\mathbf{z}_\ell &= \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell \\
\mathbf{y} &= \text{LN}(\mathbf{z}_L^0)
\end{aligned}
$$

## 三、关键实验与发现

1. **大数据预训练是前提**：在 ImageNet-1K（1.3M 图像）上训练时，ViT 比 ResNet 差 3-4%。但在 ImageNet-21K（14M）或 JFT-300M（303M）上预训练后，ViT 全面超越 SOTA CNN。

2. **ViT-H/14 在 ImageNet 达到 88.6% top-1**（在 JFT-300M 预训练后微调），成为当时最佳。

3. **ViT 的计算效率优势**：到达到相同的性能，ViT 比 CNN 需要显著更少的计算量（TPU 天数更少），因为 Transformer 的矩阵计算对硬件高度优化。

4. **注意力可视化**：ViT 的 [class] token 在最后一层的注意力图中显示出对图像语义区域的关注——即使没有明确的空间先验，模型也能学会全局的空间关系。

5. **中等数据集的预训练仍然可行**：ViT-L/16 在 ImageNet-21K 预训练后，在多个下游任务上已经超越 BiT-L（ResNet-based），说明大规模 ImageNet-21K 已足够。

## 四、局限性与后续影响

**局限性：**
- **小数据下不如 CNN**：缺少平移等变性和局部性的归纳偏置，数据不够时无法学到这些结构
- **全局注意力的计算瓶颈**：patch 数量 N 的平方复杂度（$O(N^2)$），高分辨率图像成本极高
- **位置编码非局部**：绝对位置编码无法有效处理不同于训练分辨率的输入（朴素插值效果有限）
- **缺少层级多尺度结构**：与 CNN 相比缺少特征金字塔，对密集预测任务不利

**后续影响：**
- 直接开启了 Transformer 在 CV 的时代，催生了图像分割（SETR）、检测（DETR）、视频理解（TimeSformer）等
- 启发了 DeiT（数据高效的 ViT 训练策略）、Swin Transformer（层次 Transformer）、CvT（卷积 Transformer）等一系列改进
- 视觉基础模型的 backbone 几乎全部转向 ViT架构

## 五、VLA/机器人研究中的角色

ViT 对 VLA 的影响是**直接且决定性的**：

- **OpenVLA** 使用 DINOv2（ViT-based）+ SigLIP（ViT-based）**双视觉编码器**
- **π0** 的 VLM 组件 PaliGemma 使用 ViT 作为视觉 backbone
- **RT-2** 的 PaLI-X 视觉骨干使用 ViT-G（参数量超过 2B 的超大 ViT）
- **DiT**（Diffusion Transformer）本质上是 ViT 的生成变体——将图像 patch 作为扩散状态
- 几乎所有现代 VLA 的视觉理解部分都基于 ViT 或其变体

## 六、对你的启示

1. **归纳偏置 vs 数据规模**：ViT 的核心启示是"如果数据够大，模型可以自己学会结构，不需要手动设计偏置"。当你有大量数据时，选择更通用的架构比精心设计的专用架构更好。

2. **跨界移植的成功密码**：ViT 的成功是"NLP 技术迁移到 CV"的典型案例，其成功条件是：找到两种模态的共同最小单元（patch ↔ word）、用最简单的方式做迁移、依赖大规模数据。

3. **简洁但不简单**：ViT 的代码比 ResNet 更短（约 100 行），但背后蕴含了对 Transformer 和视觉问题的深刻理解。好的算法往往在实现上简洁但在思想上深刻。

4. **承认局限性的价值**：ViT 明确指出了小数据下的不足，为后续改进（DeiT、Swin、DINO 等）指明了方向。诚实地说出局限性有时比展示强项更有价值。

## PDF

[[An Image is Worth 16x16 Words.pdf]]
