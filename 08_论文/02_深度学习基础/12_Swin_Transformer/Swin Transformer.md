---
tags:
  - 论文
  - Transformer
  - 视觉模型
  - 层次化
  - 窗口注意力
created: 2026-06-30
paper_title: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
paper_authors: "Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo"
paper_year: 2021
paper_venue: "ICCV 2021"
paper_citations: "~25,000+"
paper_url: "https://arxiv.org/abs/2103.14030"
---

# Swin Transformer

**Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
*Microsoft Research | ICCV 2021 (Best Paper, Marr Prize) | arXiv: 2103.14030*

> ICCV 2021 最佳论文（Marr Prize）。解决了 ViT 的两个核心问题——高分辨率图像计算成本高、缺少多尺度特征。引入层次化结构和移位窗口自注意力（Shifted Window Attention），使 Transformer 可以和 CNN 一样优雅地处理高分辨率输入和密集预测任务。

---

## 一、研究背景与动机

ViT 将 Transformer 引入 CV 并展示了在大数据下的强大能力，但存在两个显著缺陷：

1. **计算复杂度**：ViT 的全局自注意力随 patch 数量 $N$ 呈平方增长 $O(N^2)$，处理高分辨率图像时成本急剧上升。对于 $224\times 224$ 图像（196 patches）尚可，但 $1000\times 1000$ 图像（约 3900 patches）几乎不可行。

2. **缺少多尺度特征**：ViT 在单一分辨率上操作，输出相同尺度的特征，缺乏 CNN 的特征金字塔。这对图像分类影响较小，但对目标检测和语义分割等密集预测任务至关重要。

Swin Transformer 的核心洞察：**将自注意力限制在局部窗口内，并在层之间移动窗口边界**，从而实现"局部计算 + 全局建模"——在 $7\times 7$ 窗口内计算注意力复杂度仅为 $O(49)$ 而非 $O(N^2)$，但通过窗口移位让不同窗口的信息在深层交互。

## 二、核心方法

**层次化架构：**

Swin 将图像处理分为 4 个 stage，逐步下采样，形成类似 CNN 的特征金字塔：

| Stage | 输入分辨率 | 窗口大小 | 输出分辨率 | 说明 |
|-------|-----------|---------|-----------|------|
| Stage 1 | H/4 × W/4 | / | H/4 × W/4 | 4×4 patch，线性嵌入 |
| Stage 2 | H/4 × W/4 | 7×7 | H/8 × W/8 | Patch Merging 下采样 |
| Stage 3 | H/8 × W/8 | 7×7 | H/16 × W/16 | 最深的 stage |
| Stage 4 | H/16 × W/16 | 7×7 | H/32 × W/32 | 最终特征 |

**Shifted Window Multi-Head Self-Attention (SW-MSA)：**

核心创新在两相邻 Transformer block 之间交替使用两种注意力模式：

- **W-MSA（Window MSA）**：将特征图分割为不重叠的 $M\times M$ 窗口（$M=7$），每个窗口内计算自注意力。复杂度 $O(N \times M^2)$ 而非 $O(N^2)$。
- **SW-MSA（Shifted Window MSA）**：窗口边界偏移 $\lfloor M/2 \rfloor$ 像素，使原本在不同窗口的 patch 能互相通信。

两个 block 的序列：$\text{W-MSA} \rightarrow \text{SW-MSA} \rightarrow \text{W-MSA} \rightarrow \text{SW-MSA} \rightarrow ...$

**相对位置偏置（Relative Position Bias）：**

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V
$$

其中 $B$ 是可学习的相对位置偏置矩阵（$\hat{B} \in \mathbb{R}^{(2M-1)\times (2M-1)}$），相比 ViT 的绝对位置编码更有效——提供了平移等变性的归纳偏置。

## 三、关键实验与发现

1. **ImageNet 分类 SOTA**：Swin-B 达到 84.5% top-1，Swin-L 达到 87.3%（ImageNet-21K 预训练），在纯分类任务上接近 ViT 水平。

2. **密集预测任务全面领先**：COCO 目标检测达到 58.7 box AP（+2.7 超过 SOTA），ADE20K 语义分割达到 53.5 mIoU（+3.2 超过 SOTA）。**Swin 在检测/分割上的优势远大于分类，证明了层次化多尺度特征对密集预测任务的关键作用。**

3. **计算效率**：Swin-T 计算量约 4.5 GFLOPs，低于 ViT-B 的 16.8 GFLOPs，但在类似参数规模下性能相当或更好。

4. **窗口大小的影响**：$M=7$ 是最优选择，更大的窗口收益递减且计算成本上升，$M=7$ 等效于感受野覆盖 $49\times 49$ 区域（已足够覆盖大多数目标）。

5. **移位窗口的有效性**：SW-MSA 比 W-MSA 提升约 1-2%，证明跨窗口信息交互对性能至关重要。

## 四、局限性与后续影响

**局限性：**
- **感受野受限**：窗口自注意力天然限制感受野为窗口大小，虽然通过移位窗口可以在深层实现更大交互，但不如全局自注意力直接
- **架构更复杂**：Swin 的窗口分割、移位、masking 实现比 ViT 复杂得多，工程实现成本高
- **对需要全局注意力的任务可能不如 ViT**：如图像检索、图级别分类的某些场景
- **动态分辨率处理困难**：窗口划分假设图像尺寸是窗口大小的整数倍，非整数倍需要 padding

**后续影响：**
- Swin 成为 ImageNet 分类、COCO 检测、ADE20K 分割的通用 backbone
- 启发了 CSWin、Focal Transformer、MaxViT 等更高效的窗口注意力设计
- Swin 的层次化多尺度思想被广泛应用，包括视频理解（Video-Swin）、3D 视觉（Swin3D）

## 五、VLA/机器人研究中的角色

Swin Transformer 在 VLA 中的影响虽不如 ViT 直接，但具有重要参考价值：

- **层次化多尺度特征**对需要同时感知全局场景（如房间布局）和局部细节（如物体位姿）的机器人控制至关重要
- 窗口注意力的**效率设计**启发了 VLA 中高分辨率多相机输入的视觉编码方案
- 在一些需要高分辨率输入的 VLA 系统中（如抓取检测、零件装配），Swin 的层次设计比 ViT 更实用
- 部分机器人的视觉感知管线直接使用 Swin 作为 backbone

## 六、对你的启示

1. **计算效率与模型能力的平衡**：Swin 证明了局部计算 + 全局交互的设计范式可以在保持 Transformer 强大能力的同时大幅降低计算复杂度。这对资源受限的机器人系统尤为重要。

2. **层级金字塔是实用的视觉设计**：无论 Transformer 还是 CNN，多尺度特征对于检测/分割等理解任务几乎是必要的。Swin 证明了 Transformer 同样可以从层次化设计中获益。

3. **CV 特定归纳偏置仍然重要**：虽然 ViT 证明了"无需偏置"的可行性，但 Swin 证明了"适量偏置"（局部性、层次化）可以在更少的计算和更小的数据下达到更好的密集任务性能。偏置不是敌人的，关键是用对地方。

4. **工程复杂度是设计选择**：Swin 的窗口操作实现复杂度远高于 ViT，这也限制了其在某些场景的采用。简单的实现有时比更好的性能更值得追求（尤其在快速迭代的研究环境中）。

## PDF

[[Swin Transformer.pdf]]
