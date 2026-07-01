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
paper_venue: "ICCV 2021 (Best Paper, Marr Prize)"
paper_citations: "~25,000+"
paper_url: "https://arxiv.org/abs/2103.14030"
github: "https://github.com/microsoft/Swin-Transformer"
---

# Swin Transformer

**Swin Transformer: Hierarchical Vision Transformer using Shifted Windows**
*Microsoft Research Asia | ICCV 2021 Best Paper (Marr Prize) | arXiv: 2103.14030*

> ICCV 2021 最佳论文。解决了 [[ViT|ViT]] 的两个核心问题——高分辨率图像 O(N²) 计算成本和缺少多尺度特征。核心创新是移位窗口自注意力（Shifted Window Attention）：在局部窗口内计算自注意力（线性复杂度 O(N·M²)），并通过在相邻层间交替移动窗口实现跨窗口信息交互。Swin 的层次化设计与 CNN 的特征金字塔完全兼容，使其在目标检测和语义分割等密集预测任务上表现卓越。

---

## 一、Background / Core Idea

### 1.1 ViT 的两个核心局限

[[ViT|ViT]] 证明了纯 Transformer 可以在图像识别上超越 CNN，但存在两个根本问题（原论文 Figure 1, Section 1）：

**问题 1：计算复杂度随分辨率平方增长**

ViT 的全局 Self-Attention 复杂度为 O(N²·d)，其中 N = HW/P² 为 patches 数量。对于 224×224 图像（patch size 16：N=196），计算尚可接受。但对于密集预测任务（如语义分割），输入分辨率常为 1024×1024（N≈4096），计算量增加约 (4096/196)² ≈ 436 倍——完全不可行。

**问题 2：缺少多尺度特征**

ViT 在整个网络中保持单一空间分辨率（没有下采样），输出的是单尺度特征图。这对于需要特征金字塔的网络（FPN、U-Net）来说是个障碍——目标检测和语义分割等密集预测任务依赖多尺度特征来检测不同大小的目标。

### 1.2 核心洞察：局部窗口 + 层级特征

Swin Transformer 的核心洞察可以概括为：

1. **自注意力限制在局部窗口内**：将特征图分割为不重叠的 M×M 大小窗口（通常 M=7），每个窗口内独立计算自注意力。复杂度降至 O(N·M²)——当 M 固定时，复杂度与图像大小呈**线性关系**
2. **移位窗口实现跨窗口通信**：通过交替使用常规窗口分区和偏移窗口分区，让不同窗口的信息在层间传递
3. **层级下采样构建多尺度特征**：通过 Patch Merging 逐步下采样（类似 CNN 的 pooling），构建从 4x 到 32x 的 4 级特征金字塔

### 1.3 与以前工作的区别

此前也有工作在局部窗口内做自注意力（如 Stand-Alone Self-Attention），但使用的是**滑动窗口**——每个像素有唯一的 key set，导致实现复杂且延迟高。Swin 的**不重叠窗口 + 移位**设计在硬件实现上更高效，因为窗口内的所有 query patches 共享相同的 key set。

此外，与一些直接用 Transformer 做检测的工作不同，Swin 设计的是通用 backbone，可以无缝替换现有 CNN 检测器中的 ResNet 等骨干。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 层次化架构概览

Swin Transformer 将图像处理分为 4 个 Stage，逐步下采样，构建特征金字塔（原论文 Figure 3a）：

| Stage | 操作 | 输入分辨率 | 输出分辨率 | 说明 |
|-------|------|-----------|-----------|------|
| Stage 1 | Patch Partition → Linear Embedding → Swin Blocks | H×W×3 | H/4 × W/4 × C | 4×4 patch, 线性投影到 C(96) 维 |
| Stage 2 | Patch Merging → Swin Blocks | H/4 × W/4 × C | H/8 × W/8 × 2C | 2×2 patch 合并, 维度翻倍 |
| Stage 3 | Patch Merging → Swin Blocks | H/8 × W/8 × 2C | H/16 × W/16 × 4C | 最深的 stage (6 个 block) |
| Stage 4 | Patch Merging → Swin Blocks | H/16 × W/16 × 4C | H/32 × W/32 × 8C | 最终特征 |

Swin-T 在各 stage 的 block 分布为 {2, 2, 6, 2}，Swin-S 为 {2, 2, 18, 2}。

**Patch Merging**：类似 CNN 的 pooling。将 2×2 邻域的 patches（每个维度 C）拼接 → 得到 4C 维度 → 通过线性层压缩为 2C。这样空间分辨率减半，通道数加倍——与 CNN 的下采样方式完全对应。

### 2.2 Shifted Window Multi-Head Self-Attention (SW-MSA)

这是 Swin 最核心的创新。在相邻两个 Transformer block 之间交替使用两种注意力模式（原论文 Figure 2）：

**Block 序列**：W-MSA → SW-MSA → W-MSA → SW-MSA → ...

**W-MSA (Window-based MSA)**：
- 将特征图均匀分割为不重叠的 M×M 窗口（M=7）
- 在每个窗口内独立计算标准 Multi-Head Self-Attention
- 复杂度：O(N·M²·d) 而非 O(N²·d)

**SW-MSA (Shifted Window MSA)**：
- 窗口分区偏移 ⌊M/2⌋ = 3 个 patches
- 原本在不同窗口的 patches 现在在同一个窗口中，实现跨窗口信息交互
- 需要处理窗口边界不整除的问题——通过 cyclic-shift + masked attention 解决

### 2.3 计算复杂度对比

原论文给出了完整的复杂度推导（Eq. 1-2）：

全局 MSA 的复杂度（对于 h×w 个 patches, C 维 hidden dim）：

Ω(MSA) = 4hwC² + 2(hw)²C

窗口 MSA 的复杂度（M×M 窗口）：

Ω(W-MSA) = 4hwC² + 2M²hwC

关键差异：当 hw 增大时，前者为 O((hw)²)，后者为 O(hw)。对于 FPN/ U-Net 的高分辨率特征图，这种差异是决定性的。

### 2.4 高效批处理实现（Cyclic Shift + Masked Attention）

移位窗口带来的问题是：窗口数量从 ⌈h/M⌉×⌈w/M⌉ 增加到 (⌈h/M⌉+1)×(⌈w/M⌉+1)，许多窗口大小小于 M×M。朴素填充方案会显著增加计算量（例如 2×2 → 3×3，增加 2.25 倍）。

Swin 的解决方案（原论文 Figure 4）：
1. 对特征图进行**cyclic shift**（向左上方向循环移位）
2. 将移位后的特征图分割为常规 M×M 窗口
3. 在注意力计算中应用**masking 机制**：只允许原本在同一个窗口中（移位前）的 patches 之间计算注意力
4. 计算完成后通过**reverse cyclic shift**恢复原始位置

这样，批处理窗口数与常规分区相同，同时实现了跨窗口通信。Table 5 展示了这种方案的实际延迟优势。

### 2.5 相对位置偏置

Swin 使用**相对位置偏置**（Relative Position Bias）替代 ViT 的绝对位置编码：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

其中 B ∈ ℝ^{M²×M²} 是相对位置偏置矩阵。由于每个轴上的相对位置范围是 [-M+1, M-1]，作者参数化了一个更小的偏置矩阵 B̂ ∈ ℝ^{(2M-1)×(2M-1)}，从中取值。

**为什么相对位置偏置比绝对位置编码更好？**

原论文的消融实验（Table 4）清楚地展示了这一点：

| 位置编码 | ImageNet Top-1 | COCO box AP | ADE20K mIoU |
|----------|---------------|-------------|-------------|
| 无位置编码 | 80.1 | 49.2 | 43.8 |
| 绝对位置编码 | 80.5 | 49.0 | 43.2 |
| 绝对+相对 | 81.3 | 50.2 | 44.0 |
| **相对位置偏置** | **81.3** | **50.5** | **46.1** |

相对位置偏置在三个任务上均最佳，而绝对位置编码（ViT 方式）实际上比无位置编码在 COCO/ADE20K 上还差——**绝对位置信息对密集预测任务可能有害**。

相对位置偏置的一个重要特性是提供了**平移等变性**：无论目标出现在图像哪个位置，patch 之间的相对位置关系不变。这是 CNN 的重要归纳偏置，而通过相对位置偏置，Transformer 也能获得这一特性。

### 2.6 架构变体

Swin 定义了四种模型尺寸，在参数量和 FLOPs 上与标准的 Transformer 和 CNN 对应：

| 模型 | 参数量 | FLOPs (224²) | ImageNet Top-1 | 对应模型 |
|------|--------|-------------|---------------|---------|
| Swin-T | 28M | 4.5G | 81.3% | ResNet-50 |
| Swin-S | 50M | 8.7G | 83.0% | ResNet-101 |
| Swin-B | 88M | 15.4G | 83.3%/86.4% (IN-22K) | ViT-B/DeiT-B |
| Swin-L | 197M | 34.5G | 87.3% (IN-22K) | — |

Swin 的 FLOPs 显著低于 ViT（例如 ViT-B 需要 55.4G FLOPs，Swin-B 仅 15.4-47.0G），主要是因为窗口注意力比全局注意力更高效。

### 2.7 训练配置

- **ImageNet-1K 训练**：AdamW 优化器，学习率 0.001，weight decay 0.05，300 epochs，使用 DeiT 的增强策略（RandAugment, Mixup, CutMix, Random Erasing 等）
- **ImageNet-22K 预训练**：AdamW，学习率 0.001，weight decay 0.01，90 epochs。然后在 ImageNet-1K 上微调 30 个 epochs
- **目标检测**：基于 mmdetection 框架，搭配 Cascade Mask R-CNN, ATSS, RepPointsV2, Sparse R-CNN 等检测器

---

## 三、Experiments and Key Findings

### 3.1 ImageNet-1K 分类（从零训练）

| 模型 | 参数量 | FLOPs | Top-1 | Top-5 |
|------|--------|-------|-------|-------|
| RegNetY-8G | 39M | 8.0G | 81.7 | — |
| EfficientNet-B5 | 30M | 9.9G | 83.6 | — |
| DeiT-S | 22M | 4.6G | 79.8 | — |
| **Swin-T** | **28M** | **4.5G** | **81.3** | **95.6** |
| DeiT-B (384²) | 86M | 55.4G | 83.1 | — |
| **Swin-B (384²)** | **88M** | **47.0G** | **84.5** | **97.0** |
| ViT-B/16 (384²) | 86M | 55.4G | 77.9 | — |

**Swin-T 比 DeiT-S 高 1.5%**，Swin-B 比 DeiT-B 高 1.4-1.5%，且使用更少的 FLOPs。

### 3.2 ImageNet 分类（ImageNet-22K 预训练）

| 模型 | 参数量 | FLOPs | Top-1 |
|------|--------|-------|-------|
| ViT-B/16 (384²) | 86M | 55.4G | 84.0 |
| **Swin-B (384²)** | **88M** | **47.0G** | **86.4** |
| ViT-L/16 (384²) | 307M | 190.9G | 85.2 |
| **Swin-L (384²)** | **197M** | **103.9G** | **87.3** |

Swin-B 超过 ViT-L（86.4 vs 85.2），且 FLOPs 仅 ViT-L 的 1/4。

### 3.3 COCO 目标检测（Swin 的核心优势领域）

**Cascade Mask R-CNN 框架**（原论文 Table 2b）：

| 骨干 | box AP | mask AP | 参数量 | FLOPs |
|------|--------|---------|--------|-------|
| ResNet-50 | 46.3 | 40.1 | 82M | 739G |
| DeiT-S + 解卷积 | 48.0 | 41.4 | 80M | 889G |
| **Swin-T** | **50.5** | **43.7** | **86M** | **745G** |
| ResNeXt101-64 | 48.3 | 41.7 | 140M | 972G |
| **Swin-B** | **51.9** | **45.0** | **145M** | **982G** |

Swin-T 比 ResNet-50 高 4.2 box AP，Swin-B 超过 ResNeXt101-64。

**系统级对比**（原论文 Table 2c）：Swin-L (HTC++ 增强) 达到 58.7 box AP / 51.1 mask AP，全面超过所有此前 SOTA（包括 EfficientDet-D7 的 55.1 box AP）。

### 3.4 ADE20K 语义分割

| 骨干 | mIoU | 参数量 | FLOPs |
|------|------|--------|-------|
| DeiT-S + 解卷积 | 44.0 | — | — |
| ResNet-101 | 44.9 | 86M | 1029G |
| ResNeSt-101 | 46.9 | 66M | 1051G |
| SETR (T-Large) | 50.3 | 308M | — |
| **Swin-S** | **49.3** | **81M** | **1038G** |
| **Swin-L** | **53.5** | **235M** | **2468G** |

Swin-L 在 ADE20K 验证集上达到 53.5 mIoU，在测试集上达到 54.9 mIoU——均为当时 SOTA。

### 3.5 消融实验（Table 4）

| 变体 | ImageNet Top-1 | COCO box AP | ADE20K mIoU |
|------|---------------|-------------|-------------|
| 无移位窗口 | 80.2 | 47.7 | 43.3 |
| **移位窗口** | **81.3 (+1.1)** | **50.5 (+2.8)** | **46.1 (+2.8)** |

移位窗口在三个任务上都带来显著提升，在密集预测任务上效果更为突出（+2.8 box AP, +2.8 mIoU）。

### 3.6 窗口大小实验（Section 4.4）

作者测试了 M=7 和 M=14 两种窗口大小：
- M=14 在分类上略好（81.3% vs 81.2%），但在检测上略差（50.5 vs 50.2 box AP）
- 将移位窗口中的移位距离从 M/2 改为 M/2+1（即 3→4）对性能无影响
- M=7 在 COCO 上训练速度和 GPU 内存使用上更高效

结论：M=7 是最优选择——14×14 窗口的注意力距离已经足够覆盖大多数目标，更大的窗口收益递减且增加计算成本。

---

## 四、Limitations and Challenges

### 4.1 感受野天然受限

窗口自注意力的感受野天然限制在 M×M 窗口内（Swin 为 7×7）。虽然通过窗口移位，深层 block 之间可以实现更大范围的交互，但单层操作无法直接看到全局——这与 ViT 的全局注意力有本质区别。

### 4.2 架构和实现复杂度远高于 ViT

ViT 的核心实现约 100 行 PyTorch，而 Swin 的窗口分割、cyclic shift、masked attention 等操作需要数百行代码。Efficient batch computation 的 masking 策略正确实现也有较高门槛。这增加了调试难度和实施成本。

### 4.3 对输入分辨率的整数倍约束

Swin 假设图像尺寸是窗口大小（7x7）的整数倍。非整数倍需要通过 padding 解决，可能引入伪影。此外，Swin 的下采样操作（Patch Merging）也假设 h,w 为 2 的整数次幂。

### 4.4 全局任务可能不如 ViT

在需要全局理解的某些任务上（如图像检索、图级别的细粒度分类），Swin 的局部注意力可能不如 ViT 的全局注意力。不过原论文实验显示 Swin 在绝大多数任务上都优于或接近 ViT。

---

## 五、Relationship with Subsequent Work / Impact on the Field

Swin Transformer 因其对密集预测任务的适应性，在 2021-2023 年期间迅速成为检测/分割的标准 backbone：

| 方向 | 模型 | 继承关系 |
|------|------|----------|
| 检测 backbone | Swin + HTC++ (COCO SOTA) | Swin 成为 COCO 检测的默认 backbone 之一 |
| 分割 backbone | Swin + UperNet (ADE20K SOTA) | Swin 在语义分割中替代 ResNet |
| 视频 | Video-Swin | 将窗口注意力扩展到 3D 时空域 |
| 3D 视觉 | Swin3D | 将层次化窗口注意力用于点云处理 |
| MLP 架构 | AS-MLP | 验证了移位窗口概念在纯 MLP 架构中同样有效 |
| 高效注意力 | CSWin, Focal Transformer, MaxViT | 进一步改进窗口注意力的效率 |

**Swin 在 VLA 中的角色**：

Swin 在 VLA 中的直接使用不如 [[ViT|ViT]]（SigLIP/DINOv2 都基于 ViT 而非 Swin）广泛，但具有重要的参考价值：

1. **层次化多尺度特征对机器人物体检测至关重要**：机器人需要同时感知全局场景（房间布局）和局部细节（物体位姿），Swin 的特征金字塔天然适合
2. **窗口注意力为多相机高分辨率输入提供效率方案**：在机器人视觉中，多个高分辨率相机输入的处理要求高效架构，Swin 的 O(N) 复杂度优于 ViT 的 O(N²)
3. **Swin 的层次化设计对于需要高分辨率的操纵任务更有实用价值**：如抓取检测、精细零件装配中的细节理解
4. **动作分割/检测中的 backbone**：部分机器人感知管线（如 pick-and-place 检测）使用 Swin 作为 backbone

---

## 六、Implications for You / Hardware Compatibility

- ✅ **Swin-T (28M) 在单 GPU 上运行高效**：4.5G FLOPs 使其在消费级 GPU 上也能快速推理，是资源受限场景中的优秀选择
- ✅ **理解窗口注意力对处理高分辨率多相机 VLA 输入至关重要**：Swin 的线性复杂度设计比 ViT 更适合处理多帧高分辨率图像
- ⚠️ **Swin 的实现复杂度是实际部署的阻力**：相比 ViT 简洁的 ~100 行代码，Swin 的 cyclic shift + masking 增加了实现难度。Hugging Face 的 `transformers` 已提供 Swin 实现，建议直接使用
- ❌ **如果你的任务属性"全局理解优先"**（如图像-文本检索、整个场景分类），ViT 可能更适合——Swin 的局部窗口设计对这类任务不一定是优势
- ✅ **Swin 证明了"适量归纳偏置"的价值**：在 ViT 完全消除偏置和 CNN 强偏置之间，Swin 找到了中间点——局部性（窗口）+ 层次化（下采样）+ 平移等变性（相对位置偏置）。对 VLA 架构设计同样有启示意义
- ⚠️ **不适合动态分辨率输入**：Swin 的窗口划分和 Patch Merging 假设输入分辨率是 32 的倍数，对于非标准分辨率的处理不如 ViT 灵活

## PDF

[[Swin Transformer 原文.pdf]]
