---
tags:
  - 论文
  - CNN
  - 多尺度
  - 高效架构
created: 2026-06-30
paper_title: "Going Deeper with Convolutions"
paper_authors: "Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich"
paper_year: 2014
paper_venue: "CVPR 2015"
paper_citations: "~55,000+"
paper_url: "https://arxiv.org/abs/1409.4842"
---

# GoogLeNet (Inception v1)

**Going Deeper with Convolutions**
*Google | CVPR 2015 | arXiv 1409.4842*

> ILSVRC 2014 分类冠军（6.7% top-5 error）。提出 Inception 模块——在同一层内并行使用 1×1, 3×3, 5×5 卷积和 3×3 max pooling，网络自动学习哪个尺度最合适。1×1 卷积降维大幅减少计算量（从 AlexNet 的 60M 降到 5M 参数）。"高效+深度"路线的开创者。

---

## 一、研究背景与动机

随着深度 CNN 的流行，两个关键问题凸显出来：
1. **如何更高效地增加网络的深度和宽度？** 简单地叠加更多卷积层会导致参数量和计算量爆炸式增长
2. **目标在图像中尺度变化极大**——同一张图像中，物体可能占据大部分区域也可能只有几个像素。固定大小的卷积核无法适应不同尺度的特征

GoogLeNet 的核心洞察是：**与其在每一层固定使用一种卷积核大小，不如在每一层同时使用多种尺度的卷积核，让网络自行学习在不同位置使用哪种子特征。** 另外，借助 1×1 卷积降低通道数，可以在不牺牲表达能力的前提下大幅减少计算量。

## 二、核心方法

### 2.1 Inception 模块

Inception 模块的核心结构（Naive 版本 → 优化版本）：

**Naive Inception**：
```
输入特征图
├── 1×1 卷积
├── 3×3 卷积 (padding=1)
├── 5×5 卷积 (padding=2)
└── 3×3 MaxPool (padding=1)
        → 输出通道拼接
```

**优化 Inception（加入 1×1 降维）**：
```
输入特征图
├── 1×1 卷积 (降维) → 3×3 卷积
├── 1×1 卷积 (降维) → 5×5 卷积
├── 3×3 MaxPool → 1×1 卷积 (降维)
└── 1×1 卷积
        → 输出通道拼接
```

1×1 卷积在该模块中有两个作用：
1. **降维**：在昂贵的 3×3/5×5 卷积前减少通道数，大幅降低计算量
2. **增加非线性**：1×1 卷积后接 ReLU，增加网络的表达能力

### 2.2 整体架构

GoogLeNet 共有 22 层（含池化、不含辅助分类器），参数量仅 5M：

```
输入: 224 × 224 × 3
├── Conv 7×7 (stride 2) → MaxPool (3×3, stride 2)
├── Conv 1×1 → Conv 3×3 → MaxPool (3×3, stride 2)
├── Inception (3a) → Inception (3b) → MaxPool (3×3, stride 2)
├── Inception (4a, 4b, 4c, 4d, 4e) [中段有辅助分类器] → MaxPool
├── Inception (5a, 5b)
├── Avg Pool (7×7)
├── Dropout (40%)
└── Softmax (1000 classes)
```

### 2.3 辅助分类器

GoogLeNet 在网络的中间层（Inception 4a 和 4d 之后）添加了两个辅助分类器：
- 每个辅助分类器：Avg Pool 5×5 (stride 3) → Conv 1×1 (128) → FC 1024 → FC 1000 (softmax)
- 训练时，辅助损失以 0.3 的权重加到总损失中
- 推理时，辅助分类器被移除

设计动机：缓解深层网络的梯度消失问题（类似 ResNet 的 skip connection 的提前版本，但更粗粒度）。

## 三、关键实验与发现

### 3.1 ILSVRC 2014 结果

| 模型 | Top-5 Error (分类) | Top-5 Error (定位) | 参数量 | 计算量 |
|------|-------------------|-------------------|--------|--------|
| **GoogLeNet** | **6.67%** | **43.5%** (7 models) | 5M | 1.5 GFLOPS |
| VGG | 7.3% | - | 138M | ~15 GFLOPS |
| AlexNet | 15.3% | - | 60M | 0.7 GFLOPS |

> GoogLeNet 以分类第一（6.67% top-5）、定位第一的成绩赢得 ILSVRC 2014。

### 3.2 关键发现

1. **1×1 卷积降维极其高效**：从 256×256 输入降到 64 通道 → 3×3 卷积，计算量减少 4 倍，参数减少 16 倍
2. **多尺度特征融合有效**：不同分支捕捉不同尺度的特征，拼接后信息更丰富
3. **参数量远小于 VGG 但性能更好**：GoogLeNet 仅 5M vs VGG 138M（27 倍差距），但 top-5 error 低 0.6%
4. **辅助分类器作用有限但有助于收敛**：消融实验显示辅助分类器提升约 0.5%
5. **较深网络（22 层）训练稳定**：得益于精心设计的 Inception 模块和辅助分类器

## 四、局限性与后续影响

**局限**：
1. **架构设计复杂**：每个 Inception 模块的超参数（分支通道数、核大小比）需要精细调整
2. **Inception 模块不够通用**：难以直接迁移到其他任务（需要重新调整通道分配）
3. **辅助分类器的必要性存疑**：后续版本（Inception v2/v3）改进了设计，v4 结合了 ResNet，但 ResNet 的 skip connection 最终被证明是更好的解决方案
4. **没有解决梯度消失的根本问题**：辅助分类器只是缓解，ResNet 的 skip connection 才是根本解

**后续影响**：
- **Inception v2 (2015)**：加入 BatchNorm，5×5 改为两个 3×3
- **Inception v3 (2016)**：分解 7×7 为 7×1 + 1×7，Label Smoothing
- **Inception v4 + Inception-ResNet (2017)**：结合 ResNet 的 skip connection
- **1×1 卷积成为标准操作**：1×1 卷积降维被 ResNet bottleneck block、SENet、EfficientNet 等广泛采用
- **多分支设计思想**：影响了 ResNeXt（组卷积）、NASNet（搜索架构）等

## 五、VLA/机器人研究中的角色

GoogLeNet 虽然不是 VLA 直接使用的视觉编码器，但其核心思想在 VLA 中延续：

1. **1×1 卷积降维的普遍应用**：VLA 中视觉编码器与 LLM 之间的投影层（Projection Layer）本质上是 1×1 卷积 + 线性变换——这与 GoogLeNet 的 1×1 降维思想一致
2. **多尺度特征融合**：现代视觉编码器（如 FPN、DINOv2 的多尺度特征、SigLIP 的多分辨率）继承了 GoogLeNet 的多尺度思想
3. **高效架构设计哲学**：GoogLeNet 证明"用更少参数做更多事"——这在 VLA 中尤为重要（VLA 模型通常数十亿参数，每一层的效率优化都直接影响训练时延）
4. **辅助损失训练策略**：多任务/多损失训练（如 π0 的 VLM loss + action loss + contrastive loss）的思想可追溯至辅助分类器的多损失架构

## 六、对你的启示

1. **1×1 卷积 / 线性投影层的关键角色**：在 VLA 中，视觉编码器和语言模型之间的投影层（通常是简单的线性映射或 MLP）决定了视觉特征能否被 LLM 有效利用——这本质上是 GoogLeNet 1×1 降维思想的延续
2. **高效设计比极致准确率更重要**：GoogLeNet 以 AlexNet 1/12 的参数实现了更好的性能。在 VLA 中，模型效率直接影响能否在真实机器人上部署（推理时延要求 <50ms）
3. **Inception 的"多路径融合"思想与现代 VLA 架构**：
   - 多模态输入（视觉、语言、动作）的融合方式可视为 Inception 的延续——不同模态在不同路径处理，在高层拼接
   - π0 的 Dual-Expert 架构（VLM + Action Expert）使用了并行多路径的思路
4. **辅助损失的有效使用**：如果 VLA 模型在中间层训练不稳定（如深层 Transformer 训练困难），可以尝试类似辅助分类器的多损失策略

## PDF

[[1409.4842_GoogLeNet_Going_Deeper_with_Convolutions.pdf]]
