---
tags:
  - 论文
  - CNN
  - 残差学习
  - 深度网络
  - ResNet
  - CVPR最佳论文
created: 2026-06-30
paper_title: "Deep Residual Learning for Image Recognition"
paper_authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
paper_year: 2015
paper_venue: "CVPR 2016 (Best Paper)"
paper_citations: "~210,000+"
paper_url: "https://arxiv.org/abs/1512.03385"
github: "https://github.com/KaimingHe/deep-residual-networks"
---

# ResNet

**Deep Residual Learning for Image Recognition**
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | Microsoft Research | CVPR 2016 (Best Paper) | arXiv: 1512.03385*

> 计算机视觉史上引用量最高的论文。残差学习（Residual Learning）通过恒等捷径连接（identity shortcut connection）让训练 152 层网络变得容易，系统地解决了深层网络的退化问题。核心公式不过 $y = F(x) + x$ 这一行，却改变了整个深度学习的方向——包括 Transformer、BERT、GPT 在内的所有现代架构都依赖残差连接来训练深层模型。

---

## 一、研究背景与动机

### 1.1 退化问题：深度学习的一大反直觉谜题

在 ResNet 出现之前，深度学习社区面对一个令人困惑的问题：**为什么更深的网络反而更差？**

VGG（19 层）和 GoogLeNet（22 层）已经证明较深网络效果更好。但当研究者尝试堆叠更多层时（如 56 层 vs 20 层），观察到训练误差和测试误差**同步上升**。这不是过拟合（训练误差没下降），也不是梯度消失（使用了 Batch Normalization 后梯度信号的方差已经稳定）。

这被称为 **退化问题（Degradation Problem）**。论文使用图 1 直观展示了这一现象：在 CIFAR-10 上，56 层 plain 网络的训练 error 和 test error 都**高于** 20 层的 plain 网络，而且差距出现在训练早期并持续存在。

### 1.2 退化的本质：优化困难而非过拟合

论文的关键洞察是：

> "If the added layers can be constructed as identity mappings, a deeper model should have training error no greater than its shallower counterpart."

理论上，如果新增加的层可以学习恒等映射（identity mapping），那么深层网络至少应该和浅层网络一样好。但实践中，堆叠的非线性层很难学习恒等映射——优化的难度在于让层输出接近其输入，而非学习从零开始的映射。

作者提出：**如果让网络学习残差函数 $F(x) = H(x) - x$（其中 $H(x)$ 是目标映射），而不是直接学习 $H(x)$，那么当最优解就是恒等映射时，网络只需将残差推向 0，这比学习恒等映射容易得多。**

### 1.3 历史同期工作

ResNet 不是第一个引入捷径连接的工作：
- **Highway Networks** (Srivastava et al., 2015) 引入了带门控（gating）的捷径连接，但门控机制引入了额外参数和计算
- ResNet 的突破在于使用了**无参数的恒等捷径**（identity shortcut），比 Highway 更简洁、更有效

论文实验表明，Highway Networks 在 19 层时 error 为 7.54%，比 ResNet-20 的 8.75% 好，但当增加到 32 层时，Highway 的 error 退化到 8.80%，而 ResNet-56 继续降低到 6.97%。这证明了恒等捷径的优越性。

---

## 二、方法/架构/技术贡献

### 2.1 残差学习的数学形式

残差块的定义简洁到只有一行公式：

$$y = \mathcal{F}(x, \{W_i\}) + x$$

其中：
- $x$ 和 $y$ 是层的输入和输出向量
- $\mathcal{F}(x, \{W_i\})$ 表示需要学习的残差映射
- 对于有两个权重层的残差块：$\mathcal{F} = W_2 \sigma(W_1 x)$，其中 $\sigma$ 是 ReLU
- 加法操作 $+ x$ 是逐元素加法（element-wise addition）
- 加法后通过第二次非线性（$\sigma(y)$）

**关键属性**：当 $x$ 和 $\mathcal{F}$ 的维度不一致时（如下采样时），需要一个线性投影 $W_s$ 来匹配维度：

$$y = \mathcal{F}(x, \{W_i\}) + W_s x$$

### 2.2 恒等捷径的连接方式

ResNet 对比了三种 shortcut 选项：

| 选项 | 升维方式 | 参数量 | Top-1 Error (ImageNet, 34 层) |
|------|---------|--------|------------------------------|
| **A** | 零填充（zero-padding shortcut） | **0 额外参数** | 25.03% |
| **B** | 升维时使用 projection shortcut（1×1 conv） | 极少额外参数 | 24.52% |
| **C** | 所有 shortcut 都使用 projection | 增加较多参数 | 24.19% |

**结论**：选项 A（零填充）几乎和无参数的恒等捷径一样好（$25.03%$ vs $24.19%$），但额外参数近乎为 0。论文最终在 ResNet-50/101/152 中使用**选项 B**（升维时 projection，其他保持 identity）。

### 2.3 Bottleneck 设计：为深层网络量身定制

当网络深度超过 50 层时，原始的两层 3×3 残差块计算量过大。论文提出了 **bottleneck 残差块**：

```
两层残差块（ResNet-18/34）:
输入 (256-d)
  → 3×3, 256 → ReLU
  → 3×3, 256 → ReLU
  → + 输入 (identity)
  → ReLU

瓶颈残差块（ResNet-50/101/152）:
输入 (256-d)
  → 1×1, 64 → ReLU              ← 降维：256 → 64（减少 4 倍）
  → 3×3, 64 → ReLU              ← 核心计算（3×3 在低维空间进行）
  → 1×1, 256 → ReLU             ← 升维：64 → 256（恢复通道数）
  → + 输入 (identity)
  → ReLU
```

**计算量对比**：
- 两层残差块（256-d → 3×3, 256 → 3×3, 256）：$256 \times 9 \times 256 + 256 \times 9 \times 256 \approx 1.18M$
- 瓶颈残差块（256-d → 1×1, 64 → 3×3, 64 → 1×1, 256）：$256 \times 1 \times 64 + 64 \times 9 \times 64 + 64 \times 1 \times 256 \approx 0.074M$

**计算量减少约 94%**，同时保持了感受野（3×3 卷积）和表达能力。

### 2.4 ImageNet 架构系列

| 模型 | 层数 | 参数 | FLOPs | 顶层描述 |
|------|------|------|-------|---------|
| ResNet-18 | 18 | 11M | 1.8G | 两层残差块（无瓶颈） |
| ResNet-34 | 34 | 22M | 3.6G | 两层残差块（无瓶颈） |
| **ResNet-50** | **50** | **26M** | **3.8G** | 瓶颈结构（3 层 block） |
| ResNet-101 | 101 | 45M | 7.6G | 瓶颈结构 |
| **ResNet-152** | **152** | **60M** | **11.3G** | **瓶颈结构（ILSVRC 2015 冠军）** |

**注意**：ResNet-152 (60M 参数, 11.3 GFLOPs) 在比 VGG-16 (138M 参数, 15.3 GFLOPs) **参数更少、计算更少**的情况下，取得了远超 VGG 的性能。

#### ResNet-50 的架构表

```
输入: 224 × 224 × 3
│
├── conv1: 7 × 7, 64, stride 2 → maxpool (3 × 3, stride 2)
│   → 输出: 56 × 56 × 64
│
├── conv2_x: [1×1, 64; 3×3, 64; 1×1, 256] × 3       ← 3 个 bottleneck block
│   → 输出: 56 × 56 × 256
│
├── conv3_x: [1×1, 128; 3×3, 128; 1×1, 512] × 4     ← 4 个 bottleneck block
│   → 输出: 28 × 28 × 512
│
├── conv4_x: [1×1, 256; 3×3, 256; 1×1, 1024] × 6    ← 6 个 bottleneck block
│   → 输出: 14 × 14 × 1024
│
├── conv5_x: [1×1, 512; 3×3, 512; 1×1, 2048] × 3    ← 3 个 bottleneck block
│   → 输出: 7 × 7 × 2048
│
├── Global Average Pooling (7 × 7)
├── FC-1000 (Softmax)
```

**降采样设计**：conv3_1、conv4_1、conv5_1 使用 stride=2 从 3×3 卷积进行降采样（同时 1×1 分支也使用 stride=2 的 projection 来匹配尺寸和通道）。

### 2.5 训练细节

| 超参数 | 值 |
|--------|------|
| 优化器 | SGD + Momentum（momentum=0.9） |
| Batch size | 256 |
| 初始学习率 | 0.1 |
| 学习率调度 | error 饱和时除以 10（约 60 万次迭代） |
| 权重衰减 | $10^{-4}$ |
| 初始化 | Xavier 初始化 + BN layer 初始化 |
| 数据增强 | 随机裁剪 224×224 + 水平翻转 + 色彩增强 + 多尺度缩放 [256, 480] |
| BN 层位置 | 每个卷积后、激活之前 |

---

## 三、实验与关键发现

### 3.1 ImageNet 主实验结果

#### 单模型结果（10-crop 测试）

| 模型 | Top-1 Error | Top-5 Error | 参数量 | 备注 |
|------|-------------|-------------|--------|------|
| VGG-16 | 28.07% | 9.33% | 138M | 引用 VGG 论文报告 |
| GoogLeNet | ~ | 9.15% | 5M | |
| PReLU-Net | 24.27% | 7.38% | ~ | |
| Plain-34 | 28.54% | 10.02% | 22M | 无残差 |
| **ResNet-34 A** | **25.03%** | **7.76%** | **22M** | Zero-padding shortcuts |
| **ResNet-34 B** | **24.52%** | **7.46%** | **22M** | 仅升维时 projection |
| **ResNet-34 C** | **24.19%** | **7.40%** | **22M** | 所有 shortcut projection |
| **ResNet-50** | **22.85%** | **6.71%** | **26M** | |
| **ResNet-101** | **21.75%** | **6.05%** | **45M** | |
| **ResNet-152** | **21.43%** | **5.71%** | **60M** | |

#### 集成结果

| 模型 | Top-5 Error（测试集） |
|------|---------------------|
| VGG (ILSVRC'14) | 7.32% |
| GoogLeNet (ILSVRC'14) | 6.66% |
| PReLU-Net | 4.94% |
| BN-Inception | 4.82% |
| **ResNet (ILSVRC'15)** | **3.57%** |

ResNet-152 的集成结果（6 个不同深度的模型）在 ImageNet 测试集上实现了 **3.57%** top-5 error，首次**超越人类表现**（约 5.1%）。

### 3.2 CIFAR-10 上的极深网络验证

为了验证残差学习的可扩展性，论文在 CIFAR-10 上训练了极深网络：

| 模型 | 层数 | 参数量 | CIFAR-10 Test Error |
|------|------|--------|-------------------|
| ResNet-20 | 20 | 0.27M | 8.75% |
| ResNet-32 | 32 | 0.46M | 7.51% |
| ResNet-44 | 44 | 0.66M | 7.17% |
| ResNet-56 | 56 | 0.85M | 6.97% |
| ResNet-110 | 110 | 1.7M | 6.43% |
| **ResNet-1202** | **1202** | **19.4M** | **7.93%** |

**关键发现**：
- ResNet-110 优于 ResNet-56（随着深度增加性能提升——退化问题被解决）
- ResNet-1202 的训练 error 低于 0.1%（无优化困难），但 test error 为 7.93%（高于 110 层的 6.43%）
- **1202 层的过拟合**是主要原因（19.4M 参数，但 CIFAR-10 只有 50K 训练图像）

### 3.3 残差函数的响应分析

论文通过分析每一层的输出标准差（std）来验证残差学习的内在机制：

- Plain 网络的层响应标准差随着深度增加而增加（高 std 意味着激活值互相抵消/冲突）
- ResNet 的残差函数响应**始终很小**（接近零），说明恒等映射提供了一个很好的预条件
- 更深的 ResNet 有更小的残差响应——随着网络变深，每层对信号的修改更少

这从经验上验证了论文的核心假设：残差函数确实比原始映射更容易学习。

### 3.4 PASCAL VOC / MS COCO 检测与分割结果

ResNet 在检测任务上同样碾压了 VGG-16 基线：

**PASCAL VOC 2007（使用 Faster R-CNN）**：
| 骨干网络 | mAP |
|---------|------|
| VGG-16 | 73.2% |
| **ResNet-101** | **76.4%** |

**MS COCO（Faster R-CNN）**：
| 骨干网络 | mAP@.5 | mAP@[.5, .95] |
|---------|--------|--------------|
| VGG-16 | 41.5% | 21.2% |
| **ResNet-101** | **48.4%** | **27.2%** |

ILSVRC & COCO 2015 五项全能冠军：分类、检测、定位（ImageNet），检测、分割（COCO）。

---

## 四、局限性与挑战

### 4.1 极深网络的收益递减

虽然 ResNet 成功训练了 152 层网络，但超过一定深度后收益递减：
- ResNet-152 vs ResNet-101：top-5 error 从 6.05% 降到 5.71%（仅 0.34% 的提升）
- 1202 层的 CIFAR 网络甚至出现了过拟合
- 深度增加带来的边际收益越来越小

### 4.2 身份映射 vs 预激活的开放问题

论文发表时的 ResNet v1 使用"后激活"（post-activation）设计：Conv → BN → ReLU → Add。后来的"Identity Mappings in Deep Residual Networks"（He et al., 2016, ECCV）发现**预激活**（Pre-Activation：BN → ReLU → Conv）效果更好，特别是对 1000+ 层的网络。这说明 ResNet v1 的设计并非最优。

### 4.3 没有从根本上解决计算效率问题

虽然 ResNet 比 VGG 效率更高（60M 参数 vs 138M），但相比于 [[DenseNet]] 的密集连接和 [[GoogLeNet]] 的 Inception 模块，ResNet 仍存在特征复用不充分的问题——每个残差块需要独立学习所有特征，而 [[DenseNet]] 通过拼接实现了更高效的参数利用。

### 4.4 对特定数据集的敏感性

- 在数据集较小的场景（如医学图像、机器人数据），ResNet 容易过拟合
- 预训练权重的领域差异可能导致迁移效果不佳
- 数据分布偏移（distribution shift）对 ResNet 特征的影响需要额外的 domain adaptation

---

## 五、对后续工作的影响

残差连接成为深度学习中最基本的构建块，其影响跨域视觉、NLP、多模态、强化学习。

| 来自该论文的思想 | 被继承/改进于 |
|----------------|-------------|
| **残差块（Residual Block）** | Wide ResNet（加宽而非加深）；ResNeXt（组卷积）；ResNeSt（split-attention）；Sandwich Block（MobileNetV2） |
| **瓶颈设计（Bottleneck）** | [[DenseNet]] 也使用瓶颈; EfficientNet; MobileNetV2 的倒置瓶颈（Inverted Bottleneck, 1×1→3×3→1×1） |
| **身份捷径连接（Identity Shortcut）** | Transformer 的 Pre-LN（每一层前后的残差连接）；[[Attention Is All You Need]] 的 Add & Norm；GPT/BERT 中的残差连接 |
| **退化问题的解决范式** | [[DenseNet]] 的密集连接；Transformer 的 LayerNorm + Residual 组合 |
| **极深网络的可行性证明** | ViT（Vision Transformer）的 24+ 层 encoder；GPT-3 的 96 层 decoder |
| **ImageNet 预训练骨干网络** | Faster R-CNN、Mask R-CNN、YOLO、SSD 均以 ResNet 为主要骨干；OpenVLA 使用 ResNet 视觉编码器；Diffusion Policy 使用 ResNet 骨干 |
| **_ResNet 在 VLA 中的直接应用_** | OpenVLA（使用类似 ResNet 的视觉编码器）；ACT（Action Chunking Transformer，基于 ResNet 骨干）；RT-1（基于 ResNet 的视觉 tokenizer）；Diffusion Policy（ResNet 特征提取） |

---

## 六、对你的启示与硬件兼容性

### 6.1 硬件兼容性评估

| 组件 | 操作 | VRAM 需求 | 你的 GPU Box |
|------|------|-----------|-------------|
| **ResNet-50 训练** | 完整训练（ImageNet） | ~8-10GB (batch=256) | ✅ **RTX 3090 24GB 极轻松；4070 Ti Super 16GB 完全胜任** |
| **ResNet-50 推理** | 单张图像推理 | ~1-2GB | ✅ 极其轻松 |
| **ResNet-152 训练** | 完整训练 | ~14-18GB | ✅ RTX 3090 可胜任；4070 Ti Super 略显紧张 |
| **ResNet-152 推理** | 单张图像推理 | ~3-5GB | ✅ 轻松 |
| **ResNet-101 微调** | 作为 VLA 编码器 | ~8GB（编码器部分） | ✅ 适合在 24GB 卡上作为 VLA 骨干进行实验 |
| **作为特征提取器** | 固定权重提取特征 | ~2GB | ✅ 可以在任何 GPU 上运行 |

### 6.2 核心启示

1. **残差学习思想可以推广到非视觉问题**：ResNet 的核心是"让模块学习增量（残差）而非从头开始"。在 VLA 中，如果你发现某个模块（如动作解码器、多模态融合层）难以直接训练，可以尝试加入残差连接——让模块只学习"增量"而非完整变换。

2. **简洁的设计哲学**：ResNet 的核心创新只是 $y = F(x) + x$ 这行公式。好的研究不在于复杂度，而在于直击问题本质。如果某个设计需要多篇论文才能解释清楚，它很可能不是好设计。

3. **梯度流动的视角**：设计深度网络时始终关注梯度能否顺利从顶部流到底部。每一层的设计都应让梯度流动尽可能简单直接。在 VLA 中，如果视觉编码器、LLM、动作解码器之间的梯度流动受阻，优先检查是否有足够的残差/捷径连接。

4. **选择视觉骨干的实用指南**：
   - **快速实验/原型验证**：ResNet-18（11M 参数，1.8 GFLOPs）— 在 4070 Ti Super 上 1.5ms/图
   - **VLA 标准选择**：ResNet-50（26M 参数，3.8 GFLOPs）— OpenVLA 等工作的默认选择
   - **需要更好视觉特征**：ResNet-101（45M 参数，7.6 GFLOPs）— 更深的特征层次
   - **极致性能（代价是速度）**：ResNet-152（60M 参数，11.3 GFLOPs）

5. **预训练权重是关键**：ResNet 的 ImageNet 预训练权重仍然是机器人视觉编码器的最佳起点——即使任务领域不同（如操作室内物体），ImageNet 先验知识也能加速收敛。

6. **预激活 vs 后激活**：如果你在 VLA 中设计自定义的编码器模块，建议使用**预激活结构**（BN/ LN → Activation → Conv/Linear），这已被证明比后激活（Conv → BN → Activation → Skip Add）更稳定。

---

## PDF

[[ResNet 原文.pdf]]
