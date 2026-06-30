---
tags:
  - 论文
  - CNN
  - 残差学习
  - 深度网络
created: 2026-06-30
paper_title: "Deep Residual Learning for Image Recognition"
paper_authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
paper_year: 2015
paper_venue: "CVPR 2016"
paper_citations: "~210,000+"
paper_url: "https://arxiv.org/abs/1512.03385"
---

# ResNet

**Deep Residual Learning for Image Recognition**
*Microsoft Research | CVPR 2016 (Best Paper) | arXiv: 1512.03385*

> 计算机视觉史上引用量最高的论文。提出残差学习（Residual Learning），通过 skip connection（恒等映射）让训练 152 层网络变得容易，解决了深层网络的退化问题。

---

## 一、研究背景与动机

在 ResNet 之前，深度学习面临一个反直觉的现象：**网络越深，反而越差**——不是过拟合导致的问题，而是优化困难。VGG 和 GoogLeNet 证明了更深网络能提升性能，但当层数继续增加（如 56 层 vs 20 层），训练误差和测试误差同步上升，这种现象被称为 **退化问题（Degradation Problem）**。

作者通过实验发现，这并非梯度消失——使用了 Batch Normalization 和标准初始化后，前向/反向传播的信号已经相对稳定。**问题出在网络过于复杂，难以学习恒等映射**：理论上添加恒等映射层不应降低性能，但实际中深层网络很难拟合出恒等映射。

核心洞察：如果让网络学习残差映射 $\mathcal{H}(x) - x$ 而非原始映射 $\mathcal{H}(x)$，那当最优解就是恒等映射时，残差自然趋近于零，优化难度大大降低。

## 二、核心方法

ResNet 的核心设计是残差块（Residual Block）：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

其中 $\mathcal{F} = W_2 \sigma(W_1 \mathbf{x})$，$\sigma$ 为 ReLU 激活函数。

**关键设计元素：**

| 设计 | 描述 | 作用 |
|------|------|------|
| Skip Connection（恒等映射） | 输入直接跳跃 2-3 层加到输出 | 梯度直接流过加法门，解决梯度消失 |
| 瓶颈设计（Bottleneck） | 1×1 → 3×3 → 1×1 三明治结构（256-d → 64-d → 64-d → 256-d） | 大幅减少计算量 |
| 下采样 | stride=2 的卷积 + 1×1 conv 投影 shortcut | 跨分辨率传递信息 |
| 预激活设计 | BN-ReLU-Conv 顺序（Pre-Activation） | 改进梯度流动 |

**架构系列：**

| 模型 | 层数 | 参数 | ImageNet Top-5 Error |
|------|------|------|---------------------|
| ResNet-18 | 18 | 11M | — |
| ResNet-34 | 34 | 22M | — |
| ResNet-50 | 50 | 26M | ~7.0% |
| ResNet-101 | 101 | 45M | ~6.0% |
| ResNet-152 | 152 | 60M | 3.57%（Ensemble） |

## 三、关键实验与发现

1. **退化问题被系统解决**：ResNet-152 在 ImageNet 上首次实现 152 层的有效训练，top-5 error 仅 3.57%（Ensemble），比 VGG 更深的网络首次优于浅层网络。

2. **ILSVRC 2015 全项目冠军**：分类、检测、定位三项全部第一；COCO 2015 检测、分割双料冠军。

3. **极致深度的验证**：在 CIFAR-10 上成功训练了 1001 层的 ResNet（$n=333$，每个 block 2 层），验证了方法的可扩展性。

4. **CIFAR-10 性能**：1001 层 ResNet 达到 4.62% 测试错误率，比 110 层（6.43%）更好，残差学习在极深网络上依然有效。

5. **消融实验关键发现**：恒等映射 shortcut 比 projection shortcut 好（参数更少且性能更优），这验证了残差学习的核心思想——让 shortcut 尽可能简单，学习发生在主干路径。

## 四、局限性与后续影响

**局限性：**
- 极深层网络（1000+ 层）收益递减，更多层数带来的提升越来越小
- 梯度仍然可能在一定程度上衰减（虽然大幅缓解）
- 残差块的设计选择（shortcut 类型、block 结构）需要手工调参

**后续影响：**
- 残差连接成为深度学习的标准组件，几乎所有深度学习架构都继承了这一思想
- Pre-LN Transformer 本质上依赖残差连接来训练深层
- 启发了 DenseNet（密集连接）、ResNeXt（分组卷积）、Wide ResNet（加宽而非加深）等一系列工作

## 五、VLA/机器人研究中的角色

ResNet 在 VLA 和具身智能研究中扮演基础设施角色：

- **OpenVLA** 的视觉编码器使用了类似 ResNet 的卷积结构
- **Diffusion Policy** 的视觉编码器基于 ResNet
- **ACT**（Action Chunking Transformer）的视觉编码器使用 ResNet 骨干
- ResNet 思想被扩展到 Transformer（Pre-LN Transformer 本质上是残差连接）
- 在需要低延迟的机器人系统中，ResNet-50 仍然是高效视觉编码器的可靠选择

## 六、对你的启示

1. **残差学习思想具有普适性**：当某个模块难以直接学习时，将其设计为"增量学习"（学习残差），优化难度大幅降低。这一思想可以推广到目标检测、图像生成、视频理解等多种任务。

2. **简洁的设计哲学**：ResNet 的核心创新仅 2 行公式，但产生了巨大影响。好的研究不在于复杂度，而在于直击问题本质。

3. **梯度流动的视角**：设计深度网络时始终关注梯度能否顺利从顶部流到底部。每一层的设计都应让梯度流动尽可能简单直接。

4. **资源友好**：ResNet-50 推理仅需约 2GB VRAM，在机器人实验中是非常经济的视觉 backbone 选择。

## PDF

[[Deep Residual Learning for Image Recognition.pdf]]
