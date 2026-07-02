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
github: "https://github.com/google-research/vision_transformer"
---

# ViT

**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**
*Google Research, Brain Team | ICLR 2021 | arXiv: 2010.11929*

> 将 NLP 的 Transformer 直接应用于图像——不做任何 CNN 特定的归纳偏置。将图像切成 16x16 patches，当作"视觉单词"喂给标准 Transformer Encoder。核心贡献在于证明：当数据足够大时（JFT-300M / ImageNet-21K），纯 Transformer 可以超越 CNN；而数据不足时，CNN 的归纳偏置仍有优势。这篇文章是视觉 Transformer 时代的开创者，所有现代视觉基础模型（DINOv2, SigLIP, DiT）都基于 ViT 架构。

---

## 一、Background / Core Idea

### 1.1 CNN 的主导地位与归纳偏置

2012-2020 年，CV 领域完全由 CNN 主导。CNN 的成功部分归因于其内置的归纳偏置：
- **平移等变性**（Translation Equivariance）：卷积核对图像各位置共享，无论目标出现在图像哪边，相同效果
- **局部性**（Locality）：卷积核仅在局部邻域操作，捕捉的是局部模式
- **层级特征**：低层捕捉边缘/纹理，中层捕捉部件，高层捕捉完整物体

这些偏置在**小数据上极其有效**——模型不需要大量数据就能学会视觉的基本结构。但同时也限制了 CNN 对长距离依赖关系的建模能力（需要堆叠很多层才能连接远距离像素）。

### 1.2 NLP 中 Transformer 的成功

与此同时，[[Attention Is All You Need|Transformer]] 在 NLP 领域取得了前所未有的成功：
- [[BERT|BERT]] 和 [[GPT|GPT]] 展示了在足够数据下，**没有归纳偏置的模型可以比带有强偏置的模型学得更好**
- Transformer 的 Self-Attention 可以一次性看到序列中的所有位置，直接建模长距离依赖
- 随数据量和模型大小的扩展性极好（Scaling Laws）

### 1.3 核心问题：不用 CNN，Transformer 能做视觉吗？

**"How much of the CNN-specific inductive bias is actually necessary for image recognition?"**

作者的核心假设：如果数据足够大，模型可以自行学会视觉结构，不需要手动设计的 CNN 偏置。ViT 的目标就是尽可能地不引入 CV 特定的设计，直接测试标准 Transformer 在图像上的表现。

### 1.4 与之前工作的区别

此前将注意力机制引入视觉的工作主要分两类：
1. **将注意力作为 CNN 的补充**：Attention Augmented Convolution、Non-local Networks——注意力只替换部分卷积层
2. **特殊设计的注意力**：Sparse Transformer、轴向注意力——需要复杂的 Attention 工程才能在图像上高效运行

ViT 的做法是**最激进的**：几乎完全不做 CV 特定的改动，直接用标准 Transformer 处理图像 patches。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 Patch Embedding：将图像变成 tokens

这是 ViT 最核心的设计。对于输入图像 x ∈ ℝ^{H×W×C}，将其分割为固定大小的二维 patches，每个 patch 的大小为 (P, P)：

序列长度 N = HW / P²（对于 224² 图像和 16² patch：N = 196）

每个 patch 被展平为向量 x_p ∈ ℝ^{P²·C}（对于 16²×3：维度 768），然后通过可学习线性投影 E ∈ ℝ^{(P²·C)×D} 映射到 Transformer 的隐藏维度 D：

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

### 2.2 [class] token：借鉴 BERT

ViT 完全照搬 [[BERT|BERT]] 的 `[CLS]` token 设计——在输入序列前额外添加一个可学习的嵌入向量。该 token 经过 Transformer Encoder 后的最终输出状态作为整个图像的表示用于分类：

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

为什么用 [class] token 而不是对所有 patch 的输出做池化？作者在消融实验中比较了两种方式，发现 [class] token 略好（ImageNet top-1 差距约 0.3%），可能是由于 [class] token 可以学习自适应地"查询"它认为重要的 patches。

### 2.3 Position Embeddings：1D 可学习位置编码

ViT 使用**标准可学习的 1D 位置编码**，不引入 2D 先验。这是有意为之——虽然看起来 2D 位置编码更符合视觉直觉（考虑 x/y 坐标），但作者在消融实验（附录 D.4）中发现，使用 1D、2D、相对位置编码的差异不显著（差距 <0.1%）。

习得的位置编码（Figure 7 center）展示出良好的结构：相近的 patches 具有更相似的位置编码，且存在行/列结构。说明模型能从数据中自动学到"视觉空间"的 2D 拓扑。

### 2.4 Transformer Encoder：标准实现

与原始 Transformer 完全一致（原论文 Eq. 2-3）：

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$$
$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell$$

关键设计选择：
- **Pre-LN**：LayerNorm 在注意力/MLP 之前（而非原始的 Post-LN），这有助于更稳定的训练
- **GELU 激活**：MLP 使用 GELU 而非 ReLU
- **Transformer 编码器**（而非 Decoder）：双向 Self-Attention，不带因果掩码

### 2.5 模型配置：三种规模 × 两种 Patch Size

| 模型 | Patch Size | Layers | Hidden Dim D | MLP Dim | Heads | Params |
|------|-----------|--------|-------------|---------|-------|--------|
| ViT-B/16 | 16×16 | 12 | 768 | 3072 | 12 | 86M |
| ViT-B/32 | 32×32 | 12 | 768 | 3072 | 12 | 86M |
| ViT-L/16 | 16×16 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-L/32 | 32×32 | 24 | 1024 | 4096 | 16 | 307M |
| ViT-H/14 | 14×14 | 32 | 1280 | 5120 | 16 | 632M |

Patch size 影响序列长度：ViT-B/16 有 196 patches，ViT-B/32 只有 49 patches。**更小的 patch 提供更好的精度但计算成本更高。**

### 2.6 Hybrid 架构选项

作为替代方案，输入序列也可以从 CNN 特征图提取，而非原始图像 patches。Hybrid 模型将 ResNet 特征图的 patches 作为 Transformer 输入。在较小规模时，Hybrid 略优于纯 ViT（Figure 5），但在大规模下差距消失。

### 2.7 微调与高分辨率调整

ViT 的一个重要能力是可以**在不同分辨率下微调**（fine-tune at higher resolution）。由于保持 patch size 不变，更高分辨率 = 更多 patches，此时只需对预训练的位置嵌入做 2D 插值即可适配。

这是 ViT 中**唯二**引入 2D 空间先验的地方（另一个是 patch 提取本身）。

### 2.8 训练配置

- **优化器**：Adam (β₁=0.9, β₂=0.999)
- **Batch size**：4096
- **Weight decay**：0.1（比其他模型常用的值大 10 倍）
- **学习率调度**：线性 warmup + 线性衰减
- **微调**：SGD with momentum，batch size 512

---

## 三、Experiments and Key Findings

### 3.1 大数据预训练的威力——核心实验

| 模型 | 预训练数据 | ImageNet Top-1 | ImageNet ReaL | CIFAR-100 | VTAB (19 tasks) |
|------|-----------|----------------|---------------|-----------|-----------------|
| ViT-B/16 | ImageNet-1K | ~77.9 | — | — | — |
| ViT-L/16 | ImageNet-21K | 85.30 | 88.62 | 93.25 | 72.72 |
| ViT-H/14 | JFT-300M | **88.55** | **90.72** | **94.55** | **77.63** |
| BiT-L (ResNet152x4) | JFT-300M | 87.54 | 90.54 | 93.51 | 76.29 |
| Noisy Student (EfficientNet-L2) | ImageNet+JFT | 88.5 | 90.55 | — | — |

**核心结果**：ViT-H/14 在 ImageNet 上达到 88.55%，超过此前 SOTA——但前提是在 JFT-300M 上预训练。在 ImageNet-1K 上直接训练时，ViT 比 ResNet 差 3-4%（Figure 3）。

**训练效率**：ViT-H/14 的预训练仅需 2.5K TPUv3-core-days，而 BiT-L 需要 9.9K，Noisy Student 需要 12.3K——**ViT 用少得多的计算量达到了顶级性能**。

### 3.2 数据规模对性能的影响

原论文 Figure 3 是 ViT 最重要的发现之一：

| 预训练数据集 | 规模 | ViT vs ResNet |
|-------------|------|---------------|
| ImageNet-1K | 1.3M | ViT 落后 3-4% |
| ImageNet-21K | 14M | ViT 与 ResNet 持平或略优 |
| JFT-300M | 303M | ViT 显著领先 |

**结论**：CNN 的归纳偏置在数据不足时是优势（ViT 需要更多数据来学习同样的视觉结构），但**在数据足够大时，去除偏置的模型反而学得更好**。

### 3.3 VTAB 任务分解分析

原论文 Figure 2 展示了 VTAB 三类任务的分解：

| 模型 | Natural (7 tasks) | Specialized (4 tasks) | Structured (8 tasks) |
|------|------------------|---------------------|--------------------|
| ViT-H/14 | **83.2** | **88.1** | **66.0** |
| BiT-L | 82.3 | 88.0 | 63.8 |

ViT 在 Natural 和 Structured 任务上显著优于 ResNet，而在 Specialized（如医学/卫星图像）上接近。**Structured 任务的提升**（+2.2）最为显著——这些任务需要几何理解（如计数、深度估计），全局注意力可能在这些任务上特别有帮助。

### 3.4 计算量-性能对比 (Scaling Study)

原论文 Figure 5 展示了在相同计算预算下 ViT、ResNet 和 Hybrid 的性能对比。核心发现：
- ViT 在相同计算量下平均超出 ResNet 2-4x（达到同等性能需要 1/2 到 1/4 的计算量）
- Hybrid 在小计算量时稍优，但大模型差距消失——**说明卷积的特征处理对大规模 ViT 并不必要**
- ViT 在实验范围内没有出现饱和趋势，提示**进一步扩展可能带来更多提升**

### 3.5 注意力距离分析（Section 4.5）

原论文 Figure 7 (right) 展示了一个有趣的发现：在**最底层**（layer 0），某些注意力头已经具有很大的注意力距离（平均 ~60-80 像素），而其他头则集中在局部（~10-20 像素）。随着网络深度增加，多头注意力的平均距离更加多样化。

这揭示了 ViT 的工作原理：**在最低层即可实现全局信息整合**，而 CNN 需要堆叠很多层才能达到同样的感受野。这与 ViT 在某些需要全局理解的任务上的优势一致。

### 3.6 自监督预训练的初步探索

原论文对自监督预训练（masked patch prediction）做了初步实验：ViT-B/16 在 ImageNet 上达到 79.9%（超过从零训练的 2%），但比有监督预训练低 4%。这个问题后来由 [[MAE|MAE]] 和 [[DINOv2|DINOv2]] 等后续工作解决。

---

## 四、Limitations and Challenges

### 4.1 小数据下不如 CNN

这是 ViT 最明确的局限。在 ImageNet-1K 规模的数据集上，ViT 的性能落后于参数相当的 ResNet。这意味着 ViT 不适合数据量有限的应用场景（如医学影像、小众分类）。

### 4.2 全局注意力的 O(N²) 计算瓶颈

对于 224×224 图像（196 patches）尚可，但对于高分辨率图像（如 1024×1024，约 4096 patches），Self-Attention 的内存占用会呈平方级增加。这限制了 ViT 在高分辨率密集预测任务中的应用（后续 [[Swin Transformer|Swin Transformer]] 通过窗口注意力解决了这一问题）。

### 4.3 位置编码的插值问题

当在高于预训练的分辨率下微调时，位置编码需要进行 2D 插值。朴素的双线性插值可能对最终性能造成影响。后续工作探索了条件位置编码（CPE）等改进方案。

### 4.4 缺少多尺度特征（没有特征金字塔）

ViT 在所有层保持相同分辨率（单尺度），缺乏类似 CNN 的特征金字塔。这对密集预测任务（检测、分割）不利。[[Swin Transformer|Swin Transformer]] 通过层次化设计解决了这个问题。

---

## 五、Relationship with Subsequent Work / Impact on the Field

ViT 在 CV 领域的影响是决定性的——它开启了视觉 Transformer 时代：

| 方向 | 模型 | 与 ViT 的关系 |
|------|------|--------------|
| 数据高效的 ViT | [[../15_DINOv2/DINOv2.md|DeiT]] (Touvron et al., 2021) | 通过知识蒸馏和大规模增强，使 ViT 可以在 ImageNet-1K 上直接训练 |
| 层次化 ViT | [[../12_Swin_Transformer/Swin Transformer.md|Swin Transformer]] (Liu et al., 2021) | ViT + 层次化特征 + 窗口注意力，适合密集预测 |
| 自监督 ViT | [[../14_MAE/MAE.md|MAE]] (He et al., 2022) | 在 ViT 上做 mask 自监督预训练 |
| | [[../15_DINOv2/DINOv2.md|DINO/DINOv2]] (Caron et al., 2021/2023) | ViT 的自蒸馏自监督学习 |
| ViT + 对比学习 | **SigLIP** (Zhai et al., 2023) | ViT 作为 CLIP/SigLIP 的视觉编码器 |
| ViT 用于生成 | [[../19_DiT/DiT.md|DiT]] (Peebles & Xie, 2023) | ViT 作为扩散模型的骨干——用 ViT 替代 U-Net |

**ViT 在 VLA 中的角色——核心且直接**：

1. **OpenVLA 使用 DINOv2（ViT-based）+ SigLIP（ViT-based）双视觉编码器**——两个编码器都是 ViT 变体
2. **π0 的 VLM 组件 PaliGemma 使用 ViT 作为视觉 backbone**
3. **RT-2 的 PaLI-X 视觉骨干使用 ViT-G（参数量超过 2B 的超级 ViT）**
4. **DiT（Diffusion Transformer）本质上是 ViT 的生成变体**——将图像 patch 作为扩散状态
5. **几乎所有现代 VLA 的视觉理解部分都基于 ViT 或其变体**（SigLIP, DINOv2, CLIP）

---

## 六、Implications for You / Hardware Compatibility

- ✅ **ViT 的代码极为简洁（~100 行 PyTorch）**：实现一个可运行的 ViT 是从零理解视觉 Transformer 的最佳起点。建议先实现 Patch Embedding → [class] token → Position Embedding → Transformer Encoder 的完整流程
- ⚠️ **ViT-B/16 (86M) 在单 GPU 上可运行**：在 16GB VRAM GPU 上，batch size 32 的推理/微调可行。ViT-L/16 (307M) 在 24GB VRAM 上运行需要梯度累积
- ❌ **不建议在小数据场景（<10M 图像）使用 ViT**：应当使用 DeiT、[[Swin Transformer|Swin]] 或直接使用预训练 ViT 做微调而非从零训练
- ✅ **如果你做 VLA，直接使用 SigLIP 或 DINOv2 的预训练 ViT 权重**：这是当前 VLA 视觉编码器的事实标准，Hugging Face 上有现成的 `transformers` 实现
- ⚠️ **ViT 的全局注意力对高分辨率视频帧不友好**：当处理多帧高分辨率输入时（多个 224×224 帧），序列长度会线性增加，需考虑窗口注意力或分层采样
- ✅ **"消除归纳偏置"的哲学在 VLA 中同样成立**：ViT 启示我们——当数据量足够大时，更通用的架构比精心设计的专用架构更好。这在机器人领域意味着：大规模多机器人数据训练 > 精心设计的神经网络偏置

## PDF

[[ViT 原文.pdf]]
