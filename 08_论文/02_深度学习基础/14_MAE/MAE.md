---
tags:
  - 论文
  - 自监督学习
  - 视觉预训练
  - Masked Modeling
created: 2026-06-30
paper_title: "Masked Autoencoders Are Scalable Vision Learners"
paper_authors: "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick"
paper_year: 2021
paper_venue: "CVPR 2022"
paper_citations: "~10,000+"
paper_url: "https://arxiv.org/abs/2111.06377"
---

# MAE

**Masked Autoencoders Are Scalable Vision Learners**
*Meta AI / FAIR | CVPR 2022 | arXiv: 2111.06377*

> 把 BERT 的 MLM 思想成功移植到视觉领域。随机 mask 掉 75% 的图像 patch，让模型从剩余 25% 重建完整图像。核心洞察：非对称架构使训练极高效，图像的强空间冗余使高 mask ratio 成为可能。

---

## 一、研究背景与动机

在 NLP 领域，BERT 的 Masked Language Modeling（MLM）已经成为自监督预训练的标准范式——随机 mask 掉文本中的 tokens，训练模型从上下文重建它们。然而这一方法在视觉领域的直接迁移一直没有取得理想效果。

原因在于**图像和语言的本质差异**：
- **空间冗余**：图像中相邻像素高度相似——一个"缺失"的 patch 可以从邻近 patch 直接推测出来（例如缺失的一小块蓝色天空可以很容易的从周围的蓝色补全）
- **信息密度**：图像的信息密度远低于语言——一句话中删掉一个词可能完全改变语义，但一张图删掉一个 patch 几乎不影响对整图的理解

核心问题：**如何将 mask 自编码器有效应用于视觉领域？**

MAE 的关键洞察：**不需要像 BERT 那样只 mask 15%，视觉的自编码器应该 mask 75%，迫使模型学习全局语义理解（而非局部像素插值）**。同时采用**非对称编码器-解码器设计**——Encoder 只处理可见 patch，Decoder 轻量重建全图——使预训练效率大幅提升。

## 二、核心方法

**整体架构：**

```
原始图像 → 分成 patches → 随机 mask 75% → 可见 patches 输入 Encoder → 
添加 mask tokens → 轻量 Decoder → 重建原始像素 → MSE Loss
```

**关键设计元素：**

| 设计 | 具体实现 | 核心原因 |
|------|---------|---------|
| **极端 Mask Ratio (75%)** | 随机遮挡 75% 的 patch | 迫使模型理解图像语义，而非局部插值 |
| **非对称编码器** | 仅编码可见 patch（25%） | 大量减少计算（~4x 加速），可兼容大 ViT |
| **轻量解码器** | 8 层 Transformer，256-dim | 重建任务对解码能力要求不高，预训练完就丢弃 |
| **Mask Token** | 可学习的共享向量 | 占位符，解码时标记需要预测的位置 |
| **简单重建目标** | MSE Loss 在像素空间 | 无需对比学习、投影头、动量编码器 |

**训练流程：**

1. 图像分 patch：$224\times 224 \rightarrow 14\times 14 = 196$ patches
2. 随机 mask：保留 49 个 visible patches（25%），mask 147 个（75%）
3. Encoder（ViT-Large）处理 49 个 visible patches → 特征向量
4. 在对应位置插入 learnable mask tokens → 完整 196 序列
5. Decoder 重建像素 → 计算正常像素 MSE Loss
6. 预训练后丢弃 Decoder，仅保留 Encoder 用于下游任务

## 三、关键实验与发现

1. **Mask Ratio 的敏感性**：最佳 mask ratio 为 75%。mask 比 BERT 的 15% 好（75% → 83.6% vs 15% → ~80%），mask 90% 时还能保持相对好性能（~82%）。75% mask 意味着 Encoder 只需要处理 25% 的 tokens，**训练速度提升 4 倍以上，显存减少 2-3 倍**。

2. **非对称架构的效率验证**：Encoder 只处理可见 patch（25%）vs 处理全部 patch（100%），在相同计算预算下 MAE 显著更好。

3. **线性探测性能**：ViT-L 用 MAE 预训练后线性探测达到 77.4%（仅使用冻结特征 + 线性分类器），接近有监督 ResNet-50（79%）。

4. **端到端微调性能**：MAE 预训练 ViT-L 在 ImageNet 上达到 86.1%（仅 ImageNet-1K），ViT-H 达到 87.8%，接近或超过有监督 SOTA。

5. **数据效率**：MAE 预训练的数据效率极高——即使只在 ImageNet-1K 上预训练（无外部数据），ViT-L 也能达到 86.1% top-1，证明了纯自监督预训练的价值。

6. **可扩展性**：模型越大，MAE 预训练效果越好。ViT-H (632M) 比 ViT-L (307M) 显著更好，说明 MAE 是一种可扩展的自监督方法。

## 四、局限性与后续影响

**局限性：**
- **预训练任务与下游任务存在 gap**：MAE 训练的是像素重建能力，但下游任务（分类、检测、分割）需要的是语义理解。虽然 gap 可以通过微调弥合，但并非最优预训练目标
- **对小 ViT 的提升有限**：ViT-S (21M) 从 MAE 预训练获益较少（线性探测仅 ~74% vs 有监督 ~79%）
- **mask 策略简单**：均匀随机 mask 不是最优策略——有空间结构的 mask（block-wise、semantic-aware）可能更好
- **重建目标的选择**：像素 MSE 可能不如 perceptual loss 或 tokenizer-based target 有效

**后续影响：**
- MAE 成为视觉自监督预训练的主流方法之一
- 启发了 video-MAE、MixMAE、AIM 等一系列 masked modeling 方法
- MAE 的"高 mask ratio + 非对称架构"设计哲学影响了后续多模态预训练

## 五、VLA/机器人研究中的角色

MAE 在 VLA 中扮演视觉表征预训练的重要角色：

- MAE 预训练的视觉编码器**为 VLA 提供强大的视觉表征基础**——使得在通用图像理解上预训练的模型可以高效迁移到机器人任务
- **"视觉预训练 → 机器人微调"**的范式受益于 MAE 的高效预训练——MAE 可以在通用数据上自监督训练，然后在机器人数据上微调
- Mask 策略的思想**启发了 VLA 中处理部分观测/遮挡场景的方法**——机器人总是在部分可见的环境中工作，MAE 的训练方式本身就和机器人感知高度契合
- MAE 的超高效率预训练使实验室级别的研究者在有限算力下也能预训练大模型

## 六、对你的启示

1. **"简单"是最强大的武器**：MAE 的设计极简——无对比学习、无投影头、无动量编码器、无数据增强，仅 MSE 损失。但正是这种简洁让它在各种条件下都能稳定工作。不要为了创新而增加复杂度。

2. **理解模态的固有特性**：MAE 成功的根源是深刻理解了"图像具有高度空间冗余"，并针对这一特性设计了极端 mask ratio。每个模态都有其独特的特性，找到并利用它们比照搬其他领域的方法更重要。

3. **不对称不仅是效率问题，也是表现力问题**：非对称架构（轻量 Decoder + 重 Encoder）既提高了训练效率，也迫使 Encoder 学习更语义化的表征（Decoder 太弱，无法依赖像素级操作完成任务）。这是一种优雅的设计——约束本身成为一种正则化。

4. **"预训练 + 微调"范式的生命力**：MAE 强化了 "large-scale pre-training + task-specific fine-tuning" 的范式。在机器人领域，这意味着在通用数据上预训练视觉编码器、在机器人数据上微调，可能是数据效率最优的策略。

## PDF

[[Masked Autoencoders Are Scalable Vision Learners.pdf]]
