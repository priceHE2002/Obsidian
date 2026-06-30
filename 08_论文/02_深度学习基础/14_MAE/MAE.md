---
tags:
  - 论文
  - 自监督学习
  - 视觉预训练
  - Masked Modeling
  - ViT
created: 2026-06-30
paper_title: "Masked Autoencoders Are Scalable Vision Learners"
paper_authors: "Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick"
paper_year: 2021
paper_venue: "CVPR 2022"
paper_citations: "~10,000+"
paper_url: "https://arxiv.org/abs/2111.06377"
github: "https://github.com/facebookresearch/mae"
---

# MAE

**Masked Autoencoders Are Scalable Vision Learners**
*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick / Meta AI (FAIR) | CVPR 2022 | arXiv: 2111.06377*

> **Pitch**: 把 BERT 的 masked language modeling 思想成功移植到视觉领域。关键洞察：图像具有空间冗余（远高于语言），因此需要极端 mask ratio（75% vs BERT 的 15%），迫使模型学习全局语义而非局部像素插值。非对称架构（Encoder 只处理可见 patch）带来 3-4x 训练加速，使大 ViT 的自监督预训练变得高效实用。

---

## 一、Background / Core Idea

### 1.1 NLP 的 MLM 成功与视觉迁移困境

BERT 的 Masked Language Modeling（MLM）在 NLP 取得巨大成功——random mask 掉 15% 的 tokens，训练模型从上下文中重建它们，模型学会了深层的语言理解。但直接迁移到视觉领域长期效果不佳。MAE 论文系统性地分析了差距的根源。

### 1.2 图像 vs 语言的本质差异

**（i）信息密度差异（核心）：** 语言是高度语义化、信息密集的人造信号。删掉一个词可能完全改变句子语义。图像是自然信号，**相邻像素高度冗余**——缺失的一小块蓝色天空可以很容易从周围蓝色 patch 补全。这意味着视觉的 MLM 必须有截然不同的设计。

**（ii）架构鸿沟（已被 ViT 解决）：** CNN 在规则网格上操作，无法自然集成 "mask token" 和 positional embedding。ViT 的出现消除了这一障碍。因此 MAE 采用 ViT 作为骨干。

**（iii）Decoder 的角色差异：** 在 NLP 中，预测缺失的词本身就是高度语义化的任务。在视觉中，重建像素是低语义任务——这导致了一个微妙但重要的设计思考：Decoder 必须足够深，"吸收"重构专用性，让 Encoder 的 latent representation 更抽象。

### 1.3 MAE 的核心设计哲学

**"简单"是最强大的武器：** 无对比学习、无投影头、无动量编码器、无特殊数据增强、仅 MSE 损失。正是这种极简让 MAE 在各种条件下都能稳定工作。核心洞察：利用**极端 mask ratio（75%）** 迫使模型超越局部插值，去学习对物体和场景的整体理解。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 整体流程

```
原始图像 (224×224) → 分成 14×14=196 patches → random mask 75%（147 patches）→ 
Encoder（ViT）处理 49 个 visible patches → 添加 learnable mask tokens → 
轻量 Decoder 重建全图像素 → MSE Loss（仅计算 masked patches）
预训练后丢弃 Decoder，仅保留 Encoder
```

### 2.2 Masking 策略

**随机采样（uniform random without replacement）：** 这是默认策略，并经过消融验证为最优。相比 block-wise masking（BEiT 使用的大块遮挡）和 grid-wise masking（规则网格间隔采样），随机采样在 mask ratio 75% 时表现最好。Block-wise 在 75% 时性能严重下降（linear probe 63.9% vs random 73.5%），因为遮挡区域过大导致任务过难。Grid-wise 虽然重建更清晰，但任务过简单（相当于规则下采样），representation 质量较低。

Uniform 分布还防止了中心偏置（center bias）。更重要的是，每个 epoch 不同的随机 mask = 天然的 data augmentation。

### 2.3 非对称 Encoder-Decoder 架构

**Encoder（重）：** 标准 ViT，但**仅处理 visible patches**（25% = 49 patches）。这意味着自注意力计算量大幅减少——self-attention 复杂度是 O(n²)，patch 数从 196 降到 49 意味着注意力计算减少约 16 倍。Encoder 中没有 mask token，确保部署时（输入全图无 mask）和预训练时的输入分布一致。

**Decoder（轻量）：** 8 层 Transformer，hidden dim 512，每 token 计算量仅为 Encoder 的 9%。Decoder 接收 encoded visible patches + learnable mask tokens（共享向量）+ positional embeddings，重建全图。

**为什么非对称是关键：** 如果 Encoder 也处理 mask token（非对称的对立面），linear probing 从 73.5% 暴跌到 59.6%，且训练 FLOPs 增加 3.3×。原因：Encoder 训练时看到 mask token，部署时却看不到，造成 train-test mismatch。

### 2.4 训练目标

**重建目标：** 预测每个 masked patch 的归一化像素值（per-patch normalization：计算 patch 内所有像素的 mean 和 std 进行归一化）。归一化增强了局部对比度，比未归一化像素（84.9% → 85.4%）和 dVAE token（85.3%）都好。

Loss 函数：
$$L = \mathbb{E}\left[ ||\text{patch\_norm}(x_{masked}) - \text{patch\_norm}(\hat{x}_{masked})||^2 \right]$$

Loss **仅在 masked patches 上计算**（类似 BERT）。如果在全图计算 loss，精度下降约 0.5%。

### 2.5 简单实现

MAE 不需要任何稀疏操作。实现极简：1）对每个 input patch 做 linear projection + positional embedding；2）随机 shuffle token list，移除后 75%；3）Encoder 处理剩余 25%；4）添加 mask tokens，unshuffle 恢复原始顺序；5）Decoder 处理完整序列（196 tokens）。shuffle/unshuffle 开销可忽略。

### 2.6 Encoder 实现细节

Encoder 末尾有一个 linear projection 层（匹配 Encoder 和 Decoder 的不同宽度）。使用 sine-cosine positional embeddings。不使用 relative position bias 或 layer scaling（与 BEiT 不同）。Pre-training 设置：

| 超参数 | 值 |
|--------|-----|
| Optimizer | AdamW |
| base lr | 1.5e-4（线性缩放规则：lr = base_lr × batch_size / 256） |
| weight decay | 0.05 |
| β₁, β₂ | 0.9, 0.95 |
| batch size | 4096 |
| lr schedule | cosine decay |
| warmup | 40 epochs |
| augmentation | 仅 RandomResizedCrop |
| **无** | 无 color jitter、no drop path、no gradient clip |

### 2.7 Fine-tuning 设置

| 超参数 | 值 |
|--------|-----|
| optimizer | AdamW |
| base lr | 1e-3 |
| weight decay | 0.05 |
| β₁, β₂ | 0.9, 0.999 |
| layer-wise lr decay | 0.75 |
| training epochs | 100 (B), 50 (L/H) |
| augmentation | RandAug (9, 0.5) |
| label smoothing | 0.1 |
| mixup | 0.8 |
| cutmix | 1.0 |
| drop path | 0.1 (B/L) 0.2 (H) |

---

## 三、Experiments and Key Findings

### 3.1 Mask Ratio 的敏感性

实验发现最优 mask ratio 为 **75%**。这一比例远超 BERT 的 15%，也远超此前视觉 masked modeling 方法（iGPT 20-50%，BEiT 20-50%）。

| Mask Ratio | Fine-tuning | Linear Probing |
|------------|-------------|----------------|
| 10% | 84.7 | 54.6 |
| 25% | 84.9 | 58.9 |
| 50% | 84.9 | 67.0 |
| **75%** | **84.9** | **73.5** |
| 80% | 84.5 | 71.8 |
| 90% | 83.0 | 66.1 |

75% mask 意味着 Encoder 只需处理 25% 的 tokens，**训练速度提升 4 倍，显存减少 2-3 倍**。Linear probing 对 mask ratio 更敏感（差距 20%），fine-tuning 在 40-80% 范围内都表现良好。

### 3.2 Decoder 设计消融

| Decoder Depth | Fine-tuning | Linear Probing |
|---------------|-------------|----------------|
| 1 block | 84.8 | 65.5 |
| 2 blocks | 84.9 | 70.0 |
| 4 blocks | 84.9 | 71.9 |
| **8 blocks (default)** | **84.9** | **73.5** |
| 12 blocks | 84.4 | 73.3 |

Decoder 至少需要一定深度来"吸收"像素重建的 specialization。有趣的是，1-block decoder 在 fine-tuning 下表现良好（84.8%），但在 linear probing 下表现差（65.5%），说明浅层 Decoder 让 Encoder 学到更多重建专用特征而非语义特征。

### 3.3 训练时长的影响

训练时间越长越好，**1600 epochs 未见饱和**（linear probing 持续提升）。这与对比学习方法（MoCo v3 在 300 epochs 饱和）形成鲜明对比。原因：MAE Encoder 每 epoch 只看 25% 的 patches，需要更多 epoch 来充分看到所有数据。另一种视角：MAE 每 epoch 看到的实际像素约等于对比学习的两 crop 方法（后者每 epoch 看到 200% patches）。

### 3.4 数据增强的作用

MAE 几乎不需要数据增强——即使完全没有 augmentation（仅 center crop），linear probing 仍达 65.7%（fine-tuning 84.0%）。添加 color jitter 反而降低性能。这**与对比学习截然不同**（BYOL 去掉 augmentation 性能下降 13%，SimCLR 下降 28%）。原因：random masking 本身为每次迭代生成不同的训练样本，是一个天然的 data augmentation。

### 3.5 与 SOTA 对比

| Method | Pre-train Data | ViT-B | ViT-L | ViT-H | ViT-H(448) |
|--------|---------------|-------|-------|-------|------------|
| Scratch | - | 82.3 | 82.6 | 83.1 | - |
| DINO | IN1K | 82.8 | - | - | - |
| MoCo v3 | IN1K | 83.2 | 84.1 | - | - |
| BEiT | IN1K+dVAE | 83.2 | 85.2 | - | - |
| **MAE** | **IN1K** | **83.6** | **85.9** | **86.9** | **87.8** |

MAE 使用纯 IN1K 数据达到 87.8% top-1（ViT-H, 448 size），超越所有仅使用 IN1K 数据的方法。BEiT 需要额外的 dVAE pre-training（250M 数据），且 MAE 每 epoch 快 3.5×。

### 3.6 部分微调实验

Linear probing 和 full fine-tuning 的结果高度不相关。MAE 的 linear probing（73.5%，MoCo v3 77.6%）虽低于 MoCo v3，但**只要微调 1 个 Transformer block**，MAE 就反超（81.0%），微调 4 个 blocks 领先 2.6%。这说明 MAE 学到的是强非线性特征，线性可分离性不是唯一指标。

### 3.7 迁移学习

**COCO 目标检测与分割（Mask R-CNN）：**

| Method | Pre-train | ViT-B APbox | ViT-L APbox | ViT-B APmask | ViT-L APmask |
|--------|-----------|-------------|-------------|--------------|--------------|
| Supervised | IN1K labels | 47.9 | 49.3 | 42.9 | 43.9 |
| MoCo v3 | IN1K | 47.9 | 49.3 | 42.7 | 44.0 |
| BEiT | IN1K+DALLE | 49.8 | 53.3 | 44.4 | 47.1 |
| **MAE** | **IN1K** | **50.3** | **53.3** | **44.9** | **47.2** |

MAE 在 ViT-L 上超过 supervised 4.0 APbox。

**ADE20K 语义分割（UperNet）：**

MAE ViT-L mIoU 53.6，超过 supervised（49.9）+3.7，超过 BEiT（53.3）+0.3，超过 MoCo v3（49.1）+4.5。

### 3.8 数据效率对比

MAE 1600 epochs 在 128 TPU-v3 上训练 ViT-L 仅需 31 小时，而 MoCo v3 300 epochs 需要 36 小时。MAE 在更少的实际训练时间内达到了更高的精度。

---

## 四、Limitations and Challenges

1. **预训练-下游任务 gap**：MAE 训练像素重建能力，但下游任务（分类、检测、分割）需要语义理解。虽然可以通过微调弥合，但并非最优的预训练目标。这解释了对小模型提升有限的原因。

2. **对小模型不友好**：ViT-S（21M）从 MAE 预训练获益较少。在模型太小时，像素重建任务的"吸收"效果不明显，线性探测仅约 74%。

3. **mask 策略简单**：均匀随机 mask 可能不是最优。有空间结构的 mask（block-wise、semantic-aware）或自适应 mask 策略可能进一步提升。但简单的随机策略在效率和效果之间达到了良好的平衡。

4. **线性可分离性不足**：在 linear probing 上 MAE 低于对比学习方法（75.8% vs MoCo v3 的 77.6%）。但这不直接反映迁移能力——部分微调验证中 MAE 完胜 MoCo v3。

5. **3D/视频扩展**：原始 MAE 仅在静态图像上验证。将 masked modeling 扩展到视频（time dimension）需要额外的设计考量。

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关联 | 时间 |
|---------|------|------|
| **VideoMAE** | 将 MAE 扩展到视频，mask 时空 patches | 2022 |
| **MixMAE** | 混合 masked modeling + mixing strategies | 2022 |
| **AIM** | 将 MAE 扩展到多模态（音频-图像） | 2022 |
| **EVA** | ViT 用 MAE 风格预训练 + CLIP 蒸馏 | 2022 |
| **OpenVLA 等 VLA 系统** | MAE 预训练视觉编码器范式被继承 | 2024 |

MAE 的影响：**它确立了高 mask ratio + 非对称 Encoder-Decoder 作为视觉自监督预训练的主流设计范式**。后续很多工作将这一范式扩展到不同模态和任务。在 VLA 领域，MAE 验证了"大规模自监督预训练 + 下游微调"的数据效率，与 [[DINOv2]] 成为视觉编码器自监督预训练的两大主流方法。

与 [[BEiT]] 的对比：BEiT 预测 dVAE tokens（来自 DALL-E 预训练），而 MAE 预测像素。MAE 更简洁（无需外部 tokenizer），且效率更高（每 epoch 快 3.5×）。最终在大部分任务上 MAE 持平或优于 BEiT。

---

## 六、Implications for You / Hardware Compatibility

| 维度 | 评价 |
|------|------|
| 训练硬件要求 | ✅ 在 16GB GPU 上可训练 ViT-B（使用 gradient checkpointing）。推荐 32GB+ |
| 训练时间 | ✅ 极高效——仅需典型对比学习方法 1/3 的训练时间。ViT-L 800 epochs 在 8×A100 约 1 周 |
| 推理硬件 | ✅ Encoder 仅 0.3-2.0 GFLOPs（取决于 ViT 大小），可在任何现代 GPU/边缘设备上运行 |
| 代码复杂度 | ✅ 极简实现：无对抗训练、无动量编码器、无投影头。PyTorch 约 200 行 |
| 对 VLA 的意义 | ✅ 自监督视觉预训练基础工具，[[OpenVLA]] 等的视觉 backbone 可直接用 MAE 训练 |

**核心启示：**
1. **"简单"是最强大的武器**——MAE 极简到令人惊讶的程度，但这正是它成功的原因
2. **理解模态的固有特性**比照搬其他领域方法更重要——75% mask ratio 源于对图像空间冗余的深刻理解
3. **非对称不仅为效率，也为表现力**——轻量 Decoder + 重 Encoder 的结构约束本身就是一种正则化
4. **对于 VLA 研究**，MAE 预训练的视觉编码器可以作为强基线。与 [[DINOv2]] 相比，MAE 更轻量、训练更快，但在 frozen feature 质量上不如 DINOv2

---

## PDF

[[MAE 原文.pdf]]
