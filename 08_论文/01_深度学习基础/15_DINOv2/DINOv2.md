---
tags:
  - 论文
  - 自监督学习
  - 视觉特征
  - ViT
  - 蒸馏
  - 基础模型
created: 2026-06-30
paper_title: "DINOv2: Learning Robust Visual Features without Supervision"
paper_authors: "Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski"
paper_year: 2023
paper_venue: "TMLR 2024"
paper_citations: "~5,000+"
paper_url: "https://arxiv.org/abs/2304.07193"
github: "https://github.com/facebookresearch/dinov2"
---

# DINOv2

**DINOv2: Learning Robust Visual Features without Supervision**
*Meta AI Research + Inria | TMLR 2024 | arXiv: 2304.07193*

> **Pitch**: Meta AI 推出的自监督视觉基础模型，通过组合 DINO（对比学习）+ iBOT（masked image modeling）+ KoLeo（多样性正则化）三种损失，在 142M 精选图像上训练出 1.1B ViT-g 模型。其 frozen features 同时适用于图像级（分类）和像素级（分割、深度估计）任务，无需任何微调。OpenVLA 的**空间编码器**——与 SigLIP 语义编码器互补构成 VLA 的双视觉编码器范式。

---

## 一、Background / Core Idea

### 1.1 自监督视觉预训练的困境

自监督视觉预训练在 DINOv2 之前已经取得了显著进展（DINO、iBOT、MAE 等），但存在一个根本问题：**没有一个方法能同时适用于所有类型的视觉任务**。对比学习方法（DINO、MoCo v3）在图像级任务（分类、检索）上表现好，但在像素级任务（分割、深度估计）上特征质量不足。MAE 需要微调才能在下游任务取得好结果。强监督方法（CLIP、SigLIP）需要大量图文对数据，且特征的空间理解能力相对有限。

### 1.2 DINOv2 的核心洞见

DINOv2 的答案是：**将多种自监督目标的优点结合起来**，在更大、更干净的数据集上训练更大的模型，然后通过知识蒸馏产出不同规模的变体。

三个关键支柱：
1. **更好的训练目标组合**：DINO（全局对比）+ iBOT（局部 MIM）+ KoLeo（多样性正则）+ Sinkhorn-Knopp 居中
2. **更好的数据**：LVD-142M——从 1.2B 候选图像中通过自监督检索精选出的 142M 高质量图像
3. **更好的工程实现**：FlashAttention、sequence packing、FSDP、efficient stochastic depth 等加速技术

### 1.3 为什么单独一个目标不够

| 方法 | 优点 | 缺点 |
|------|------|------|
| DINO (对比学习) | 图像级特征好、语义分离 | 局部细节不足 |
| iBOT (MIM) | 像素级特征好 | 图像级表征弱于对比学习 |
| 两者组合 | 互补优势 | 需要额外正则化防坍塌 |

---

## 二、Method / Architecture / Technical Contribution

### 2.1 总体训练目标

$$\mathcal{L} = \mathcal{L}_{\text{DINO}} + \mathcal{L}_{\text{iBOT}} + \mathcal{L}_{\text{KoLeo}}$$

### 2.2 DINO 损失（图像级）

基于教师-学生框架的 self-distillation。用 cross-entropy loss 对齐 student 和 teacher 的 class token 输出：

$$\mathcal{L}_{DINO} = -\sum p_t \log p_s$$

- Student 和 teacher 网络架构相同，teacher 参数是 student 的 EMA（exponential moving average）
- 从同一图像的不同 crop 提取特征
- Student DINO head 输出 prototype scores → softmax → $p_s$
- Teacher DINO head 输出 → softmax + Sinkhorn-Knopp centering → $p_t$
- **关键改进**：使用 **Sinkhorn-Knopp（SK）** 替代 DINO 原始的平均移动中心化（moving average centering），该方法来自 SwAV。在 teacher 端运行 3 次 SK 迭代，student 端使用标准 softmax。

### 2.3 iBOT 损失（patch/像素级）

Masked Image Modeling（MIM）组件：随机 mask student 的部分输入 patch，用 teacher 对应位置的输出作为重建目标：

$$\mathcal{L}_{iBOT} = -\sum_i p_t^i \log p_s^i$$

- $i$ 遍历被 mask 的 patch indices
- 这与 [[MAE]] 类似，但重建目标不是像素而是 teacher 的 feature token
- 本质上是在 feature space 做 masked prediction

**关键改进**：将 DINO 和 iBOT 的 projection head 权重**解耦**（untie）。iBOT 原论文认为共享 head 更好，但 DINOv2 发现大规模训练时解耦效果更好，因此使用两个独立的 MLP heads。

### 2.4 KoLeo 正则化

来自 Kozachenko-Leonenko 微分熵估计器，鼓励 batch 内特征均匀分布（特征多样性）：

$$\mathcal{L}_{KoLeo} = -\frac{1}{n}\sum_{i=1}^n \log(d_{n,i})$$

其中 $d_{n,i} = \min_{j \neq i} ||x_i - x_j||$，即 $x_i$ 与 batch 中所有其他点的最小距离。计算前对特征做 ℓ2-normalization。

**效果**：KoLeo 在实例检索（Oxford-M）上提升 8% mAP，对其他指标无负面影响。它防止了特征坍塌，确保 batch 内特征分布均匀。

### 2.5 数据工程：LVD-142M

DINOv2 构建了一个自动化的数据筛选流水线（无需元数据或标注）：

1. **Curated data sources**：ImageNet-22k、ImageNet-1k、Google Landmarks、若干细粒度数据集
2. **Uncurated data**：从公开 web crawls 收集 1.2B 图像，经过 NSFW 过滤、人脸模糊、PCA hash 去重
3. **Self-supervised image retrieval**：
   - 用自监督 ViT-H/16（INet-22k 预训练）计算图像 embedding
   - 对 uncurated data 做 k-means 聚类
   - 对 curated dataset 的每张图像，从对应 cluster 检索 N=4 个最近邻
   - 使用 Faiss GPU 加速（20 节点 × 8 V100-32GB，不到 2 天）
4. **最终得到 LVD-142M**：142M 精选图像

数据质量 > 数据数量：在 LVD-142M（142M 精选）训练 vs 相同数量的 uncurated 图像（随机采样 142M），前者在几乎所有 benchmark 上明显更好。

### 2.6 模型架构与训练配置

**ViT-g/14（1.1B 参数）：**
- Embedding dim: 1536（不是原 ViT-g 的 1408——为了与 FlashAttention 的 64 倍数对齐）
- 24 heads, 64 dim/head
- 40 层 Transformer blocks
- Patch size 14（优先于 16，因为更多 token = 更好的密集预测性能）
- SwiGLU FFN（替代标准的 MLP）
- LayerScale + Stochastic Depth（drop rate=0.4）
- 128k prototypes

**训练效率提升（对比 iBOT）：**
- FlashAttention：自定义实现，比原版覆盖更多场景和硬件
- Sequence packing：将不同大小的 crop（224 + 98）的 token 序列拼接成单序列，用 block-diagonal attention mask 分离
- Efficient stochastic depth：跳过 dropped residuals 的计算而非 masking 结果
- FSDP（Fully-Sharded Data Parallel）：4 个 model replicas（student/teacher/optimizer ×2）共 16GB，shard 到多 GPU，通信用 fp16 节省 50%
- **总体效果**：比 iBOT 实现 **2× 更快，仅用 1/3 内存**

**高分辨率适应（resolution adaptation）：** 训练最后阶段将分辨率从 224 提升到 518 训练少量迭代（~10K），成本仅为全程高分辨率训练的 1/3，但性能接近。

### 2.7 知识蒸馏

DINOv2 不是让小模型从零训练，而是从训练好的 ViT-g 蒸馏：
- 使用同一个训练循环
- 大模型作为 frozen teacher
- 保留 student 的 EMA 作为最终模型
- 移除 masking 和 stochastic depth
- 在两个 global crop 上应用 iBOT loss

蒸馏的 ViT-L 在 12 个 benchmark 上全部优于从零训练的 ViT-L，有时甚至超过蒸馏目标（ViT-g）。

---

## 三、Experiments and Key Findings

### 3.1 ImageNet 线性探测

| Method | Arch | Data | Text sup? | Linear | kNN |
|--------|------|------|-----------|--------|-----|
| CLIP | ViT-L/14 | WIT-400M | ✅ | 84.3 | 79.8 |
| OpenCLIP | ViT-G/14 | LAION-2B | ✅ | 86.2 | 83.2 |
| MAE | ViT-H/14 | IN1K | ❌ | 76.6 | 49.4 |
| iBOT | ViT-L/16 | INet-22k | ❌ | 82.3 | 72.9 |
| **DINOv2 ViT-g/14** | **ViT-g/14** | **LVD-142M** | **❌** | **86.5** | **83.5** |

DINOv2 在纯自监督设置下超越所有弱监督方法（OpenCLIP-G 86.2%，EVA-CLIP 86.4%），达到 86.5%。在 ImageNet-V2（泛化测试）上领先更明显（78.4 vs EVA-CLIP 77.4）。

### 3.2 Dense Prediction：语义分割

**冻结特征 + 线性分类器（无需微调）：**

| Method | ADE-20k (linear) | ADE-20k (+ms) |
|--------|-----------------|---------------|
| OpenCLIP ViT-G/14 | 45.5 | 51.7 |
| iBOT ViT-L/16 | 47.1 | - |
| **DINOv2 ViT-g/14** | **53.0** | **57.9** |

DINOv2 的冻结特征 + 简单 linear layer 即可达到 MAE 用 UperNet 全微调的性能（53.0 vs 53.6 mIoU）。接入 ViT-Adapter + Mask2former head（冻结 backbone）可达 60.2 mIoU，接近 SOTA（62.9）。

### 3.3 Dense Prediction：单目深度估计

| Method | NYUd (lin.4) | KITTI (lin.4) |
|--------|-------------|---------------|
| OpenCLIP ViT-G/14 | 0.510 | 3.21 |
| iBOT ViT-L/16 | 0.387 | 3.07 |
| **DINOv2 ViT-g/14** | **0.298** | **2.35** |

DINOv2 在深度估计上远超自监督和弱监督方法。特征甚至能在 NYUd → SUN RGB-D 零样本迁移（lin.4: 0.362），验证了特征的跨域泛化能力。

### 3.4 消融实验

| 组件消融 | k-NN (IN1K) | Linear (IN1K) |
|---------|-------------|---------------|
| iBOT baseline | 72.9 | 82.3 |
| +LayerScale, Stochastic Depth | 75.4 | 82.0 |
| +128k prototypes | 76.6 | 81.9 |
| **+KoLeo** | **78.9** | **82.5** |
| +SwiGLU FFN | 78.7 | 83.1 |
| +Patch size 14 | 78.9 | 83.5 |
| +Batch size 3k | 81.7 | 84.7 |
| +Sinkhorn-Knopp | 81.7 | 84.7 |
| +Untying heads = **DINOv2** | **82.0** | **84.5** |

KoLeo 是单次提升最大的组件（+2.3% k-NN），batch scaling 贡献巨大（+1.2% k-NN +0.9% linear）。

### 3.5 KoLeo 与 MIM 的消融

| 消融 | INet-1k | Im-A | ADE-20k | Oxford-M |
|------|---------|------|---------|----------|
| w/o KoLeo | 85.3 | 70.6 | 47.2 | 55.6 |
| **w/ KoLeo** | **85.8** | **72.8** | 47.1 | **63.9** |
| w/o MIM | 85.3 | 72.0 | 44.2 | 64.3 |
| **w/ MIM** | **85.8** | **72.8** | **47.1** | 63.9 |

KoLeo 对检索（Oxford-M）提升显著（+8.3 mAP）。MIM 对分割（ADE-20k）提升显著（+2.9 mIoU），因为像素级任务需要局部细节。

### 3.6 域泛化

| Method | Im-A | Im-R | Im-C↓ | Sketch |
|--------|------|------|-------|--------|
| OpenCLIP ViT-G/14 | 63.8 | 87.8 | 45.3 | 66.4 |
| iBOT ViT-L/16 | 41.5 | 51.0 | 43.9 | 38.5 |
| **DINOv2 ViT-g/14** | **75.9** | **78.8** | **28.2** | **62.5** |

DINOv2 在域外泛化上表现突出，尤其是在自然对抗样本（ImageNet-A）上超越弱监督方法 12 个百分点。

---

## 四、Limitations and Challenges

1. **预训练计算成本极高**：ViT-g（1.1B）训练需要大量 GPU 资源（预估 ~2000 GPU-days），大多数实验室无法复现。不过 DINOv2 提供了蒸馏后的中小模型。

2. **特征维度固定**：输出特征维度不可调整（依赖 ViT patch size），某些场景过杀或不足。Patch size 14 在分类上可能不如更小的 patch 精细。

3. **缺少语言对齐**：纯视觉模型，不包含语言理解，需要配合 [[SigLIP]]、[[CLIP]] 等语言对齐模型使用。这是它与 CLIP 相比的根本差距。

4. **地理偏差**：模型在非洲地区性能比欧洲低 25.7%，低收入地区比高收入低 31.7%，反映出训练数据的分布偏差（尽管 LVD-142M 做了精选）。

5. **特征虽强但仍是"冻结的"**：在 VLA 等下游任务中，DINOv2 frozen features 虽然很强，但微调仍然带来显著提升（+2-9%），说明并非完全 task-agnostic。

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 关联 | 时间 |
|---------|------|------|
| **OpenVLA** | 双编码器（DINOv2 + SigLIP），DINOv2 负责空间特征 | 2023 |
| **FLOWER** | 继承双视觉编码器设计，验证范式有效性 | 2024 |
| **Cosmos** | 使用类似的自监督视觉编码器 | 2025 |
| **视觉基础模型系列** | DINOv2 开启了"自监督视觉基础模型"范式 | 2023-2025 |

DINOv2 的**双视觉编码器范式**（空间 + 语义）被 [[OpenVLA]]、[[FLOWER]] 等大量验证：一个模型理解"是什么"（[[SigLIP]]），一个模型理解"在哪里"（DINOv2）——这已成为 VLA 视觉处理的标配。

DINOv2 vs MAE：DINOv2 的 frozen features 质量远高于 MAE（linear probing 86.5% vs 76.6%），因此对于需要 frozen backbone 的场景（例如 VLA 的视觉编码器），DINOv2 是更好的选择。但 MAE 训练更轻量，且微调后差距缩小。

与 [[SigLIP]] 的对比：SigLIP 在分类和语义理解任务上更强，但 DINOv2 在分割、深度等密集预测任务上显著优于 SigLIP（视觉上，SigLIP 的分割 mask 有许多伪影和断裂组件，DINOv2 则平滑准确）。

---

## 六、Implications for You / Hardware Compatibility

| 维度 | 评价 |
|------|------|
| 训练硬件要求 | ❌ ViT-g 训练需要大量 TPU/A100 集群，不可复现。ViT-S/B 约 32GB VRAM |
| 推理硬件 | ✅ ViT-g 推理 ~3-4GB VRAM，ViT-L ~1.5GB，16GB GPU 完全可运行 |
| frozen features 使用 | ✅ 可直接冻结使用，classification / segmentation / depth 各自训练简单 linear head（极小计算量） |
| 对 VLA 的意义 | ✅ **核心视觉编码器**——理解 DINOv2 是理解 [[OpenVLA]]、[[FLOWER]] 架构的前提 |
| 蒸馏模型可用性 | ✅ Meta 开源了 ViT-S/B/L/g 全部模型，可直接使用 |

**核心启示：**
1. **组合优于单一**：三种损失的组合是 DINOv2 成功的关键——对比学习处理全局语义，MIM 保障局部细节，KoLeo 防止特征坍塌
2. **视觉编码器是 VLA 的基石**：[[OpenVLA]] 使用 DINOv2 后在 7B 规模超越 55B 的 [[RT-2-X]]，证明视觉编码器的质量对 VLA 整体性能有决定性影响
3. **空间 vs 语义是互补维度**：DINOv2 + SigLIP 揭示了一个通用原则——VLA 需要同时捕获"什么东西"（语义）和"在哪里"（空间）
4. **大规模蒸馏是实用策略**：用 ViT-g 蒸馏出小模型，保持质量同时大幅降低部署成本

---

## PDF

[[DINOv2 原文.pdf]]
