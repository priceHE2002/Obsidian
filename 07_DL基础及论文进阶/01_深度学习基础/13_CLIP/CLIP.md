---
tags:
  - 论文
  - 多模态
  - 视觉-语言
  - 对比学习
  - 零样本
  - CLIP
created: 2026-06-30
paper_title: "Learning Transferable Visual Models From Natural Language Supervision"
paper_authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever"
paper_year: 2021
paper_venue: "ICML 2021"
paper_citations: "~30,000+"
paper_url: "https://arxiv.org/abs/2103.00020"
github: "https://github.com/OpenAI/CLIP"
---

# CLIP

**Learning Transferable Visual Models From Natural Language Supervision**
*OpenAI | ICML 2021 | arXiv: 2103.00020*

> 连接视觉和语言的关键桥梁。在 400M 互联网图文对上用对比学习训练，使模型学会将图像和文本映射到同一语义嵌入空间。CLIP 的视觉编码器成为 VLA 视觉 backbone 的事实标准——[[DINOv2|SigLIP]]（CLIP 的改进版）是 OpenVLA 和 π0 的核心视觉编码器。CLIP 也首次展示了零样本图像分类与有监督模型相当的能力，以及对分布偏移的显著鲁棒性。

---

## 一、Background / Core Idea

### 1.1 传统视觉模型的局限：固定类别标签

2021 年之前，计算机视觉的标准范式是：在固定类别集（如 ImageNet 1000 类）上训练，输出 P(y|x) 的条件概率分布。这个范式有三个根本局限：

1. **封闭类别集**：模型只能区分训练时见过的类别，无法识别"新"概念
2. **标注成本高**：每个新任务都需要从零收集和标注数据
3. **泛化性差**：模型学到的只是区分 1000 个特定类别的特征，而非通用的视觉概念

原论文指出："State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability."

### 1.2 自然语言作为监督信号

**核心洞察**：自然语言远比固定类别标签包含更丰富的信息。互联网上存在数以亿计的图文对——图片搭配描述文本（如"一只金毛叼着红色飞盘"）。这些文本描述包含了物体、动作、空间关系、属性等远比"golden retriever"单个标签丰富的语义。

CLIP 的目标是：直接从自然语言文本中学习视觉概念，使得模型可以**用语言作为接口**来执行任意视觉分类任务。

### 1.3 为什么对比学习而不是生成式？

作者在训练方法上做了关键的效率权衡（原论文 Figure 2）：

1. **生成式方法（预测完整文本）**：训练图像编码器 + 文本解码器生成图像描述。这种方法虽然可以学到丰富的表示，但效率极低——生成式目标需要模型预测文本中的每个词，包括大量对视觉理解不重要的内容（如语法词、修饰语）
2. **对比方法（预测图文对是否匹配）**：只需区分"这批图文对中，哪张图配哪段文字"。这是一个更为简单但同样有效的代理任务，效率比生成式高约 4 倍

### 1.4 与零样本学习的联系

此前 Li et al. (2017) 的 Visual N-Grams 尝试了类似的零样本分类，但性能很低（ImageNet 仅 11.5%）。CLIP 将其大幅提升到 76.2%——这是量变到质变的跃迁，核心原因是：更大的模型、更多的数据、更优的训练目标。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 对比学习框架

CLIP 的核心框架（原论文 Figure 1 和 3）极为简洁：双塔架构（Dual-Encoder），将图像和文本映射到共享嵌入空间。

对于一个批次中的 N 个图文对（I₁,T₁),...,(I_N,T_N)：

1. 图像编码器将图像编码为 I_e ∈ ℝ^{N×d_e}
2. 文本编码器将文本编码为 T_e ∈ ℝ^{N×d_e}
3. 计算 N×N 的余弦相似度矩阵
4. 对比损失：最大化 N 个正对的相似度，最小化 N²-N 个负对

**对称 InfoNCE Loss**：

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^N \Bigg[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(I_j, T_i)/\tau)}\Bigg]$$

其中 sim(I, T) = I·T / (∥I∥·∥T∥)（余弦相似度），τ 是可学习的温度参数。

**两部分损失的含义**：
- 第一项（图像→文本方向）：对每张图，从 N 个文本中找出配对的文本
- 第二项（文本→图像方向）：对每段文本，从 N 张图中找出配对的图像

对称性确保了嵌入空间中双向的对齐。

### 2.2 温度参数 τ

温度 τ 是一个可学习的标量参数（初始值经指数变换后控制相似度分布的 sharpness）：
- τ 越小：softmax 分布越尖锐（只有最匹配的对获得高概率）
- τ 越大：分布越平滑（更多的负对也获得一定概率）

作者选择了可学习的 τ，让模型自动在训练的锐度-平滑度之间权衡。代码中实际实现为 logits = np.dot(I_e, T_e.T) * np.exp(t)，其中 t 是学习参数。

### 2.3 模型规模：图像编码器

CLIP 同时探索了 ResNet 和 ViT 两种图像编码器：

**ResNet 基**：
- 使用改进的 ResNet（ResNet-50, ResNet-101, ResNet-50x4, ResNet-50x16, ResNet-50x64）
- 改进点：使用 ResNet-D（改进的 stem）、添加抗锯齿模糊池化、用**注意力池化**替代全局平均池化
- 扩展策略：同时增加宽度、深度和分辨率（类似 EfficientNet 方法）
- 最大的 ResNet-50x64 相当于 3.05 倍宽度、4 倍深度、3.15 倍分辨率的缩放

**ViT 基**：
- 使用 ViT-B/32, ViT-B/16, ViT-L/14
- 在原 ViT 基础上添加了额外的 LayerNorm（patch + position embedding 后）
- 最大的 ViT-L/14 是最强的 CLIP 模型

### 2.4 模型规模：文本编码器

文本编码器使用 [[GPT|GPT-2]] 风格的 Transformer 架构：

| 参数 | 数值 |
|------|------|
| 层数 | 12 |
| 宽度 | 512 |
| Attention Heads | 8 |
| 参数量 | 63M |
| 词汇表 | BPE，49,152 |
| 最大序列长度 | 76 |
| Attention 类型 | Masked Self-Attention（保留语言建模潜力） |

文本序列被 [SOS] 和 [EOS] token 包裹，[EOS] token 的最高层激活经 LayerNorm 和线性投影后作为文本特征表示。

### 2.5 WIT 数据集（WebImageText）

| 属性 | 数值 |
|------|------|
| 总图文对数 | 400M |
| 数据来源 | 互联网（通过 500,000 个查询词条收集） |
| 查询构造 | Wikipedia 高频率文章标题 + WordNet 同义词集 + 高 PMI 二元组 |
| 每查询上限 | 20,000 对（以平衡分布） |
| 文本总词数 | 与 GPT-2 的 WebText 相当 |
| 训练曝光 | 32 epochs = 12.8B 样本 |
| Batch size | 32,768 |

WIT 的数据规模（400M pairs）是此前工作的数倍——比如 Visual N-Grams 使用的 YFCC100M 只有约 100M 有文本的图像。**这是 CLIP 成功的关键因素之一**。

### 2.6 训练配置

- **硬件**：592 块 V100 GPU（ResNet-50x64），256 块 GPU（ViT-L/14）
- **优化器**：Adam
- **Batch size**：32,768（大 batch 提供大量负样本）
- **训练时间**：约 12 天（ResNet-50x64），更长时间（ViT-L/14）
- **精度**：混合精度训练
- **梯度检查点**：减少内存

大 batch size（32,768）的选择至关重要——对比学习的效果高度依赖于负样本的数量和质量，更多的负样本意味着更快的收敛和更好的最终性能。

### 2.7 Prompt Engineering 与集成

**为什么需要 prompt？**

标准图像分类数据集通常只给类别名称（如"dog", "cat"），这对零样本分类不够。CLIP 需要对每个类别构造完整的 prompt：

基线："a photo of a {label}"
改进：对 Fine-grained 任务添加上下文，如
- Oxford-IIIT Pets："A photo of a {label}, a type of pet."
- Food101："A photo of {label}, a type of food."
- OCR：用引号包裹文本（如换行版）→ 提升 1.4 倍
- 卫星图像："a satellite photo of a {label}."

**Prompt 集成**：对每个类别使用 80 种不同的 prompt 模板，在嵌入空间平均后再做分类。这比在概率空间平均更高效（可缓存一个平均嵌入）。

原论文 Figure 4 展示了 prompt engineering + ensembling 的效果：对 ResNet-50 从 ~60% 提升到 ~64%（4 分），对更大的模型提升更明显。

---

## 三、Experiments and Key Findings

### 3.1 零样本 ImageNet 分类——核心结果

| 模型 | ImageNet Top-1 | 需要 ImageNet 训练？ |
|------|---------------|-------------------|
| ResNet-50 (有监督) | 76.3 | 是 |
| **CLIP ViT-L/14 (零样本)** | **76.2** | **否** |
| ResNet-101 (有监督) | 77.4 | 是 |
| BiT-M (200M) | 87.5 | 是 |

**CLIP 零样本性能与有监督 ResNet-50 几乎持平（76.2% vs 76.3%）**，但从未见过任何 ImageNet 训练样本——这是通过 "A photo of a {class}" 的文本 prompt 实现的。

这标志着视觉零样本分类的能力跃迁——从 Visual N-Grams 的 11.5% 提升到 CLIP 的 76.2%。

### 3.2 零样本分类——30+ 数据集的广泛评估

| 数据集 | 任务类型 | CLIP ViT-L/14 | 此前最佳 | 差距 |
|--------|---------|---------------|---------|------|
| ImageNet | 通用分类 | 76.2 | 88.4 (SOTA) | -12.2 |
| CIFAR-10 | 通用分类 | 95.6 | 99.5 | -3.9 |
| CIFAR-100 | 通用分类 | 79.3 | 93.5 | -14.2 |
| Oxford Pets | 细粒度 | 89.1 | 97.6 | -8.5 |
| Caltech-101 | 通用分类 | 92.7 | 98.3 | -5.6 |
| Food-101 | 细粒度 | 89.2 | 96.5 | -7.3 |
| SUN397 | 场景识别 | 68.5 | 83.1 | -14.6 |
| Stanford Cars | 细粒度 | 60.9 | 95.8 | -34.9 |
| DTD (纹理) | 纹理分类 | 47.5 | 74.0 | -26.5 |
| EuroSAT | 卫星图像 | 48.3 | 99.0 | -50.7 |
| Country211 | 地理定位 | 33.7 | — | — |
| KITTI | 驾驶场景 | 49.6 | — | — |
| MNIST | 数字识别 | 73.4 | 99.8 | -26.4 |

CLIP 在通用或自然图像数据集上表现突出，但在细粒度分类、纹理、卫星图像等专业领域上的零样本性能仍显著不如有监督模型。

**线性探测（线性分类器在提取的特征上训练）的结果更好**：在 ImageNet 上达到 85.4%（ResNet-50x64），超越当时的众多有监督模型。这说明 CLIP 的视觉表示质量很高，零样本性能受限于 prompt 工程而非表示本身。

### 3.3 分布偏移鲁棒性——CLIP 最重要的发现之一

原论文 Section 3.3 和 Figure 7 展示了 CLIP 最引人注目的特性：**对自然分布偏移的显著鲁棒性**。

由作者整理的 ImageNet 变体测试集结果：

| 测试集 | 说明 | CLIP ViT-L/14 (零样本) | ResNet-101 (有监督) | 差距 |
|--------|------|----------------------|--------------------|------|
| ImageNet | 标准集 | 76.2 | 77.4 | -1.2 |
| ImageNet-A | 自然对抗样本 | **77.1** | **2.4** | **+74.7** |
| ImageNet-R | 渲染/艺术画 | **88.9** | 36.1 | **+52.8** |
| ImageNet-Sketch | 素描 | **60.2** | 25.1 | **+35.1** |
| ImageNet-V2 | 复刻版 | 70.0 | 70.2 | -0.2 |
| ObjectNet | 视角/背景变化 | **51.4** | 15.0 | **+36.4** |

**在 ImageNet-A 上，CLIP 的零样本准确率 77.1% 对比 ResNet-101 的 2.4%。** 这不是"更好一点点"的问题——而是天壤之别。在有监督时代被认为"不可克服"的自然对抗样本，CLIP 几乎完美地解决了。

**为什么 CLIP 对分布偏移如此鲁棒？**

作者提出了一个重要假设：CLIP 的零样本评估方式（非任务特定训练）避免了监督学习的"捷径学习"问题——有监督模型会利用训练数据中的表面特征（如纹理、背景），而不是理解物体的核心视觉概念。

### 3.4 Few-shot 学习

CLIP 的零样本性能与 BiT-M 的 16-shot（每类 16 个样本）性能相当。给予少量样本微调后，CLIP 可以快速超过 SOTA。

### 3.5 表示学习分析

线性探测结果：

| 模型 | ImageNet Top-1 | 计算量 (GFLOPs) |
|------|---------------|----------------|
| ResNet-50 (有监督) | 76.3 | 4.1 |
| CLIP ResNet-50 (线性探测) | 76.7 | 4.1 |
| ResNet-101 (有监督) | 77.4 | 7.9 |
| CLIP ResNet-101 (线性探测) | 78.0 | 7.9 |
| Noisy Student EfficientNet-L2 | 88.4 | — |
| CLIP ResNet-50x64 (线性探测) | **85.4** | — |

CLIP ResNet-50 的线性探测结果超过有监督 ResNet-50，说明 CLIP 的表示质量更高。

---

## 四、Limitations and Challenges

### 4.1 细粒度理解不足

CLIP 只做全局的图文匹配，缺乏细粒度的空间/位置理解能力。例如，CLIP 可以区分"狗"和"猫"，但不知道"狗坐在猫的右边"这样的空间关系。这限制了其在需要精确空间推理的任务上的表现。

### 4.2 专业领域零样本性能差

在医学影像（如 ChestX-ray）、遥感图像（如 EuroSAT）、纹理分类（如 DTD）等专业领域，CLIP 的零样本性能显著低于有监督模型。原因是在预训练数据中这类图像较少，且描述文本不够专业。

### 4.3 抽象/组合理解能力弱

CLIP 对复杂组合概念（"一个蓝色的球在红色盒子的右边"）的理解能力有限。对比学习目标只要求全局匹配，不要求理解语义组合。这启发了后续的组合零样本学习（Compositional Zero-shot Learning）研究。

### 4.4 数据污染问题

WIT 数据集来自互联网，可能与下游测试集存在重叠。原论文未做严格的数据去重分析，这意味着 CLIP 的零样本性能可能被高估（部分 ImageNet 图像在 WIT 中以某种形式出现过）。

### 4.5 训练成本极高

ViT-L/14 的训练需要 256 块 GPU，普通实验室无法复现。这也解释了为什么 CLIP 的改进工作（如 SigLIP, OpenCLIP）主要由大型机构推动。

### 4.6 对比学习的限制

对比学习只关心"匹配/不匹配"，不学习文本的生成能力——CLIP 的文本编码器不能做文本生成。这与生成式方法（CoCa, SimVLM）形成对比，后者同时学习视觉表示和语言生成。

---

## 五、Relationship with Subsequent Work / Impact on the Field

CLIP 是视觉-语言多模态学习的里程碑，以 CLIP 为起点形成了完整的 VLM 研究树：

| 方向 | 代表性工作 | 改进内容 | VLA 角色 |
|------|-----------|---------|----------|
| 更高效的对比学习 | **SigLIP** (Zhai et al., 2023) | Sigmoid loss 替代 InfoNCE，无需大 batch size | OpenVLA 的视觉编码器 |
| 更大数据 | ALIGN (Jia et al., 2021) | 1.8B 图文对，更弱的数据过滤 | — |
| 生成+对比混合 | CoCa (Yu et al., 2022) | 对比学习 + 字幕生成联合训练 | — |
| 自监督视觉 | [[../15_DINOv2/DINOv2.md|DINOv2]] (Oquab et al., 2023) | CLIP + 自蒸馏，无需文本标注 | OpenVLA 的第二视觉编码器 |
| VLM 多模态 | LLaVA (Liu et al., 2023) | CLIP 视觉 + Llama 语言 → 对话式 AI | VLA 视觉语言对齐的基础 |
| 文生图 | DALL·E 2, Stable Diffusion | CLIP 文本嵌入作为条件输入 | 扩散模型的文本条件 |

**CLIP 在 VLA 中的角色——核心且直接**：

1. **SigLIP（CLIP 改进版）是 OpenVLA 和 π0 的视觉编码器核心组件**：OpenVLA 使用 SigLIP + [[DINOv2|DINOv2]] 双视觉编码器，其中 SigLIP 提供了语义对齐能力
2. **"视觉和语言共享嵌入空间"是 VLA 多模态融合的基础**：没有 CLIP 的思想，就无法让模型理解"拿起杯子"这个文本指令对应什么样的视觉场景
3. **CLIP 预训练权重广泛用于初始化 VLA 的视觉编码器**：由于 VLA 的机器人数据有限，使用在 400M 图文对上预训练的 CLIP 视觉编码器可以大幅减少微调需求
4. **零样本对齐能力对开放世界机器人至关重要**：机器人在训练中遇到的环境/物体/光照与训练集总是不同的——CLIP 的开放世界理解能力相比有监督模型有天然优势
5. **零样本提示工程在 VLA 中的应用**：VLA 的语义任务（如图像描述、指令理解）需要将机器人场景作为"prompt"来适配视觉编码器

---

## 六、Implications for You / Hardware Compatibility

- ✅ **对比学习是连接不同模态最高效的方式**：不生成文本、只看匹配/不匹配——这个简单策略在连接视觉和语言上如此有效，启示我们在连接视觉、语言、动作三种模态时也优先考虑对比学习
- ✅ **不要低估分布偏移——这在机器人的视觉感知中是常态**：CLIP 最重要的启示是——有监督学习对分布偏移极其脆弱，而大规模对比预训练天然更鲁棒。在机器人应用中（训练环境 vs 真实环境去始终不同），这是关键优势
- ⚠️ **CLIP ResNet-50x4 和 ViT-B/16 可在单 24GB GPU 上推理**：但 ViT-L/14 需要更多显存或使用半精度。完整训练 CLIP 需要大量 GPU（256+），不可行。建议直接使用预训练权重（Hugging Face `openai/clip-vit-large-patch14`）
- ✅ **使用 SigLIP 而非原始 CLIP**：SigLIP 的 Sigmoid loss 不需要大 batch size（更高效、更稳定），并且 OpenVLA 和 π0 都使用 SigLIP。建议直接使用 SigLIP 作为 VLA 的视觉编码器
- ✅ **数据覆盖比模型大小更重要**：CLIP 的核心是 400M 图文对的数据规模和质量。在机器人场景中，数据覆盖（各种场景、光照、视角、物体、背景）比模型复杂度更能决定泛化能力
- ⚠️ **CLIP 零样本不适用于精细操纵任务**：如果 VLA 需要精确的空间推理（如"在蓝色杯子里放红色方块"），CLIP 的全局匹配能力不足。需要 [[DINOv2|DINOv2]] 这样的 patch-level 编码器来补充空间理解

## PDF

[[CLIP 原文.pdf]]
