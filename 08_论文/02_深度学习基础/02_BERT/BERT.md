---
tags:
  - 论文
  - 预训练
  - 双向注意力
  - NLP
created: 2026-06-30
paper_title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
paper_authors: "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova"
paper_year: 2018
paper_venue: "NAACL 2019"
paper_citations: "~100,000+"
paper_url: "https://arxiv.org/abs/1810.04805"
github: "https://github.com/google-research/bert"
---

# BERT

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
*Google AI Language | NAACL 2019 | arXiv 1810.04805*

> "预训练+微调"范式的里程碑。提出 Masked Language Model (MLM) + Next Sentence Prediction (NSP) 两项预训练任务，在 11 项 NLP 任务上全面刷新 SOTA，证明了**双向上下文信息**对语言理解的关键作用。虽然不是 VLA 模型直接采用的架构，但 MLM 启发了 [[../14_MAE/MAE.md|MAE]]，segment embedding 启发了多模态 token 编码，而"预训练+微调"范式是所有现代 VLA 训练流程的基础。

---

## 一、Background / Core Idea

### 1.1 预训练语言表示的两条路线

在 BERT 之前，预训练语言表示分为两条路线（原论文 Section 1）：
1. **基于特征（feature-based）**：如 ELMo，使用任务特定架构，将预训练表示作为额外特征。ELMo 双向拼接了从左到右和从右到左的 LSTM 表示，但两端的信息只在浅层融合，不是真正的深层双向
2. **基于微调（fine-tuning）**：如 OpenAI [[../03_GPT/GPT.md|GPT]]，引入最少任务特定参数，在预训练后对整个模型进行微调。但 GPT 使用单向（从左到右）语言模型，限制了对双向上下文的理解能力

两条路线共享相同的预训练目标（单向语言建模），这成为它们的根本局限。

### 1.2 核心问题：为什么单向预训练不够？

原论文明确指出："The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training."

具体来说：
- 在 NER 任务中，判断"华盛顿"是人名还是地名需要看上下文两侧的信息
- 在 SQuAD 问答中，判断答案在文档中的跨度需要同时考虑问题文本和答案前后的内容
- GPT 虽然通过 Transformer 可以看到最多 512 个 token 的上下文，但只能是**左侧**的

传统的语言建模目标（预测下一个 token）天然是单向的——如果允许模型看到右侧信息，预测将退化为复制任务。因此需要一个**新的预训练目标**来实现双向表示。

### 1.3 关键洞察：Cloze 任务作为预训练目标

BERT 的核心洞察：将传统语言建模替换为**"完形填空"（Cloze）任务**——随机遮盖输入中的部分 token，让模型预测被遮盖的词。由于模型看不到被遮盖的词本身，即使有双向上下文也不会导致信息泄露。这实际上就是在 Transformer Encoder（双向 Self-Attention）上做去噪表示学习。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 模型架构：Multi-layer Bidirectional Transformer Encoder

BERT 使用 [[../01_Attention_Is_All_You_Need/Attention Is All You Need.md|Transformer]] 的 Encoder 部分（仅 Encoder），包含两个配置：

| 配置 | 层数 (L) | Hidden Size (H) | Attention Heads (A) | 参数量 |
|------|---------|----------------|--------------------|--------|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |

两种配置的 FFN 维度均为 4H（即 BERT-base 为 3072，BERT-large 为 4096），沿用 Transformer 的设计。BERT-base 与 OpenAI GPT 的模型配置几乎完全相同（12 层，768 维，12 个头），唯一的区别是注意力掩码——BERT 使用双向注意力，GPT 使用从左到右的因果注意力。

### 2.2 输入表示：Token + Segment + Position Embeddings

这是 BERT 对 Transformer 的重要贡献之一——设计了一个能统一表示单句和句子对（<Question, Answer>）的输入方案。原论文 Figure 2 展示了输入表示的具体构成：

输入序列格式：`[CLS] token1 token2 ... [SEP] sentence B tokens [SEP]`

每个 token 的输入表示由三部分求和得到（原论文 Figure 2）：
1. **Token Embeddings**：WordPiece 分词，词汇表 30,000 个 token
2. **Segment Embeddings**：可学习的分段嵌入，A/B 句用不同嵌入（区分上下句）
3. **Position Embeddings**：可学习的位置编码，最大序列长度 512

关键设计：
- `[CLS]` token 借鉴自 Transformer 分类任务——其最终隐藏状态作为整个序列的聚合表示用于分类。原始 Transformer 用类似方式处理分类
- `[SEP]` token 分隔不同句子
- Segment Embeddings 的思路被多模态模型继承，用于区分图像 token 和文本 token（如 LLaVA 中用不同 segment 表示不同的模态输入）

### 2.3 预训练任务 1：Masked Language Model (MLM)

**为什么选 15%？**

对于每个输入序列，随机选择 15% 的 WordPiece token 进行遮盖。这个比例是权衡的结果：
- 比例太低（如 5%）：预训练信号不足，模型收敛慢
- 比例太高（如 25%+）：破坏太多上下文信息，难以学习有效的表示

**80%-10%-10% 策略：缓解预训练-微调 mismatch**

选中的 token 并非全部替换为 `[MASK]`，而是：
- **80%** 替换为 `[MASK]`
- **10%** 替换为随机词
- **10%** 保持不变

原因：`[MASK]` token 在微调阶段不存在，如果预训练时只用 `[MASK]`，模型会对这个 token 产生特定的响应模式，但微调时却永远不会看到 `[MASK]`（distribution mismatch）。通过引入随机替换和不改变，迫使模型保持对每个输入 token 的上下文理解能力，而不依赖 `[MASK]` 作为"需要预测的标记"。

**效率问题**：MLM 只预测 15% 的 token，因此比标准的从左到右语言建模需要更多训练步数才能收敛。BERT-base 在 4 块 Cloud TPU 上训练了 4 天（约 1M 步数）。这是 MLM 的一个根本效率瓶颈。

### 2.4 预训练任务 2：Next Sentence Prediction (NSP)

从语料中构建训练样本：50% 正例（实际相邻的句子对），50% 负例（从语料中随机抽取的句子对）。模型通过 `[CLS]` token 的表示来预测两句话是否连续。

**为什么需要 NSP？**

许多下游任务需要理解句子间的关系——问答（问题-文档）、蕴含（前提-假设）、对话历史理解。纯 MLM 只在 token 级别操作，缺乏句子级别的理解。NSP 为模型提供了跨句的语义关系学习信号。

**为何被废弃？**

RoBERTa (Liu et al., 2019) 发现移除 NSP 后，在大部分 GLUE 任务上性能反而小幅提升。可能原因：
1. NSP 任务相对简单——判断"是否是相邻句子"可以通过话题词重叠、命名实体等表面特征完成
2. NSP 的负例（随机抽取）太过简单，模型学习的是"话题是否一致"而非"逻辑是否连续"
3. 更大的 batch size 和更多的训练数据（RoBERTa 的改进）可能使 MLM 本身就能学到句子级别的表示

ALBERT 提出的 SOP（Sentence Order Prediction）——判断两个连续句子的顺序是否正确——是一个更有意义的改进。

### 2.5 预训练数据

| 数据源 | 规模 | 特点 |
|-------|------|------|
| BooksCorpus | 8 亿词 | 约 11,000 本未出版书籍，包含长程上下文 |
| English Wikipedia | 25 亿词 | 仅提取文本段落，省略列表/表格/标题 |

原论文强调："It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark in order to extract long contiguous sequences." 文档级语料（包含段落内连续句子）对于训练模型理解长程依赖关系至关重要。而 Billion Word Benchmark 的句子被随机打乱，破坏了这种结构。

### 2.6 微调范式

BERT 的微调极为简洁（原论文 Section 3.2）：

- **分类任务**：将 `[CLS]` 的最终隐藏状态送入一个简单的分类层
- **序列标注**：将每个 token 的最终隐藏状态送入分类层
- **问答**：预测答案跨度的起始和结束位置，输入格式为 [CLS] 问题 [SEP] 文档 [SEP]

核心优势：**"We plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end."** 不需要为每个任务设计新的网络架构。

微调成本：原论文称所有结果可以在 1 小时内（单块 Cloud TPU）或数小时内（GPU）复现。

---

## 三、Experiments and Key Findings

### 3.1 GLUE 结果

GLUE 基准测试包含 9 项自然语言理解任务。BERT 在所有任务上的结果（原论文 Table 1）：

| 系统 | MNLI-(m/mm) | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | 平均 |
|------|-------------|-----|------|-------|------|-------|------|-----|------|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| **BERT-base** | **84.6/83.4** | **71.2** | **90.5** | **93.5** | **52.1** | **85.8** | **88.9** | **66.4** | **79.6** |
| **BERT-large** | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

关键发现：
- BERT-base（双向注意力）在几乎所有任务上超过 GPT（单向注意力），尽管模型架构几乎完全相同——**证明了双向上下文的核心价值**
- CoLA（语言可接受性）提升巨大，BERT-large 比 GPT 高 15.1 分——这是对语言学知识要求最高的任务
- BERT-large 在 MNLI 上比此前最佳提升 4.6 个百分点
- BERT-large 在 GLUE 官方排行榜上得分 80.5，GPT 为 72.8

### 3.2 SQuAD 结果

| 模型 | SQuAD v1.1 EM/F1 | SQuAD v2.0 EM/F1 |
|------|------------------|------------------|
| 此前最佳 (Dec 2018) | 85.8/91.7 (nlnet) | 71.4/74.9 (unet) |
| **BERT-large (单模型)** | **84.1/90.9** | **78.7/81.9** |
| **BERT-large (集成)** | **85.8/91.8** | — |
| **BERT-large (Ensemble+TriviaQA)** | **87.4/93.2** | — |

在 SQuAD v2.0（包含无答案检测）上，BERT-large 以 83.1 F1 绝对领先，比此前最佳提升 5.1 F1。SQuAD v2.0 的解决方案也很优雅：将无答案情况表示为 `[CLS]` token（起始和结束都在 `[CLS]`），通过与最佳跨度分数比较来决定是否回答。

### 3.3 消融实验（Section 5.2）

| 实验 | MNLI-m | QNLI | MRPC | SST-2 | SQuAD |
|------|--------|------|------|-------|-------|
| BERT-base | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| 去除 NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| 单向 LM (GPT-style) | 82.1 | 87.4 | 82.3 | 91.3 | — |
| BiLSTM + ELMo | 76.4 | 79.8 | 84.9 | 90.4 | — |

结论：
1. NSP 的影响：去除 NSP 后各任务下降 0.5-3.5 分，QNLI（问答蕴含）受影响最大
2. 双向 vs 单向：BERT-base (双向) 在 MNLI 上比 GPT (单向) 高 2.5 分，在 MRPC 上高 4.4 分
3. 模型大小：BERT-large 在所有任务上显著优于 BERT-base，尤其在小数据任务（CoLA, RTE）上提升尤为明显

### 3.4 训练细节

- **优化器**：Adam，学习率 1e-4（BERT-base）/ 5e-5（微调）
- **批次大小**：256 序列
- **训练步数**：1,000,000 步
- **硬件**：4 块 Cloud TPU（BERT-base 训练约 4 天）
- **GLUE 微调**：batch size 32，3 epochs，从 {5e-5, 4e-5, 3e-5, 2e-5} 中选择最佳学习率

---

## 四、Limitations and Challenges

### 4.1 MLM 的计算效率低于自回归 LM

MLM 只预测 15% 的 token，因此每个训练样本只有 15% 的信号。相比之下，自回归语言模型（GPT）利用序列中的每个 token 作为预测目标。这意味着 BERT 需要更多训练步数或更大 batch size 才能达到与自回归模型同等的训练效率。

### 4.2 NSP 任务的效果争议

虽然原论文消融实验显示 NSP 有一定帮助，但后续（RoBERTa, ALBERT）发现：
- NSP 在理解"逻辑连续性"方面很弱——模型只需判断是否来自同一文档，这是简单的表面特征学习
- 改进方案：Sentence Order Prediction (SOP, ALBERT) 或直接移除（RoBERTa）

### 4.3 不擅长生成任务

BERT 天生是 Encoder-only 架构，没有自回归生成能力。虽然可以通过非自回归解码（如 Mask-Predict）或添加 Decoder 来生成文本，但这需要额外的架构设计，不是 BERT 的自然用法。

### 4.4 输入长度限制

最大序列长度为 512 tokens，受限于 Self-Attention 的 O(n²) 计算复杂度。对于长文档（如论文、法律文档），需要分段处理或使用 Longformer/BigBird 等长序列 Transformer 变体。

### 4.5 [MASK] token 的预训练-微调差异

虽然 80%-10%-10% 策略缓解了 mismatch，但问题并未完全消除。预训练时模型看到了大量 [MASK] token，但微调时一个都没有。ELECTRA 的替代方案（用生成器替换而非遮盖）试图从根本上解决这个问题。

---

## 五、Relationship with Subsequent Work / Impact on the Field

BERT 的核心思想"预训练+微调"和"掩码预训练"在两个方向上产生了深远影响：

| 方向 | 模型 | 关系 |
|------|------|------|
| NLP 预训练改进 | RoBERTa (2019) | 去除 NSP，更大 batch，更多数据，更好性能 |
| | ALBERT (2019) | 参数共享，Sentence Order Prediction |
| | DistilBERT (2019) | 知识蒸馏压缩 |
| | ELECTRA (2020) | 替代 MLM，用判别器替换生成器 |
| | XLNet (2019) | 排列语言建模，兼具双向和自回归优点 |
| 视觉预训练 (继承 MLM) | [[../14_MAE/MAE.md|MAE]] (He et al., 2022) | 图像上的随机 mask 预训练——BERT 的 MLM 在像素空间的直接继承者 |
| | VideoMAE | 视频帧 mask 预训练 |
| 多模态预训练 | LLaVA (2023) | 视觉-语言对齐训练，继承"预训练+微调"范式 |
| 机器人/动作 预训练 | π0 (2024) | 预训练（通用多模态理解）→ 后训练（动作对齐），直接沿用 BERT 范式 |

**BERT 与 VLA 的具体连接**：

1. **"预训练+微调"是一切 VLA 的基础**：π0 的训练流程直接分为两步——大规模多模态预训练（通用理解）→ 后训练（动作对齐）。这正是 BERT 开创的范式
2. **MLM 启发了 MAE**：BERT 的随机 mask 策略 → MAE 的高比例随机 mask 图像 patch → 视觉自监督学习的标准范式。DINOv2 的 masked image modeling 也继承了这一思路
3. **Segment Embeddings**：BERT 用于区分句子 A/B 的 segment embeddings，被多模态模型用于区分图像 token 和文本 token（如 LLaVA）
4. **`[CLS]` token 的通用性**：BERT 的 `[CLS]` token 思路被 [[../11_ViT/ViT.md|ViT]] 直接采用（`[class]` token），现为视觉 Transformer 的标准设计
5. **双向注意力的应用**：GR00T N1 使用双向 Transformer 处理视觉-语言-动作（非自回归理解部分），继承了 BERT 的核心设计

---

## 六、Implications for You / Hardware Compatibility

- ✅ **理解"预训练+微调"是 VLA 训练流程的前提**：无论你做哪个 VLA 模型（OpenVLA, π0, RT-2），训练策略都是"大规模预训练 → 任务特定微调/后训练"
- ✅ **Mask 策略具有通用性**：MLM 的随机 mask 预训练适用于图像（[[../14_MAE/MAE.md|MAE]]）、视频（VideoMAE）、甚至动作序列。如果你需要做自监督表示学习，这个思路是你最值得尝试的第一个方案
- ⚠️ **BERT-base (110M) 在单 GPU 上可运行**：在 16GB VRAM 以上的 GPU 上可以完整运行 BERT-base 的微调。BERT-large (340M) 在 24GB VRAM 的 GPU 上需要梯度累积
- ✅ **代码实践**：推荐从 Hugging Face 的 `BertForMaskedLM` 实现入手，理解输入表示（token_ids + token_type_ids + attention_mask）的构造和 MLM 训练流程
- ❌ **不要直接将 BERT 用于生成任务**：Encoder-only 架构没有自回归生成能力，需要改用 Encoder-Decoder（T5）或 Decoder-only（GPT）架构
- ✅ **对于 VLA 中需要"理解"的部分（语义分类、意图识别）**，BERT-style 双向注意力仍然是最佳选择

## PDF

[[BERT 原文.pdf]]
