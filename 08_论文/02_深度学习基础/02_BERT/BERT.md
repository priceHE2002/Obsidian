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
---

# BERT

**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
*Google AI | NAACL 2019 | arXiv 1810.04805*

> "预训练+微调"范式的里程碑。提出 Masked Language Model (MLM) 和 Next Sentence Prediction (NSP) 两项预训练任务，在 11 项 NLP 任务上刷新 SOTA。虽然不是 VLA 的直接组件，但其"预训练+微调"思想被所有现代 VLA 继承。

---

## 一、研究背景与动机

在 BERT 之前，NLP 的预训练方法主要是单向的（如 ELMo 使用从左到右 + 从右到左的 LSTM 拼接，GPT 使用从左到右的 Transformer Decoder）。这些方法在生成任务上表现良好，但在需要深层理解的任务（如分类、阅读理解）上存在根本性局限——模型无法同时看到被预测 token 左右两侧的上下文。

作者提出一个关键问题：如果预训练语言模型能同时看到 token 两侧的上下文，是否能在理解任务上取得突破？这不仅需要一个双向架构（Transformer Encoder），还需要一种新的预训练目标——传统的语言建模（预测下一个 token）不允许双向信息。

## 二、核心方法

### 2.1 模型架构

BERT 使用 Transformer Encoder（即 Transformer 论文中的编码器部分），有两种配置：

| 配置 | 层数 | Hidden Size | Attention Heads | 参数量 |
|------|------|-------------|-----------------|--------|
| BERT Base | 12 | 768 | 12 | 110M |
| BERT Large | 24 | 1024 | 16 | 340M |

### 2.2 预训练任务

#### Task 1: Masked Language Model (MLM)

随机 mask 15% 的 token，让模型预测被 mask 的词：
- 在选定 token 中：80% 替换为 [MASK]，10% 替换为随机词，10% 保持不变
- 这样做是为了缓解预训练阶段到微调阶段的 mismatch（因为 [MASK] token 在微调时不存在）

#### Task 2: Next Sentence Prediction (NSP)

判断句子 B 是否为句子 A 的后续句子：
- 50% 正例（实际连续的句子对）
- 50% 负例（从语料中随机抽取的句子对）

NSP 的目的是让模型学习句子间的关系，这对问答、自然语言推理等任务至关重要。后来自 RoBERTa 以来被证明效果有限。

### 2.3 输入表示

每个输入序列为 `[CLS] token1 token2 ... [SEP] sentence B tokens [SEP]`，其中：
- `[CLS]` 的最终隐层状态用于分类任务
- `[SEP]` 分隔不同句子
- 使用 Token Embeddings + Segment Embeddings + Position Embeddings 三者的和

### 2.4 预训练数据

- BooksCorpus (800M 词)
- English Wikipedia (2,500M 词)

## 三、关键实验与发现

| 任务 | 此前 SOTA | BERT Base | BERT Large |
|------|-----------|-----------|------------|
| GLUE | 80.0 | 85.7 | 87.4 |
| MultiNLI | 80.6 | 84.7 | 86.7 |
| SQuAD v1.1 (F1) | 91.2 | 88.5 (w/o data aug) | 93.2 |
| SQuAD v2.0 (F1) | 80.1 | 79.1 | 89.9 |
| SWAG | 83.0 | 86.3 | 89.2 |

关键发现：
1. **双向注意力至关重要**：BERT 比 GPT (单向 Transformer) 在 GLUE 上高 7%+，在 SQuAD 上高 5%+
2. **Large 模型提升显著**：340M 比 110M 在所有任务上都有大幅提升
3. **预训练数据质量重要**：使用 BooksCorpus + Wikipedia（文档级数据）比句子级数据更好

## 四、局限性与后续影响

**局限**：
1. **NSP 任务效果有限**：RoBERTa (Liu et al., 2019) 发现去除 NSP 后性能反而略有提升
2. **MLM 效率低于自回归 LM**：因为只预测 15% 的 token，每个 token 需要更多的训练步数
3. **生成任务不擅长**：BERT 天生是 Encoder-only 架构，不适合文本生成（需要额外的 Decoder 或非自回归解码策略）
4. **输入长度限制**：最大 512 tokens，受限于自注意力 O(n²) 复杂度

**后续影响**：
- RoBERTa：优化训练策略，去除 NSP，使用更多数据、更大 batch → 更强
- ALBERT：参数共享 → 更小的模型
- DistilBERT：知识蒸馏 → 更快的推理
- "BERTology"：大量分析 BERT 内部表示的研究方向

## 五、VLA/机器人研究中的角色

BERT 虽然不是 VLA 模型直接采用的架构（现代 VLA 更多使用 Decoder-only 架构），但其核心思想深刻影响了 VLA：

1. **"预训练+微调"范式**：这是所有现代 VLA 的基础——在大规模多模态数据上预训练 → 在目标机器人数据上微调。π0 的预训练/后训练分离直接沿用了 BERT 开创的范式
2. **MLM 启发了 MAE**：BERT 的 Mask 策略启发了 Masked Autoencoder (MAE)（He et al., 2022），图像上的随机 mask 预训练成为视觉基础模型的标准训练方式
3. **双向注意力在 VLA 中的应用**：GR00T N1 和 RT-2 的某些组件使用双向注意力处理视觉-语言输入（不同于纯自回归的 GPT）
4. **Segment Embeddings**：来自 BERT 的 segment/sentence 编码方式被多模态模型沿用来区分图像和文本 token

## 六、对你的启示

1. **"预训练+微调"是 VLA 的核心策略**：理解 BERT 的预训练-微调分离是掌握 VLA 训练流程的前提——大规模预训练（通用能力）、后训练/微调（任务对齐）
2. **注意哪种注意力适合你的任务**：
   - 需要**理解**（分类、描述）：双向注意力（BERT-style）
   - 需要**生成**（对话、代码）：单向注意力（GPT-style）
   - 需要**对齐**（翻译、caption）：交叉注意力（Encoder-Decoder）
3. **Mask 策略的通用性**：BERT 的 mask 策略启发了图像、视频、动作序列的统一 pre-training 方法（MAE、VideoMAE）
4. **代码实践**：建议实现一个简化版 BERT（在 tiny 数据上做 MLM 预训练），这会帮助你理解现代预训练流水线

## PDF

[[1810.04805_BERT.pdf]]
