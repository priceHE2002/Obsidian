---
tags:
  - 论文
  - 生成式预训练
  - 自回归
  - NLP
created: 2026-06-30
paper_title: "Improving Language Understanding by Generative Pre-Training"
paper_authors: "Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever"
paper_year: 2018
paper_venue: "Technical Report (OpenAI)"
paper_citations: "~30,000+"
paper_url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
---

# GPT

**Improving Language Understanding by Generative Pre-Training**
*OpenAI | Technical Report 2018*

> Decoder-only Transformer 路线的开山之作。提出"Generative Pre-Training"范式——在大规模无标注语料上做自回归语言建模预训练（预测下一个 token），再在下游任务上微调。和同期 [[BERT|BERT]] 双向预训练路线形成二元格局，最终被证明在**生成任务和扩展性上更有优势**——GPT-2→GPT-3→GPT-4→ChatGPT 的进化史证明了这一路线的正确性。VLA 中 OpenVLA 等模型使用 Decoder-only 骨干（Llama 2），直接继承自 GPT 的设计。

---

## 一、Background / Core Idea

### 1.1 NLP 任务依赖任务特定架构的时代

在 GPT 之前（2018 年前），NLP 任务的解决方式是高度碎片化的：每个任务（分类、蕴含、相似度、问答）需要不同的网络架构设计，并且需要大量标注数据。论文开篇指出："Most deep learning methods require substantial amounts of manually labeled data, which restricts their applicability in many domains。"

虽然词嵌入（word2vec, GloVe）提供了词级别的预训练表示，但**更高级的文本表示**（句法、语义、长程依赖）需要从无标注数据中学习。

### 1.2 两个关键挑战

论文明确提出了从无标注文本中学习更高级表示的**两个挑战**（原论文 Section 1）：

1. **优化目标不明确**：什么类型的无监督目标对学习可迁移的文本表示最有效？语言建模？机器翻译？话语一致性？不同方法在不同任务上表现各异
2. **迁移方法不统一**：没有共识认为哪种方式是将学习到的表示迁移到目标任务的最佳方式。现有技术涉及"多层次变化"，使迁移过程复杂化

### 1.3 核心洞察：半监督学习 + Transformer Decoder

GPT 的解决方案（论文 Section 1）：

1. **两阶段训练**：无监督预训练（语言建模）+ 有监督微调（任务特定）
2. **使用 Transformer Decoder**：替代先前常见的 LSTM/GRU，利用 Transformer 处理长程依赖的能力
3. **统一的输入格式**：将所有 NLP 任务通过"遍历式"（traversal-style）输入转换统一为序列格式

与 BERT 的分界线：GPT 使用**从左到右的因果注意力**（预测下一个 token），而 BERT 使用**双向注意力**（预测遮盖 token）。这个差异决定了 GPT 更适合生成任务，BERT 更适合理解任务。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 两阶段训练框架

**阶段一：无监督预训练**

在大规模无标注语料上进行标准语言建模（因果语言建模）：

$$L_1(\mathcal{U}) = \sum_i \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)$$

其中 $\mathcal{U}$ 是无标注语料，每个 token 基于前 k 个 token 预测。模型架构是 multi-layer Transformer Decoder（原论文 Eq. 2）：

$$h_0 = U W_e + W_p$$
$$h_l = \text{transformer\_block}(h_{l-1}) \quad \forall i \in [1, n]$$
$$P(u) = \text{softmax}(h_n W_e^T)$$

与原始 [[Attention Is All You Need|Transformer]] 的不同：GPT 移除了 Decoder 中的 cross-attention 层（因为只有纯自回归单模态生成，不需要从 Encoder 读取信息），只保留 masked self-attention + FFN。

**阶段二：有监督微调**

在目标任务上微调预训练参数。对于带标签数据集 $\mathcal{C}$，输入经过预训练模型得到最后一层的激活 $h_l^m$，通过一个线性输出层预测标签 y：

$$P(y | x_1, ..., x_m) = \text{softmax}(h_l^m W_y)$$

**辅助目标**：微调时加入语言建模 loss 作为辅助目标（原论文发现这有助于改善泛化和加速收敛）。

### 2.2 模型架构详解

| 参数 | 数值 |
|------|------|
| 层数 | 12 |
| Hidden Size | 768 |
| Attention Heads | 12（每头维度 64） |
| Feed-Forward 维度 | 3072 |
| 参数量 | 117M |
| 最大序列长度 | 512 |
| 激活函数 | GELU（Gaussian Error Linear Unit） |
| 位置编码 | 可学习（非正弦） |
| 词汇表 | BPE, 40,000 merges |
| LayerNorm | Extensive use（权重初始化 N(0, 0.02)） |

**重要设计选择**：
- 使用 **GELU** 激活函数替代 ReLU——GELU 在 2016 年被提出，是 ReLU 的平滑近似，被后续几乎所有 Transformer 模型继承
- 使用 **可学习位置编码** 而非正弦编码——实验发现两者效果接近，但可学习版本更灵活
- **权重初始化**：N(0, 0.02) ——由于 LayerNorm 的广泛使用，简单的初始化就足够
- **L2 正则化**：变体版本，w=0.01，仅用于非 bias 和 gain 权重

### 2.3 任务统一输入格式

这是 GPT 的重要贡献——将所有 NLP 任务统一为序列输入输出格式，避免了对每个任务架构的修改。具体转换方式（原论文 Figure 1 右）：

| 任务 | 输入格式 | 输出 |
|------|---------|------|
| 分类 | [Start] 文本 [Extract] | 线性层 → softmax |
| 蕴含 | [Start] 前提 [Delim] 假设 [Extract] | 线性层 → softmax |
| 相似度 | [Start] 文本1 [Delim] 文本2 [Extract] + [Start] 文本2 [Delim] 文本1 [Extract]（两个方向） | 两个表示逐元素相加 → 线性层 |
| 问答/常识推理 | [Start] 文档 [Delim] 问题 [Delim] 答案1 [Extract] ... 答案 N | 各序列独立处理 → softmax |

这种设计的优雅之处：**微调只需要修改输入格式，不需要改变模型架构**。这个思想在 VLA 领域被继承——将视觉 token、语言 token、动作 token 统一为单一序列。

### 2.4 预训练数据

- **BooksCorpus**：约 7,000 本未出版书籍，涵盖冒险、奇幻、爱情等多种类型。论文强调："It contains long stretches of contiguous text, which allows the generative model to learn to condition on long-range information."
- 与 ELMo 使用的 1B Word Benchmark 对比：后者虽然规模相似，但在句子级别打乱了顺序，**破坏了长程结构**
- 语言模型在 BooksCorpus 上达到 **perplexity 18.4**

GPT 的数据量和质量相比后来的工作都有限——GPT-2 使用 800 万网页的 WebText，GPT-3 使用 45TB 的 Common Crawl——但作为第一代已经足够验证 Decoder-only 路线的可行性。

### 2.5 训练细节

- **优化器**：Adam，最大学习率 2.5e-4
- **学习率调度**：前 2000 步从 0 线性增加到最大值，然后用余弦调度衰减到 0
- **Batch size**：64（每个样本为 512 token 的连续片段）
- **训练轮数**：100 epochs
- **正则化**：残差/嵌入/注意力 dropout rate = 0.1
- **微调细节**：分类器 dropout rate = 0.1，学习率 6.25e-5，batch size 32，3 epochs。微调时使用线性衰减，warmup 占 0.2% 的训练步数

---

## 三、Experiments and Key Findings

### 3.1 自然语言推理 (NLI) 结果

| 方法 | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
|------|--------|---------|------|---------|------|-----|
| ESIM + ELMo (5x ensemble) | — | — | 89.3 | — | — | — |
| CAFE (5x) | 80.2 | 79.0 | 89.3 | — | — | — |
| Stochastic Answer Network (3x) | 80.6 | 80.1 | — | — | — | — |
| CAFE (single) | 78.7 | 77.9 | 88.5 | 83.3 | — | — |
| GenSen | 71.4 | 71.3 | — | — | 82.3 | 59.2 |
| Multi-task BiLSTM + Attn | 72.2 | 72.1 | — | — | 82.1 | 61.7 |
| **GPT (ours)** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | **56.0** |

GPT 在 6 项 NLI 任务中取得 5 项 SOTA，并且不使用任务特定的架构设计。

### 3.2 问答与常识推理结果

| 方法 | Story Cloze | RACE-m | RACE-h | RACE (avg) |
|------|------------|--------|--------|------------|
| 此前最佳 | 77.6 | 60.2 | 50.3 | 53.3 |
| **GPT** | **86.5** | **62.9** | **57.4** | **59.0** |

RACE 数据集的提升（+5.7%）尤为显著——这是对长程推理能力要求很高的初中/高中英语考试题。

### 3.3 语义相似度与分类结果

| 方法 | CoLA (mc) | SST-2 (acc) | MRPC (F1) | STS-B (pc) | QQP (F1) | GLUE |
|------|-----------|-------------|-----------|------------|----------|------|
| 此前 SOTA | 35.0 | 93.2 | 86.0 | 81.0 | 66.1 | 64.8 |
| **GPT** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

GLUE 整体得分大幅超越此前 SOTA（+8.0），但 SST-2 和 MRPC 未达 SOTA，反映单向注意力在分类任务上的局限（相比之下 BERT 后来在这些任务上是碾压性的）。

### 3.4 消融实验

| 方法 | 平均分 | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|------|--------|------|-------|------|-------|-----|------|------|-----|
| Full (Transformer + aux LM) | 74.7 | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| w/o 预训练 | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| w/o 辅助 LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| LSTM + aux LM | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

**关键发现**：
1. **预训练的巨大影响**：无预训练的 Transformer 平均分下降 14.8 分（从 74.7 降到 59.9），CoLA 从 45.4 骤降至 18.9
2. **辅助 LM 的效果**：去除后平均分几乎不变（74.7 vs 75.0），但 CoLA 从 45.4 提升到 47.9，表明在数据量小或特定类型任务上辅助 LM 可能干扰而非帮助
3. **Transformer vs LSTM**：Transformer + aux LM 比 LSTM + aux LM 高 5.6 分（74.7 vs 69.1），证明 Transformer 的处理长程依赖的能力对迁移学习至关重要

### 3.5 零样本性能随预训练的变化

原论文 Figure 2 (right) 展示了零样本性能随预训练步数的变化。关键发现：
- GPT 的零样本性能（不做任何微调）在预训练过程中持续提升，说明语言建模预训练确实学会了与下游任务相关的功能
- Transformer 的零样本性能比 LSTM 更稳定（方差更小），表明 Transformer 的归纳偏置有助于迁移

具体的零样本方法：
- CoLA：以 token 平均对数概率是否超过阈值判断
- SST-2：在句子后添加 "very" 限制模型输出仅为 "positive" 或 "negative"
- RACE：选择文档/问题条件下平均 token 对数概率最高的答案
- Winograd Schema：替换代词为候选词，选择使后续 token 概率更高的候选

---

## 四、Limitations and Challenges

### 4.1 单向注意力限制了理解能力

GPT 的单向注意力（只能看左侧上下文）在需要双向理解的任务上不如 [[BERT|BERT]]。同期 BERT 在 GLUE 上比 GPT 高 7+ 分，在 SQuAD 上高 5+ 分，直接反映了双向注意力的优势。2018 年的时点来看，BERT 似乎是更优越的路线。

### 4.2 参数量和数据量有限

GPT-1 仅 117M 参数，训练数据仅 BooksCorpus——这在当时是合理的规模，但后来 GPT-2 (1.5B) 和 GPT-3 (175B) 展示了超大规模带来的涌现能力（如上下文学习）。GPT-1 的生成能力也远未成熟，论文更多侧重理解任务评估。

### 4.3 统一的输入格式但不完美

虽然"统一格式"思想优雅，但实际上对不同类型任务（尤其是相似度任务，需要双向编码）的适配不够自然。BERT 的 [SEP] + Segment Embeddings 对句子对任务的适配更优雅。

### 4.4 固定长度上下文窗口

512 token 的最大序列长度限制了对更长文档的处理能力，这是 Transformer 架构的通病（O(n²) 复杂度），而非 GPT 特有的局限。

---

## 五、Relationship with Subsequent Work / Impact on the Field

GPT-1 作为 Decoder-only Transformer 的奠基之作，其影响通过 GPT 系列不断放大：

| 模型 | 年份 | 参数量 | 关键创新 | VLA 关系 |
|------|------|--------|---------|----------|
| **GPT-1** | 2018 | 117M | 生成式预训练 + 微调范式 | Decoder-only 架构的奠基 |
| GPT-2 | 2019 | 1.5B | 零样本迁移，WebText 数据集 | 规模化证明了自回归的扩展能力 |
| GPT-3 | 2020 | 175B | 少样本/上下文学习 (In-Context Learning) | 涌现能力的概念验证 |
| GPT-4 | 2023 | 多模态 | 视觉-语言理解，推理能力 | 多模态 Decoder-only 的验证 |
| **Llama 2/3** | 2023 | 7B-405B | Decoder-only Transformer，开源 | [[../20_Llama2/Llama2.md|Llama 2]] 是 OpenVLA 的骨干 |
| **Qwen2-VL** | 2024 | 2B-72B | Decoder-only 视觉语言模型 | 新一代 VLA 骨架候选 |

**GPT-1 在 VLA 中的核心影响**：

1. **OpenVLA 的核心架构是 Decoder-only Transformer**：OpenVLA 使用 Llama 2 (Decoder-only) 作为骨干，将视觉 token 通过投影层拼接后输入——这正是 GPT"统一序列格式"思路在 VLA 中的直接应用
2. **"所有任务都统一为 token 序列预测"**：VLA 模型将视觉、语言、动作统一为 token 序列，使用同一个 Transformer 自回归预测。动作 token（Spaeter/Lowlevel tokenizers）的预测本质上就是"预测下一个 token"
3. **RT-2 的动作 token 化**：将连续动作离散化为 token，用自回归方式预测——直接继承自 GPT 的语言建模方式
4. **"统一格式"思想的延续**：GPT 将不同 NLP 任务统一为序列格式 → VLA 将视觉/语言/动作统一为 token 序列 → 同一架构处理多模态输入
5. **跨任务泛化**：GPT-3 的上下文学习能力在 VLA 中对应为少样本任务适应（IROS 2024 的 few-shot VLA 方向）

---

## 六、Implications for You / Hardware Compatibility

- ✅ **Decoder-only 是当前最主流的 VLA 骨干架构**：OpenVLA、RT-2 都使用 Decoder-only。了解 GPT-style 架构是理解这些模型的基础
- ✅ **从 nanoGPT 开始代码实践**：Andrej Karpathy 的 nanoGPT (~300 行 PyTorch) 是实现 Decoder-only Transformer 的最佳学习资源，然后可以扩展到 Prismatic VLM（OpenVLA 的前置视觉-语言预训练）
- ⚠️ **注意因果掩码的实现**：理解 causal attention mask 的构造（上三角矩阵掩码）和 KV cache（推理时缓存 Key/Value 以避免重复计算）是部署 Decoder-only 模型的关键
- ❌ **动作 tokenization 是 VLA 特有的难点**：语言建模天然适用于离散 token，但机器人动作是连续的。将其离散化（Spaeter, Lowlevel, Uniform tokenizer）会引入量化误差，需要权衡 token 数量和精度
- ✅ **上下文学习在 VLA 中的潜力**：GPT-3 的少样本能力在 VLA 中可转化为"在新任务上通过 few-shot demonstration 快速适应"——这是人形机器人泛化的关键方向之一
- ⚠️ **单 GPU 可以运行 GPT-1 级别 (117M) 的模型**：在 8GB VRAM 以上的 GPU 上，完整的预训练和微调都可行。但 GPT-3 级别 (175B) 需要分布式系统

## PDF

[[GPT 原文.pdf]]
