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
paper_venue: "Technical Report"
paper_citations: "~30,000+"
paper_url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
---

# GPT

**Improving Language Understanding by Generative Pre-Training**
*OpenAI | Technical Report 2018*

> 提出 "Generative Pre-Training" 范式——在大规模无标注语料上做自回归语言建模预训练，再在下游任务上微调。这是 GPT 系列（GPT-2→GPT-3→GPT-4）的起点，也是 Decoder-only Transformer 路线的开创者。VLA 中从 RT-2 到 OpenVLA 都使用 Decoder-only 骨干。

---

## 一、研究背景与动机

在 GPT 之前，NLP 任务需要大量标注数据和任务特定的架构设计。每个任务（分类、蕴含、相似度、问答）都有不同的网络拓扑结构，研究者需要为每个任务设计专门的模型。

虽然 BERT 同期也提出预训练方法，但 GPT 选择了不同的路径——**Decoder-only Transformer + 自回归语言建模**。核心动机是：无标注文本极其丰富，如果能通过语言建模目标从这些文本中学习通用的语言表示，再通过少量标注数据微调，就能大幅减少对标注数据的依赖。

GPT 的关键洞察是：**所有 NLP 任务都可以统一为序列输入输出格式**，从而使用同一个架构和微调流程。

## 二、核心方法

### 2.1 两阶段训练

#### 阶段一：无监督预训练

在大规模无标注语料上进行标准语言建模（因果语言建模）：

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i | u_{i-k}, ..., u_{i-1}; \Theta)
$$

其中 $\mathcal{U}$ 是无标注语料，每个 token 基于前 $k$ 个 token 预测。

#### 阶段二：有监督微调

在目标任务上微调预训练参数：

$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y | x_1, ..., x_m)
$$

### 2.2 模型架构

GPT 使用 12 层 Transformer Decoder（仅 decoder 的 masked self-attention）：

| 配置 | 数值 |
|------|------|
| 层数 | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Feed-Forward | 3072 |
| 参数量 | 117M |
| 最大序列长度 | 512 |

与 Transformer Decoder 不同：GPT 移除了 decoder 中的 cross-attention 层（因为只有单向生成，无需从 encoder 读取信息），只保留 masked self-attention + FFN。

### 2.3 任务统一格式

GPT 将所有下游任务转换为序列输入格式：

| 任务 | 输入格式 | 输出 |
|------|---------|------|
| 分类 | [Start] 文本 [Extract] | 类别标签 |
| 蕴含 | [Start] 前提 [Delim] 假设 [Extract] | 蕴含/矛盾/中立 |
| 相似度 | [Start] 文本1 [Delim] 文本2 [Extract] + [Start] 文本2 [Delim] 文本1 [Extract] | 相似度分数 |
| 问答 | [Start] 问题 [Delim] 文档 [Delim] 候选1 ... [Extract] | 选择候选 |

这种"统一格式"思想的优雅之处在于：微调只需要修改输入输出格式，不需要改变模型架构。

### 2.4 预训练数据

- BooksCorpus：约 7,000 本未出版书籍（文本质量高，包含长程依赖关系）
- 1B Word Benchmark（用于消融实验）

## 三、关键实验与发现

| 任务 | 数据集 | SOTA | GPT | 结果 |
|------|--------|------|-----|------|
| 情感分类 | SST-5 | 53.3 | **54.2** | 超越 SOTA |
| 自然语言推理 | MultiNLI | 76.4 | 76.5 | 超越 SOTA |
| 语义相似度 | MRPC (F1) | 83.0 | **87.4** | 大幅超越 |
| 文本蕴含 | RTE | 67.9 | **69.2** | 超越 SOTA |
| 问答 | RACE (Acc) | 72.0 | **72.8** | 超越 SOTA |

关键发现：
1. **预训练质量与数据量正相关**：更大的预训练周期持续提升下游性能
2. **零样本迁移能力**：即使不微调，GPT 在某些任务上已有一定能力（这启发了 GPT-2 的零样本实验）
3. **辅助语言建模目标有帮助**：微调时加入语言建模 loss (auxiliary LM objective) 提升泛化性

## 四、局限性与后续影响

**局限**：
1. **仅 117M 参数**，远小于后来的 GPT-2 (1.5B) 和 GPT-3 (175B)
2. **单向注意力**：只能看到左侧上下文，在需要双向理解的任务（如阅读理解）上不如 BERT
3. **预训练数据有限**：仅 BooksCorpus（GPT-2 使用 WebText，GPT-3 使用 Common Crawl）
4. **生成能力有限**：作为第一代 GPT，更多侧重理解任务（2018 年时未强调生成能力）

**后续影响**：
- GPT-2 (2019)：1.5B 参数，零样本迁移能力，文本生成
- GPT-3 (2020)：175B 参数，少样本学习能力（In-Context Learning）
- GPT-4 (2023)：多模态，推理能力
- ChatGPT (2022)：基于 GPT-3.5 + RLHF 的对话模型
- 整个 Decoder-only Transformer 路线的奠基

## 五、VLA/机器人研究中的角色

GPT 虽然不是 VLA 文献中直接引用的对象，但其开创的 Decoder-only 路线是 VLA 模型的核心架构：

1. **OpenVLA**：使用 Llama 2（Decoder-only Transformer）作为骨干，将视觉 token 投影后拼接输入
2. **π0 / PaliGemma**：虽然整体是 Encoder-Decoder，但文本解码器是 Decoder-only 的自回归生成
3. **GR00T N1**：使用 Decoder-only 变体的双向注意力处理视觉-语言-动作
4. **RT-2 的动作 token 化**：将动作离散化为 token，用自回归方式预测——这直接继承自 GPT 的语言建模方式
5. **Rh20t**：统一动作 token 序列预测

**"统一格式"思想在 VLA 中的体现**：VLA 模型将视觉、语言、动作统一为 token 序列，使用同一个 Transformer 自回归预测——这正是 GPT 的最核心思想在机器人领域的应用。

## 六、对你的启示

1. **Decoder-only 是当前 VLA 的主流架构**：了解 GPT-style Decoder-only Transformer 的优势和局限是选择 VLA 骨干网络的基础
2. **自回归生成与连续动作的偏离**：动作是连续的，将其离散化为 token 是 GPT 方法用于 VLA 的关键适配。需要理解 Gumbel-Softmax 和 tokenization 策略（如 Spaeter, Lowlevel, Uniform 等 tokenizer）
3. **In-Context Learning 在 VLA 中的潜力**：GPT-3 的少样本学习能力在 VLA 中可以对应为"少样本任务适应"——让模型在新任务上通过 few-shot demonstration 快速适应
4. **代码实践建议**：从 Andrej Karpathy 的 nanoGPT 入手理解和实现 Decoder-only Transformer，然后扩展为 VLA 架构（添加视觉编码器、动作解码器）

## PDF

[[GPT_Improving_Language_Understanding_by_Generative_Pre-Training.pdf]]
