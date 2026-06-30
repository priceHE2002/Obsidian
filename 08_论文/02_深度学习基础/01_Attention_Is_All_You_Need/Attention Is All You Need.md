---
tags:
  - 论文
  - Transformer架构
  - 注意力机制
  - 自注意力
  - 位置编码
created: 2026-06-30
paper_title: "Attention Is All You Need"
paper_authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin"
paper_year: 2017
paper_venue: "NeurIPS 2017"
paper_citations: "~140,000+"
paper_url: "https://arxiv.org/abs/1706.03762"
---

# Attention Is All You Need

**Attention Is All You Need**
*Google Brain | NeurIPS 2017 | arXiv 1706.03762*

> 深度学习领域最革命性的论文之一。提出 Transformer 架构，用纯注意力机制替代 RNN/LSTM，彻底改变了 NLP 和 CV 的格局。VLA 中所有模型的骨干网络（GPT/Llama/PaliGemma/Qwen）都基于此架构。

---

## 一、研究背景与动机

在 Transformer 之前，序列建模的主流方法是 RNN（LSTM、GRU）。RNN 的核心问题在于其**顺序计算的本质**——每个时间步必须等待前一个时间步完成才能计算，这使得训练无法并行化，限制了在长序列和大规模数据上的扩展能力。同时，RNN 在处理长序列时存在梯度消失/爆炸问题，即便使用 LSTM 的门控机制也只能缓解而无法根除。

注意力机制此前主要作为 RNN 的辅助组件（如 Bahdanau Attention）用于机器翻译的编码-解码对齐。作者提出一个激进的问题：如果完全丢弃 RNN，只用注意力机制，能否构建一个更好的序列模型？

## 二、核心方法

### 2.1 整体架构

Transformer 采用 Encoder-Decoder 架构：
- **Encoder**：6 层，每层包含 Multi-Head Self-Attention + Feed-Forward Network，每层后接 Add & LayerNorm（残差连接 + 层归一化）
- **Decoder**：6 层，每层包含 Masked Multi-Head Self-Attention + Cross-Attention + FFN，同样每层后接 Add & LayerNorm

### 2.2 Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q$, $K$, $V$ 分别代表查询、键、值矩阵
- $\sqrt{d_k}$ 缩放因子防止内积过大导致 softmax 梯度消失

### 2.3 Multi-Head Attention

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O
$$
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

作者使用 $h=8$ 个头，每个头的维度 $d_k = d_v = 64$。多个头允许模型在不同子空间关注不同位置的信息。

### 2.4 Positional Encoding

由于自注意力本身没有序列顺序信息，作者使用正弦函数编码位置：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

这种编码方式允许模型外推到更长的序列。

### 2.5 参数配置

| 配置 | 参数量 | d_model | h | d_ff | 层数 |
|------|--------|---------|---|------|------|
| Base | ~65M | 512 | 8 | 2048 | 6 |
| Big | ~213M | 1024 | 16 | 4096 | 6 |

## 三、关键实验与发现

| 任务 | 指标 | Transformer | 此前 SOTA | 提升 |
|------|------|-------------|-----------|------|
| WMT 2014 英德翻译 | BLEU | 28.4 | 26.2 (Seq2Seq + Attn) | +2.2 |
| WMT 2014 英法翻译 | BLEU | 41.8 | 39.2 (Seq2Seq + Attn) | +2.6 |
| 训练速度 | hours | 3.5 days (8 P100) | 3-14 days | 3-4x faster |

关键发现：
1. **训练速度远快于 RNN**：由于可并行计算，训练时间减少一个数量级
2. **多头注意力的不同头关注不同模式**：有些头关注句法关系，有些关注语义关系
3. **Big 模型翻译质量显著优于 Base**：更大的模型从并行计算中受益

## 四、局限性与后续影响

**局限**：
1. **自注意力复杂度 O(n²)**：对长序列（如文档、视频帧）计算和内存开销随序列长度平方增长
2. **位置编码固定**：原始论文使用固定的正弦编码（BERT 改为可学习位置编码，GPT-2 使用可学习的）
3. **缺少局部归纳偏置**：CNN 天然有局部性先验，Transformer 需要大量数据来学习

**后续影响**：
- BERT (2018)：Transformer Encoder + MLM → 理解任务 SOTA
- GPT 系列 (2018-)：Transformer Decoder + 自回归 → 生成任务 SOTA
- ViT (2020)：Transformer 应用于图像 → 视觉基础模型
- 几乎取代了所有 RNN/LSTM 应用

## 五、VLA/机器人研究中的角色

Transformer 是 VLA（Vision-Language-Action）模型的基础计算架构：

1. **RT-2**：骨干 PaLI-X 基于 Encoder-Decoder Transformer
2. **OpenVLA**：骨干 Llama 2 基于 Decoder-only Transformer
3. **π0**：VLM 使用 PaliGemma（基于 Transformer），Action Expert 使用 DiT（基于 Transformer 的扩散架构）
4. **GR00T N1**：使用双向 Transformer 处理视觉-语言-动作
5. **Octo**：基于 T5 Encoder-Decoder Transformer

几乎所有现代 VLA 模型的架构选择都是"用哪种 Transformer 变体"——Encoder-Decoder、Decoder-only、或 Encoder-only——而非考虑是否使用 Transformer。

## 六、对你的启示

1. **实现一个最小 Transformer 是理解 VLA 的最佳起点**：从零实现 Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding、LayerNorm，对理解所有 VLA 模型至关重要
2. **注意复杂度选择**：在面对长序列场景（如视频帧序列、高维动作轨迹）时，O(n²) 的复杂度是主要瓶颈，需要关注 FlashAttention、Ring Attention 等优化
3. **代码实践建议**：仔细阅读 PyTorch 的 `nn.MultiheadAttention` 源码和 Andrej Karpathy 的 minGPT/nanoGPT 实现
4. **在 VLA 中关注"哪种 Transformer 变体最适合"**：Decoder-only（生成灵活但缺少交叉注意力）、Encoder-Decoder（编码理解和生成分离）、Encoder-only（纯理解）

## PDF

[[1706.03762_Attention_Is_All_You_Need.pdf]]
