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
github: "https://github.com/tensorflow/tensor2tensor"
---

# Transformer

**Attention Is All You Need**
*Google Brain / University of Toronto | NeurIPS 2017 | arXiv 1706.03762*

> 深度学习领域最具革命性的论文，没有之一。提出 Transformer 架构，用纯注意力机制完全替代 RNN/LSTM 和卷积，解决了序列建模的并行化难题。这是所有后续 VLA 模型（GPT、Llama、PaliGemma、Qwen2-VL）的架构基石。理解 Transformer 是理解整个 VLA 领域的前提。

---

## 一、Background / Core Idea

### 1.1 此前序列建模的困境：RNN 的致命缺陷

在 Transformer 之前（2017 年以前），序列建模的标准方案是 RNN（LSTM/GRU），如 [[1706.03762_Attention_Is_All_You_Need.pdf|原论文]] 所述。RNN 的核心问题在于其**顺序计算的本质**——每个时间步必须等待前一个时间步计算完成才能继续，这使得训练**完全无法并行化**。对于长度为 n 的序列，RNN 需要 O(n) 步顺序操作，而 Transformer 只需要 O(1) 步。

RNN 的另一问题是**梯度消失/爆炸**。即便使用 LSTM 的门控机制，当序列长度超过 100-200 token 时，早期位置的梯度信息仍然会指数级衰减。原论文（第 2 页）明确指出："This inherently sequential nature precludes parallelization within training examples。"

### 1.2 注意力机制的历史：从辅助到主角

注意力机制此前主要作为 RNN 的辅助组件：
- **Bahdanau Attention（2014）**：在 Seq2Seq 模型中，Decoder 在每个时间步对 Encoder 所有隐藏状态做加权求和，解决了编码信息瓶颈
- **Luong Attention（2015）**：简化了 Bahdanau 的计算方式，提出了 global/local 两种注意力变体

但关键点是——"In all but a few cases, such attention mechanisms are used in conjunction with a recurrent network." 没有人想过完全用注意力替代 RNN。

### 1.3 核心洞察：自注意力可以替代循环

作者提出了一个激进的问题：**如果完全丢弃 RNN，只用注意力机制，能否构建一个更好的序列模型？** 这个问题的答案后来改变了整个深度学习格局。

三个关键动机（第 6 页，Section 4 "Why Self-Attention"）：
1. **每层计算复杂度**：Self-Attention 为 O(n^2·d)，RNN 为 O(n·d^2)。当序列长度 n 小于表示维度 d（实际中通常如此），Self-Attention 反而更高效
2. **可并行化的计算量**：Self-Attention 只需要 O(1) 步顺序操作，RNN 需要 O(n) 步
3. **长距离依赖路径长度**：Self-Attention 的最大路径长度为 O(1)，RNN 为 O(n)，卷积为 O(log_k(n))。路径越短，前向/反向信号越容易传播

---

## 二、Method / Architecture / Technical Contribution

### 2.1 整体架构：Encoder-Decoder 框架

Transformer 沿用了 Encoder-Decoder 架构（原论文 Figure 1），但内部全部用注意力+前馈网络替代了 RNN：

- **Encoder**：6 层 (N=6) 相同结构。每层包含两个子层：(1) Multi-Head Self-Attention, (2) Position-wise Feed-Forward Network。每个子层后接残差连接和 LayerNorm
- **Decoder**：6 层 (N=6) 相同结构。每层包含三个子层：(1) Masked Multi-Head Self-Attention, (2) Cross-Attention（关注 Encoder 输出）, (3) FFN。同样每个子层后接残差连接 + LayerNorm

所有子层和嵌入层的输出维度均为 d_model = 512（Base 模型）或 1024（Big 模型）。

**残差连接 + LayerNorm 的精确形式**：
```
output = LayerNorm(x + Sublayer(x))
```
其中 Sublayer(x) 是子层自身的函数。这种"先加后归一"（Post-LN）的设计与后来 GPT 使用的 Pre-LN 不同。Post-LN 在训练较深模型时可能不稳定，这启发了后续的 Pre-LN 改进。

### 2.2 Scaled Dot-Product Attention：数学推导与直觉

这是 Transformer 最核心的公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 Q ∈ ℝ^{n×d_k}, K ∈ ℝ^{m×d_k}, V ∈ ℝ^{m×d_v}。

**为什么需要除以 √d_k？**

原论文第 4 页给出了严谨的解释：假设 q 和 k 的分量是独立的随机变量，均值为 0，方差为 1。则点积 q·k = Σ_{i=1}^{d_k} q_i k_i 的均值为 0，方差为 d_k。当 d_k 较大时，点积的绝对值会很大，将 softmax 推入梯度极小的区域（梯度消失）。

除以 √d_k 后，点积的方差变为 1，softmax 的输入保持在一个合理的范围内，梯度可以良好传播。

论文通过消融实验验证了这一设计：当 d_k 从 64 减小到 32（行 B），BLEU 从 25.8 降至 25.1；当不使用缩放（等效于 d_k=1），性能进一步下降。这证明了缩放不是可选的，而是必要的。

**为什么用点积注意力而不是加性注意力？**

点积注意力（dot-product attention）虽然理论上与加性注意力（additive attention）复杂度相近，但"can be implemented using highly optimized matrix multiplication code"——即可以利用高度优化的矩阵乘法库（cuBLAS），实际速度快得多且更省内存。

### 2.3 Multi-Head Attention：并行化的注意力

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

**为什么用 8 个头？**

作者使用了 h=8，每个头的维度 d_k = d_v = d_model / h = 64。这样总计算量与单头注意力（d_model=512）基本相同——"Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."

**每个头学到了什么？**

原论文的附录（Figure 3-5）展示了不同注意力头的学习模式：
- 有些头关注句法关系（如动词与其宾语的长距离依赖）
- 有些头关注指代消解（如"it"指向哪个名词）
- 不同头在相同位置关注不同的语义模式

消融实验（Table 3 行 A）显示：单头注意力（h=1）比最佳设置（h=8）低 0.9 BLEU；但头数过多（h=16）也会轻微下降，说明 8 个头是权衡计算和表达能力的合理选择。

### 2.4 注意力在模型中的三种应用

Transformer 在三种场景中使用 Multi-Head Attention（原论文 Section 3.2.3）：

1. **Encoder-Decoder Attention**：Decoder 的 Query 来自前一 Decoder 层，Key/Value 来自 Encoder 输出。这使得 Decoder 的每个位置都能关注输入序列的所有位置——这是标准的 Seq2Seq Attention 机制

2. **Encoder Self-Attention**：Q/K/V 全部来自同一 Encoder 层的输出。Encoder 的每个位置关注前一层的所有位置。这是双向的（没有 mask），允许每个 token 看到完整上下文

3. **Decoder Masked Self-Attention**：与 Encoder 类似但加上了因果掩码（causal mask）——每个位置只能关注当前位置及之前的 token。通过将非法位置的 softmax 输入设为 -∞ 实现。这保证了自回归生成特性

### 2.5 Position-wise Feed-Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

这是一个包含一个隐藏层的 MLP，使用 ReLU 激活（后来 GPT 改为 GELU）：
- 输入/输出维度：d_model = 512
- 隐藏层维度：d_ff = 2048（d_model 的 4 倍）

FFN 的特点是"position-wise"——对序列中的每个位置使用**相同**的线性变换（参数在不同位置间共享），但不同层使用不同的参数。这等价于两个 kernel size=1 的卷积层。

Big 模型的 FFN 维度为 4096（d_model 的 4 倍），这也是后续模型（如 GPT）沿用的 4x 比例。

### 2.6 Positional Encoding：序列顺序的来源

由于 Transformer "contains no recurrence and no convolution"，必须注入位置信息。原论文使用**正弦函数编码**：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**为什么选择这个形式？**

1. 波长从 2π 到 10000·2π 呈几何级数——不同维度以不同"频率"振荡，低维度编码短距离位置，高维度编码长距离
2. 对任意偏移 k，PE_{pos+k} 可以表示为 PE_{pos} 的线性函数（因为 sin(α+β)=sinα·cosβ+cosα·sinβ）——"the model can easily learn to attend by relative positions"
3. 允许外推到比训练时更长的序列（相比可学习位置编码）

消融实验（Table 3 行 E）显示，替换为可学习位置编码后结果几乎相同（BLEU 25.8 vs 25.7），表明两种编码方式在实际效果上等价。作者选择正弦版本主要是为了外推能力。

### 2.7 参数配置对比

| 配置 | 参数量 | d_model | h | d_k | d_ff | 层数 | P_drop | 训练步数 |
|------|--------|---------|---|-----|------|------|--------|---------|
| Base | ~65M | 512 | 8 | 64 | 2048 | 6 | 0.1 | 100K |
| Big | ~213M | 1024 | 16 | 64 | 4096 | 6 | 0.3(EN-DE)/0.1(EN-FR) | 300K |

### 2.8 训练细节

**优化器——Adam with Custom Schedule**：

使用 Adam 优化器，β₁=0.9, β₂=0.98, ε=10⁻⁹。学习率采用**预热 + 衰减**的定制调度：

$$\text{lrate} = d_{model}^{-0.5} \cdot \min(\text{step\_num}^{-0.5}, \text{step\_num} \cdot \text{warmup\_steps}^{-1.5})$$

前 warmup_steps=4000 步线性增加，之后按步数的平方根倒数衰减。这种调度在预热阶段允许早期训练稳定，在后期逐步精细调整。

**正则化**：
- **Dropout**：对每个子层的输出应用 P_drop=0.1 的 dropout（残差连接之前）。同时对嵌入层和位置编码的和也应用 dropout
- **Label Smoothing**：使用 ε_ls=0.1 的标签平滑。原论文指出："This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."

**硬件**：
- 8 块 NVIDIA P100 GPU 单机训练
- Base 模型：每步约 0.4 秒，总 100,000 步（12 小时）
- Big 模型：每步约 1.0 秒，总 300,000 步（3.5 天）

**推理**：
- 最后 5 个 checkpoint 平均（Base）
- 最后 20 个 checkpoint 平均（Big）
- 波束搜索 beam size = 4，长度惩罚 α = 0.6

### 2.9 计算复杂度分析

原论文 Table 1 给出了关键对比：

| 层类型 | 每层复杂度 | 顺序操作数 | 最大路径长度 |
|--------|-----------|-----------|-------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| Recurrent | O(n·d²) | O(n) | O(n) |
| Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
| 受限 Self-Attention | O(r·n·d) | O(1) | O(n/r) |

关键洞察：当 n < d（实际中通常如此），Self-Attention 的 O(n²·d) 实际上小于 RNN 的 O(n·d²)。但 Self-Attention 面临 n² 的内存开销，这启发了后续的 FlashAttention、稀疏注意力等优化。

### 2.10 数据与分词

- **WMT 2014 英德翻译**：约 450 万句子对，BPE 编码，共享源-目标词汇表约 37,000 个 token
- **WMT 2014 英法翻译**：约 3600 万句子对，WordPiece 编码，32,000 词汇表
- 训练批次按近似序列长度排列，每批约 25,000 源 token + 25,000 目标 token

---

## 三、Experiments and Key Findings

### 3.1 机器翻译：BLEU 分数

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本 (FLOPs) |
|------|------------|------------|-----------------|
| ByteNet [18] | 23.75 | — | — |
| Deep-Att + PosUnk [39] | — | 39.2 | 1.0·10²⁰ |
| GNMT + RL [38] | 24.6 | 39.92 | 2.3·10¹⁹ / 1.4·10²⁰ |
| ConvS2S [9] | 25.16 | 40.46 | 9.6·10¹⁸ / 1.5·10²⁰ |
| MoE [32] | 26.03 | 40.56 | 2.0·10¹⁹ / 1.2·10²⁰ |
| **Transformer Base** | **27.3** | **38.1** | **3.3·10¹⁸** |
| **Transformer Big** | **28.4** | **41.8** | **2.3·10¹⁹** |

Transformer Big 在英德翻译上超越所有先前模型（包括集成模型）2+ BLEU，在英法翻译上以不到之前 SOTA 1/4 的训练成本取得新的单模型 SOTA。

### 3.2 消融实验：Table 3 的详细分析

原论文 Table 3 是理解 Transformer 设计选择的关键：

| 实验 | 变动 | 训练 PPL | 验证 BLEU | 参数量 |
|------|------|----------|-----------|--------|
| Base | — | 4.92 | 25.8 | 65M |
| (A) h=1 | 1 个头而不是 8 个 | 5.29 | 24.9 | — |
| (A) h=16 | 16 个头 | 5.01 | 25.8 | — |
| (B) d_k=16 | 减小 key 维度 | 5.16 | 25.1 | 58M |
| (C) d_ff=256 | 减小 FFN 维度 | 6.11 | 23.7 | 36M |
| (C) d_ff=4096 | 增大 FFN 维度 | 4.75 | 26.2 | 90M |
| (D) P_drop=0.0 | 无 dropout | 5.77 | 24.6 | — |
| (D) P_drop=0.2 | 更大 dropout | 4.95 | 25.5 | — |
| (E) 学习位置编码 | 替代正弦 | 4.92 | 25.7 | — |

关键发现：
1. 头数和维度间存在权衡：h=1 差 0.9 BLEU，但太多头也无效
2. 增大 FFN 维度收益显著（+0.4 BLEU），但计算成本增加
3. Dropout 对防止过拟合至关重要

### 3.3 英语成分句法分析：向其他任务的泛化

Transformer 在 Penn Treebank WSJ 数据集（仅 40K 训练句子）上达到 91.3 F1，在 1700 万句子的半监督设置下达到 92.7 F1，优于除 RNN Grammar 外的所有先前模型。这证明了 Transformer 即使在小数据场景下也能泛化。

---

## 四、Limitations and Challenges

### 4.1 O(n²) 的计算瓶颈

Self-Attention 的计算和内存复杂度随序列长度平方增长，这是 Transformer 最根本的局限。对于长文档（如论文全文）、高分辨率图像（视频帧序列）、长时间动作轨迹等场景，朴素的自注意力在实践上不可行。这催生了：
- FlashAttention：通过 IO-aware 算法减少内存读写（见 [[FlashAttention]]）
- 稀疏注意力（Longformer, BigBird）：限制注意力邻近范围
- 分层注意力（Swin Transformer）：窗口内注意力 [[Swin Transformer]]

### 4.2 位置编码的局限性

原始的正弦位置编码是固定的，无法根据训练数据自适应。可学习位置编码（如 BERT 使用的）虽然效果相当但无法外推到更长的序列。后续改进包括：
- RoPE（Rotary Position Embedding）：在注意力计算中融入相对位置信息
- ALiBi（Attention with Linear Biases）：对远距离位置添加线性偏置

### 4.3 缺少局部归纳偏置

Transformer 的 Self-Attention 是全局的——每个 token 都能直接关注所有其他 token。这虽然灵活（不需要 CNN 的局部性先验），但意味着模型需要**大量数据**来学习哪些关系是重要的（如相邻像素应该有更强的关联）。数据不足时会过拟合。

### 4.4 训练稳定性

Post-LN 的设计（残差连接在 LayerNorm 之前）在深层网络中可能导致训练不稳定，尤其是学习率设置不当、权重初始化不合适时。这启发了后来的 Pre-LN 设计（GPT 开始使用的），其在 LayerNorm 后才做残差连接。

---

## 五、Relationship with Subsequent Work / Impact on the Field

Transformer 是深度学习历史上影响最深远的单一架构，衍生出三个主要分支：

| 分支 | 代表性模型 | 核心变体 | VLA 中的角色 |
|------|-----------|---------|-------------|
| Encoder-only（理解） | [[../02_BERT/BERT.md|BERT]], RoBERTa, ALBERT | 双向 Self-Attention + MLM | 视觉编码器（[[../11_ViT/ViT.md|ViT]]）、图像理解 |
| Decoder-only（生成） | [[../03_GPT/GPT.md|GPT]], [[../20_Llama2/Llama2.md|Llama 2/3]], GPT-4 | 因果 Masked Self-Attention + 自回归 | [[../13_CLIP/CLIP.md|OpenVLA]] 骨干、RT-2、GR00T N1 |
| Encoder-Decoder（理解+生成） | T5, PaLI-X, PaliGemma | 双向编码 + 自回归解码 | [[../18_DiT/DiT.md|π0]] 的 VLM 组件, RT-2 |

**VLA 中的直接继承**：
- **RT-2**：骨干 PaLI-X 基于 Encoder-Decoder Transformer
- **OpenVLA**：骨干 Llama 2 基于 Decoder-only Transformer
- **π0**：VLM 使用 PaliGemma（基于 Transformer），Action Expert 使用 DiT（基于 Transformer 的扩散架构）
- **GR00T N1**：使用双向 Transformer 处理视觉-语言-动作
- **Octo**：基于 T5 Encoder-Decoder Transformer

几乎所有现代 VLA 模型的核心问题都是"用哪种 Transformer 变体"，而不是"要不要用 Transformer"。

---

## 六、Implications for You / Hardware Compatibility

- ✅ **从零实现 Transformer 是理解 VLA 的最佳起点**：逐层实现 Scaled Dot-Product Attention、Multi-Head Attention、Positional Encoding、LayerNorm，能帮助你理解所有后续模型
- ⚠️ **注意 O(n²) 复杂度边界**：在 8 块 RTX 4090 上，Base 模型可处理约 2048 token 的序列。超过此长度需使用 FlashAttention、稀疏注意力或窗口注意力
- ✅ **推荐学习路径**：先理解原论文公式 → 阅读 Andrej Karpathy 的 nanoGPT（~300 行 PyTorch 代码）→ 扩展到 ViT/BERT → 理解 VLA 中的多模态变体
- ⚠️ **Big 模型（213M）在单 GPU 上可运行但训练困难**：在 24GB VRAM 的 GPU 上，推理可行但完整训练需要多 GPU 或混合精度训练
- ✅ **理解三种注意力模式**：Self-Attention（编码依赖）、Cross-Attention（融合不同模态）、Causal Attention（自回归生成）是理解所有 VLA 变体的基础
- ❌ **不要在新任务上使用原始 Post-LN**：Pre-LN 在现代实践中更稳定，原始 Transformer 的训练设置（预热、学习率调度）在现代框架中可能需要调整

## PDF

[[Attention Is All You Need 原文.pdf]]
