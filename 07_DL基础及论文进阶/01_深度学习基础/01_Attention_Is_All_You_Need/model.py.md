---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Transformer (Attention Is All You Need) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Transformer (Attention Is All You Need)
=======================================
论文: "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)
核心贡献: 用纯注意力机制完全替代 RNN/LSTM，提出 Scaled Dot-Product Attention、
         Multi-Head Attention、正弦位置编码等关键组件。
架构: Encoder-Decoder，各6层，d_model=512，h=8，d_ff=2048
代码结构:
  1. ScaledDotProductAttention - 核心注意力计算（含数学注释）
  2. MultiHeadAttention - 多头并行注意力
  3. PositionWiseFFN - 位置前馈网络
  4. PositionalEncoding - 正弦位置编码（含公式推导）
  5. EncoderLayer / Encoder - 编码器（Self-Attn + FFN + 残差 + LayerNorm）
  6. DecoderLayer / Decoder - 解码器（Masked-Self-Attn + Cross-Attn + FFN）
  7. Transformer - 完整 Encoder-Decoder 模型

与后续论文的关系:
  - [[../02_BERT/BERT|BERT]] 只用了 Encoder 部分
  - [[../03_GPT/GPT|GPT]] 只用了 Decoder 部分（去掉 Cross-Attention）
  - [[../21_Llama2/Llama 2|Llama 2]] 继承了 Decoder-only 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ==============================================================================
# 1. Scaled Dot-Product Attention —— Transformer 最核心的公式
# ==============================================================================
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    公式: Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    为什么除以 √d_k？
    假设 q 和 k 的分量是独立随机变量，均值为0，方差为1。
    则点积 q·k = Σ q_i·k_i 的均值为0，方差为 d_k。
    当 d_k 较大时，点积的绝对值会很大 → softmax 输出接近 one-hot，
    梯度接近 0（梯度消失）。除以 √d_k 后，方差变为 1，保持梯度良好。

    为什么用点积注意力而非加性注意力？
    点积注意力可以利用高度优化的矩阵乘法库（cuBLAS），
    实际速度远快于加性注意力且更省内存。
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,    # (batch, n_heads, seq_len_q, d_k)
                key: torch.Tensor,      # (batch, n_heads, seq_len_k, d_k)
                value: torch.Tensor,    # (batch, n_heads, seq_len_k, d_v)
                mask: torch.Tensor = None  # (batch, 1, seq_len_q, seq_len_k) 或 None
                ) -> tuple[torch.Tensor, torch.Tensor]:

        d_k = query.size(-1)

        # 步骤1: 计算注意力分数 QK^T
        # (batch, n_heads, seq_len_q, d_k) × (batch, n_heads, d_k, seq_len_k)
        # → (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1))

        # 步骤2: 缩放 —— 核心！除以 √d_k 防止点积过大导致梯度消失
        scores = scores / math.sqrt(d_k)

        # 步骤3: 应用 mask（如果有）
        # - Encoder Self-Attention: 无 mask（每个 token 可以看到所有 token）
        # - Decoder Masked Self-Attention: causal mask（只看当前及之前的 token）
        # - Cross-Attention: 通常无额外 mask（但可能有 padding mask）
        if mask is not None:
            # mask 中值为 True 的位置会被设为 -∞，softmax 后概率为 0
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 步骤4: Softmax 归一化 → 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步骤5: 加权求和 → 最终输出
        # (batch, n_heads, seq_len_q, seq_len_k) × (batch, n_heads, seq_len_k, d_v)
        # → (batch, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


# ==============================================================================
# 2. Multi-Head Attention —— 并行化的注意力
# ==============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    公式:
      MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
      head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

    为什么用 8 个头（h=8）？
    - 每个头 d_k = d_model / h = 64，总计算量 ≈ 单头全维度注意力
    - 不同的头可以关注不同的语义模式：
      有的关注句法关系（动词-宾语），有的关注指代消解（"it" 指向哪个名词）
    - 消融实验: h=1 差 0.9 BLEU，h=16 不再改善甚至轻微下降

    注意区分三种使用场景:
    1. Encoder Self-Attention: Q=K=V 来自同一 Encoder 层，无 mask（双向）
    2. Decoder Masked Self-Attention: Q=K=V 来自 Decoder，带 causal mask
    3. Cross-Attention: Q 来自 Decoder，K=V 来自 Encoder 输出
    """

    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) 必须能被 h ({h}) 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h  # 每个头的维度 = 512/8 = 64

        # 线性投影层：将 d_model 维度映射到 h × d_k = d_model
        # 与原文略有不同，将 Q/K/V 的投影合并为一个矩阵（实践中常用）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 (batch, seq_len, d_model) 拆分为 (batch, h, seq_len, d_k)"""
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将 (batch, h, seq_len, d_k) 合并为 (batch, seq_len, d_model)"""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self,
                query: torch.Tensor,   # (batch, seq_len_q, d_model)
                key: torch.Tensor,     # (batch, seq_len_k, d_model)
                value: torch.Tensor,   # (batch, seq_len_k, d_model)
                mask: torch.Tensor = None
                ) -> torch.Tensor:

        # 线性投影并拆分为多头
        Q = self._split_heads(self.W_q(query))  # (batch, h, seq_len_q, d_k)
        K = self._split_heads(self.W_k(key))    # (batch, h, seq_len_k, d_k)
        V = self._split_heads(self.W_v(value))  # (batch, h, seq_len_k, d_k)

        # Scaled Dot-Product Attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # 合并多头并通过输出投影
        output = self.W_o(self._combine_heads(attn_output))
        return self.dropout(output)


# ==============================================================================
# 3. Position-wise Feed-Forward Network
# ==============================================================================
class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network

    公式: FFN(x) = max(0, x·W_1 + b_1)·W_2 + b_2

    - 对序列中每个位置独立应用相同的线性变换（参数共享）
    - 等价于两个 kernel_size=1 的卷积
    - 隐藏层维度 d_ff = 2048 = 4 × d_model（后来 GPT 等模型沿用此比例）
    - 使用 ReLU 激活（后来 GPT 改为 GELU）
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两层 MLP: d_model → d_ff → d_model
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==============================================================================
# 4. Positional Encoding —— 正弦位置编码
# ==============================================================================
class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Sinusoidal Positional Encoding）

    公式:
      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    为什么选择正弦函数？
    1. 波长从 2π 到 10000·2π 呈几何级数 ——
       低维度编码短距离位置，高维度编码长距离
    2. 对任意偏移 k: PE(pos+k) 可以表示为 PE(pos) 的线性函数
       （因为 sin(α+β) = sinα·cosβ + cosα·sinβ）——
       模型可以容易地学习相对位置关系
    3. 允许外推到比训练时更长的序列（相比可学习位置编码）

    消融实验（Table 3 行 E）: 可学习位置编码效果几乎相同(BLEU 25.8 vs 25.7)
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000):
        super().__init__()

        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # 计算除数项: 10000^(2i / d_model)
        # div_term: (d_model/2,)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度 (2i):   sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度 (2i+1): cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码加到输入嵌入上。
        x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


# ==============================================================================
# 5. Encoder Layer —— Post-LN 设计
# ==============================================================================
class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层

    每层包含两个子层:
    1. Multi-Head Self-Attention（无 mask，双向）
    2. Position-wise FFN

    每个子层后接残差连接 + LayerNorm:
      output = LayerNorm(x + Sublayer(x))
    这是 Post-LN 设计（原始 Transformer 使用）。
    后来的 GPT/LLaMA 改用 Pre-LN: output = x + Sublayer(LayerNorm(x))
    """

    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)   # 注意：Post-LN 在子层之后
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 子层1: Self-Attention + 残差 + LayerNorm (Post-LN)
        attn_output = self.self_attn(x, x, x, mask)  # Q=K=V=x, 无 mask = 双向注意力
        x = self.norm1(x + self.dropout(attn_output))

        # 子层2: FFN + 残差 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ==============================================================================
# 6. Decoder Layer
# ==============================================================================
class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层

    每层包含三个子层:
    1. Masked Multi-Head Self-Attention（因果 mask）
    2. Cross-Attention（Q 来自 Decoder，K/V 来自 Encoder）
    3. Position-wise FFN

    每个子层后接残差连接 + LayerNorm (Post-LN)
    """

    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        # 第一子层: 因果自注意力（只看左侧上下文）
        self.masked_self_attn = MultiHeadAttention(d_model, h, dropout)
        # 第二子层: 交叉注意力（关注 Encoder 输出）
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        # 第三子层: FFN
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,              # Decoder 输入
                enc_output: torch.Tensor,     # Encoder 输出 (K, V 来源)
                src_mask: torch.Tensor = None,      # Encoder 的 padding mask
                tgt_mask: torch.Tensor = None       # Decoder 的因果 mask
                ) -> torch.Tensor:

        # 子层1: Masked Self-Attention（带因果 mask，确保自回归）
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 子层2: Cross-Attention（Q 来自 Decoder，K/V 来自 Encoder）
        cross_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # 子层3: FFN
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ==============================================================================
# 7. 完整 Encoder
# ==============================================================================
class Encoder(nn.Module):
    """堆叠 N 层 EncoderLayer"""

    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048,
                 N: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)  # 最终 LayerNorm

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ==============================================================================
# 8. 完整 Decoder
# ==============================================================================
class Decoder(nn.Module):
    """堆叠 N 层 DecoderLayer"""

    def __init__(self, d_model: int = 512, h: int = 8, d_ff: int = 2048,
                 N: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


# ==============================================================================
# 9. 完整 Transformer 模型
# ==============================================================================
class Transformer(nn.Module):
    """
    完整 Transformer (Encoder-Decoder) 模型

    Base 配置: d_model=512, h=8, d_ff=2048, N=6, 约65M参数
    Big 配置: d_model=1024, h=16, d_ff=4096, N=6, 约213M参数
    """

    def __init__(self,
                 src_vocab_size: int,    # 源语言词表大小
                 tgt_vocab_size: int,    # 目标语言词表大小
                 d_model: int = 512,
                 h: int = 8,
                 d_ff: int = 2048,
                 N: int = 6,
                 max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()

        # 源/目标语言 Embedding（共享权重？原文使用共享，但这里分开以便灵活使用）
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码（源和目标共用同一个正弦编码）
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder 和 Decoder
        self.encoder = Encoder(d_model, h, d_ff, N, dropout)
        self.decoder = Decoder(d_model, h, d_ff, N, dropout)

        # 输出投影: d_model → 目标词表大小
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # 缩放因子: embedding * √d_model（原文 Section 3.4）
        self.d_model = d_model

    def forward(self,
                src: torch.Tensor,      # (batch, src_seq_len)
                tgt: torch.Tensor,      # (batch, tgt_seq_len)
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None
                ) -> torch.Tensor:

        # Encoder 前向
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        enc_output = self.encoder(src_emb, src_mask)

        # Decoder 前向
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)

        # 输出投影
        return self.output_proj(dec_output)


# ==============================================================================
# 工具函数: 生成 causal mask
# ==============================================================================
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    生成因果掩码（上三角矩阵），用于 Decoder 的 Masked Self-Attention。

    mask[i, j] = True  表示位置 i 可以关注位置 j
    mask[i, j] = False 表示位置 i 不能关注位置 j（被遮挡）

    对于自回归生成: 每个位置只能看到自己和之前的位置
    即下三角为 True，上三角为 False
    """
    # (seq_len, seq_len) 的下三角矩阵
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)  # → (1, 1, seq_len, seq_len)


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Transformer 架构演示")
    print("=" * 60)

    # ---- 1. Scaled Dot-Product Attention ----
    print("\n--- 1. Scaled Dot-Product Attention ---")
    attn = ScaledDotProductAttention()
    batch, h, seq_len, d_k = 2, 8, 10, 64
    dummy_q = torch.randn(batch, h, seq_len, d_k)
    dummy_k = torch.randn(batch, h, seq_len, d_k)
    dummy_v = torch.randn(batch, h, seq_len, d_k)
    output, weights = attn(dummy_q, dummy_k, dummy_v)
    print(f"输入 Q: {dummy_q.shape}  K: {dummy_k.shape}  V: {dummy_v.shape}")
    print(f"输出: {output.shape}  注意力权重: {weights.shape}")

    # 验证: 注意力权重每行之和为 1
    print(f"注意力权重每行之和: {weights[0, 0, 0].sum().item():.4f} (应 ≈ 1.0)")

    # ---- 2. Multi-Head Attention ----
    print("\n--- 2. Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model=512, h=8)
    dummy_x = torch.randn(2, 10, 512)
    # Encoder Self-Attention (无 mask)
    enc_out = mha(dummy_x, dummy_x, dummy_x)
    print(f"Encoder Self-Attn  输入: {dummy_x.shape} → 输出: {enc_out.shape}")
    # Decoder Masked Self-Attention (带 causal mask)
    causal_mask = generate_causal_mask(10)
    dec_out = mha(dummy_x, dummy_x, dummy_x, causal_mask)
    print(f"Decoder Masked-Attn 输入: {dummy_x.shape} → 输出: {dec_out.shape}")

    # ---- 3. Positional Encoding ----
    print("\n--- 3. 正弦位置编码 ---")
    pe = PositionalEncoding(d_model=512, max_len=100)
    dummy_seq = torch.zeros(1, 50, 512)
    pos_out = pe(dummy_seq)
    print(f"位置编码后的形状: {pos_out.shape}")
    # 可视化前几个位置的第0维（偶数维 = sin）
    pe_vals = pe.pe[0, :5, 0]  # 取位置 0~4，维度 0
    print(f"前5个位置的第0维 (sin) 值: {pe_vals.tolist()}")
    # 可视化前几个位置的第1维（奇数维 = cos）
    pe_vals_cos = pe.pe[0, :5, 1]
    print(f"前5个位置的第1维 (cos) 值: {pe_vals_cos.tolist()}")

    # ---- 4. 完整 Transformer ----
    print("\n--- 4. 完整 Transformer (Encoder-Decoder) ---")
    src_vocab, tgt_vocab = 1000, 1000
    model = Transformer(src_vocab, tgt_vocab, d_model=512, h=8, N=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    dummy_src = torch.randint(0, src_vocab, (2, 15))  # 源序列
    dummy_tgt = torch.randint(0, tgt_vocab, (2, 12))  # 目标序列
    tgt_mask = generate_causal_mask(12)
    logits = model(dummy_src, dummy_tgt, tgt_mask=tgt_mask)
    print(f"输入 src: {dummy_src.shape}, tgt: {dummy_tgt.shape}")
    print(f"输出 logits: {logits.shape} (batch=2, seq_len=12, vocab_size={tgt_vocab})")

    print("\n" + "=" * 60)
    print("三种注意力模式总结:")
    print("  1. Encoder Self-Attention:  无 mask, Q=K=V, 双向")
    print("  2. Decoder Masked Self-Attn: causal mask, 自回归")
    print("  3. Cross-Attention:         Q=Decoder, K/V=Encoder")
    print("=" * 60)

```
