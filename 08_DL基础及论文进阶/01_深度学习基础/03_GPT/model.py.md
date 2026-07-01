---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# GPT: Improving Language Understanding by Generative Pre-Training - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
GPT: Improving Language Understanding by Generative Pre-Training
================================================================
论文: "Improving Language Understanding by Generative Pre-Training" (Radford et al., OpenAI 2018)
核心贡献: Decoder-only Transformer 路线奠基，自回归语言建模预训练 + 微调范式。
架构: 12层因果Transformer Decoder, d_model=768, 12heads, 117M参数, 无Cross-Attention
代码结构:
  1. GPTSelfAttention - 因果自注意力 (causal mask)
  2. GPTBlock - Decoder Block (含因果 Self-Attn + FFN, Pre-LN)
  3. GPT - 完整 Decoder-only 语言模型

关键设计选择:
  - 移除了原始 Transformer Decoder 中的 Cross-Attention 层
    （因为没有 Encoder 输出需要关注）
  - 使用 GELU 激活（首次，替代 Transformer 的 ReLU）
  - 可学习位置编码（而非正弦编码）
  - Pre-LN 设计

与 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的关系:
  GPT = Transformer Decoder 减去 Cross-Attention
与 [[../02_BERT/BERT|BERT]] 的区别:
  GPT 单向因果注意力 vs BERT 双向注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 因果自注意力 —— 带 causal mask 的 Multi-Head Self-Attention
# ==============================================================================
class GPTSelfAttention(nn.Module):
    """
    GPT 的因果自注意力

    与原始 Transformer 中 Multi-Head Attention 的区别:
    - 总是使用 causal mask（上三角矩阵为 -∞）
    - Q = K = V = 同一输入（因为没有 Cross-Attention）

    因果 mask 确保位置 i 只能关注位置 j ≤ i:
      自然实现自回归生成——预测"下一个"token 时只能看到之前的内容
    """

    def __init__(self, d_model: int = 768, n_head: int = 12, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.c_attn = nn.Linear(d_model, 3 * d_model)  # 合并 Q/K/V 投影
        self.c_proj = nn.Linear(d_model, d_model)      # 输出投影

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_resid = nn.Dropout(dropout)

        # 注册因果 mask（不会更新，但会随设备移动）
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Q/K/V 投影
        qkv = self.c_attn(x)  # (batch, seq_len, 3*d_model)
        q, k, v = qkv.split(d_model, dim=-1)

        # 拆分为多头: (batch, n_head, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention with causal mask
        # scores: (batch, n_head, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用因果 mask —— GPT 的核心！
        # mask 中 0 的位置会被设为 -∞，确保未来 token 不被关注
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)  # (batch, n_head, seq_len, d_k)

        # 合并多头并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        return self.dropout_resid(self.c_proj(attn_output))


# ==============================================================================
# 2. GPT Transformer Block (Decoder-only)
# ==============================================================================
class GPTBlock(nn.Module):
    """
    GPT 的 Transformer Block (Decoder-only)

    与原始 Transformer 的 Decoder Layer 的区别:
    - 没有 Cross-Attention 子层！（因为无 Encoder 输出）
    - 使用 Pre-LN: x + Sublayer(LayerNorm(x))
    - 使用 GELU 激活（而非 ReLU）

    结构:
      x → LayerNorm → Causal Self-Attn → + x (残差)
      x → LayerNorm → FFN              → + x (残差)
    """

    def __init__(self, d_model: int = 768, n_head: int = 12,
                 d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # Pre-LN: 在注意力之前
        self.attn = GPTSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)  # Pre-LN: 在 FFN 之前

        # FFN: d_model → 4*d_model → d_model, GELU 激活
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT 首次使用 GELU 替代 ReLU，后来成为标配
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN + Causal Self-Attention + 残差
        x = x + self.attn(self.ln1(x))
        # Pre-LN + FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x


# ==============================================================================
# 3. 完整 GPT 模型 (Decoder-only)
# ==============================================================================
class GPT(nn.Module):
    """
    完整 GPT 模型 (Decoder-only)

    GPT-1 配置: 12层, d_model=768, 12heads, d_ff=3072, 117M 参数
    训练数据: BooksCorpus (约7,000本未出版书籍)
    两阶段训练: 无监督预训练 (语言建模) + 有监督微调

    自回归生成公式:
      P(u) = softmax(h_n · W_e^T)     ← 最后的输出投影与 token embedding 共享权重
    """

    def __init__(self,
                 vocab_size: int = 40000,     # BPE 词表，40,000 merges
                 d_model: int = 768,
                 n_head: int = 12,
                 d_ff: int = 3072,
                 n_layer: int = 12,
                 max_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Token Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # 可学习位置编码（GPT 使用可学习而非正弦）
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.drop = nn.Dropout(dropout)

        # 堆叠 Decoder Block
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 语言模型头: d_model → vocab_size（输出预测每个 token 的概率）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 通常将 lm_head 的权重与 token_embed 绑定（weight tying）
        # 这减少了参数量并改善了泛化性能
        self.lm_head.weight = self.token_embed.weight

        self.d_model = d_model
        self.max_len = max_len

        # 初始化权重（GPT 使用 N(0, 0.02)）
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: (batch, seq_len) — token 序列
        returns: (batch, seq_len, vocab_size) — 每个位置的词表 logits
        """
        batch_size, seq_len = input_ids.shape

        # Token + Position Embedding
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_emb = self.token_embed(input_ids)        # (batch, seq_len, d_model)
        pos_emb = self.pos_embed(positions)             # (1, seq_len, d_model)
        x = self.drop(token_emb + pos_emb)

        # 逐层通过 Decoder Block
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # 语言模型头: 预测下一个 token
        return self.lm_head(x)  # (batch, seq_len, vocab_size)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20,
                 temperature: float = 1.0) -> torch.Tensor:
        """简单的自回归生成（贪心解码）"""
        for _ in range(max_new_tokens):
            # 截断到最大长度
            x = input_ids[:, -self.max_len:]
            logits = self.forward(x)  # (batch, seq_len, vocab_size)
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GPT 架构演示 (Decoder-only Transformer)")
    print("=" * 60)

    # 小型 GPT 配置（演示用）
    model = GPT(
        vocab_size=1000,
        d_model=256,
        n_head=8,
        d_ff=1024,
        n_layer=4,
        max_len=128,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GPT 参数量: {total_params:,}")

    # 模拟输入
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))

    # 前向传播
    logits = model(input_ids)
    print(f"\n输入 shape: {input_ids.shape}")
    print(f"输出 logits shape: {logits.shape}")
    print(f"  (batch={batch_size}, seq_len={seq_len}, vocab_size=1000)")
    print(f"\n每个位置输出词表大小的 logits → 用于预测下一个 token")

    # 演示自回归生成
    print("\n--- 自回归生成 ---")
    prompt = torch.randint(1, 1000, (1, 5))  # 5 个 token 的 prompt
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"输入 prompt: {prompt.shape} (batch=1, seq_len=5)")
    print(f"生成结果: {generated.shape} (batch=1, seq_len=15)")

    print("\n--- GPT vs BERT vs Transformer 架构对比 ---")
    print("  GPT:          Decoder-only, 因果注意力, 自回归生成")
    print("  BERT:         Encoder-only, 双向注意力, 预训练+微调")
    print("  Transformer:  Encoder-Decoder, 完整架构, 序列转换")

    print("\n--- GPT 系列演进 ---")
    print("  GPT-1 (2018): 117M, BooksCorpus, 预训练+微调")
    print("  GPT-2 (2019): 1.5B, WebText, 零样本迁移")
    print("  GPT-3 (2020): 175B, 上下文学习/涌现能力")
    print("  GPT-4 (2023): 多模态, 视觉-语言理解")

```
