---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Llama 2 - 代码实现

> 本文档包含 Llama 2 架构的 PyTorch 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Llama 2: Open Foundation and Fine-Tuned Chat Models
=====================================================
论文: "Llama 2" (Touvron et al., Meta AI, 2023)
核心架构: Decoder-only Transformer + RMSNorm (Pre-Norm) + RoPE + SwiGLU + GQA
代码结构:
  1. RMSNorm —— Pre-LN 归一化 (见 [[../20_RMSNorm/RMSNorm|RMSNorm]])
  2. RoPE —— Rotary Position Embedding (旋转位置编码)
  3. SwiGLU —— 门控 FFN (SiLU 激活 + 门控线性单元)
  4. GQA —— Grouped-Query Attention (分组查询注意力)
  5. TransformerBlock —— 完整 Decoder Block
  6. LlamaModel —— 完整 Llama 2 模型 (嵌入 + N层Block + 输出头)

关键设计要点:
  - Pre-Norm: RMSNorm 在 Attention/FFN 之前 → 梯度恒等路径
  - RoPE: 通过旋转矩阵编码相对位置 → 外推能力强
  - SwiGLU: 3 个权重矩阵 (门控+上投影+下投影) → 比 ReLU/GELU 更好
  - GQA: KV heads < Q heads → 减少 KV Cache 显存

与后续论文的关系:
  - OpenVLA 使用 Llama 2 7B 作为 VLA 骨干
  - Llama 3 继承相同架构，GQA 拓展到全系列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. RMSNorm —— Pre-LN 归一化
# ==============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    公式: RMSNorm(x) = x / RMS(x) * γ
    其中 RMS(x) = sqrt(mean(x²) + ε)

    为什么用 RMSNorm 而非 LayerNorm？
    - 去掉均值中心化 → 单遍计算 → 快 7-15%
    - 去掉 β 偏置参数 → 省参数 + 省计算
    - 残差连接使均值自然趋近于 0 → re-centering 不必要
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提升到 fp32 计算以避免 bf16/fp16 的精度问题
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight).type_as(x)


# ==============================================================================
# 2. RoPE —— Rotary Position Embedding (旋转位置编码)
# ==============================================================================
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转频率的复数表示

    RoPE 的核心思想: 将位置信息编码为旋转矩阵
    - 对于位置 m: 将 q, k 的第 (2i, 2i+1) 维度对旋转 m·θ_i
    - θ_i = theta^(-2i/dim) —— 频率呈几何级数分布

    为什么用几何级数频率？
    - 低维度 (小 i): 高频率 → 编码短距离位置 (快速变化)
    - 高维度 (大 i): 低频率 → 编码长距离位置 (慢变化)
    - 类似傅里叶变换的多尺度表示

    为什么用复数表示？
    - 复数的乘法天然实现 2×2 旋转矩阵: e^{imθ}·(x+iy) = 旋转后的向量
    - 将 (x, y) 视为复数 z = x+iy，旋转 z' = z·e^{imθ}
    """
    # 计算频率: θ_i = theta^(-2i/dim) for i = 0, 1, ..., dim/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 位置索引: m = 0, 1, ..., max_seq_len - 1
    t = torch.arange(max_seq_len, dtype=torch.float)
    # 外积: (max_seq_len, dim/2) 的频率矩阵
    freqs = torch.outer(t, freqs)
    # 转换为复数: e^{i·m·θ}
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将预计算的旋转频率应用到 query 和 key 上

    步骤:
    1. 将 xq, xk 的最后维拆分为 (dim/2, 2)，然后视为复数
    2. 与 freqs_cis (旋转因子) 相乘实现旋转
    3. 转回实数表示

    为什么只对 q 和 k 应用 RoPE？
    Attention 分数 = softmax(q·k^T)，位置信息只需在点积中体现
    而 q·k^T 经过旋转矩阵后: (R_m q)·(R_n k)^T = q·(R_m^T R_n)·k
    由于 R_m^T R_n = R_{n-m} (旋转矩阵的性质)，注意力分数只依赖相对位置 (n-m)
    """
    # 将最后维拆分为两半, 视为复数实部和虚部
    # shape: (batch, n_heads, seq_len, dim//2, 2) → (..., dim//2) as complex
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)

    # 广播 freqs_cis 到正确的维度
    freqs_cis = freqs_cis[:xq.size(2)]  # 截取到当前 seq_len
    # (seq_len, dim//2) → (1, 1, seq_len, dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # 旋转: (a+ib) * (cos θ + i sin θ) = (a·cos θ - b·sin θ) + i(a·sin θ + b·cos θ)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================================
# 3. SwiGLU FFN —— 门控前馈网络
# ==============================================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit)

    公式: SwiGLU(x) = (SiLU(x·W_g)) ⊙ (x·W_u) · W_d

    其中:
      SiLU(x) = x · σ(x)  (σ 是 sigmoid, 也称 Swish 激活)
      W_g: d_model → hidden_dim  (门控权重)
      W_u: d_model → hidden_dim  (上投影权重)
      W_d: hidden_dim → d_model  (下投影权重)

    为什么 SwiGLU 比 ReLU/GELU 更好？
    1. 门控机制: SiLU 的非线性 (在 -∞ 不是完全为 0) 保留了更多梯度信息
    2. 三个权重矩阵: 虽然参数量增加 ~50%，但表达能力显著提升
    3. 消融实验: SwiGLU > Swish > GELU > ReLU (在同等 FLOPs 下)

    为什么 hidden_dim = 8/3 * d_model (而非通常的 4×)？
    因为有了第三个权重矩阵 W_g，隐藏层稍小（8/3 ≈ 2.67）就能匹配
    传统 4× FFN 的参数量和 FLOPs。
    """

    def __init__(self, d_model: int, multiple_of: int = 256):
        super().__init__()
        # 计算 hidden_dim: 2/3 * 8 * d_model，并向上取整到 multiple_of 的倍数
        # 这样设计保证参数量与传统 4× d_model 的 FFN 相当
        hidden_dim = int(2 * (4 * d_model) / 3)  # 8/3 * d_model
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_g = nn.Linear(d_model, hidden_dim, bias=False)  # 门控
        self.w_u = nn.Linear(d_model, hidden_dim, bias=False)  # 上投影
        self.w_d = nn.Linear(hidden_dim, d_model, bias=False)  # 下投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门控路径: SiLU(x·W_g) —— 非线性门
        gate = F.silu(self.w_g(x))
        # 上投影路径: x·W_u —— 线性投影
        up = self.w_u(x)
        # 门控融合 + 下投影
        return self.w_d(gate * up)


# ==============================================================================
# 4. GQA —— Grouped-Query Attention (分组查询注意力)
# ==============================================================================
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)

    对比三种注意力机制:
    | 类型 | Q heads | KV heads | KV Cache 显存 | 质量  |
    |------|--------|---------|--------------|------|
    | MHA  | h      | h       | 最大           | 最高  |
    | GQA  | h      | g       | 折中           | ≈MHA |
    | MQA  | h      | 1       | 最小           | 有损  |

    Llama 2 中的使用:
    - 7B/13B: g = h (即 MHA，无分组)
    - 34B/70B: g = 8, h = 64 → 每组 8 个 Q heads 共享 1 个 KV head

    为什么 GQA 对推理如此关键？
    在自回归生成中，每个新 token 都要与历史所有 token 做注意力。
    KV Cache 存储所有历史的 K 和 V 矩阵。
    - MHA (h=64): 每层 KV Cache = 64 × seq_len × d_head
    - GQA (g=8):  每层 KV Cache = 8 × seq_len × d_head  → 减少 8x！

    对于 OpenVLA (7 步动作 × 256 tokens = 1792 tokens)，
    KV Cache 的节省对消费级 GPU 推理至关重要。
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        assert n_heads % n_kv_heads == 0, f"n_heads({n_heads}) 必须被 n_kv_heads({n_kv_heads}) 整除"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV head 对应几个 Q head
        self.head_dim = d_model // n_heads

        # Q 投影: 完整的 n_heads 个
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # K, V 投影: 仅 n_kv_heads 个 (GQA 的核心节约)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        # 输出投影
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 预计算 RoPE 频率
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(self.head_dim, max_seq_len, theta),
            persistent=False
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # 线性投影
        Q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE (只对 Q 和 K)
        Q, K = apply_rotary_emb(Q, K, self.freqs_cis)

        # 扩展 KV heads 以匹配 Q heads (分组共享)
        if self.n_rep > 1:
            # 每个 KV head 复制 n_rep 次
            # (bsz, n_kv_heads, seq, dim) → (bsz, n_kv_heads, 1, seq, dim)
            # → (bsz, n_kv_heads, n_rep, seq, dim) → (bsz, n_heads, seq, dim)
            K = K.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(bsz, self.n_heads, -1, self.head_dim)
            V = V.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(bsz, self.n_heads, -1, self.head_dim)

        # Scaled Dot-Product Attention
        # scores: (bsz, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask  # mask 中 0 → 0, -inf 位置 → -inf

        attn_weights = F.softmax(scores, dim=-1)
        # attn_output: (bsz, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头: (bsz, n_heads, seq_len, head_dim) → (bsz, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.w_o(attn_output)


# ==============================================================================
# 5. TransformerBlock —— 完整 Decoder Block (Pre-Norm)
# ==============================================================================
class TransformerBlock(nn.Module):
    """
    Llama 2 Decoder Block (Pre-Norm 设计)

    数据流:
      x → RMSNorm → Self-Attention (GQA + RoPE) → +x (残差)
        → RMSNorm → SwiGLU FFN → +x (残差)

    为什么是 Pre-Norm 而非 Post-Norm？
    | 方面       | Post-Norm (原版 Transformer)     | Pre-Norm (Llama 2)            |
    |-----------|-------------------------------|-------------------------------|
    | 归一化位置 | 子层之后                       | 子层之前                       |
    | 梯度路径   | LayerNorm(x + SubLayer(x))    | x + SubLayer(RMSNorm(x))      |
    | 训练稳定性 | 深层容易梯度爆炸/消失          | 恒等路径保证梯度畅通           |
    | 学习率     | 需要 warmup + 小心调参         | 允许更高学习率，更鲁棒         |

    直观理解: output = x + F(Norm(x))
    - 反向传播时: gradient 通过恒等路径 (x) 直通
    - 即使 F 的梯度很小，x 的梯度也能正常传播 → 训练深层模型不会退化
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        # Pre-Norm: 注意 RMSNorm 在子层之前
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-Norm: 先归一化，再过 Attention，再加残差
        x = x + self.attn(self.attn_norm(x), mask)
        # Pre-Norm: 先归一化，再过 FFN，再加残差
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ==============================================================================
# 6. LlamaModel —— 完整 Llama 2 模型
# ==============================================================================
class LlamaModel(nn.Module):
    """
    完整 Llama 2 模型 (Decoder-only Transformer)

    架构流程:
      Input tokens → Embedding → [TransformerBlock × N] → RMSNorm → LM Head → Logits

    配置示例:
    | 参数    | 7B      | 13B     | 70B      |
    |--------|---------|---------|---------|
    | d_model| 4096    | 5120    | 8192    |
    | n_heads| 32      | 40      | 64      |
    | n_kv_heads| 32   | 40      | 8       |
    | n_layers| 32     | 40      | 80      |

    注意: 7B/13B 使用 MHA (n_kv_heads = n_heads)，70B 使用 GQA (n_kv_heads = 8)
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, n_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # N 层 Decoder Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, max_seq_len)
            for _ in range(n_layers)
        ])

        # 最终 RMSNorm (在 LM Head 之前)
        self.norm = RMSNorm(d_model)

        # LM Head (输出投影到词表)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 可选: 将 embedding 和 lm_head 权重共享 (tied weights)
        # 在许多 LLM 实践中会这样做以减少参数量

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        mask: (batch, 1, seq_len, seq_len) causal mask, 可选
        """
        # Token Embedding (注意: Llama 没有加 position embedding，RoPE 在 Attention 内部处理)
        x = self.embedding(token_ids)

        # 通过 N 层 Decoder Block
        for layer in self.layers:
            x = layer(x, mask)

        # 最后的 RMSNorm + LM Head
        x = self.norm(x)
        return self.lm_head(x)


# ==============================================================================
# 工具: 生成 Causal Mask
# ==============================================================================
def generate_causal_mask(seq_len: int) -> torch.Tensor:
    """
    生成因果掩码 (自回归生成用)

    返回: (1, 1, seq_len, seq_len) 的 mask
    - mask[i, j] = 0  if j <= i (可以关注)
    - mask[i, j] = -inf if j > i  (不能关注未来的 token)
    """
    # 上三角为 -inf
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Llama 2 架构演示")
    print("=" * 60)

    # ---- 1. RoPE 旋转位置编码 ----
    print("\n--- 1. RoPE 旋转位置编码 ---")
    head_dim = 16
    max_seq = 32
    freqs_cis = precompute_freqs_cis(head_dim, max_seq, theta=10000.0)
    print(f"频率矩阵形状: {freqs_cis.shape} (max_seq={max_seq}, dim/2={head_dim//2})")
    print(f"前3个位置的频率 (theta): {freqs_cis[:3, 0].angle().tolist()}")
    print("说明: 低维度 (dim 0) 旋转最快 → 编码短距离位置关系")

    # 模拟 q, k 并施加 RoPE
    bsz, seq, n_heads = 2, 8, 4
    dummy_q = torch.randn(bsz, n_heads, seq, head_dim)
    dummy_k = torch.randn(bsz, n_heads, seq, head_dim)
    q_rope, k_rope = apply_rotary_emb(dummy_q, dummy_k, freqs_cis)
    print(f"RoPE 后 Q shape: {q_rope.shape}, K shape: {k_rope.shape}")

    # ---- 2. SwiGLU FFN ----
    print("\n--- 2. SwiGLU FFN ---")
    d_model = 128
    ffn = SwiGLUFFN(d_model)
    # 参数量对比
    ffn_params = sum(p.numel() for p in ffn.parameters())
    # 传统 4× FFN 参数量: d_model * (d_model*4) * 2 = 8 * d_model²
    trad_ffn_params = 8 * d_model ** 2
    print(f"SwiGLU FFN 参数量: {ffn_params:,} (d_model={d_model})")
    print(f"传统 4× FFN 参数量: {trad_ffn_params:,}")
    print(f"SwiGLU hidden_dim = {ffn.w_g.out_features} (≈ 8/3 × d_model)")

    dummy_x = torch.randn(2, 16, d_model)
    ffn_out = ffn(dummy_x)
    print(f"SwiGLU FFN 输入: {dummy_x.shape} → 输出: {ffn_out.shape}")

    # ---- 3. GQA (Grouped-Query Attention) ----
    print("\n--- 3. GQA 分组查询注意力 ---")
    n_heads, n_kv_heads = 8, 2  # 4 Q heads 共享 1 KV head
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len=64)
    
    gqa_params = sum(p.numel() for p in gqa.parameters())
    # MHA: 所有 Q/K/V 投影都是 d_model × d_model
    mha_params = 4 * d_model ** 2
    print(f"GQA 参数量 (h={n_heads}, kv_h={n_kv_heads}): {gqa_params:,}")
    print(f"MHA 参数量 (对比): {mha_params:,}")
    print(f"KV 参数节省: {(1 - gqa_params/mha_params)*100:.1f}%")

    causal_mask = generate_causal_mask(16)
    gqa_out = gqa(dummy_x, causal_mask)
    print(f"GQA 输入: {dummy_x.shape} → 输出: {gqa_out.shape}")
    
    # 演示 KV head 共享机制
    print(f"\n  n_rep = {gqa.n_rep} (每个 KV head 被 {gqa.n_rep} 个 Q head 共享)")
    print(f"  Q heads: {n_heads} (完整), KV heads: {n_kv_heads} (共享)")
    print(f"  KV Cache 节省: {(1 - n_kv_heads/n_heads)*100:.0f}%")

    # ---- 4. TransformerBlock ----
    print("\n--- 4. TransformerBlock (Pre-Norm) ---")
    block = TransformerBlock(d_model, n_heads, n_kv_heads)
    block_params = sum(p.numel() for p in block.parameters())
    print(f"单个 Block 参数量: {block_params:,}")

    block_out = block(dummy_x, causal_mask)
    print(f"Block 输入: {dummy_x.shape} → 输出: {block_out.shape}")
    # 验证残差连接 (输入和输出应该在同一量级)
    print(f"输入均值: {dummy_x.mean().item():.4f} → 输出均值: {block_out.mean().item():.4f}")

    # ---- 5. 完整 LlamaModel ----
    print("\n--- 5. 完整 LlamaModel (小型演示) ---")
    demo_config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layers': 4,
        'n_heads': 8,
        'n_kv_heads': 2,
        'max_seq_len': 64
    }
    model = LlamaModel(**demo_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"小型 Llama 配置: {demo_config}")
    print(f"总参数量: {total_params:,}")

    token_ids = torch.randint(0, 1000, (2, 32))  # (batch=2, seq=32)
    mask = generate_causal_mask(32)
    
    with torch.no_grad():
        logits = model(token_ids, mask)
    
    print(f"输入 token_ids: {token_ids.shape}")
    print(f"输出 logits: {logits.shape} (batch=2, seq=32, vocab=1000)")
    
    # 计算下一 token 的预测损失 (演示)
    # 即用位置 i 的 logits 预测位置 i+1 的 token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, demo_config['vocab_size']),
        shift_labels.view(-1)
    )
    print(f"下一 token 预测交叉熵: {loss.item():.4f} (随机初始化, 应 ≈ ln(vocab) ≈ {math.log(demo_config['vocab_size']):.1f})")

    # ---- 6. Llama 2 7B 理论参数量估算 ----
    print("\n--- 6. Llama 2 7B 参数量理论估算 ---")
    config_7b = {
        'd_model': 4096,
        'n_layers': 32,
        'n_heads': 32,
        'n_kv_heads': 32,  # 7B 使用 MHA
        'vocab_size': 32000
    }
    
    d, L, h, V = config_7b['d_model'], config_7b['n_layers'], \
                 config_7b['n_heads'], config_7b['vocab_size']
    
    # Embedding 层
    emb_params = V * d
    # 每层 Attention (Q/K/V/O 投影)
    attn_params_per_layer = 4 * d ** 2
    # 每层 SwiGLU FFN (3 个投影, hidden_dim ≈ 8/3 d)
    hidden_dim = int(2 * 4 * d / 3)
    ffn_params_per_layer = 3 * d * hidden_dim
    # RMSNorm (每层 2 个 + 最后 1 个)
    norm_params = (2 * L + 1) * d
    # LM Head
    lm_head_params = d * V
    
    est_total = (
        emb_params + L * attn_params_per_layer + L * ffn_params_per_layer +
        norm_params + lm_head_params
    )
    print(f"Embedding:       {emb_params/1e9:.2f}B")
    print(f"Attention ({L}层):  {(L * attn_params_per_layer)/1e9:.2f}B")
    print(f"SwiGLU FFN ({L}层): {(L * ffn_params_per_layer)/1e9:.2f}B")
    print(f"RMSNorm:         {norm_params/1e9:.4f}B")
    print(f"LM Head:         {lm_head_params/1e9:.2f}B")
    print(f"估算总参数量:     {est_total/1e9:.2f}B (实际 ~6.74B, 不含 Embedding 共享)")

    print("\n" + "=" * 60)
    print("Llama 2 架构要点总结:")
    print("  1. RMSNorm (Pre-Norm): 归一化在子层之前 → 梯度恒等路径")
    print("  2. RoPE: 旋转位置编码 → 注意力分数只依赖相对位置")
    print("  3. SwiGLU: 门控 FFN (SiLU 激活) → 3 个权重矩阵")
    print("  4. GQA: 分组查询注意力 → KV Cache 节省 (72B 用)")
    print("  5. OpenVLA 骨干 = Llama 2 7B + 动作 token 替换")
    print("=" * 60)

```
