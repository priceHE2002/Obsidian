---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Adapter 实现 - 基于 [[Adapter]] (Houlsby et al., ICML 2019) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Adapter 实现 - 基于 [[Adapter]] (Houlsby et al., ICML 2019)

实现 Houlsby Adapter、Pfeiffer Adapter 和 AdapterFusion 简化版。
Adapter 是 PEFT 领域的奠基性工作：冻结预训练权重，在每层插入瓶颈适配器。

Houlsby Adapter: 在每个子层 (MHA + FFN) 之后插入
Pfeiffer Adapter: 仅在 FFN 之后插入（更精简）
AdapterFusion: 多任务 Adapter 的知识融合

参考：
- [[Adapter]] - Houlsby Bottleneck Adapter (ICML 2019)
- [[AdapterFusion]] (Pfeiffer et al., 2021)
- [[LoRA]] - 后续零延迟 PEFT 方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckAdapter(nn.Module):
    """
    Houlsby Adapter 的瓶颈结构。

    公式: Adapter(x) = x + ReLU(x @ W_down) @ W_up

    - W_down: h -> b (下投影，压缩到瓶颈维度)
    - W_up: b -> h (上投影，恢复到原始维度)
    - 残差连接保证训练稳定性，近零初始化使初始行为接近恒等映射
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏层维度 h
            bottleneck_dim: 瓶颈维度 b，典型值 64
            dropout: dropout 比例
        """
        super().__init__()
        # 下投影: h -> b，将高维特征压缩到瓶颈空间
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        # 上投影: b -> h，从瓶颈空间恢复
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # LayerNorm: 在每个 Adapter 后做归一化，稳定跨任务的训练
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """
        近零初始化策略。
        初始化时 W_down 和 W_up 都接近零，使 Adapter(0) ≈ 0，
        从而初始阶段模型行为 = 原始预训练模型。
        """
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            [batch_size, seq_len, hidden_dim]
        """
        residual = x
        # 下投影 + ReLU 非线性（瓶颈处引入表达能力）
        x = self.down_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        # 上投影恢复维度
        x = self.up_proj(x)
        x = self.dropout(x)
        # 残差连接 + LayerNorm（AdapterLN 变体）
        return self.layer_norm(residual + x)


class HoulsbyAdapterLayer(nn.Module):
    """
    Houlsby Adapter: 在每个 MHA 和 FFN 子层之后各注入一个 Adapter。

    结构:
    x -> MHA -> Adapter1 -> FFN -> Adapter2 -> output

    这是 Adapter 论文的原始设计，通过 2 个 Adapter/层 提供最大灵活性。
    缺点：推理时增加顺序计算，小批量下有显著延迟。
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.attn_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim, dropout)
        self.ffn_adapter = BottleneckAdapter(hidden_dim, bottleneck_dim, dropout)
        # 这两个 Adapter 的参数是唯一可训练的，基模型权重全部冻结
        self.num_trainable_params = (hidden_dim * bottleneck_dim + bottleneck_dim * hidden_dim) * 2

    def forward(self, x: torch.Tensor, attn_fn, ffn_fn) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attn_fn: 冻结的注意力函数
            ffn_fn: 冻结的 FFN 函数
        """
        # Step 1: 注意力 + Houlsby Adapter
        attn_out = attn_fn(x)
        x = self.attn_adapter(x + attn_out)
        # Step 2: FFN + Houlsby Adapter
        ffn_out = ffn_fn(x)
        x = self.ffn_adapter(x + ffn_out)
        return x


class PfeifferAdapterLayer(nn.Module):
    """
    Pfeiffer Adapter: 仅在 FFN 子层之后插入一个 Adapter。

    结构:
    x -> MHA -> FFN -> Adapter -> output

    相比 Houlsby Adapter，参数减半，但仍能捕获任务特定特征。
    Pfeiffer Adapter 在后来的多任务学习中更常用（效率更高）。
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.adapter = BottleneckAdapter(hidden_dim, bottleneck_dim, dropout)
        self.num_trainable_params = hidden_dim * bottleneck_dim + bottleneck_dim * hidden_dim

    def forward(self, x: torch.Tensor, attn_fn, ffn_fn) -> torch.Tensor:
        attn_out = attn_fn(x)
        x = x + attn_out
        ffn_out = ffn_fn(x)
        # 仅一个 Adapter 在 FFN 后
        return self.adapter(x + ffn_out)


class AdapterFusion(nn.Module):
    """
    AdapterFusion 简化版 ([[AdapterFusion]], Pfeiffer et al., 2021)。

    多任务场景中，每个任务有独立 Adapter。AdapterFusion 学习一个
    "知识融合"模块，从多个任务的 Adapter 输出中动态组合知识。

    核心机制:
    - 每个 Adapter 对同一个输入产生不同输出
    - 融合层通过学习注意力权重 a_i 组合这些输出
    - Query = 原始输入 x, Key/Value = 各 Adapter 的输出

    这使模型能够在不同任务之间共享知识，类似于 LoRA 的多任务热切换。
    """

    def __init__(self, hidden_dim: int, num_adapters: int):
        """
        Args:
            hidden_dim: 隐藏维度
            num_adapters: 任务/Adapter 数量
        """
        super().__init__()
        # 融合层的注意力参数
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.num_adapters = num_adapters

    def forward(self, x: torch.Tensor, adapter_outputs: list) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] - 原始输入作为 Query
            adapter_outputs: list of [batch_size, seq_len, hidden_dim]
        Returns:
            [batch_size, seq_len, hidden_dim] - 融合后的输出
        """
        # 将所有 Adapter 输出堆叠: [batch, seq, num_adapters, hidden]
        stacked = torch.stack(adapter_outputs, dim=2)

        # Query: 原始输入决定需要什么知识
        q = self.query(x)  # [batch, seq, hidden]
        # Key: 每个 Adapter 输出
        k = self.key(stacked)  # [batch, seq, num_adapters, hidden]

        # 分数: QK^T 的点积注意
        scores = torch.einsum('bsh,bsnh->bsn', q, k)  # [batch, seq, num_adapters]
        scores = scores / (x.shape[-1] ** 0.5)  # 缩放
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq, num_adapters]

        # 加权融合
        fused = torch.einsum('bsn,bsnh->bsh', attn_weights, stacked)
        return fused


# ============================================================
# 演示代码
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Adapter 实现演示")
    print("参考: Houlsby et al. ICML 2019 / Pfeiffer et al. 2021")
    print("=" * 70)

    batch, seq, hidden, bottleneck = 2, 16, 768, 64

    # 模拟输入
    x = torch.randn(batch, seq, hidden)

    # 模拟冻结的子层（在实际使用中这些是预训练模型的子层）
    def mock_attn(x):
        return 0.1 * torch.randn_like(x)

    def mock_ffn(x):
        return 0.1 * torch.randn_like(x)

    # --- Houlsby Adapter ---
    print("\n[1] Houlsby Adapter")
    houlsby = HoulsbyAdapterLayer(hidden, bottleneck)
    y_houlsby = houlsby(x, mock_attn, mock_ffn)
    print(f"    输入形状: {x.shape}")
    print(f"    输出形状: {y_houlsby.shape}")
    total_params = sum(p.numel() for p in houlsby.parameters())
    print(f"    可训练参数: {total_params:,} "
          f"({total_params / (hidden * hidden) * 100:.2f}% 的等效全参数层)")

    # --- Pfeiffer Adapter ---
    print("\n[2] Pfeiffer Adapter (仅 FFN 后)")
    pfeiffer = PfeifferAdapterLayer(hidden, bottleneck)
    y_pfeiffer = pfeiffer(x, mock_attn, mock_ffn)
    print(f"    输入形状: {x.shape}")
    print(f"    输出形状: {y_pfeiffer.shape}")
    total_params_pf = sum(p.numel() for p in pfeiffer.parameters())
    print(f"    可训练参数: {total_params_pf:,} (Houlsby 的 {total_params_pf/total_params*100:.0f}%)")

    # --- AdapterFusion ---
    print("\n[3] AdapterFusion (3 个任务的输出融合)")
    num_tasks = 3
    fusion = AdapterFusion(hidden, num_tasks)

    # 模拟 3 个任务的 Adapter 各自产生不同输出
    adapter_outputs = [
        torch.tanh(torch.randn(batch, seq, hidden) * 0.1) for _ in range(num_tasks)
    ]

    y_fused = fusion(x, adapter_outputs)
    print(f"    Adapter 数量: {num_tasks}")
    print(f"    融合后输出形状: {y_fused.shape}")

    print("\n" + "=" * 70)
    print("核心对比:")
    print(f"  Houlsby: 每层 2 个 Adapter → {total_params:,} 参数")
    print(f"  Pfeiffer: 每层 1 个 Adapter → {total_params_pf:,} 参数")
    print(f"  LoRA (对比): r=8 时仅 {128*hidden*4:,} 参数，且推理零延迟")
    print("=" * 70)

```
