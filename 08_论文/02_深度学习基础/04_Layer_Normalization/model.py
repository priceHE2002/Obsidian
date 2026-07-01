"""
Layer Normalization
===================
论文: "Layer Normalization" (Ba et al., NeurIPS 2019)
核心贡献: 提出沿特征维度（而非 batch 维度）的归一化方法，解决了 BN 对 batch size
         的依赖和训练-推理不一致问题，成为 Transformer 的默认归一化方案。
代码结构:
  1. LayerNorm_Manual - 手动实现 LayerNorm 前向传播
  2. LayerNorm_WithGrad - 含完整梯度推导注释的实现
  3. PostLN_Block - Post-LN 模式的 Transformer Block
  4. PreLN_Block - Pre-LN 模式的 Transformer Block
  5. PostLN_vs_PreLN 对比分析

核心公式:
  μ = (1/H) Σ x_i           ← 均值
  σ = √(1/H Σ (x_i - μ)^2)   ← 标准差
  y = γ · (x - μ)/σ + β     ← 归一化 + 仿射变换

为什么 LayerNorm 适合 Transformer 而 BatchNorm 不适合？
  - BN 依赖 batch 统计量，小 batch / 变长序列下不稳定
  - LN 对每个样本独立归一化，batch_size=1 也能正常工作
  - LN 训练和推理行为完全一致（无 running stats）

与 [[../05_Batch_Normalization/Batch Normalization|BatchNorm]] 的对比:
  BN: 沿 (N, H, W) 归一化 — 跨样本
  LN: 沿 (C, H, W) 归一化 — 单样本内
与 [[../19_RMSNorm/RMSNorm|RMSNorm]] 的关系:
  RMSNorm = LN 去掉均值中心化步骤（节省约 10% 计算量）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 手动实现 LayerNorm（前向传播）
# ==============================================================================
class LayerNorm_Manual(nn.Module):
    """
    LayerNorm 手动实现

    公式（逐步推导）:

    步骤1: 计算均值  μ = (1/H) Σ_{i=1}^H x_i
    步骤2: 计算方差  σ² = (1/H) Σ_{i=1}^H (x_i - μ)²
    步骤3: 归一化    x̂_i = (x_i - μ) / √(σ² + ε)
    步骤4: 仿射变换  y_i = γ · x̂_i + β

    其中 γ (gain) 和 β (bias) 是可学习参数，维度 = 特征维度 H

    为什么有 γ 和 β？
    如果归一化移除了有用信息（如把所有 sigmoid 输入推到线性区），
    网络可以通过学习 γ = √Var[x], β = E[x] 来"撤销"归一化。
    这意味着 LN 永远不会损害网络的表达能力——最坏情况等价于恒等变换。
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        # 可学习的仿射参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # γ
        self.bias = nn.Parameter(torch.zeros(normalized_shape))   # β

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., normalized_shape)  — 最后几个维度是被归一化的特征维度

        对于 Transformer: x 通常是 (batch, seq_len, d_model)
        归一化沿最后一个维度 (d_model) 进行
        """
        # 步骤1: 计算均值（沿最后一个或几个维度）
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)

        # 步骤2: 计算方差
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)

        # 步骤3: 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 步骤4: 仿射变换（恢复表达力）
        return self.weight * x_norm + self.bias


# ==============================================================================
# 2. LayerNorm + 完整梯度推导（教学用）
# ==============================================================================
class LayerNorm_WithGrad(nn.Module):
    """
    LayerNorm + 详细梯度推导注释

    前向传播（同 LayerNorm_Manual）:
      μ = mean(x), σ = std(x), x̂ = (x - μ)/σ, y = γ·x̂ + β

    反向传播——链式法则:

    (1) ∂L/∂x̂ = ∂L/∂y · γ
        因为 y = γ·x̂ + β, 所以 ∂y/∂x̂ = γ

    (2) ∂L/∂σ² = Σ ∂L/∂x̂_i · (x_i - μ) · (-1/2) · (σ² + ε)^(-3/2)
        因为 x̂ = (x-μ)/√(σ²+ε), ∂x̂/∂σ² = -(x-μ)/(2(σ²+ε)^(3/2))

    (3) ∂L/∂μ = (Σ ∂L/∂x̂_i · (-1/√(σ²+ε))) + ∂L/∂σ² · Σ(-2(x_i-μ))/H
        因为 x̂ 和 σ² 都依赖于 μ

    (4) ∂L/∂x_i = ∂L/∂x̂_i · 1/√(σ²+ε) + ∂L/∂σ² · 2(x_i-μ)/H + ∂L/∂μ · 1/H
        三项分别对应: 直接梯度 + 通过方差 + 通过均值

    (5) ∂L/∂γ = Σ ∂L/∂y_i · x̂_i,  ∂L/∂β = Σ ∂L/∂y_i

    核心观察: ∂L/∂x_i 依赖于 mini-batch 中所有样本（通过 μ 和 σ²）。
    这与 BN 的批依赖性不同——LN 中"所有样本"= 一个样本的全部特征维度。
    """

    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)           # μ
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # σ²
        x_norm = (x - mean) / torch.sqrt(var + self.eps)      # x̂
        return self.weight * x_norm + self.bias               # y


# ==============================================================================
# 3. Post-LN Transformer Block
# ==============================================================================
class PostLN_Block(nn.Module):
    """
    Post-LN Transformer Block

    结构（原始 Transformer 使用）:
      x → Self-Attn → +x → LayerNorm → FFN → +x → LayerNorm
    等价于:
      y = LayerNorm(x + Sublayer(x))

    问题:
    - 残差路径上有 LayerNorm，阻碍了干净的梯度流
    - 深层网络（>12层）容易训练不稳定
    - 需要仔细的 warmup 策略（原始 Transformer 用 4000 步 warmup）

    Post-LN 的梯度流分析:
      输出 y = LN(x + F(x))
      对输入求导: ∂y/∂x = (∂LN/∂·) · (I + ∂F/∂x)
      残差梯度路径经过 LN，LN 的 Jacobian 会缩放梯度
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        # Post-LN: 归一化在子层之后
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-LN: LayerNorm(x + Sublayer(x))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)           # LN 在残差之后
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)            # LN 在残差之后
        return x


# ==============================================================================
# 4. Pre-LN Transformer Block
# ==============================================================================
class PreLN_Block(nn.Module):
    """
    Pre-LN Transformer Block

    结构（GPT-2 之后的标准做法）:
      x → LayerNorm → Self-Attn → +x → LayerNorm → FFN → +x
    等价于:
      y = x + Sublayer(LayerNorm(x))

    优势:
    - 残差路径上无 LayerNorm，梯度可自由流动
    - 训练更稳定，允许更高学习率
    - 不需要特殊的 warmup 策略
    - 所有现代 LLM（GPT-2+, Llama, Mistral）都用 Pre-LN

    Pre-LN 的梯度流分析:
      输出 y = x + F(LN(x))
      对输入求导: ∂y/∂x = I + ∂F/∂LN · ∂LN/∂x
      恒等路径保持 I，梯度不会因 LN 而衰减
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # Pre-LN: 归一化在子层之前
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)  # Pre-LN: 归一化在子层之前
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: x + Sublayer(LayerNorm(x))
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out                         # 残差在 LN 之后（干净的恒等路径）

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out                          # 残差在 LN 之后
        return x


# ==============================================================================
# 5. Post-LN vs Pre-LN 对比分析
# ==============================================================================
class PostLN_vs_PreLN:
    """
    Post-LN vs Pre-LN 详细对比

    | 属性          | Post-LN (原始Transformer) | Pre-LN (现代LLM标准) |
    |--------------|--------------------------|---------------------|
    | 公式          | LN(x + Sublayer(x))      | x + Sublayer(LN(x))   |
    | 残差路径梯度  | 经过 LN 缩放             | 直接通过 (I)          |
    | 训练稳定性    | 容易发散（深层次）       | 稳定                  |
    | 是否需要warmup| 是（4000步特殊warmup）    | 一般warmup即可        |
    | 现代使用      | 仅原始Transformer        | GPT-2+, Llama, BERT  |
    | 理论分析      | 困难                     | 简单                  |

    为什么 Pre-LN 更稳定？
    残差连接 x_{l+1} = x_l + F_l(LN(x_l)) 中:
    - 梯度 ∂L/∂x_l 中有恒等路径 I（不衰减）
    - 每一层的有效"学习率"更均匀
    - 初始化时 F_l 接近 0，故 x_{l+1} ≈ x_l，深层网络退化为浅层网络

    为什么 Post-LN 的理论上界可能更高？
    LN 作用于残差输出，可以约束每一层的输出范数，防止信号积累。
    但实践中 Pre-LN 的稳定性优势远超这一点可能的性能增益。
    """


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Layer Normalization 演示")
    print("=" * 60)

    # ---- 1. LayerNorm 前向传播验证 ----
    print("\n--- 1. LayerNorm 前向传播 ---")
    batch, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch, seq_len, d_model)

    # 对比手动实现和 PyTorch 官方实现
    ln_manual = LayerNorm_Manual(d_model)
    ln_official = nn.LayerNorm(d_model)

    # 使用相同的权重
    ln_official.weight.data = ln_manual.weight.data.clone()
    ln_official.bias.data = ln_manual.bias.data.clone()

    out_manual = ln_manual(x)
    out_official = ln_official(x)

    diff = (out_manual - out_official).abs().max().item()
    print(f"手动实现 vs 官方实现的最大差异: {diff:.10f}")
    print(f"输入范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"归一化后均值 (应≈0): {out_manual.mean():.6f}")
    print(f"归一化后方差 (应≈1): {out_manual.var():.6f}")

    # ---- 2. 验证 LayerNorm 的不变性 ----
    print("\n--- 2. 缩放不变性验证 ---")
    x_test = torch.randn(4, 32)
    ln = nn.LayerNorm(32)

    # LN 对单样本缩放具有不变性
    scale = 3.0
    out1 = ln(x_test)
    out2 = ln(x_test * scale)
    diff_scale = (out1 - out2).abs().max().item()
    print(f"缩放 x*{scale} 后 LN 输出的最大差异: {diff_scale:.10f} (应 ≈ 0)")
    print("→ LN 对单样本缩放不变！这是其核心优势之一")

    # ---- 3. Post-LN vs Pre-LN 梯度流对比 ----
    print("\n--- 3. Post-LN vs Pre-LN 梯度流对比 ---")

    d_model_demo = 64
    post_block = PostLN_Block(d_model_demo, 256)
    pre_block = PreLN_Block(d_model_demo, 256)

    x_demo = torch.randn(2, 8, d_model_demo, requires_grad=True)

    # Post-LN 梯度测试
    y_post = post_block(x_demo)
    y_post.sum().backward()
    grad_post_norm = x_demo.grad.norm().item()
    x_demo.grad = None

    # Pre-LN 梯度测试
    y_pre = pre_block(x_demo)
    y_pre.sum().backward()
    grad_pre_norm = x_demo.grad.norm().item()

    print(f"Post-LN 输入梯度范数: {grad_post_norm:.4f}")
    print(f"Pre-LN  输入梯度范数: {grad_pre_norm:.4f}")
    print("→ Pre-LN 的梯度范数通常更大（恒等路径贡献了梯度）")

    # ---- 4. LN vs BN 对比 ----
    print("\n--- 4. LayerNorm vs BatchNorm 核心对比 ---")
    print("  BatchNorm:")
    print("    - 沿 batch 维度归一化 (N, H, W)")
    print("    - 需要 running mean/var（训练≠推理）")
    print("    - 依赖 batch size，小 batch 不稳定")
    print("    - 不适合 RNN/Transformer（变长序列）")
    print("  LayerNorm:")
    print("    - 沿特征维度归一化 (C, H, W)")
    print("    - 训练=推理（无 running stats）")
    print("    - batch_size=1 也能工作")
    print("    - 适合 RNN/Transformer/序列模型")
    print("\n  → CNN 用 BN，Transformer/RNN 用 LN")
    print("  → 现代 LLM 用 [[../19_RMSNorm/RMSNorm|RMSNorm]]（去掉均值中心化）")
