---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# RMSNorm - 代码实现

> 本文档包含 RMSNorm 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
RMSNorm (Root Mean Square Layer Normalization)
===============================================
论文: "Root Mean Square Layer Normalization" (Zhang & Sennrich, NeurIPS 2019)
核心贡献: 去掉 LayerNorm 的均值中心化步骤（μ=0），仅通过 RMS 统计量归一化。
         计算量减少约 7-15%，效果等价甚至更优。
架构要点: RMS(x) = sqrt(mean(x²))，仅保留可学习 γ 参数（去掉 β）
代码结构:
  1. NumPy 版 LayerNorm —— 作为对比基线
  2. NumPy 版 RMSNorm —— 核心算法（逐步骤演示）
  3. PyTorch 版 RMSNorm —— 实际训练可用的 nn.Module
  4. LayerNorm vs RMSNorm 对比演示 —— 数值验证
  5. Pre-LN 用法演示 —— Llama 风格归一化位置

与后续论文的关系:
  - [[../04_Layer_Normalization/Layer Normalization|LayerNorm]] 是 RMSNorm 的前身
  - [[../21_Llama2/Llama 2|Llama 2]] 全系列使用 RMSNorm + Pre-LN
  - [[../19_DiT/DiT|DiT]] 在 AdaLN 中使用 RMSNorm 作为基底
"""

import torch
import torch.nn as nn
import numpy as np


# ==============================================================================
# 1. NumPy 版 LayerNorm —— 对比基线
# ==============================================================================
def layer_norm_numpy(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """
    LayerNorm: y = (x - μ) / σ * γ + β

    LayerNorm 需要两遍遍历:
    第一遍: 计算均值 μ
    第二遍: 计算标准差 σ (需要先知道 μ 才能算 (x-μ)²)
    """
    # 步骤1: 计算均值 —— 第一遍遍历
    mean = np.mean(x, axis=-1, keepdims=True)

    # 步骤2: 计算标准差 —— 第二遍遍历（依赖步骤1的均值）
    centered = x - mean
    variance = np.mean(centered ** 2, axis=-1, keepdims=True)
    std = np.sqrt(variance + eps)

    # 步骤3: 归一化 + 缩放 + 平移
    return centered / std * gamma + beta


# ==============================================================================
# 2. NumPy 版 RMSNorm —— 核心算法，单遍遍历
# ==============================================================================
def rms_norm_numpy(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    RMSNorm: y = x / RMS(x) * γ

    核心公式: RMS(x) = sqrt(mean(x²))

    为什么可以去掉均值中心化？
    1. 残差连接自然使激活均值趋近于 0: x' = x + F(x)
    2. 权重衰减间接约束了激活偏移
    3. 实证发现: LayerNorm 的 β 偏置参数梯度极小（≈ 0），β 几乎没有贡献

    关键优势:
    - 单遍计算: 只需计算 x² 的均值，不依赖 μ
    - 不需要 β 参数: 节省参数和计算
    - 隐式惩罚均值偏移: RMS² = σ² + μ²，RMS 天然包含均值信息
    """
    # 步骤1: 计算 RMS —— 一次遍历即可
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    # 步骤2: 归一化 + 缩放（无平移项）
    return x / rms * gamma


# ==============================================================================
# 3. PyTorch 版 RMSNorm —— 训练可用的 nn.Module
# ==============================================================================
class RMSNorm(nn.Module):
    """
    生产级 RMSNorm 实现（与 Llama 官方实现一致）

    参数:
      hidden_size: 归一化维度（通常是 d_model）
      eps: 数值稳定项。Llama 官方推荐 1e-5（而非常见的 1e-8）
           因为在 bf16 下，RMS 值可能很小，更大的 eps 更安全

    与 LayerNorm 的关键差异:
    | 属性          | LayerNorm               | RMSNorm               |
    |--------------|------------------------|----------------------|
    | 统计量        | μ + σ (两次遍历)         | RMS (一次遍历)          |
    | 可学习参数    | γ + β                  | 仅 γ                  |
    | 均值中心化    | 是                      | 否                     |
    | 计算开销      | 基准                    | 快 7-15% (相对 LN 本身) |
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        # 可学习增益参数 γ（初始化全 1）
        # 注意: 没有 β (偏置) 参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, hidden_size) 或 (batch, hidden_size)

        rms = sqrt(mean(x²) + eps)
        output = x / rms * γ
        """
        # 计算 RMS: 对最后一维求 mean(x²)
        # keepdim=True 保证广播形状正确
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化 + 缩放
        # 注意: x.float() 提升精度以避免 bf16 溢出
        return (x.float() / rms * self.weight).type_as(x)


# ==============================================================================
# 演示: LayerNorm vs RMSNorm 数值对比
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RMSNorm vs LayerNorm 对比演示")
    print("=" * 60)

    # ---- 1. NumPy 版基础验证 ----
    print("\n--- 1. NumPy 版 LayerNorm vs RMSNorm ---")
    d = 8
    x_np = np.random.randn(2, 3, d).astype(np.float32)  # 模拟 (batch=2, seq=3, d=8)
    gamma_np = np.ones(d)
    beta_np = np.zeros(d)

    ln_out = layer_norm_numpy(x_np, gamma_np, beta_np)
    rn_out = rms_norm_numpy(x_np, gamma_np)

    print(f"输入形状: {x_np.shape}")
    print(f"LayerNorm 输出均值: {ln_out.mean(axis=-1)}")
    print(f"LayerNorm 输出标准差: {ln_out.std(axis=-1)}")
    print(f"RMSNorm 输出 RMS: {np.sqrt(np.mean(rn_out ** 2, axis=-1))}")

    # ---- 2. 关键验证: RMS² = σ² + μ² (对非零均值信号) ----
    print("\n--- 2. 关键验证: RMS² = σ² + μ² ---")
    # 构造一个有明显均值偏移的信号
    x_offset = np.array([1.0, 2.0, 3.0, 4.0])  # mean=2.5
    mean = np.mean(x_offset)
    std = np.std(x_offset)
    rms_raw = np.sqrt(np.mean(x_offset ** 2))
    print(f"信号: {x_offset}")
    print(f"均值 μ: {mean:.4f}")
    print(f"标准差 σ: {std:.4f}")
    print(f"RMS: {rms_raw:.4f}")
    print(f"验证 RMS² = σ² + μ²: {std**2 + mean**2:.4f} ≈ RMS²: {rms_raw**2:.4f}")
    print("说明: RMS 天然包含均值信息——即使不显式减去均值，RMS 也会因均值偏移而变大")

    # ---- 3. PyTorch 版 RMSNorm ----
    print("\n--- 3. PyTorch 版 RMSNorm (nn.Module) ---")
    d_model = 64
    rms_norm = RMSNorm(hidden_size=d_model, eps=1e-5)
    
    # 模拟真实场景: (batch=4, seq_len=16, d_model=64)
    x_pt = torch.randn(4, 16, d_model)
    with torch.no_grad():
        out_pt = rms_norm(x_pt)
    
    # 验证: 输出的 RMS 应该接近 1（归一化效果）
    out_rms = torch.sqrt(torch.mean(out_pt ** 2, dim=-1))
    print(f"输入形状: {x_pt.shape}")
    print(f"输出 RMS 均值: {out_rms.mean().item():.4f} (应按近 1.0)")
    
    # 对比 PyTorch 内置 LayerNorm
    pt_ln = nn.LayerNorm(d_model, elementwise_affine=True)
    with torch.no_grad():
        out_ln = pt_ln(x_pt)
    ln_std = out_ln.std(dim=-1).mean()
    print(f"LayerNorm 输出标准差均值: {ln_std.item():.4f} (也应接近 1.0)")
    
    # ---- 4. Pre-LN 用法演示（Llama 风格）----
    print("\n--- 4. Pre-LN 用法演示 (Llama 风格) ---")
    print("""
    传统 Post-LN (原始 Transformer):
      x → SubLayer → LayerNorm → residual_add
    
    Pre-LN (Llama 2, RMSNorm):
      x → RMSNorm → SubLayer → residual_add
    
    关键差异:
    - Pre-LN 的梯度路径: output = x + SubLayer(RMSNorm(x))
      恒等项 x 提供直接梯度路径 → 深层模型训练更稳定
    - Post-LN 的梯度路径: output = LayerNorm(x + SubLayer(x))
      梯度经过 LayerNorm → 深层模型容易梯度爆炸/消失
    """)
    
    # 模拟一个简化的 Pre-LN Decoder Block
    class DummyDecoderBlock(nn.Module):
        """简化版 Llama 风格 Decoder Block（仅演示 RMSNorm 位置）"""
        def __init__(self, d_model: int):
            super().__init__()
            # Pre-LN: RMSNorm 放在 Attention/FFN 之前
            self.attn_norm = RMSNorm(d_model)
            self.ffn_norm = RMSNorm(d_model)
            # 模拟 Attention 和 FFN
            self.attn = nn.Linear(d_model, d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Linear(d_model * 4, d_model),
            )
        
        def forward(self, x):
            # Pre-LN: 先归一化，再过子层，再加残差
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x
    
    block = DummyDecoderBlock(d_model)
    x_block = torch.randn(2, 8, d_model)
    out_block = block(x_block)
    print(f"Pre-LN Decoder Block: 输入 {x_block.shape} → 输出 {out_block.shape}")
    print("RMSNorm 位置: 在 Attention 和 FFN 的输入端（Pre-LN）")

    # ---- 5. ε 参数对数值稳定性的影响 ----
    print("\n--- 5. ε 参数敏感性演示 ---")
    # 模拟 bf16 场景：很小的值
    x_small = torch.tensor([1e-10, 1e-9, 1e-8, 1e-7]).reshape(1, 1, 4)
    
    for eps_val in [1e-8, 1e-6, 1e-5]:
        rms_small = RMSNorm(hidden_size=4, eps=eps_val)
        with torch.no_grad():
            out_small = rms_small(x_small)
            out_rms_val = torch.sqrt(torch.mean(out_small ** 2, dim=-1)).item()
        print(f"  eps={eps_val:.0e}: 输出RMS={out_rms_val:.6f}")
    
    print("说明: eps=1e-8 在 bf16 训练中可能导致除零，"+
          "Llama 官方推荐 1e-5")

    print("\n" + "=" * 60)
    print("总结:")
    print("  1. RMSNorm = LayerNorm - 均值中心化 - β 参数")
    print("  2. RMS(x) = sqrt(mean(x²))，单遍计算（vs LN 的两遍）")
    print("  3. Pre-LN: RMSNorm 在子层前 → 更稳定的深层训练")
    print("  4. ε 推荐 1e-5 (bf16 训练) 而非默认的 1e-8")
    print("=" * 60)

```
