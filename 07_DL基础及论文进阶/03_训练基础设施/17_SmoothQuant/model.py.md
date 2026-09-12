---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# SmoothQuant 完整实现 - 基于 [[SmoothQuant]] (Xiao et al., ICML 2023) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
SmoothQuant 完整实现 - 基于 [[SmoothQuant]] (Xiao et al., ICML 2023)

实现逐通道平滑因子、数学等价变换、INT8 矩阵乘法。
核心洞察：LLM 激活值的量化难度在不同通道间极度不均衡——
通过在权重和激活值之间"平滑"迁移量化难度（引入缩放因子 s），
使所有通道的激活值分布变平滑，实现 W8A8 纯 INT8 计算。

与 [[LLM.int8()]] 的关键区别：
- LLM.int8(): 隔离异常值通道（混合精度，W8A8 但部分 fp16）
- SmoothQuant: 迁移量化难度（纯 INT8，W8A8 全 INT8 GEMM）

核心公式: Y = (X · diag(s)^{-1}) · (diag(s) · W)^T = X̂ · Ŵ^T
        输出不变，但量化难度从激活值迁移到权重上。

参考:
- [[SmoothQuant]] - 原始论文 (ICML 2023)
- [[LLM.int8()]] - W8A8 混合精度前驱
- [[AWQ]] - 同团队的 W4A16 缩放方案
- [[GPTQ]] - 可组合使用的权重量化方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================
# 一、平滑因子计算
# ============================================================

class SmoothFactor:
    """
    逐通道平滑因子的计算与迁移。

    WHY 需要平滑因子？
    LLM 激活值中某些通道（输出通道维度）的幅度可达 ±60，
    而权重的范围通常在 [-1.5, 1.5]。如果直接量化激活值，
    异常值通道迫使 INT8 的量化步长按 ±60 设置，
    正常通道的信息几乎完全被压缩掉。

    SmoothQuant 的思路是"迁移"而非"隔离"：
    - 激活值 X 除以 s → 异常值被缩小 → 激活值量化容易
    - 权重 W 乘以 s → 权重被放大 → 权重量化变难（但仍可控）
    - 输出不变: Y = (X/s) · (W·s)^T = XW^T
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: 迁移强度参数。
                   α=0: 完全不迁移（等价于直接量化激活值）→ 激活值量化最困难
                   α=0.5: 均衡迁移（推荐）→ 权重和激活值量化难度均等
                   α=1.0: 完全迁移到权重 → 权重量化最困难
                   WHY α=0.5? 在 OPT 和 LLaMA 系列上均最优——说明
                   "均衡"是最佳策略，两边都不承受过多量化负担。
        """
        self.alpha = alpha

    def compute(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算逐通道平滑因子 s。

        WHY s_j = max(|X_j|)^α / max(|W_j|)^{1-α}?
        - max(|X_j|) 量化了第 j 个通道激活值的量化难度
        - max(|W_j|) 量化了第 j 个通道权重的量化难度
        - α 控制迁移比例

        Args:
            X: 校准激活值，形状 (n_tokens, in_features)
            W: 权重矩阵，形状 (out_features, in_features)

        Returns:
            s: 平滑因子，形状 (in_features,)
        """
        if X.dim() > 2:
            X = X.reshape(-1, X.size(-1))

        # 通道级别的最大幅度
        x_max = X.abs().max(dim=0).values  # (in_features,)
        w_max = W.abs().max(dim=0).values  # (in_features,)
        # 等价于沿输出通道方向取 max（因为 s_j 逐通道作用于输入维度）

        # 防止除零
        x_max = torch.clamp(x_max, min=1e-12)
        w_max = torch.clamp(w_max, min=1e-12)

        # 平滑因子公式
        s = (x_max ** self.alpha) / (w_max ** (1 - self.alpha))
        # WHY 限制 s 的范围？避免极端缩放导致数值不稳定
        s = torch.clamp(s, min=1e-12)

        return s

    def apply_smooth(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        s: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用平滑变换：X̂ = X / s,  Ŵ = W * s。

        WHY 数学等价？
        X̂Ŵ^T = (X/s)(W·s)^T = XW^T  ← 输出完全不变
        但 X̂ 的幅度分布更均匀（量化友好），Ŵ 的幅度虽有增加
        但在可控范围内。

        Args:
            X: 原始激活值
            W: 原始权重
            s: 平滑因子 (in_features,)
        """
        X_smooth = X / s.unsqueeze(0)
        W_smooth = W * s.unsqueeze(0)
        return X_smooth, W_smooth


# ============================================================
# 二、INT8 对称量化（W8A8）
# ============================================================

class Int8Quantizer:
    """
    INT8 对称量化器——支持 per-tensor, per-token, per-channel 粒度。

    WHY 对称量化（非对称量化需要零点修正）？
    SmoothQuant 平滑后，激活值和权重的分布都接近零中心对称，
    对称量化（无零点）既简单又足够精确。
    """

    @staticmethod
    def quantize_per_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        逐张量对称量化（一个缩放因子）。

        用于 SmoothQuant O1 变体——最快的 W8A8 方案。
        """
        amax = x.abs().max()
        scale = amax / 127.0 if amax > 0 else 1e-12
        x_int8 = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return x_int8, scale

    @staticmethod
    def quantize_per_token(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        逐行对称量化（每个 token 独立缩放）。

        WHY 逐 token？激活值在不同 token 上的幅度差异很大——
        BOS token 可能远小于其他 token。逐 token 量化
        保护了每个 token 的量化精度。

        用于 SmoothQuant O2 变体。
        """
        amax = x.abs().max(dim=-1, keepdim=True).values  # (n_tokens, 1)
        amax = torch.clamp(amax, min=1e-12)
        scale = amax / 127.0
        x_int8 = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
        return x_int8, scale.squeeze(-1)

    @staticmethod
    def quantize_per_channel(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        逐列对称量化（每个输出通道独立缩放）。

        WHY 逐 channel 量化权重？不同输出通道的权重幅度差异大
        （某些输出对应高频 token，权重被训练得大），逐通道量化保留了
        这种结构性差异。
        """
        amax = w.abs().max(dim=1, keepdim=True).values  # (out_features, 1)
        amax = torch.clamp(amax, min=1e-12)
        scale = amax / 127.0
        w_int8 = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        return w_int8, scale.squeeze(-1)


# ============================================================
# 三、SmoothQuant 线性层
# ============================================================

class SmoothQuantLinear(nn.Module):
    """
    SmoothQuant W8A8 线性层。

    WHY W8A8?
    W8A8 可以利用 INT8 Tensor Core 做真正的 INT8 矩阵乘法
    （而非 W4A16 需要反量化为 fp16 再计算），在批量推理场景中
    W8A8 的吞吐可接近 fp16 的 2x。

    三种变体（来自论文 Table 2）：
    - O1: per-tensor X + per-channel W → 最快
    - O2: per-token X + per-channel W → 略精
    - O3: per-token + per-channel X + per-group W → 最精（极少使用）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 0.5,
        variant: str = "O2",
    ):
        """
        Args:
            alpha: 平滑迁移强度（默认 0.5 = 均衡）
            variant: 量化变体 ("O1", "O2", "O3")
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.variant = variant

        # ---- 权重 ----
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ---- SmoothQuant 状态 ----
        self.s: Optional[torch.Tensor] = None  # 平滑因子 (in_features,)
        self.W_smooth: Optional[torch.Tensor] = None  # 平滑后的权重
        self.W_int8: Optional[torch.Tensor] = None  # INT8 量化权重
        self.W_scale: Optional[torch.Tensor] = None  # 权重缩放因子

        # 工具
        self.smoother = SmoothFactor(alpha=alpha)
        self.quantizer = Int8Quantizer()

    def calibrate(self, X_calib: torch.Tensor):
        """
        校准阶段：计算平滑因子，转换并量化权重。

        WHY 校准只需做一次？
        平滑因子 s 是统计量——基于校准集估计后固定不变。
        量化后的权重也在推理中保持不变。
        """
        if X_calib.dim() > 2:
            X_calib = X_calib.reshape(-1, X_calib.size(-1))

        W = self.weight.data

        # ---- 步骤 1: 计算平滑因子 ----
        self.s = self.smoother.compute(X_calib, W)

        # ---- 步骤 2: 应用平滑变换 ----
        # 仅作用于权重（推理时输入侧平滑在线做）
        self.W_smooth = W * self.s.unsqueeze(0)

        # ---- 步骤 3: 量化权重 ----
        self.W_int8, self.W_scale = Int8Quantizer.quantize_per_channel(self.W_smooth)

        print(f"  [SmoothQuant] α={self.alpha}, 变体={self.variant}")
        print(f"  [SmoothQuant] s_j 范围: [{self.s.min():.4f}, {self.s.max():.4f}]")
        print(f"  [SmoothQuant] W 缩放因子范围: [{self.W_scale.min():.6f}, {self.W_scale.max():.6f}]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SmoothQuant W8A8 前向传播。

        计算过程: Y = Q_per_token(X / s) @ Q_per_channel(W * s)^T
                然后反量化输出
        """
        if self.s is None:
            raise RuntimeError("请先调用 calibrate() 完成 SmoothQuant 校准。")

        original_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.size(-1))

        # ---- 步骤 1: 输入侧平滑 ----
        # X_smooth = X / s  (激活值中异常值被缩小)
        x_smooth = x / self.s.unsqueeze(0).to(device=x.device)

        # ---- 步骤 2: 量化激活值 ----
        if self.variant == "O1":
            # Per-tensor 量化（最快）
            x_int8, scale_x = Int8Quantizer.quantize_per_tensor(x_smooth)
            # 扩展到 per-token 以配合逐通道权重输出
            scale_x_vec = torch.full((x_smooth.size(0),), scale_x, device=x.device)
        elif self.variant in ("O2", "O3"):
            # Per-token 量化（每 token 独立缩放）
            x_int8, scale_x_vec = Int8Quantizer.quantize_per_token(x_smooth)
        else:
            raise ValueError(f"未知变体: {self.variant}")

        # ---- 步骤 3: INT8 矩阵乘法 ----
        # 使用 INT8 GEMM（如果可用）或回退到浮点模拟
        if hasattr(torch, '_int_mm'):
            result_int32 = torch._int_mm(x_int8, self.W_int8.t().to(x.device))
        else:
            result_int32 = torch.matmul(x_int8.float(), self.W_int8.t().float().to(x.device))

        # ---- 步骤 4: 反量化 ----
        # Y ≈ scale_x[i] * scale_w[j] * sum(x_int8 * w_int8)
        outer_scale = scale_x_vec.unsqueeze(-1) * self.W_scale.unsqueeze(0).to(x.device)
        output = result_int32.float() * outer_scale

        if self.bias is not None:
            output = output + self.bias

        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], self.out_features)

        return output


# ============================================================
# 四、LayerNorm 融合（SmoothQuant 的关键优化）
# ============================================================

def fuse_layernorm_smooth(layernorm_weight: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    将平滑因子 s 融合到 LayerNorm 的 γ 参数中。

    WHY 融合？
    Transformer 层中: X → LayerNorm(X) → Linear(W) → ...
    LayerNorm 的输出已做了逐通道归一化: X_norm = (X - μ)/σ * γ + β
    将 s 合并到 γ 中: γ' = γ / s
    这意味着平滑操作零额外开销——LayerNorm 本身就要做逐通道缩放。

    这是 SmoothQuant 在性能上的关键优势之一。

    Args:
        layernorm_weight: LayerNorm 的 γ 参数 (in_features,)
        s: 平滑因子 (in_features,)

    Returns:
        融合后的 γ' (in_features,)
    """
    return layernorm_weight / s


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SmoothQuant 演示: 平滑因子 + 数学等价变换 + W8A8 INT8")
    print("=" * 60)

    # ---- 1. 平滑因子计算 ----
    print("\n[1] 平滑因子与量化难度迁移")
    torch.manual_seed(42)

    # 模拟激活值中有异常值的场景
    X = torch.randn(256, 512) * 0.5
    X[:, 50] *= 30.0   # 通道 50: 大幅度异常值
    X[:, 200] *= 25.0  # 通道 200: 大幅异常值

    W = torch.randn(1024, 512) * 0.1

    smoother = SmoothFactor(alpha=0.5)
    s = smoother.compute(X, W)

    # 展示平滑前后的分布变化
    X_smooth, W_smooth = smoother.apply_smooth(X, W, s)

    print(f"  平滑前: 激活值范围 [{X.min():.2f}, {X.max():.2f}], "
          f"权重范围 [{W.min():.4f}, {W.max():.4f}]")
    print(f"  平滑后: 激活值范围 [{X_smooth.min():.2f}, {X_smooth.max():.2f}], "
          f"权重范围 [{W_smooth.min():.4f}, {W_smooth.max():.4f}]")
    print(f"  平滑因子 s 范围: [{s.min():.4f}, {s.max():.4f}]")
    print(f"  异常值通道 50: s=[{s[50].item():.2f}], 激活值 {X[:,50].abs().max():.1f} → "
          f"{X_smooth[:,50].abs().max():.1f}")

    # ---- 2. 数学等价性验证 ----
    print("\n[2] 数学等价性验证")
    x_test = torch.randn(4, 512)
    # 原始输出
    orig_out = F.linear(x_test, W)
    # 平滑后的输出（应完全相等）
    smooth_out = F.linear(x_test / s, W * s.unsqueeze(0))
    eq_error = (orig_out - smooth_out).abs().max().item()
    print(f"  ||原始输出 - 平滑输出||_∞ = {eq_error:.10f}")
    print(f"  数学等价性: {'✓ 成立' if eq_error < 1e-6 else '✗ 不成立（数值误差）'}")

    # ---- 3. W8A8 量化与精度 ----
    print("\n[3] SmoothQuant W8A8 量化精度")
    sq_layer = SmoothQuantLinear(512, 1024, alpha=0.5, variant="O2")
    sq_layer.weight.data = W.clone()

    # 校准
    sq_layer.calibrate(X)

    # 前向对比
    with torch.no_grad():
        fp16_out = F.linear(x_test, sq_layer.weight, sq_layer.bias)
        sq_out = sq_layer(x_test)

    mae = (fp16_out - sq_out).abs().mean().item()
    print(f"  fp16 vs SmoothQuant W8A8 MAE: {mae:.6f}")
    print(f"  输出范围: fp16=[{fp16_out.min():.4f}, {fp16_out.max():.4f}]")

    # ---- 4. SmoothQuant 变体对比 ----
    print("\n[4] SmoothQuant 变体对比")
    for variant in ["O1", "O2"]:
        layer_var = SmoothQuantLinear(512, 1024, alpha=0.5, variant=variant)
        layer_var.weight.data = W.clone()
        layer_var.calibrate(X)
        out_var = layer_var(x_test)
        var_mae = (fp16_out - out_var).abs().mean().item()
        print(f"  {variant}: MAE={var_mae:.6f}")

    # ---- 5. 与 LLM.int8() 的对比 ----
    print("\n[5] SmoothQuant vs LLM.int8() 对比")
    print("  | 维度       | LLM.int8()          | SmoothQuant             |")
    print("  |------------|---------------------|-------------------------|")
    print("  | 处理方式   | 隔离异常值（混合精度）| 迁移难度（纯 INT8）      |")
    print("  | 计算模式   | INT8 + fp16 混合    | 纯 INT8 GEMM            |")
    print("  | 加速比     | 1.0x-1.3x          | 1.7x-2.0x               |")
    print("  | 理论依据   | 涌现特征发现        | 数学等价变换 + 均衡策略  |")

    print("\n" + "=" * 60)
    print("演示完成。SmoothQuant 实现了 W8A8 纯 INT8 推理，接近 2x 加速。")
    print("=" * 60)


```
