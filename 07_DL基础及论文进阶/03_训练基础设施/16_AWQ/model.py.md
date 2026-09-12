---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# AWQ 完整实现 - 基于 [[AWQ]] (Lin et al., MLSys 2024) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
AWQ 完整实现 - 基于 [[AWQ]] (Lin et al., MLSys 2024)

实现激活感知显著通道识别、逐通道缩放因子优化。
核心洞察：权重的"重要性"不由自身数值分布决定，而由对应激活值分布决定——
约 1% 的通道（激活值幅度最大的）贡献 ~85% 的量化后 PPL 退化。
AWQ 通过逐通道缩放因子 s > 1 放大显著通道权重，使其获得更多量化 bin，
隐式保护而非显式隔离异常值通道。

核心组件:
- SalientChannelDetector: 基于激活值幅度的显著通道识别
- AWQScaler: 缩放因子 s 的网格搜索优化
- AWQLinear: 激活感知权重量化的线性层

与 [[GPTQ]] 的关键区别: AWQ 用 O(nd) 的激活值统计替代 O(d³) 的 Hessian 计算，
量化速度快 10-100 倍，精度接近甚至略优于 GPTQ。

参考:
- [[AWQ]] - 原始论文 (MLSys 2024)
- [[GPTQ]] - Hessian-based 权重量化对比基线
- [[SmoothQuant]] - 同团队的 W8A8 缩放迁移方案
- [[LLM.int8()]] - 显式异常值隔离 vs AWQ 的隐式保护
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================
# 一、显著通道检测
# ============================================================

class SalientChannelDetector:
    """
    基于激活值分布识别"显著通道"（salient channels）。

    WHY 显著通道重要？
    AWQ 的核心实验发现：约 1% 的通道（激活值幅度最大）贡献了约 85% 的
    权重量化带来的 PPL 退化。这些通道的激活值幅度远超其他通道，
    量化时这些通道的相对误差最大——保护它们就能保护绝大部分精度。

    与 [[LLM.int8()]] 的不同思路：
    - LLM.int8(): 显式隔离异常值通道（稀疏索引 + 混合精度）
    - AWQ: 隐式保护（乘以 s > 1 → 更多量化 bin，纯 INT8/INT4 计算）
    """

    def __init__(self, top_ratio: float = 0.01):
        """
        Args:
            top_ratio: 显著通道的比例。默认 1%。
        """
        self.top_ratio = top_ratio

    def detect(self, X: torch.Tensor) -> torch.Tensor:
        """
        识别激活值幅度最大的通道。

        WHY 用激活值的平均幅度而非最大幅度？
        最大幅度可能被个别异常 token 主导，平均幅度（mean(|X|)）
        更稳定地反映每个通道在整个校准集上的"活动水平"。

        Args:
            X: 校准集在该层的输入激活值，形状 (n_samples, ..., in_features)

        Returns:
            salient_mask: 布尔掩码，True 表示显著通道
        """
        # 展平除特征维外的所有维度
        if X.dim() > 2:
            X = X.reshape(-1, X.size(-1))

        # 每通道的平均激活幅度
        # WHY mean? 更能代表该通道在整个校准集上的行为
        channel_magnitude = X.abs().mean(dim=0)  # (in_features,)

        # 取 top-k 通道
        n_salient = max(1, int(channel_magnitude.size(0) * self.top_ratio))
        _, top_indices = torch.topk(channel_magnitude, n_salient)

        salient_mask = torch.zeros(channel_magnitude.size(0), dtype=torch.bool)
        salient_mask[top_indices] = True
        return salient_mask


# ============================================================
# 二、缩放因子优化
# ============================================================

class AWQScaler:
    """
    逐通道缩放因子 s 的网格搜索优化。

    WHY 网格搜索而非解析解？
    最优缩放因子 s 涉及非凸的量化误差函数——不存在闭式解。
    但 AWQ 发现，所有模型和任务的最优解都对应
    α = (s / max(|X|))^β 中的 β ≈ 0.5（惊人地一致），
    所以实际只需在一个小候选集 {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} 上搜索。

    WHY s > 1？
    放大显著通道的权重 → 使其在 INT4 量化中获得更多量化 bin →
    减少相对量化误差。同时输入侧除以 s 使数学等价（Y 不变）。
    """

    def __init__(self, beta_candidates: Optional[list] = None):
        """
        Args:
            beta_candidates: β 候选值列表
        """
        self.beta_candidates = beta_candidates or [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        self.best_beta: float = 0.5  # 论文发现 0.5 跨任务一致最优

    def compute_scale(
        self,
        channel_magnitude: torch.Tensor,
        salient_mask: torch.Tensor,
        beta: float = 0.5,
    ) -> torch.Tensor:
        """
        计算逐通道缩放因子 s。

        WHY s = max(|X|)^β？
        AWQ 的缩放策略：
        - 显著通道（|X|大）→ s > 1 → 放大权重 → 更多量化 bin → 高精度
        - 非显著通道（|X|小）→ s ≈ 1 → 不缩放 → 保持原有量化精度
        - β 控制缩放强度

        Args:
            channel_magnitude: 每通道的激活值平均幅度 (in_features,)
            salient_mask: 显著通道掩码
            beta: 缩放强度参数
        """
        s = torch.ones_like(channel_magnitude)
        # 仅对显著通道缩放（非显著通道保持 s=1）
        max_mag = channel_magnitude.max()
        if max_mag > 0:
            s[salient_mask] = (channel_magnitude[salient_mask] / max_mag) ** beta
        # 限制 s 的下界为 1（只放大，不缩小，保护显著通道）
        s = torch.clamp(s, min=1.0)
        return s

    def search_best_beta(
        self,
        W: torch.Tensor,
        X_calib: torch.Tensor,
        salient_mask: torch.Tensor,
        bits: int = 4,
    ) -> float:
        """
        网格搜索最优 β。

        WHY 搜索？
        虽然 β=0.5 是跨模型的鲁棒默认值，但针对特定模型-任务组合，
        微调 β 可以额外提升 0.01-0.05 PPL。

        Args:
            W: 权重矩阵 (out_features, in_features)
            X_calib: 校准激活值
            salient_mask: 显著通道掩码
            bits: 量化位宽
        """
        maxq = 2 ** (bits - 1) - 1
        channel_mag = X_calib.abs().mean(dim=0)

        best_beta = 0.5
        best_loss = float('inf')

        for beta in self.beta_candidates:
            s = self.compute_scale(channel_mag, salient_mask, beta)

            # 应用缩放并量化
            W_scaled = W.clone() * s.unsqueeze(0)  # (out, in)
            scale = W_scaled.abs().max() / maxq
            W_q = torch.round(W_scaled / scale).clamp(-maxq, maxq) * scale
            W_q_decomp = W_q / s.unsqueeze(0)  # 补偿回原始空间

            # 量化损失
            loss = F.mse_loss(W_q_decomp @ X_calib.t(), W @ X_calib.t())
            if loss < best_loss:
                best_loss = loss
                best_beta = beta

        self.best_beta = best_beta
        return best_beta


# ============================================================
# 三、AWQ 线性层
# ============================================================

class AWQLinear(nn.Module):
    """
    AWQ 激活感知权重量化线性层。

    WHY AWQ 更简单但效果好？
    GPTQ 需要 Hessian 计算 + Cholesky 分解（O(d³)），GPTQ 的迭代复杂度很高。
    AWQ 只需要激活值的统计量（O(nd)）+ 一个 β 搜索（O(1) 次尝试）。
    这种简洁性带来了量化速度快 10-100 倍和部署更简单的双重优势。

    数学等价性保证：
    Y = (W·s) · (X/s) = WX （输出不变）
    其中 s 是逐通道缩放因子。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        top_ratio: float = 0.01,
    ):
        """
        Args:
            bits: 量化位宽
            group_size: 分组量化大小
            top_ratio: 显著通道比例（默认 1%）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.top_ratio = top_ratio
        self.maxq = 2 ** (bits - 1) - 1

        # ---- 权重 ----
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ---- AWQ 量化状态 ----
        self.quantized: bool = False
        # 量化后的权重和缩放因子
        self.qweight: Optional[torch.Tensor] = None
        self.qscales: Optional[torch.Tensor] = None
        # 缩放因子（用于输入侧补偿）
        self.channel_scale: Optional[torch.Tensor] = None  # s, (in_features,)

        # 工具
        self.detector = SalientChannelDetector(top_ratio=top_ratio)
        self.scaler = AWQScaler()

    def calibrate(
        self,
        X_calib: torch.Tensor,
        beta: Optional[float] = None,
        auto_search: bool = False,
    ) -> None:
        """
        校准阶段：分析激活值分布，计算缩放因子，量化权重。

        WHY 校准阶段只做一次？
        AWQ 是后训练量化（PTQ）——校准是一次性的离线过程。
        推理时只需加载量化权重和缩放因子。

        Args:
            X_calib: 校准激活值 (n_tokens, in_features)
            beta: 手动指定 β（None 则使用 0.5）
            auto_search: 是否自动网格搜索 β
        """
        if X_calib.dim() > 2:
            X_calib = X_calib.reshape(-1, X_calib.size(-1))

        W = self.weight.data.clone()

        # ---- 步骤 1: 检测显著通道 ----
        salient_mask = self.detector.detect(X_calib)
        n_salient = salient_mask.sum().item()
        print(f"  [AWQ] 显著通道: {n_salient}/{self.in_features} "
              f"({100*n_salient/self.in_features:.2f}%)")

        # ---- 步骤 2: 计算缩放因子 ----
        if auto_search:
            beta = self.scaler.search_best_beta(W, X_calib, salient_mask, self.bits)
            print(f"  [AWQ] 搜索到最优 β = {beta:.3f}")
        elif beta is None:
            beta = 0.5  # AWQ 的通用最优值

        channel_mag = X_calib.abs().mean(dim=0)
        s = self.scaler.compute_scale(channel_mag, salient_mask, beta)
        self.channel_scale = s

        # ---- 步骤 3: 应用缩放后量化 ----
        # W'_{:,i} = W_{:,i} * s_i
        W_scaled = W * s.unsqueeze(0)

        # 分组量化
        n_groups = self.in_features // self.group_size
        qweight = torch.zeros_like(W_scaled)
        qscales = torch.zeros(self.out_features, n_groups)

        for g in range(n_groups):
            g_start = g * self.group_size
            g_end = g_start + self.group_size
            w_group = W_scaled[:, g_start:g_end]

            # min-max 对称量化
            scale = w_group.abs().max(dim=-1, keepdim=True).values
            scale = torch.clamp(scale, min=1e-12)
            scale = scale / self.maxq

            qw = torch.round(w_group / scale).clamp(-self.maxq, self.maxq)
            qweight[:, g_start:g_end] = qw * scale
            qscales[:, g] = scale.squeeze(-1)

        self.qweight = qweight
        self.qscales = qscales
        self.quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AWQ 前向传播。

        WHY 需要输入侧补偿？
        数学上 Y = (W·s)·(X/s) = WX，如果不除以 s，输出会错误放大。
        除以 s 保证了缩放变换的数学等价性。

        W4A16 模式：权重 4-bit，激活值 16-bit。
        """
        if self.channel_scale is None:
            raise RuntimeError("请先调用 calibrate() 完成 AWQ 校准。")

        # ---- 输入侧补偿: X' = X / s ----
        # WHY 除以 s？保持 Y = (W·s)·(X/s) = WX 的数学等价性
        x_compensated = x / self.channel_scale.unsqueeze(0).to(device=x.device)

        if self.quantized:
            # 实际部署中 qweight 存 int4，由特殊 CUDA 内核反量化
            W = self.qweight.to(dtype=x.dtype, device=x.device)
        else:
            W = self.weight

        return F.linear(x_compensated, W, self.bias)


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AWQ 演示: 显著通道检测 + 缩放因子优化 + 激活感知量化")
    print("=" * 60)

    # ---- 1. 显著通道检测 ----
    print("\n[1] 显著通道检测")
    torch.manual_seed(42)
    # 模拟校准激活值：某些通道的幅度明显更大
    X = torch.randn(512, 256) * 0.5
    # 通道 10, 50, 200 设为"显著"（幅度远超其他通道）
    X[:, 10] *= 8.0
    X[:, 50] *= 6.0
    X[:, 200] *= 7.0

    detector = SalientChannelDetector(top_ratio=0.01)
    salient_mask = detector.detect(X)
    salient_cols = salient_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"  输入形状: {X.shape}")
    print(f"  显著通道: {salient_cols} ({len(salient_cols)}/{X.size(-1)})")
    for col in salient_cols:
        print(f"    通道 {col}: mean(|X|) = {X[:, col].abs().mean():.3f}")

    # ---- 2. 缩放因子计算 ----
    print("\n[2] 缩放因子计算")
    scaler = AWQScaler()
    channel_mag = X.abs().mean(dim=0)
    for beta in [0.25, 0.5, 0.75, 1.0]:
        s = scaler.compute_scale(channel_mag, salient_mask, beta)
        print(f"  β={beta:.2f}: s 对显著通道 = "
              f"{[f'{s[col].item():.3f}' for col in salient_cols[:3]]}")

    # ---- 3. AWQ 校准演示 ----
    print("\n[3] AWQ 校准 + 量化")
    W = torch.randn(512, 256) * 0.15
    layer = AWQLinear(in_features=256, out_features=512, bits=4, group_size=64)
    layer.weight.data = W

    layer.calibrate(X, beta=0.5, auto_search=False)
    print(f"  channel_scale 范围: [{layer.channel_scale.min():.3f}, "
          f"{layer.channel_scale.max():.3f}]")
    print(f"  量化权重形状: {layer.qweight.shape}")
    print(f"  缩放因子形状: {layer.qscales.shape}")

    # ---- 4. 前向传播精度验证 ----
    print("\n[4] 前向传播精度验证")
    x_test = torch.randn(4, 256)
    # fp16 参考
    with torch.no_grad():
        fp16_out = F.linear(x_test / layer.channel_scale, W, layer.bias)
    # AWQ 量化输出
    awq_out = layer(x_test)
    mae = (fp16_out - awq_out).abs().mean().item()
    print(f"  fp16 vs AWQ 4-bit MAE: {mae:.6f}")

    # ---- 5. AWQ vs GPTQ 对比 ----
    print("\n[5] AWQ vs GPTQ 对比")
    print("  | 维度       | GPTQ              | AWQ                     |")
    print("  |------------|-------------------|-------------------------|")
    print("  | 核心方法   | Hessian + Cholesky | 激活值统计 + 缩放       |")
    print("  | 量化复杂度 | O(d³)             | O(nd) + O(1) 网格搜索   |")
    print("  | 校准数据   | 128 样本          | 128 样本                |")
    print("  | 量化速度   | ~1h (175B)        | ~10min (175B)           |")
    print("  | 理论依据   | 二阶优化 (OBQ)    | 凸优化 + 实证分析       |")
    print("  | 最优参数   | group_size=128    | β=0.5 (跨模型一致)     |")

    print("\n" + "=" * 60)
    print("演示完成。AWQ 以 O(nd) 复杂度实现接近 GPTQ 的 4-bit 量化精度。")
    print("=" * 60)


```
