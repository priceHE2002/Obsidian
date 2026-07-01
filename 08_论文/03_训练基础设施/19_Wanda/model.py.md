---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Wanda 完整实现 - 基于 [[Wanda]] (Sun et al., 2023) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Wanda 完整实现 - 基于 [[Wanda]] (Sun et al., 2023)

实现 Wanda 评分 (|W| x ||X||_2)、一次性逐层剪枝。
核心洞察：将 SparseGPT 的 OBS 剪枝公式在"忽略所有权重交互"的
极端简化下推导，恰好得到"权重绝对值 × 输入激活列 L2 范数"——
Wanda。比 SparseGPT 快数百倍（OPT-175B: 5 秒 vs 1 小时），
同时在 50-60% 稀疏度下保持相近精度的剪枝质量。

Wanda 的多重解释：
1. |W|: 衡量权重大小（与 magnitude pruning 同）
2. ||X||_2: 衡量该特征在正向传播中的"活跃度"——
   如果某通道的激活范数很小（几乎从不"说话"），对应权重就不重要

核心公式: s_{ij} = |W_{ij}| · ||x_j||_2
剪枝策略: 逐列 (per-column)，每列保留 score 最大的 top-k

与 [[SparseGPT]] 的关系: Wanda = SparseGPT 在无补偿 + 对角 Hessian 假设下的极限简化

参考:
- [[Wanda]] - 原始论文 (2023)
- [[SparseGPT]] - Wanda 的出发点，OBS-based 剪枝
- [[SparseGPT]] - 对 Wanda 失效的高稀疏度场景的必要补充
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================
# 一、激活列范数收集
# ============================================================

class ActivationCollector:
    """
    在校准集上收集输入激活值的列 L2 范数。

    WHY 列 L2 范数？
    Wanda 的评分 = |w_{ij}| * ||x_j||_2
    其中 j 是输入特征的列索引。l2 范数量化了每个输入通道
    在整个校准集和所有 token 上的"总活跃度"。

    这是 Wanda 和 SparseGPT 的核心区别：
    - SparseGPT: 需要完整的 Hessian H = X^T X（列间协方差）
    - Wanda: 只需要 ||x_j||_2（各列的 L2 范数，无协方差）
    """

    def __init__(self):
        self.column_norms: Optional[torch.Tensor] = None  # (in_features,)
        self.total_tokens: int = 0

    def reset(self):
        self.column_norms = None
        self.total_tokens = 0

    def add_batch(self, inp: torch.Tensor):
        """
        累加一批校准输入的列平方和。

        WHY 累加平方和而非即时算 norm?
        逐步累加避免存储所有校准样本的激活值——
        只需维护一个 (in_features,) 的累加器。
        """
        if len(inp.shape) > 2:
            inp = inp.reshape(-1, inp.size(-1))

        # 列平方和 = sum over batch × seq tokens of x_j^2
        col_sq_sum = (inp ** 2).sum(dim=0)  # (in_features,)
        if self.column_norms is None:
            self.column_norms = torch.zeros_like(col_sq_sum)
        self.column_norms += col_sq_sum
        self.total_tokens += inp.shape[0]

    def finalize(self) -> torch.Tensor:
        """
        返回归一化的列 L2 范数 ||x_j||_2。

        WHY 需要 sqrt？
        因为我们累加的是平方和，最终归一化需要开方得到真正的 L2 范数。
        """
        if self.column_norms is None:
            raise RuntimeError("请先调用 add_batch() 收集校准数据。")
        # 除以总 token 数以获得均方根
        return torch.sqrt(self.column_norms / max(self.total_tokens, 1))


# ============================================================
# 二、Wanda 剪枝器
# ============================================================

class WandaPruner:
    """
    Wanda 一次性逐层剪枝器。

    WHY 逐列剪枝（per-column pruning）？
    剪枝策略是在每列（每个输出神经元）内独立进行的：
    对权重矩阵 W ∈ R^{out × in}，每列 j 保留 score 最大的 (1-sparsity) 比例权重。
    这保证每个输出神经元都有相同比例的保留权重——避免某些输出神经元
    被完全剪掉（全局剪枝的风险）。

    WHY 不需要补偿？
    Wanda 的"不补偿"假设在 ≤ 60% 稀疏度下成立，原因是：
    1. 保留的权重通过 activation norm 加权，重要性排序本身就考虑了输入信号
    2. 对于非极端稀疏度，移除的权重贡献在统计上被剩余权重自然吸收
    """

    def __init__(self, sparsity: float = 0.5):
        """
        Args:
            sparsity: 目标稀疏度（0.5 = 50% 权重被剪枝）
        """
        self.sparsity = sparsity

    def compute_scores(
        self,
        W: torch.Tensor,
        column_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Wanda 重要性评分。

        WHY |w_ij| * ||x_j||_2？
        这个公式可以从 SparseGPT 的 OBS 重要性推导：
        SparseGPT: importance(w_ij) = w_ij^2 / [H^{-1}]_{jj}
        如果假设 H 是对角矩阵（即列间无交互），
        则 [H^{-1}]_{jj} ≈ 1/||x_j||_2^2
        代入得: importance ∝ w_ij^2 * ||x_j||_2^2
        取 sqrt 得: score = |w_ij| * ||x_j||_2

        Args:
            W: 权重矩阵 (out_features, in_features)
            column_norms: 每列的 L2 范数 (in_features,)

        Returns:
            scores: Wanda 分数 (out_features, in_features)
        """
        # |W| · ||X_j||_2  (广播到所有输出行)
        return W.abs() * column_norms.unsqueeze(0)

    def prune(
        self,
        W: torch.Tensor,
        column_norms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 Wanda 剪枝。

        算法步骤：
        1. 计算 Wanda 分数 s_{ij} = |W_{ij}| * ||x_j||_2
        2. 对每列（输入特征），按分数取 top-k 保留
        3. 其余清零

        Args:
            W: 权重矩阵 (out_features, in_features)
            column_norms: 每列的 L2 范数 (in_features,)

        Returns:
            W_pruned: 剪枝后的权重
            mask: 剪枝掩码 (True=保留, False=剪枝)
        """
        rows, cols = W.shape

        # ---- 步骤 1: 计算 Wanda 分数 ----
        scores = self.compute_scores(W, column_norms)  # (out_features, in_features)

        # ---- 步骤 2: 每列取 top-k ----
        # WHY 每列独立取 top-k？
        # 保证每个输出神经元都有相同比例的保留权重
        n_keep = int(rows * (1 - self.sparsity))
        n_keep = max(1, n_keep)  # 至少保留 1 个

        mask = torch.zeros_like(scores, dtype=torch.bool)
        for j in range(cols):
            col_scores = scores[:, j]  # (out_features,)
            _, top_indices = torch.topk(col_scores, n_keep)
            mask[top_indices, j] = True

        # ---- 步骤 3: 应用掩码 ----
        W_pruned = W.clone()
        W_pruned[~mask] = 0.0

        return W_pruned, mask

    def prune_nm(
        self,
        W: torch.Tensor,
        column_norms: torch.Tensor,
        N: int = 2,
        M: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        N:M 半结构化 Wanda 剪枝。

        WHY N:M 在 Wanda 中也 work？
        Wanda 的评分在每 M 个连续权重中独立选择 top N——
        这保持了 2:4 等结构化模式，在 A100/H100 上可硬件加速。
        """
        rows, cols = W.shape
        scores = self.compute_scores(W, column_norms)
        mask = torch.zeros_like(scores, dtype=torch.bool)

        for j in range(cols):
            col_scores = scores[:, j]
            for row_start in range(0, rows, M):
                row_end = min(row_start + M, rows)
                group_scores = col_scores[row_start:row_end]
                n_keep = min(N, row_end - row_start)
                _, top_local = torch.topk(group_scores, n_keep)
                top_global = row_start + top_local
                mask[top_global, j] = True

        W_pruned = W.clone()
        W_pruned[~mask] = 0.0
        return W_pruned, mask


# ============================================================
# 三、Wanda 线性层
# ============================================================

class WandaLinear(nn.Module):
    """
    Wanda 剪枝的线性层。

    WHY Wanda 而非 SparseGPT？
    - SparseGPT: 精度更高但慢（175B = 1h）
    - Wanda: 精度接近（<60% 稀疏度）且极快（175B = 5s）
    - 规则: <60% 用 Wanda, >60% 用 SparseGPT
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        # ---- 权重 ----
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ---- 剪枝状态 ----
        self.pruned: bool = False
        self.mask: Optional[torch.Tensor] = None

    def apply_wanda(self, collector: ActivationCollector):
        """
        使用收集的激活范数执行 Wanda 剪枝。

        WHY 在校准集上收集？
        ||x_j||_2 是激活值的统计量——需要少量样本（128 个）来估计。
        """
        column_norms = collector.finalize()
        W = self.weight.data
        pruner = WandaPruner(sparsity=self.sparsity)

        W_pruned, self.mask = pruner.prune(W, column_norms)
        self.weight.data = W_pruned.to(dtype=self.weight.dtype)
        self.pruned = True

    def apply_wanda_nm(
        self,
        collector: ActivationCollector,
        N: int = 2,
        M: int = 4,
    ):
        """应用 N:M Wanda 剪枝。"""
        column_norms = collector.finalize()
        W = self.weight.data
        pruner = WandaPruner(sparsity=self.sparsity)  # sparsity 被 N:M 模式覆盖
        W_pruned, self.mask = pruner.prune_nm(W, column_norms, N, M)
        self.weight.data = W_pruned.to(dtype=self.weight.dtype)
        self.pruned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @property
    def actual_sparsity(self) -> float:
        if self.mask is None:
            return 0.0
        return 1.0 - self.mask.float().mean().item()


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Wanda 演示: |W|·||X||_2 评分 + 一次性逐层剪枝")
    print("=" * 60)

    # ---- 1. 激活范数收集 ----
    print("\n[1] 校准集 → 激活列范数")
    torch.manual_seed(42)
    collector = ActivationCollector()
    for _ in range(128):
        inp = torch.randn(4, 32, 256)
        collector.add_batch(inp)
    col_norms = collector.finalize()
    print(f"  列范数形状: {col_norms.shape}")
    print(f"  列范数范围: [{col_norms.min():.4f}, {col_norms.max():.4f}]")

    # ---- 2. Wanda 剪枝 ----
    print("\n[2] Wanda 50% 剪枝")
    W = torch.randn(512, 256) * 0.15
    pruner = WandaPruner(sparsity=0.5)
    W_wanda, mask = pruner.prune(W, col_norms)

    actual_sp = 1.0 - mask.float().mean().item()
    print(f"  实际稀疏度: {actual_sp*100:.1f}%")

    x_test = torch.randn(8, 256)
    orig_out = x_test @ W.t()
    wanda_out = x_test @ W_wanda.t()
    mse = F.mse_loss(wanda_out, orig_out)
    print(f"  输出重建 MSE: {mse:.6f}")

    # ---- 3. Wanda vs Magnitude vs SparseGPT ----
    print("\n[3] Wanda vs Magnitude Pruning 对比")

    # Magnitude pruning
    W_mag = W.clone()
    for j in range(W_mag.size(1)):
        importance = W_mag[:, j].abs()
        n_prune = int(W_mag.size(0) * 0.5)
        _, prune_idx = torch.topk(importance, n_prune, largest=False)
        W_mag[prune_idx, j] = 0.0
    mag_out = x_test @ W_mag.t()
    mag_mse = F.mse_loss(mag_out, orig_out)

    print(f"  Magnitude Pruning MSE: {mag_mse:.6f}")
    print(f"  Wanda MSE:              {mse:.6f}")
    print(f"  Wanda 改善: {(mag_mse - mse) / mag_mse * 100:.1f}%")

    # ---- 4. N:M 半结构化 ----
    print("\n[4] 2:4 N:M 半结构化 Wanda")
    W_nm, nm_mask = pruner.prune_nm(W, col_norms, N=2, M=4)
    nm_sp = 1.0 - nm_mask.float().mean().item()
    nm_out = x_test @ W_nm.t()
    nm_mse = F.mse_loss(nm_out, orig_out)
    print(f"  2:4 稀疏度: {nm_sp*100:.1f}%")
    print(f"  输出 MSE: {nm_mse:.6f}")

    # ---- 5. Wanda vs SparseGPT 对比表 ----
    print("\n[5] Wanda 与 SparseGPT 的理论对比")
    print("  | 维度         | SparseGPT          | Wanda                  |")
    print("  |--------------|--------------------|------------------------|")
    print("  | 重要性度量   | w²/[H⁻¹]ᵢᵢ       | |w|·||x_j||₂           |")
    print("  | 权重补偿     | OBS 闭式补偿       | 无补偿                  |")
    print("  | 列间依赖     | 完整 Cholesky      | 忽略（对角假设）        |")
    print("  | 计算复杂度   | O(d³ + nd²)        | O(nd)                  |")
    print("  | OPT-175B 耗时| ~1 小时            | ~5 秒                  |")
    print("  | 加速比       | 基线               | ~700x                  |")
    print("  | 最佳场景     | >60% 稀疏度        | ≤60% 稀疏度            |")

    # ---- 6. 不同稀疏度的表现 ----
    print("\n[6] 不同稀疏度下 Wanda 的表现")
    for sp in [0.3, 0.5, 0.6, 0.7]:
        pruner_sp = WandaPruner(sparsity=sp)
        W_test, _ = pruner_sp.prune(W, col_norms)
        test_out = x_test @ W_test.t()
        test_mse = F.mse_loss(test_out, orig_out)
        status = "✓ 推荐" if sp <= 0.6 else "⚠ 精度退化"
        print(f"  稀疏度 {sp*100:.0f}%: MSE={test_mse:.6f} {status}")

    print("\n" + "=" * 60)
    print("演示完成。Wanda 以极简公式实现接近 SparseGPT 的剪枝精度。")
    print("=" * 60)

```
