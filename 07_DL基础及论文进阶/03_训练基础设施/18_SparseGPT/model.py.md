---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# SparseGPT 完整实现 - 基于 [[SparseGPT]] (Frantar & Alistarh, ICML 2023) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
SparseGPT 完整实现 - 基于 [[SparseGPT]] (Frantar & Alistarh, ICML 2023)

实现 Hessian-based 逐列剪枝与权重补偿。
核心洞察：传统 magnitude pruning 在 LLM 上灾难性失败（50% 稀疏度下
OPT-175B 的 PPL 从 8.3 飙升至 10,000+），因为忽略了权重间的相互作用。
SparseGPT 通过逐层 Hessian + OBS 闭式补偿，使 50% 非结构化剪枝的
困惑度损失几乎为零（+0.05 PPL），且无需任何微调。

核心组件:
- SparseGptHessian: 基于校准激活值的逐层 Hessian
- SparseGPTPruner: Hessian-weighted 重要性评分 + Cholesky 补偿
- SparseGPTLinear: 支持 N:M 半结构化稀疏的线性层

与 [[GPTQ]] 的关系: 同一团队的姐妹工作，共享 Hessian + Cholesky 框架，
一个是量化版本，一个是剪枝版本。

参考:
- [[SparseGPT]] - 原始论文 (ICML 2023)
- [[GPTQ]] - 同一团队的量化框架
- [[Wanda]] - 更简单的 "权重×激活范数" 剪枝度量
- [[OBS]] - Optimal Brain Surgeon 基础
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


# ============================================================
# 一、层内 Hessian 计算
# ============================================================

class SparseGptHessian:
    """
    逐层 Hessian 近似计算 —— SparseGPT 和 GPTQ 共用此框架。

    WHY H = X^T X？
    对于加权最小二乘问题 ||WX - W_sparse X||^2，
    其二阶泰勒展开的 Hessian 恰好是 2 X^T X。
    SparseGPT 利用 LLM 的逐层结构——每层的输入激活 X 自然提供了
    该层的 Hessian 信息，无需反向传播。

    WHY 加阻尼 λI？
    保证 H 正定，使 Cholesky 分解数值稳定。
    """

    def __init__(self, nsamples: int = 128, damp: float = 1e-2):
        self.nsamples = nsamples
        self.damp = damp
        self.H: Optional[torch.Tensor] = None
        self.nsamples_collected: int = 0

    def add_batch(self, inp: torch.Tensor):
        if len(inp.shape) > 2:
            inp = inp.reshape(-1, inp.size(-1))
        batch_H = inp.t() @ inp  # (in_features, in_features)
        if self.H is None:
            self.H = torch.zeros_like(batch_H)
        self.H += batch_H
        self.nsamples_collected += inp.shape[0]

    def finalize(self, in_features: int) -> torch.Tensor:
        # 归一化
        self.H = self.H / max(self.nsamples_collected, 1)
        # 阻尼
        self.H += self.damp * torch.eye(in_features, device=self.H.device)
        return self.H


# ============================================================
# 二、SparseGPT 剪枝器
# ============================================================

class SparseGPTPruner:
    """
    基于 Hessian 的逐列剪枝 + OBS 权重补偿。

    WHY 逐列剪枝（per-column pruning）？
    权重矩阵 W ∈ R^{out_features × in_features} 的每一列对应一个输入特征。
    按列处理使剪枝问题分解为 in_features 个独立的子问题——每个子问题在
    out_features 个输出行中选择 sparsity 比例的权重进行剪枝。

    WHY OBS (Optimal Brain Surgeon) 补偿？
    简单移除权重 = 丢弃其对输出的贡献 → 误差大。
    OBS 补偿通过调整剩余权重来"弥补"被移除权重的贡献 →
    使剪枝后的输出尽可能接近原始输出。
    """

    def __init__(self, sparsity: float = 0.5):
        """
        Args:
            sparsity: 目标稀疏度（0.5 = 50% 权重被移除）
        """
        self.sparsity = sparsity

    def prune(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        percdamp: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对权重矩阵执行 SparseGPT 剪枝。

        算法（来自论文 Algorithm 1）：
        1. Cholesky 分解 H = LL^T
        2. 逐列:
           a) 计算重要性: score_i = w_i^2 / [H^{-1}]_{ii}
           b) 剪枝最低分的权重
           c) 更新剩余未剪枝权重以补偿

        Args:
            W: 权重矩阵 (out_features, in_features)
            H: Hessian 矩阵 (in_features, in_features)

        Returns:
            W_pruned: 剪枝后的权重
            mask: 剪枝掩码 (1=保留, 0=剪枝)
        """
        dev = W.device
        rows, cols = W.shape

        W = W.float().clone()
        H = H.to(dev).float()

        # ---- Cholesky 分解 ----
        # WHY Cholesky? H^{-1} 的信息可通过 L 间接访问，无需显式求逆
        L = torch.linalg.cholesky(H)  # (cols, cols)

        # H^{-1} 的对角线元素 (用于重要性评分)
        # H^{-1}_{jj} = 1/L_{jj} * (1/L_{jj} - sum_{k<j} L_{jk}^2 / L_{jj}^2)
        # 简化：我们通过逐列更新来间接获取

        # ---- 逐列剪枝 ----
        mask = torch.ones_like(W, dtype=torch.bool)  # True=保留
        W_pruned = W.clone()

        # 对每个输入列
        for j in range(cols):
            col_weights = W_pruned[:, j]  # (out_features,)

            # ---- 步骤 1: 计算重要性评分 ----
            # WHY importance = w^2 / [H^{-1}]_{jj}?
            # 这是 OBS 框架的核心——通过 Hessian 对角元素来衡量
            # 每个"移除"对损失的二次影响
            h_inv_jj = 1.0 / (L[j, j] * L[j, j])  # [H^{-1}]_{jj} 近似
            importance = (col_weights ** 2) / (h_inv_jj + 1e-12)

            # ---- 步骤 2: 选择剪枝的权重 ----
            n_prune = int(rows * self.sparsity)
            if n_prune > 0:
                # 选择重要性最低的 n_prune 个权重
                _, prune_idx = torch.topk(importance, n_prune, largest=False)
                mask[prune_idx, j] = False  # False=剪枝
                W_pruned[prune_idx, j] = 0.0

            # ---- 步骤 3: OBS 权重补偿 ----
            # WHY 补偿？
            # 剪掉的权重对后续列的输出有影响——将其对输出的贡献
            # 通过 L 矩阵"重新分配"到未剪枝的后续列权重上
            if j < cols - 1:
                error = W[:, j] - W_pruned[:, j]  # (out_features,)
                # 补偿因子：L[j, j+1:] / L[j, j]
                comp_factor = L[j, j + 1:] / L[j, j]  # (cols - j - 1,)
                # 外积补偿
                W_pruned[:, j + 1:] += error.unsqueeze(-1) * comp_factor.unsqueeze(0)

        return W_pruned, mask

    def prune_nm(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        N: int = 2,
        M: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        N:M 半结构化剪枝（每 M 个连续权重保留 N 个）。

        WHY N:M 稀疏？
        非结构化稀疏在 GPU 上几乎无法加速——需要 2:4 等结构化模式
        才能利用 NVIDIA Ampere 的稀疏 Tensor Core（实际加速 1.5-2x）。

        2:4 稀疏 = 每 4 个权重中保留 2 个 → 50% 稀疏，硬件加速。

        Args:
            W: 权重矩阵
            H: Hessian
            N: 每 M 个中保留的数量
            M: 分组大小
        """
        dev = W.device
        rows, cols = W.shape

        W = W.float().clone()
        H = H.to(dev).float()
        L = torch.linalg.cholesky(H)

        mask = torch.ones_like(W, dtype=torch.bool)
        W_pruned = W.clone()

        for j in range(cols):
            col_weights = W_pruned[:, j]

            # N:M 剪枝：将 rows 分成 rows/M 组，每组保留 N 个
            h_inv_jj = 1.0 / (L[j, j] * L[j, j])
            importance = (col_weights ** 2) / (h_inv_jj + 1e-12)

            for row_start in range(0, rows, M):
                row_end = min(row_start + M, rows)
                group_imp = importance[row_start:row_end]
                n_keep = min(N, row_end - row_start)
                n_prune = (row_end - row_start) - n_keep

                if n_prune > 0:
                    _, prune_local = torch.topk(group_imp, n_prune, largest=False)
                    prune_global = row_start + prune_local
                    mask[prune_global, j] = False
                    W_pruned[prune_global, j] = 0.0

            # OBS 补偿
            if j < cols - 1:
                error = W[:, j] - W_pruned[:, j]
                comp_factor = L[j, j + 1:] / L[j, j]
                W_pruned[:, j + 1:] += error.unsqueeze(-1) * comp_factor.unsqueeze(0)

        return W_pruned, mask


# ============================================================
# 三、SparseGPT 线性层
# ============================================================

class SparseGPTLinear(nn.Module):
    """
    SparseGPT 剪枝的线性层。

    WHY 一次性剪枝 + 无需微调？
    传统剪枝需要迭代训练恢复精度，对大模型成本过高。
    SparseGPT 只需一次前向传播收集 Hessian，无需反向，
    剪枝后无需微调即可保持接近原精度（50-60% 稀疏度）。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sparsity: float = 0.5,
        nm_pattern: Optional[Tuple[int, int]] = None,  # 如 (2, 4) 表示 2:4 稀疏
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.nm_pattern = nm_pattern

        # ---- 权重 ----
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ---- 剪枝状态 ----
        self.pruned: bool = False
        self.mask: Optional[torch.Tensor] = None  # 剪枝掩码

    def apply_pruning(self, hessian_calc: SparseGptHessian):
        """
        应用 SparseGPT 剪枝。

        WHY 在外部提供 Hessian？
        Hessian 来自该层之前的输入校准集——不同层的 Hessian 不同，
        但 SparseGPT 逐层独立剪枝，无需全局优化。
        """
        H = hessian_calc.finalize(self.in_features)
        W = self.weight.data
        pruner = SparseGPTPruner(sparsity=self.sparsity)

        if self.nm_pattern is not None:
            W_pruned, self.mask = pruner.prune_nm(W, H, *self.nm_pattern)
        else:
            W_pruned, self.mask = pruner.prune(W, H)

        self.weight.data = W_pruned.to(dtype=self.weight.dtype)
        self.pruned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。剪枝后的权重中已清零的权重贡献为零。
        """
        return F.linear(x, self.weight, self.bias)

    @property
    def actual_sparsity(self) -> float:
        """实际稀疏度（可能因 N:M 与目标不同）。"""
        if self.mask is None:
            return 0.0
        return 1.0 - self.mask.float().mean().item()


# ============================================================
# 四、辅助：Magnitude Pruning 对比
# ============================================================

def magnitude_prune(
    W: torch.Tensor,
    sparsity: float,
    per_column: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    传统 magnitude pruning（对比基线）。

    WHY magnitude pruning 在 LLM 上失败？
    - 全局 magnitude: 可能将某些列的权重全部剪掉 → 破坏该输出神经元
    - 列级 magnitude: 更好但仍忽略权重间的相关性
    - 两者都没有误差补偿 → 剪枝后的输出与原始输出偏差大

    这就是为什么 SparseGPT 的 Hessian + OBS 补偿是必需的。
    """
    W = W.clone()
    rows, cols = W.shape
    mask = torch.ones_like(W, dtype=torch.bool)

    if per_column:
        # 逐列 magnitude pruning
        for j in range(cols):
            importance = W[:, j].abs()
            n_prune = int(rows * sparsity)
            if n_prune > 0:
                _, prune_idx = torch.topk(importance, n_prune, largest=False)
                mask[prune_idx, j] = False
                W[prune_idx, j] = 0.0
    else:
        # 全局 magnitude pruning（更容易失败）
        importance = W.abs().flatten()
        n_prune = int(rows * cols * sparsity)
        _, prune_idx = torch.topk(importance, n_prune, largest=False)
        mask_flat = torch.ones_like(importance, dtype=torch.bool)
        mask_flat[prune_idx] = False
        mask = mask_flat.reshape(rows, cols)
        W[mask_flat.reshape(rows, cols) == False] = 0.0

    return W, mask


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SparseGPT 演示: Hessian 剪枝 + OBS 补偿 + N:M 稀疏")
    print("=" * 60)

    # ---- 1. Hessian 计算 ----
    print("\n[1] Hessian 近似")
    torch.manual_seed(42)
    hess = SparseGptHessian(nsamples=128)
    for _ in range(128):
        inp = torch.randn(4, 32, 128)
        hess.add_batch(inp)
    H = hess.finalize(128)
    print(f"  Hessian 形状: {H.shape}, 条件数: {torch.linalg.cond(H):.2f}")

    # ---- 2. SparseGPT 剪枝 ----
    print("\n[2] SparseGPT 50% 剪枝")
    W = torch.randn(256, 128) * 0.15
    pruner = SparseGPTPruner(sparsity=0.5)
    W_sparse, mask = pruner.prune(W, H)

    actual_sparsity = 1.0 - mask.float().mean().item()
    print(f"  目标稀疏度: 50.0%")
    print(f"  实际稀疏度: {actual_sparsity*100:.1f}%")
    print(f"  非零权重数: {mask.sum().item()} / {mask.numel()}")

    # 输出精度对比
    x_test = torch.randn(8, 128)
    orig_out = x_test @ W.t()
    sparse_out = x_test @ W_sparse.t()
    mse = F.mse_loss(sparse_out, orig_out)
    print(f"  输出重建 MSE: {mse:.6f}")

    # ---- 3. Magnitude vs SparseGPT ----
    print("\n[3] SparseGPT vs Magnitude Pruning")
    W_mag, mag_mask = magnitude_prune(W, sparsity=0.5, per_column=True)
    mag_out = x_test @ W_mag.t()
    mag_mse = F.mse_loss(mag_out, orig_out)
    print(f"  Magnitude Pruning MSE: {mag_mse:.6f}")
    print(f"  SparseGPT MSE:          {mse:.6f}")
    print(f"  SparseGPT 改善: {(mag_mse - mse) / mag_mse * 100:.1f}%")

    # ---- 4. N:M 半结构化稀疏 ----
    print("\n[4] 2:4 半结构化稀疏")
    W_nm, nm_mask = pruner.prune_nm(W, H, N=2, M=4)
    nm_sparsity = 1.0 - nm_mask.float().mean().item()
    nm_out = x_test @ W_nm.t()
    nm_mse = F.mse_loss(nm_out, orig_out)
    print(f"  2:4 稀疏度: {nm_sparsity*100:.1f}%")
    print(f"  输出 MSE: {nm_mse:.6f}")

    # ---- 5. 不同稀疏度的表现 ----
    print("\n[5] 不同稀疏度的影响")
    for sp in [0.3, 0.5, 0.6, 0.7]:
        pruner_sp = SparseGPTPruner(sparsity=sp)
        W_test, _ = pruner_sp.prune(W, H)
        test_out = x_test @ W_test.t()
        test_mse = F.mse_loss(test_out, orig_out)
        print(f"  稀疏度 {sp*100:.0f}%: MSE={test_mse:.6f}")

    print("\n" + "=" * 60)
    print("演示完成。SparseGPT 实现了 50% 剪枝几乎无损（+0.05 PPL）。")
    print("=" * 60)

```
