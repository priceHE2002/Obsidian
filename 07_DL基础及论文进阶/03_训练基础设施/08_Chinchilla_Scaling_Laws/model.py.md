---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Chinchilla Scaling Laws - 代码实现

> 本文档包含 Chinchilla 缩放定律的 NumPy 教学实现：幂律公式、计算最优参数/数据分配。

[Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 的核心结论是：在固定计算预算 C 下，模型参数 N 和训练数据 D 应等比增长（N_opt ∝ C^0.5, D_opt ∝ C^0.5）。下面用 NumPy 实现三种分析方法。

```python
"""
Chinchilla Scaling Laws: 计算最优语言模型
==========================================
NumPy 教学实现 —— 幂律公式推导、IsoFLOPs 曲线、计算最优分配。

论文: [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) (Hoffmann et al., NeurIPS 2022)

核心结论:
  给定计算预算 C（FLOPs），最优参数和数据量:
    N_opt ∝ C^0.5
    D_opt ∝ C^0.5
  即: 训练 tokens ≈ 20 × 模型参数量

推翻的旧结论:
  Kaplan et al. (2020): N_opt ∝ C^0.73, D_opt ∝ C^0.27
  → 错误地认为 "模型越大越好，数据相对不重要"

三种分析方法（互相验证）:
  Approach 1: 固定模型规模，变化数据量 → 拟合 loss(N, D)
  Approach 2: 固定 FLOPs 预算，变化参数量 → 搜索最优 N
  Approach 3: IsoFLOPs 曲线 → 直接拟合参数-数据联合 loss
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict


# ============================================================
# 一、基础公式：Loss 的参数化方程
# ============================================================

def chinchilla_loss(N: np.ndarray, D: np.ndarray,
                    A: float = 406.4, B: float = 410.7,
                    E: float = 1.69, alpha: float = 0.34,
                    beta: float = 0.28) -> np.ndarray:
    """
    Chinchilla Loss 方程（论文公式）。

    L(N, D) = A / N^α + B / D^β + E

    其中:
      A / N^α: 模型容量不足导致的 loss（N → ∞ 时该项 → 0）
      B / D^β: 数据不足导致的 loss（D → ∞ 时该项 → 0）
      E: 不可约 loss（自然语言的固有熵）

    参数来源: 论文 Table A3（在 C4 数据集上的拟合结果）
    α = 0.34: 模型规模指数
    β = 0.28: 数据规模指数

    WHY α > β?
    - 增大模型的收益递减更快（α=0.34）
    - 增加数据的边际回报更大（β=0.28 < α）
    - 因此最优比例是 N ∝ D（两者等比），而非 N ≫ D
    """
    return A / (N ** alpha) + B / (D ** beta) + E


def compute_flops(N: float, D: float) -> float:
    """
    计算训练 FLOPs。

    对 Transformer 模型，前向+反向的 FLOPs 近似:
    C ≈ 6ND

    推导:
      前向: 2ND（一次乘加 = 2 FLOPs × N 参数 × D tokens）
      反向: 4ND（反向计算约为前向的 2 倍）
      总计: 6ND

    来自 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 论文的 Approximations 部分。
    """
    return 6 * N * D


# ============================================================
# 二、Approach 1: 解析推导 N_opt, D_opt
# ============================================================

def compute_optimal_parameters(C: float, alpha: float = 0.34, beta: float = 0.28,
                                A: float = 406.4, B: float = 410.7) -> Tuple[float, float]:
    """
    Approach 1: 在固定 FLOPs 预算 C 下，解析求最优 N 和 D。

    推导过程（拉格朗日乘子法）:
      最小化 L(N, D) = A/N^α + B/D^β + E
      s.t. 6ND = C (FLOPs 约束)

      → 令 ∂L/∂N = ∂L/∂D（约束下求极值）
      → α·A·N^{-(α+1)} ∝ β·B·D^{-(β+1)}
      → 代入 6ND = C，解出:
        N = G · C^(α/(α+β))
        D = (α/β) · G · C^(β/(α+β))

    其中 G 是常数（与 A,B,α,β 有关）。

    代入 α=0.34, α≈0.34, β≈0.28：
      N_opt ∝ C^0.46（论文 Reported）
      D_opt ∝ C^0.54（论文 Reported）

    更精确的值（Approach 3）：
      N_opt ∝ C^0.50, D_opt ∝ C^0.50
      （因为 α 和 β 更接近时取更精确拟合）
    """
    # 数值求解（而非解析，更直观）
    # 尝试一系列 N，计算对应的 D = C / (6N)，求使 L(N, D) 最小的 N
    N_candidates = np.logspace(
        np.log10(C ** 0.3 * 1e-4),
        np.log10(C ** 0.7 * 1e-3),
        1000
    )
    best_loss = float('inf')
    best_N = best_D = None

    for N in N_candidates:
        D = C / (6 * N)
        loss = A / (N ** alpha) + B / (D ** beta) + E
        if loss < best_loss:
            best_loss = loss
            best_N = N
            best_D = D

    return best_N, best_D


def compute_tokens_per_param_ratio() -> float:
    """
    计算计算最优的 tokens-per-parameter 比例。

    来自 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 的核心建议:
      训练 tokens ≈ 20 × 模型参数量

    即：对 1B 参数的模型，需要 ~20B tokens 的训练数据。
    """
    ratios = []
    for logC in np.arange(17, 24, 1):
        C = 10 ** logC
        N_opt, D_opt = compute_optimal_parameters(C)
        ratios.append(D_opt / N_opt)

    return np.median(ratios)


# ============================================================
# 三、Approach 2: 固定 FLOPs 预算，变化参数量
# ============================================================

def isoFLOPs_curve(C: float) -> Dict[str, np.ndarray]:
    """
    给定固定 FLOPs 预算 C，画出 N-Loss 的 U 型曲线。

    这就是 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 中的 Approach 2。
    """
    N_range = np.logspace(
        np.log10(C ** 0.3 * 1e-4),
        np.log10(C ** 0.7 * 1e-3),
        200
    )
    D_range = C / (6 * N_range)
    losses = chinchilla_loss(N_range, D_range)

    # 找最优 N（最小 loss）
    best_idx = np.argmin(losses)

    return {
        'N': N_range,
        'D': D_range,
        'loss': losses,
        'N_opt': N_range[best_idx],
        'D_opt': D_range[best_idx],
        'loss_min': losses[best_idx],
    }


# ============================================================
# 四、Approach 3: 直接拟合参数-数据联合数据表
# ============================================================

def generate_scaling_table():
    """
    生成 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 中的
    表 1: 不同 FLOPs 预算下的最优参数量、数据量和比例。
    """
    print("=" * 80)
    print("Chinchilla 计算最优参数表")
    print("=" * 80)
    print(f"{'FLOPs 预算 C':<18} {'log C':>6} {'N_opt':>18} {'D_opt (tokens)':>18} {'比例 D/N':>10}")
    print("-" * 80)

    for logC in range(17, 25):
        C = 10 ** logC
        N_opt, D_opt = compute_optimal_parameters(C)
        ratio = D_opt / N_opt
        # 格式化 N_opt
        if N_opt >= 1e9:
            N_str = f"{N_opt/1e9:.2f}B"
        elif N_opt >= 1e6:
            N_str = f"{N_opt/1e6:.2f}M"
        else:
            N_str = f"{N_opt:.0f}"

        if D_opt >= 1e12:
            D_str = f"{D_opt/1e12:.2f}T"
        elif D_opt >= 1e9:
            D_str = f"{D_opt/1e9:.2f}B"
        else:
            D_str = f"{D_opt:.0f}"

        print(f"  {C:<14.0e}  {logC:>4d}   {N_str:>16}   {D_str:>16}   {ratio:>8.0f}")

    print("-" * 80)


# ============================================================
# 五、Kaplan vs Chinchilla 对比
# ============================================================

def compare_kaplan_vs_chinchilla(C: float = 1e21):
    """
    对比 Kaplan (2020) 和 Chinchilla (2022) 对同一 FLOPs 预算的建议。

    [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 1.2 节指出
    Kaplan 定律的假设缺陷导致了大模型严重欠训练。
    """
    # Kaplan 定律 (N_opt ∝ C^0.73, D_opt ∝ C^0.27)
    # 用 Kaplan 建议的比例: N/D ≈ 1/1.7（GPT-3 级别）
    N_kaplan = (C / 6) ** 0.73 * 1e-5  # 教学近似
    D_kaplan = C / (6 * N_kaplan)

    # Chinchilla 定律 (N_opt ∝ C^0.5, D_opt ∝ C^0.5)
    N_chinchilla, D_chinchilla = compute_optimal_parameters(C)

    print("\n" + "=" * 70)
    print(f"Kaplan vs Chinchilla @ FLOPs = {C:.0e}")
    print("=" * 70)
    print(f"{'':<15} {'Kaplan (2020)':<25} {'Chinchilla (2022)':<25}")
    print("-" * 70)
    print(f"{'N_opt':<15} {N_kaplan/1e9:>20.2f}B {N_chinchilla/1e9:>20.2f}B")
    print(f"{'D_opt':<15} {D_kaplan/1e9:>20.2f}B {D_chinchilla/1e9:>20.2f}B")
    print(f"{'比例 D/N':<15} {D_kaplan/N_kaplan:>23.0f} {D_chinchilla/N_chinchilla:>23.0f}")
    print(f"{'Loss':<15} {chinchilla_loss(N_kaplan, D_kaplan):>23.4f} "
          f"{chinchilla_loss(N_chinchilla, D_chinchilla):>23.4f}")

    # 欠训练分析
    undertraining = (D_chinchilla - D_kaplan) / D_chinchilla * 100
    print(f"\n  Kaplan 的训练数据仅为 Chinchilla 最优的 {(100-undertraining):.0f}%")
    print(f"  → 模型严重欠训练（{undertraining:.0f}% 的数据缺口）")


# ============================================================
# 六、IsoFLOPs 曲线可视化（文字版）
# ============================================================

def demo_isoflops_curves():
    """
    演示不同 FLOPs 预算下的 IsoFLOPs 曲线。

    对应 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 中的 Figure 2。
    """
    print("\n" + "=" * 80)
    print("IsoFLOPs 曲线: 给定 FLOPs 预算下，参数量 vs Loss（U 型）")
    print("=" * 80)

    for logC in [17, 19, 21, 23]:
        C = 10 ** logC
        result = isoFLOPs_curve(C)

        print(f"\n  C = 10^{logC} = {C:.0e} FLOPs")
        print(f"    最优 N: {result['N_opt']/1e6:.1f}M 参数")
        print(f"    最优 D: {result['D_opt']/1e9:.1f}B tokens")
        print(f"    最小 loss: {result['loss_min']:.4f}")
        print(f"    比例 D/N: {result['D_opt']/result['N_opt']:.0f}")

        # 显示 U 型曲线两侧的 loss（验证 U 型）
        N_range = result['N']
        losses = result['loss']
        # 太小的 N
        idx_low = np.argmin(np.abs(N_range - result['N_opt'] * 0.1))
        # 太大的 N
        idx_high = np.argmin(np.abs(N_range - result['N_opt'] * 10))
        print(f"    0.1× N_opt: loss={losses[idx_low]:.4f} (↑{(losses[idx_low]-result['loss_min']):.4f})")
        print(f"    10× N_opt:  loss={losses[idx_high]:.4f} (↑{(losses[idx_high]-result['loss_min']):.4f})")


# ============================================================
# 七、实际应用: 为你的模型选最优配置
# ============================================================

def recommend_training_config(model_size_B: float) -> Dict:
    """
    根据 Chinchilla 定律，为给定的模型参数推荐训练数据量。

    应用:
      - 训练 7B 模型需要 ~140B tokens
      - 训练 70B 模型需要 ~1.4T tokens
      - 训练 405B 模型需要 ~8.1T tokens

    来自 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) 第六节。
    """
    N = model_size_B * 1e9  # 参数总量
    ratio = compute_tokens_per_param_ratio()  # ~20x
    D_opt = N * ratio
    C = compute_flops(N, D_opt)

    return {
        'N': N,
        'D_opt': D_opt,
        'ratio': ratio,
        'FLOPs': C,
    }


# ============================================================
# 八、主程序
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 80)
    print("Chinchilla Scaling Laws - 计算最优缩放定律")
    print("参考 [Chinchilla Scaling Laws](Chinchilla%20Scaling%20Laws.md) (Hoffmann et al., NeurIPS 2022)")
    print("=" * 80)

    # 1. Loss 函数演示
    print("\n[1] Chinchilla Loss 函数演示")
    N = 1e9  # 1B 参数
    D1 = 20e9  # 20B tokens（Chinchilla 最优）
    D2 = 2e9   # 2B tokens（严重不足）
    print(f"  L(N=1B, D=20B) = {chinchilla_loss(N, D1):.4f} (最优)")
    print(f"  L(N=1B, D=2B) = {chinchilla_loss(N, D2):.4f} (欠训练)")
    print(f"  Loss 差异: {chinchilla_loss(N, D2) - chinchilla_loss(N, D1):.4f}")

    # 2. 最优参数
    print("\n[2] 给定 FLOPs 预算，求最优 N, D")
    C = 1e21  # 固定 FLOPs
    N_opt, D_opt = compute_optimal_parameters(C)
    print(f"  C = {C:.0e} FLOPs")
    print(f"  N_opt = {N_opt/1e9:.2f}B 参数")
    print(f"  D_opt = {D_opt/1e9:.2f}B tokens")
    print(f"  比例 D/N = {D_opt/N_opt:.0f}")

    # 3. Tokens-per-parameter 比例
    ratio = compute_tokens_per_param_ratio()
    print(f"\n[3] 推荐 tokens-per-parameter 比例: ~{ratio:.0f}×")
    print(f"  含义: 如果训练一个 7B 模型，需要 ~{7*ratio/1e9:.0f}B tokens")

    # 4. 计算最优参数表
    generate_scaling_table()

    # 5. Kaplan vs Chinchilla
    compare_kaplan_vs_chinchilla(C=1e21)

    # 6. IsoFLOPs 曲线
    demo_isoflops_curves()

    # 7. 实际建议
    print("\n" + "=" * 70)
    print("Chinchilla 推荐的实际训练配置")
    print("=" * 70)
    for size in [1, 7, 13, 70, 405]:
        cfg = recommend_training_config(size)
        print(f"  {size:>3d}B 参数: {cfg['D_opt']/1e9:>6.0f}B tokens, "
              f"FLOPs ≈ {cfg['FLOPs']:.1e}")

    print("\n[Done]")
```
