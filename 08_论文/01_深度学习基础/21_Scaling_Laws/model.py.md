---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Scaling Laws - 代码实现

> 本文档包含 Scaling Laws 的 NumPy/PyTorch 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Scaling Laws for Neural Language Models
=======================================
论文: "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
核心贡献: 建立 LM 性能与 N(参数量)/D(数据量)/C(计算量) 之间的幂律关系
           L(N) = (Nc/N)^α_N, α_N ≈ 0.076
           L(D) = (Dc/D)^α_D, α_D ≈ 0.095
代码结构:
  1. 幂律拟合 —— 最小二乘法拟合 logL = a·logX + b
  2. Kaplan 定律 —— N/D/C 三组幂律的公式与可视化
  3. Chinchilla 对比 —— Kaplan vs Chinchilla 的 D∝N 比例差异
  4. 计算最优分配 —— 给定 C 的最优 N/D 分配策略
  5. 外推演示 —— 用小规模数据预测大规模性能

与后续论文的关系:
  - [[Chinchilla]] 修正了 Kaplan 的 D∝N 比例 (N^0.74 → N^1.0)
  - [[../20_Llama2/Llama 2|Llama 2]] 的 2T tokens 预训练体现了数据规模的重要性
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，可在 Obsidian 中运行
import matplotlib.pyplot as plt


# ==============================================================================
# 1. 幂律拟合工具 —— numpy 最小二乘法
# ==============================================================================
def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    拟合幂律关系 y = a * x^b

    方法: 对两边取对数 log(y) = log(a) + b * log(x)
          → 线性回归: Y = B0 + B1 * X
          → a = exp(B0), b = B1

    为什么取对数？
    幂律在对数空间中变为线性关系，可以用简单的最小二乘法求解。
    """
    log_x = np.log(x)
    log_y = np.log(y)
    
    # 最小二乘法: Y = B0 + B1 * X
    B1, B0 = np.polyfit(log_x, log_y, 1)
    
    a = np.exp(B0)
    b = B1
    return a, b


def compute_r_squared(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    """计算 R² 以评估拟合质量"""
    y_pred = a * (x ** b)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


# ==============================================================================
# 2. Kaplan 幂律公式
# ==============================================================================
def kaplan_loss_N(N: np.ndarray) -> np.ndarray:
    """
    模型规模幂律: L(N) = (Nc / N)^α_N
    
    论文拟合值 (WebText2 数据集):
      Nc ≈ 8.8e13 (非嵌入参数)
      α_N ≈ 0.076
    
    含义: 参数量翻倍，损失减少约 1 - 2^(-0.076) ≈ 5.1%
          收益递减 (α_N < 1)
    """
    Nc = 8.8e13   # 临界参数量
    alpha_N = 0.076
    return (Nc / N) ** alpha_N


def kaplan_loss_D(D: np.ndarray) -> np.ndarray:
    """
    数据量幂律: L(D) = (Dc / D)^α_D
    
    论文拟合值:
      Dc ≈ 5.4e13 (tokens)
      α_D ≈ 0.095
    
    含义: 数据量翻倍，损失减少约 1 - 2^(-0.095) ≈ 6.4%
          注意 α_D > α_N —— 数据略优于参数（但优势不大）
    """
    Dc = 5.4e13
    alpha_D = 0.095
    return (Dc / D) ** alpha_D


def kaplan_loss_C(Cmin: np.ndarray) -> np.ndarray:
    """
    计算量幂律 (最优分配下): L(Cmin) = (Cmin_c / Cmin)^α_C
    
    论文拟合值:
      Cmin_c ≈ 3.1e8 PF-days
      α_C ≈ 0.050
    
    含义: 计算量翻倍，损失减少约 1 - 2^(-0.050) ≈ 3.4%
          收益递减最快 (α_C 最小)
    """
    Cmin_c = 3.1e8
    alpha_C = 0.050
    return (Cmin_c / Cmin) ** alpha_C


# ==============================================================================
# 3. 联合模型 L(N, D) 和计算最优分配
# ==============================================================================
def joint_loss_ND(N: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    联合模型 L(N, D) = [(Nc/N)^(α_N/α_D) + Dc/D]^α_D
    
    这个公式满足:
    1. N → ∞ 时 L → L(D) (仅受数据限制)
    2. D → ∞ 时 L → L(N) (仅受模型限制)
    3. D = ∞ 处有 1/D 解析展开
    """
    Nc = 8.8e13
    Dc = 5.4e13
    alpha_N = 0.076
    alpha_D = 0.095
    return ((Nc / N) ** (alpha_N / alpha_D) + Dc / D) ** alpha_D


def compute_optimal_allocation(Cmin: float) -> dict:
    """
    给定计算预算 Cmin，计算最优资源配置 (Kaplan 版本)
    
    论文核心发现 (Kaplan 视角):
      N_opt ∝ Cmin^0.73   → 计算增加主要用来扩大模型
      D_opt ∝ Cmin^0.27   → 数据量增长缓慢
      B_opt ∝ Cmin^0.24   → batch size 适度增长
      S_min ∝ Cmin^0.03   → 串行步数几乎不变
    
    这说明: "优先扩大模型规模" 是计算高效的选择
    """
    # 使用近似比例常数（论文未给出确切的截距，这里采用相对比例）
    N_opt = 1e6 * (Cmin ** 0.73)
    D_opt = 1e5 * (Cmin ** 0.27)
    B_opt = 1000 * (Cmin ** 0.24)
    S_min = 1000 * (Cmin ** 0.03)
    
    return {
        'N_opt': N_opt,      # 最优参数量
        'D_opt': D_opt,      # 最优数据量
        'B_opt': B_opt,      # 最优 batch size  
        'S_min': S_min       # 最优串行步数
    }


# ==============================================================================
# 4. Chinchilla 对比 —— 修正后的 Scaling Law
# ==============================================================================
def chinchilla_optimal(N_target: float) -> float:
    """
    Chinchilla 定律 (Hoffmann et al., 2022):
      D_opt ≈ 20 × N_target
    
    Chinchilla 修正了 Kaplan 的结论:
    - Kaplan:  D ∝ N^0.74  (模型增速远大于数据增速)
    - Chinchilla: D ∝ N^1.0  (同步增长)
    
    为什么 Kaplan 低估了数据需求？
    1. Kaplan 只用了 1.5B 参数的最大模型，外推有限
    2. 大参数模型需要更多数据才能避免过拟合
    3. Kaplan 的学习率调度不够优化
    
    实际影响:
    - Kaplan 方案: 7B 模型只需要 ~2B tokens
    - Chinchilla 方案: 7B 模型需要 ~140B tokens
    - 实践量 (Llama 2): 7B 模型用了 2T tokens (远超 Chinchilla 推荐)
    """
    return 20.0 * N_target


# ==============================================================================
# 5. 过拟合判定
# ==============================================================================
def check_overfitting(N: float, D: float) -> dict:
    """
    判断给定 (N, D) 组合是否会过拟合
    
    论文结论: 过拟合程度仅依赖于 N^α_N/α_D / D ≈ N^0.8 / D
    
    实际经验法则: D ≳ 5 × 10^3 × N^0.74 才能避免过拟合
    """
    ratio = N ** 0.74 / D
    threshold = 5e3
    
    is_overfit = ratio > threshold
    return {
        'ratio': ratio,
        'threshold': threshold,
        'is_overfit': is_overfit,
        'status': '过拟合风险' if is_overfit else '安全'
    }


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Scaling Laws 幂律拟合演示")
    print("=" * 60)
    
    # ---- 1. 模拟实验数据进行幂律拟合 ----
    print("\n--- 1. 幂律拟合 (最小二乘法) ---")
    # 模拟不同参数规模的损失值（加噪声以模拟真实实验）
    np.random.seed(42)
    N_exp = np.array([1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9])  # 非嵌入参数
    # 理论 L(N) + 噪声
    L_true = kaplan_loss_N(N_exp)
    noise = np.random.normal(0, 0.02, size=len(N_exp))
    L_obs = L_true + noise * L_true  # 相对噪声
    
    a_fit, b_fit = fit_power_law(N_exp, L_obs)
    r2 = compute_r_squared(N_exp, L_obs, a_fit, b_fit)
    
    print(f"拟合结果: L(N) = {a_fit:.2e} * N^{b_fit:.4f}")
    print(f"R² = {r2:.4f}")
    print(f"理论 α_N = 0.076, 拟合 α ≈ {abs(b_fit):.4f}")
    
    # 外推: 用拟合参数预测更大模型
    N_extrap = 1e10  # 外推到 10B 参数
    L_predicted = a_fit * (N_extrap ** b_fit)
    print(f"外推: N={N_extrap:.0e} → 预测 L = {L_predicted:.4f}")
    
    # ---- 2. 三种幂律的可视化数据 ----
    print("\n--- 2. Kaplan 三种幂律对比 ---")
    
    # N-range: 1M → 1B 参数
    N_range = np.logspace(np.log10(1e6), np.log10(1e9), 20)
    L_N = kaplan_loss_N(N_range)
    
    # D-range: 1M → 1B tokens
    D_range = np.logspace(np.log10(1e6), np.log10(1e9), 20)
    L_D = kaplan_loss_D(D_range)
    
    # C-range: 相对计算量
    C_range = np.logspace(0, 3, 20)
    L_C = kaplan_loss_C(C_range)
    
    print(f"  模型规模 (N=1e6→1e9): L = {L_N[0]:.3f} → {L_N[-1]:.3f}")
    print(f"  数据量 (D=1e6→1e9):   L = {L_D[0]:.3f} → {L_D[-1]:.3f}")
    print(f"  计算量 (C=1→1000):    L = {L_C[0]:.3f} → {L_C[-1]:.3f}")
    print(f"α_N={0.076}, α_D={0.095}, α_C={0.050} (指数越小, 边际收益递减越快)")
    
    # ---- 3. Kaplan vs Chinchilla: 数据需求对比 ----
    print("\n--- 3. Kaplan vs Chinchilla: 数据需求对比 ---")
    
    models_N = np.array([1e6, 1e7, 1e8, 1e9])  # 1M, 10M, 100M, 1B 参数
    
    print(f"{'参数 N':>12s} | {'Kaplan D_opt':>14s} | {'Chinchilla D_opt':>18s} | {'判定':>14s}")
    print("-" * 70)
    
    for N_val in models_N:
        # Kaplan: D ∝ N^0.74 → 近似 D_opt
        D_kaplan = N_val ** 0.74  # 相对比例
        # Chinchilla: D ≈ 20 * N
        D_chinchilla = chinchilla_optimal(N_val)
        # 过拟合检测
        overfit_info = check_overfitting(N_val, D_chinchilla)
        
        N_label = f"{N_val/1e6:.0f}M" if N_val < 1e9 else f"{N_val/1e9:.0f}B"
        print(f"{N_label:>12s} | {D_kaplan:>14.1e} | {D_chinchilla:>18.1e} | {overfit_info['status']:>14s}")
    
    print("\n说明: Kaplan 低估了大数据的重要性")
    print("  - Chinchilla 修正: 模型和数据应同步增长 (N ∝ D)")
    print("  - Llama 2 实践: 7B 用了 2T tokens (远超前两者推荐)")
    
    # ---- 4. 联合模型 L(N,D) 演示 ----
    print("\n--- 4. 联合模型 L(N, D) 演示 ---")
    
    N_test = np.array([1e7, 1e8, 1e9])
    D_test = np.array([1e7, 1e8, 1e9])
    
    for n in N_test:
        for d in D_test:
            L_val = joint_loss_ND(n, d)
            overfit = check_overfitting(n, d)
            n_label = f"{n/1e6:.0f}M" if n < 1e9 else f"{n/1e9:.0f}B"
            d_label = f"{d/1e6:.0f}M" if d < 1e9 else f"{d/1e9:.0f}B"
            print(f"  N={n_label:>4s}, D={d_label:>4s}: L={L_val:.3f} ({overfit['status']})")
    
    # ---- 5. 计算最优分配策略 ----
    print("\n--- 5. 计算最优分配策略 ---")
    
    # 模拟不同计算预算
    C_budgets = [1.0, 10.0, 100.0, 1000.0]
    
    print(f"{'计算预算 C':>12s} | {'N_opt (相对)':>14s} | {'D_opt (相对)':>14s}")
    print("-" * 50)
    for C_val in C_budgets:
        opt = compute_optimal_allocation(C_val)
        print(f"{C_val:>12.1f} | {opt['N_opt']:>14.1e} | {opt['D_opt']:>14.1e}")
    
    print("\nKaplan 结论: 计算预算增加时，优先扩大模型而非数据")
    print("Chinchilla 修正: 应等比例增加模型和数据")
    
    # ---- 6. 规模收益递减演示 ----
    print("\n--- 6. 规模收益递减演示 ---")
    
    steps = [1, 2, 4, 8, 16]  # 规模翻倍次数
    L_improve_N = (1 - 0.5 ** 0.076) * 100  # 每次翻倍的损失降幅
    L_improve_D = (1 - 0.5 ** 0.095) * 100
    
    print(f"参数量每翻一倍:  损失减少 ≈ {L_improve_N:.1f}%")
    print(f"数据量每翻一倍:  损失减少 ≈ {L_improve_D:.1f}%")
    
    print(f"\n规模乘数 | 损失 (模型缩放) | 损失 (数据缩放)")
    print("-" * 45)
    
    base_L = 3.0
    for mult in steps:
        L_N = base_L * (mult ** (-0.076))
        L_D = base_L * (mult ** (-0.095))
        print(f"   ×{mult:>3d}   |   {L_N:.3f}       |   {L_D:.3f}")
    
    print("\n核心启示: 幂律指数 < 1 → 收益递减 → 无限扩大模型的极限")
    
    print("\n" + "=" * 60)
    print("总结:")
    print("  1. L(N) = (Nc/N)^α_N, α_N=0.076 → 参数翻倍, 损失降 5.1%")
    print("  2. L(D) = (Dc/D)^α_D, α_D=0.095 → 数据翻倍, 损失降 6.4%")
    print("  3. Kaplan: D ∝ N^0.74 → 优先扩大模型")
    print("  4. Chinchilla: D ∝ N^1.0 → 等比扩大模型和数据")
    print("  5. 过拟合判据: D ≳ 5×10^3 × N^0.74")
    print("=" * 60)

```
