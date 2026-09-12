---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# FlashAttention: 分块 Online Softmax + Tiling 实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
FlashAttention: 分块 Online Softmax + Tiling 实现
==================================================
通过 IO-aware 的 tiling 策略, 将标准注意力的显存复杂度从 O(N^2) 降低到 O(N),
并在不损失精度的前提下实现加速。

论文: [[FlashAttention]] (Dao et al., NeurIPS 2022)
核心思想:
  - GPU 瓶颈不在 FLOPs, 而在 HBM ↔ SRAM 的数据移动
  - 通过 tiling 将所有中间计算保留在 SRAM 中, 避免将 O(N^2) 的 S 矩阵写入 HBM
  - Online Softmax: 按 block 处理 softmax, 无需全局归约

关键设计:
  1. Tiling: 将 Q, K, V 按块加载到 SRAM, 每次只计算一个 Q-block × K-block
  2. Online Softmax: 逐块更新 rowmax 和 rowsum, 数学等价于标准 softmax
  3. 反向重计算: 反向传播时不读 S 矩阵, 而是从 QKV 重算 (计算换显存)

Note: 这是算法级模拟, 不包含 CUDA kernel 实现.
      FlashAttention-2/3 的优化请参考原文.
"""

import numpy as np
import time


# ============================================================
# 标准 Attention 实现 (作为对比基线)
# ============================================================

def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       causal: bool = False) -> np.ndarray:
    """标准 Scaled Dot-Product Attention.

    显存: O(N^2) —— 需要存储完整的 S ∈ R^{N×N} 矩阵.
    HBM 读写: O(N^2) —— S 矩阵至少被读写 4 次.
    """
    N, d = Q.shape
    scale = np.sqrt(d)

    # Step 1: S = QK^T / sqrt(d)  ← O(N^2) 矩阵写入 HBM
    S = Q @ K.T / scale

    # Step 2: 可选 causal mask
    if causal:
        mask = np.triu(np.ones((N, N), dtype=Q.dtype) * -1e9, k=1)
        S = S + mask

    # Step 3: Softmax (需要从 HBM 读取 S)
    row_max = S.max(axis=-1, keepdims=True)
    S_safe = S - row_max
    P = np.exp(S_safe)
    P = P / P.sum(axis=-1, keepdims=True)  # 写入 HBM

    # Step 4: O = PV
    O = P @ V

    return O


# ============================================================
# Online Softmax 算法 (分块 softmax 的数学基础)
# ============================================================

def online_softmax_update(
    m_old: float,
    d_old: float,
    o_old: np.ndarray,
    x_block: np.ndarray,
    v_block: np.ndarray,
) -> tuple:
    """Online Softmax 的增量更新步骤.

    这是 FlashAttention 的核心数学技巧:
      不需要知道全局的 S 矩阵就能逐块更新 softmax 输出.

    算法推导:
      设 m_new = max(m_old, max(x_block))
      d_new = d_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
      o_new = o_old * exp(m_old - m_new) + sum(exp(x_block - m_new) * v_block)

    为什么正确:
      这等价于先收集所有 block 再统一做 softmax, 但不需要存储 O(N^2) 矩阵.
    """
    x_max = x_block.max()
    m_new = max(m_old, x_max)

    scale_old = np.exp(m_old - m_new)
    d_new = scale_old * d_old + np.sum(np.exp(x_block - m_new))
    o_new = scale_old * o_old + np.sum(
        np.exp(x_block - m_new)[:, np.newaxis] * v_block, axis=0
    )

    return m_new, d_new, o_new


# ============================================================
# FlashAttention 分块实现
# ============================================================

def flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    Br: int = None, Bc: int = None,
                    causal: bool = False) -> np.ndarray:
    """FlashAttention 的分块前向传播 (算法级模拟).

    Tiling 策略:
      - 将 Q 按行分成 Br 大小的块 (外循环)
      - 将 K, V 按行分成 Bc 大小的块 (内循环)
      - 每个 (Q-block, K-block) 的 S 计算完全在 "SRAM" 中完成 (模拟)

    关键: S_ij ∈ R^{Br × Bc} 从不写入 "HBM"——只在 SRAM 中存在, 用完即弃.

    为什么显存是 O(N):
      每个时刻只有 Q_i (Br×d), K_j (Bc×d), V_j (Bc×d),
      和 O_i (Br×d) 四个小矩阵在 "HBM" 中.
      没有任何 O(N^2) 大小的矩阵需要存储.
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    if Br is None:
        Br = min(64, N)
    if Bc is None:
        Bc = min(64, N)

    O = np.zeros((N, d), dtype=np.float32)

    # ---- 外循环: 遍历 Q 的块 ----
    for i_start in range(0, N, Br):
        i_end = min(i_start + Br, N)
        Q_i = Q[i_start:i_end]  # 加载 Q 块到 SRAM (Br, d)
        Br_actual = i_end - i_start

        # Online Softmax 的状态 (每个 query 行独立)
        m_i = np.full(Br_actual, -np.inf, dtype=np.float32)
        d_i = np.zeros(Br_actual, dtype=np.float32)
        O_i = np.zeros((Br_actual, d), dtype=np.float32)

        # ---- 内循环: 遍历 K, V 的块 ----
        for j_start in range(0, N, Bc):
            j_end = min(j_start + Bc, N)
            K_j = K[j_start:j_end]  # 加载 K 块到 SRAM (Bc, d)
            V_j = V[j_start:j_end]  # 加载 V 块到 SRAM (Bc, d)

            # ---- 计算 S_ij = Q_i @ K_j^T / sqrt(d) (在 SRAM 中!) ----
            S_ij = Q_i @ K_j.T * scale  # (Br, Bc) —— 在 SRAM 中

            # ---- Causal mask ----
            if causal:
                for qi in range(Br_actual):
                    qi_abs = i_start + qi
                    for kj in range(j_end - j_start):
                        kj_abs = j_start + kj
                        if kj_abs > qi_abs:
                            S_ij[qi, kj] = -1e9

            # ---- Online Softmax 更新 ----
            for qi in range(Br_actual):
                m_i[qi], d_i[qi], O_i[qi] = online_softmax_update(
                    m_i[qi], d_i[qi], O_i[qi],
                    S_ij[qi],
                    V_j,
                )

            # S_ij 在这里被丢弃——不写回 HBM

        O[i_start:i_end] = O_i

    return O


# ============================================================
# 显存分析
# ============================================================

def estimate_memory(N: int, d: int, bytes_per_element: int = 2) -> dict:
    """估算标准 Attention 和 FlashAttention 的显存使用."""
    qkv_bytes = 3 * N * d * bytes_per_element
    standard_extra = 2 * N * N * bytes_per_element
    standard_total = qkv_bytes + standard_extra
    flash_hbm = qkv_bytes + N * d * bytes_per_element

    return {
        "sequence_length": N,
        "head_dim": d,
        "bytes_per_element": bytes_per_element,
        "standard_HBM_GB": standard_total / 1e9,
        "flash_HBM_GB": flash_hbm / 1e9,
        "memory_reduction": standard_total / flash_hbm,
    }


# ============================================================
# 数值精度验证
# ============================================================

def verify_precision(N: int = 256, d: int = 64):
    """验证 FlashAttention 与标准 Attention 的数值等价性.

    论文定理 1: FlashAttention 与标准 attention 在数学上完全等同,
    仅受浮点舍入误差影响.
    """
    np.random.seed(42)
    Q = np.random.randn(N, d).astype(np.float32) * 0.1
    K = np.random.randn(N, d).astype(np.float32) * 0.1
    V = np.random.randn(N, d).astype(np.float32) * 0.1

    t0 = time.time()
    out_standard = standard_attention(Q, K, V, causal=True)
    t_standard = time.time() - t0

    t0 = time.time()
    out_flash = flash_attention(Q, K, V, Br=32, Bc=32, causal=True)
    t_flash = time.time() - t0

    max_diff = np.abs(out_standard - out_flash).max()
    mean_diff = np.abs(out_standard - out_flash).mean()
    rel_error = max_diff / (np.abs(out_standard).max() + 1e-10)

    return {
        "max_absolute_error": max_diff,
        "mean_absolute_error": mean_diff,
        "relative_error": rel_error,
        "standard_time": t_standard,
        "flash_time": t_flash,
    }


# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("FlashAttention: 分块 Online Softmax + Tiling 模拟")
    print("=" * 60)

    # ---- 1. Online Softmax 验证 ----
    print("\n1. Online Softmax 数学验证")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    standard_softmax = np.exp(x - x.max()) / np.exp(x - x.max()).sum()

    m, d_val, o = -np.inf, 0.0, 0.0
    m, d_val, o = online_softmax_update(m, d_val, o, x[:3], np.ones(3))
    m, d_val, o = online_softmax_update(m, d_val, o, x[3:], np.ones(3))
    online_probs = np.exp(x - m) / d_val
    print(f"  标准 softmax:  {standard_softmax}")
    print(f"  Online softmax: {online_probs}")
    print(f"  最大差异: {np.abs(standard_softmax - online_probs).max():.10f}")

    # ---- 2. 显存分析 ----
    print(f"\n{'='*50}")
    print("2. 显存分析对比 (FP16)")
    print(f"  {'N':>6} {'d':>4} | {'标准 Attention':>14} | {'FlashAttention':>14} | {'节省比例':>10}")
    print("  " + "-" * 55)
    for N in [1024, 4096, 16384, 65536, 131072]:
        d = 64
        mem = estimate_memory(N, d, bytes_per_element=2)
        print(f"  {N:>6} {d:>4} | {mem['standard_HBM_GB']:>12.2f} GB | "
              f"{mem['flash_HBM_GB']:>12.4f} GB | {mem['memory_reduction']:>8.1f}x")
        if mem['standard_HBM_GB'] > 80:
            print(f"         {'':>10} 标准 Attention 已超过 A100 80GB ❌")

    # ---- 3. 数值精度验证 ----
    print(f"\n{'='*50}")
    print("3. 数值精度验证 (N=256, d=64, causal=True)")
    result = verify_precision(N=256, d=64)
    print(f"  最大绝对误差: {result['max_absolute_error']:.8f}")
    print(f"  平均绝对误差: {result['mean_absolute_error']:.8f}")
    print(f"  相对误差: {result['relative_error']:.8f}")
    print(f"  标准 Attention 耗时: {result['standard_time']:.4f}s")
    print(f"  FlashAttention 耗时: {result['flash_time']:.4f}s")

    # ---- 4. 分块计算演示 ----
    print(f"\n{'='*50}")
    print("4. 分块计算演示 (N=8, d=4, Br=4, Bc=4)")
    N, d = 8, 4
    Q = np.arange(N * d, dtype=np.float32).reshape(N, d) * 0.01
    K = np.arange(N * d, dtype=np.float32).reshape(N, d) * 0.01
    V = np.ones((N, d), dtype=np.float32)

    out_std = standard_attention(Q, K, V, causal=False)
    out_flash = flash_attention(Q, K, V, Br=4, Bc=4, causal=False)
    print(f"  标准 Attention 输出 (前 2 行):\n{out_std[:2]}")
    print(f"  FlashAttention 输出 (前 2 行):\n{out_flash[:2]}")
    print(f"  差异: {np.abs(out_std - out_flash).max():.10f}")

    print("\n✅ FlashAttention 分块模拟完成")

```
