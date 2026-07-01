---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# AdamW - 代码实现

> 本文档包含 AdamW（解耦权重衰减）的 PyTorch 教学实现。

[AdamW](AdamW.md) 的核心贡献是将权重衰减（weight decay）从 Adam 的自适应梯度更新中解耦，避免了 L2 正则化被 Adam 的 1/sqrt(v) 因子扭曲的问题。

```python
"""
AdamW: Decoupled Weight Decay Regularization
=============================================
PyTorch 教学实现 —— 完整实现 + 与标准 Adam（带 L2）的对比。

论文: [AdamW](AdamW.md) (Loshchilov & Hutter, ICLR 2019)

核心问题: Adam 中的 L2 正则化失效
  标准 Adam + L2 正则化:
    w_{t+1} = w_t - η * m_t / (√v_t + ε) - η * λ * w_t / (√v_t + ε)
    ↑ 权重衰减由 λ 控制，但被 1/√v_t 扭曲

  AdamW（解耦）:
    w_{t+1} = w_t - η * m_t / (√v_t + ε) - η * λ * w_t
    ↑ 权重衰减项不被 1/√v_t 缩放，与 SGD 的正则化语义一致

为什么这对 LLM 训练至关重要:
  - β_2 = 0.95 (LLaMA 系列) 时，√v_t 在不同维度差异可达 100x
  - 被扭曲的权重衰减会导致部分维度过度衰减或衰减不足
  - AdamW 的解耦保证所有维度受到均匀的正则化力度
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


# ============================================================
# 一、标准 Adam（带 L2 正则化）
# ============================================================

class AdamWithL2:
    """
    标准 Adam 优化器 + L2 正则化 → 耦合的权重衰减。

    这是 AdamW 论文中批评的 "错误实现" ——
    L2 正则化的梯度被 Adam 的自适应学习率缩放，
    导致权重衰减不是常数，而是随梯度方差变化。
    """
    def __init__(self, params: torch.Tensor, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.01, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay  # 在 Adam 中这是 "L2 正则化系数"
        self.eps = eps
        self.step_count = 0

        # 优化器状态（fp32，和参数同形状）
        self.m = torch.zeros_like(params)
        self.v = torch.zeros_like(params)

    def step(self, grad: torch.Tensor):
        """
        关键问题: L2 梯度被 Adam 自适应缩放

        loss = data_loss(w) + (λ/2) * ||w||^2
        ∇w = ∇data_loss + λ * w

        Adam 更新（问题版本）:
          m = β1*m + (1-β1)*(∇data_loss + λ*w)
          v = β2*v + (1-β2)*(∇data_loss + λ*w)^2
          w = w - η * m_hat / (√v_hat + ε)

        λ*w 项先进入 m 和 v 的指数平均，再被 1/√v_hat 缩放
        → 权重衰减率 = η * λ / √v_hat —— 不是常数！
        """
        self.step_count += 1
        beta1, beta2 = self.betas

        # L2 正则化梯度 = data_grad + λ*w（耦合）
        grad_with_l2 = grad + self.weight_decay * self.params

        # 标准 Adam 更新
        self.m = beta1 * self.m + (1 - beta1) * grad_with_l2
        self.v = beta2 * self.v + (1 - beta2) * (grad_with_l2 ** 2)

        m_hat = self.m / (1 - beta1 ** self.step_count)
        v_hat = self.v / (1 - beta2 ** self.step_count)

        # 关键: L2 正则化被 1/√v_hat 扭曲
        # 真正的权重衰减效果: η * λ / √(v + ε) —— 不是常数
        self.params -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# ============================================================
# 二、AdamW（解耦权重衰减）
# ============================================================

class AdamW:
    """
    AdamW 优化器 —— 解耦权重衰减。

    核心公式:
      g_t = ∇L(w_t)                          # 纯数据梯度（无 L2）
      m_t = β₁·m_{t-1} + (1-β₁)·g_t          # 一阶矩
      v_t = β₂·v_{t-1} + (1-β₂)·g_t²         # 二阶矩
      m̂_t = m_t / (1-β₁^t)                    # 偏差校正
      v̂_t = v_t / (1-β₂^t)
      w_{t+1} = w_t - η·m̂_t/(√v̂_t+ε) - η·λ·w_t  # 解耦的权重衰减
                               ^^^^^^^^^^^^^^^^
                               不被 1/√v_hat 缩放！

    为什么解耦后更好:
      - 每步权重衰减率为常数 η·λ
      - 与 SGD 的权重衰减语义对齐
      - 对 λ 的鲁棒性远高于 Adam + L2
    """
    def __init__(self, params: torch.Tensor, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0.01, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.step_count = 0

        self.m = torch.zeros_like(params)
        self.v = torch.zeros_like(params)

    def step(self, grad: torch.Tensor):
        """
        AdamW 更新步骤。

        与 Adam+L2 的关键区别在第 (4) 步：
          权重衰减是纯标量乘 (1-ηλ)，不经过 Adam 的自适应缩放。
        """
        self.step_count += 1
        beta1, beta2 = self.betas

        # 1. 更新有偏矩估计（只用数据梯度）
        self.m = beta1 * self.m + (1 - beta1) * grad
        self.v = beta2 * self.v + (1 - beta2) * (grad ** 2)

        # 2. 偏差校正
        m_hat = self.m / (1 - beta1 ** self.step_count)
        v_hat = self.v / (1 - beta2 ** self.step_count)

        # 3. Adam 自适应更新
        self.params -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        # 4. 解耦的权重衰减（不涉及 m_hat / √v_hat）
        #    这就是 "Decoupled Weight Decay" 的核心！
        self.params -= self.lr * self.weight_decay * self.params


# ============================================================
# 三、AdamW 的不同变体
# ============================================================

class AdamWGrouped:
    """
    分组权重衰减 AdamW（LLaMA 风格）。

    Bias 和 LayerNorm 的权重不应该施加 weight decay，
    因为它们的正则化语义不同：
    - weight decay = L2 正则化 → 鼓励权重稀疏 → 适用于大矩阵
    - bias = 偏移量，天然接近零
    - LayerNorm γ = 缩放因子，应保持为 1 附近

    来自 [AdamW](AdamW.md) 的第六节实践建议：
    "weight decay 应用于所有参数除 bias 和 LayerNorm 外"
    """
    def __init__(self, params_list, lr=1e-3, betas=(0.9, 0.95),
                 weight_decay=0.1, eps=1e-8):
        """
        params_list: [(tensor, apply_wd_bool), ...]
                    例如: [(linear.weight, True), (linear.bias, False), (ln.weight, False)]
        """
        self.param_groups = [
            {'params': p, 'apply_wd': apply_wd}
            for p, apply_wd in params_list
        ]
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.step_count = 0

        self.m = {}
        self.v = {}
        for i, group in enumerate(self.param_groups):
            p = group['params']
            self.m[i] = torch.zeros_like(p)
            self.v[i] = torch.zeros_like(p)

    def step(self, grads):
        self.step_count += 1
        beta1, beta2 = self.betas

        for i, group in enumerate(self.param_groups):
            p = group['params']
            g = grads[i]

            # 1. Adam 更新（同标准 AdamW）
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - beta2 ** self.step_count)

            p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

            # 2. 条件权重衰减
            if group['apply_wd']:
                p -= self.lr * self.weight_decay * p


# ============================================================
# 四、对比实验：Adam+L2 vs AdamW
# ============================================================

def compare_adam_l2_vs_adamw():
    """
    直观验证 AdamW 的优势。

    实验设置:
      一个简单的优化问题（2D Rosenbrock 函数），
      对比 Adam+L2 和 AdamW 的收敛行为。

    Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    最优解: (1, 1)
    """
    print("=" * 70)
    print("Adam+L2 vs AdamW 对比实验 (Rosenbrock 优化)")
    print("=" * 70)

    def rosenbrock(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x ** 2)
        dy = 200 * (y - x ** 2)
        return torch.tensor([dx, dy])

    # 初始点
    init_point = torch.tensor([-1.0, 2.0], dtype=torch.float32)
    num_steps = 500

    # --- Adam + L2 ---
    params_l2 = init_point.clone()
    opt_l2 = AdamWithL2(params_l2, lr=5e-2, weight_decay=0.01)
    loss_history_l2 = []

    for _ in range(num_steps):
        g = rosenbrock_grad(params_l2[0], params_l2[1])
        opt_l2.step(g)
        loss_history_l2.append(rosenbrock(params_l2[0], params_l2[1]).item())

    # --- AdamW ---
    params_w = init_point.clone()
    opt_w = AdamW(params_w, lr=5e-2, weight_decay=0.01)
    loss_history_w = []

    for _ in range(num_steps):
        g = rosenbrock_grad(params_w[0], params_w[1])
        opt_w.step(g)
        loss_history_w.append(rosenbrock(params_w[0], params_w[1]).item())

    print(f"  初始 loss: {rosenbrock(init_point[0], init_point[1]):.2f}")
    print(f"  Adam+L2:    final loss={loss_history_l2[-1]:.4f}, "
          f"params=({params_l2[0]:.4f}, {params_l2[1]:.4f})")
    print(f"  AdamW:      final loss={loss_history_w[-1]:.4f}, "
          f"params=({params_w[0]:.4f}, {params_w[1]:.4f})")

    final_l2_dist = torch.norm(params_l2 - torch.tensor([1.0, 1.0]))
    final_w_dist = torch.norm(params_w - torch.tensor([1.0, 1.0]))
    print(f"  到最优解距离: Adam+L2={final_l2_dist:.4f}, AdamW={final_w_dist:.4f}")


# ============================================================
# 五、权重衰减系数鲁棒性测试
# ============================================================

def test_weight_decay_robustness():
    """
    验证 AdamW 对 weight_decay 参数的鲁棒性。

    来自 [AdamW](AdamW.md) 论文 Table:
    λ = 1.0 时 Adam 下降 7% 准确率，AdamW 仅下降 1%。
    """
    print("\n" + "=" * 70)
    print("权重衰减系数 λ 的鲁棒性测试")
    print("=" * 70)

    def simple_ridge_regression(wd_coeff, use_adamw):
        """简单岭回归优化（含噪声）。"""
        N, d = 100, 20
        w_true = torch.randn(d) * 0.5
        X = torch.randn(N, d)
        y = X @ w_true + torch.randn(N) * 0.1

        w = torch.randn(d) * 0.1

        if use_adamw:
            opt = AdamW(w, lr=0.01, weight_decay=wd_coeff)
        else:
            opt = AdamWithL2(w, lr=0.01, weight_decay=wd_coeff)

        for _ in range(200):
            grad = 2 * X.T @ (X @ w - y) / N
            opt.step(grad)

        return torch.norm(w - w_true).item()

    print(f"{'λ':<10} {'Adam+L2 误差':<18} {'AdamW 误差':<18} {'AdamW 优势':<12}")
    print("-" * 65)

    wd_list = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    for wd in wd_list:
        torch.manual_seed(42)
        err_l2 = simple_ridge_regression(wd, use_adamw=False)
        torch.manual_seed(42)
        err_w = simple_ridge_regression(wd, use_adamw=True)

        advantage = (err_l2 - err_w) / max(err_l2, 1e-8) * 100
        print(f"  {wd:<8.4f} {err_l2:<16.4f} {err_w:<16.4f} {advantage:>+8.1f}%")


# ============================================================
# 六、LLaMA 风格 AdamW 完整示例
# ============================================================

def llama_style_adamw_example():
    """
    演示 [AdamW](AdamW.md) 中描述的 LLaMA 风格 AdamW 配置。

    LLaMA 系列的超参数:
      - β₁ = 0.9, β₂ = 0.95（注意: 不是默认的 0.999）
      - weight_decay = 0.1
      - cosine schedule（warmup 后衰减至 10%）
      - 不对 bias/LayerNorm 应用 weight decay
    """
    print("\n" + "=" * 70)
    print("LLaMA 风格 AdamW 配置演示")
    print("=" * 70)

    # 模拟 LLaMA 的优化器设置
    d_model = 4096
    linear_w = torch.randn(d_model, d_model) * 0.02
    linear_b = torch.zeros(d_model)
    ln_weight = torch.ones(d_model)

    param_groups = [
        (linear_w, True),       # Linear weight: apply WD
        (linear_b, False),      # Linear bias: NO WD
        (ln_weight, False),     # LayerNorm weight: NO WD
    ]

    opt = AdamWGrouped(
        param_groups,
        lr=3e-4,
        betas=(0.9, 0.95),      # LLaMA 的 β₂ = 0.95
        weight_decay=0.1,        # LLaMA 的 weight_decay
    )

    print("  参数设置:")
    print(f"    lr = 3e-4 (LLaMA 起始 LR)")
    print(f"    betas = (0.9, 0.95)  # β₂=0.95, 不是 0.999!")
    print(f"    weight_decay = 0.1")
    print(f"    Linear.weight: apply weight_decay = True")
    print(f"    Linear.bias:   apply weight_decay = False")
    print(f"    LayerNorm.weight: apply weight_decay = False")

    # 为什么 LLaMA 用 β₂=0.95 而非 0.999?
    print("\n  为什么 β₂=0.95?")
    print("    β₂=0.999 → v_t 过于平滑 → 无法快速适应梯度方差变化")
    print("    β₂=0.95  → 更快响应近期梯度 → 训练更稳定")
    print("    实践教训: β₂=0.999 在 LLM 预训练中可能导致不稳定")


# ============================================================
# 七、显存估算
# ============================================================

def estimate_adamw_memory():
    """
    估算 AdamW 优化器状态在不同配置下的显存消耗。

    每个参数:
      - fp32 AdamW: m (4B) + v (4B) = 8B per param
      - bf16 AdamW: m (2B) + v (2B) = 4B per param
      - 8-bit AdamW: m (1B) + v (1B) = 2B per param
      - Lion:        m (4B) = 4B per param (仅动量)
    """
    print("\n" + "=" * 70)
    print("AdamW 优化器显存估算")
    print("=" * 70)

    models = [1e9, 7e9, 13e9, 70e9]  # 1B, 7B, 13B, 70B

    print(f"{'模型':<12} {'fp32 AdamW':<15} {'bf16 AdamW':<15} {'8-bit AdamW':<15} {'Lion':<15}")
    print("-" * 75)
    for params in models:
        fp32 = params * 8 / 1e9
        bf16 = params * 4 / 1e9
        eight_bit = params * 2 / 1e9
        lion = params * 4 / 1e9
        print(f"  {params/1e9:>4.0f}B       {fp32:>8.1f} GB     {bf16:>8.1f} GB     "
              f"{eight_bit:>8.1f} GB     {lion:>8.1f} GB")


# ============================================================
# 八、主程序
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 70)
    print("AdamW - 解耦权重衰减 教学实现")
    print("参考 [AdamW](AdamW.md) (Loshchilov & Hutter, ICLR 2019)")
    print("=" * 70)

    # 1. 基本验证
    print("\n[1] AdamW 基本步骤演示")
    w = torch.randn(10)
    opt = AdamW(w, lr=0.01, weight_decay=0.1)
    print(f"  初始 w 范数: {w.norm():.4f}")
    for i in range(3):
        grad = torch.randn(10)  # 模拟梯度
        opt.step(grad)
        print(f"  Step {i+1}: |w| = {w.norm():.4f} "
              f"(被 weight_decay 逐步收缩)")

    # 2. Adam+L2 vs AdamW 对比
    compare_adam_l2_vs_adamw()

    # 3. 鲁棒性测试
    test_weight_decay_robustness()

    # 4. LLaMA 风格
    llama_style_adamw_example()

    # 5. 显存估算
    estimate_adamw_memory()

    print("\n[Done]")
```
