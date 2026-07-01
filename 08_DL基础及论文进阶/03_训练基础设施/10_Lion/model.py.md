---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Lion - 代码实现

> 本文档包含 PyTorch 教学实现。参考 [[AdamW|AdamW]]

```python
"""
Lion: Symbolic Discovery of Optimization Algorithms
=====================================================
论文: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
核心贡献: 通过进化搜索自动发现优化器，结果是 Lion（EvoLved Sign Momentum）。
         比 AdamW 省 2-3x 显存，训练更快。

核心公式（极度简洁）:
  1. u_t = sign(β₁·m_{t-1} + (1-β₁)·g_t)  # 用 sign 而非实际值
  2. θ_t = θ_{t-1} - η·u_t                  # 统一更新量
  3. m_t = β₂·m_{t-1} + (1-β₂)·g_t         # 动量更新（不同于 u_t！）
  
  ↑ 注意: 更新方向 u_t 和动量 m_t 使用不同的 β！

关键差异 vs Adam/AdamW:
  - Lion:     只使用 sign(梯度+动量)，每个参数更新量相同（±η）
  - Adam:     自适应学习率（每个参数不同），需要存储 m 和 v
  - AdamW:    Adam + 解耦权重衰减

为什么 sign 有效？
  - sign 是天然的正则化：限制更新步长一致（不像 Adam 对某些参数极小更新）
  - 省内存: 不需要存储二阶矩 v（Adam 需要）
  - 搜索发现: 进化算法从数千种候选优化器中选出了这个设计

推荐超参数:
  - lr: 通常比 AdamW 小 3-10 倍（因为 sign 让更新幅度统一）
  - β₁=0.9, β₂=0.99 (β₁ ≠ β₂ 是关键！)
  - weight_decay: 比 AdamW 大 2-3 倍
"""

import torch
import numpy as np


class Lion:
    """
    从零实现 Lion 优化器

    Args:
        lr: 学习率（建议 AdamW 的 1/3 ~ 1/10）
        betas: (β₁, β₂) — 注意不是 (0.9, 0.999)！
               论文推荐 (0.9, 0.99)
        weight_decay: 权重衰减（建议比 AdamW 大 2-3 倍）
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99),
                 weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        """执行一步更新"""
        self.t += 1

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.data

            # Step 1: 计算更新方向 u = sign(β₁·m + (1-β₁)·grad)
            #         注意: u_t 使用 β₁ 混合当前梯度
            update = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            u = torch.sign(update)  # ← 核心！只用方向，不用大小

            # Step 2: 参数更新 θ -= lr * u
            p.data -= self.lr * u

            # Step 3: 权重衰减（解耦，与 AdamW 相同方式）
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

            # Step 4: 更新动量 m_t = β₂·m_{t-1} + (1-β₂)·grad
            #         注意: 动量更新用 β₂（不同于 u_t 的 β₁！）
            #         这是 Lion 与标准 Momentum 的关键差异
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class AdamW:
    """对比用：简化的 AdamW"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 weight_decay=0.01, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad.data
            # 动量更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            # 偏差校正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 参数更新
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            # 权重衰减
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Lion vs AdamW 对比演示")
    print("=" * 60)

    # 简单优化问题: f(w₁,w₂) = (w₁-1)² + 10*(w₂-2)²
    # 这是一个条件数很差的函数（两个维度曲率差 10 倍）
    def loss_fn(w1, w2):
        return (w1 - 1) ** 2 + 10 * (w2 - 2) ** 2

    print(f"\n优化目标: f(w₁,w₂) = (w₁-1)² + 10(w₂-2)²")
    print(f"最优解: w* = (1, 2)\n")

    for name, opt_cls, lr, kwargs in [
        ("Lion", Lion, 3e-4, {"betas": (0.9, 0.99), "weight_decay": 0.1}),
        ("AdamW", AdamW, 1e-3, {"betas": (0.9, 0.999), "weight_decay": 0.01}),
    ]:
        # 初始化参数
        w1 = torch.tensor([0.0], requires_grad=True)
        w2 = torch.tensor([0.0], requires_grad=True)
        opt = opt_cls([w1, w2], lr=lr, **kwargs)

        history = []
        for step in range(200):
            opt.zero_grad()
            loss = loss_fn(w1, w2)
            loss.backward()
            opt.step()
            history.append((w1.item(), w2.item()))

        final_loss = loss_fn(w1, w2).item()
        # 找到 loss < 0.01 所需的步数
        steps_to_converge = next(
            (i for i, (a, b) in enumerate(history)
             if loss_fn(torch.tensor(a), torch.tensor(b)) < 0.01),
            200
        )

        print(f"  {name} (lr={lr}):")
        print(f"    最终 w = ({w1.item():.4f}, {w2.item():.4f})")
        print(f"    最终 loss = {final_loss:.6f}")
        print(f"    收敛到 loss<0.01 需 {steps_to_converge} 步")
        print(f"    显存占用: {'2N' if 'Lion' in name else '3N'} (M+V)")

    print(f"\n关键差异:")
    print(f"  Lion:  u = sign(β₁·m + (1-β₁)·g)  — 只存 m, 符号更新")
    print(f"  AdamW: u = m̂ / √v̂              — 存 m+v, 自适应学习率")
    print(f"\nLion 的优势:")
    print(f"  1. 省 1/3 显存 (不需要 v)")
    print(f"  2. sign 天然正则化 → 泛化更好")
    print(f"  3. 计算更简单 (sign vs sqrt+division)")
    print(f"  4. 训练速度快 (减少 HBM 读写)")
    print(f"\n注意事项:")
    print(f"  - lr 要比 AdamW 小 3-10 倍")
    print(f"  - weight_decay 要比 AdamW 大 2-3 倍")
    print(f"  - β₁ ≠ β₂ 是 Lion 特有的关键设计")
```
