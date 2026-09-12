---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Adam - 代码实现

> 本文档包含 Adam 优化器的 NumPy/PyTorch 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
Adam: A Method for Stochastic Optimization
===========================================
论文: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015)
核心贡献: 融合动量 (一阶矩) 和自适应学习率 (二阶矩)，
         加入偏差校正解决早期步的零初始化偏差。
代码结构:
  1. SGD —— 随机梯度下降 (对比基线)
  2. SGD + Momentum —— 带动量的 SGD
  3. RMSprop —— 自适应学习率 (无动量)
  4. Adam —— 完整 Adam (动量 + 自适应 + 偏差校正)
  5. AdamW —— 解耦权重衰减 (Transformer 标准)
  6. 优化器对比 —— 同一函数上的收敛行为对比

关键设计:
  - 一阶矩 m_t: 梯度的指数移动平均 → 加速收敛 (动量)
  - 二阶矩 v_t: 平方梯度的指数移动平均 → 逐参数自适应步长
  - 偏差校正: m_t_hat = m_t/(1-β1^t) → 修正初始化偏差
  - 更新: θ -= α * m_t_hat / (sqrt(v_t_hat) + ε)

与后续论文的关系:
  - AdamW 是 Llama 2/OpenVLA 等所有现代 Transformer 的标准优化器
  - SwiGLU 需要 Adam 的自适应学习率 (SGD 训练不稳定)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# 1. SGD —— 随机梯度下降 (最简基线)
# ==============================================================================
class SGD:
    """
    标准 SGD: θ_t = θ_{t-1} - α * g_t

    优点: 简单，无额外状态
    缺点: 慢，容易在沟壑地形震荡，对学习率敏感
    """
    def __init__(self, params: list, lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data


# ==============================================================================
# 2. SGD + Momentum —— 引入动量加速
# ==============================================================================
class SGDMomentum:
    """
    SGD + Momentum: v_t = μ·v_{t-1} + (1-μ)·g_t
                    θ_t = θ_{t-1} - α·v_t

    为什么需要动量？
    1. 加速穿越平坦区域：历史梯度方向上的分量累积
    2. 减少震荡：与当前梯度方向垂直的分量相互抵消
    3. 类似物理中的惯性——在持续下降方向加速

    μ 的作用：
    - μ=0.9: 当前梯度贡献 10%，历史方向保留 90% → 平滑方向变化
    - μ=0: 退化为 SGD
    - μ=0.99: 几乎忽略瞬时波动

    缺陷: 所有参数共享一个全局学习率 → 不适合梯度尺度差异大的网络
    """
    def __init__(self, params: list, lr: float = 0.01, mu: float = 0.9):
        self.params = params
        self.lr = lr
        self.mu = mu
        # 为每个参数维护独立的动量状态
        self.velocity = [torch.zeros_like(p) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # 动量更新: 指数移动平均
                self.velocity[i] = self.mu * self.velocity[i] + (1 - self.mu) * p.grad.data
                p.data -= self.lr * self.velocity[i]


# ==============================================================================
# 3. RMSprop —— 自适应学习率 (无动量)
# ==============================================================================
class RMSprop:
    """
    RMSprop: v_t = β_2·v_{t-1} + (1-β_2)·g_t²
             θ_t = θ_{t-1} - α·g_t / (√v_t + ε)

    为什么用平方梯度？
    平方梯度反映每个参数的"梯度尺度"：
    - 大梯度的参数 (jumps): v_t 大 → 步长自动缩小
    - 小梯度的参数 (flat): v_t 小 → 步长自动增大
    → 实现逐参数自适应学习率

    为什么用指数移动平均 (EMA) 而不是 AdaGrad 的累加？
    AdaGrad: v_t = Σ g_i² → v_t 单调递增 → 学习率最终趋于 0
    RMSprop: v_t = EMA → v_t 可以减小 (旧梯度指数衰减) → 学习率可恢复

    缺陷: 
    1. 无动量，在高曲率地形中收敛慢
    2. 无偏差校正，β_2 接近 1 时早期步不稳定
    """
    def __init__(self, params: list, lr: float = 0.001, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta2 = beta2
        self.eps = eps
        self.v = [torch.zeros_like(p) for p in params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                # 平方梯度的指数移动平均
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad.data ** 2
                # 逐参数自适应步长
                p.data -= self.lr * p.grad.data / (torch.sqrt(self.v[i]) + self.eps)


# ==============================================================================
# 4. Adam —— 完整实现 (动量 + 自适应 + 偏差校正)
# ==============================================================================
class Adam:
    """
    Adam: Adaptive Moment Estimation

    完整算法 (每个时间步 t):
    1. g_t = ∇_θ L(θ_{t-1})               — 计算梯度
    2. m_t = β1·m_{t-1} + (1-β1)·g_t      — 一阶矩 (动量)
    3. v_t = β2·v_{t-1} + (1-β2)·g_t²    — 二阶矩 (自适应)
    4. m̂_t = m_t / (1 - β1^t)             — 一阶偏差校正
    5. v̂_t = v_t / (1 - β2^t)             — 二阶偏差校正
    6. θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε) — 参数更新

    为什么需要偏差校正？(Section 3.2, 图 2)
    m_t 和 v_t 都从 0 初始化。
    - 第 1 步: m_1 = (1-β1)·g_1, 期望值 = (1-β1)·E[g]  → 严重偏小
    - 第 1 步: v_1 = (1-β2)·g_1², 期望值 = (1-β2)·E[g²] → 更严重偏小
    
    偏差校正:
    E[m̂_t] = E[m_t] / (1-β1^t) = E[g]  (偏小 β1^t → 校正后无偏)
    
    为什么 β1=0.9, β2=0.999？
    - β1=0.9: 动量窗口 ≈ 1/(1-0.9) = 10 步 (捕捉近期梯度方向)
    - β2=0.999: 自适应窗口 ≈ 1/(1-0.999) = 1000 步 (需要更多历史来估计方差)
    """

    def __init__(self, params: list, lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # 时间步计数器 (偏差校正用)
        # 状态变量
        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                g = p.grad.data

                # 步骤1: 更新有偏一阶矩 (动量)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                # 步骤2: 更新有偏二阶矩 (自适应学习率)
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

                # 步骤3: 偏差校正 —— 核心创新！
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)  # 除以 1-β1^t
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)  # 除以 1-β2^t

                # 步骤4: 参数更新
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


# ==============================================================================
# 5. AdamW —— 解耦权重衰减 (Transformer 标准优化器)
# ==============================================================================
class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay

    为什么需要 AdamW 而非 Adam + L2 正则化？
    Adam + L2: θ_t = θ_{t-1} - α * (m̂_t + λ·θ_{t-1}) / (√v̂_t + ε)
               ↑ L2 项也被 √v̂ 缩放了！不正确的！

    AdamW:     θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε) - α·λ·θ_{t-1}
               ↑ 权重衰减与自适应学习率解耦

    什么问题？在 Adam 中加 L2:
    - 梯度大的参数: √v̂ 大 → L2 正则被弱化 → 不够正则化
    - 梯度小的参数: √v̂ 小 → L2 正则被强化 → 过度正则化
    → 不同参数获得不平衡的正则化

    解耦权重衰减 (AdamW) 对所有参数施加均匀的正则化力度。

    这在 Transformer 中尤其重要：
    - Attention 层的 Q/K/V/O 投影梯度尺度差异大
    - Embedding 层梯度极小
    → L2 会导致 Embedding 过度正则化而 Attention 几乎无正则化
    """

    def __init__(self, params: list, lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is not None:
                g = p.grad.data

                # 一阶矩 + 二阶矩 (与 Adam 相同)
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

                # 偏差校正 (与 Adam 相同)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 核心差异: 权重衰减解耦
                # 先应用 Adam 更新
                p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                # 再独立施加权重衰减 (对所有参数均等)
                p.data -= self.lr * self.weight_decay * p.data


# ==============================================================================
# 演示: 优化器对比
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Adam 优化器对比演示")
    print("=" * 60)

    # ---- 1. 偏差校正的数值演示 ----
    print("\n--- 1. 偏差校正的必要性 ---")
    beta2 = 0.999
    for t in [1, 10, 100, 1000]:
        correction = 1 - beta2 ** t
        print(f"  t={t:>4d}: 1 - β2^t = {correction:.4f} → v̂_t = v_t / {correction:.4f}")
    print("说明: 早期步偏差巨大 (t=1 时偏 1000x), 需要偏差校正")

    # ---- 2. 偏差校正的有/无对比 ----
    print("\n--- 2. 偏差校正对早期步的影响 ---")
    beta1_test = 0.9
    for t in [1, 2, 3, 5, 10, 50]:
        # 假设梯度恒为 1
        m_biased = (1 - beta1_test ** t) * 1.0  # EMA 累积
        m_corrected = m_biased / (1 - beta1_test ** t)
        print(f"  t={t:>2d}: 有偏 m_t={m_biased:.4f}, 校正后 m̂_t={m_corrected:.4f}")
    print("说明: 偏差校正确保 m̂_t 的期望等于 E[g], 不受初始化的影响")

    # ---- 3. 优化器在 Rosenbrock 函数上的收敛对比 ----
    print("\n--- 3. 优化器收敛对比 (Rosenbrock 函数) ---")

    def rosenbrock(xy):
        """Rosenbrock 函数: (1-x)² + 100(y-x²)²
        - 全局最小值: (1, 1), f(1,1) = 0
        - 典型的沟壑地形: 狭窄的弯曲山谷
        - 对优化器挑战: 需要沿沟壑方向加速，同时避免震荡
        """
        x, y = xy[0], xy[1]
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def train_optimizer(opt_class, opt_params: dict, lr: float, steps: int = 1000):
        """用指定优化器在 Rosenbrock 上训练"""
        # 初始化在 (-1.5, 1.5)，远离最小值 (1,1)
        param = torch.tensor([-1.5, 1.5], dtype=torch.float32, requires_grad=True)
        optimizer = opt_class([param], lr=lr, **opt_params)

        history = []
        for t in range(steps):
            loss = rosenbrock(param)
            loss.backward()
            optimizer.step()
            param.grad.zero_()
            history.append((param[0].item(), param[1].item(), loss.item()))
        return np.array(history)

    steps = 500

    # 运行四种优化器
    print("  运行 SGD...")
    sgd_hist = train_optimizer(SGD, {}, lr=0.001, steps=steps)
    print("  运行 SGD + Momentum...")
    sgd_m_hist = train_optimizer(SGDMomentum, {'mu': 0.9}, lr=0.001, steps=steps)
    print("  运行 RMSprop...")
    rmsprop_hist = train_optimizer(RMSprop, {}, lr=0.001, steps=steps)
    print("  运行 Adam...")
    adam_hist = train_optimizer(Adam, {}, lr=0.01, steps=steps)  # Adam 可用更大的学习率

    # 打印最终结果
    results = [
        ("SGD", sgd_hist, "绿色"),
        ("SGD+Momentum", sgd_m_hist, "蓝色"),
        ("RMSprop", rmsprop_hist, "橙色"),
        ("Adam", adam_hist, "红色")
    ]
    
    print(f"\n  {'优化器':>15s} | {'最终 Loss':>10s} | {'最终位置 (x, y)':>25s}")
    print("  " + "-" * 55)
    for name, hist, color in results:
        final_loss = hist[-1, 2]
        final_x = hist[-1, 0]
        final_y = hist[-1, 1]
        print(f"  {name:>15s} | {final_loss:>10.6f} | ({final_x:.4f}, {final_y:.4f})")

    best = min(results, key=lambda r: r[1][-1, 2])
    print(f"\n  最佳收敛: {best[0]} (loss={best[1][-1, 2]:.6f})")
    print("  目标最小值: (1.0, 1.0), f=0")

    # ---- 4. 参数 m_t, v_t 的状态可视化 ----
    print("\n--- 4. m_t 和 v_t 的动态变化 (Adam) ---")
    # 重新训练并记录内部状态
    param = torch.tensor([-1.5, 1.5], dtype=torch.float32, requires_grad=True)
    adam_opt = Adam([param], lr=0.01)
    
    m_history_x = []
    v_history_x = []
    for t in range(200):
        loss = rosenbrock(param)
        loss.backward()
        # 记录优化器状态 (参数 x 维度)
        m_history_x.append(adam_opt.m[0][0].item())
        v_history_x.append(adam_opt.v[0][0].item())
        adam_opt.step()
        param.grad.zero_()
    
    print(f"  x 维度的 m_t 变化: {m_history_x[0]:.4f} → {m_history_x[50]:.4f} → {m_history_x[-1]:.4f}")
    print(f"  x 维度的 v_t 变化: {v_history_x[0]:.4f} → {v_history_x[50]:.4f} → {v_history_x[-1]:.4f}")
    print("  m_t 收敛于 0 (接近最小值 → 梯度小)")
    print("  v_t 收敛于 0 → 自适应机制在收敛后弱化")
    
    # ---- 5. Adam vs AdamW: 权重衰减的差异 ----
    print("\n--- 5. Adam vs AdamW: 权重衰减差异 ---")
    
    # 用一个小线性模型演示
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 两个参数，梯度尺度不同
            self.w_big = nn.Parameter(torch.tensor([10.0]))  # 大权重
            self.w_small = nn.Parameter(torch.tensor([0.1]))  # 小权重
        
        def forward(self, x):
            return self.w_big * x + self.w_small

    def train_with_wd(opt_class):
        model = TinyModel()
        params = list(model.parameters())
        if opt_class == AdamW:
            opt = AdamW(params, lr=0.01, weight_decay=0.1)
        else:
            # Adam + L2 (通过 grad 手动加)
            opt = Adam(params, lr=0.01)

        for t in range(100):
            x = torch.randn(1)
            y = model(x)
            # 模拟含 L2 正则化的损失
            loss = y + 0.05 * (model.w_big ** 2 + model.w_small ** 2).sum()
            loss.backward()
            opt.step()
            model.zero_grad()
        
        return model.w_big.item(), model.w_small.item()

    # 注意: 这是一个示意性对比，真实的 Adam+L2 在 grad 中统一计算
    w_big_aw, w_small_aw = train_with_wd(AdamW)
    print(f"  AdamW 结果:    w_big={w_big_aw:.4f}, w_small={w_small_aw:.4f}")
    print("  说明: AdamW 的权重衰减对所有参数均等施加")
    print("        Adam+L2 中，大梯度参数 (w_big) 的衰减被 √v̂ 弱化")

    # ---- 6. 超参数选择指南 ----
    print("\n--- 6. 超参数推荐 ---")
    print("""
    超参数选择:
    | 场景         | LR     | β1    | β2    | ε      | weight_decay |
    |-------------|--------|-------|-------|--------|--------------|
    | Vision (CNN)| 1e-3   | 0.9   | 0.999 | 1e-8   | 0             |
    | Transformer | 1e-4   | 0.9   | 0.999 | 1e-8   | 0.01          |
    | LoRA 微调    | 2e-5   | 0.9   | 0.999 | 1e-8   | 0.01          |
    | bf16 训练    | 同上   | 0.9   | 0.99  | 1e-6   | 同上           |
    
    ε 的注意点:
    - fp32 训练: ε=1e-8 安全
    - bf16/fp16 训练: ε=1e-6 防止除零 (√v̂ 可能很小)
    """)

    print("=" * 60)
    print("Adam 核心要点总结:")
    print("  1. 一阶矩 m_t (动量) + 二阶矩 v_t (自适应) → 融合两大优化器范式")
    print("  2. 偏差校正 m̂_t = m_t/(1-β1^t) → 修正零初始化偏差")
    print("  3. 有效步长 |Δt| ≈ α → 构建参数空间的信任域")
    print("  4. AdamW: 权重衰减从梯度中解耦 → Transformer 训练的必要改进")
    print("  5. SwiGLU + RMSNorm → 必须用 AdamW， SGD 无法稳定训练")
    print("=" * 60)

```
