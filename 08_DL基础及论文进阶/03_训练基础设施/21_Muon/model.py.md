---
tags:
  - 代码
  - PyTorch
  - 优化器
created: 2026-07-02
---

# Muon - 代码实现

> 本文档包含 Muon 优化器的 PyTorch 教学实现。参考 [[AdamW|AdamW]]、[[Lion|Lion]]

```python
"""
Muon: MomentUm Orthogonalized by Newton-Schulz
===============================================
原作者: Keller Jordan (2024, blog post)
论文: "Muon: An optimizer for hidden layers in neural networks"
扩展: "Muon is Scalable for LLM Training" (Liu et al., Moonshot AI, 2025)

核心思路:
  1. 计算 SGD + Nesterov 动量产生更新矩阵 G
  2. 用 Newton-Schulz 迭代（五次多项式）将 G 近似正交化 → O
  3. 用 O 替代 G 进行参数更新

与 AdamW/Lion 的本质区别:
  - AdamW: 逐元素自适应步长 (m̂ / √v̂)
  - Lion:   逐元素符号更新 sign(β₁m + (1-β₁)g)
  - Muon:   矩阵级正交化 → 所有方向统一更新幅度

为什么有效?
  - 梯度更新矩阵通常条件数极高（几乎是低秩的）
  - 正交化使"稀有方向"的更新幅度与主导方向相同
  - Newton-Schulz 迭代在 bf16 中稳定、在 GPU 上高效（<1% FLOP 开销）

适用范围:
  - 仅用于 2D hidden layer 权重（Linear 层、flattened Conv）
  - embedding/head/bias/LayerNorm 仍用 AdamW
  - Q、K、V 分别应用（不要合并为 QKV 矩阵）
  - 预训练场景为主；微调效果待验证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ==============================================================================
# 设备选择：优先 MPS > CUDA > CPU
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==============================================================================
# 1. Newton-Schulz 迭代：核心正交化算子
# ==============================================================================
def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz 五次多项式迭代 —— 近似矩阵正交化。

    原理: 给定 SVD G = U·S·Vᵀ，正交化的目标是 UVᵀ（即丢弃所有奇异值，
    只保留方向信息）。本函数用多项式迭代 φ⁵(x) → 1 逼近这一目标，
    避免昂贵的 SVD 计算。

    关键设计:
    - 系数 (3.4445, -4.7750, 2.0315) 是故意不收敛的 —— 它们最大化
      在 x=0 处的斜率以快速放大小奇异值，代价是最终奇异值在 ~[0.5, 1.5]
      而非精确的 1.0。经验上这完全不损害训练。
    - 在 bfloat16 中运行以利用 Tensor Core 加速
    - 转置优化：确保 XXᵀ 的计算是对 m×m 而非 n×n (m = min(rows, cols))

    Args:
        G: 2D 张量，形状为 (rows, cols)
        steps: 迭代步数（默认 5）
        eps: 数值稳定性的小量

    Returns:
        O: 正交化后的张量，形状与输入相同
    """
    assert G.ndim == 2, f"NewtonSchulz5 expects 2D input, got {G.ndim}D"

    a, b, c = (3.4445, -4.7750, 2.0315)  # 精心调优的五次多项式系数

    # 转换为 bf16 加速。如果设备不支持 bf16，回退到 float32
    if device.type == 'mps':
        compute_dtype = torch.float32  # MPS 的 bf16 支持有限
    else:
        compute_dtype = torch.bfloat16

    X = G.to(dtype=compute_dtype)

    # 归一化：除以 Frobenius 范数 → 确保最大奇异值 ≤ 1
    X = X / (X.norm() + eps)

    # 转置优化：确保 XXᵀ 的尺寸为 min(rows, cols) × min(rows, cols)
    # 当 rows > cols 时转置可以减少计算量
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        # A = X @ Xᵀ — (m × m) 矩阵
        A = X @ X.T
        # B = b·A + c·A² — 多项式的一部分
        B = b * A + c * (A @ A)
        # X = a·X + B·X — 完整五次多项式: a·X + b·X³ + c·X⁵
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X


# ==============================================================================
# 2. Muon 优化器（教学版，从零实现）
# ==============================================================================
class Muon:
    """
    Muon 优化器的教学实现。

    注意:
    - 仅接受 2D 参数（Linear 层的权重矩阵、flattened conv 核）
    - embedding、bias、LayerNorm、分类头等参数应使用 AdamW
    - 在实践中通常用 HybridMuonAdamW 管理两类参数（见下方）

    Args:
        params: 2D 参数列表
        lr: 学习率（与 AdamW 相同即可，得益于 RMS-一致缩放）
        momentum: 动量系数（默认 0.95，Nesterov 风格推荐）
        nesterov: 是否使用 Nesterov 动量（推荐 True）
        ns_steps: Newton-Schulz 迭代步数（默认 5）
        weight_decay: 权重衰减（扩展版；原始 Muon 无此参数，Moonshot AI 扩展添加）
        use_rms_scale: 是否使用 RMS-一致缩放（Moonshot AI 扩展）
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0, use_rms_scale=True):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.use_rms_scale = use_rms_scale

        # 为每个参数独立维护动量缓冲区（只需要 m，不需要 v —— 比 AdamW 省 50%）
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self):
        """执行一步更新"""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad.data

            # === Step 1: 更新动量 ===
            # m_t = β · m_{t-1} + g_t
            self.m[i] = self.momentum * self.m[i] + grad

            # === Step 2: 准备正交化的输入（Nesterov 或普通动量） ===
            if self.nesterov:
                # Nesterov: G = g_t + β · m_t
                # — "前瞻"一步，经验上在所有场景优于普通动量
                G = grad + self.momentum * self.m[i]
            else:
                G = self.m[i]  # 直接用动量

            # === Step 3: 正交化！ ===
            O = newtonschulz5(G, steps=self.ns_steps)

            # === Step 4: RMS-一致缩放（Moonshot AI 扩展） ===
            # Muon 的原始正交化输出 RMS 依赖矩阵形状
            # ~1/√(max(rows, cols))，需调整以匹配 AdamW
            if self.use_rms_scale:
                scale_factor = 0.2 * math.sqrt(max(p.shape[0], p.shape[1]))
                O = O * scale_factor

            # === Step 5: 参数更新（含可选权重衰减） ===
            # W = W - lr · O
            p.data -= self.lr * O

            # 权重衰减（解耦，与 AdamW 相同方式）
            if self.weight_decay != 0:
                p.data -= self.lr * self.weight_decay * p.data

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


# ==============================================================================
# 3. 对比实现：AdamW（教学版）
# ==============================================================================
class SimpleAdamW:
    """简化的 AdamW，用于对比（逐元素更新 vs 矩阵级更新）"""
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8):
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
            # 自适应步长更新
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
# 4. HybridMuonAdamW: 混合优化器（实战版）
# ==============================================================================
class HybridMuonAdamW:
    """
    混合优化器：对于 2D hidden 权重用 Muon，对于其余参数用 AdamW。

    这是生产环境的推荐用法。在 Moonshot AI 的训练中，Muon 用于:
    - Q/K/V 投影矩阵（分别）
    - MLP 的 W_in 和 W_out
    - 所有 hidden layer 的 Linear 权重

    AdamW 用于:
    - Embedding (词汇表 + 位置)
    - 分类头 (lm_head)
    - Bias 参数
    - LayerNorm 的 γ 和 β
    - 任何 1D 或 0D 参数

    用法:
        muon_params = [p for n, p in model.named_parameters()
                       if p.ndim == 2 and 'embed' not in n and 'head' not in n]
        adamw_params = [p for n, p in model.named_parameters()
                        if p not in set(muon_params)]

        # 分离 bias 和 LayerNorm 的 weight_decay
        no_decay = [p for p in adamw_params if p.ndim < 2]

        optimizer = HybridMuonAdamW(
            muon_params=muon_params,
            adamw_params=[p for p in adamw_params if p not in set(no_decay)],
            adamw_no_decay_params=no_decay,
            lr=3e-4, momentum=0.95,
            adamw_betas=(0.9, 0.95),
            weight_decay=0.1
        )
    """

    def __init__(self, muon_params, adamw_params, adamw_no_decay_params=None,
                 lr=3e-4, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_betas=(0.9, 0.95), weight_decay=0.1, adamw_eps=1e-8,
                 use_rms_scale=True):
        # Muon 子优化器：用于 2D hidden 权重
        self.muon = Muon(
            muon_params, lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
            use_rms_scale=use_rms_scale
        )
        # AdamW 子优化器：用于 embedding/head/bias/LayerNorm
        self.adamw = SimpleAdamW(
            adamw_params, lr=lr, betas=adamw_betas,
            weight_decay=weight_decay, eps=adamw_eps
        )
        # AdamW 无衰减组：bias, LayerNorm
        self.adamw_no_decay = None
        if adamw_no_decay_params:
            self.adamw_no_decay = SimpleAdamW(
                adamw_no_decay_params, lr=lr, betas=adamw_betas,
                weight_decay=0.0, eps=adamw_eps
            )

    def step(self):
        self.muon.step()
        self.adamw.step()
        if self.adamw_no_decay:
            self.adamw_no_decay.step()

    def zero_grad(self):
        self.muon.zero_grad()
        self.adamw.zero_grad()
        if self.adamw_no_decay:
            self.adamw_no_decay.zero_grad()


# ==============================================================================
# 5. 演示：Muon vs AdamW 对比实验
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("Muon vs AdamW: 矩阵优化问题对比")
    print("=" * 65)

    # 构造一个玩具问题：2D 权重的二次优化
    # 目标: min_W || W·X - Y ||²_ F
    # 这是一个 6×4 的线性映射优化问题
    torch.manual_seed(42)

    dim_out, dim_in = 6, 4
    batch_size = 128

    # 真实权重（我们的优化目标）
    W_true = torch.randn(dim_out, dim_in, device=device) * 2.0

    # 合成数据
    X = torch.randn(batch_size, dim_in, device=device)
    Y = X @ W_true.T  # (batch, dim_out)

    def compute_loss(W):
        pred = X @ W.T
        return F.mse_loss(pred, Y)

    print(f"\n优化目标: min_W || W·X - Y_true ||²")
    print(f"W 形状: ({dim_out}, {dim_in})")
    print(f"数据: {batch_size} samples, X ∈ R^({batch_size},{dim_in})")
    print()

    results = {}
    for name, opt_cls, lr, kwargs in [
        ("AdamW", SimpleAdamW, 3e-3, {"betas": (0.9, 0.999), "weight_decay": 0.0}),
        ("Muon (standard)", Muon, 3e-4, {"momentum": 0.95, "nesterov": True,
                                          "ns_steps": 5, "use_rms_scale": True}),
        ("Muon (no Nesterov)", Muon, 3e-4, {"momentum": 0.95, "nesterov": False,
                                             "ns_steps": 5, "use_rms_scale": True}),
        ("Muon (ns_steps=2)", Muon, 3e-4, {"momentum": 0.95, "nesterov": True,
                                            "ns_steps": 2, "use_rms_scale": True}),
    ]:
        W = torch.randn(dim_out, dim_in, device=device, requires_grad=True)
        opt = opt_cls([W], lr=lr, **kwargs)

        history = []
        for step in range(500):
            opt.zero_grad()
            loss = compute_loss(W)
            loss.backward()
            opt.step()
            history.append(loss.item())

        final_loss = history[-1]
        # 找到 loss < 0.01 所需的步数
        steps_needed = next((i for i, v in enumerate(history) if v < 0.01), 500)

        results[name] = {
            "final_loss": final_loss,
            "steps_to_0.01": steps_needed,
            "history": history,
        }

        print(f"  {name} (lr={lr}):")
        print(f"    最终 loss = {final_loss:.6f}")
        print(f"    收敛到 loss<0.01 需 {steps_needed} 步")
        print(f"    最终 ||W - W_true||_F = "
              f"{torch.norm(W - W_true).item():.4f}")
        print()

    # 显存对比
    print("-" * 65)
    print("显存占用对比（以 7B 模型为例）:")
    print("-" * 65)

    total_params = 7e9
    bytes_per_param = 2  # bf16

    muon_params_ratio = 0.80   # ~80% 参数是 2D hidden weights
    adamw_params_ratio = 0.20  # ~20% 参数是 embedding/head/LayerNorm

    muon_mem_params = total_params * muon_params_ratio * bytes_per_param
    muon_mem_momentum = total_params * muon_params_ratio * bytes_per_param  # 只要 m
    adamw_mem_params = total_params * adamw_params_ratio * bytes_per_param
    adamw_mem_m = total_params * adamw_params_ratio * bytes_per_param
    adamw_mem_v = total_params * adamw_params_ratio * bytes_per_param

    print(f"  Muon (用于 ~80% Hidden 参数):")
    print(f"    动量 m: {muon_mem_momentum/1e9:.1f} GB")
    print(f"  AdamW (用于 ~20% 其他参数):")
    print(f"    动量 m: {adamw_mem_m/1e9:.1f} GB")
    print(f"    方差 v: {adamw_mem_v/1e9:.1f} GB")
    total_muon_hybrid = muon_mem_momentum + adamw_mem_m + adamw_mem_v
    total_adamw_full = total_params * 2 * bytes_per_param  # m+v for all
    print(f"\n  Hybrid Muon+AdamW 优化器总显存: {total_muon_hybrid/1e9:.1f} GB")
    print(f"  全 AdamW 优化器总显存:         {total_adamw_full/1e9:.1f} GB")
    print(f"  节省:                           "
          f"{(1 - total_muon_hybrid/total_adamw_full)*100:.0f}%")

    print(f"\n总览:")
    print(f"  Muon 将梯度更新从'逐元素自适应'转变为'矩阵级正交化'")
    print(f"  Newton-Schulz 迭代 FLOP 开销: <1% (典型 LLM 训练)")
    print(f"  最重要的实践创新: 正交化让小方向和小方向同等重要")
    print(f"  生产级验证: Moonshot AI Moonlight (16B) + Kimi K2 (1T)")
    print(f"  建议: 预训练用 Muon+AdamW Hybrid，微调保持 AdamW")
```

---

## 核心公式速查

**Muon 更新规则：**

$$M_t = \beta \cdot M_{t-1} + G_t$$
$$G' = G_t + \beta \cdot M_t \quad \text{(Nesterov)}$$
$$O = \text{NewtonSchulz5}(G')$$
$$W_t = W_{t-1} - \eta \cdot (0.2 \cdot O \cdot \sqrt{\max(d_{\text{out}}, d_{\text{in}})} + \lambda \cdot W_{t-1})$$

**Newton-Schulz 五次多项式迭代：**

$$X_0 = G / \|G\|_F$$
$$X_{k+1} = a X_k + b (X_k X_k^\top) X_k + c (X_k X_k^\top)^2 X_k$$
$$(a, b, c) = (3.4445, -4.7750, 2.0315)$$

**与 AdamW 的关键区别：**

| | AdamW | Muon |
|:--|:-----|:-----|
| 更新粒度 | 逐元素 | 矩阵级 |
| 步长计算 | $\hat{m} / \sqrt{\hat{v} + \epsilon}$ | $\text{NewtonSchulz5}(G_{\text{momentum}})$ |
| 额外状态 | 2× (m, v) | 1× (m) |
| 理论性质 | 自适应学习率 | 正交化 → 方向均衡 |
| FLOP 开销 | baseline | <1% (NS 迭代) |

**使用限制：**
- ✅ 仅用于 2D hidden layer 权重
- ✅ Q、K、V 分别应用（不合并）
- ✅ 预训练为主（微调待验证）
- ❌ 不能用 embedding / head / bias / LayerNorm（这些用 AdamW）
- ❌ 小 batch size (< 64) 效果差（正交化噪声大）
```
