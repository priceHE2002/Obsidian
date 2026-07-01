---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# GaLore 完整实现 - 基于 [[GaLore]] (Zhao et al., ICML 2024) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
GaLore 完整实现 - 基于 [[GaLore]] (Zhao et al., ICML 2024)

实现 SVD 梯度分解、低秩投影累积、梯度外推、训练循环。
核心洞察：LLM 训练中的梯度矩阵 ∇W 的有效秩约为满秩的 10%——
前 10% 的奇异值贡献了 90% 的梯度能量。GaLore 将优化器状态
（Adam 的 momentum 和 variance）存储在低秩子空间中，
使全参数训练的显存从 O(mn) 降到 O((m+n)r)。

与 [[LoRA]] 的根本区别:
- LoRA: 参数更新本身就低秩（ΔW = BA，秩 ≤ r）→ PEFT
- GaLore: 参数更新是全秩的，但优化器状态在低秩空间管理 → 全参数训练

核心公式: ρ = P^T ∇W Q  (投影到低秩空间)
         M̃, Ṽ = Adam(ρ)   (低秩空间优化)
         ΔW = -η P · M̃/√(Ṽ+ε) · Q^T  (反投影到全秩)
         其中 P, Q 周期性通过 SVD(∇W) 更新

参考:
- [[GaLore]] - 原始论文 (ICML 2024)
- [[LoRA]] - 参数低秩更新的 PEFT 对比
- [[ReLoRA]] - 周期性合并低秩更新的相关工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


# ============================================================
# 一、SVD 梯度分解与投影
# ============================================================

class GradientProjector:
    """
    梯度低秩投影器——维护 P, Q 投影矩阵。

    WHY 用 SVD 更新投影矩阵？
    梯度 ∇W 的 SVD 分解 ∇W = U Σ V^T 提供了最优的低秩近似：
    前 r 个奇异值对应的左奇异向量 U_r 和右奇异向量 V_r
    构成最优的秩-r 近似子空间。

    WHY 周期性更新（而非每步更新）？
    SVD 计算开销约为正常前向一步的 0.1-0.5%。梯度子空间在训练中
    变化缓慢——实践表明每 200-1000 步更新一次足够。
    """

    def __init__(self, m: int, n: int, r: int, update_interval: int = 200):
        """
        Args:
            m: 权重矩阵的行数（out_features）
            n: 权重矩阵的列数（in_features）
            r: 低秩子空间的秩。GaLore 推荐 r ∈ [64, 256]
            update_interval: 投影矩阵的更新间隔（步数）
        """
        self.m = m
        self.n = n
        self.r = r
        self.update_interval = update_interval

        # ---- 投影矩阵 ----
        # P: 左投影矩阵 (m, r)
        # Q: 右投影矩阵 (n, r)
        # 随机初始化——将在第一次 update 时被 SVD 结果替换
        self.P: torch.Tensor = torch.randn(m, r)
        self.Q: torch.Tensor = torch.randn(n, r)

        # ---- 梯度累加器（用于周期性 SVD）----
        self.grad_buffer: Optional[torch.Tensor] = None  # (m, n)
        self.step_count: int = 0

    def project(self, grad: torch.Tensor) -> torch.Tensor:
        """
        将全秩梯度投影到低秩子空间。

        WHY 投影？
        直接存储 ∇W 的优化器状态需要 2 × m × n 浮点数（Adam）。
        在低秩空间中，ρ = P^T ∇W Q 的优化器状态只需 2 × r × r 浮点数。

        Args:
            grad: 全秩梯度 (m, n)

        Returns:
            rho: 低秩投影梯度 (r, r)
        """
        # 累加梯度（用于后续可能的 SVD 更新）
        if self.grad_buffer is None:
            self.grad_buffer = grad.detach().clone()
        else:
            self.grad_buffer += grad.detach()

        self.step_count += 1

        # 投影到低秩空间
        # P 和 Q 应与 grad 在同一设备
        P_dev = self.P.to(grad.device)
        Q_dev = self.Q.to(grad.device)
        rho = P_dev.t() @ grad @ Q_dev  # (r, r)
        return rho

    def project_back(self, rho_update: torch.Tensor) -> torch.Tensor:
        """
        将低秩空间的更新反投影回全秩空间。

        WHY 反投影？
        优化器在低秩空间 (r, r) 中计算参数更新后，
        需要将其映射回原始参数空间 (m, n)。

        Args:
            rho_update: 低秩空间的参数更新 (r, r)（如 Adam 的 Δ）

        Returns:
            full_update: 全秩参数更新 (m, n)
        """
        P_dev = self.P.to(rho_update.device)
        Q_dev = self.Q.to(rho_update.device)
        full_update = P_dev @ rho_update @ Q_dev.t()  # (m, n)
        return full_update

    def maybe_update_projections(self, force: bool = False):
        """
        周期性更新投影矩阵（基于累积梯度的 SVD）。

        WHY SVD 而非在线方法？
        SVD 提供最优低秩近似（Eckart-Young-Mirsky 定理）。
        虽然计算量大，但更新间隔 T ≥ 200 使总体开销可忽略。

        Args:
            force: 是否强制更新（忽略 update_interval 检查）
        """
        if not force and self.step_count < self.update_interval:
            return False
        if self.grad_buffer is None:
            return False

        # ---- SVD 分解累积梯度 ----
        # WHY 用累积梯度而非单步梯度？
        # 累积梯度提供了更稳定的子空间估计——单步梯度受 batch 噪声影响
        U, S, Vt = torch.linalg.svd(self.grad_buffer.float(), full_matrices=False)
        # U: (m, min(m,n)), S: (min(m,n),), Vt: (min(m,n), n)

        # 取前 r 个奇异向量
        r_actual = min(self.r, S.size(0))
        self.P = U[:, :r_actual].to(self.grad_buffer.dtype)  # (m, r)
        self.Q = Vt[:r_actual, :].t().to(self.grad_buffer.dtype)  # (n, r)

        # ---- 重置累加器和计数器 ----
        self.grad_buffer = None
        self.step_count = 0

        # ---- 计算有效秩（用于监控）----
        total_energy = (S ** 2).sum().item()
        captured = (S[:r_actual] ** 2).sum().item() / total_energy
        print(f"  [GaLore SVD] 更新投影矩阵: r={r_actual}, "
              f"捕获梯度能量: {captured*100:.1f}%")

        return True


# ============================================================
# 二、GaLore Adam 优化器
# ============================================================

class GaLoreAdam:
    """
    在低秩空间中运行的 Adam 优化器。

    WHY 不用标准 Adam？
    标准 Adam 为每个参数存储 momentum (M) 和 variance (V)——
    对于 m×n 的权重，即 2×m×n 浮点数。
    GaLore Adam 只存储低秩投影 ρ 的 M̃ 和 Ṽ —— 2×r×r 浮点数。
    当 r=128, m=n=4096 时，这是 ~30x 的显存节省。
    """

    def __init__(
        self,
        r: int,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.r = r
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # ---- 低秩空间的优化器状态 ----
        # 大小仅为 2 × r × r （vs 标准 Adam 的 2 × m × n）
        self.M: Optional[torch.Tensor] = None  # momentum (r, r)
        self.V: Optional[torch.Tensor] = None  # variance (r, r)
        self.t: int = 0  # 步数计数器

    def step(self, rho: torch.Tensor) -> torch.Tensor:
        """
        在低秩空间中执行一步 Adam 更新。

        Args:
            rho: 低秩投影梯度 (r, r)

        Returns:
            rho_update: 低秩空间的参数更新 (r, r)
        """
        self.t += 1

        # ---- 初始化优化器状态 ----
        if self.M is None:
            self.M = torch.zeros_like(rho)
            self.V = torch.zeros_like(rho)

        # ---- Adam 更新 ----
        beta1, beta2 = self.betas

        # 偏差修正
        self.M = beta1 * self.M + (1 - beta1) * rho
        self.V = beta2 * self.V + (1 - beta2) * (rho ** 2)

        m_hat = self.M / (1 - beta1 ** self.t)
        v_hat = self.V / (1 - beta2 ** self.t)

        # 低秩空间的参数更新量
        rho_update = m_hat / (torch.sqrt(v_hat) + self.eps)

        # 权重衰减（直接在 rho 上而非原始权重上）
        if self.weight_decay > 0:
            rho_update = rho_update + self.weight_decay * rho

        return -self.lr * rho_update


# ============================================================
# 三、GaLore 线性层
# ============================================================

class GaLoreLinear(nn.Module):
    """
    GaLore 线性层——全参数更新 + 低秩优化器状态。

    WHY 梯度外推（Gradient Scaling）？
    GaLore 论文发现，低秩投影后的梯度 ρ = P^T ∇W Q 在幅度上
    与全秩梯度有系统性偏差。为保证与标准 Adam 等价的更新幅度，
    需要对投影梯度进行适当的缩放（通常在 1.0-2.0 范围）。

    训练流程:
    1. 前向: 正常计算 (fp16/bf16)
    2. 反向: 正常计算梯度 ∇W (autograd)
    3. GaLore: 投影 ∇W → ρ
    4. Adam: 在低秩空间更新 M̃, Ṽ
    5. 反投影: ΔW = P · Adam(ρ) · Q^T
    6. 应用更新: W -= η · ΔW
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 128,
        lr: float = 1e-3,
        svd_interval: int = 200,
        gradient_scale: float = 1.0,
    ):
        """
        Args:
            in_features: 输入维度 n
            out_features: 输出维度 m
            r: 低秩子空间的秩
            lr: 学习率
            svd_interval: SVD 更新间隔（步数）
            gradient_scale: 投影梯度的缩放因子（梯度外推）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.gradient_scale = gradient_scale

        # ---- 全秩权重（完整更新）----
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # ---- GaLore 组件 ----
        self.projector = GradientProjector(out_features, in_features, r, svd_interval)
        self.optimizer = GaLoreAdam(r=r, lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准前向传播——与普通 nn.Linear 完全相同。

        WHY 前向不变？
        GaLore 只改变优化器状态的管理方式，不影响前向/反向的计算图。
        """
        return F.linear(x, self.weight, self.bias)

    def galore_step(self):
        """
        执行一步 GaLore 参数更新。

        WHY 在外部调用？
        通常在 optimizer.step() 的地方改为调用此方法。
        GaLore 的梯度处理流程替代了标准优化器的 step。
        """
        if self.weight.grad is None:
            return

        grad = self.weight.grad  # (out_features, in_features)

        # ---- 步骤 1: 梯度外推（可选）----
        # WHY 梯度外推？
        # 低秩投影可能系统性地低估梯度幅度。
        # 外推因子通常在 [1.0, 2.0] 范围。
        grad = grad * self.gradient_scale

        # ---- 步骤 2: 投影到低秩空间 ----
        rho = self.projector.project(grad)  # (r, r)

        # ---- 步骤 3: 低秩空间 Adam 更新 ----
        rho_update = self.optimizer.step(rho)  # (r, r)

        # ---- 步骤 4: 反投影到全秩空间 ----
        full_update = self.projector.project_back(rho_update)  # (out_features, in_features)

        # ---- 步骤 5: 应用参数更新 ----
        with torch.no_grad():
            self.weight.data += full_update.to(dtype=self.weight.dtype, device=self.weight.device)

        # ---- 步骤 6: 周期性更新投影矩阵 ----
        self.projector.maybe_update_projections()

    def reset_optimizer_state(self):
        """重置优化器状态（如切换训练阶段时）。"""
        self.optimizer.M = None
        self.optimizer.V = None
        self.optimizer.t = 0


# ============================================================
# 四、GaLore 训练循环
# ============================================================

class GaLoreTrainer:
    """
    GaLore 全参数训练循环。

    WHY 自定义训练循环？
    GaLore 需要替换标准 PyTorch 优化器的 step() 逻辑——
    梯度先从全秩投影到低秩、优化、再反投影。这与
    optimizer.step() 的"直接更新参数"范式不兼容。
    """

    def __init__(
        self,
        galore_layers: Dict[str, GaLoreLinear],
        lr: float = 1e-3,
    ):
        self.galore_layers = galore_layers
        self.lr = lr

    def training_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        执行一个训练步（含前向、反向、GaLore 更新）。

        Returns:
            loss: 训练损失
        """
        # ---- 前向传播 ----
        # （在实际模型中，需要通过整个 Transformer 层）
        # 此处简化为直接调用各 GaLore 层的 forward

        # ---- 反向传播 ----
        # loss.backward() —— 计算所有参数的梯度 ∇W

        # ---- GaLore 参数更新 ----
        for name, layer in self.galore_layers.items():
            layer.galore_step()  # 替代 optimizer.step()

        return 0.0  # 简化的演示

    def memory_estimate(self, m: int, n: int, r: int) -> dict:
        """
        估算 GaLore 的显存节省。

        WHY 这个估算？
        在实际使用前需要知道能否在目标 GPU 上运行。
        """
        # 标准 Adam 优化器状态
        standard_adam_bytes = 2 * m * n * 4  # fp32 momentum + variance
        # GaLore 优化器状态（低秩空间）
        galore_adam_bytes = 2 * r * r * 4  # fp32 M̃, Ṽ
        # 投影矩阵存储
        proj_bytes = (m + n) * r * 2  # bf16 P, Q

        return {
            "standard_adam_gb": standard_adam_bytes / 1e9,
            "galore_adam_gb": galore_adam_bytes / 1e9,
            "projections_gb": proj_bytes / 1e9,
            "total_galore_gb": (galore_adam_bytes + proj_bytes) / 1e9,
            "saving_ratio": (galore_adam_bytes + proj_bytes) / standard_adam_bytes,
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GaLore 演示: SVD 分解 + 低秩投影 + Adam 更新 + 训练循环")
    print("=" * 60)

    # ---- 1. SVD 分解与梯度能量 ----
    print("\n[1] 梯度低秩性展示")
    torch.manual_seed(42)
    # 模拟一个 4096×4096 权重的梯度（模拟 LLM 中的注意力投影）
    grad = torch.randn(256, 256)
    # 使其低秩（模拟真实梯度流）
    U_true = torch.randn(256, 16)
    V_true = torch.randn(256, 16)
    grad = U_true @ V_true.t() + torch.randn(256, 256) * 0.01  # 低秩 + 噪声

    U, S, Vt = torch.linalg.svd(grad, full_matrices=False)
    total_energy = (S ** 2).sum().item()
    cumulative = torch.cumsum(S ** 2, dim=0) / total_energy
    for r_test in [4, 8, 16, 32, 64]:
        print(f"  r={r_test:>3d}: 捕获 {cumulative[r_test-1]*100:.1f}% 梯度能量")

    # ---- 2. 投影矩阵创建与更新 ----
    print("\n[2] 投影矩阵创建与梯度投影")
    projector = GradientProjector(m=256, n=256, r=16, update_interval=10)

    # 模拟多步梯度
    for step in range(10):
        fake_grad = U_true @ V_true.t() + torch.randn(256, 256) * 0.02
        rho = projector.project(fake_grad)
        if step == 0:
            print(f"  步 {step}: ∇W ({fake_grad.shape}) → ρ ({rho.shape})")
            print(f"  ρ 范数: {rho.norm().item():.4f}")

    # 更新投影矩阵
    projector.maybe_update_projections(force=True)
    print(f"  P 形状: {projector.P.shape}, Q 形状: {projector.Q.shape}")

    # ---- 3. GaLore Adam 优化 ----
    print("\n[3] GaLore Adam 低秩空间优化")
    adam = GaLoreAdam(r=16, lr=1e-3)

    projector2 = GradientProjector(m=256, n=256, r=16)
    for step in range(5):
        fake_grad = U_true @ V_true.t() + torch.randn(256, 256) * 0.02
        rho = projector2.project(fake_grad)
        rho_update = adam.step(rho)
        if step < 3:
            print(f"  步 {step}: ||ρ_update|| = {rho_update.norm().item():.4f}")

    # 反投影
    full_update = projector2.project_back(rho_update)
    print(f"  反投影更新形状: {full_update.shape}")
    print(f"  ||full_update|| = {full_update.norm().item():.4f}")

    # ---- 4. GaLore 训练演示 ----
    print("\n[4] GaLore 训练步骤演示")
    galore_layer = GaLoreLinear(256, 256, r=16, svd_interval=10)

    x = torch.randn(4, 256)
    out = galore_layer(x)
    loss = out.sum()
    loss.backward()

    # 记录更新前的权重
    w_before = galore_layer.weight.data.clone()
    galore_layer.galore_step()
    w_after = galore_layer.weight.data

    update_norm = (w_after - w_before).norm().item()
    print(f"  权重更新范数: {update_norm:.6f}")
    print(f"  学习率: {galore_layer.optimizer.lr}")

    # ---- 5. 显存节省估算 ----
    print("\n[5] 显存节省估算")
    trainer = GaLoreTrainer({})
    for m_n, label in [((1024, 1024), "QKV 投影"), ((4096, 4096), "注意力"), ((11008, 4096), "FFN")]:
        est = trainer.memory_estimate(m_n[0], m_n[1], r=128)
        print(f"  {label} ({m_n[0]}×{m_n[1]}): "
              f"标准 Adam≈{est['standard_adam_gb']:.4f}GB, "
              f"GaLore≈{est['total_galore_gb']:.4f}GB "
              f"(节省 {1/est['saving_ratio']:.1f}x)")

    # ---- 6. GaLore vs LoRA 对比 ----
    print("\n[6] GaLore vs LoRA 对比")
    print("  | 维度         | LoRA            | GaLore                   |")
    print("  |--------------|-----------------|--------------------------|")
    print("  | 参数更新     | 低秩 ΔW=BA      | 全秩 ΔW=P·Adam(ρ)·Q^T   |")
    print("  | 优化器状态   | 适配器参数量级  | 低秩 r×r                 |")
    print("  | 表达能力     | rank(ΔW) ≤ r    | rank(ΔW) ≤ min(m,n)      |")
    print("  | 训练模式     | PEFT / 微调     | 全参数预训练 + 微调      |")
    print("  | 7B 训练显存  | 18 GB           | 28 GB                    |")
    print("  | 预训练可用性 | ❌ 不可用        | ✅ 可从头预训练          |")

    print("\n" + "=" * 60)
    print("演示完成。GaLore 使 7B 全参数训练在单卡 RTX 4090 上成为可能。")
    print("=" * 60)

```
