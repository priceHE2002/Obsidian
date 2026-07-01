---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# AttenA+ - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
AttenA+ (Action Inequality Rectification) — PyTorch 教学实现

论文: AttenA+: Rectifying Action Inequality in Robotic Foundation Models (arXiv 2605.13548)
核心思想:
  - "动作不平等": 低速精细操作比高速位移更重要，但训练时损失权重相同
  - 解决: 用动作速度作为"重要性"的自然代理变量，对低速步骤加权
  - 零架构修改、零额外参数 — 纯靠损失重加权

方法:
  损失 = w(v) * L_original
  其中 v 是动作速度（从动作向量直接计算），w(v) 随 v 增大而减小
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 一、核心：速度感知损失加权
# ============================================================
class AttenAPlusLoss(nn.Module):
    """
    AttenA+ 的多种速度加权策略。
    
    设计哲学:
      动作速度慢 → 机器人正在精细操作（抓取、插入） → 这步很重要 → 高损失权重
      动作速度快 → 机器人正在空载移动 → 容错率高 → 低损失权重
    
    为什么用速度作为重要性的代理变量？
      - 不需要人工标注（自动化）
      - 速度从动作向量可直接计算（无需额外数据）
      - 慢动作几乎总是对应精细操作（物理约束）
    """
    def __init__(self, strategy: str = "inverse", epsilon: float = 1e-6,
                 max_weight: float = 10.0, lambda_param: float = 1.0,
                 normalize: bool = True):
        """
        Args:
            strategy: 权重映射策略
                - "inverse": w(v) = 1/(v+ε)
                - "inverse_sq": w(v) = 1/(v²+ε)  更激进强调低速
                - "exponential": w(v) = exp(-λ·v)  平滑衰减
                - "logarithmic": w(v) = -log(v+ε)  对速度变化不敏感
            epsilon: 防止除零的小常数
            max_weight: 权重截断上限（防止极低速度产生过大梯度）
            lambda_param: 指数衰减的 λ 参数
            normalize: 是否归一化权重使得 E[w] = 1（保持总损失量级不变）
        """
        super().__init__()
        self.strategy = strategy
        self.epsilon = epsilon
        self.max_weight = max_weight
        self.lambda_param = lambda_param
        self.normalize = normalize

    def compute_weight(self, velocity):
        """
        根据动作速度计算损失权重。
        Args:
            velocity: (B,) 或 (B, T) — 每个样本/时间步的动作速度 (标量)
        Returns:
            weight: 同 shape — 损失权重
        """
        if self.strategy == "inverse":
            # 最简单的反比权重
            weight = 1.0 / (velocity + self.epsilon)
        elif self.strategy == "inverse_sq":
            # 平方反比: 对低速的强调更激进
            # 适用于精细操作占比很小的任务
            weight = 1.0 / (velocity.pow(2) + self.epsilon)
        elif self.strategy == "exponential":
            # 指数衰减: 通过 λ 控制衰减速度
            # λ 越大 → 低速权重相对越大
            weight = torch.exp(-self.lambda_param * velocity)
        elif self.strategy == "logarithmic":
            # 对数: 对极低速度不敏感，适合速度分布极不均匀的场景
            weight = -torch.log(velocity + self.epsilon)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # 权重截断: 防止极低速度导致的数值不稳定
        weight = torch.clamp(weight, max=self.max_weight)

        # 归一化: 保持总损失量级不变
        # 如果不归一化，加权后的总损失可能远大于 / 小于原始损失
        if self.normalize:
            mean_w = weight.mean().clamp(min=self.epsilon)
            weight = weight / mean_w

        return weight


# ============================================================
# 二、动作速度计算（从动作向量提取标量速度）
# ============================================================
def compute_action_velocity(actions, method: str = "l2"):
    """
    从动作向量计算"速度"标量。
    
    为什么需要这一步？
      动作向量包含多个维度的位移（关节位置、末端位姿等），
      我们需要一个标量来代表"这步动作有多大"。
    
    Args:
        actions: (B, action_dim) 或 (B, T, action_dim) — 动作向量
        method: "l2" (L2范数) | "max" (最大分量) | "mean_abs" (平均绝对值)
    Returns:
        velocity: (B,) 或 (B, T) — 动作速度标量
    """
    if actions.dim() == 2:
        dim = -1
    else:
        dim = -1

    if method == "l2":
        # L2 范数: 综合所有维度的位移大小
        velocity = torch.norm(actions, p=2, dim=dim)
    elif method == "max":
        # 最大分量: 关注"最剧烈的那个维度"
        velocity = actions.abs().max(dim=dim).values
    elif method == "mean_abs":
        # 平均绝对值: 各维度等权重
        velocity = actions.abs().mean(dim=dim)
    else:
        raise ValueError(f"Unknown method: {method}")

    return velocity


# ============================================================
# 三、AttenA+ 包装器 — 对任意损失的即插即用增强
# ============================================================
def attena_plus_wrap(loss_per_sample, actions, strategy: str = "inverse",
                     epsilon: float = 1e-6, max_weight: float = 10.0,
                     lambda_param: float = 1.0, vel_method: str = "l2",
                     normalize: bool = True):
    """
    对任意逐样本损失施加 AttenA+ 速度加权。
    
    这是最实用的接口 — 直接包装你现有的损失函数。
    
    用法示例:
        loss_per_sample = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
        weighted_loss = attena_plus_wrap(loss_per_sample, target, strategy='inverse')
    
    Args:
        loss_per_sample: (B,) — 每个样本的损失值
        actions: (B, action_dim) — 动作向量（用于计算速度）
        strategy: 权重映射策略
        epsilon: 防除零常数
        max_weight: 权重截断上限
        lambda_param: 指数衰减参数
        vel_method: 速度计算方法
        normalize: 是否归一化权重
    Returns:
        weighted_loss: 标量 — 加权平均损失
    """
    velocity = compute_action_velocity(actions, method=vel_method)
    loss_module = AttenAPlusLoss(
        strategy=strategy, epsilon=epsilon, max_weight=max_weight,
        lambda_param=lambda_param, normalize=normalize,
    )
    weight = loss_module.compute_weight(velocity)
    return (weight * loss_per_sample).mean()


# ============================================================
# 四、__main__ 演示：对比不同策略
# ============================================================
if __name__ == "__main__":
    # -- 模拟数据: 假设 8 个样本的动作速度分布 --
    B = 8
    action_dim = 7

    # 模拟动作: 前4个是低速精细操作，后4个是高速位移
    actions = torch.randn(B, action_dim)
    actions[:4] *= 0.1    # 低速: 精细操作（抓取/插入）
    actions[4:] *= 1.0    # 高速: 空载移动

    # 按速度排序便于观察
    velocities = compute_action_velocity(actions, method="l2")
    sorted_idx = torch.argsort(velocities)
    print("动作速度分布（升序）:")
    for i, idx in enumerate(sorted_idx):
        label = "低速精细" if idx < 4 else "高速位移"
        print(f"  样本{idx} ({label}): 速度={velocities[idx]:.4f}")

    # -- 模拟损失: 所有样本损失相同（便于对比） --
    # 假设所有样本都有相同的 MSE —— 但 AttenA+ 会给低速样本更高权重
    loss_per_sample = torch.ones(B)  # 每个样本损失 = 1.0

    print("\n--- 策略对比 (所有样本原始损失 = 1.0) ---")
    strategies = ["inverse", "inverse_sq", "exponential", "logarithmic"]
    for strategy in strategies:
        weighted_loss = attena_plus_wrap(
            loss_per_sample, actions,
            strategy=strategy,
            vel_method="l2",
        )
        print(f"  {strategy:>12s}: 加权后总损失 = {weighted_loss:.4f}")

    # -- 演示: 低速样本的权重 vs 高速样本的权重 --
    print("\n--- 各样本的 inverse 权重 ---")
    loss_module = AttenAPlusLoss(strategy="inverse")
    for idx in sorted_idx:
        vel = velocities[idx]
        w = loss_module.compute_weight(vel.unsqueeze(0)).item()
        label = "低速精细" if idx < 4 else "高速位移"
        print(f"  样本{idx} ({label}): 速度={vel:.4f} → 权重={w:.4f}")

    # -- 演示: 即插即用到判别式 VLA --
    print("\n--- 演示: 判别式 VLA 集成 AttenA+ ---")
    pred = torch.randn(B, action_dim)
    target = torch.randn(B, action_dim)
    # 原始 MSE
    mse_original = F.mse_loss(pred, target)
    # AttenA+ 加权的 MSE
    loss_each = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
    mse_attena = attena_plus_wrap(loss_each, target, strategy='inverse')
    print(f"  原始 MSE: {mse_original.item():.4f}")
    print(f"  AttenA+ MSE: {mse_attena.item():.4f}")

    # -- 演示: 即插即用到 Flow Matching 损失 --
    print("\n--- 演示: Flow Matching 集成 AttenA+ ---")
    velocity_pred = torch.randn(B, action_dim)
    z_0 = torch.randn(B, action_dim)
    z_1 = target  # 真实动作
    target_vel = z_1 - z_0
    fm_loss_each = F.mse_loss(velocity_pred, target_vel, reduction='none').mean(dim=-1)
    fm_attena = attena_plus_wrap(fm_loss_each, z_1, strategy='inverse')
    print(f"  AttenA+ Flow Matching 损失: {fm_attena.item():.4f}")
```
