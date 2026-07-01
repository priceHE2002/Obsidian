"""
Batch Normalization
===================
论文: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal
      Covariate Shift" (Ioffe & Szegedy, ICML 2015)
核心贡献: 在每个 mini-batch 上将层输入归一化为零均值单位方差，允许使用 5-30 倍
         更高的学习率，降低对初始化的敏感性，提供正则化效应。
代码结构:
  1. BatchNorm_Manual - 手动实现 BN（含完整数学注释）
  2. BatchNorm_WithRunningStats - 含 running mean/var 的完整实现
  3. BN_vs_LN_vs_GN 对比分析
  4. 为什么 BN 不适合小 batch / 序列模型

核心公式（训练时）:
  μ_B = (1/m) Σ x_i              ← mini-batch 均值
  σ²_B = (1/m) Σ (x_i - μ_B)²    ← mini-batch 方差
  x̂_i = (x_i - μ_B) / √(σ²_B + ε) ← 归一化
  y_i = γ · x̂_i + β               ← 仿射变换

推理时使用 running statistics:
  y = γ · (x - E[x]) / √(Var[x] + ε) + β

与 [[../04_Layer_Normalization/Layer Normalization|LayerNorm]] 的对比:
  BN: 沿 (N, H, W) 归一化 — 跨样本 → 依赖 batch size
  LN: 沿 (C, H, W) 归一化 — 单样本内 → batch 无关
"""

import torch
import torch.nn as nn
import math


# ==============================================================================
# 1. 手动实现 BatchNorm（含完整数学注释）
# ==============================================================================
class BatchNorm_Manual(nn.Module):
    """
    手动实现 Batch Normalization

    归一化轴的区别（以 CNN 特征图 (N, C, H, W) 为例）:
    - BN:  在 (N, H, W) 上计算 μ 和 σ²，逐通道 (C) 独立      ← 跨样本
    - LN:  在 (C, H, W) 上计算 μ 和 σ²，逐样本 (N) 独立      ← 单样本内
    - IN:  在 (H, W) 上计算 μ 和 σ²，逐 (N, C) 独立           ← 单样本单通道
    - GN:  在 (H, W) 上计算 μ 和 σ²，逐 (N, 通道组) 独立

    为什么需要 γ (scale) 和 β (shift)？
    如果归一化把输入限制在零均值单位方差区域，可能移除有用信息。
    γ 和 β 允许网络"撤销"归一化——如果最优表示不是零均值单位方差的。
    例如 sigmoid 的最佳工作区间不是均值 0 方差 1。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数（每个通道独立）
        self.gamma = nn.Parameter(torch.ones(num_features))   # γ，缩放
        self.beta = nn.Parameter(torch.zeros(num_features))   # β，偏移

        # Running statistics（推理时使用，不参与梯度更新）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W) — CNN 特征图
        或 (N, C) — 全连接层
        或 (N, C, L) — 序列

        归一化在除 C 外的所有维度上进行
        """
        # 确定归一化维度
        if x.dim() == 4:  # (N, C, H, W)
            dims = (0, 2, 3)  # 在 N, H, W 上计算统计量
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        elif x.dim() == 2:  # (N, C)
            dims = (0,)  # 在 N 上计算统计量
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")

        if self.training:
            # ===== 训练模式: 使用当前 mini-batch 的统计量 =====

            # 步骤1: mini-batch 均值
            mean = x.mean(dim=dims, keepdim=True)  # μ_B

            # 步骤2: mini-batch 方差
            # 注意：这里用有偏估计（除以 m，不是 m-1），与原论文一致
            var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)  # σ²_B

            # 步骤3: 归一化
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

            # 步骤4: 更新 running statistics (EMA)
            # running_mean = momentum * running_mean + (1-momentum) * mean
            # PyTorch 默认 momentum=0.1，即新值权重 0.9
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * mean.squeeze()
                # 注意：running_var 使用无偏估计 (除以 m-1)
                n = x.numel() // x.size(1)  # 每个通道的元素数
                unbiased_var = var.squeeze() * n / (n - 1) if n > 1 else var.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * unbiased_var
        else:
            # ===== 推理模式: 使用累积的 running statistics =====
            mean = self.running_mean
            var = self.running_var

            if x.dim() == 4:
                mean = mean.view(1, -1, 1, 1)
                var = var.view(1, -1, 1, 1)

            x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 步骤5: 仿射变换（训练和推理共用）
        return gamma * x_norm + beta


# ==============================================================================
# 2. 带完整反向传播推导注释的 BatchNorm
# ==============================================================================
class BatchNorm_WithGrad(nn.Module):
    """
    BatchNorm + 完整梯度推导注释

    前向传播:
      μ_B = mean(x), σ²_B = var(x), x̂ = (x-μ_B)/√(σ²_B+ε), y = γ·x̂+β

    反向传播链式法则（注意：μ_B 和 σ²_B 依赖于 mini-batch 中所有样本）:

    (1) ∂L/∂x̂_i = ∂L/∂y_i · γ

    (2) ∂L/∂σ²_B = Σ ∂L/∂x̂_i · (x_i - μ_B) · (-1/2) · (σ²_B + ε)^(-3/2)

    (3) ∂L/∂μ_B = (Σ ∂L/∂x̂_i · -1/√(σ²_B+ε))
                  + ∂L/∂σ²_B · (Σ -2(x_i-μ_B) / m)

    (4) ∂L/∂x_i = ∂L/∂x̂_i · 1/√(σ²_B+ε)
                  + ∂L/∂σ²_B · 2(x_i-μ_B)/m
                  + ∂L/∂μ_B · 1/m

    (5) ∂L/∂γ = Σ ∂L/∂y_i · x̂_i
    (6) ∂L/∂β = Σ ∂L/∂y_i

    关键: 公式(4)显示 ∂L/∂x_i 依赖于 batch 中所有样本（通过 μ_B 和 σ²_B），
    这意味着 BN 改变了梯度的流——这是一种隐式的正则化。

    为什么 BN 能允许更高学习率？
    Santurkar et al. (2018) 发现，BN 通过平滑损失景观（Loss Landscape）
    来实现这一点。具体来说:
    - BN 使损失函数的 Lipschitz 常数变小
    - 参数的小变化不会被放大为激活值和梯度的剧烈变化
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """同 BatchNorm_Manual"""
        if x.dim() == 4:
            dims = (0, 2, 3)
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
        else:
            dims = (0,)
            gamma = self.gamma.view(1, -1)
            beta = self.beta.view(1, -1)

        if self.training:
            mean = x.mean(dim=dims, keepdim=True)
            var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * mean.squeeze()
                n = x.numel() // x.size(1)
                unbiased_var = var.squeeze() * n / (n - 1) if n > 1 else var.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * unbiased_var
        else:
            mean = self.running_mean.view(1, -1, 1, 1) if x.dim() == 4 \
                   else self.running_mean.view(1, -1)
            var = self.running_var.view(1, -1, 1, 1) if x.dim() == 4 \
                  else self.running_var.view(1, -1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return gamma * x_norm + beta


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Batch Normalization 演示")
    print("=" * 60)

    # ---- 1. BN 前向传播验证 ----
    print("\n--- 1. BN 前向传播 ---")
    N, C, H, W = 4, 3, 8, 8
    x = torch.randn(N, C, H, W) * 2 + 1  # 均值偏移、方差放大

    bn = BatchNorm_Manual(C)

    # 训练模式
    bn.train()
    out_train = bn(x)
    print(f"输入: shape={x.shape}, mean={x.mean():.3f}, std={x.std():.3f}")
    print(f"训练模式输出: mean≈{out_train.mean():.6f}, std≈{out_train.std():.6f}")

    # 验证 BN 归一化到零均值单位方差（训练时）
    channel_means = out_train.mean(dim=(0, 2, 3))
    channel_stds = out_train.std(dim=(0, 2, 3))
    print(f"逐通道均值: {channel_means.detach()} (应 ≈ 0)")
    print(f"逐通道标准差: {channel_stds.detach()} (应 ≈ 1)")

    # 推理模式
    bn.eval()
    out_eval = bn(x)
    print(f"\n推理模式: running_mean={bn.running_mean}, running_var={bn.running_var}")
    print(f"推理模式输出: mean={out_eval.mean():.4f}, std={out_eval.std():.4f}")

    # ---- 2. BN 对小 batch 的影响 ----
    print("\n--- 2. BN 对小 batch 的影响 ---")
    print("当 batch_size 很小（如 2 或 4）时:")
    print("  - μ_B 和 σ²_B 的估计噪声很大")
    print("  - 训练不稳定（高方差）")
    print("  - 这是 BN 的根本局限之一")
    print("  → 解决方案: 使用 [[../04_Layer_Normalization/Layer Normalization|LayerNorm]] 或 GroupNorm")

    # ---- 3. BN vs LN vs GN vs IN 归一化轴对比 ----
    print("\n--- 3. 归一化轴对比 (以 (N, C, H, W) 为例) ---")
    print("  BatchNorm:      沿 (N, H, W) 归一化 → 逐通道")
    print("  LayerNorm:      沿 (C, H, W) 归一化 → 逐样本")
    print("  InstanceNorm:   沿 (H, W) 归一化   → 逐样本逐通道")
    print("  GroupNorm:      沿 (H, W) 归一化   → 逐样本逐通道组")
    print("  [[../19_RMSNorm/RMSNorm|RMSNorm]]:      仅 RMS 缩放（无均值中心化）")

    # ---- 4. BN 为什么不适合 Transformer ----
    print("\n--- 4. BN 为什么不适合序列模型？ ---")
    print("  1. RNN: 不同时间步需要不同的统计量，推理时可能遇到未见时间步")
    print("  2. Transformer: 序列长度可变，padding 破坏 batch 统计量")
    print("  3. 小 batch: 大模型训练时 batch size 通常很小，BN 统计量噪声大")
    print("  → 因此 Transformer 使用 [[../04_Layer_Normalization/Layer Normalization|LayerNorm]]/[[../19_RMSNorm/RMSNorm|RMSNorm]]")
    print("  → 但 CNN 视觉编码器仍推荐 BN（batch>=16 时）")
