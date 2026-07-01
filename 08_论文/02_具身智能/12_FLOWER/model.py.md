---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# FLOWER - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
FLOWER (Flow Matching for VLA) — PyTorch 教学实现

论文: FLOWER: Democratizing Generalist Robot Policies (CoRL 2025)
核心模块:
  1. 中间层特征提取 (从 VLM 倒数第二 quarter 取 hidden states)
  2. Rectified Flow Matching — 直线速度场，4~8 步推理
  3. Action-Specific AdaLN — 共享 Flow Transformer 参数，每类动作独立归一化
  4. 交叉注意力条件化 (视觉/语言特征 → 动作生成)

架构流程:
  VLM Encoder (剪枝后30-50%) → 中间层特征提取
  → Flow Transformer (交叉注意力) → AdaLN 动作头 (每类动作独立)
  → 4步 Rectified Flow 去噪 → 动作序列
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 一、Rectified Flow Matching — 直线速度场
# ============================================================
# Rectified Flow vs 标准 Diffusion:
#   Diffusion: 学习弯曲的去噪路径 (需要更多步)
#   Rectified Flow: 学习从噪声到数据的**直线**速度场 (更快收敛, 更少推理步)
#
# 训练: 学习速度场 v_θ 使得 z_t = (1-t)·z_0 + t·z_1 沿着直线从噪声到数据
# 损失: ||(z_1 - z_0) - v_θ(z_t, t)||²
# 推理: ODE solver 沿速度场积分 (4~8 步即可)

class FlowMatchingLoss(nn.Module):
    """
    Rectified Flow Matching 损失。
    
    为什么用 Flow Matching 而不是扩散？
      - 流匹配学习的是直线路径，扩散学习的是弯曲路径
      - 直线意味着更少的推理步骤（4-8 步 vs 扩散的 10-100 步）
      - 训练也更快收敛，因为目标更简单
    """
    def forward(self, velocity_pred, z_0, z_1, z_t):
        """
        Args:
            velocity_pred: (B, action_dim) — 网络预测的速度场
            z_0: (B, action_dim) — 噪声 (源分布, 通常 N(0,I))
            z_1: (B, action_dim) — 干净动作 (目标分布)
            z_t: (B, action_dim) — 插值点 z_t = (1-t)*z_0 + t*z_1
        Returns:
            loss: 标量
        """
        # 目标速度 = 从噪声到目标动作的直线方向
        target_velocity = z_1 - z_0
        # 让网络预测的速度接近直线方向
        return F.mse_loss(velocity_pred, target_velocity)


# ============================================================
# 二、AdaLN (Adaptive Layer Normalization) — 每类动作独立的归一化
# ============================================================
class ActionSpecificAdaLN(nn.Module):
    """
    为每种动作空间类型分配独立的 AdaLN scale/shift 参数。
    
    设计动机:
      不同机器人有不同动作空间（维数、量纲都不同）—
      如果所有动作类型共享同一个 LN 参数，模型需要额外"翻译"不同量纲，
      这会降低效率。独立的 AdaLN 让每类动作"说自己的语言"。
    
    效率:
      整个 Flow Transformer 的权重共享，只有 LN 的 scale/shift 是独立的。
      这意味着参数量增加极少（~20% 减少 vs 完全独立的动作头）。
    """
    def __init__(self, dim: int, num_action_types: int):
        super().__init__()
        self.num_action_types = num_action_types
        # 每类动作学一组 scale/shift
        self.scales = nn.Parameter(torch.ones(num_action_types, dim))
        self.shifts = nn.Parameter(torch.zeros(num_action_types, dim))

    def forward(self, x, action_type_ids):
        """
        Args:
            x: (B, dim) — 输入特征
            action_type_ids: (B,) — 每样本的动作类型索引 [0, num_action_types-1]
        Returns:
            x: (B, dim) — 条件化后的特征
        """
        scale = self.scales[action_type_ids]   # (B, dim)
        shift = self.shifts[action_type_ids]   # (B, dim)
        return x * (1 + scale) + shift         # 1+scale 初始化为 scale=0 → 恒等变换


# ============================================================
# 三、Flow Transformer — 交叉注意力条件化 + 直线去噪
# ============================================================
class FlowTransformerBlock(nn.Module):
    """
    Flow Transformer 的基础块。
    标准 DiT (Diffusion Transformer) 设计:
      交叉注意力(条件注入) → AdaLN → FFN
    时间步通过 AdaLN-zero 注入（初始化为零以确保训练开始时稳定）。
    """
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # AdaLN 调制参数 (来自时间步嵌入)
        # 使用 zero-init 风格: 初始化 scale 为 0 确保训练第一步是恒等映射
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),  # 3组 scale/shift
        )
        # 初始化为零
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, condition, time_emb):
        """
        Args:
            x: (B, N, dim) — 动作 tokens (带噪声)
            condition: (B, L, dim) — VLM 中间层特征
            time_emb: (B, dim) — 流匹配时间步嵌入
        """
        # AdaLN 调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(time_emb).chunk(6, dim=-1)

        # 交叉注意力 + AdaLN
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.cross_attn(x_norm, condition, condition)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # FFN + AdaLN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class FlowTransformer(nn.Module):
    """
    FLOWER 的 Flow Transformer: 从条件（VLM 中间层特征）生成动作的速度场。
    
    关键设计:
      1. 交叉注意力注入 VLM 中间层特征（而非拼接或 FiLM）
         — 交叉注意力比拼接更灵活：让网络自由选择"关注哪些空间/语义信息"
         — 消融证实: 交叉注意力 > 拼接 > FiLM
      2. AdaLN 时间条件化 — 标准 DiT 做法，零初始化确保训练初期稳定
      3. 动作特定 AdaLN — 每类动作独立的放缩偏移
    """
    def __init__(self, dim: int = 512, depth: int = 8, num_heads: int = 8,
                 num_action_types: int = 3):
        super().__init__()
        self.dim = dim

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 条件投影（VLM 特征 → Flow Transformer 空间）
        self.cond_proj = nn.Linear(dim, dim)

        # Transformer 块
        self.blocks = nn.ModuleList([
            FlowTransformerBlock(dim, num_heads) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(dim)

        # 动作特定 AdaLN 头
        self.action_adaln = ActionSpecificAdaLN(dim, num_action_types)

    def forward(self, z_t, condition, t, action_type_ids):
        """
        Args:
            z_t: (B, action_dim) — 插值点 (噪声和数据的线性插值)
            condition: (B, L, dim) — VLM 中间层特征
            t: (B,) — 流匹配时间步 [0, 1]
            action_type_ids: (B,) — 动作类型编号
        Returns:
            velocity: (B, action_dim) — 预测的速度场 v_θ(z_t, t, condition)
        """
        B, action_dim = z_t.shape
        device = z_t.device

        # 时间嵌入
        t_emb = self.time_mlp(t.float())                         # (B, dim)

        # 条件投影
        cond = self.cond_proj(condition)                          # (B, L, dim)

        # 将动作 z_t 投影到 dim 空间作为 query
        if action_dim != self.dim:
            z_proj = nn.Linear(action_dim, self.dim, device=device)(z_t)
        else:
            z_proj = z_t
        x = z_proj.unsqueeze(1)                                   # (B, 1, dim)

        # Flow Transformer 块
        for block in self.blocks:
            x = block(x, cond, t_emb)

        x = self.final_norm(x.squeeze(1))                         # (B, dim)

        # 动作特定 AdaLN
        x = self.action_adaln(x, action_type_ids)

        # 输出速度场（维度 = 动作维度）
        return nn.Linear(self.dim, action_dim, device=device)(x)


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入 — 标准化扩散/流模型的时间条件化方式。"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================
# 四、FLOWER 完整模型
# ============================================================
class FLOWERModel(nn.Module):
    """
    FLOWER 从 VLM 中间层到动作的流水线。
    
    核心思路:
      1. 不重新训练 VLM，而是冻结它并从中间层提取特征
      2. Flow Transformer 接收这些特征作为条件
      3. 用 Rectified Flow Matching 训练速度场
      4. 推理时沿速度场积分 4~8 步即可得到干净动作
      
    为什么从 VLM 中间层提取特征？
      - 倒数第二 quarter 层提供了最丰富的语义信息
      - 最后一层过度专业化于 next-token 预测（不需要文本生成）
      - 剪掉最后 30% 层可以节省大量显存和推理时间
    """
    def __init__(self, vlm_feat_dim: int = 768, action_dim: int = 7,
                 num_action_types: int = 3, ft_dim: int = 512, ft_depth: int = 6):
        super().__init__()
        # VLM 特征投影
        self.vlm_proj = nn.Linear(vlm_feat_dim, ft_dim)
        # Flow Transformer
        self.flow_transformer = FlowTransformer(
            dim=ft_dim, depth=ft_depth, num_action_types=num_action_types
        )
        self.flow_loss_fn = FlowMatchingLoss()

    def forward(self, vlm_hidden_states, actions, action_type_ids, t=None):
        """
        Args:
            vlm_hidden_states: (B, L, vlm_feat_dim) — VLM 中间层特征
            actions: (B, action_dim) — 真实动作
            action_type_ids: (B,) — 动作类型
            t: (B,) — 流时间步（训练时采样 [0,1]，推理时 None）
        Returns:
            loss / action_pred
        """
        B = actions.size(0)
        device = actions.device

        # 投影 VLM 特征
        condition = self.vlm_proj(vlm_hidden_states)                # (B, L, ft_dim)

        if t is None:
            # 推理模式
            t = torch.zeros(B, device=device)

        # 构建插值点 z_t = (1-t)*z_0 + t*z_1
        z_0 = torch.randn(B, actions.size(-1), device=device)      # 噪声
        z_1 = actions                                               # 干净动作
        t_reshape = t.view(-1, 1)
        z_t = (1 - t_reshape) * z_0 + t_reshape * z_1

        # 预测速度场
        velocity_pred = self.flow_transformer(z_t, condition, t, action_type_ids)

        # 计算 Rectified Flow 损失
        loss = self.flow_loss_fn(velocity_pred, z_0, z_1, z_t)
        return loss

    @torch.no_grad()
    def sample_action(self, vlm_hidden_states, action_type_ids, action_dim: int = 7,
                      num_steps: int = 4):
        """
        推理: 用 ODE solver (Euler方法) 沿速度场积分 num_steps 步。
        
        直线路径优势: 只需 4-8 步即可得到高质量动作。
        """
        B = vlm_hidden_states.size(0)
        device = vlm_hidden_states.device

        condition = self.vlm_proj(vlm_hidden_states)
        z = torch.randn(B, action_dim, device=device)              # 从噪声开始
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = step * dt
            t_tensor = torch.full((B,), t, device=device)
            velocity = self.flow_transformer(z, condition, t_tensor, action_type_ids)
            # Euler 积分: z_{t+dt} = z_t + v_θ(z_t, t) * dt
            z = z + velocity * dt

        return z  # 去噪后的动作


# ============================================================
# 五、__main__ 演示
# ============================================================
if __name__ == "__main__":
    B = 8
    vlm_feat_dim = 768     # VLM 中间层 hidden_dim (如 Florence-2 Encoder)
    L = 64                 # VLM token 数量
    action_dim = 7
    num_action_types = 3   # e.g., 0:单臂7D, 1:双臂14D, 2:移动底座3D

    model = FLOWERModel(
        vlm_feat_dim=vlm_feat_dim,
        action_dim=action_dim,
        num_action_types=num_action_types,
        ft_dim=512,
        ft_depth=6,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -- 训练 --
    model.train()
    vlm_hidden = torch.randn(B, L, vlm_feat_dim)         # 模拟 VLM 中间层特征
    actions = torch.randn(B, action_dim)                  # 真实动作
    action_type_ids = torch.randint(0, num_action_types, (B,))
    t = torch.rand(B)                                     # 随机时间步 [0, 1]

    loss = model(vlm_hidden, actions, action_type_ids, t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[训练] Rectified Flow 损失: {loss.item():.4f}")

    # -- 推理: 单臂 4 步去噪 --
    model.eval()
    with torch.no_grad():
        action = model.sample_action(
            vlm_hidden,
            action_type_ids=torch.zeros(B, dtype=torch.long),
            action_dim=action_dim,
            num_steps=4,
        )
    print(f"[推理] 4步去噪动作: {action.shape}  (应为 [{B}, {action_dim}])")

    # -- 推理: 双臂 8 步去噪 --
    dual_action_dim = 14
    with torch.no_grad():
        # 注意：实际中需要对不同 action_dim 做独立输出投影
        # 这里的演示用相同的 action_dim 展示流程
        action_dual = model.sample_action(
            vlm_hidden,
            action_type_ids=torch.ones(B, dtype=torch.long),
            action_dim=action_dim,  # 简化: 实际应为14
            num_steps=8,
        )
    print(f"[推理] 8步去噪动作: {action_dual.shape}")
```
