---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# LDA-1B - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
LDA-1B (Latent Dynamics Action Model) — PyTorch 教学实现

论文: LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion (RSS 2026)
核心模块:
  1. MM-DiT (Multi-Modal Diffusion Transformer) in DINO latent space
  2. 深度可分离卷积视觉编码器（轻量化设计）
  3. 全域数据利用 (三种训练角色: 动作预测 / 逆动力学 / 视觉预测)
  4. 知识蒸馏 (大模型 → 小模型)

架构流程:
  图像 → 深度可分离卷积编码 → DINO 语义潜在空间
  → MM-DiT (视觉预测 + 逆动力学 + 动作预测)
  → 动作输出 / 下一帧预测
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 一、深度可分离卷积 — 轻量化视觉骨干
# ============================================================
class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积: 将标准卷积分解为 Depthwise + Pointwise。
    
    为什么用深度可分离卷积？
      LDA-1B 的目标是 1.6B 参数的轻量 WAM——
      相比标准卷积，深度可分离卷积参数量减少约 8-9 倍，
      同时保持相近的表达能力。这对消费级硬件推理至关重要。
    
    参数量对比: K²×C_in×C_out → K²×C_in + C_in×C_out
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: 每个输入通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels, bias=False,
        )
        # Pointwise: 1×1 卷积混合通道
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # SiLU (= Swish) 比 ReLU 在深层网络中更平滑

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class LightweightVisualEncoder(nn.Module):
    """
    轻量级视觉编码器: 多层深度可分离卷积 + 全局池化。
    
    设计目标: 在保持足够语义提取能力的前提下，最大限度减少参数和计算量。
    输出投影到 DINO 语义潜在空间维度（通常是 768 或 1024 维）。
    """
    def __init__(self, in_channels=3, base_channels=32, latent_dim=768):
        super().__init__()
        # Stem: 第一层用标准卷积（输入只有3通道，depthwise 无优势）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )

        # 逐阶段加深和下采样
        self.stage1 = self._make_stage(base_channels, base_channels * 2, stride=2)     # 64
        self.stage2 = self._make_stage(base_channels * 2, base_channels * 4, stride=2)  # 128
        self.stage3 = self._make_stage(base_channels * 4, base_channels * 8, stride=2)  # 256
        self.stage4 = self._make_stage(base_channels * 8, base_channels * 8, stride=1)  # 256

        # 输出投影到 DINO 潜在空间
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(base_channels * 8, latent_dim)

    def _make_stage(self, in_ch, out_ch, stride):
        return nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch, stride=stride),
            DepthwiseSeparableConv(out_ch, out_ch),
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.pool(feat).flatten(1)    # (B, C)
        return self.proj(feat)                # (B, latent_dim)


# ============================================================
# 二、MM-DiT 核心 — 多模态扩散 Transformer
# ============================================================
class MMDiTBlock(nn.Module):
    """
    MM-DiT 的基本块。
    同时处理视觉 token 和动作 token 两条流，
    通过交叉注意力交换信息。
    
    核心设计:
      两条流独立做 self-attention，然后互相做 cross-attention。
      这比把所有 token 混在一起更有效——因为视觉和动作的语义空间不同。
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # -- 视觉流 --
        self.vis_norm1 = nn.LayerNorm(dim)
        self.vis_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.vis_norm2 = nn.LayerNorm(dim)
        self.vis_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.vis_norm3 = nn.LayerNorm(dim)
        self.vis_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # -- 动作流 --
        self.act_norm1 = nn.LayerNorm(dim)
        self.act_self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.act_norm2 = nn.LayerNorm(dim)
        self.act_cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.act_norm3 = nn.LayerNorm(dim)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, vis_tokens, act_tokens):
        # -- 视觉流: self-attn + cross-attn(到动作) + MLP --
        v = vis_tokens
        v = v + self.vis_self_attn(
            self.vis_norm1(v), self.vis_norm1(v), self.vis_norm1(v)
        )[0]
        v = v + self.vis_cross_attn(
            self.vis_norm2(v), self.act_norm2(act_tokens), self.act_norm2(act_tokens)
        )[0]
        v = v + self.vis_mlp(self.vis_norm3(v))

        # -- 动作流: self-attn + cross-attn(到视觉) + MLP --
        a = act_tokens
        a = a + self.act_self_attn(
            self.act_norm1(a), self.act_norm1(a), self.act_norm1(a)
        )[0]
        a = a + self.act_cross_attn(
            self.act_norm2(a), self.vis_norm2(vis_tokens), self.vis_norm2(vis_tokens)
        )[0]
        a = a + self.act_mlp(self.act_norm3(a))

        return v, a


class MMDiT(nn.Module):
    """
    MM-DiT (Multi-Modal Diffusion Transformer): LDA-1B 的核心。
    
    两条 token 流:
      视觉流: 当前帧的语义特征 (来自 DINO latent space)
      动作流: 噪声动作 / 中间特征
    
    为什么在 DINO 潜在空间工作？
      - 像素空间中预测 → 55.8M 参数模型仅 14.2% 成功率
      - DINO 空间中预测 → 同参数模型 55.4% 成功率
      - DINO 天然聚焦于物体语义（"这是什么、它在哪"），
        而非纹理、光照等无关细节
    """
    def __init__(self, dim=512, depth=8, num_heads=8, num_vis_tokens=64):
        super().__init__()
        self.num_vis_tokens = num_vis_tokens

        # 可学习的视觉 query tokens（类似 Perceiver IO）
        self.vis_queries = nn.Parameter(torch.randn(1, num_vis_tokens, dim) * 0.02)

        # 视觉编码器输出投影
        self.vis_input_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # 动作 token 投影
        self.act_input_proj = nn.Linear(dim, dim)  # action_dim → dim

        # MM-DiT blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(dim, num_heads) for _ in range(depth)
        ])

        # 输出头
        self.vis_output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 预测下一帧的 DINO 特征
        )
        self.act_output_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),  # 预测动作 / 逆动力学
        )

    def forward(self, vis_feat, act_tokens):
        """
        Args:
            vis_feat: (B, vis_token_dim) — 当前帧的 DINO 特征
            act_tokens: (B, n_act_tokens, dim) — 噪声动作 tokens (扩散用) 
                       或 (B, 1, dim) — 查询 token (逆动力学用)
        Returns:
            vis_pred: (B, vis_token_dim) — 预测的下一帧特征
            act_out: (B, n_act_tokens, dim) — 动作特征
        """
        B = vis_feat.size(0)

        # 视觉流: query tokens + 输入特征
        vis_queries = self.vis_queries.expand(B, -1, -1)
        vis_input = self.vis_input_proj(vis_feat.unsqueeze(1))          # (B, 1, dim)
        vis_tokens = torch.cat([vis_input, vis_queries[:, 1:]], dim=1)  # (B, num_vis_tokens, dim)

        # 动作流
        act_tokens = self.act_input_proj(act_tokens)                     # (B, n_act, dim)

        # MM-DiT blocks
        for block in self.blocks:
            vis_tokens, act_tokens = block(vis_tokens, act_tokens)

        # 输出: 视觉预测 (取第一个 token)
        vis_pred = self.vis_output_head(vis_tokens[:, 0, :])            # (B, dim)

        # 输出: 动作预测
        act_out = self.act_output_head(act_tokens)                       # (B, n_act, dim)

        return vis_pred, act_out


# ============================================================
# 三、LDA-1B 的三种训练模式
# ============================================================
class LDA1BModel(nn.Module):
    """
    LDA-1B 完整模型: 轻量视觉编码器 + MM-DiT + 三任务训练。
    
    三种训练角色（全域数据利用的核心）:
      1. 动作预测 (Policy Learning): 高质量遥操作数据 → (obs, act) 对
      2. 逆动力学 (Inverse Dynamics): 低质量数据 → (obs, next_obs) 对，
         学习"什么样的动作连接了这两个状态"
      3. 视觉预测 (Visual Forecasting): 无动作标签人类视频 →
         用光流推导的"假想动作"驱动视觉预测，
         学习"动作会导致什么视觉变化"
    
    所有数据都用上了——没有数据被丢弃。
    """
    def __init__(self, img_size=224, latent_dim=768, action_dim=7,
                 mmdit_dim=512, mmdit_depth=8, num_heads=8):
        super().__init__()
        self.action_dim = action_dim

        self.visual_encoder = LightweightVisualEncoder(
            in_channels=3, base_channels=32, latent_dim=latent_dim,
        )
        self.vis_proj_to_mmdit = nn.Linear(latent_dim, mmdit_dim)

        self.mmdit = MMDiT(
            dim=mmdit_dim, depth=mmdit_depth, num_heads=num_heads,
        )

        # 动作投影（将原始动作投影到 MM-DiT 维度）
        self.action_embed = nn.Linear(action_dim, mmdit_dim)

        # 输出头
        self.action_head = nn.Linear(mmdit_dim, action_dim)      # 动作预测
        self.inv_dyn_head = nn.Linear(mmdit_dim, action_dim)     # 逆动力学

    def encode_visual(self, images):
        """编码 RGB 图像为 DINO 语义特征。"""
        return self.visual_encoder(images)  # (B, latent_dim)

    def forward_policy(self, images, actions):
        """
        模式 1: 动作预测 (Policy Learning)
        用于高质量遥操作数据。
        """
        B = images.size(0)
        device = images.device

        # 编码视觉
        vis_feat = self.visual_encoder(images)                    # (B, latent_dim)
        vis_feat = self.vis_proj_to_mmdit(vis_feat)              # (B, mmdit_dim)

        # 动作 token（训练时用真实动作作为条件）
        act_embed = self.action_embed(actions).unsqueeze(1)      # (B, 1, mmdit_dim)

        # MM-DiT
        vis_pred, act_out = self.mmdit(vis_feat, act_embed)

        # 动作预测损失
        action_pred = self.action_head(act_out.squeeze(1))       # (B, action_dim)
        policy_loss = F.mse_loss(action_pred, actions)

        # 视觉预测损失（辅助）
        vis_loss = F.mse_loss(vis_pred, vis_feat)

        return policy_loss + 0.1 * vis_loss, {
            'policy_loss': policy_loss.item(),
            'vis_loss': vis_loss.item(),
        }

    def forward_inverse_dynamics(self, images, next_images):
        """
        模式 2: 逆动力学 (Inverse Dynamics)
        用于低质量 / 噪声机器人数据。
        
        学习: 给定 (obs, next_obs)，预测中间的动作。
        为什么有用？低质量数据的动作标签不精确，但状态转移是可靠的。
        训练逆动力学可以让模型理解"物理世界中的运动规律"，
        而这些知识会迁移到动作预测任务中。
        """
        B = images.size(0)

        vis_feat = self.visual_encoder(images)                    # (B, latent_dim)
        next_feat = self.visual_encoder(next_images)              # (B, latent_dim)

        vis_feat = self.vis_proj_to_mmdit(vis_feat)              # (B, mmdit_dim)
        next_feat = self.vis_proj_to_mmdit(next_feat)            # (B, mmdit_dim)

        # 用可学习的 query 代替动作 token
        query = self.action_embed(torch.zeros(B, self.action_dim,
                                              device=images.device)).unsqueeze(1)

        # MM-DiT: 输入 = 当前帧，期望输出 ≈ next_feat + 中间动作
        # 这里简化: 用 MM-DiT 同时输出 vis_pred 和 action_pred
        vis_pred, act_out = self.mmdit(vis_feat, query)

        # 逆动力学: 预测动作
        inv_action = self.inv_dyn_head(act_out.squeeze(1))

        # 视觉预测损失（主要）: 鼓励模型理解状态转移
        vis_forecast_loss = F.mse_loss(vis_pred, next_feat)

        return vis_forecast_loss, {
            'vis_forecast_loss': vis_forecast_loss.item(),
        }

    def forward_visual_forecasting(self, images, pseudo_actions):
        """
        模式 3: 视觉预测 (Visual Forecasting)
        用于无动作标签的人类视频。
        
        pseudo_actions: 通过光流推导的"假想"动作（如从光流场中提取的 2D 位移）。
        模型学习"给定当前帧和假想动作，预测下一帧的 DINO 特征"。
        
        为什么有用？人类视频包含丰富的物体交互和长周期任务知识，
        即使没有精确的动作标签，视觉预测也能让模型学到"物体如何运动"。
        """
        B = images.size(0)

        vis_feat = self.visual_encoder(images)                    # (B, latent_dim)
        vis_feat = self.vis_proj_to_mmdit(vis_feat)              # (B, mmdit_dim)

        act_embed = self.action_embed(pseudo_actions).unsqueeze(1)

        vis_pred, _ = self.mmdit(vis_feat, act_embed)

        # 视觉预测损失
        # 注意: 实际训练中 next_images 的真值也需要
        # 这里简化表示损失结构
        forecast_loss = torch.tensor(0.0, device=images.device)  # 占位

        return forecast_loss, {
            'forecast_loss': forecast_loss.item(),
        }


# ============================================================
# 四、知识蒸馏 — 大模型指导小模型
# ============================================================
def distillation_loss(student_output, teacher_output, temperature=3.0):
    """
    知识蒸馏损失: 让大模型（teacher）的"软标签"指导小模型（student）。
    
    为什么用蒸馏？
      LDA-1B 只有 1.6B 参数，但可以通过蒸馏从更大的 WAM
      （如 FastWAM ~5B）获取知识。蒸馏的核心是匹配 teacher 的
      输出分布（不仅仅是 hard label），这样 student 可以学到
      teacher 的"推理过程"而不仅是"答案"。
    
    temperature 控制软化的程度:
      - 大 temperature → 更软的分布 → 更多"次要知识"被传递
      - 小 temperature → 接近 hard label → 只保留最确定的知识
    """
    soft_student = F.log_softmax(student_output / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_output / temperature, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)


# ============================================================
# 五、__main__ 演示
# ============================================================
if __name__ == "__main__":
    B = 4
    latent_dim = 768
    action_dim = 7

    model = LDA1BModel(
        img_size=224, latent_dim=latent_dim, action_dim=action_dim,
        mmdit_dim=512, mmdit_depth=4, num_heads=8,
    )
    # 注意: 完整 1.6B 模型需要 mmdit_depth=24+ 和更宽的维度
    # 这里用浅层配置做教学演示

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # -- 模式 1: 动作预测 --
    images = torch.randn(B, 3, 224, 224)
    actions = torch.randn(B, action_dim)
    loss, logs = model.forward_policy(images, actions)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[动作预测] 总损失: {loss.item():.4f} | "
          f"策略损失: {logs['policy_loss']:.4f} | "
          f"视觉损失: {logs['vis_loss']:.4f}")

    # -- 模式 2: 逆动力学 --
    next_images = torch.randn(B, 3, 224, 224)
    loss_id, logs_id = model.forward_inverse_dynamics(images, next_images)
    optimizer.zero_grad()
    loss_id.backward()
    optimizer.step()
    print(f"[逆动力学] 视觉预测损失: {loss_id.item():.4f}")

    # -- 模式 3: 视觉预测 --
    pseudo_actions = torch.randn(B, action_dim)
    loss_vf, logs_vf = model.forward_visual_forecasting(images, pseudo_actions)
    print(f"[视觉预测] 预测损失: {loss_vf.item():.4f}")

    # -- 知识蒸馏演示 --
    teacher_logits = torch.randn(B, action_dim)
    student_logits = torch.randn(B, action_dim)
    distill_loss = distillation_loss(student_logits, teacher_logits, temperature=3.0)
    print(f"[知识蒸馏] 蒸馏损失: {distill_loss.item():.4f}")
```
