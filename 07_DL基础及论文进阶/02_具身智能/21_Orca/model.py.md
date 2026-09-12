---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Orca: The World is in Your Mind — 核心组件实现

> 本文档包含 Orca 核心组件的 PyTorch 教学实现：世界隐空间编码器、无意识/有意识学习损失、DiT Action Expert。完整预训练代码需参考 [Orca 官方仓库](https://orca-wm.github.io)。

```python
"""
Orca: The World is in Your Mind (BAAI, 2026)
==============================================
核心贡献: Next-State-Prediction 范式——先学习统一世界隐空间，再按需读取出文本/图像/动作。
代码结构:
  1. WorldLatentEncoder - 从视频帧中提取世界隐空间表征
  2. UnconsciousLoss - 无意识学习：预测相邻帧 latent
  3. ConsciousLoss - 有意识学习：事件条件下的状态转移
  4. ActionExpert - DiT-based 动作生成头（Flow Matching）
  5. OrcaForAction - 冻结 Encoder + Action Expert 的完整动作生成模型

与相关工作的关系:
  - [[../08_pi0/π0|π0]] 的 Action Expert 同样使用 DiT + Flow Matching
  - [[../14_Motus/Motus|Motus]] 的 WAM 范式与 Orca 互补——Motus 直接建模动作空间
  - [[../07_OpenVLA/OpenVLA|OpenVLA]] 和 [[../12_FLOWER/FLOWER|FLOWER]] 验证了冻结骨干 + 轻量适配
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# 1. World Latent Encoder
# ============================================
class WorldLatentEncoder(nn.Module):
    """
    Orca 的 Encoder：从视觉信号中提取世界隐空间表征。

    注意：实际 Orca 使用 Qwen3.5 VLM 作为骨干，这里简化为 Vision Transformer +
    可学习的 latent queries 来演示核心机制。

    三个输出：
    - obs_latent: 用于无意识学习（observation-only state transition）
    - evt_latent: 用于有意识学习（event-conditioned state transition）
    - vqa_latent: 用于 VQA response generation
    """

    def __init__(
        self,
        vision_dim: int = 1024,       # 视觉编码器输出维度
        latent_dim: int = 512,        # 世界隐空间维度
        num_obs_queries: int = 64,    # 无意识学习 query 数量
        num_evt_queries: int = 64,    # 有意识学习 query 数量
        num_vqa_queries: int = 32,    # VQA query 数量
        num_layers: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 可学习的 query tokens（每种学习范式各一组）
        self.obs_queries = nn.Parameter(torch.randn(1, num_obs_queries, latent_dim))
        self.evt_queries = nn.Parameter(torch.randn(1, num_evt_queries, latent_dim))
        self.vqa_queries = nn.Parameter(torch.randn(1, num_vqa_queries, latent_dim))

        # 视觉特征投影
        self.vision_proj = nn.Linear(vision_dim, latent_dim)

        # 语言特征投影（用于有意识学习和 VQA）
        self.text_proj = nn.Linear(latent_dim, latent_dim)

        # 共享 Transformer 层（cross-attention 到视觉特征）
        self.layers = nn.ModuleList([
            TransformerLayer(latent_dim, num_heads) for _ in range(num_layers)
        ])

        # 输出投影（latent → 预测的下一帧 latent）
        self.obs_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.evt_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        vision_features: torch.Tensor,    # [B, N_patches, vision_dim]
        text_features: torch.Tensor = None,  # [B, text_len, latent_dim] for conscious
    ):
        B = vision_features.shape[0]

        # 投影视觉特征
        vis = self.vision_proj(vision_features)  # [B, N, latent_dim]

        # 处理无意识学习 queries：预测相邻帧 latent
        obs_q = self.obs_queries.expand(B, -1, -1)  # [B, Q_obs, D]
        for layer in self.layers:
            obs_q = layer(obs_q, vis)
        obs_latent = obs_q.mean(dim=1)  # [B, D] — 全局池化
        obs_pred = self.obs_head(obs_latent)  # [B, D]

        # 处理有意识学习 queries：事件条件下的状态转移
        evt_q = self.evt_queries.expand(B, -1, -1)
        if text_features is not None:
            evt_q = evt_q + self.text_proj(text_features.mean(dim=1, keepdim=True))
        for layer in self.layers:
            evt_q = layer(evt_q, vis)
        evt_latent = evt_q.mean(dim=1)
        evt_pred = self.evt_head(evt_latent)

        # VQA queries（简化：复用 evt 逻辑，实际 Orca 用 LM head）
        vqa_q = self.vqa_queries.expand(B, -1, -1)
        if text_features is not None:
            vqa_q = vqa_q + self.text_proj(text_features.mean(dim=1, keepdim=True))
        for layer in self.layers:
            vqa_q = layer(vqa_q, vis)
        vqa_latent = vqa_q.mean(dim=1)

        return {
            "obs_pred": obs_pred,       # 预测的下一帧 latent
            "evt_pred": evt_pred,       # 预测的目标事件帧 latent
            "vqa_latent": vqa_latent,   # VQA 隐空间（接 LM head）
            "world_latent": evt_latent, # 主世界隐空间（用于下游 Decoder）
        }


# ============================================
# 2. Transformer Layer (简化的 Cross-Attention)
# ============================================
class TransformerLayer(nn.Module):
    """单层 Cross-Attention Transformer block"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor):
        # Cross-attention: queries attend to visual context
        x = self.norm1(queries)
        x = self.cross_attn(x, context, context)[0] + queries
        x = self.ffn(self.norm2(x)) + x
        return x


# ============================================
# 3. 无意识学习 Loss（Unconscious Learning）
# ============================================
class UnconsciousLoss(nn.Module):
    """
    无意识学习：预测相邻帧的 latent 表征。

    L_obs = MSE(obs_pred, target_latent)

    其中 target_latent 由冻结的 vision encoder 提取，
    obs_pred 由 Encoder 的 obs_head 从当前帧预测。

    关键设计（Orca Section 3.1.1）：
    - 不需要任何标签——纯自监督
    - 监督在 latent 空间，而非像素空间
    - 学习自然的物理动态（物体运动、场景变化）
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        obs_pred: torch.Tensor,       # [B, D] 预测的下一帧 latent
        target_latent: torch.Tensor,   # [B, D] 真实的下一帧 latent（冻结 encoder 提取）
    ):
        return F.mse_loss(obs_pred, target_latent)


# ============================================
# 4. 有意识学习 Loss（Conscious Learning）
# ============================================
class ConsciousLoss(nn.Module):
    """
    有意识学习：事件条件下的状态转移 + VQA。

    L = λ_evt * L_evt + λ_vqa * L_vqa

    L_evt: 在事件描述 e_{t+Δ} 的条件下，预测目标事件帧的 latent
    L_vqa: 给定视频和问题，生成答案（标准 next-token prediction）

    关键设计（Orca Section 3.1.1）：
    - L_evt 在 latent 空间做 teacher forcing
    - L_vqa 用标准 LM loss，保持语义理解能力
    """

    def __init__(
        self,
        lambda_evt: float = 1.0,
        lambda_vqa: float = 0.5,
    ):
        super().__init__()
        self.lambda_evt = lambda_evt
        self.lambda_vqa = lambda_vqa

    def forward(
        self,
        evt_pred: torch.Tensor,        # [B, D] 预测的目标事件帧 latent
        target_evt_latent: torch.Tensor, # [B, D] 真实的目标事件帧 latent
        vqa_logits: torch.Tensor = None,  # [B, seq_len, vocab_size]
        vqa_targets: torch.Tensor = None,  # [B, seq_len]
    ):
        # Event-conditioned state transition loss
        loss_evt = F.mse_loss(evt_pred, target_evt_latent)

        # VQA loss（标准 cross-entropy）
        loss_vqa = torch.tensor(0.0, device=evt_pred.device)
        if vqa_logits is not None and vqa_targets is not None:
            loss_vqa = F.cross_entropy(
                vqa_logits.view(-1, vqa_logits.size(-1)),
                vqa_targets.view(-1),
                ignore_index=-100,
            )

        total = self.lambda_evt * loss_evt + self.lambda_vqa * loss_vqa
        return total, {"loss_evt": loss_evt.item(), "loss_vqa": loss_vqa.item()}


# ============================================
# 5. DiT-based Action Expert（动作生成）
# ============================================
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入（DiT 标准做法）"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: [B] — 扩散时间步（0~1 for flow matching）
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DiTBlock(nn.Module):
    """DiT 基本块：Self-Attention + Cross-Attention（condition）+ FFN"""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        # AdaLN: 条件（时间嵌入 + 世界隐空间）通过 scale + shift 调制
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),  # scale_1, shift_1, scale_2, shift_2, scale_3, shift_3
        )

    def forward(
        self,
        x: torch.Tensor,         # [B, N_action, D]
        condition: torch.Tensor,  # [B, D] — 时间嵌入 + 世界隐空间 + 本体感觉
    ):
        # AdaLN 调制参数
        params = self.adaln(condition)  # [B, 6*D]
        scale1, shift1, scale2, shift2, scale3, shift3 = params.chunk(6, dim=-1)

        # Self-attention with AdaLN
        x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = self.self_attn(x, x, x)[0] + x

        # Cross-attention (condition as context) with AdaLN
        x = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        cond = condition.unsqueeze(1)  # [B, 1, D]
        x = self.cross_attn(x, cond, cond)[0] + x

        # FFN with AdaLN
        x = self.norm3(x) * (1 + scale3.unsqueeze(1)) + shift3.unsqueeze(1)
        x = self.ffn(x) + x

        return x


class ActionExpert(nn.Module):
    """
    Orca 的动作生成头：DiT-based，Flow Matching。

    输入：
    - world_latent: [B, D] 世界隐空间
    - proprio: [B, proprio_dim] 机器人本体感觉
    - noisy_action: [B, chunk_len, action_dim] 噪声动作

    输出：
    - velocity: [B, chunk_len, action_dim] 速度场（Flow Matching）

    架构（Orca Section 3.2.3）：
    - 冻结的 Encoder → world_latent → MLP adaptor → condition
    - condition = time_emb + adapted_latent + proprio_emb
    - 多层 DiTBlock → 最终线性投影 → velocity
    - Flow matching loss: v_θ ≈ target_action - noise
    """

    def __init__(
        self,
        latent_dim: int = 512,
        proprio_dim: int = 16,       # 关节角度、末端位置等
        action_dim: int = 7,         # Δx, Δy, Δz, Δθx, Δθy, Δθz, gripper
        chunk_len: int = 16,         # 预测 16 步动作
        num_layers: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.chunk_len = chunk_len

        # MLP adaptor：将世界隐空间映射到 Action Expert 的条件空间
        self.latent_adaptor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # 时间嵌入
        self.time_emb = SinusoidalTimeEmbedding(latent_dim)
        self.time_proj = nn.Linear(latent_dim, latent_dim)

        # 本体感觉嵌入
        self.proprio_proj = nn.Linear(proprio_dim, latent_dim)

        # 动作嵌入（将噪声动作投影到 DiT 空间）
        self.action_proj = nn.Linear(action_dim, latent_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, num_heads) for _ in range(num_layers)
        ])

        # 输出投影：latent → action velocity
        self.output_proj = nn.Linear(latent_dim, action_dim)

    def forward(
        self,
        world_latent: torch.Tensor,   # [B, D] — 来自冻结 Encoder
        proprio: torch.Tensor,         # [B, proprio_dim]
        noisy_action: torch.Tensor,    # [B, chunk_len, action_dim]
        t: torch.Tensor,               # [B] — 时间步 [0, 1]
    ):
        B = world_latent.shape[0]

        # 构建条件
        latent_cond = self.latent_adaptor(world_latent)       # [B, D]
        time_cond = self.time_proj(self.time_emb(t))          # [B, D]
        proprio_cond = self.proprio_proj(proprio)             # [B, D]
        condition = latent_cond + time_cond + proprio_cond     # [B, D]

        # 投影噪声动作
        x = self.action_proj(noisy_action)  # [B, chunk_len, D]

        # DiT blocks
        for block in self.blocks:
            x = block(x, condition)

        # 输出速度场
        velocity = self.output_proj(x)  # [B, chunk_len, action_dim]
        return velocity


# ============================================
# 6. Flow Matching Loss
# ============================================
class FlowMatchingLoss(nn.Module):
    """
    Flow Matching 损失（Orca 使用与 π0 相同的公式）。

    给定 target action a_1 和高斯噪声 a_0 = N(0, 1)，
    在时间 t 构造插值：a_t = (1-t) * a_0 + t * a_1
    速度场目标：v = a_1 - a_0

    L = MSE(v_θ(a_t, t, c), a_1 - a_0)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        velocity_pred: torch.Tensor,  # [B, chunk_len, action_dim]
        target_action: torch.Tensor,  # [B, chunk_len, action_dim]
        noise: torch.Tensor,           # [B, chunk_len, action_dim]
        t: torch.Tensor,               # [B]
    ):
        # 构造插值样本
        t = t.view(-1, 1, 1)  # [B, 1, 1]
        a_t = (1 - t) * noise + t * target_action

        # 速度场目标：直线路径的恒定速度
        target_velocity = target_action - noise

        return F.mse_loss(velocity_pred, target_velocity)


# ============================================
# 7. OrcaForAction: 冻结 Encoder + Action Expert
# ============================================
class OrcaForAction(nn.Module):
    """
    Orca 动作生成完整模型。

    与论文 Section 3.2.3 完全对应：
    - Encoder 冻结（只提取 world_latent）
    - Action Expert（MLP adaptor + DiT）从头训练
    - 使用 Flow Matching loss

    推理时：从 N(0,1) 采样噪声，用 Action Expert 做 multi-step denoising
    """

    def __init__(
        self,
        encoder: WorldLatentEncoder,
        action_expert: ActionExpert,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_expert = action_expert
        self.num_inference_steps = num_inference_steps

        # 冻结 Encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        vision_features: torch.Tensor,
        proprio: torch.Tensor,
        target_action: torch.Tensor = None,
    ):
        """
        Training forward pass.
        """
        # 提取世界隐空间
        enc_outputs = self.encoder(vision_features)
        world_latent = enc_outputs["world_latent"]

        if target_action is not None:
            B, chunk_len, action_dim = target_action.shape
            # 采样噪声和时间步
            noise = torch.randn_like(target_action)
            t = torch.rand(B, device=target_action.device)

            # 构造插值样本
            t_expanded = t.view(-1, 1, 1)
            noisy_action = (1 - t_expanded) * noise + t_expanded * target_action

            # 预测速度场
            velocity_pred = self.action_expert(
                world_latent, proprio, noisy_action, t
            )

            return velocity_pred, noise, target_action, t
        else:
            return world_latent

    @torch.no_grad()
    def generate_action(
        self,
        vision_features: torch.Tensor,
        proprio: torch.Tensor,
    ):
        """
        推理：multi-step denoising 生成动作。

        Orca 使用 Flow Matching，所以使用简单的 Euler 积分：
        a_{t+Δt} = a_t + v_θ(a_t, t, c) * Δt

        π0 使用更高级的 DPM-Solver，这里用 Euler 做教学演示。
        """
        B = vision_features.shape[0]
        chunk_len = self.action_expert.chunk_len
        action_dim = self.action_expert.action_dim

        # 提取世界隐空间
        enc_outputs = self.encoder(vision_features)
        world_latent = enc_outputs["world_latent"]

        # 初始化噪声
        action = torch.randn(B, chunk_len, action_dim, device=vision_features.device)

        # Multi-step denoising
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((B,), step * dt, device=vision_features.device)
            velocity = self.action_expert(world_latent, proprio, action, t)
            action = action + velocity * dt

        return action


# ============================================
# 8. 完整预训练 Loss（Orca Equation 2）
# ============================================
class OrcaPreTrainingLoss(nn.Module):
    """
    Orca 预训练总损失。

    L = λ_obs * L_obs + λ_evt * L_evt + λ_vqa * L_vqa

    对应论文 Equation 2。
    """

    def __init__(
        self,
        lambda_obs: float = 1.0,
        lambda_evt: float = 1.0,
        lambda_vqa: float = 0.5,
    ):
        super().__init__()
        self.lambda_obs = lambda_obs
        self.lambda_evt = lambda_evt
        self.lambda_vqa = lambda_vqa
        self.unconscious_loss = UnconsciousLoss()
        self.conscious_loss = ConsciousLoss(lambda_evt, lambda_vqa)

    def forward(
        self,
        enc_outputs: dict,
        target_obs_latent: torch.Tensor,
        target_evt_latent: torch.Tensor,
        vqa_logits: torch.Tensor = None,
        vqa_targets: torch.Tensor = None,
    ):
        # 无意识学习 loss
        loss_obs = self.unconscious_loss(
            enc_outputs["obs_pred"], target_obs_latent
        )

        # 有意识学习 loss
        loss_cons, loss_dict = self.conscious_loss(
            enc_outputs["evt_pred"],
            target_evt_latent,
            vqa_logits,
            vqa_targets,
        )

        total = self.lambda_obs * loss_obs + loss_cons
        return total, {
            "loss_obs": loss_obs.item(),
            "loss_evt": loss_dict["loss_evt"],
            "loss_vqa": loss_dict["loss_vqa"],
            "total": total.item(),
        }


# ============================================
# 9. 消融实验配置（Orca Section 4.3）
# ============================================
def get_ablation_config(ablation: str):
    """
    返回不同消融实验的 λ 配置。

    Orca Table 5 的消融实验：
    - "obs_only":      仅无意识学习（只有 L_obs）
    - "obs_evt":       无意识 + 事件条件（L_obs + L_evt）
    - "obs_vqa":       无意识 + VQA（L_obs + L_vqa）
    - "evt_vqa":       有意识（L_evt + L_vqa）
    - "full":          全部三个（L_obs + L_evt + L_vqa）
    """
    configs = {
        "obs_only": {"lambda_obs": 1.0, "lambda_evt": 0.0, "lambda_vqa": 0.0},
        "obs_evt":  {"lambda_obs": 1.0, "lambda_evt": 1.0, "lambda_vqa": 0.0},
        "obs_vqa":  {"lambda_obs": 1.0, "lambda_evt": 0.0, "lambda_vqa": 0.5},
        "evt_vqa":  {"lambda_obs": 0.0, "lambda_evt": 1.0, "lambda_vqa": 0.5},
        "full":     {"lambda_obs": 1.0, "lambda_evt": 1.0, "lambda_vqa": 0.5},
    }
    return configs.get(ablation, configs["full"])


# ============================================
# 10. 训练吞吐优化要点（Orca Section 3.3 / Appendix D）
# ============================================
# Orca 使用自研 FlagScale（FSDP2），通过以下优化实现 4.4× 加速：
#
# 1. Chunked Cross-Entropy Loss:
#    避免在 loss 计算时 materialize 完整 logits，分块计算
#    → 大幅降低峰值显存
#
# 2. Activation Recomputation:
#    对计算重但显存轻的层（Attention、FFN）不存中间激活
#    → 反向时重新计算，用 33% 额外计算换 5× 显存节省
#
# 3. Forward/Backward Pre-fetching:
#    将 FSDP all-gather 通信与计算重叠
#    → 前向：预取下一层参数 | 反向：预取上一层梯度
#
# 4. Visual Block Sharding Optimization:
#    移除视觉模块不必要的 FSDP sharding
#    → 减少通信开销
#
# 结果：吞吐从 0.66 → 2.91 Samples/Sec/GPU
```
