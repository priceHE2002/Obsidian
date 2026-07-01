---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# GR00T N1 模型实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
GR00T N1 模型实现
=================
DiT action head + Flow Matching + 多具身条件化

GR00T N1 是 NVIDIA 的 VLA 旗舰模型，架构与 pi0 类似（双系统 VLM+DiT），
但专为人形机器人设计，支持多具身条件化（EmbodimentTag），
通过 Flow Matching 输出连续关节动作。Apache 2.0 完全开源。

参考: [[GR00T N1]] | [[GR00T N1 原文.pdf]]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# 多具身条件化
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EmbodimentConfig:
    """
    具身形态配置。

    WHY: 人形机器人（20+ 关节）和机械臂（7 DoF）的动作空间完全不同。
    通过 EmbodimentTag 机制，模型知道自己控制的机器人形态，
    从而激活相应的 action head 和 AdaLN 参数。
    """
    name: str                 # 形态名称标识
    action_dim: int           # 动作空间维度
    action_mode: str          # "joint" | "eef_delta" | "eef_absolute"
    joint_names: List[str]    # 关节名称列表
    control_freq: float       # 控制频率 (Hz)
    proprio_dim: int          # 本体感觉维度

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


# 预定义的具身形态
EMBODIMENT_REGISTRY = {
    "GR1": EmbodimentConfig(
        name="GR1", action_dim=40, action_mode="joint",
        joint_names=[f"joint_{i}" for i in range(40)],
        control_freq=50.0, proprio_dim=80,
    ),
    "OXE_DROID": EmbodimentConfig(
        name="OXE_DROID", action_dim=7, action_mode="eef_delta",
        joint_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        control_freq=15.0, proprio_dim=14,
    ),
    "AGIBOT_GENIE1": EmbodimentConfig(
        name="AGIBOT_GENIE1", action_dim=52, action_mode="joint",
        joint_names=[f"joint_{i}" for i in range(52)],
        control_freq=50.0, proprio_dim=104,
    ),
    "FRANKA_PANDA": EmbodimentConfig(
        name="FRANKA_PANDA", action_dim=7, action_mode="eef_delta",
        joint_names=["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        control_freq=20.0, proprio_dim=14,
    ),
}


class EmbodimentConditioning(nn.Module):
    """
    多具身条件化模块。

    WHY: 不同机器人的动作空间不同（维度、控制模式），需要让
    AdaLN 根据当前具身形态选择对应的 scale & shift 参数。
    所有具身共享主干，只有 AdaLN 参数是形态特定的。
    """

    def __init__(
        self,
        hidden_size: int,
        num_embodiments: int = 4,
        num_layers: int = 12,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # WHY: 每种具身形态有独立的 AdaLN 参数
        self.ada_ln_params = nn.ParameterDict({
            f"emb_{i}": nn.Parameter(torch.zeros(num_layers, 6, hidden_size))
            for i in range(num_embodiments)
        })

        self.embodiment_embed = nn.Embedding(num_embodiments, hidden_size)

    def forward(self, embodiment_ids: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, ...]:
        """返回当前层的 AdaLN scale & shift 参数，根据具身形态选择。"""
        B = embodiment_ids.shape[0]
        params = torch.stack([
            self.ada_ln_params[f"emb_{eid.item()}"][layer_idx]
            for eid in embodiment_ids
        ])
        return tuple(params[:, i, :] for i in range(6))


# ═══════════════════════════════════════════════════════════════════
# DiT Block with Multi-Embodiment AdaLN
# ═══════════════════════════════════════════════════════════════════

class GR00TDiTBlock(nn.Module):
    """
    GR00T N1 的 DiT Block：Self-Attention + Cross-Attention to VLM + MLP。

    WHY AdaLN 用 EmbodimentConditioning：
    统一主干 + 每个具身独立 AdaLN 参数 = 共享特征提取 + 形态特定调制。
    比独立 action head 更参数高效（省 ~20%）。
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self._ada_ln: Optional[Tuple[torch.Tensor, ...]] = None

    def set_ada_ln_params(self, params: Tuple[torch.Tensor, ...]):
        self._ada_ln = params

    def forward(
        self,
        x: torch.Tensor,
        vlm_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca = self._ada_ln

        # Self-Attn
        x_normed = self.norm1(x) * (1 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # Cross-Attn to VLM
        if vlm_embeddings is not None:
            x_normed = self.norm2(x) * (1 + scale_ca.unsqueeze(1)) + shift_ca.unsqueeze(1)
            cross_out, _ = self.cross_attn(x_normed, vlm_embeddings, vlm_embeddings)
            x = x + cross_out

        # MLP
        x_normed = self.norm3(x) * (1 + gate_ca.unsqueeze(1)) + gate_ca.unsqueeze(1)
        mlp_out = self.mlp(x_normed)
        x = x + gate_ca.unsqueeze(1) * mlp_out
        return x


# ═══════════════════════════════════════════════════════════════════
# Rectified Flow
# ═══════════════════════════════════════════════════════════════════

class RectifiedFlowScheduler:
    """
    Rectified Flow 调度器。

    WHY Rectified Flow 而非 DDPM？
    - 确定性直线路径 -> 更少去噪步骤
    - 单臂 4 步，双臂 8 步
    - 与 NVIDIA 生态（TensorRT、Jetson）配合更好
    """

    def __init__(self, num_inference_steps: int = 8):
        self.num_inference_steps = num_inference_steps

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    def add_noise(
        self, x_1: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_1)
        t = t.view(-1, *([1] * (x_1.dim() - 1)))
        return (1.0 - t) * noise + t * x_1

    def flow_loss(self, v_pred: torch.Tensor, x_1: torch.Tensor, x_0: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(v_pred, x_1 - x_0)

    @torch.no_grad()
    def sample_euler(
        self, model_fn, noise: torch.Tensor, context: Dict
    ) -> torch.Tensor:
        x = noise
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device)
            v = model_fn(x, t, context)
            x = x + v * dt
        return x


# ═══════════════════════════════════════════════════════════════════
# GR00T N1 完整模型
# ═══════════════════════════════════════════════════════════════════

class GR00TN1(nn.Module):
    """
    GR00T N1: NVIDIA 双系统 VLA（System 2 VLM + System 1 DiT）。

    完整流程：
    1. System 2 (VLM, Cosmos-Reason-2B): 图像+语言 -> 语义 embedding（~1-5Hz）
    2. System 1 (DiT): VLM embedding + 本体感觉 -> Rectified Flow 生成关节动作（~22-50Hz）
    3. EmbodimentTag 选择对应 AdaLN 参数和 action head

    WHY 双系统？
    - VLM 太大 -> 不能每步都跑（低频即可）
    - 高频控制需要小网络 -> DiT Action Expert
    - 专为人形机器人优化 -> 双手+移动+多关节协调
    """

    def __init__(
        self,
        vlm_hidden_size: int = 2048,
        dit_hidden_size: int = 1024,
        num_dit_layers: int = 32,
        num_dit_heads: int = 16,
        num_action_chunks: int = 50,
        num_inference_steps: int = 8,
        max_embodiments: int = 4,
    ):
        super().__init__()
        self.vlm_hidden_size = vlm_hidden_size
        self.dit_hidden_size = dit_hidden_size
        self.num_action_chunks = num_action_chunks

        # System 2: VLM 编码器
        self.vlm = nn.Sequential(
            nn.Linear(1152, vlm_hidden_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=vlm_hidden_size, nhead=16,
                    dim_feedforward=vlm_hidden_size * 4,
                    batch_first=True,
                ),
                num_layers=24,
            ),
        )

        self.vlm_to_dit = nn.Linear(vlm_hidden_size, dit_hidden_size) \
            if vlm_hidden_size != dit_hidden_size else nn.Identity()

        # Embodiment Conditioning
        self.emb_conditioning = EmbodimentConditioning(
            dit_hidden_size, num_embodiments=max_embodiments,
            num_layers=num_dit_layers,
        )

        # 本体感觉投影
        self.prop_proj = nn.Linear(128, dit_hidden_size)

        # 时间嵌入
        time_dim = dit_hidden_size * 4
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(dit_hidden_size),
            nn.Linear(dit_hidden_size, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, dit_hidden_size),
        )

        # System 1: DiT blocks
        self.dit_blocks = nn.ModuleList([
            GR00TDiTBlock(dit_hidden_size, num_dit_heads)
            for _ in range(num_dit_layers)
        ])

        # 动作 embedding
        self.action_embed = nn.Linear(64, dit_hidden_size)  # 最大 action_dim
        self.step_embed = nn.Embedding(num_action_chunks, dit_hidden_size)

        # WHY: 每个机器人 action_dim 不同，需要独立的输出头
        self.action_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(dit_hidden_size),
                nn.Linear(dit_hidden_size, config.action_dim),
            )
            for name, config in EMBODIMENT_REGISTRY.items()
        })

        self.final_norm = nn.LayerNorm(dit_hidden_size)
        self.scheduler = RectifiedFlowScheduler(num_inference_steps=num_inference_steps)

    def forward(
        self,
        vision_features: torch.Tensor,       # [B, T, vision_dim]
        text_ids: torch.Tensor,              # [B, text_len]
        proprio: torch.Tensor,               # [B, prop_dim]
        embodiment_ids: torch.Tensor,        # [B]
        embodiment_name: str,                # 当前具身形态名称
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = vision_features.shape[0]
        device = vision_features.device

        # System 2: VLM 编码
        vlm_output = self.vlm(vision_features)
        vlm_proj = self.vlm_to_dit(vlm_output)

        if actions is not None:
            noise = torch.randn_like(actions)
            t = self.scheduler.sample_time(B, device)
            x_t = self.scheduler.add_noise(actions, t, noise)

            x = self.action_embed(x_t) + self.step_embed(
                torch.arange(self.num_action_chunks, device=device)
            )

            t_emb = self.time_embed(t).unsqueeze(1)
            p_emb = self.prop_proj(proprio).unsqueeze(1)

            for i, block in enumerate(self.dit_blocks):
                ada_params = self.emb_conditioning(embodiment_ids, layer_idx=i)
                block.set_ada_ln_params(ada_params)
                x = block(x + t_emb + p_emb, vlm_proj)

            x = self.final_norm(x)
            v_pred = self.action_heads[embodiment_name](x)

            loss = self.scheduler.flow_loss(v_pred, actions, noise)
            return {"loss": loss, "v_pred": v_pred}

        return {"vlm_output": vlm_output}

    @torch.no_grad()
    def generate_actions(
        self,
        vision_features: torch.Tensor,
        text_ids: torch.Tensor,
        proprio: torch.Tensor,
        embodiment_ids: torch.Tensor,
        embodiment_name: str,
    ) -> torch.Tensor:
        """推理：Flow Matching 采样生成动作。"""
        B = vision_features.shape[0]
        device = vision_features.device
        action_dim = EMBODIMENT_REGISTRY[embodiment_name].action_dim

        vlm_output = self.vlm(vision_features)
        vlm_proj = self.vlm_to_dit(vlm_output)
        proprio_emb = self.prop_proj(proprio).unsqueeze(1)

        def model_fn(x_t, t, context):
            x = self.action_embed(x_t) + self.step_embed(
                torch.arange(self.num_action_chunks, device=device)
            )
            t_emb = self.time_embed(t).unsqueeze(1)
            x = x + t_emb + proprio_emb

            for i, block in enumerate(self.dit_blocks):
                ada_params = self.emb_conditioning(embodiment_ids, layer_idx=i)
                block.set_ada_ln_params(ada_params)
                x = block(x, vlm_proj)

            x = self.final_norm(x)
            return self.action_heads[embodiment_name](x)

        noise = torch.randn(B, self.num_action_chunks, action_dim, device=device)
        actions = self.scheduler.sample_euler(model_fn, noise, {})
        return actions


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ═══════════════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GR00T N1 模型演示")
    print("=" * 60)

    print(f"\n支持的具身形态: {list(EMBODIMENT_REGISTRY.keys())}")
    for name, cfg in EMBODIMENT_REGISTRY.items():
        print(f"  {name}: {cfg.action_dim}维动作, {cfg.control_freq}Hz")

    scheduler = RectifiedFlowScheduler(num_inference_steps=8)
    t = scheduler.sample_time(4, torch.device("cpu"))
    x1 = torch.randn(4, 50, 40)
    noise = torch.randn_like(x1)
    x_t = scheduler.add_noise(x1, t, noise)
    print(f"\nRectified Flow: x_t 形状 {x_t.shape}（{scheduler.num_inference_steps}步推理）")

    emb_cond = EmbodimentConditioning(hidden_size=1024, num_layers=12)
    emb_ids = torch.tensor([0, 1, 2, 3])
    params = emb_cond(emb_ids, layer_idx=0)
    print(f"AdaLN params: {len(params)} 个, 每个形状 {params[0].shape} (B, H)")

    model = GR00TN1(
        vlm_hidden_size=2048, dit_hidden_size=1024,
        num_dit_layers=8, num_dit_heads=16,
        num_action_chunks=50, num_inference_steps=8,
    )

    vision = torch.randn(2, 256, 1152)
    text = torch.randint(0, 50000, (2, 32))
    prop = torch.randn(2, 128)
    eid = torch.tensor([0, 0])
    actions = torch.randn(2, 50, 40)

    output = model(vision, text, prop, eid, "GR1", actions)
    print(f"\n训练损失: {output['loss'].item():.4f}")

    generated = model.generate_actions(vision, text, prop, eid, "GR1")
    print(f"推理动作: {generated.shape} (GR1: 50步 x 40维关节)")

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n总参数: {total_params:.0f}M")
    print("GR00T N1 演示完成")

```
