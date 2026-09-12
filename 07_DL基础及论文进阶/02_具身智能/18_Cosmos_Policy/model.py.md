---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Cosmos Policy - 代码实现

> 本文档包含 [[Cosmos Policy]] 的 PyTorch/NumPy 教学实现，涵盖视频预测世界模型、Latent Frame Injection、策略优化（Direct Policy + Model-based Planning）以及价值函数学习。

```python
"""
Cosmos Policy 教学实现 — 视频扩散模型微调为机器人策略
- Latent Frame Injection: 动作/本体感觉/未来帧编码为 latent frames
- DiT (Diffusion Transformer): 统一去噪 latent 序列
- 一个模型三种能力: Policy + World Model + Value Function
- Model-based Planning: 世界模型展开 + 价值函数选最优
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 一、Latent Frame Injection（核心创新）
# ============================================================

class LatentFrameEncoder(nn.Module):
    """将机器人专属模态编码为可插入的 latent frames
    
    WHY Latent Frame Injection: Cosmos-Predict2 是视频扩散模型，
    只懂怎么去噪"视频帧"。Latent Frame Injection 的技巧是
    把动作、本体感觉、未来帧等机器人模态也编码为"视频帧形状"——
    这样不需要任何架构修改就能复用视频扩散模型的能力。
    
    参考 [[Cosmos Policy]] Section 2.1。
    """
    def __init__(self, d_latent=256, frame_size=16):
        """
        d_latent: 每帧 latent 维度
        frame_size: latent frame 空间大小 (frame_size × frame_size)
        """
        super().__init__()
        self.frame_dim = d_latent * frame_size * frame_size
        self.frame_size = frame_size
        self.d_latent = d_latent
        
        # 动作 → latent frame 编码器
        # WHY: 将 7 维动作向量膨胀为 16×16×256 的 latent frame
        # 让视频 DiT 把它当"特殊的帧"来去噪
        self.action_to_frame = nn.Sequential(
            nn.Linear(7, 512), nn.ReLU(),
            nn.Linear(512, self.frame_dim),
            nn.Unflatten(1, (d_latent, frame_size, frame_size)),
        )
        # 本体感觉 → latent frame 编码器
        self.proprio_to_frame = nn.Sequential(
            nn.Linear(7, 512), nn.ReLU(),
            nn.Linear(512, self.frame_dim),
            nn.Unflatten(1, (d_latent, frame_size, frame_size)),
        )
        # 未来帧预测编码（可选）
        self.future_frame_encoder = nn.Sequential(
            nn.Conv2d(3, d_latent, 3, padding=1),  # 保持空间
        )
        # 价值估计 → latent frame（一个小 token）
        self.value_encoder = nn.Linear(1, self.frame_dim)
    
    def encode_action(self, action):
        """action: (B, action_dim) → (B, d_latent, H, W)"""
        return self.action_to_frame(action)
    
    def encode_proprio(self, proprio):
        return self.proprio_to_frame(proprio)
    
    def encode_future_frame(self, image):
        """image: (B, 3, H, W) → (B, d_latent, H, W)"""
        return self.future_frame_encoder(image)
    
    def encode_value(self, value):
        """value: (B, 1) → (B, d_latent*H*W)"""
        val_flat = self.value_encoder(value)
        return val_flat.view(val_flat.shape[0], self.d_latent, self.frame_size, self.frame_size)


class LatentFrameInjector:
    """将机器人 latent frames 插入到视频 latent 序列中
    
    序列结构:
    [视频帧 latent 1] [视频帧 latent 2] [...] [动作 latent] [本体感觉 latent] [未来帧 latent(可选)] [value latent(可选)]
    
    WHY 这种结构: Diffusion Transformer 本来就懂怎么去噪 latent frames——
    现在只是多了一些"特殊"的 frame。视频模型里已经有的物理知识
    （物体怎么运动、碰撞后怎么反弹）自然应用到这些特殊 frame 上。
    """
    def __init__(self, d_latent=256, frame_size=16, num_video_frames=4):
        self.d_latent = d_latent
        self.frame_size = frame_size
        self.num_video_frames = num_video_frames
    
    def inject(self, video_latents, action_latent, proprio_latent, 
               future_latent=None, value_latent=None):
        """构建完整的 latent 序列"""
        B = video_latents.shape[0]
        sequence = [video_latents]
        sequence.append(action_latent.unsqueeze(1) if action_latent.dim() == 4 else action_latent)
        sequence.append(proprio_latent.unsqueeze(1) if proprio_latent.dim() == 4 else proprio_latent)
        if future_latent is not None:
            sequence.append(future_latent.unsqueeze(1) if future_latent.dim() == 4 else future_latent)
        if value_latent is not None:
            sequence.append(value_latent.unsqueeze(1) if value_latent.dim() == 4 else value_latent)
        return torch.cat(sequence, dim=1)  # (B, total_frames, d_latent, H, W)


# ============================================================
# 二、Diffusion Transformer (DiT) 骨干
# ============================================================

class DiTBlock(nn.Module):
    """Diffusion Transformer 基本块（AdaLN + 自注意力 + FFN）
    
    WHY AdaLN (Adaptive Layer Norm): DiT 使用 AdaLN 将时间步条件
    注入每一层。与 Cosmos-Predict2 的原生架构保持一致——
    不修改架构就是用视频模型的全部能力。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # AdaLN 调制参数（由时间步 embedding 生成）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim)
        )
    
    def forward(self, x, c):
        # c: 时间步条件 embedding
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # 自注意力 + AdaLN
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_norm, x_norm, x_norm)[0]
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP + AdaLN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


class DiffusionTransformer(nn.Module):
    """DiT: 去噪 latent 序列的 Transformer
    
    与 Cosmos-Predict2 兼容的架构:
    - Patchify: 把 latent frame 切成 patches → token 序列
    - DiT blocks: 自注意力 + AdaLN 条件化
    - Unpatchify: token 序列 → latent frame
    
    参考 [[Cosmos Policy]] Section 2.3，基于 DiT 架构。
    """
    def __init__(self, d_latent=256, frame_size=16, patch_size=4, 
                 dim=512, depth=12, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_per_frame = (frame_size // patch_size) ** 2
        self.patch_dim = d_latent * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        # 可学习位置编码
        max_patches = 20 * self.num_patches_per_frame  # 最多 20 帧
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, dim) * 0.02)
        # 时间步编码
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(dim), nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)
        # 输出头
        self.output_head = nn.Linear(dim, self.patch_dim)
    
    def patchify(self, frames):
        """frames: (B, num_frames, d_latent, H, W) → (B, num_frames*num_patches, patch_dim)"""
        B, F, C, H, W = frames.shape
        p = self.patch_size
        # 切 patch
        patches = frames.reshape(B, F, C, H//p, p, W//p, p)
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6)  # (B, F, H/p, W/p, C, p, p)
        patches = patches.reshape(B, F * (H//p) * (W//p), C * p * p)
        return patches
    
    def unpatchify(self, patches, num_frames):
        """反向操作: patches → frames"""
        B = patches.shape[0]
        C, H, W = 256, 16, 16  # d_latent, frame_size, frame_size
        p = self.patch_size
        patches = patches.reshape(B, num_frames, H//p, W//p, C, p, p)
        patches = patches.permute(0, 1, 4, 2, 5, 3, 6)  # (B, F, C, H/p, p, W/p, p)
        frames = patches.reshape(B, num_frames, C, H, W)
        return frames
    
    def forward(self, frames, timestep, num_frames):
        """
        frames: (B, total_frames, d_latent, H, W) — 包含视频+动作+本体感觉 latents
        timestep: (B,) — 扩散时间步
        """
        B = frames.shape[0]
        # Patchify
        patches = self.patchify(frames)  # (B, total_patches, patch_dim)
        x = self.patch_embed(patches)
        # 位置编码
        x = x + self.pos_embed[:, :x.shape[1]]
        # 时间步条件
        c = self.time_embed(timestep)
        # DiT blocks
        for block in self.blocks:
            x = block(x, c)
        x = self.final_norm(x)
        # 预测噪声（输出和输入同形状）
        noise_pred_patches = self.output_head(x)
        noise_pred = self.unpatchify(noise_pred_patches, num_frames)
        return noise_pred  # (B, total_frames, d_latent, H, W)


# ============================================================
# 三、Cosmos Policy 主模型
# ============================================================

class CosmosPolicy(nn.Module):
    """Cosmos Policy: Fine-Tuned Video Model for Control
    
    一个模型 → 三种输出:
    1. Policy: 给定观测 → 去噪动作 latent → 解码为动作
    2. World Model: 给定观测+动作 → 去噪未来帧
    3. Value Function: 给定轨迹 → 预测累计奖励
    """
    def __init__(self, d_latent=256, frame_size=16, num_video_frames=4):
        super().__init__()
        self.num_video_frames = num_video_frames
        # Latent Frame 编码/注入
        self.frame_encoder = LatentFrameEncoder(d_latent, frame_size)
        self.injector = LatentFrameInjector(d_latent, frame_size, num_video_frames)
        # DiT 骨干
        self.dit = DiffusionTransformer(d_latent=d_latent, frame_size=frame_size)
        # 动作解码器（从 latent frame → 连续动作值）
        self.action_decoder = nn.Sequential(
            nn.Flatten(), nn.Linear(d_latent * frame_size * frame_size, 512),
            nn.ReLU(), nn.Linear(512, 7),  # 7 维动作
        )
        # 价值解码器
        self.value_decoder = nn.Linear(512, 1)
        # 噪声调度
        self.num_timesteps = 1000
        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1-alphas_cumprod).sqrt())
    
    def add_noise(self, x0, noise, t):
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise
    
    def forward_policy(self, video_latents, proprio, timestep=None, action=None):
        """策略模式: 观测 → 动作
        
        训练时传入 action（真实动作），推理时只传观测。
        """
        B = video_latents.shape[0]
        # 总是编码本体感觉
        prop_latent = self.frame_encoder.encode_proprio(proprio)
        
        if action is not None:
            # 训练——对动作 latent 加噪并预测噪声
            action_flat = action.unsqueeze(1) if action.dim() == 2 else action
            action_latent = self.frame_encoder.encode_action(action_flat)
            noise = torch.randn_like(action_latent)
            t = torch.randint(0, self.num_timesteps, (B,), device=action_latent.device)
            noisy_action = self.add_noise(action_latent, noise, t)
            
            sequence = self.injector.inject(video_latents, noisy_action, prop_latent)
            noise_pred = self.dit(sequence, t, self.num_video_frames + 2)
            return F.mse_loss(noise_pred, torch.cat([
                torch.zeros_like(video_latents), noise, torch.zeros_like(prop_latent)
            ], dim=1))
        else:
            # 推理——从纯噪声去噪生成动作
            action_latent = torch.randn(B, 1, *video_latents.shape[2:], device=video_latents.device)
            for step_i in range(self.num_timesteps - 1, -1, -1):
                t = torch.full((B,), step_i, device=video_latents.device)
                sequence = self.injector.inject(video_latents, action_latent, prop_latent)
                noise_pred = self.dit(sequence, t, self.num_video_frames + 2)
                # DDPM 去噪一步
                alpha = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t-1] if step_i > 0 else torch.tensor(1.0)
                action_latent = (action_latent - (1-alpha).sqrt() * noise_pred) / alpha.sqrt()
                if step_i > 0:
                    action_latent = action_latent + (1-alpha_prev).sqrt() * torch.randn_like(action_latent)
            
            action = self.action_decoder(action_latent.squeeze(1))
            return action
    
    def forward_world_model(self, video_latents, action, proprio):
        """世界模型模式: 观测+动作 → 未来帧
        
        WHY 有用: 在 Model-based Planning 中，
        用世界模型"想象"不同动作序列的后果。
        """
        # 编码当前观测和动作
        action_latent = self.frame_encoder.encode_action(action.unsqueeze(1))
        prop_latent = self.frame_encoder.encode_proprio(proprio)
        # 去噪生成未来帧
        B = video_latents.shape[0]
        future_latent = torch.randn(B, 1, *video_latents.shape[2:], device=video_latents.device)
        for step_i in range(self.num_timesteps - 1, -1, -1):
            t = torch.full((B,), step_i, device=video_latents.device)
            sequence = self.injector.inject(
                video_latents, action_latent, prop_latent, future_latent=future_latent
            )
            noise_pred = self.dit(sequence, t, self.num_video_frames + 3)
            future_latent = (future_latent - (1-self.alphas_cumprod[t]).sqrt() * noise_pred) / \
                            self.alphas_cumprod[t].sqrt()
        return future_latent  # 预测的未来帧
    
    def forward_value(self, video_latents, action, proprio):
        """价值函数: 估计累积奖励"""
        B = video_latents.shape[0]
        action_latent = self.frame_encoder.encode_action(action.unsqueeze(1))
        prop_latent = self.frame_encoder.encode_proprio(proprio)
        sequence = self.injector.inject(video_latents, action_latent, prop_latent)
        # 通过 DiT 编码 + 价值解码
        x = self.dit.patch_embed(self.dit.patchify(sequence))
        c = self.dit.time_embed(torch.zeros(B, device=video_latents.device).long())
        for block in self.dit.blocks:
            x = block(x, c)
        return self.value_decoder(x.mean(dim=1))


# ============================================================
# 四、Model-based Planning
# ============================================================

def model_based_planning(model, video_latents, proprio, num_candidates=32):
    """通过世界模型展开 + 价值函数选择最优动作序列
    
    WHY Planning 能提升 12.5%: Direct Policy 只输出
    一个"最优"动作——但有时会陷入次优。Planning 模式
    采样 N 个候选动作 → 世界模型展开未来 → 价值函数
    选最优 —— 不需要额外模型。
    参考 [[Cosmos Policy]] Section 2.4。
    """
    best_action = None
    best_value = float('-inf')
    
    for i in range(num_candidates):
        # 采样候选动作（加噪声产生多样性）
        with torch.no_grad():
            action = model.forward_policy(video_latents, proprio, action=None)
            noise = torch.randn_like(action) * 0.1
            candidate = action + noise
            
            # 世界模型预测未来
            future = model.forward_world_model(video_latents, candidate, proprio)
            # 价值函数评估
            value = model.forward_value(video_latents, candidate, proprio).item()
            
            if value > best_value:
                best_value = value
                best_action = candidate
    
    return best_action


# ============================================================
# 五、演示
# ============================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


if __name__ == "__main__":
    print("=" * 60)
    print("[Cosmos Policy] 代码演示 — Video Model for Control")
    print("=" * 60)
    
    batch_size = 1
    d_latent, frame_size = 16, 8  # 缩小的 latent 空间（真实是 256×16×16）
    num_video_frames = 4
    
    model = CosmosPolicy(d_latent, frame_size, num_video_frames)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n参数量: {params:.1f}M（教学版本，真实 2B）")
    
    # 伪造输入
    video_latents = torch.randn(batch_size, num_video_frames, d_latent, frame_size, frame_size)
    proprio = torch.randn(batch_size, 7)
    action_gt = torch.randn(batch_size, 7)
    
    # --- Policy 模式 ---
    print("\n1. Policy 模式（观测 → 动作）")
    loss = model.forward_policy(video_latents, proprio, action=action_gt)
    print(f"   训练 loss: {loss.item():.4f}")
    
    with torch.no_grad():
        action_pred = model.forward_policy(video_latents, proprio, action=None)
    print(f"   推理动作: shape={action_pred.shape}, range=[{action_pred.min():.3f}, {action_pred.max():.3f}]")
    
    # --- World Model 模式 ---
    print("\n2. World Model 模式（观测+动作 → 未来帧）")
    future = model.forward_world_model(video_latents, action_gt, proprio)
    print(f"   预测未来帧: shape={future.shape}")
    
    # --- Value Function ---
    print("\n3. Value Function 模式（轨迹 → 价值估计）")
    value = model.forward_value(video_latents, action_gt, proprio)
    print(f"   价值估计: {value.item():.4f}")
    
    # --- Model-based Planning ---
    print("\n4. Model-based Planning（采样 + 评估 + 选最优）")
    best_action = model_based_planning(model, video_latents, proprio, num_candidates=8)
    print(f"   最优动作: shape={best_action.shape}")
    
    print("\n关键设计要点:")
    print("  - Latent Frame Injection: 动作/本体感觉编码为 latent frames")
    print("  - 架构零修改: 复用 Cosmos-Predict2 视频 DiT")
    print("  - 一个模型 = Policy + World Model + Value Function")
    print("  - Planning 提升 12.5%: 世界模型展开 + 价值函数选最优")
    print("  - 视频模型的物理先验: 不需要从零学物理规律")
    print("\n参考: [[Cosmos Policy]] Section 2, 基于 [[Cosmos-Predict2]]")
```

## 设计说明

- **Latent Frame Injection**：将动作/本体感觉/未来帧编码为 latent frame，插入视频 DiT 序列——零架构修改
- **一个模型三种能力**：Policy（观测到动作）、World Model（预测未来帧）、Value Function（评估轨迹）
- **Model-based Planning**：采样 N 个候选动作 → 世界模型展开 → 价值函数选最优（+12.5% vs Direct Policy）
- **视频模型物理先验**：Cosmos-Predict2 预训练提供了运动/碰撞/形变的物理知识，微调收敛更快
- 对照 [[Cosmos Policy]] Section 2-3，融合了 VLA 和 WAM 的能力
```
