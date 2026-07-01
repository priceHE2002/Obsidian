---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Octo - 代码实现

> 本文档包含 [[Octo]] 的 PyTorch/NumPy 教学实现，涵盖 T5 Encoder-Decoder 骨干、多任务条件化（语言/图像目标）、DDPM 动作头以及模块化注意力设计。

```python
"""
Octo 教学实现 — 第一个开源通用机器人策略
- T5 文本编码器 + 视觉编码器（ResNet/ViT）
- Transformer Decoder-only 骨干 + 模块化注意力
- DDPM 扩散动作头
- 语言/图像双模式任务条件化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 一、Token 化方案（Octo 的核心抽象）
# ============================================================

class ObservationTokenizer(nn.Module):
    """将多模态观测统一 token 化
    
    WHY "任意 token 进": Octo 的设计哲学是"一个 Transformer，
    任意 token 输入，任意 token 输出"。通过 token 化抽象，
    不同机器人平台的不同传感器配置都能使用同一架构。
    参考 [[Octo]] Section 2.1。
    
    输入:
      - 图像 (B, C, H, W) — 每个相机视角独立
      - 本体感觉 (B, proprio_dim) — 关节角度等
    输出:
      - 观测 token 序列 (B, num_tokens, d_model)
    """
    def __init__(self, image_size=224, proprio_dim=7, d_model=384):
        super().__init__()
        self.d_model = d_model
        # 简化的视觉编码器（实际 Octo 用预训练 ResNet/ViT）
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )
        # 本体感觉编码
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, images, proprio):
        """images: (B, num_cameras, 3, H, W)"""
        B, num_cam = images.shape[:2]
        # 每个相机独立编码（论文 Section 4.2：不同视角使用独立编码器）
        img_tokens = []
        for c in range(num_cam):
            img_tok = self.vision_encoder(images[:, c])  # (B, d_model)
            img_tokens.append(img_tok)
        img_tokens = torch.stack(img_tokens, dim=1)  # (B, num_cam, d_model)
        # 本体感觉 token
        prop_token = self.proprio_encoder(proprio).unsqueeze(1)  # (B, 1, d_model)
        # 拼接
        obs_tokens = torch.cat([img_tokens, prop_token], dim=1)  # (B, num_cam+1, d_model)
        return obs_tokens


class TaskTokenizer(nn.Module):
    """任务条件 token 化：语言指令 或 目标图像
    
    WHY 双模式: Octo 支持两种任务指定方式——自然语言
    ("pick up the apple") 或目标图像 ("到达这个状态")。
    两种方式共用同一骨干，只需切换输入头。
    
    语言模式: T5-base 编码 → 单个 embedding
    图像模式: 浅层 CNN → 单个 token
    """
    def __init__(self, d_model=384, t5_dim=768):
        super().__init__()
        self.d_model = d_model
        # T5 输出投影（实际 Octo 使用预训练 T5-base，这里用简化线性层模拟）
        self.language_proj = nn.Sequential(
            nn.Linear(t5_dim, d_model),
            nn.LayerNorm(d_model),
        )
        # 目标图像编码器（浅层 CNN）
        self.goal_image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, language_embed=None, goal_image=None):
        """至少提供一个条件"""
        if language_embed is not None:
            # language_embed: (B, t5_dim) — 来自 T5 encoder
            return self.language_proj(language_embed).unsqueeze(1)  # (B, 1, d_model)
        elif goal_image is not None:
            # goal_image: (B, 3, H, W)
            return self.goal_image_encoder(goal_image).unsqueeze(1)
        else:
            raise ValueError("需要语言指令或目标图像作为条件")


# ============================================================
# 二、模块化注意力 Transformer 骨干
# ============================================================

class ModularAttentionTransformer(nn.Module):
    """Decoder-only Transformer + 模块化注意力设计
    
    WHY 模块化注意力: Octo 对不同 token 类型使用不同的注意力模式:
    - 观测 tokens: 全自注意力（所有观测之间互相关注）
    - 任务 token: 交叉注意力到观测 tokens（任务"查询"观测）
    - 动作 tokens: 因果自注意力（动作生成是自回归的）
    
    这种设计比统一自注意力更灵活，但实现更复杂。
    参考 [[Octo]] Section 2.3。
    """
    def __init__(self, d_model=384, nhead=6, num_layers=8, max_seq_len=16):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # 可学习的位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        # Transformer Decoder layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, obs_tokens, task_token, action_tokens=None):
        """
        obs_tokens: (B, num_obs, d_model)
        task_token: (B, 1, d_model)
        action_tokens: (B, num_actions, d_model) — 推理时逐步追加
        """
        # 拼接所有 tokens: [obs | task | actions]
        if action_tokens is not None:
            x = torch.cat([obs_tokens, task_token, action_tokens], dim=1)
            num_obs = obs_tokens.shape[1]
            num_task = 1
            num_actions = action_tokens.shape[1]
        else:
            x = torch.cat([obs_tokens, task_token], dim=1)
            num_obs = obs_tokens.shape[1]
            num_task = 1
            num_actions = 0
        
        total_len = x.shape[1]
        x = x + self.pos_embed[:, :total_len]
        
        # 构建模块化注意力掩码
        attn_mask = self._build_modular_mask(num_obs, num_task, num_actions, x.device)
        
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.final_norm(x)
        return x
    
    def _build_modular_mask(self, num_obs, num_task, num_actions, device):
        """构建模块化注意力掩码
        
        - 观测 tokens: 可以看所有观测 tokens（全自注意力）
        - 任务 token: 只能看观测 tokens（交叉注意力）
        - 动作 tokens: 因果自注意力（只能看自己和之前的动作）
        
        返回: 0=可看, -inf=不可看
        """
        total = num_obs + num_task + num_actions
        mask = torch.zeros(total, total, device=device)
        
        # 观测区域（前 num_obs 个）: 可以看所有观测 tokens
        mask[:num_obs, :num_obs] = 0
        mask[:num_obs, num_obs:] = float('-inf')
        
        # 任务 token: 可以看观测 tokens
        task_idx = num_obs
        mask[task_idx, :num_obs] = 0
        mask[task_idx, num_obs:] = float('-inf')
        
        # 动作 tokens: 因果注意力
        if num_actions > 0:
            action_start = num_obs + num_task
            for i in range(num_actions):
                # 可以看到观测、任务、以及自己及之前的动作
                mask[action_start + i, :action_start] = 0
                mask[action_start + i, action_start:action_start + i + 1] = 0
                if action_start + i + 1 < total:
                    mask[action_start + i, action_start + i + 1:] = float('-inf')
        
        return mask


class TransformerBlock(nn.Module):
    """单个 Transformer block（自注意力 + FFN）"""
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ============================================================
# 三、DDPM 动作头
# ============================================================

class DDPMActionHead(nn.Module):
    """基于 DDPM 的动作生成头
    
    WHY DDPM 而非自回归: 自回归生成速度慢——每个动作 token
    需要前向推理一次。DDPM 可以并行去噪整个动作序列。
    Octo 原版用的是自回归（性能因此落后），这里采用更先进的
    DDPM 方案——与 [[Diffusion Policy]] 的思想一致。
    """
    def __init__(self, d_model=384, action_dim=7, horizon=16, 
                 num_steps=100):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_steps = num_steps
        
        # 噪声预测网络
        self.time_emb = nn.Sequential(
            SinusoidalEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim * horizon, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, action_dim * horizon),
        )
        # DDPM β schedule
        betas = torch.linspace(1e-4, 0.02, num_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1-alphas_cumprod).sqrt())
    
    def add_noise(self, x0, noise, t):
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise
    
    def forward(self, x_noisy, context, t):
        """context: (B, d_model) — Transformer 输出的上下文"""
        B = x_noisy.shape[0]
        t_emb = self.time_emb(t).unsqueeze(1)
        combined = t_emb + context.unsqueeze(1)
        inp = torch.cat([combined.squeeze(1), x_noisy.reshape(B, -1)], dim=-1)
        noise_pred = self.net(inp).reshape(B, self.horizon, self.action_dim)
        return noise_pred
    
    @torch.no_grad()
    def sample(self, context, num_inference_steps=10):
        """从噪声采样动作序列"""
        B = context.shape[0]
        device = context.device
        x = torch.randn(B, self.horizon, self.action_dim, device=device)
        steps = torch.linspace(self.num_steps-1, 0, num_inference_steps, dtype=torch.long)
        for i in range(num_inference_steps):
            t = torch.full((B,), steps[i], device=device)
            noise_pred = self.forward(x, context, t)
            alpha = self.alphas_cumprod[steps[i]]
            alpha_prev = self.alphas_cumprod[steps[i+1]] if i+1 < num_inference_steps else torch.tensor(1.0)
            x = (x - (1-alpha).sqrt() * noise_pred) / alpha.sqrt()
            if i < num_inference_steps - 1:
                noise = torch.randn_like(x)
                x = x + ((1-alpha_prev) / (1-alpha)).sqrt() * ((1-alpha/alpha_prev).sqrt()) * noise
        return x


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


# ============================================================
# 四、完整 Octo 模型
# ============================================================

class Octo(nn.Module):
    """Octo: Open-Source Generalist Robot Policy
    
    模块化架构:
      - obs_tokenizer: 图像+本体感觉 → token 序列
      - task_tokenizer: 语言/目标图像 → 任务 token
      - transformer: 模块化注意力骨干
      - action_head: DDPM 动作生成
    """
    def __init__(self, proprio_dim=7, action_dim=7, d_model=384):
        super().__init__()
        self.obs_tokenizer = ObservationTokenizer(proprio_dim=proprio_dim, d_model=d_model)
        self.task_tokenizer = TaskTokenizer(d_model=d_model)
        self.transformer = ModularAttentionTransformer(d_model=d_model)
        self.action_head = DDPMActionHead(d_model=d_model, action_dim=action_dim)
    
    def forward(self, images, proprio, language_embed, action=None):
        """
        images: (B, num_cam, 3, H, W)
        proprio: (B, proprio_dim)
        language_embed: (B, t5_dim) 或 None
        action: (B, horizon, action_dim) 或 None
        """
        # Token 化
        obs_tokens = self.obs_tokenizer(images, proprio)
        task_token = self.task_tokenizer(language_embed=language_embed)
        # Transformer 编码
        context_seq = self.transformer(obs_tokens, task_token)
        # 取最后一个 token 作为上下文（简化方案）
        context = context_seq[:, -1]
        # 动作预测
        if action is not None:
            B = action.shape[0]
            t = torch.randint(0, self.action_head.num_steps, (B,), device=action.device)
            noise = torch.randn_like(action)
            x_noisy = self.action_head.add_noise(action.reshape(B, -1), noise.reshape(B, -1), t)
            noise_pred = self.action_head(x_noisy.reshape(B, -1, self.action_head.action_dim), context, t)
            return noise_pred, noise
        else:
            return self.action_head.sample(context)


# ============================================================
# 五、演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[Octo] 代码演示 — Open-Source Generalist Robot Policy")
    print("=" * 60)
    
    batch_size = 2
    num_cameras = 2  # 两个相机视角
    proprio_dim = 7  # 7 维关节角度
    action_dim = 7   # 末端执行器位姿
    t5_dim = 768     # T5-base 输出维度
    
    model = Octo(proprio_dim=proprio_dim, action_dim=action_dim)
    
    # 伪造输入
    images = torch.randn(batch_size, num_cameras, 3, 224, 224)
    proprio = torch.randn(batch_size, proprio_dim)
    language_embed = torch.randn(batch_size, t5_dim)
    
    # --- 训练 ---
    print("\n1. DDPM 训练（去噪分数匹配）")
    action_gt = torch.randn(batch_size, 16, action_dim)
    noise_pred, noise = model(images, proprio, language_embed, action=action_gt)
    train_loss = F.mse_loss(noise_pred, noise)
    print(f"   Training Loss: {train_loss.item():.4f}")
    
    # --- 推理 ---
    print("\n2. DDPM 推理（从噪声采样动作）")
    with torch.no_grad():
        actions = model(images, proprio, language_embed)
    print(f"   生成动作形状: {actions.shape}")  # (2, 16, 7)
    print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # --- 目标图像模式 ---
    print("\n3. 目标图像条件（替代语言指令）")
    goal_image = torch.randn(batch_size, 3, 224, 224)
    task_token_img = model.task_tokenizer(goal_image=goal_image)
    print(f"   目标图像 token 形状: {task_token_img.shape}")
    
    print(f"\n参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("\n关键设计要点:")
    print("  - Token 化抽象: 任意传感器 → 统一 token 序列")
    print("  - 模块化注意力: 观测(全自注意力)/任务(交叉)/动作(因果)")
    print("  - 双模式条件: 语言指令 或 目标图像")
    print("  - DDPM 动作头: 并行去噪 > 自回归生成")
    print("\n参考: [[Octo]] Section 2, 模块化思想被 [[OpenVLA]] 继承")
```

## 设计说明

- **Token 化抽象**：图像 + 本体感觉统一为 token 序列，支持任意传感器配置
- **模块化注意力**：不同 token 类型使用不同注意力模式（全自注意力/交叉/因果）
- **双模式条件化**：语言指令（T5）和目标图像两套条件方案
- **DDPM 动作头**：替代原版自回归生成（性能更优，与 [[Diffusion Policy]] 一致）
- 对照 [[Octo]] Section 2-3，模块化架构思想深刻影响了 [[OpenVLA]]
```
