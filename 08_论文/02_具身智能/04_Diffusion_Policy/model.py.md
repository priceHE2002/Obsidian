---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Diffusion Policy - 代码实现

> 本文档包含 [[Diffusion Policy]] 的 PyTorch/NumPy 教学实现，涵盖 CNN-based（1D Conv + FiLM）和 Transformer-based 两种扩散策略，以及 DDIM 加速采样。

```python
"""
Diffusion Policy 教学实现
- CNN-based: 1D Temporal Conv + FiLM conditioning
- Transformer-based: Causal Transformer Decoder + Cross-Attention
- DDIM 采样加速（100步训练 → 10步推理，0.1s延迟）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 一、基础组件
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """去噪步数 k 的正弦位置编码
    
    WHY: 扩散模型需要知道当前处于哪个去噪步数。
    正弦编码让相近的步数有相近的 embedding，
    有助于网络学习"噪声水平"的连续变化。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FiLM(nn.Module):
    """Feature-wise Linear Modulation
    
    WHY: 将条件信息注入卷积网络。FiLM 通过 γ（缩放）和 β（偏移）
    对每个通道施加仿射变换，比简单拼接条件向量更有效。
    参考 [[Diffusion Policy]] Fig.3 的 CNN 架构。
    """
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, out_dim * 2)
    
    def forward(self, x, cond):
        scale_shift = self.linear(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return x * (scale + 1.0) + shift


# ============================================================
# 二、CNN-based Diffusion Policy（推荐首试方案）
# ============================================================

class CNNResBlock(nn.Module):
    """1D 卷积残差块 + FiLM 条件化
    
    WHY GroupNorm 而非 BatchNorm: BatchNorm 的 running mean/variance
    与 EMA（指数移动平均）不兼容，而 DDPM 训练常用 EMA。
    """
    def __init__(self, dim, cond_dim, kernel_size=5):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2)
        self.film1 = FiLM(cond_dim, dim)
        self.film2 = FiLM(cond_dim, dim)
        self.act = nn.Mish()
    
    def forward(self, x, cond):
        residual = x
        x = self.conv1(self.act(self.film1(self.norm1(x), cond)))
        x = self.conv2(self.act(self.film2(self.norm2(x), cond)))
        return x + residual


class CNNDiffusionPolicy(nn.Module):
    """CNN-based 扩散策略
    
    WHY 1D Conv: 动作序列是时间序列，1D 卷积天然捕获时序模式。
    WHY FiLM conditioning: 观测 O_t 和去噪步数 k 通过 FiLM 注入每个残差块。
    论文发现位置控制比速度控制效果更好（Section 4.4）。
    """
    def __init__(self, obs_dim=256, action_dim=7, action_horizon=16, 
                 hidden_dim=256, num_res_blocks=6):
        super().__init__()
        self.action_horizon = action_horizon
        self.time_emb = SinusoidalPositionEmbedding(hidden_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_proj = nn.Conv1d(action_dim, hidden_dim, 1)
        self.res_blocks = nn.ModuleList([
            CNNResBlock(hidden_dim, hidden_dim) for _ in range(num_res_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, 1)
        )
    
    def forward(self, x_noisy, obs, timestep):
        """x_noisy: (B, action_dim, action_horizon), obs: (B, obs_dim)"""
        B = x_noisy.shape[0]
        t_emb = self.time_emb(timestep)
        cond = self.cond_encoder(torch.cat([obs, t_emb], dim=-1))
        x = self.input_proj(x_noisy)
        for block in self.res_blocks:
            x = block(x, cond)
        return self.output_proj(x)


# ============================================================
# 三、Transformer-based Diffusion Policy
# ============================================================

class TransformerDiffusionPolicy(nn.Module):
    """Transformer-based 扩散策略
    
    WHY Transformer: CNN 的时间平滑 inductive bias 在需要高频动作切换时
    反而成障碍。Transformer 的多头交叉注意力更灵活。
    论文建议：新任务先用 CNN，时间精细度不够再切换到 Transformer。
    """
    def __init__(self, obs_dim=256, action_dim=7, action_horizon=16, 
                 d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.action_horizon = action_horizon
        self.d_model = d_model
        self.time_emb = SinusoidalPositionEmbedding(d_model)
        self.action_embed = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, action_horizon, d_model) * 0.02)
        self.time_proj = nn.Linear(d_model, d_model)
        self.obs_proj = nn.Linear(obs_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, action_dim)
    
    def forward(self, x_noisy, obs, timestep):
        """x_noisy: (B, action_horizon, action_dim)"""
        B, H, _ = x_noisy.shape
        x = self.action_embed(x_noisy) + self.pos_embed[:, :H, :]
        t_emb = self.time_proj(self.time_emb(timestep).unsqueeze(1))
        x = x + t_emb
        obs_memory = self.obs_proj(obs).unsqueeze(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(H, device=x.device)
        x = self.decoder(tgt=x, memory=obs_memory, tgt_mask=causal_mask)
        return self.output_head(x)


# ============================================================
# 四、扩散过程（前向加噪 + DDPM训练 + DDIM采样）
# ============================================================

class DiffusionProcess:
    """扩散数学过程
    
    WHY DDIM: 训练 100 步 DDPM，推理 10 步 DDIM → 0.1s 延迟。
    """
    def __init__(self, num_train_steps=100, num_inference_steps=10, 
                 beta_start=1e-4, beta_end=0.02):
        self.num_train = num_train_steps
        self.num_inference = num_inference_steps
        betas = torch.linspace(beta_start, beta_end, num_train_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus = (1.0 - alphas_cumprod).sqrt()
    
    def add_noise(self, x0, noise, timestep):
        """前向加噪 x_k = sqrt(ᾱ_k)·x0 + sqrt(1-ᾱ_k)·ε
        
        WHY 闭式解: 不需要迭代加噪，随机采样 k → 解析计算 x_k。
        """
        t = timestep.long()
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1m = self.sqrt_one_minus[t].view(-1, 1, 1)
        return sqrt_a * x0 + sqrt_1m * noise
    
    def ddim_step(self, model, x, obs, t, t_prev, eta=0.0):
        """DDIM 单步去噪（eta=0 确定性，eta=1 等价 DDPM）"""
        alpha_cum = self.alphas_cumprod[t]
        alpha_cum_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        noise_pred = model(x, obs, t)
        pred_x0 = (x - (1-alpha_cum).sqrt() * noise_pred) / alpha_cum.sqrt()
        sigma = eta * ((1-alpha_cum_prev)/(1-alpha_cum) * (1-alpha_cum/alpha_cum_prev)).sqrt()
        noise_term = torch.randn_like(x) if eta > 0 else 0
        return alpha_cum_prev.sqrt() * pred_x0 + (1-alpha_cum_prev-sigma**2).sqrt() * noise_pred + sigma * noise_term
    
    @torch.no_grad()
    def ddim_sample(self, model, obs, action_dim, action_horizon):
        """完整 DDIM 采样：N(0,I) → 逐步去噪 → 动作序列"""
        device = next(model.parameters()).device
        B = obs.shape[0]
        step_indices = torch.linspace(0, self.num_train-1, self.num_inference, dtype=torch.long)
        step_indices = torch.cat([step_indices, torch.tensor([-1])])
        x = torch.randn(B, action_dim, action_horizon, device=device)
        for i in range(len(step_indices) - 1):
            t = step_indices[i]
            t_prev = step_indices[i+1]
            x = self.ddim_step(model, x, obs,
                               torch.full((B,), t, device=device),
                               torch.full((B,), t_prev, device=device))
        return x


# ============================================================
# 五、训练损失
# ============================================================

def diffusion_loss(model, diffusion, x0, obs):
    """去噪分数匹配损失
    
    WHY MSE 就够了: ε_θ 学习 score function ∇log p(a|O_t)，
    梯度场估计与归一化常数无关 → 训练极度稳定。
    没有对抗训练、没有负采样。对照 [[Diffusion Policy]] Section 2.3。
    """
    B = x0.shape[0]
    t = torch.randint(0, diffusion.num_train, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    x_noisy = diffusion.add_noise(x0, noise, t)
    noise_pred = model(x_noisy, obs, t)
    return F.mse_loss(noise_pred, noise)


# ============================================================
# 六、演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[Diffusion Policy] 代码演示")
    print("=" * 60)
    
    batch_size = 4
    obs_dim = 256
    action_dim = 7     # (dx, dy, dz, drx, dry, drz, gripper)
    action_horizon = 16
    
    obs = torch.randn(batch_size, obs_dim)
    x0 = torch.randn(batch_size, action_dim, action_horizon) * 0.1
    
    # --- CNN-based ---
    print("\n1. CNN-based Diffusion Policy")
    cnn_model = CNNDiffusionPolicy(obs_dim, action_dim, action_horizon)
    diffusion = DiffusionProcess(num_train_steps=100, num_inference_steps=10)
    loss = diffusion_loss(cnn_model, diffusion, x0, obs)
    print(f"   CNN 训练 loss: {loss.item():.4f}")
    actions = diffusion.ddim_sample(cnn_model, obs, action_dim, action_horizon)
    print(f"   生成动作形状: {actions.shape}")
    print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
    
    # --- Transformer-based ---
    print("\n2. Transformer-based Diffusion Policy")
    tf_model = TransformerDiffusionPolicy(obs_dim, action_dim, action_horizon)
    x0_tf = x0.transpose(1, 2)
    loss = diffusion_loss(tf_model, diffusion, x0_tf, obs)
    print(f"   Transformer 训练 loss: {loss.item():.4f}")
    
    print(f"\n参数量对比:")
    print(f"   CNN-based:   {sum(p.numel() for p in cnn_model.parameters())/1e6:.1f}M")
    print(f"   Transformer: {sum(p.numel() for p in tf_model.parameters())/1e6:.1f}M")
    print(f"\nDDIM: {diffusion.num_inference} 步推理替代 {diffusion.num_train} 步训练")
    
    print("\n关键设计要点:")
    print("  - FiLM conditioning: 观测+时间步优雅注入 CNN 每层")
    print("  - GroupNorm 而非 BatchNorm: 与 EMA 兼容")
    print("  - DDIM 10 步推理: 0.1s 真实机器人延迟")
    print("  - 位置控制 > 速度控制: 扩散策略的独特优势")
    print("  - T_a=8: 动作执行长度的最优平衡")
    print("  - MSE 极简训练: 无负采样 → 极度稳定")
```

## 设计说明

- **CNN-based 优先**：论文建议新任务先用 CNN 版本，超参更鲁棒
- **FiLM conditioning**：观测 O_t 和去噪步数 k 通过 FiLM 调制每个卷积层
- **DDIM 加速**：100 步训练 → 10 步推理，延迟 0.1s
- **关键发现**：扩散策略绕过了 EBM 的归一化常数问题，score function 估计稳定
- 对照 [[Diffusion Policy]] Section 2-3，[[π0]] 的 Flow Matching 是其变体
