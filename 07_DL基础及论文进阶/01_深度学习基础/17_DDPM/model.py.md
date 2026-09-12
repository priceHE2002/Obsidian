---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DDPM (Denoising Diffusion Probabilistic Models) - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DDPM (Denoising Diffusion Probabilistic Models) - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
DDPM (Denoising Diffusion Probabilistic Models)
================================================
论文: "Denoising Diffusion Probabilistic Models"
      (Ho et al., UC Berkeley, NeurIPS 2020)
核心贡献: 让扩散模型从理论可行变为工程可用。定义前向加噪过程
         (马尔可夫链逐步破坏数据→纯噪声) 和反向去噪过程
         (学习神经网络从噪声恢复数据)。关键发现: 噪声预测网络 ε_θ
         等价于学习 score function，简化损失使训练稳定。
架构: 前向扩散(重参数化) → U-Net 噪声预测 → DDPM/DDIM 采样

与 [[../18_Flow_Matching/Flow Matching.md|Flow Matching]] 的关系: DDPM 是随机扩散，
  Flow Matching 是确定性传播，后者在 VLA 动作生成中渐进取代前者
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# 1. 噪声调度 —— 定义前向过程的 β_t
# ==============================================================================
def linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """线性噪声调度（DDPM 默认）。

    β_t 从 beta_start 到 beta_end 线性增长。
    早期步骤 β_t 小 → 信息量损失少 → 学得精细
    后期步骤 β_t 大 → 快速逼近纯噪声

    为什么用线性调度？
    简单有效。在高分辨率图像上表现好。
    缺点是低分辨率（如 CIFAR-10）对数似然不够优。
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps=1000, s=0.008):
    """余弦噪声调度（Improved DDPM 改进）。

    公式: α̅_t = f(t)/f(0), f(t) = cos²((t/T+s)/(1+s) · π/2)

    为什么余弦比线性好？
    在 t 中等时的噪声添加更慢（α̅_t 衰减更平缓），
    避免了高噪声水平的过早信息"浪费"。
    DDPM 原始用线性，余弦是 Improved DDPM 的改进。
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    ft = torch.cos(((t / timesteps + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = ft / ft[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, max=0.999)


# ==============================================================================
# 2. 前向扩散 —— 重参数化技巧
# ==============================================================================
class GaussianDiffusion:
    """前向扩散过程 + 训练/采样逻辑。

    前向过程（马尔可夫链）:
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)

    重参数化 → 直接从 x_0 采样 x_t（无需迭代）:
    x_t = √α̅_t · x_0 + √(1-α̅_t) · ε,  ε~N(0,I)

    其中 α_t = 1-β_t, α̅_t = ∏_{s=1}^{t} α_s

    为什么重参数化如此重要？
    - 训练时不需要迭代 1000 步生成 x_t —— 直接一步到位
    - 每个训练 step 可以随机采样不同的 t，高效利用数据
    """

    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 前向扩散 q(x_t | x_0) 的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 反向过程 p(x_{t-1} | x_t) 的参数
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.posterior_variance = (
            betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """前向扩散: 从 x_0 直接采样 x_t（重参数化）。

        公式: x_t = √α̅_t · x_0 + √(1-α̅_t) · ε

        这是训练循环的核心一步 —— 不需要模拟整个扩散过程。
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def p_sample(self, model, x_t, t):
        """DDPM 单步采样: x_t → x_{t-1}。

        公式（重参数化 ε_θ）:
        x_{t-1} = 1/√α_t · (x_t - β_t/√(1-α̅_t) · ε_θ(x_t, t)) + σ_t · z

        其中 z ~ N(0,I) if t>1 else 0（最后一步不添加噪声）。

        为什么用噪声预测而非直接预测 x_0？
        预测噪声 ε 是一个 easier 的回归任务 ——
        噪声 ~ N(0,I)，而 x_0 的分布复杂得多。
        DDPM 证明这种方法显著提升生成质量。
        """
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )

        # 预测噪声 ε_θ(x_t, t)
        predicted_noise = model(x_t, t)

        # 估计 x_0 的"预测"（可用于理解去噪过程）
        pred_x0 = sqrt_recip_alpha_t * (
            x_t - betas_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )

        # 计算 x_{t-1} 的均值
        model_mean = sqrt_recip_alpha_t * (
            x_t - betas_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )
        # 加上 posterior_variance 中的 x_t 贡献项
        posterior_variance_t = self._extract(
            self.posterior_variance, t, x_t.shape
        )
        model_mean = model_mean + posterior_variance_t * (
            x_t - self._extract(self.sqrt_alphas_cumprod, t, x_t.shape) * pred_x0
        ) / (1. - self._extract(self.alphas_cumprod, t, x_t.shape))

        # 添加噪声（最后一步不加）
        if (t == 0).any():
            noise = torch.zeros_like(x_t)
        else:
            noise = torch.randn_like(x_t)
        variance = torch.sqrt(posterior_variance_t) * noise

        return model_mean + variance

    def p_sample_loop(self, model, shape, device='cpu'):
        """DDPM 完整采样循环（T=1000 步）。

        从纯噪声 x_T ~ N(0,I) 开始，逐步去噪得到 x_0。
        这是 DDPM 生成图像的标准流程。
        """
        b = shape[0]
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)

        return x

    def ddim_sample(self, model, x_t, t, prev_t, eta=0.0):
        """DDIM 确定性采样（单步跳步采样）。

        DDIM (Denoising Diffusion Implicit Models) 将反向过程改为
        确定性 ODE，允许跳过中间步骤。
        eta=0 → 完全确定性（DDIM）
        eta=1 → 等价于 DDPM

        公式（DDIM, eta=0）:
        x_{prev} = √α̅_{prev} · x̂_0 + √(1-α̅_{prev}) · ε_θ

        为什么 DDIM 可以跳步？
        DDIM 将其重新解释为隐式概率模型，
        不再依赖于马尔可夫链的逐步约束。
        """
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_prev = self._extract(self.alphas_cumprod_prev, prev_t, x_t.shape)

        # 预测噪声
        predicted_noise = model(x_t, t)

        # 估计 x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
                   ) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)  # 确保在合法范围内

        # 指向 x_{prev} 的方向
        pred_dir = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise

        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir

        # 随机性: eta=0 不加噪声 (DDIM), eta>0 添加随机成分
        if eta > 0:
            variance = eta * torch.sqrt(
                (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
                * (1 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            x_prev = x_prev + variance * torch.randn_like(x_t)

        return x_prev

    def ddim_sample_loop(self, model, shape, device='cpu',
                         sampling_timesteps=50, eta=0.0):
        """DDIM 加速采样循环。

        sampling_timesteps=50: 1000 步中均匀选取 50 步做 DDIM 采样。

        为什么 DDIM 加速至关重要？
        DDPM 1000 步: 约 20-30 秒 / 单张图 (V100)
        DDIM 50 步:   约 1-2 秒 / 单张图
        DDIM 10 步:   约 0.2-0.5 秒 ← 接近实时控制需求
        """
        b = shape[0]
        x = torch.randn(shape, device=device)

        # 均匀采样 timesteps（倒序）
        times = torch.linspace(self.timesteps - 1, 0, sampling_timesteps)
        times = times.long().tolist()

        time_pairs = list(zip(times[:-1], times[1:]))

        for t, prev_t in time_pairs:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            prev_t_batch = torch.full((b,), prev_t, device=device, dtype=torch.long)
            x = self.ddim_sample(model, x, t_batch, prev_t_batch, eta=eta)

        return x

    def training_loss(self, model, x_0):
        """DDPM 简化训练损失。

        公式: L_simple = E_{t, x_0, ε} [||ε - ε_θ(x_t, t)||²]

        采样 t ~ Uniform(1, T)，生成噪声 ε，
        用重参数化生成 x_t，让模型预测 ε。

        为什么这是简化损失？
        原始 VLB (变分下界) 对各 t 加权不均（小 t 权重过大）。
        L_simple 对所有 t 等权 → 模型在所有噪声水平都学好。
        """
        b = x_0.shape[0]
        t = torch.randint(1, self.timesteps, (b,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @staticmethod
    def _extract(arr, t, broadcast_shape):
        """从 1D 张量中按索引提取并广播到目标形状。"""
        res = arr.to(t.device)[t].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


# ==============================================================================
# 3. U-Net 噪声预测网络
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal 时间步嵌入（与 Transformer 位置编码同族）。

    将时间步 t ∈ [0, T-1] 映射到高维连续空间。
    公式: PE(t, 2i) = sin(t / 10000^(2i/d))
          PE(t, 2i+1) = cos(t / 10000^(2i/d))

    为什么用 sinusodial 而非可学习？
    - 连续值: 即使 t 不是整数也能产出合理嵌入
    - 外推: 训练用 0-999，推理时可扩展到 1000+
    - 高频-低频混合: 不同维度编码不同频率信息
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """U-Net 残差块。

    每个残差块由两个卷积 + GroupNorm + SiLU 激活组成。
    时间嵌入通过 FiLM-style 调制注入
    （先 scale + shift 再进入残差路径）。
    """

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2),  # scale + shift
        )
        self.dropout = nn.Dropout(dropout)

        # 通道数不匹配时用 1×1 conv 做 shortcut
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        # 时间调制: scale + shift
        scale_shift = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = h * (scale + 1) + shift
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """空间自注意力块（在特定分辨率上使用）。

    DDPM 的 U-Net 在 16×16 分辨率处插入自注意力，
    让模型理解长距离空间依赖关系。

    为什么只在中间层用 attention？
    高分辨率（如 256²）全注意力计算量过大 O((HW)²)，
    仅在分辨率较低的 bottleneck 附近用 attention 是最优效率/质量平衡。
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        scale = self.head_dim ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class UpDownBlock(nn.Module):
    """下采样或上采样 + 残差块组合。"""

    def __init__(self, in_ch, out_ch, time_emb_dim,  mode='down', dropout=0.0):
        super().__init__()
        self.mode = mode
        self.res = ResBlock(in_ch, out_ch, time_emb_dim, dropout)
        if mode == 'down':
            self.sample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        elif mode == 'up':
            self.sample = nn.Upsample(scale_factor=2, mode='bilinear',
                                       align_corners=False)
            self.conv = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        if self.mode == 'down':
            x = self.sample(x)
        elif self.mode == 'up':
            x = self.sample(x)
            x = self.conv(x)
        return x


class UNet(nn.Module):
    """DDPM 的去噪 U-Net。

    架构:
    - 编码器: 4 次下采样 (H/2, H/4, H/8, H/16)
    - 瓶颈: ResBlock + Self-Attention + ResBlock
    - 解码器: 4 次上采样 + skip connections

    时间条件注入: 每个 ResBlock 通过 FiLM 接收时间嵌入。
    自注意力: 在最低分辨率 (16×16) 处插入。

    为什么用 U-Net 而非 ViT？
    DDPM 时代 (2020) ViT 尚未成熟。
    U-Net 的多尺度特征 + skip connection 对像素级重建很有效。
    后来的 DiT (2022) 证明 ViT 在扩散模型中更好。
    """

    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2, time_emb_dim=256, dropout=0.0):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # 输入投影
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 编码器
        self.down_blocks = nn.ModuleList()
        ch = base_ch
        chs = [ch]
        for level, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch
                chs.append(ch)
            if level < len(ch_mult) - 1:
                self.down_blocks.append(
                    UpDownBlock(ch, ch, time_emb_dim, 'down', dropout)
                )
                chs.append(ch)

        # 瓶颈（含自注意力）
        mid_ch = base_ch * ch_mult[-1]
        self.mid_res1 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_res2 = ResBlock(mid_ch, mid_ch, time_emb_dim, dropout)

        # 解码器（与编码器对称，含 skip connection）
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(ch_mult))):
            out_ch = base_ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                self.up_blocks.append(
                    ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)
                )
                ch = out_ch
            if level > 0:
                self.up_blocks.append(
                    UpDownBlock(ch, ch, time_emb_dim, 'up', dropout)
                )

        # 输出层
        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        """
        x: (B, C, H, W) — 加噪后的图像 x_t
        t: (B,) — 时间步
        返回: (B, C, H, W) — 预测的噪声 ε
        """
        t_emb = self.time_mlp(t)

        # 编码器
        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        # 瓶颈
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # 解码器（含 skip connection）
        for block in self.up_blocks:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            if isinstance(block, UpDownBlock):
                h = block(h, t_emb)
            else:
                h = block(h, t_emb)
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear')

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


# ==============================================================================
# 4. DDPM 完整包装
# ==============================================================================
class DDPM(nn.Module):
    """DDPM 完整模型：U-Net + 扩散过程 + 采样。

    训练: model.training_loss(x_0) → MSE loss
    DDPM 采样: model.ddpm_sample(shape) → x_0 (1000 步)
    DDIM 采样: model.ddim_sample(shape, steps=50) → x_0 (加速)
    """

    def __init__(self, unet, timesteps=1000, beta_schedule='linear'):
        super().__init__()
        self.unet = unet
        self.diffusion = GaussianDiffusion(timesteps, beta_schedule)

    def forward(self, x, t):
        """Unet 噪声预测。"""
        return self.unet(x, t)

    def training_loss(self, x_0):
        """计算简化训练损失 L_simple。"""
        return self.diffusion.training_loss(self.unet, x_0)

    @torch.no_grad()
    def ddpm_sample(self, shape, device='cpu'):
        """DDPM 完全采样（1000 步）。"""
        return self.diffusion.p_sample_loop(self.unet, shape, device)

    @torch.no_grad()
    def ddim_sample(self, shape, device='cpu', steps=50, eta=0.0):
        """DDIM 加速采样。"""
        return self.diffusion.ddim_sample_loop(
            self.unet, shape, device, steps, eta
        )


# ==============================================================================
# 演示
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DDPM — Denoising Diffusion Probabilistic Models 演示")
    print("=" * 60)

    device = 'cpu'
    img_size = 32

    # 创建 U-Net 和 DDPM
    unet = UNet(in_ch=3, base_ch=64, ch_mult=(1, 2, 4), num_res_blocks=1)
    model = DDPM(unet, timesteps=1000, beta_schedule='linear').to(device)

    # 训练步骤演示
    x_0 = torch.randn(4, 3, img_size, img_size).to(device)
    loss = model.training_loss(x_0)
    print(f"\n训练步骤:")
    print(f"  输入 x_0 形状: {x_0.shape}")
    print(f"  L_simple (MSE 噪声预测): {loss.item():.4f}")
    print(f"  目标: 预测添加的噪声 ε ~ N(0,I)")

    # 前向扩散演示
    diff = model.diffusion
    t = torch.tensor([250, 500, 750, 999]).to(device)
    x_t = diff.q_sample(x_0[:1], t)
    print(f"\n前向扩散 (x_0 → x_t):")
    for i, ti in enumerate(t.tolist()):
        snr = (diff.alphas_cumprod[ti] / (1 - diff.alphas_cumprod[ti])).item()
        print(f"  t={ti:4d}: α̅_t={diff.alphas_cumprod[ti].item():.4f}, "
              f"SNR={snr:.3f}, ||x_t||_mean={x_t[i].norm().item() / (3*32*32)**0.5:.3f}")

    # DDPM 采样（1000 步，简化演示 - 只做 1 张图）
    print(f"\nDDPM 采样 (1000 步)...")
    sample = model.ddpm_sample((1, 3, img_size, img_size), device=device)
    print(f"  生成样本形状: {sample.shape}")
    print(f"  样本值范围: [{sample.min().item():.3f}, {sample.max().item():.3f}]")

    # DDIM 采样（50 步，加速）
    print(f"\nDDIM 采样 (50 步, 确定性)...")
    sample_ddim = model.ddim_sample((1, 3, img_size, img_size),
                                     device=device, steps=50, eta=0.0)
    print(f"  生成样本形状: {sample_ddim.shape}")
    print(f"  加速比: 1000→50 = 20x")

    # DDIM 10 步（极限加速）
    print(f"\nDDIM 采样 (10 步, 极限加速)...")
    sample_fast = model.ddim_sample((1, 3, img_size, img_size),
                                     device=device, steps=10, eta=0.0)
    print(f"  生成样本形状: {sample_fast.shape}")
    print(f"  加速比: 1000→10 = 100x")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params / 1e6:.1f}M")
    print(f"真实 DDPM U-Net (CIFAR-10): ~35M params")
    print(f"真实 DDPM U-Net (ImageNet 256): ~550M params")
    print(f"\n关键: L_simple 均匀重加权使训练稳定，噪声预测等价于 score matching")

```

```
