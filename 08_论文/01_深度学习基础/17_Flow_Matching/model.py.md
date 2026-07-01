---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Flow Matching - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。参考 [[../16_DDPM/DDPM.md|DDPM]]

```python
"""
Flow Matching 教学实现
======================
一步步手动实现: 直线插值 → 速度场网络 → CFM 损失 → ODE 采样

核心公式 (OT 路径, sigma_min → 0 时的最简形式):
  x_t  = (1-t) * x_0 + t * x_1    —— 噪声和数据之间画一条直线
  目标  = x_1 - x_0                 —— 速度方向恒定, 与时间 t 无关

对比 DDPM:
  DDPM 目标 = 预测噪声 ε (公式复杂, 涉及 √α̅)
  FM 目标   = 预测速度 v = x_1 - x_0 (清晰直观)

这就是 Flow Matching 的优雅之处——
  路线是直的, 目标是恒定的, 模型只需要往前走就行。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===================================================================
# 一、直线路径 —— Flow Matching 的核心
# ===================================================================
# DDPM: 弯曲随机路径, 去噪 1000 步, 前 800 步全是噪声——绕路了。
# FM:   直接在噪声和数据之间画直线。x_t = (1-t)*x_0 + t*x_1
# 为什么这是对的？最优传输 (OT) 理论保证直线是最短路径。
# 速度 v = x_1 - x_0 与 t 无关——整条线恒速——ODE 大步长也不偏航。

def compute_ot_path(x_0, x_1, t):
    """OT 直线上 t 时刻的位置和速度。返回 (x_t, v_t)。"""
    while t.dim() < x_1.dim():
        t = t.unsqueeze(-1)
    x_t = (1 - t) * x_0 + t * x_1
    v_t = x_1 - x_0  # 恒定方向, 与 t 无关!
    return x_t, v_t


# ===================================================================
# 二、时间嵌入
# ===================================================================
# DDPM 的 t 是离散步数, FM 的 t 是连续值 ∈ [0,1], 但编码方式相同。
# t=0→纯噪声, t=1→纯数据, t=0.5→各半。

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        # freq_i = 1 / 10000^(2i/d), i=0,1,...,d/2-1
        freq = torch.exp(
            -math.log(10000) * torch.arange(0, self.dim, 2, device=t.device) / self.dim
        )
        args = t[:, None] * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ===================================================================
# 三、向量场预测网络 —— U-Net
# ===================================================================
# DDPM 预测噪声 ε, FM 预测速度 v。输出 shape 相同, 网络可复用。
# 带 skip connection 和时间条件 (scale+shift) 注入。

class ResBlock(nn.Module):
    """残差块 + 时间条件注入。h = norm(h) * (1+scale) + shift"""

    def __init__(self, in_ch, out_ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        scale, shift = self.t_proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.norm2(h) * (scale + 1) + shift  # 时间条件注入
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class VectorFieldUNet(nn.Module):
    """预测速度场 v_θ(x_t, t) 的 U-Net。输入=输出 shape。"""

    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 4), t_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            TimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4), nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 编码器: 通道翻倍, 空间减半 (stride=2 conv)
        self.enc = nn.ModuleList()
        ch = base_ch
        self.skip_channels = [ch]
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            self.enc.append(ResBlock(ch, out_ch, t_dim))
            ch = out_ch
            self.skip_channels.append(ch)
            if i < len(ch_mult) - 1:
                self.enc.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                self.skip_channels.append(ch)

        # 瓶颈
        mid_ch = base_ch * ch_mult[-1]
        self.mid1 = ResBlock(mid_ch, mid_ch, t_dim)
        self.mid2 = ResBlock(mid_ch, mid_ch, t_dim)

        # 解码器: skip connection 恢复细节
        self.dec = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            out_ch = base_ch * mult
            self.dec.append(ResBlock(ch + self.skip_channels.pop(), out_ch, t_dim))
            ch = out_ch
            if i < len(ch_mult) - 1:
                self.dec.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        skips = [h]

        for layer in self.enc:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            skips.append(h)

        h = self.mid2(self.mid1(h, t_emb), t_emb)

        for layer in self.dec:
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ===================================================================
# 四、CFM 损失 —— 论文最关键的贡献
# ===================================================================
# 问题: 真正的边际向量场 u_t(x) 需要整个分布 p_t(x)——不可行。
# 定理 2: 用条件向量场 u_t(x|x_1) 替代, 梯度不变!
#         ∇_θ L_FM = ∇_θ L_CFM
# 条件场太好算了: u_t(x|x_1) = x_1 - x_0 (只需一个数据点)
# → 训练 = 每次采样 x_1, x_0 → 直线插值 → MSE(预测, x_1 - x_0)
# DDPM 那些 √α̅、β_t、马尔可夫链在 FM 里全部消失。

def cfm_loss(model, x_1):
    """CFM 损失: E[||v_θ(x_t,t) - (x_1-x_0)||²], 梯度等价 FM (定理2)。"""
    B = x_1.shape[0]
    t = torch.rand(B, device=x_1.device)
    x_0 = torch.randn_like(x_1)
    x_t, v_target = compute_ot_path(x_0, x_1, t)
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v_target)


# ===================================================================
# 五、ODE 采样 —— 从噪声走到数据
# ===================================================================
# 求解 dx/dt = v_θ(x,t), 初值 x(0)~N(0,I), 得到 x(1)=生成样本。
# 为什么可大步长? 速度场方向恒定 → 不偏航。DDPM 弯曲路径则不行。
# Euler(1阶) 最快 | Midpoint(2阶) 平衡 | RK4(4阶) 最稳(4x前向/步)

@torch.no_grad()
def sample(model, shape, num_steps=10, method='euler'):
    """ODE 求解 dx/dt=v_θ(x,t), 从 t=0 积到 t=1, x(0)~N(0,I)。

    时间网格: t_i = i/N  (i=0,1,...,N-1), 每步跨 dt=1/N。
    最后一步 (i=N-1): 从 t=1-dt 积到 t=1, 终点 x(1) 即生成样本。
    """
    B = shape[0]
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((B,), i * dt, device=device)

        if method == 'euler':
            x = x + dt * model(x, t)          # 1阶, 最简
        elif method == 'midpoint':
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
            x = x + dt * k2                    # 2阶, 平衡
        elif method == 'rk4':
            k1 = model(x, t)
            k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = model(x + dt * k3, t + dt)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)  # 4阶, 最稳
        else:
            raise ValueError(f"Unknown method: {method}")

    return x


# ===================================================================
# 演示
# ===================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("Flow Matching —— 教学演示")
    print("=" * 55)

    C, H, W = 3, 32, 32
    vf_net = VectorFieldUNet(in_ch=C, base_ch=64, ch_mult=(1, 2, 4))
    print(f"参数量: {sum(p.numel() for p in vf_net.parameters())/1e6:.1f}M")

    # 验证损失计算图
    x_1 = torch.randn(4, C, H, W)
    print(f"\nCFM 损失 (未训练): {cfm_loss(vf_net, x_1).item():.4f}")

    # 验证 OT 路径: ||v|| 在整条线上应恒定
    print(f"\nOT 路径线性性 (||v|| 恒定):")
    x_0, x_1_d = torch.randn(1, C, H, W), torch.randn(1, C, H, W)
    for ti in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _, v = compute_ot_path(x_0, x_1_d, torch.tensor([ti]))
        print(f"  t={ti:.2f}  ||v||={v.norm():.1f}")

    # 多种求解器 + 多种步数对比
    print(f"\n采样对比:")
    for method in ['euler', 'midpoint', 'rk4']:
        for steps in [5, 10]:
            out = sample(vf_net, (1, C, H, W), num_steps=steps, method=method)
            print(f"  {method:>8s} {steps:2d}步  range=[{out.min():+.2f}, {out.max():+.2f}]")

    print(f"\nDDPM vs Flow Matching:")
    print(f"  DDPM 1000步 SDE 弯曲路径 → ~1-2s/图")
    print(f"  FM   10步  ODE 直线路径  → ~0.2s/图")
    print(f"  FM    5步  ODE 直线路径  → ~0.1s/图 (实时级)")
    print(f"\n直线路径 + 恒定速度 = 大步长不跑偏 → 5-20步即可")
```

## 说明

### 和 DDPM 的本质区别

DDPM 问的是"这噪声里有多少是噪声成分?" —— 预测 ε, 公式里根号满天飞。
Flow Matching 问的是"从噪声到数据, 现在该往哪个方向走?" —— 预测 v, 目标就是终点减起点。

| 维度 | DDPM | Flow Matching |
|------|------|---------------|
| 路径形状 | 随机的、弯曲的 | 确定的、直的 |
| 预测目标 | 噪声 ε | 速度 v |
| 目标公式 | ε = (x_t - √α̅·x_0) / √(1-α̅) | **v = x_1 - x_0** |
| 推理步数 | 1000 (或 DDIM 50-100) | **5-20** |
| 训练框架 | SDE + 马尔可夫链 | ODE + 向量场 |
| 实时控制 | 勉强 | 轻松 (< 20ms) |

### 为什么推理可以这么快

DDPM 的弯曲路径 → 方向一直变 → 大步长就走偏 → 必须小步 (1000 步)。
Flow Matching 的 OT 直线 → 方向全程恒定 → 大步长也不偏 → 10 步即够。

体现在代码上就是: `sample()` 函数里 num_steps 从 1000 变成了 10,
其他几乎不变。这个看似微小的改动, 让 Flow Matching 成为 π0、
GR00T N1、FLOWER 等 VLA 动作生成的首选方案。
