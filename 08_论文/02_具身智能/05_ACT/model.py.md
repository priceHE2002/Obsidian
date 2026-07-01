---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# ACT - 代码实现

> 本文档包含 [[ACT]] 的 PyTorch/NumPy 教学实现，涵盖 Transformer Encoder-Decoder + CVAE（KL 正则）+ Action Chunking 三大核心组件。

```python
"""
ACT (Action Chunking with Transformers) 教学实现
- Transformer Encoder-Decoder 架构
- CVAE 隐变量建模（KL 正则化）
- Action Chunking + Temporal Ensembling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ============================================================
# 一、位置编码
# ============================================================

class SinusoidalPositionEncoding(nn.Module):
    """正弦位置编码（Transformer 标准）
    
    WHY 需要位置编码: Transformer 自身不知道序列顺序，
    必须显式注入位置信息。ACT 处理的是 4 个相机视角的
    图像序列 + 历史帧，位置编码至关重要。
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


# ============================================================
# 二、CVAE 核心：风格隐变量建模
# ============================================================

class CVAEEncoder(nn.Module):
    """CVAE 编码器：从 (观测, 真实动作) 推断风格隐变量 z
    
    WHY CVAE: 人类演示者的动作风格差异极大（快/慢、左/右绕），
    如果只用 MSE 回归会学到"平均风格"——结果四不像。
    CVAE 通过学习隐变量 z 来捕获"风格"——除给定观测外、
    导致动作变化的其他因素。
    
    训练时：编码器可以看到真实动作 a，z ~ q_φ(z|o,a)
    推理时：从先验 p(z) = N(0,I) 直接采样
    
    KL 正则化让 p(z|o,a) 接近 N(0,I)，保证训练和推理时
    z 的分布不会差太远。对照 [[ACT]] Section 2.3。
    """
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        # 将观测和动作编码为联合表示
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 输出分布参数 μ 和 log σ²
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, obs, action):
        """返回 z 和 KL 散度"""
        x = torch.cat([obs, action], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # KL 散度: D_KL(N(μ,σ²) || N(0,I))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return z, kl


# ============================================================
# 三、ACT 主网络
# ============================================================

class ACT(nn.Module):
    """ACT: Action Chunking with Transformers
    
    完整架构:
    1. 多视角图像 → ResNet 编码 → 拼接 + 位置编码
    2. Transformer Encoder → 时-空融合
    3. CVAE Encoder → 风格隐变量 z
    4. Transformer Decoder（z + 编码特征 → 动作序列）
    
    WHY Action Chunking: 一次预测 k 步动作，减少组合误差 k 倍。
    有效任务长度 = 原始长度 / k。
    类似"动作作为一种 open-loop 宏指令，但每 k 步重规划以保持闭环"。
    """
    def __init__(self, obs_dim=256, action_dim=14, chunk_size=100,
                 latent_dim=32, d_model=512, nhead=8, num_encoder_layers=4,
                 num_decoder_layers=6):
        super().__init__()
        self.action_dim = action_dim    # 双臂 14 维（7×2）
        self.chunk_size = chunk_size     # k 步动作块
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # 观测编码器
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.pos_encoding = SinusoidalPositionEncoding(d_model)
        
        # Transformer Encoder：融合时序+多视角信息
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # CVAE Encoder：推断风格 z
        # 输入: 编码后的观测特征 + 全部动作序列
        encoder_out_dim = d_model  # 简化：取 encoder 输出均值
        self.cvae_encoder = CVAEEncoder(
            obs_dim=encoder_out_dim, 
            action_dim=action_dim * chunk_size,  # 全动作序列
            latent_dim=latent_dim
        )
        
        # 动作查询 token（decoder 输入）
        self.action_query = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        # 隐变量投影到 decoder 空间
        self.z_proj = nn.Linear(latent_dim, d_model)
        
        # Transformer Decoder：z + 编码特征 → 动作序列
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        # 动作输出头
        self.action_head = nn.Linear(d_model, action_dim)
    
    def encode_obs(self, obs_seq):
        """编码观测序列 → 条件特征
        
        obs_seq: (B, num_frames, obs_dim)
        典型配置: 4 个相机 × 2 历史帧 = 8 个 token
        """
        x = self.obs_encoder(obs_seq)  # (B, num_frames, d_model)
        x = self.pos_encoding(x)
        memory = self.encoder(x)  # (B, num_frames, d_model)
        return memory
    
    def forward(self, obs_seq, action_seq=None, z=None):
        """
        训练模式: action_seq 非 None，使用 CVAE 编码器
        推理模式: action_seq=None, z 从 N(0,I) 采样
        
        obs_seq: (B, num_frames, obs_dim)
        action_seq: (B, chunk_size, action_dim) — 训练时提供
        z: (B, latent_dim) — 推理时提供（否则从先验采样）
        """
        B = obs_seq.shape[0]
        # 编码观测
        memory = self.encode_obs(obs_seq)
        
        # ------ CVAE: 获取风格隐变量 ------
        if action_seq is not None:
            # 训练模式：用真实动作推断 z
            action_flat = action_seq.reshape(B, -1)  # (B, chunk_size * action_dim)
            memory_pooled = memory.mean(dim=1)  # (B, d_model)
            z, kl_loss = self.cvae_encoder(memory_pooled, action_flat)
        else:
            # 推理模式：从先验 N(0,I) 采样
            if z is None:
                z = torch.randn(B, self.latent_dim, device=obs_seq.device)
            kl_loss = torch.tensor(0.0, device=obs_seq.device)
        
        # ------ Decoder: z + memory → 动作序列 ------
        z_emb = self.z_proj(z).unsqueeze(1)  # (B, 1, d_model)
        # 将 z embedding 加到 action queries 上
        tgt = self.action_query.expand(B, -1, -1) + z_emb
        # WHY: tgt_mask 使用因果注意力，动作 token 按序生成
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.chunk_size, device=obs_seq.device
        )
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        # 输出动作
        actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        return actions, kl_loss


# ============================================================
# 四、Temporal Ensembling（让动作过渡更平滑）
# ============================================================

def temporal_ensemble(pred_buffers, exec_step, chunk_size, exp_weight=0.01):
    """
    对重叠的动作块取加权平均，实现平滑的动作过渡。
    
    WHY Temporal Ensembling: 如果每 k 步预测一个动作块，
    相邻块之间可能不连续。通过高频查询策略并对重叠块
    做指数衰减加权平均，越近的预测权重越大。
    
    pred_buffers: list of (chunk_size, action_dim) — 历史预测
    exec_step: 当前执行步（用于计算每个预测的"新鲜度"）
    返回: (action_dim,) — 加权平均后的动作
    """
    num_preds = len(pred_buffers)
    if num_preds == 0:
        return None
    
    weighted_sum = 0.0
    weight_total = 0.0
    for i, pred in enumerate(pred_buffers):
        # 指数衰减权重：越新的预测权重越大
        # 第 i 个预测的"年龄" = num_preds - 1 - i
        age = num_preds - 1 - i
        w = math.exp(-exp_weight * age)
        weighted_sum += w * pred[0]  # 取当前步的动作
        weight_total += w
    
    return weighted_sum / weight_total


# ============================================================
# 五、完整训练 + 推理流程
# ============================================================

def act_loss(model, obs_seq, action_seq, kl_beta=0.01):
    """ACT 训练损失 = MSE + β * KL
    
    WHY β 权衡: β 太小 → z 不服从 N(0,I) → 推理时从先验采样不准
    β 太大 → z 被拉向 N(0,I) → 丢失风格信息 → 动作多样性减弱
    论文推荐 β ∈ [0.001, 0.1]，大多数任务 β=0.01 表现最优
    """
    pred_actions, kl_loss = model(obs_seq, action_seq)
    mse_loss = F.mse_loss(pred_actions, action_seq)
    total_loss = mse_loss + kl_beta * kl_loss
    return total_loss, mse_loss, kl_loss


@torch.no_grad()
def act_inference(model, obs_seq, num_styles=1):
    """ACT 推理：从不同 z 采样可产生不同风格的动作
    
    每次从 N(0,I) 采样不同的 z → 不同的动作风格。
    WHY 有用: 在真实部署时，你可以采样多个 z，
    用 collision checker 筛选，选最安全的动作。
    """
    B = obs_seq.shape[0]
    all_actions = []
    for _ in range(num_styles):
        z = torch.randn(B, model.latent_dim, device=obs_seq.device)
        actions, _ = model(obs_seq, action_seq=None, z=z)
        all_actions.append(actions)
    return torch.stack(all_actions, dim=0)  # (num_styles, B, chunk_size, action_dim)


# ============================================================
# 六、演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[ACT] 代码演示 — Action Chunking with Transformers")
    print("=" * 60)
    
    batch_size = 2
    obs_dim = 512       # ResNet 编码输出
    action_dim = 14     # 双臂：每臂 7 维 (x,y,z,rx,ry,rz,grip)
    chunk_size = 100    # 一次预测 100 步（~2秒@50Hz）
    num_frames = 4      # 4 个相机视角（顶部+正面+双腕）
    
    model = ACT(obs_dim, action_dim, chunk_size)
    # 伪造观测序列
    obs_seq = torch.randn(batch_size, num_frames, obs_dim)
    action_seq = torch.randn(batch_size, chunk_size, action_dim)
    
    # --- 训练 ---
    print("\n1. 训练模式（使用 CVAE Encoder）")
    total_loss, mse_loss, kl_loss = act_loss(model, obs_seq, action_seq)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   MSE Loss:   {mse_loss.item():.4f}")
    print(f"   KL Loss:    {kl_loss.item():.4f}")
    
    # --- 推理 ---
    print("\n2. 推理模式（从 N(0,I) 采样 z）")
    actions = act_inference(model, obs_seq, num_styles=3)
    print(f"   3 种风格的动作形状: {actions.shape}")
    for i in range(3):
        print(f"   风格 {i}: mean={actions[i].mean().item():.4f}, std={actions[i].std().item():.4f}")
    
    # --- Temporal Ensembling 演示 ---
    print("\n3. Temporal Ensembling 演示")
    chunks = [torch.randn(chunk_size, action_dim) for _ in range(5)]
    for step in range(chunk_size):
        # 模拟执行：每步取前几个预测的加权平均
        available = [c[step:] for c in chunks if step < len(c)]
        if available:
            ensembled = temporal_ensemble(available, step, chunk_size)
            if step < 3:  # 只打印前几步
                print(f"   Step {step}: 使用 {len(available)} 个预测，加权平均 norm={torch.norm(ensembled).item():.3f}")
    
    # --- 模型统计 ---
    print(f"\n参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print("\n关键设计要点:")
    print("  - Action Chunking: 预测 100 步，减少组合误差 100×")
    print("  - CVAE 隐变量: 捕获演示者的动作风格差异")
    print("  - KL 正则化: 保证推理时的 z 采样与训练时一致")
    print("  - Temporal Ensembling: 重叠块加权平均，平滑动作过渡")
    print("  - 硬件友好: 仅需 2-6GB VRAM（4070 Ti Super 完美支持）")
    print("\n参考: [[ACT]] Section 2, ALOHA 硬件参考 [[SO-100 arm]]")
```

## 设计说明

- **Action Chunking**：一次预测 k=100 步，组合误差减少 100 倍
- **CVAE + KL 正则**：z 捕获风格多样性，训练/推理分布对齐
- **Temporal Ensembling**：指数衰减权重平滑相邻动作块之间的过渡
- **硬件友好**：仅 2-6GB VRAM，4070 Ti Super 完美支持
- 对照 [[ACT]] Section 2.1-2.3，思想被 [[π0]] 直接继承
```
