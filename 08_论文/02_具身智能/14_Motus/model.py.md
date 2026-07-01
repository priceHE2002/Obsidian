---
tags: [代码, PyTorch]
created: 2026-07-01
---
# Motus - 代码实现
> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
Motus: A Unified Latent Action World Model

基于 [[Motus]] 论文实现。核心架构 MoT (Mixture-of-Transformer)：将理解专家（Qwen3-VL-2B）、
视频生成专家（Wan 2.2-5B）、动作专家（Transformer 300M）三个专家通过共享注意力层融合。
关键创新——潜在动作（Latent Action）：用光流压缩为 14 维潜在向量，让模型从海量无标签
人类视频中学习通用运动先验。

参考: [[Motus]] | CVPR 2026 | 清华TSAIL + 北大 + Horizon Robotics
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 人体关键点编码器
# ═══════════════════════════════════════════════════════════════════════════════

class HumanPoseEncoder(nn.Module):
    """
    将人体关键点（如 17+ 关键点）编码为结构化的姿态 embedding。

    WHY: Motus 需要从人类操作视频中学习运动先验。人体关键点是最紧凑的姿态表示——
    相比原始视频帧，关键点数量级更小（17-21 个点 vs 224x224 像素），
    且天然具有语义（"手腕关节" vs "肘关节"），模型更容易学习关节间的运动协调。

    输入: (B, T, num_keypoints, 3) — 每帧的关键点 (x, y, confidence)
    输出: (B, T, d_model) — 时序姿态特征
    """

    def __init__(
        self,
        num_keypoints: int = 17,    # COCO 17 关键点格式
        d_model: int = 512,
        num_layers: int = 3,
        nhead: int = 8,
    ):
        super().__init__()
        # 将每个关键点 (x, y, c) 投影为 d_model
        self.kp_proj = nn.Linear(3, d_model)

        # 关键点间的空间注意力 —— WHY: 学习关节间的空间关系（如手腕相对肘部位置）
        self.spatial_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, batch_first=True)
            for _ in range(num_layers)
        ])

        # 时序 Transformer —— WHY: 学习运动动态（关键点如何随时间变化）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 可学习的全局姿态 token —— 类似 ViT 的 CLS token
        self.pose_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, kp_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kp_seq: (B, T, K, 3) — 关键点序列
        Returns:
            pose_feat: (B, T, d_model) — 时序姿态特征
        """
        B, T, K, _ = kp_seq.shape

        # 空间编码: 每帧独立处理（B*T 个独立的关键点图）
        kp_flat = kp_seq.view(B * T, K, 3)
        x = self.kp_proj(kp_flat)  # (B*T, K, D)

        for attn in self.spatial_attn:
            x = x + attn(x, x, x)[0]

        # 全局姿态 token 聚合——WHY: 类似 ViT 的 CLS，聚合所有关键点信息
        pose_token = self.pose_token.expand(B * T, -1, -1)
        x_pooled = torch.cat([pose_token, x], dim=1)
        x_pooled = x_pooled[:, 0, :]  # 取 pose_token 输出
        x_spatial = x_pooled.view(B, T, -1)

        # 时序编码——WHY: 捕捉关键点的运动模式和时序依赖性
        x_temporal = self.temporal_encoder(x_spatial)
        return x_temporal


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 光流 -> 潜在动作 压缩器
# ═══════════════════════════════════════════════════════════════════════════════

class OpticalFlowLatentCompressor(nn.Module):
    """
    将高维光流图压缩为 14 维潜在动作向量。

    WHY: Motus 的核心创新——互联网视频没有动作标签，但有光流（像素级运动信息）。
    通过将光流压缩为极低维（14 维）的"潜在动作"，模型可以:
    1. 从海量无标签人类视频学习通用运动先验
    2. 用少量机器人数据将 14 维潜在动作与真实动作空间对齐（只需学一个映射）

    维度选择 14 的原因: 足够捕捉复杂运动模式（如 7-DoF 末端位姿 x 2 臂），
    同时足够小以保证压缩是有意义的（不是简单的恒等映射）。
    """

    def __init__(self, latent_dim: int = 14, input_channels: int = 2):
        """
        Args:
            latent_dim: 潜在动作维度（论文中为 14）
            input_channels: 光流通道数（水平+垂直 = 2）
        """
        super().__init__()
        # 编码器: 光流 -> 低维潜在表示
        # WHY: 使用步长卷积下采样，逐步压缩空间信息，保留运动模式
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),  # H/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),               # H/4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),              # H/8
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),             # H/16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # 投影到 latent_dim
        self.latent_head = nn.Linear(256, latent_dim)

        # 解码器（用于重建光流，确保 latent 保留了运动信息）
        # WHY: 重建损失确保压缩是有信息量的——如果 latent 是 trivial 的，重建会很差
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
        )

    def forward(self, optical_flow: torch.Tensor, return_recon: bool = False):
        """
        Args:
            optical_flow: (B, 2, H, W) 光流图
        Returns:
            latent_action: (B, latent_dim)
            flow_recon: (B, 2, H, W) [可选] 重建的光流
        """
        h = self.encoder(optical_flow).squeeze(-1).squeeze(-1)  # (B, 256)
        latent = self.latent_head(h)                              # (B, 14)

        if return_recon:
            recon = self.decoder(latent)
            return latent, recon
        return latent


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 多模态融合层（Tri-model Joint Attention）
# ═══════════════════════════════════════════════════════════════════════════════

class TriModalFusion(nn.Module):
    """
    Motus MoT 架构的共享注意力融合层。

    WHY: 三个专家（理解、视频生成、动作）不是独立运行——
    在共享的交叉注意力层中交换信息，形成"理解->想象->执行"的闭环:
    - 理解专家提供场景语义（"这是什么物体"）
    - 视频生成专家推演未来（"如果我这样做会看到什么"）
    - 动作专家接收两者信息决定下一步动作
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()

        # Cross-attention: 理解 + 视频 -> 动作
        # WHY: 动作专家需要知道"场景里有什么"（理解）和"可能发生什么"（视频），
        # 才能做出最优决策
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Self-attention: 动作内部的自注意
        # WHY: 动作 token 之间也有时序依赖（当前动作影响下一步可执行的动作）
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # LayerNorm（Pre-LN 风格，训练更稳定）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        action_feat: torch.Tensor,      # (B, N_act, D)
        context_feat: torch.Tensor,     # (B, N_ctx, D) 来自理解+视频专家的融合
    ) -> torch.Tensor:
        """
        Args:
            action_feat: 动作专家当前的特征
            context_feat: 理解和视频专家的融合上下文
        Returns:
            增强后的动作特征
        """
        # 交叉注意力: 动作查询理解+视频上下文
        x = self.norm1(action_feat)
        cross_out = self.cross_attn(query=x, key=context_feat, value=context_feat)[0]
        action_feat = action_feat + cross_out

        # 自注意力
        x = self.norm2(action_feat)
        self_out = self.self_attn(query=x, key=x, value=x)[0]
        action_feat = action_feat + self_out

        # FFN
        x = self.norm3(action_feat)
        action_feat = action_feat + self.ffn(x)

        return action_feat


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Motus 主模型
# ═══════════════════════════════════════════════════════════════════════════════

class MotusModel(nn.Module):
    """
    Motus 大一统世界动作模型（简化版 MoT 架构）。

    WHY: 在简化实现中，我们将三个专家的核心功能合成为:
    - 视觉编码器：提取场景语义 + 运动特征（模拟理解+视频生成专家的输出）
    - 姿态编码器：从人体关键点提取运动先验
    - 融合层：Tri-model Joint Attention
    - 动作头：输出连续动作

    完整版 Motus 还包括视频生成输出和价值函数输出，
    但核心的潜在动作和多模态融合思想在此保留。
    """

    def __init__(
        self,
        img_feat_dim: int = 1024,       # 视觉编码器输出维度
        pose_dim: int = 512,
        d_model: int = 512,
        action_dim: int = 7,
        num_fusion_layers: int = 4,
        nhead: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # 各模态投影
        self.img_proj = nn.Linear(img_feat_dim, d_model)
        self.pose_proj = nn.Linear(pose_dim, d_model)
        self.proprio_proj = nn.Sequential(
            nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # 多模态融合层
        self.fusion_layers = nn.ModuleList([
            TriModalFusion(d_model, nhead) for _ in range(num_fusion_layers)
        ])

        # 动作输出头
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        # 潜在动作对齐映射 —— WHY: 将 14 维潜在动作映射到真实动作空间
        # 只用少量机器人数据即可完成对齐
        self.latent_to_action = nn.Linear(14, action_dim)

    def forward(
        self,
        img_feat: torch.Tensor,         # (B, N_img, img_feat_dim)
        pose_feat: torch.Tensor,        # (B, T_pose, pose_dim)
        proprio: torch.Tensor,          # (B, proprio_dim)
        latent_action: torch.Tensor = None,  # (B, 14) 可选的潜在动作
    ) -> dict:
        """
        Returns:
            dict with:
                - action: (B, action_dim) 预测动作
                - action_from_latent: (B, action_dim) 从潜在动作对齐的动作
        """
        B = img_feat.size(0)

        # 投影
        img_emb = self.img_proj(img_feat)       # (B, N_img, D)
        pose_emb = self.pose_proj(pose_feat)     # (B, T_pose, D)
        prop_emb = self.proprio_proj(proprio).unsqueeze(1)  # (B, 1, D)

        # 构建上下文特征（理解+视频专家的融合输出）
        context = torch.cat([img_emb, pose_emb, prop_emb], dim=1)  # (B, C, D)

        # 动作 token 从本体感觉初始化
        # WHY: 初始动作查询包含当前关节状态信息，有利于快速收敛
        action_tokens = prop_emb

        # 通过融合层
        for fusion in self.fusion_layers:
            action_tokens = fusion(action_tokens, context)

        # 输出动作
        action = self.action_head(action_tokens.squeeze(1))  # (B, action_dim)

        output = {'action': action}

        if latent_action is not None:
            output['action_from_latent'] = self.latent_to_action(latent_action)

        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 5. __main__ 演示
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Motus: Unified Latent Action World Model")
    print("=" * 60)

    B = 4

    # 人体姿态编码器演示
    print("\n1. 人体关键点编码")
    pose_encoder = HumanPoseEncoder(num_keypoints=17, d_model=512)
    kp_seq = torch.randn(B, 16, 17, 3)  # 16 帧，17 关键点，(x,y,c)
    pose_feat = pose_encoder(kp_seq)
    print(f"   输入: {kp_seq.shape} (B,T,K,3)")
    print(f"   输出: {pose_feat.shape} (B,T,D)")
    print(f"   K=17 关键点 -> D=512 姿态特征")

    # 光流压缩演示
    print("\n2. 光流潜在动作压缩")
    compressor = OpticalFlowLatentCompressor(latent_dim=14)
    flow = torch.randn(B, 2, 128, 128)  # 模拟光流
    latent, recon = compressor(flow, return_recon=True)
    print(f"   光流输入: {flow.shape} -> latent {latent.shape}")
    print(f"   重建光流: {recon.shape}")
    compress_ratio = flow.numel() // latent.numel()
    print(f"   压缩比: {compress_ratio}x")
    print(f"   WHY: 14 维 latent 从海量无标签视频中学习通用运动先验")

    # Motus 主模型演示
    print("\n3. Motus 多模态融合策略")
    model = MotusModel(img_feat_dim=1024, pose_dim=512, action_dim=7)
    img_feat = torch.randn(B, 8, 1024)   # 8 个图像 token
    proprio = torch.randn(B, 7)           # 本体感觉
    out = model(img_feat, pose_feat, proprio, latent)
    print(f"   动作预测: {out['action'].shape}")
    print(f"   潜在动作对齐: {out['action_from_latent'].shape}")
    print(f"   总参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    print("\n关键设计要点:")
    print("  - MoT 架构: 理解专家 + 视频生成专家 + 动作专家融合")
    print("  - 潜在动作: 光流 -> 14 维 latent -> 动作空间对齐")
    print("  - 利用无标签人类视频学习运动先验（数据效率 13.55x vs pi0.5）")
    print("  - RoboTwin 2.0 上 88%，比 pi0.5 高 45%")
    print("\n参考: [[Motus]] | CVPR 2026")
```
