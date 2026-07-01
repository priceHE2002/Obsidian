---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# FastWAM: Do World Action Models Need Test-time Future Imagination? - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# FastWAM: Do World Action Models Need Test-time Future Imagination? - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
FastWAM: Do World Action Models Need Test-time Future Imagination?

基于 [[FastWAM]] 论文实现。核心发现：WAM 的性能增益来自训练时的视频联合训练
（学到更好的世界表征），而非推理时的显式"想象"。因此推理时可以完全跳过
未来视频生成和迭代去噪，只需从 Video DiT 做单次前向提取 latent world representation，
然后用 Action DiT 直接解码动作。推理延迟从 >190ms 优化到 <90ms。

参考: [[FastWAM]] | arXiv 2026.3 | 清华MARS Lab + 星海图Galaxea AI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HyperNetwork 风格快速权重适应模块
# ═══════════════════════════════════════════════════════════════════════════════

class HyperNetworkAdapter(nn.Module):
    """
    HyperNetwork 为 Action DiT 的每一层动态生成条件化权重。

    WHY: Fast-WAM 推理时，Video DiT 的输出需要高效地条件化 Action DiT。
    HyperNetwork 将 Video DiT 的表征直接转化为 Action DiT 的参数调制，
    避免了在两个 DiT 之间做昂贵的交叉注意力，大幅降低推理延迟。

    设计思路: 参考 HyperNetwork 经典范式——
    用一个网络生成另一个网络的部分权重，实现快速条件适应。
    """

    def __init__(self, latent_dim: int = 512, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        # WHY: 两层 MLP 足够将 latent 映射为调制参数，更深会导致过拟合
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * 2),  # scale + shift
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: (B, D) 来自 Video DiT 的 latent world representation
        Returns:
            scale: (B, D_out) 通道级缩放
            shift: (B, D_out) 通道级偏移
        """
        params = self.net(latent)
        scale, shift = params.chunk(2, dim=-1)
        return scale, shift


class HyperAdaptedBlock(nn.Module):
    """
    被 HyperNetwork 调制的 Transformer Block。

    WHY: 标准 AdaLN（如 DiT）用时间步 t 生成 scale/shift。
    Fast-WAM 的创新是用 Video DiT 的输出（而非 t）来生成 scale/shift，
    使 Action DiT 能直接利用视频模型学到的物理世界表征。
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        scale_attn: torch.Tensor,
        shift_attn: torch.Tensor,
        scale_ffn: torch.Tensor,
        shift_ffn: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model)
            scale_attn/shift_attn: 来自 HyperNetwork 的注意力层调制
            scale_ffn/shift_ffn: 来自 HyperNetwork 的 FFN 层调制
        """
        # AdaLN 调制 + 自注意力
        # WHY: 不是简单加法，而是 scale+shift 调制——视频表征同时控制"幅度"和"偏向"
        x_mod = self.norm1(x) * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + self.attn(x_mod, x_mod, x_mod)[0]

        x_mod = self.norm2(x) * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        x = x + self.ffn(x_mod)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 世界表征提取器（Video DiT 的快速单次前向）
# ═══════════════════════════════════════════════════════════════════════════════

class WorldRepExtractor(nn.Module):
    """
    从观察（图像 + 本体感觉）中提取 latent world representation。

    WHY: Fast-WAM 的核心创新——推理时不生成未来视频帧。
    只需 Video DiT 做单次前向传播（而非多次去噪迭代），
    提取一个紧凑的 latent world representation，
    这个表征已经融合了训练时通过视频 co-training 学到的物理知识。
    """

    def __init__(self, img_feat_dim: int = 1024, proprio_dim: int = 7, latent_dim: int = 512):
        super().__init__()
        self.img_proj = nn.Linear(img_feat_dim, latent_dim)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # 融合层
        self.fusion = nn.MultiheadAttention(latent_dim, 8, batch_first=True)

    def forward(
        self,
        img_feat: torch.Tensor,   # (B, N_img, img_feat_dim)
        proprio: torch.Tensor,    # (B, proprio_dim)
    ) -> torch.Tensor:
        """
        Returns:
            world_repr: (B, latent_dim) 紧凑的世界表征
        """
        img_emb = self.img_proj(img_feat)         # (B, N_img, D)
        prop_emb = self.proprio_proj(proprio).unsqueeze(1)  # (B, 1, D)

        # 图像 attend 到本体感觉（两者融合）
        fused = self.fusion(query=prop_emb, key=img_emb, value=img_emb)[0]
        return fused.squeeze(1)  # (B, D)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Action DiT（HyperNetwork 调制版）
# ═══════════════════════════════════════════════════════════════════════════════

class HyperModulatedActionDiT(nn.Module):
    """
    被 HyperNetwork 逐层调制的 Action Diffusion Transformer。

    WHY: 每层都有独立的 HyperNetwork 生成 scale/shift，
    确保 Video DiT 的世界表征在不同抽象层次上都能条件化动作生成。
    这种设计比"只在第一层注入条件"更有效——
    深层特征往往需要不同的条件化方式。
    """

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        world_latent_dim: int = 512,
    ):
        super().__init__()
        self.action_dim = action_dim

        # 动作输入投影
        self.action_embed = nn.Linear(action_dim, d_model)

        # 每层配一个独立的 HyperNetwork
        # WHY: 不同层对"物理知识"的需求不同——
        # 浅层需要运动学信息，深层需要动力学信息
        self.hyper_networks = nn.ModuleList([
            nn.ModuleDict({
                'attn': HyperNetworkAdapter(world_latent_dim, d_model // 2, d_model),
                'ffn': HyperNetworkAdapter(world_latent_dim, d_model // 2, d_model),
            })
            for _ in range(num_layers)
        ])

        # Hyper-modulated blocks
        self.blocks = nn.ModuleList([
            HyperAdaptedBlock(d_model, nhead, d_model * 4)
            for _ in range(num_layers)
        ])

        # 输出头
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, action_dim)
        world_repr: torch.Tensor,       # (B, world_latent_dim)
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: 当前带噪动作
            world_repr: 来自 WorldRepExtractor 的世界表征
        Returns:
            预测的去噪后动作 (B, action_dim)
        """
        B = noisy_actions.size(0)
        x = self.action_embed(noisy_actions).unsqueeze(1)  # (B, 1, d_model)

        for hyper, block in zip(self.hyper_networks, self.blocks):
            scale_attn, shift_attn = hyper['attn'](world_repr)
            scale_ffn, shift_ffn = hyper['ffn'](world_repr)
            x = block(x, scale_attn, shift_attn, scale_ffn, shift_ffn)

        return self.action_head(x.squeeze(1))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FastWAM 完整模型
# ═══════════════════════════════════════════════════════════════════════════════

class FastWAM(nn.Module):
    """
    Fast-WAM：跳过推理时"想象"的世界动作模型。

    WHY: 完整 WAM 推理时需多步去噪生成未来视频帧 → 从中解码动作。
    Fast-WAM 的发现：训练时视频 co-training 已让世界表征足够好，
    推理时只需 Video DiT 单次前向 → HyperNetwork 调制 Action DiT → 直接输出动作。
    这使推理速度与纯 VLA 持平（<90ms vs 190ms+），同时保留 WAM 的物理理解优势。
    """

    def __init__(
        self,
        img_feat_dim: int = 1024,
        proprio_dim: int = 7,
        action_dim: int = 7,
        world_latent_dim: int = 512,
        d_model: int = 512,
        num_action_layers: int = 4,
    ):
        super().__init__()

        # 世界表征提取器（替代 Video DiT 的多步去噪）
        self.world_extractor = WorldRepExtractor(img_feat_dim, proprio_dim, world_latent_dim)

        # Action DiT（HyperNetwork 调制版）
        self.action_dit = HyperModulatedActionDiT(
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_action_layers,
            world_latent_dim=world_latent_dim,
        )

    def forward(
        self,
        img_feat: torch.Tensor,       # (B, N_img, img_feat_dim)
        proprio: torch.Tensor,        # (B, proprio_dim)
        noisy_actions: torch.Tensor,  # (B, action_dim)
    ) -> dict:
        """
        训练时前向——同时计算动作预测和世界表征。

        实际训练中还会加入视频预测的辅助损失（co-training），
        这里简化为主路径。
        """
        world_repr = self.world_extractor(img_feat, proprio)
        action_pred = self.action_dit(noisy_actions, world_repr)

        return {
            'action_pred': action_pred,
            'world_repr': world_repr,
        }

    @torch.no_grad()
    def fast_inference(
        self,
        img_feat: torch.Tensor,
        proprio: torch.Tensor,
        num_denoising_steps: int = 5,
    ) -> torch.Tensor:
        """
        快速推理：单次 Video DiT 前向 + 少量 Action DiT 去噪步。

        WHY: 相比于完整 WAM 的"先多次去噪生成视频帧 → 再多次去噪生成动作"，
        Fast-WAM 只需: 1 次世界表征提取 + N 步动作去噪。
        这就是速度从 >190ms 降至 <90ms 的原因。
        """
        world_repr = self.world_extractor(img_feat, proprio)
        B = img_feat.size(0)
        device = img_feat.device

        # Flow Matching 去噪（简化版）
        x = torch.randn(B, self.action_dit.action_dim, device=device)
        dt = 1.0 / num_denoising_steps

        for _ in range(num_denoising_steps):
            # 单步去噪
            v_pred = self.action_dit(x, world_repr)
            x = x - dt * v_pred  # Euler 步

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 视频 Co-training 辅助损失
# ═══════════════════════════════════════════════════════════════════════════════

class VideoCoTrainingLoss:
    """
    Fast-WAM 的核心发现——训练时视频 co-training 是关键。

    WHY: 论文做了精巧的消融实验:
    - 推理时生成未来帧 vs 跳过 → 无显著差异
    - 训练时移除视频 co-training → 性能崩溃（91.8% → 83.8%）

    这说明 WAM 的性能来自训练阶段通过视频预测学到的更好世界表征，
    而非推理时的显式想象。本模块实现训练时的视频预测辅助损失。
    """

    @staticmethod
    def compute_video_pred_loss(
        world_repr: torch.Tensor,       # (B, D) 世界表征
        future_frame_feat: torch.Tensor,  # (B, D) 真实未来帧特征
    ) -> torch.Tensor:
        """
        训练时强制世界表征能预测未来视觉状态。

        WHY: 视频预测作为辅助任务：
        1. 提供稠密的物理监督信号（物体不会凭空消失，运动会保持连续性）
        2. 让模型学到更好的视觉表征（不仅识别"这是什么"，还理解"它怎么动"）
        3. 隐式数据增强——预测未来要求理解物体的 3D 结构和物理属性
        """
        # 简单的对比/回归损失
        # 实际实现可能用 cosine similarity 或 MSE + 梯度停止
        return F.mse_loss(world_repr, future_frame_feat.detach())


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 演示
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("FastWAM: HyperNetwork-style Fast Weight Adaptation")
    print("=" * 60)

    B, A = 4, 7

    # 模型
    model = FastWAM(
        img_feat_dim=1024,
        proprio_dim=7,
        action_dim=A,
        world_latent_dim=512,
        d_model=512,
        num_action_layers=4,
    )

    # 模拟输入
    img_feat = torch.randn(B, 8, 1024)
    prop = torch.randn(B, 7)
    noise_act = torch.randn(B, A)

    # 训练前向
    out = model(img_feat, prop, noise_act)
    print(f"训练输出:")
    print(f"  action_pred:   {out['action_pred'].shape}")
    print(f"  world_repr:    {out['world_repr'].shape}")

    # 快速推理
    action = model.fast_inference(img_feat, prop, num_denoising_steps=5)
    print(f"\n快速推理:")
    print(f"  去噪后动作:    {action.shape}")
    print(f"  推理路径: 1次 Video DiT 前向 + 5步 Action 去噪")
    print(f"  (vs 完整 WAM: N次视频去噪 + M次动作去噪)")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params/1e6:.2f}M")

    # HyperNetwork 参数
    hn_params = sum(
        sum(p.numel() for p in hn.parameters())
        for hn_dict in model.action_dit.hyper_networks
        for hn in hn_dict.values()
    )
    print(f"HyperNetwork 参数: {hn_params/1e3:.1f}K")
    print(f"  说明: 极小的 HyperNetwork 实现了高效的条件化权重适应")

```

```
