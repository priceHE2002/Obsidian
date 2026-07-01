---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# HybridVLA: Collaborative Diffusion and Autoregression in a Unified VLA - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# HybridVLA: Collaborative Diffusion and Autoregression in a Unified VLA - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
HybridVLA: Collaborative Diffusion and Autoregression in a Unified VLA

基于 [[HybridVLA]] 论文实现。核心创新：在同一 LLM 内部同时进行扩散去噪和自回归生成，
不添加外部扩散头。通过特殊 token <BOD>/<EOD> 标记扩散区间，联合优化两种损失。
推理时根据自回归 token 置信度自适应融合两种模式——需要语义理解时信自回归，
需要精确控制时信扩散。

参考: [[HybridVLA]] | ICLR 2026 Poster | 北大 + 智源BAAI + 港中文CUHK
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 扩散动作 Head（嵌入 LLM 内部）
# ═══════════════════════════════════════════════════════════════════════════════

class DiffusionActionHead(nn.Module):
    """
    在 LLM hidden state 上直接进行扩散去噪，生成连续动作。

    WHY: HybridVLA 的关键设计——不在 LLM 外部加独立的 Diffusion Transformer，
    而是直接在 LLM 的 hidden state 上做去噪。这确保了:
    1. LLM 的语义理解能力直接影响去噪过程
    2. 不需要额外的大规模参数（没有单独的 DiT）
    3. 扩散和自回归两种模式共享相同的 LLM 骨干
    """

    def __init__(self, d_model: int = 4096, action_dim: int = 7, num_denoising_steps: int = 10):
        super().__init__()
        self.num_denoising_steps = num_denoising_steps
        self.action_dim = action_dim

        # 噪声预测网络 —— WHY: 轻量 MLP，因为 LLM hidden state 已经足够丰富
        self.noise_pred = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, action_dim),
        )

        # 时间步嵌入 —— WHY: 让去噪网络知道当前处于扩散过程的哪个阶段
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, seq_len, d_model) LLM 的 hidden states
        noisy_actions: torch.Tensor,  # (B, action_dim) 当前带噪动作
        timestep: torch.Tensor,       # (B,) 扩散时间步 [0, 1]
    ) -> torch.Tensor:
        """
        预测添加到 noisy_actions 上的噪声。

        Returns:
            noise_pred: (B, action_dim) 预测的噪声
        """
        # 聚合序列信息 → 条件向量
        # WHY: 取最后一个 diffusion token 位置的 hidden state，
        # 它已经通过 LLM 的自注意力聚合了全部观察信息
        if hidden_states.dim() == 3:
            cond = hidden_states[:, -1, :]  # (B, d_model)
        else:
            cond = hidden_states

        # 时间嵌入调制
        t_emb = self.time_embed(timestep.unsqueeze(-1).float())  # (B, d_model)
        cond = cond + t_emb

        # 噪声预测
        noise = self.noise_pred(cond)
        return noise

    @torch.no_grad()
    def denoise(
        self,
        hidden_states: torch.Tensor,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        从纯噪声开始逐步去噪，生成连续动作。

        WHY: 推理时的去噪过程。从 x_T ~ N(0,I) 开始，逐步去噪得到 x_0。
        与标准 DDPM 的区别是每一步都通过 LLM hidden state 进行条件化。

        这是 HybridVLA 的"扩散模式"——输出连续动作而非离散 token。
        """
        if num_steps is None:
            num_steps = self.num_denoising_steps

        B = hidden_states.size(0)
        # 从噪声开始
        x = torch.randn(B, self.action_dim, device=hidden_states.device)
        dt = 1.0 / num_steps

        for i in range(num_steps - 1, -1, -1):
            t = torch.full((B,), i * dt, device=hidden_states.device)
            noise_pred = self.forward(hidden_states, x, t)
            # 简单 Euler 步（Flow Matching 去噪的一阶近似）
            x = x - dt * noise_pred

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 混合动作 Head（联合扩散 + 自回归）
# ═══════════════════════════════════════════════════════════════════════════════

class HybridActionHead(nn.Module):
    """
    统一的混合动作 Head —— 同时支持扩散和自回归两种模式。

    WHY: 不在 LLM 外部加独立扩散头，而是让 LLM 本身同时学会扩散和自回归。
    两种模式的损失在同一优化过程中联合训练，共享全部 LLM 参数。
    这确保了扩散能从 LLM 的语义理解中获益（而非仅仅是"特征提取"）。
    """

    def __init__(
        self,
        d_model: int = 4096,
        action_dim: int = 7,
        num_action_bins: int = 256,     # 自回归模式的离散化 bin 数
        num_denoising_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = action_dim

        # 扩散组件 —— WHY: 连续动作生成，无离散化损失
        self.diffusion_head = DiffusionActionHead(d_model, action_dim, num_denoising_steps)

        # 自回归组件 —— WHY: 离散 token 预测（充分利用 LLM 的 next-token 能力）
        self.ar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_action_bins * action_dim),
        )
        self.num_action_bins = num_action_bins

        # 特殊 token 嵌入
        # WHY: <BOD> 和 <EOD> 标记告诉模型"接下来进入扩散模式"，
        # 类似于编程语言中的 begin/end 块
        self.bod_embed = nn.Parameter(torch.randn(d_model) * 0.02)  # Begin of Diffusion
        self.eod_embed = nn.Parameter(torch.randn(d_model) * 0.02)  # End of Diffusion

    def forward_diffusion(
        self,
        hidden_states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """扩散模式前向: 预测噪声（用于训练 loss）"""
        return self.diffusion_head(hidden_states, noisy_actions, timestep)

    def forward_ar(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        自回归模式前向: 预测离散 action token 的 logits。

        Returns:
            logits: (B, action_dim, num_bins) 每个动作维度的 bin logits
        """
        if hidden_states.dim() == 3:
            h = hidden_states[:, -1, :]
        else:
            h = hidden_states
        logits = self.ar_head(h)  # (B, action_dim * num_bins)
        return logits.view(-1, self.action_dim, self.num_action_bins)

    def bin_to_continuous(self, bin_ids: torch.LongTensor, action_min: float = -1.0, action_max: float = 1.0) -> torch.Tensor:
        """
        将离散 bin ID 转回连续值。

        WHY: 自回归模式输出的是离散 token，需要映射回连续动作空间
        才能与扩散模式的动作进行融合。
        """
        bins = torch.linspace(action_min, action_max, self.num_action_bins,
                              device=bin_ids.device)
        return bins[bin_ids]


class HybridVLA(nn.Module):
    """
    HybridVLA 完整模型（简化版，以 HybridActionHead 为核心）。

    WHY: 真实 HybridVLA 使用 LLaMA-2 7B 或 Phi-2 2.7B 作为骨干。
    这里将 LLM 抽象为 Transformer encoder，聚焦于关键的 Hybrid Head 设计。

    训练流程:
    1. 标准自回归模式: [obs] → [action_token_1] → [action_token_2] → ...
    2. 扩散模式: [obs] → <BOD> → [noise_1, ..., noise_N] → <EOD>
    两种模式的损失联合优化。
    """

    def __init__(
        self,
        d_model: int = 4096,
        nhead: int = 32,
        num_layers: int = 12,
        action_dim: int = 7,
        num_action_bins: int = 256,
        num_denoising_steps: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # —— 简化的 Transformer 骨干（替代 LLM）——
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # —— 混合动作 Head ——
        self.hybrid_head = HybridActionHead(d_model, action_dim, num_action_bins, num_denoising_steps)

        # —— 输入投影 ——
        self.obs_proj = nn.Linear(1024, d_model)     # 视觉特征 → d_model
        self.proprio_proj = nn.Linear(7, d_model)    # 本体感觉 → d_model

    def forward(
        self,
        obs_feat: torch.Tensor,            # (B, N_obs, 1024)
        proprio: torch.Tensor,             # (B, proprio_dim)
        noisy_actions: torch.Tensor = None,  # (B, action_dim) 扩散模式
        timestep: torch.Tensor = None,       # (B,) 扩散时间步
        mode: str = 'ar',                    # 'ar' 或 'diffusion' 或 'both'
    ) -> dict:
        B = obs_feat.size(0)

        # 投影观察
        obs_emb = self.obs_proj(obs_feat)
        prop_emb = self.proprio_proj(proprio).unsqueeze(1)

        # 编码
        seq = torch.cat([obs_emb, prop_emb], dim=1)
        hidden = self.backbone(seq)  # (B, seq_len, d_model)

        output = {}

        if mode in ('ar', 'both'):
            output['ar_logits'] = self.hybrid_head.forward_ar(hidden)

        if mode in ('diffusion', 'both') and noisy_actions is not None:
            output['noise_pred'] = self.hybrid_head.forward_diffusion(
                hidden, noisy_actions, timestep
            )

        if mode == 'both' and noisy_actions is not None:
            # 训练时同时计算两种 loss
            output['hidden_for_fusion'] = hidden

        return output


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 自适应融合器（推理时）
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFusion:
    """
    推理时自适应融合扩散和自回归的动作预测。

    WHY: 论文发现不同任务类型适合不同的模式——
    语义密集型任務（"把红色瓶子放进标注'3'的抽屉"）自回归模式置信度高，
    精度密集型任務（"把针插入针孔"）扩散模式更可靠。
    自适应融合让模型自动根据当前情境在两种模式间切换。
    """

    @staticmethod
    def fuse(
        ar_logits: torch.Tensor,        # (B, action_dim, num_bins)
        diff_action: torch.Tensor,       # (B, action_dim)
        num_bins: int = 256,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据自回归置信度自适应融合。

        Returns:
            fused_action: (B, action_dim) 融合后的动作
            alpha: (B,) 融合权重（接近 1 表示更信自回归）
        """
        # 自回归的置信度（softmax 的最大概率）
        ar_probs = F.softmax(ar_logits / temperature, dim=-1)
        confidence = ar_probs.max(dim=-1).values.mean(dim=-1)  # (B,)

        # α = f(confidence): 高置信度 → α 大 → 侧重自回归
        alpha = torch.sigmoid((confidence - 0.5) * 10.0)  # (B,)

        # 自回归生成连续动作
        ar_actions = torch.sum(
            ar_probs * torch.linspace(-1, 1, num_bins,
                                       device=ar_logits.device).view(1, 1, -1),
            dim=-1
        )  # (B, action_dim)

        # 融合
        alpha_expanded = alpha.unsqueeze(-1)  # (B, 1)
        fused = alpha_expanded * ar_actions + (1 - alpha_expanded) * diff_action

        return fused, alpha


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 混合损失函数
# ═══════════════════════════════════════════════════════════════════════════════

class HybridLoss(nn.Module):
    """
    L_hybrid = L_diff + L_ce

    WHY: 两种损失在同一优化过程中联合优化共享的 LLM 参数。
    自回归的交叉熵损失负责语义理解学习，
    扩散的 MSE 损失负责连续动作精度学习。
    两者互相增强而非竞争——这是 HybridVLA 的核心赌注。
    """

    def __init__(self, ce_weight: float = 1.0, diff_weight: float = 1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.diff_weight = diff_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        model_output: dict,
        action_target: torch.Tensor,     # (B, action_dim) 连续值目标
        action_bin_target: torch.LongTensor,  # (B, action_dim) 离散 bin ID
    ) -> dict:
        total = 0.0
        losses = {}

        if 'ar_logits' in model_output:
            # 自回归 CE 损失
            # ar_logits: (B, action_dim, num_bins)
            # WHY: 每个动作维度独立预测 bin，语义信息通过 LLM 共享
            ce = self.ce_loss(
                model_output['ar_logits'].reshape(-1, model_output['ar_logits'].size(-1)),
                action_bin_target.reshape(-1)
            )
            losses['ce_loss'] = self.ce_weight * ce
            total = total + losses['ce_loss']

        if 'noise_pred' in model_output:
            # 扩散 MSE 损失
            # WHY: 标准 Flow Matching / DDPM 噪声预测损失
            mse = F.mse_loss(model_output['noise_pred'], action_target)
            losses['diff_loss'] = self.diff_weight * mse
            total = total + losses['diff_loss']

        losses['total'] = total
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 演示
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("HybridVLA: Collaborative Diffusion + Autoregression")
    print("=" * 60)

    B, D, A = 4, 512, 7  # 用较小的 d_model 演示

    # 模型
    model = HybridVLA(d_model=D, nhead=8, num_layers=6,
                      action_dim=A, num_action_bins=256, num_denoising_steps=10)

    # 模拟输入
    obs = torch.randn(B, 8, 1024)      # 8 个视觉 token
    prop = torch.randn(B, 7)           # 本体感觉

    # 自回归模式
    out_ar = model(obs, prop, mode='ar')
    print(f"自回归模式 logits: {out_ar['ar_logits'].shape}")
    print(f"  期望: (B={B}, action_dim={A}, num_bins=256)")

    # 扩散模式
    noise_act = torch.randn(B, A)
    t = torch.rand(B)
    out_diff = model(obs, prop, noisy_actions=noise_act, timestep=t, mode='diffusion')
    print(f"\n扩散模式 noise_pred: {out_diff['noise_pred'].shape}")
    print(f"  期望: (B={B}, action_dim={A})")

    # Both 模式（训练时）
    out_both = model(obs, prop, noisy_actions=noise_act, timestep=t, mode='both')
    print(f"\nBoth 模式 keys: {list(out_both.keys())}")

    # 模拟推理融合
    diff_action = torch.randn(B, A)  # 模拟扩散去噪结果
    fused, alpha = AdaptiveFusion.fuse(out_both['ar_logits'], diff_action)
    print(f"\n自适应融合:")
    print(f"  融合动作 shape: {fused.shape}")
    print(f"  融合权重 α: {alpha.tolist()}")
    print(f"  α→1 表示更信自回归（语义密集型任务）")
    print(f"  α→0 表示更信扩散（精度密集型任务）")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params/1e6:.2f}M")

```

```
