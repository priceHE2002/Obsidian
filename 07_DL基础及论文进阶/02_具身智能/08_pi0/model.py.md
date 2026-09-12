---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# pi0 模型实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
pi0 模型实现
============
Flow Matching 向量场预测 + DiT Action Expert + VLM+Action Expert 双系统架构

pi0 是第一个把 Flow Matching 引入 VLA 的模型，用"VLM 大脑 + Action Expert 小脑"
的双系统设计取代了传统 VLA 的"离散 token + 自回归"范式。50Hz 高频精控，
是当前公认最强的开源 VLA。

参考: [[π0]] | [[π0 原文.pdf]]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


# ═══════════════════════════════════════════════════════════════════
# Flow Matching 核心
# ═══════════════════════════════════════════════════════════════════

class FlowMatchingScheduler:
    """
    Flow Matching 调度器：定义从噪声到数据的确定性最优传输路径。

    WHY Flow Matching 而非 DDPM？
    - DDPM: 随机路径，每一步"走一步加一点噪声"，蜿蜒到达目标
    - Flow Matching: 确定性直线路径（最优传输），更少步数到达
    - 连续建模天然适合机器人连续动作空间，不需要离散化

    WHY 特别适合机器人？
    1. 动作空间是连续的（关节角度、末端位移）-> FM 输出连续向量
    2. 真实动作分布通常非高斯（多峰/非对称/尖峰）-> FM 理论上可建模任意分布
    3. 可产出 50Hz 的平滑轨迹 -> 对高频控制友好
    """

    def __init__(self, num_inference_steps: int = 10, sigma_min: float = 0.001):
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min

    def get_linear_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """最优传输路径: x_t = (1-t)*x_0 + t*x_1"""
        return t, 1.0 - t

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """从 [0, 1] 均匀采样时间步。"""
        return torch.rand(batch_size, device=device)

    def add_noise(
        self, x_1: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        线性插值: x_t = (1-t)*noise + t*x_1
        t=0 -> x_t 是纯噪声
        t=1 -> x_t 是真实数据
        """
        if noise is None:
            noise = torch.randn_like(x_1)
        t = t.view(-1, *([1] * (x_1.dim() - 1)))
        return (1.0 - t) * noise + t * x_1

    def flow_loss(
        self,
        model_output: torch.Tensor,  # 向量场预测 v_theta(x_t, t)
        x_1: torch.Tensor,           # 真实动作
        x_0: torch.Tensor,           # 噪声
        t: torch.Tensor,             # 时间步
    ) -> torch.Tensor:
        """
        Flow Matching 目标函数：
        L_FM = E[ ||v_theta(x_t, t) - (x_1 - x_0)||^2 ]

        WHY: 学习的是从噪声到数据的"直线方向"向量 (x_1 - x_0)，
        理论上沿此直线积分即可从噪声走向数据。
        """
        target = x_1 - x_0  # 直接方向向量
        return F.mse_loss(model_output, target)

    @torch.no_grad()
    def sample_euler(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        context: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Euler 法采样：从 t=0 (噪声) 积分到 t=1 (动作)。

        WHY 欧拉法就够了？
        - Flow Matching 的路径是直线 -> 最简单的 ODE 求解器已足够
        - ~10 步迭代即可达到高质量生成
        - 每步只是 x_{t+dt} = x_t + v_theta(x_t, t) * dt
        """
        x = noise
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((x.shape[0],), step * dt, device=x.device)
            v = model(x, t, context)  # 向量场预测
            x = x + v * dt  # 欧拉步
        return x


# ═══════════════════════════════════════════════════════════════════
# DiT Action Expert (小脑)
# ═══════════════════════════════════════════════════════════════════

class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization + MLP Block。
    时间步 t 和语义 context 通过 AdaLN 调制每一层的 scale & shift。
    每个 block 可交叉注意力到 VLM 的语义 embedding。
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Self-Attention
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-Attention to VLM embeddings
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # AdaLN 参数（由 time + context conditioning 生成）
        self.ada_ln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),  # 6 = 2*(norm1 + norm2 + mlp_gate)
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,                    # conditioning vector [B, H]
        vlm_embeddings: Optional[torch.Tensor] = None,  # [B, T_vlm, H]
    ) -> torch.Tensor:
        # AdaLN modulation: scale, shift, gate * 3
        modulation = self.ada_ln_modulation(c)  # [B, 6H]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        # Self-Attention with AdaLN
        x_normed = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Cross-Attention to VLM（可选）
        # WHY: Action Expert 的每一层都可以"查询"VLM 对场景的理解
        # 这样 VLM 的语义知识直接影响动作生成的每一步
        if vlm_embeddings is not None:
            x_normed2 = self.norm2(x)
            cross_out, _ = self.cross_attn(x_normed2, vlm_embeddings, vlm_embeddings)
            x = x + cross_out

        # MLP with AdaLN
        x_normed = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_normed)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class DiTActionExpert(nn.Module):
    """
    DiT (Diffusion Transformer) Action Expert。

    WHY 需要独立的 Action Expert（而不是让 VLM 直接输出动作）？
    1. VLM 参数量巨大（3B+），逐 step 推理太慢
    2. VLM 的设计目标不是生成平滑动作轨迹——它擅长"理解"，不擅长"执行"
    3. 专门的小网络（300M）可以 50Hz 高频运行

    WHY DiT 架构？
    - Transformer 的全局注意力适合建模长程动作依赖
    - AdaLN 条件化让时间和语义信息精准调控每一层
    """

    def __init__(
        self,
        action_dim: int = 32,          # 动作维度（可含多步 action chunk）
        hidden_size: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        context_dim: int = 512,       # VLM embedding 维度
        prop_dim: int = 128,          # 本体感觉维度
        num_action_chunks: int = 50,  # 输出 50 步 action chunk (= 1秒 @ 50Hz)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.hidden_size = hidden_size

        # 噪声动作 embedding
        self.action_embed = nn.Linear(action_dim, hidden_size)
        # 动作步位置编码（1D RoPE 风格）
        self.step_embed = nn.Embedding(num_action_chunks, hidden_size)

        # 时间步 embedding
        time_embed_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_size),
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 本体感觉 projection
        self.prop_proj = nn.Linear(prop_dim, hidden_size)

        # VLM context projection
        self.context_proj = nn.Linear(context_dim, hidden_size)

        # 条件组合
        self.condition_proj = nn.Linear(time_embed_dim + hidden_size + hidden_size, hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList([
            AdaLNBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # 输出头：预测速度场
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, action_dim)

    def forward(
        self,
        x_t: torch.Tensor,                    # [B, num_chunks, action_dim] — 当前噪声动作
        t: torch.Tensor,                      # [B] — 时间步
        vlm_embeddings: torch.Tensor,         # [B, T_vlm, context_dim]
        proprio: torch.Tensor,                # [B, prop_dim]
    ) -> torch.Tensor:
        B, C, _ = x_t.shape

        # 动作 -> embedding + 位置编码
        x = self.action_embed(x_t) + self.step_embed(torch.arange(C, device=x_t.device))  # [B, C, H]

        # 时间 embedding
        t_emb = self.time_embed(t)  # [B, 4H]

        # 本体感觉
        p_emb = self.prop_proj(proprio)  # [B, H]

        # 条件向量
        cond = self.condition_proj(torch.cat([t_emb, p_emb], dim=-1))  # [B, H]

        # VLM context projection
        vlm_proj = self.context_proj(vlm_embeddings)  # [B, T_vlm, H]

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond, vlm_proj)

        x = self.final_norm(x)
        v = self.output_head(x)  # [B, C, action_dim] — 向量场预测
        return v


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码，将标量时间步映射为高维向量。"""
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
# VLM 编码器（大脑）
# ═══════════════════════════════════════════════════════════════════

class VLMEncoder(nn.Module):
    """
    VLM 编码器：处理图像和语言指令，输出语义 embedding。

    WHY 冻住 VLM 的大部分参数？
    - VLM 的语义知识不会被"稀释"（小学习率微调）
    - 可以灵活替换 VLM 骨干（PaliGemma -> 更强的新 VLM）
    - Action Expert 参数独立，架构不变
    """

    def __init__(
        self,
        vision_dim: int = 1152,    # PaliGemma 视觉特征维度
        text_vocab_size: int = 256000,
        hidden_size: int = 2048,   # PaliGemma 3B 的维度
        num_layers: int = 24,
        num_heads: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # 视觉投影
        self.vision_proj = nn.Linear(vision_dim, hidden_size)

        # 文本 embedding
        self.text_embed = nn.Embedding(text_vocab_size, hidden_size)

        # VLM Transformer（简化）
        self.layers = nn.ModuleList([
            VLMTransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_size)

    def forward(
        self,
        vision_features: torch.Tensor,  # [B, N_patches, vision_dim]
        text_ids: torch.Tensor,         # [B, text_len]
    ) -> torch.Tensor:
        vision_embeds = self.vision_proj(vision_features)
        text_embeds = self.text_embed(text_ids)
        x = torch.cat([vision_embeds, text_embeds], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        return x


class VLMTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out, _ = self.attn(x_normed, x_normed, x_normed)
        x = x + attn_out
        x_normed = self.norm2(x)
        x = x + self.mlp(x_normed)
        return x


# ═══════════════════════════════════════════════════════════════════
# pi0 完整模型：双系统架构
# ═══════════════════════════════════════════════════════════════════

class Pi0(nn.Module):
    """
    pi0: VLM + Action Expert 双系统 VLA。

    完整流程：
    1. VLM 接收图像+语言指令 -> 输出语义 embedding（大脑，低频 1-5Hz）
    2. Action Expert 接收 VLM embedding + 本体感觉 -> Flow Matching 生成连续动作（小脑，50Hz）
    3. 预训练阶段：大量多样化数据 -> 学习"应对不确定性"
    4. 后训练阶段：少量高质量数据 -> 学习"精准执行"

    WHY 预训练/后训练分离（借鉴 LLM 的成功经验）？
    - 高质量数据里没有恢复行为——全是完美演示 -> 模型以为世界是确定的
    - 预训练混入不完美数据 -> 模型学会纠错和应对不确定性
    - 后训练在高质数据上精细化 -> 在知道"世界不确定"基础上学习最优行为
    """

    def __init__(
        self,
        vlm_hidden_size: int = 2048,
        action_dim: int = 32,
        expert_hidden_size: int = 512,
        expert_num_layers: int = 12,
        prop_dim: int = 128,
        num_action_chunks: int = 50,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.num_action_chunks = num_action_chunks
        self.num_inference_steps = num_inference_steps

        # 双系统
        self.vlm = VLMEncoder(hidden_size=vlm_hidden_size)
        self.action_expert = DiTActionExpert(
            action_dim=action_dim,
            hidden_size=expert_hidden_size,
            num_layers=expert_num_layers,
            context_dim=vlm_hidden_size,
            prop_dim=prop_dim,
            num_action_chunks=num_action_chunks,
        )

        # VLM -> Action Expert context 的投影（如果维度不同）
        self.vlm_to_expert = nn.Linear(vlm_hidden_size, expert_hidden_size) \
            if vlm_hidden_size != expert_hidden_size else nn.Identity()

        # Flow Matching 调度器
        self.scheduler = FlowMatchingScheduler(num_inference_steps=num_inference_steps)

        # 冻结/小学习率标记
        self._freeze_vlm_core()

    def _freeze_vlm_core(self):
        """
        WHY: VLM 的大部分参数保持冻结或极小学习率。
        - 防止语义知识被"稀释"
        - Action Expert 承担主要训练压力
        - 后训练阶段甚至可以完全冻结 VLM
        """
        for param in self.vlm.parameters():
            param.requires_grad = False
        # 只放开 vision_proj 用于对齐
        self.vlm.vision_proj.weight.requires_grad = True

    def forward(
        self,
        vision_features: torch.Tensor,    # [B, N_patches, vision_dim]
        text_ids: torch.Tensor,           # [B, text_len]
        proprio: torch.Tensor,            # [B, prop_dim]
        actions: Optional[torch.Tensor] = None,  # [B, num_chunks, action_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向：计算 Flow Matching 损失。
        """
        B = vision_features.shape[0]
        device = vision_features.device

        # Step 1: VLM 编码（大脑）
        vlm_output = self.vlm(vision_features, text_ids)  # [B, T_vlm, vlm_H]

        # Step 2: 采样噪声和时间步
        if actions is not None:
            noise = torch.randn_like(actions)  # [B, C, action_dim]
            t = self.scheduler.sample_time(B, device)
            x_t = self.scheduler.add_noise(actions, t, noise)

            # Step 3: Action Expert 预测向量场
            v_pred = self.action_expert(
                x_t, t,
                vlm_embeddings=self.vlm_to_expert(vlm_output),
                proprio=proprio,
            )

            # Step 4: Flow Matching 损失
            loss = self.scheduler.flow_loss(v_pred, actions, noise, t)

            return {"loss": loss, "vlm_output": vlm_output, "v_pred": v_pred}

        # 推理模式（无动作监督）
        return {"vlm_output": vlm_output}

    @torch.no_grad()
    def generate_actions(
        self,
        vision_features: torch.Tensor,
        text_ids: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理：Flow Matching 采样生成连续动作 chunk。

        WHY 可产出 50Hz 平滑轨迹？
        - ~10 步去噪即可 -> 每步只需一次 Action Expert 前向
        - Action Expert 只有 300M -> 前向很快
        - 不需要逐 token 自回归 -> 一次性生成整个 chunk
        """
        B = vision_features.shape[0]
        device = vision_features.device

        # VLM 编码（可以缓存！低频调用）
        vlm_output = self.vlm(vision_features, text_ids)

        # Flow Matching 采样
        noise = torch.randn(B, self.num_action_chunks, self.action_expert.action_dim, device=device)

        # 定义闭包给 scheduler
        class ModelWrapper(nn.Module):
            def __init__(self_, parent, vlm_emb, prop):
                super().__init__()
                self_.parent = parent
                self_.vlm_emb = vlm_emb
                self_.prop = prop

            def forward(self_, x_t, t, context):
                return self_.parent.action_expert(x_t, t, self_.vlm_emb, self_.prop)

        wrapped = ModelWrapper(self, self.vlm_to_expert(vlm_output), proprio)
        actions = self.scheduler.sample_euler(wrapped, noise, {})

        return actions


# ═══════════════════════════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("pi0 模型演示")
    print("=" * 60)

    # Flow Matching 调度器演示
    scheduler = FlowMatchingScheduler(num_inference_steps=10)
    print(f"\nFlow Matching: {scheduler.num_inference_steps} 步推理（确定性直线路径）")

    # 模拟 Flow Matching 过程
    t = scheduler.sample_time(4, torch.device("cpu"))
    x1 = torch.randn(4, 50, 32)    # 真实动作 chunk
    noise = torch.randn_like(x1)
    x_t = scheduler.add_noise(x1, t, noise)
    print(f"t=0.3 时的 x_t 形状: {x_t.shape}（线性插值: 70%噪声 + 30%数据）")

    # DiT Action Expert 演示
    expert = DiTActionExpert(
        action_dim=32, hidden_size=512, num_layers=6,
        num_heads=8, num_action_chunks=50,
    )
    x_t_input = torch.randn(2, 50, 32)
    t_input = torch.tensor([0.3, 0.7])
    vlm_emb = torch.randn(2, 256, 512)  # VLM context
    prop = torch.randn(2, 128)          # 本体感觉
    v_pred = expert(x_t_input, t_input, vlm_emb, prop)
    print(f"\nDiT Action Expert 向量场预测: {v_pred.shape} (B, 50步, 32维动作)")

    # pi0 完整模型
    model = Pi0(
        vlm_hidden_size=2048, action_dim=32,
        expert_hidden_size=512, expert_num_layers=6,
        prop_dim=128, num_action_chunks=50, num_inference_steps=10,
    )
    vision = torch.randn(2, 256, 1152)
    text = torch.randint(0, 256000, (2, 32))
    prop_input = torch.randn(2, 128)
    actions = torch.randn(2, 50, 32)

    # 训练
    output = model(vision, text, prop_input, actions)
    print(f"\n训练损失: {output['loss'].item():.4f}")

    # 推理
    generated = model.generate_actions(vision, text, prop_input)
    print(f"推理动作: {generated.shape} (一次性生成 50 步 x 32维)")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数: {total_params/1e6:.0f}M, 可训练: {trainable_params/1e6:.0f}M ({100*trainable_params/total_params:.1f}%)")

    print("\npi0 演示完成")

```
