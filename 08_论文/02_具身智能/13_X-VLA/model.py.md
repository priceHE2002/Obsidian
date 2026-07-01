---
tags: [代码, PyTorch]
created: 2026-07-01
---
# X-VLA - 代码实现
> 本文档包含 PyTorch/NumPy 教学实现。

```python
"""
X-VLA: Soft-Prompted Transformer for Scalable Cross-Embodiment VLA

基于 [[X-VLA]] 论文实现。核心思想：给每个数据源/机器人形态分配一组可学习的
Soft Prompt embedding（仅占 0.04% 额外参数），让 Transformer 自动学会编码该
数据源的"身份"——坐标系方向、控制模式、视角偏差等。

X-VLA 最关键的设计选择是不使用 DiT 的 AdaLN——直接用标准 Transformer decoder +
Flow Matching 达到 SOTA，证明架构复杂度不是关键，Soft Prompt 的条件化策略才是核心。

参考: [[X-VLA]] | ICLR 2026 | 清华AIR + 上海AI Lab + 北大
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Soft Prompt Embedding —— 跨形态条件化的核心
# ═══════════════════════════════════════════════════════════════════════════════

class SoftPromptEmbedding(nn.Module):
    """
    每个数据源/机器人形态对应一组可学习的 Soft Prompt embedding。

    WHY: NLP 中的 Soft Prompt Tuning 证明，少量可学习向量就能让冻住的 LLM 适应新任务。
    在 VLA 中，不同机器人有不同的动作空间（绝对/增量、关节/末端）、相机位姿、控制频率等。
    传统方案需要为每种形态设计不同的输入头，参数膨胀且不能泛化到新形态。

    Soft Prompt 的优雅之处：不需要为每种形态设计不同的输入格式，
    而是让模型自己学会"这个 prompt vector 代表什么意思"——
    比如某个维度学会了编码"这是增量控制模式"，另一个维度学会了"相机装在手腕上"。

    参数极少：K * d_model（通常 16 * 512 ≈ 8K），仅占模型总参数的 0.04%。
    """

    def __init__(self, num_sources: int, num_prompts: int = 16, d_model: int = 512):
        """
        Args:
            num_sources: 数据源数量（不同机器人 + 同一机器人的不同数据集）
            num_prompts: 每个数据源的 prompt token 数量 K（论文用 16-64）
            d_model: token embedding 维度

        注意：num_sources 可以随着新机器人加入而扩展——只需要新增 prompt embedding 行。
        """
        super().__init__()
        self.prompts = nn.Parameter(
            torch.randn(num_sources, num_prompts, d_model) * 0.02
        )
        self.d_model = d_model

    def forward(self, source_ids: torch.LongTensor):
        """
        Args:
            source_ids: (batch,) 每个样本所属的数据源索引
        Returns:
            prompt_tokens: (batch, num_prompts, d_model)
        """
        return self.prompts[source_ids]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 位置编码
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦位置编码。

    WHY: X-VLA 使用标准 Transformer（非 DiT），需要显式位置编码。
    正弦编码的周期特性让模型能泛化到训练时未见过的序列长度——
    这对于跨形态场景很重要，不同机器人的 token 序列长度不同。
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        """x: (B, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Flow Matching 条件流模型
# ═══════════════════════════════════════════════════════════════════════════════

class FlowMatchingScheduler:
    """
    Flow Matching 噪声调度器。

    Flow Matching 的核心公式:
        x_t = (1 - t) * x_0 + t * noise    # 线性插值路径
        v_target = noise - x_0             # 目标速度场（直线方向）

    WHY Flow Matching 而非 DDPM: Flow Matching 用直线路径连接数据和噪声，
    去噪步数可以大幅减少（通常 5-10 步即可），同时保持高质量。
    X-VLA、FLOWER、pi0 都选择了 Flow Matching。
    """

    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    def sample_t(self, batch_size: int, device: torch.device):
        """训练时随机采样时间步 t in [0, 1)"""
        return torch.rand(batch_size, device=device)

    def interpolate(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """线性插值路径: x_t = (1-t)*x0 + t*noise"""
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * noise

    def target_velocity(self, x0: torch.Tensor, noise: torch.Tensor):
        """目标速度场: v = noise - x0（从噪声指向数据的直线方向）"""
        return noise - x0

    @torch.no_grad()
    def denoise(self, model, source_ids, img_tokens, txt_tokens,
                proprio_tokens, action_dim: int, N_act: int, device: torch.device):
        """推理时从噪声逐步去噪生成动作。Euler 法积分。"""
        B = source_ids.size(0)
        x_t = torch.randn(B, N_act, action_dim, device=device)
        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.full((B,), 1.0 - step * dt, device=device)
            v_pred = model(source_ids, img_tokens, txt_tokens, proprio_tokens, x_t)
            x_t = x_t - v_pred * dt
        return x_t


# ═══════════════════════════════════════════════════════════════════════════════
# 4. X-VLA Transformer 主模型
# ═══════════════════════════════════════════════════════════════════════════════

class XVLATransformer(nn.Module):
    """
    X-VLA 的完整 Transformer 架构。

    WHY 不用 DiT: 论文明确指出，用标准 Transformer decoder blocks
    可以达到和 DiT 相同的性能，但架构更简单、更易训练。
    这表明"Soft Prompt 的条件化"比"复杂的架构设计"更重要。

    输入序列结构:
      [Prompt_Tokens | Image_Tokens | Text_Tokens | Proprio_Tokens | Noise_Action_Tokens]

    Prompt tokens 放在最前面是关键——因果注意力让所有后续 token
    都能"看到"形态条件，类似 NLP 中把 task instruction 放在最前面。
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        num_sources: int = 10,
        num_prompts: int = 16,
        img_token_dim: int = 768,
        text_token_dim: int = 512,
        proprio_dim: int = 7,
        action_dim: int = 7,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim

        # Soft Prompt 模块
        self.soft_prompt = SoftPromptEmbedding(num_sources, num_prompts, d_model)

        # 各模态投影层 —— WHY: 不同模态的维度不同，需要投影到统一的 d_model 空间
        self.img_proj = nn.Linear(img_token_dim, d_model)
        self.text_proj = nn.Linear(text_token_dim, d_model)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_embed = nn.Linear(action_dim, d_model)

        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len)

        # Transformer Decoder —— WHY: Decoder（因果注意力）而非 Encoder（双向注意力）
        # 因为动作 token 不应看到"未来"信息，且 decoder 自回归特性天然适合逐步去噪
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出投影
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def build_input_sequence(
        self,
        prompt_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        noise_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        拼接所有模态 token 为统一的输入序列。

        WHY: Prompt tokens 在最前面——因果注意力让后续所有 token
        都能看到形态条件，类似 NLP 中把 task instruction 放在最前面。
        """
        img_emb = self.img_proj(image_tokens)
        txt_emb = self.text_proj(text_tokens)
        prop_emb = self.proprio_proj(proprio_tokens)
        act_emb = self.action_embed(noise_actions)

        seq = torch.cat([prompt_tokens, img_emb, txt_emb, prop_emb, act_emb], dim=1)
        seq = self.pos_encoding(seq)
        return seq

    def forward(
        self,
        source_ids: torch.LongTensor,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        noise_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            预测的速度场 v_pred: (B, N_act, action_dim)
            Flow Matching 损失: ||v_pred - (noise - action_target)||^2
        """
        prompt_tokens = self.soft_prompt(source_ids)
        seq = self.build_input_sequence(
            prompt_tokens, image_tokens, text_tokens,
            proprio_tokens, noise_actions
        )

        out = self.transformer(tgt=seq, memory=None)

        # 只取 action 部分输出
        N_act = noise_actions.size(1)
        action_out = out[:, -N_act:, :]
        return self.action_head(action_out)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Prompt Warm-up —— 新机器人快速适配
# ═══════════════════════════════════════════════════════════════════════════════

class PromptWarmupAdapter:
    """
    两阶段新机器人适配策略。

    WHY 两阶段:
    1. Prompt Warm-up: 冻住全部 Transformer 参数，只为新机器人学习新 prompt tokens。
       只有 K*d_model 个参数需要学习，数据需求极低（几十条轨迹即可）。
    2. Joint Fine-tuning: 解冻最后几层 + prompt tokens 联合微调，进一步提升性能。

    对比: OpenVLA 需要 LoRA 微调几百 M 参数，而 X-VLA 只需学习几千个参数。
    """

    @staticmethod
    def freeze_transformer(model: XVLATransformer):
        """冻住 Transformer 主体，只留 prompt tokens 可训练"""
        for name, param in model.named_parameters():
            if 'soft_prompt' not in name:
                param.requires_grad = False

    @staticmethod
    def unfreeze_top_layers(model: XVLATransformer, num_layers: int = 3):
        """Joint Fine-tuning: 解冻最后 num_layers 层"""
        total_layers = model.transformer.num_layers
        for name, param in model.named_parameters():
            if 'soft_prompt' in name:
                continue
            for i in range(total_layers - num_layers, total_layers):
                if f'.layers.{i}.' in name or f'.layer.{i}.' in name:
                    param.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# 6. __main__ 演示
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("X-VLA: Cross-Embodiment Soft Prompt Transformer")
    print("=" * 60)

    B = 4
    device = torch.device("cpu")
    model = XVLATransformer(num_sources=7)

    # 模拟输入
    source_ids = torch.randint(0, 7, (B,))
    img_tokens = torch.randn(B, 32, 768)
    txt_tokens = torch.randn(B, 16, 512)
    prop_tokens = torch.randn(B, 4, 7)
    noise_act = torch.randn(B, 8, 7)

    # 前向传播
    v_pred = model(source_ids, img_tokens, txt_tokens, prop_tokens, noise_act)
    print(f"输入 source_ids: {source_ids.tolist()}")
    print(f"预测速度场 v_pred shape: {v_pred.shape}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    prompt_params = sum(p.numel() for p in model.soft_prompt.parameters())
    print(f"\n总参数量: {total_params/1e6:.1f}M")
    print(f"Soft Prompt 参数: {prompt_params} ({100*prompt_params/total_params:.3f}%)")

    # Prompt Warm-up 演示
    PromptWarmupAdapter.freeze_transformer(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPrompt Warm-up 后可训练参数: {trainable} ({100*trainable/total_params:.3f}%)")
    print("  说明: 仅需学习少量 prompt tokens 即可适配新机器人")

    # Flow Matching 去噪演示
    scheduler = FlowMatchingScheduler(num_steps=10)
    x_t = torch.randn(B, 8, 7, device=device)
    t = scheduler.sample_t(B, device)
    noise = torch.randn_like(x_t)
    x_t = scheduler.interpolate(x_t, noise, t)
    v_target = scheduler.target_velocity(x_t, noise)
    print(f"\nFlow Matching:")
    print(f"  时间步 t: [{t.min().item():.3f}, {t.max().item():.3f}]")
    print(f"  目标速度场 v_target shape: {v_target.shape}")
    print(f"  损失: MSE(v_pred, v_target)")

    print("\n关键设计要点:")
    print("  - Soft Prompt: 每数据源仅 K*d_model 参数（0.04%）编码形态身份")
    print("  - 标准 Transformer: 不用 DiT 的 AdaLN，架构极简")
    print("  - Prompt Warm-up: 冻住模型，只学 prompt tokens（几十条轨迹即可适配）")
    print("  - Flow Matching: 直线路径去噪（5-10 步），比 DDPM 快 10-20 倍")
    print("  - 0.9B 参数横扫 LIBERO (97-98%)，IROS 2025 世界冠军")
    print("\n参考: [[X-VLA]] | ICLR 2026")
```
