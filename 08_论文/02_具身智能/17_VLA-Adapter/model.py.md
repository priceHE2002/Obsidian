---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# VLA-Adapter: An Effective Paradigm for Tiny-Scale VLA Model - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# VLA-Adapter: An Effective Paradigm for Tiny-Scale VLA Model - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
VLA-Adapter: An Effective Paradigm for Tiny-Scale VLA Model

基于 [[VLA-Adapter]] 论文实现。核心策略：冻住 VLM（Qwen2.5-0.5B），
不修改其内部参数。在 VLM 外部添加 Bridge Attention + 轻量 Policy Network，
将 VLM 的语义表征"翻译"为连续动作。可训练参数仅 ~4.7M，8 小时单 GPU 即可
达到 LIBERO 98.5% 成功率，超越 7B 的 OpenVLA-OFT。

参考: [[VLA-Adapter]] | arXiv 2025.9 | OpenHelix Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Houlsby Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class HoulsbyAdapter(nn.Module):
    """
    Houlsby Adapter: 在每个 Transformer 层后插入的轻量瓶颈模块。

    WHY: 经典的 Adapter 设计（Houlsby et al., 2019）——
    在每个 Transformer 层的自注意力和 FFN 之后各插入一个 bottleneck。
    先降维（d_model → bottleneck_dim）减少参数，再升维回来。
    这样在不修改原始 VLM 权重的前提下，学会下游任务特定的表征变换。

    适用场景: 需要每一层都做适配的任务（精细控制可能受益于多层 adapter）。
    """

    def __init__(self, d_model: int = 896, bottleneck_dim: int = 64, dropout: float = 0.1):
        """
        Args:
            d_model: VLM 的 hidden dimension（Qwen2.5-0.5B 为 896）
            bottleneck_dim: 瓶颈维度（越小参数越少）
        """
        super().__init__()
        # 降维 → 激活 → 升维
        # WHY: bottleneck 结构在 NLP PEFT 中被广泛验证——
        # 少量参数就能学到有意义的任务适配
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化: up 投影接近零，使得初始状态近似恒等映射
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(x))))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Pfeiffer Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class PfeifferAdapter(nn.Module):
    """
    Pfeiffer Adapter: 只在 FFN 后插入的轻量瓶颈模块。

    WHY: Pfeiffer Adapter 比 Houlsby 更轻量——
    只在 FFN 之后插入一个 adapter（而非每个子层后两个），
    参数减半，推理时 overhead 更小。许多 PEFT 研究发现
    Pfeiffer 在大多数任务上与 Houlsby 性能持平。

    VLA-Adapter 论文推荐：对外部 Policy Network 使用 Pfeiffer 风格，
    对 VLM 内部使用 Houlsby 风格（如果需要微调 VLM 的话）。
    """

    def __init__(self, d_model: int = 896, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up(self.act(self.down(x))))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bridge Attention —— 核心创新
# ═══════════════════════════════════════════════════════════════════════════════

class BridgeAttention(nn.Module):
    """
    VLA-Adapter 的核心创新：将 VLM 多层特征"桥接"到动作空间。

    WHY: VLA-Adapter 的关键洞察——
    "如何连接 VLM 表征和动作空间，比 VLM 有多大更重要。"

    Bridge Attention 解决的核心问题:
    - 从 VLM 的哪些层取特征？（中间层最优——语义丰富而不过度专业化）
    - 如何融合不同层的特征？（交叉注意力 + 可学习门控）
    - 如何将语义表征"翻译"为动作？（ActionQuery tokens 作为中间语言）

    三个子模块:
    1. Raw-to-Action 交叉注意力: VLM 中间层 hidden states → 动作空间
    2. ActionQuery-to-Action 交叉注意力: 深层 ActionQuery tokens → 动作空间
    3. 自注意力融合 + 可学习门控
    """

    def __init__(
        self,
        d_model: int = 896,       # VLM hidden dim
        action_dim: int = 7,
        num_action_queries: int = 8,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # —— Raw-to-Action 交叉注意力 ——
        # WHY: VLM 中间层的 raw hidden states 蕴含最丰富且通用的语义信息
        self.raw_to_action = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # —— ActionQuery ——
        # WHY: 可学习的 tokens，充当"中间语言"——在 VLM 表征和动作之间做翻译
        self.action_queries = nn.Parameter(
            torch.randn(1, num_action_queries, d_model) * 0.02
        )
        # ActionQuery 自注意力（学习 query 之间的协调关系）
        self.query_self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # —— ActionQuery-to-Action 交叉注意力 ——
        self.aq_to_action = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # —— 融合 ——
        # 自注意力融合层
        self.fusion_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 可学习门控参数
        # WHY: g 控制了注入多少"原始 VLM 特征" vs "ActionQuery 特征"
        # g 小 → 更多依赖 ActionQuery（任务特定）
        # g 大 → 更多依赖原始特征（更通用）
        # 这是 VLA-Adapter 最精妙的设计之一
        self.gate = nn.Parameter(torch.zeros(1))  # 初始化为 0，偏向 ActionQuery

        # 最终投影到动作空间
        self.action_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        raw_features: torch.Tensor,      # (B, N_raw, d_model) VLM 中间层 hidden states
        action_query_features: torch.Tensor,  # (B, N_aq, d_model) 深层 ActionQuery
    ) -> torch.Tensor:
        """
        Args:
            raw_features: 从 VLM 中间层提取的 hidden states
            action_query_features: 从 VLM 深层/最终层提取的特征
        Returns:
            action: (B, action_dim) 连续动作
        """
        B = raw_features.size(0)

        # 1. Raw-to-Action 交叉注意力
        # WHY: 让模型学会从 VLM 的中间层表征中"提取"动作相关信息
        raw_action = self.raw_to_action(
            query=raw_features.mean(dim=1, keepdim=True),  # 聚合为单个查询
            key=raw_features,
            value=raw_features,
        )[0]  # (B, 1, d_model)

        # 2. ActionQuery 处理
        aq = self.action_queries.expand(B, -1, -1)  # (B, K, d_model)
        aq = self.query_self_attn(aq, aq, aq)[0]    # 自注意

        # 交叉注意力: ActionQuery attend 到深层特征
        aq_action = self.aq_to_action(
            query=aq,
            key=action_query_features,
            value=action_query_features,
        )[0].mean(dim=1, keepdim=True)  # (B, 1, d_model) 聚合

        # 3. 可学习门控融合
        # WHY: tanh(g) 的取值范围 (-1, 1)，控制两种特征的混合比例
        gate_val = torch.tanh(self.gate)  # 标量
        fused = raw_action * gate_val + aq_action  # (B, 1, d_model)

        # 4. 自注意力融合
        fused = self.fusion_attn(fused, fused, fused)[0]
        fused = self.norm(fused.squeeze(1))

        # 5. 投影到动作空间
        return self.action_proj(fused)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VLA-Adapter 完整模型
# ═══════════════════════════════════════════════════════════════════════════════

class VLAAdapter(nn.Module):
    """
    VLA-Adapter: 冻住 VLM + Bridge Attention + 轻量 Policy Network。

    WHY: VLA-Adapter 的核心哲学——
    VLM 在互联网数据上学到的语义表征极其宝贵，不应该被"微调为动作预测器"污染。
    更好的方案是冻住 VLM，在外部加一个精巧的"翻译器"。

    架构: 冻住的 VLM → 多层特征提取 → Bridge Attention → 连续动作

    参数效率（论文数据）:
    - VLM: 0.5B（冻住）
    - Bridge + Policy: ~4.7M（可训练）
    - 训练 VRAM: ~9.6GB（batch=1，适合 16GB 显卡）
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 896,       # Qwen2.5-0.5B
        vlm_num_layers: int = 24,
        action_dim: int = 7,
        num_action_queries: int = 8,
        adapter_bottleneck: int = 64,
        use_vlm_adapters: bool = False,  # 是否在 VLM 内部加 Adapter
    ):
        super().__init__()
        self.vlm_hidden_dim = vlm_hidden_dim

        # —— VLM 内部的 Adapter（可选）——
        # WHY: 如果需要在 VLM 内部做轻量适配（而非完全冻住），
        # 可以插入 Houlsby/Pfeiffer Adapter。论文的 Pro 版使用了少量 VLM adapter。
        if use_vlm_adapters:
            self.vlm_adapters = nn.ModuleList([
                HoulsbyAdapter(vlm_hidden_dim, adapter_bottleneck)
                for _ in range(vlm_num_layers)
            ])
        else:
            self.vlm_adapters = None

        # —— Bridge Attention ——
        self.bridge = BridgeAttention(
            d_model=vlm_hidden_dim,
            action_dim=action_dim,
            num_action_queries=num_action_queries,
        )

        # —— 轻量 Policy Network ——
        # WHY: 在 Bridge Attention 之后加一个轻量 Transformer 做最终的动作精修。
        # 这是一个迷你 Transformer（2 层，远小于 VLM），专注于动作精炼。
        self.policy_net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=action_dim * 2,  # 动作 + 置信度特征
                nhead=2,
                dim_feedforward=256,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.policy_proj_in = nn.Linear(action_dim, action_dim * 2)
        self.policy_proj_out = nn.Linear(action_dim * 2, action_dim)

    def _simulate_vlm_forward(
        self,
        img_feat: torch.Tensor,   # (B, N_img, vlm_hidden_dim)
        text_feat: torch.Tensor,  # (B, N_txt, vlm_hidden_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        模拟 VLM 的前向传播。

        WHY: 真实 VLA-Adapter 使用 Qwen2.5-0.5B 作为冻住的 VLM。
        这里用简化的 Transformer 模拟，重点展示 Bridge Attention 的设计。
        """
        B = img_feat.size(0)

        # 模拟中间层输出（论文发现中间层特征最适合动作生成）
        # 在实际实现中，这来自 VLM 的 hidden states
        mid_features = img_feat + torch.randn_like(img_feat) * 0.02  # 模拟 VLM 中间层

        # 模拟深层 ActionQuery 特征
        # 在实际实现中，ActionQuery tokens 被附加到 VLM 输入序列的末尾
        deep_features = torch.cat([img_feat, text_feat], dim=1)

        return mid_features, deep_features

    def forward(
        self,
        img_feat: torch.Tensor,
        text_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            action: (B, action_dim)
        """
        # VLM 前向（在实际实现中，VLM 参数冻住）
        raw_features, aq_features = self._simulate_vlm_forward(img_feat, text_feat)

        # Bridge Attention
        action = self.bridge(raw_features, aq_features)  # (B, action_dim)

        # 轻量 Policy Network 精修
        # WHY: Bridge 输出后加一层轻量精修，类似"后处理"，
        # 确保输出动作在机器人动力学约束内
        x = self.policy_proj_in(action).unsqueeze(1)  # (B, 1, action_dim*2)
        x = self.policy_net(x)
        action = self.policy_proj_out(x.squeeze(1))

        return action

    def count_trainable_params(self) -> dict:
        """统计可训练 vs 总参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 演示
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("VLA-Adapter: Houlsby/Pfeiffer Adapter + Bridge Attention")
    print("=" * 60)

    B = 4

    # 演示 Houlsby Adapter
    adapter_h = HoulsbyAdapter(d_model=896, bottleneck_dim=64)
    x = torch.randn(B, 10, 896)
    y = adapter_h(x)
    print(f"Houlsby Adapter: {x.shape} → {y.shape}")
    print(f"  参数: {sum(p.numel() for p in adapter_h.parameters())/1e3:.1f}K")

    # 演示 Pfeiffer Adapter
    adapter_p = PfeifferAdapter(d_model=896, bottleneck_dim=64)
    y2 = adapter_p(x)
    print(f"Pfeiffer Adapter: {x.shape} → {y2.shape}")
    print(f"  参数: {sum(p.numel() for p in adapter_p.parameters())/1e3:.1f}K")

    # 演示 VLA-Adapter 完整模型
    model = VLAAdapter(
        vlm_hidden_dim=896,
        vlm_num_layers=24,
        action_dim=7,
        num_action_queries=8,
        use_vlm_adapters=False,  # 标准版: VLM 完全冻住
    )

    img_feat = torch.randn(B, 32, 896)  # VLM 编码的图像 token
    text_feat = torch.randn(B, 16, 896)  # 文本 token
    action = model(img_feat, text_feat)
    print(f"\nVLA-Adapter 动作输出: {action.shape}")

    params = model.count_trainable_params()
    print(f"\n参数统计:")
    print(f"  总参数:     {params['total']/1e6:.1f}M")
    print(f"  可训练:     {params['trainable']/1e6:.2f}M")
    print(f"  冻住(VLM):  {params['frozen']/1e6:.1f}M")
    print(f"  可训练占比: {100*params['trainable']/params['total']:.2f}%")
    print(f"  (论文: Bridge + Policy ~4.7M, VLM 0.5B 冻住)")

    # 模拟带 VLM Adapter 的 Pro 版
    model_pro = VLAAdapter(use_vlm_adapters=True)
    params_pro = model_pro.count_trainable_params()
    print(f"\nPro 版可训练参数: {params_pro['trainable']/1e6:.2f}M")

```

```
