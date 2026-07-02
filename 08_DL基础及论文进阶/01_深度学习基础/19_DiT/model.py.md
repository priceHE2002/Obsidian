---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DiT (Diffusion Transformer) 完整实现 - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
DiT (Diffusion Transformer) 完整实现
======================================
论文: "Scalable Diffusion Models with Transformers" (Peebles & Xie, ICCV 2023)
核心思想: 用纯 Transformer 替代 U-Net 作为扩散模型的骨干,
          通过 AdaLN (Adaptive Layer Normalization) 注入条件信息。

参考: [[DiT]], [[DDPM]], [[Flow Matching]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# 1. 辅助组件
# ============================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    AdaLN 调制操作: y = (1 + scale) * x + shift

    为什么用 (1 + scale) 而非直接 scale?
    - 初始 scale=0 → 行为等同恒等映射, 保证训练初期稳定
    - 与 DiT 官方实现一致
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    时间步嵌入: 将标量 t 转换为高维向量

    使用正弦位置编码风格 (类似 ViT 的位置编码), 将连续时间步
    映射为频率编码, 再通过 MLP 转换为条件向量。

    为什么需要频率编码?
    - 标量 t 的信息量太少, 直接 Feed 效果极差
    - 类似 NeRF 中的位置编码, 高频分量让 MLP 容易学习
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        # 将频率编码映射到 hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """生成正弦/余弦频率编码 (类似 Transformer 位置编码)"""
        half = dim // 2
        # 指数衰减频率: [1, 1/10000^(2/dim), 1/10000^(4/dim), ...]
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]  # [B, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """
    类别标签嵌入: 将类别 ID 映射为高维向量

    支持 classifier-free guidance:
    - 训练时随机将部分样本的标签替换为 null token
    - null token 是可学习的 token, 代表"无条件"生成

    为什么用 Embedding 而非 One-hot?
    - Embedding 可学习, 能编码类别间的语义关系
    - null token 参数与类别 token 共享同一 Embedding 表
    """
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.use_cfg_embedding = dropout_prob > 0
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + int(self.use_cfg_embedding), hidden_size)
        self.num_classes = num_classes

    def token_drop(self, labels: torch.Tensor, force_drop_ids=None):
        """
        随机丢弃标签 (classifier-free guidance 训练所需)

        被丢弃的标签替换为 num_classes 索引 (即 null token)
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool = False, force_drop_ids=None):
        if train:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


# ============================================================
# 2. DiT 核心组件
# ============================================================

class Mlp(nn.Module):
    """
    标准 Transformer MLP: Linear → SiLU → Linear

    与 ViT 的 MLP 设计一致, 不使用 SwiGLU (DiT 跟随 ViT 传统)
    """
    def __init__(self, in_features: int, hidden_features=None, out_features=None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()  # Swish / SiLU 激活
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-Head Self-Attention (标准 ViT Attention)

    为什么 DiT 使用 Self-Attention 而非 Cross-Attention 注入条件?
    - 实验证明 AdaLN (调节 LN 参数) 比 Cross-Attention 更高效
    - AdaLN 几乎不增加 Gflops, Cross-Attention 增加约 15%
    - 效果上 AdaLN 与 Cross-Attention 相当甚至更好
    """
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) 必须能被 num_heads ({num_heads}) 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # sqrt(d_k) 缩放

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        # QKV 投影 + reshape → [B, 3, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    DiT Block: AdaLN-Zero 条件 Transformer Block

    结构:
      x = x + α₁ · MultiHeadAttention(AdaLN₁(x, c))
      x = x + α₂ · MLP(AdaLN₂(x, c))

    其中 c = t_emb + c_emb (时间步 + 类别条件)
    AdaLN(x, c) = γ(c) · LayerNorm(x) + β(c)

    为什么 α 初始化为零?
    - 确保 Block 初始行为为恒等函数 (identity function)
    - 训练从"无条件"开始逐步学习条件调制 → 训练最稳定, 效果最好
    - 这是 DiT 论文中 AdaLN-Zero 相比普通 AdaLN 的关键改进
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)

        # AdaLN 调制参数由条件 c 通过一个小 MLP 回归得到
        # 输出 6 * hidden_size: γ₁,β₁,α₁,γ₂,β₂,α₂ 各 hidden_size 维
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        参数:
          x: [B, N, C] token 序列
          c: [B, C] 条件向量 (时间步 + 类别的融合 embedding)
        """
        # 从条件向量回归 6 组调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention with AdaLN
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # MLP with AdaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    DiT 输出层: AdaLN + Linear → 噪声/协方差预测

    输出维度为 patch_size² × out_channels:
    - 前半部分: 噪声预测 ε
    - 后半部分 (若 learn_sigma=True): 对角协方差预测 Σ

    为什么输出协方差?
    - ADM 论文 (Diffusion Models Beat GANs) 表明同时预测噪声和协方差
      可以提升 Log-likelihood
    - 协方差仅用于评估, 采样时只用噪声预测
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # AdaLN 调制参数 (零初始化)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ============================================================
# 3. Patchify: 将 2D Latent 特征图转换为 Token 序列
# ============================================================

class PatchEmbed(nn.Module):
    """
    Patch Embedding: 切分 2D 特征图为固定大小的 patch

    输入: [B, C, H, W]  (e.g. VAE latent 32×32×4)
    输出: [B, (H/p)*(W/p), hidden_size]

    为什么用 patchify 而非全像素?
    - 自注意力复杂度 O(N²), 逐像素处理 cost 太高
    - Patch 类比 ViT 的分词, 保留足够空间信息的同时控制 token 数
    - patch_size 是 DiT 设计空间中的独立缩放维度 (越小 → token 越多 → FID 越低)
    """
    def __init__(self, patch_size: int = 2, in_channels: int = 4, hidden_size: int = 1152):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, hidden_size, H/p, W/p]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, hidden_size], N = H * W
        return x


# ============================================================
# 4. DiT 主模型
# ============================================================

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 完整模型

    架构流程:
      1. PatchEmbed: latent z (32×32×4) → token 序列
      2. TimestepEmbedder: t → t_emb
      3. LabelEmbedder: class_id → c_emb
      4. t_emb + c_emb → 条件向量 c
      5. DiTBlock × N: 逐块用 AdaLN 注入条件信息
      6. FinalLayer: 输出噪声预测 + 协方差

    实验配置 (DiT 论文 Table 1):
      | 名称  | Layers | Hidden dim | Heads | 参数量  |
      |-------|--------|-----------|-------|---------|
      | DiT-S | 12     | 384       | 6     | ~33M    |
      | DiT-B | 12     | 768       | 12    | ~130M   |
      | DiT-L | 24     | 1024      | 16    | ~458M   |
      | DiT-XL| 28     | 1152      | 16    | ~675M   |

    关键设计决策:
    - AdaLN-Zero: 零初始化残差门控, 训练从恒等函数开始
    - elementwise_affine=False 在 LayerNorm: 因为 γ,β 由 AdaLN 动态提供
    - 无 dropout, 无 positional encoding (latent 本身隐含位置信息)
    """
    def __init__(
        self,
        input_size: int = 32,          # latent spatial size (VAE 下采样 8x)
        patch_size: int = 2,           # patch 大小 (p=2 → 256 tokens)
        in_channels: int = 4,          # VAE latent 通道数
        hidden_size: int = 1152,       # Transformer 隐藏维度
        depth: int = 28,               # DiT Block 层数
        num_heads: int = 16,           # 注意力头数
        mlp_ratio: float = 4.0,        # MLP 隐藏层扩张比
        num_classes: int = 1000,       # ImageNet 类别数
        class_dropout_prob: float = 0.1, # CFG 训练时的标签丢弃率
        learn_sigma: bool = True,      # 是否学习协方差
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # 1. Patchify 输入
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)

        # 2. 时间步 + 类别条件嵌入
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # 3. Positional Embedding
        num_patches = (input_size // patch_size) ** 2
        # 使用固定正弦位置编码 (与 ViT 一致), 不可学习
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # 4. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # 5. 最终输出层
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重: AdaLN 调制层零初始化 (保证训练初始为恒等函数)"""
        # 初始化位置编码
        pos_embed = self._get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
                                                   int(self.pos_embed.shape[1] ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 初始化 PatchEmbed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化 LabelEmbedder
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # 初始化 TimestepEmbedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # DiT Blocks: 零初始化 AdaLN 调制
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # FinalLayer: 零初始化
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    @staticmethod
    def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int):
        """生成 2D 正弦-余弦位置编码 (与 ViT/MAE 一致)"""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # w first
        grid = np.stack(grid, axis=0)
        grid = grid.reshape([2, 1, grid_size, grid_size])

        pos_embed = DiT._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    @staticmethod
    def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
        """从网格坐标生成位置编码"""
        assert embed_dim % 2 == 0
        emb_h = DiT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = DiT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        return np.concatenate([emb_h, emb_w], axis=1)

    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
        """1D 正弦-余弦位置编码"""
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000 ** omega
        pos = pos.reshape(-1)
        out = np.einsum('m,d->md', pos, omega)
        emb_sin = np.sin(out)
        emb_cos = np.cos(out)
        return np.concatenate([emb_sin, emb_cos], axis=1)

    def unpatchify(self, x: torch.Tensor):
        """
        Token 序列 → 2D 图像 (patchify 的逆操作)

        输入: [B, N, patch_size² * out_channels]
        输出: [B, out_channels, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], "输入 token 数必须是完全平方数"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        前向传播

        参数:
          x: [B, C, H, W] noisy latent at timestep t
          t: [B] 扩散时间步 (0~T)
          y: [B] 类别标签

        返回:
          [B, out_channels, H, W] 噪声预测 (+ 协方差若 learn_sigma=True)
        """
        # 1. Patchify: 2D latent → token 序列
        x = self.x_embedder(x) + self.pos_embed  # [B, N, hidden_size]

        # 2. 构建条件向量: t_emb + c_emb
        t_emb = self.t_embedder(t)                    # [B, hidden_size]
        y_emb = self.y_embedder(y, train=self.training)  # [B, hidden_size]
        c = t_emb + y_emb                             # [B, hidden_size]

        # 3. 逐 DiT Block 处理
        for block in self.blocks:
            x = block(x, c)

        # 4. 最终输出 + 反 patchify
        x = self.final_layer(x, c)  # [B, N, patch_size² * out_channels]
        x = self.unpatchify(x)      # [B, out_channels, H, W]
        return x


# ============================================================
# 5. 演示
# ============================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("DiT (Diffusion Transformer) 演示")
    print("=" * 60)

    # 使用小型配置快速演示
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiT(
        input_size=32,
        patch_size=4,        # p=4 → 64 tokens (快速演示)
        in_channels=4,
        hidden_size=384,     # DiT-S 配置
        depth=12,
        num_heads=6,
        num_classes=1000,
        learn_sigma=True,
    ).to(device)

    # 模拟一次前向传播
    B = 2
    x = torch.randn(B, 4, 32, 32).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    y = torch.randint(0, 1000, (B,)).to(device)

    output = model(x, t, y)
    print(f"输入 shape: {x.shape}")
    print(f"时间步 t: {t.tolist()}")
    print(f"类别 y: {y.tolist()}")
    print(f"输出 shape: {output.shape}")  # [B, 8, 32, 32] (4 通道噪声 + 4 通道协方差)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"DiT Block 数: {len(model.blocks)}")
    print(f"Token 数: {(32 // 4) ** 2}")

    # 验证 AdaLN-Zero 初始化 (首次前向应接近恒等)
    print(f"\n输出均值 (应接近 0): {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")

    print("\n✅ DiT 关键设计:")
    print("  1. Patchify: 2D latent → token 序列 (无位置编码=用固定正弦)")
    print("  2. AdaLN-Zero: 条件调制参数零初始化 → 训练从恒等开始")
    print("  3. AdaLN 而非 Cross-Attention: 参数高效, 不增加 Gflops")
    print("  4. learn_sigma=True: 同时预测噪声和协方差 (DiT-XXL/2 用这个)")
```
