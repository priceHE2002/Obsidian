---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# Open X-Embodiment & RT-X - 代码实现

> 本文档包含 PyTorch/NumPy 教学实现。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# 1. OXE 数据标准化流程
# ============================================================
# OXE 的核心贡献之一：将来自 22 种不同机器人的数据统一为
# 7 维动作向量 [dx, dy, dz, droll, dpitch, dyaw, gripper]
# 不同数据集有不同的控制模式（绝对位置/增量/速度），
# 这里实现数据集级别的归一化来统一处理

class OXEDataNormalizer:
    """数据集级别归一化器：每个数据集维护自己的统计量，
    确保不同机器人的动作和观测被缩放到相近的范围。
    这是 OXE 论文中「粗略对齐」策略的核心。"""

    def __init__(self):
        self.obs_mean = None
        self.obs_std = None
        self.action_mean = None
        self.action_std = None
        self._count = 0
        self._obs_sum = None
        self._obs_sq_sum = None
        self._act_sum = None
        self._act_sq_sum = None

    def update(self, obs: np.ndarray, action: np.ndarray):
        """用 Welford 在线算法逐步更新统计量，避免一次加载全部数据导致 OOM"""
        self._count += 1
        if self._obs_sum is None:
            self._obs_sum = np.zeros(obs.shape, dtype=np.float64)
            self._obs_sq_sum = np.zeros(obs.shape, dtype=np.float64)
            self._act_sum = np.zeros(action.shape, dtype=np.float64)
            self._act_sq_sum = np.zeros(action.shape, dtype=np.float64)

        self._obs_sum += obs
        self._obs_sq_sum += obs ** 2
        self._act_sum += action
        self._act_sq_sum += action ** 2

    def finalize(self):
        """计算最终的均值和标准差"""
        self.obs_mean = self._obs_sum / self._count
        self.obs_std = np.sqrt(self._obs_sq_sum / self._count - self.obs_mean ** 2)
        # 防止除零：如果某个维度方差为 0，将其 std 设为 1
        self.obs_std = np.clip(self.obs_std, a_min=1e-6, a_max=None)

        self.action_mean = self._act_sum / self._count
        self.action_std = np.sqrt(self._act_sq_sum / self._count - self.action_mean ** 2)
        self.action_std = np.clip(self.action_std, a_min=1e-6, a_max=None)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_mean) / self.obs_std

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        return action_norm * self.action_std + self.action_mean


# ============================================================
# 2. FiLM 条件化模块
# ============================================================
# FiLM 让模型感知"当前在第几步任务"——每层特征都会
# 被条件向量（如语言指令的 embedding）调制。
# 为什么用 FiLM 而不是简单拼接？
# FiLM 对特征做逐通道的 scale + shift，比拼接更强大、更稳定。

class FiLMGen(nn.Module):
    """FiLM 生成器：将通用条件信息调制到卷积层特征通道上。"""
    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(cond_dim, feature_dim)
        self.shift_proj = nn.Linear(cond_dim, feature_dim)

    def forward(self, features: torch.Tensor, cond: torch.Tensor):
        """features: [B, C, ...], cond: [B, cond_dim]"""
        scale = self.scale_proj(cond)
        shift = self.shift_proj(cond)
        while scale.dim() < features.dim():
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return features * (1 + scale) + shift


class FiLMResBlock(nn.Module):
    """带 FiLM 的残差卷积块"""
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.film1 = FiLMGen(cond_dim, channels)
        self.film2 = FiLMGen(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.film1(x, cond)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.film2(x, cond)
        return F.relu(x + residual)


# ============================================================
# 3. RT-1-X 策略网络（35M 参数）
# ============================================================
# 组件：视觉 CNN → 逐帧处理历史图像 → Transformer → 输出动作
# 核心设计：每帧图像独立编码（共享权重），然后通过 Transformer 融合时序

class RT1XPolicy(nn.Module):
    """RT-1-X 策略网络。
    与纯 RT-1 的关键区别：RT-1-X 通过 dataset_id embedding
    让模型感知当前控制的是哪种机器人形态。"""

    def __init__(self,
                 num_action_dim: int = 7,
                 num_history: int = 6,        # 输入多少帧历史图像
                 feature_dim: int = 512,       # 视觉编码器输出维度
                 hidden_dim: int = 256,
                 num_transformer_layers: int = 4,
                 num_heads: int = 4,
                 max_datasets: int = 60):      # OXE 最多 60 个数据集
        super().__init__()

        # 简化的视觉编码器（原论文用 EfficientNet）
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.ReLU(),
        )

        # 数据集 ID embedding —— RT-1-X 的独特设计
        # 让模型知道当前处理的数据来自哪种机器人（Franka/WidowX/...）
        self.dataset_embedding = nn.Embedding(max_datasets, hidden_dim)

        # 投影 + 时序位置编码
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_history, hidden_dim) * 0.02
        )

        # Transformer encoder —— 融合多帧时序信息
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # 输出头：7 维动作 [dx,dy,dz,dr,dp,dy,gripper] + 1 维 terminate flag
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_action_dim + 1),
        )

    def forward(self,
                images: torch.Tensor,       # [B, T, 3, H, W]
                dataset_ids: torch.Tensor,  # [B] — 每条数据属于哪个数据集
                ):
        B, T, C, H, W = images.shape

        # 每帧独立编码（共享视觉编码器权重）
        images_flat = images.view(B * T, C, H, W)
        feat = self.vision_encoder(images_flat)          # [B*T, feature_dim]
        feat = feat.view(B, T, -1)                       # [B, T, feature_dim]
        feat = self.feature_proj(feat)                   # [B, T, hidden_dim]

        # 数据集 ID 注入 —— 广播到每一帧
        ds_emb = self.dataset_embedding(dataset_ids)     # [B, hidden_dim]
        ds_emb = ds_emb.unsqueeze(1)                     # [B, 1, hidden_dim]
        feat = feat + ds_emb + self.temporal_pos_embed[:, :T, :]

        # Transformer 建模时序关系
        out = self.transformer(feat)                     # [B, T, hidden_dim]
        # 取最后一帧（聚合了全部历史信息的 token）预测动作
        last_frame = out[:, -1, :]                       # [B, hidden_dim]
        action_output = self.action_head(last_frame)     # [B, 8]

        action = action_output[:, :7]     # [B, 7] — 末端执行器位移
        terminate = action_output[:, 7:8] # [B, 1] — 终止概率 logit
        return action, terminate


# ============================================================
# 4. RT-2-X 动作 token 化（教学简化版）
# ============================================================
# RT-2-X 核心思想：把连续动作离散化到 256 个 bin，
# 每个 bin 对应一个 token，让 VLM 像生成文本一样生成动作。

class RT2XActionTokenizer:
    """连续动作 <-> 离散 token 的编码/解码器。
    每个动作维度被等分到 256 个 bin，
    整个 7 维动作被映射为 7 个整数 token。
    这是 RT-2 论文中最巧妙的设计：「动作即语言」。"""

    def __init__(self, action_dim: int = 7, num_bins: int = 256):
        self.action_dim = action_dim
        self.num_bins = num_bins
        # 动作范围（示例值，实际会根据数据集统计）
        self.action_min = np.array([-0.1, -0.1, -0.05, -1.0, -1.0, -1.0, 0.0])
        self.action_max = np.array([ 0.1,  0.1,  0.05,  1.0,  1.0,  1.0, 1.0])

    def discretize(self, action: np.ndarray) -> np.ndarray:
        """将连续动作映射为 0~255 的离散整数 token"""
        action_clipped = np.clip(action, self.action_min, self.action_max)
        normalized = (action_clipped - self.action_min) / (
            self.action_max - self.action_min + 1e-8
        )
        tokens = (normalized * (self.num_bins - 1)).astype(np.int32)
        return tokens  # shape: [7]

    def undiscretize(self, tokens: np.ndarray) -> np.ndarray:
        """将 0~255 的离散 token 还原为连续动作值"""
        normalized = tokens.astype(np.float32) / (self.num_bins - 1)
        normalized = np.clip(normalized, 0.0, 1.0)
        action = normalized * (self.action_max - self.action_min) + self.action_min
        return action  # shape: [7]


# ============================================================
# 演示
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OXE Data Normalizer 演示")
    print("=" * 60)
    normalizer = OXEDataNormalizer()
    for _ in range(100):
        obs = np.random.randn(512) * 2
        act = np.random.randn(7) * 0.1
        normalizer.update(obs, act)
    normalizer.finalize()
    print(f"Obs std range: "
          f"[{normalizer.obs_std.min():.3f}, {normalizer.obs_std.max():.3f}]")
    print(f"Action std range: "
          f"[{normalizer.action_std.min():.3f}, {normalizer.action_std.max():.3f}]")
    print()

    print("=" * 60)
    print("RT-1-X 策略网络演示")
    print("=" * 60)
    model = RT1XPolicy()
    dummy_images = torch.randn(2, 6, 3, 224, 224)
    dummy_dataset_ids = torch.randint(0, 60, (2,))
    action, terminate = model(dummy_images, dummy_dataset_ids)
    print(f"输入: images={dummy_images.shape}, dataset_ids={dummy_dataset_ids}")
    print(f"输出: action={action.shape}, terminate={terminate.shape}")
    print(f"Action (batch 0): {action[0].detach().numpy()}")
    print(f"Terminate logit (batch 0): {terminate[0].item():.4f}")
    print()

    print("=" * 60)
    print("RT-2-X 动作 Token 化演示")
    print("=" * 60)
    tokenizer = RT2XActionTokenizer()
    raw_action = np.array([0.05, -0.02, 0.01, 0.3, -0.4, 0.2, 1.0])
    tokens = tokenizer.discretize(raw_action)
    restored = tokenizer.undiscretize(tokens)
    print(f"原始动作: {raw_action}")
    print(f"离散token: {tokens}")
    print(f"还原动作: {restored}")
    print(f"量化误差: {np.mean(np.abs(raw_action - restored)):.6f}")
```
