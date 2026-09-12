---
tags:
  - 代码
  - PyTorch
created: 2026-07-01
---

# DenseNet: Densely Connected Convolutional Networks - 代码实现

> 本文档包含 `model.py` 的 PyTorch/NumPy 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
DenseNet: Densely Connected Convolutional Networks
==================================================
论文: "Densely Connected Convolutional Networks" (Huang et al., CVPR 2017 Best Paper)
核心贡献: 密集连接——每一层接收前面所有层的特征图作为输入（通道拼接，而非加法）。
         DenseNet-201 (20M) ≈ ResNet-152 (60M) 性能，参数仅 1/3。
架构: DenseBlock（密集块）+ Transition Layer（过渡层），增长率 k 控制
代码结构:
  1. DenseLayer - 密集层（BN→ReLU→1×1→BN→ReLU→3×3, 输出 k 个通道）
  2. DenseBlock - 密集块（L 层的密集连接）
  3. TransitionLayer - 过渡层（BN→1×1→AvgPool, 压缩通道数）
  4. DenseNet - 完整模型 (121/169/201)

核心公式: x_l = H_l([x_0, x_1, ..., x_{l-1}])
第 l 层的输入是前面所有层输出的通道拼接。

Growth Rate k: 每层贡献 k 个新特征（通常 k=32），所有历史特征通过拼接保留。
Transition Layer 使用压缩因子 θ (通常 0.5) 压缩通道数。

与 [[../09_ResNet/ResNet|ResNet]] 的区别:
  ResNet: 加法连接 (y = F(x) + x) → 信息可能混合丢失
  DenseNet: 拼接连接 (y = [x, F(x)]) → 所有历史信息完整保留
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ==============================================================================
# 1. DenseLayer —— 密集层
# ==============================================================================
class _DenseLayer(nn.Module):
    """
    密集块内的单个层（Bottleneck 版本，即 DenseNet-B）

    结构: BN → ReLU → 1×1 Conv(4k) → BN → ReLU → 3×3 Conv(k)

    1×1 卷积降维到 4k 通道（k = growth rate），
    然后 3×3 卷积输出 k 个新特征。

    为什么用 1×1 降维？
    当输入通道数很大时（如第 12 层有 k×(11)+64 = 416 个输入通道），
    直接在这么多通道上做 3×3 卷积计算量巨大。
    1×1 卷积将通道压缩到 4k，大幅降低计算量。
    """

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        # BN → ReLU → 1×1 (4k 通道)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate,
                               kernel_size=1, bias=False)

        # BN → ReLU → 3×3 (k 通道)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意: 这里是"预激活"结构 (BN→ReLU→Conv)，
        # 与 [[../09_ResNet/ResNet|ResNet]] 的 Pre-Activation 设计一致
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        # 返回的是新特征，会在 DenseBlock 中与历史特征拼接
        return out


# ==============================================================================
# 2. DenseBlock —— 密集块
# ==============================================================================
class _DenseBlock(nn.Module):
    """
    密集块（DenseBlock）

    包含 num_layers 个密集连接的层。每层的输出在通道维度与前面所有
    层的输出拼接，形成"集体知识库"（collective knowledge）。

    以 growth_rate=32, num_layers=6 为例:
    - 输入: 64 通道
    - Layer 1: 输入 64 → 输出 32 → 拼接 → 96 通道
    - Layer 2: 输入 96 → 输出 32 → 拼接 → 128 通道
    - ...
    - Layer 6: 输入 224 → 输出 32 → 拼接 → 256 通道
    - 最终输出: 64 + 6×32 = 256 通道

    为什么 growth rate 那么小（k=32）？
    - 每层只需贡献少量"新知识"，历史信息通过拼接完整保留
    - 窄层设计最大化了参数效率
    - DenseNet-201 (20M) ≈ ResNet-152 (60M) 性能，参数只有 1/3
    """

    def __init__(self, num_layers: int, in_channels: int,
                 growth_rate: int, bn_size: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,  # 输入通道随层数增长
                growth_rate,
                bn_size,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]  # 保存前面所有层的特征
        for layer in self.layers:
            # 拼接所有历史特征作为当前层的输入
            # 这是 DenseNet 的核心 —— 不是加法（ResNet），而是拼接
            concat_features = torch.cat(features, dim=1)
            new_features = layer(concat_features)  # 输出 k 个新特征
            features.append(new_features)
        # 返回所有特征的拼接
        return torch.cat(features, dim=1)


# ==============================================================================
# 3. TransitionLayer —— 过渡层
# ==============================================================================
class _TransitionLayer(nn.Module):
    """
    过渡层——连接两个密集块

    功能:
    1. 降采样（2×2 AvgPool, stride=2）
    2. 通道压缩（1×1 Conv, 压缩因子 θ∈(0,1]）

    结构: BN → ReLU → 1×1 Conv → 2×2 AvgPool

    压缩因子 θ 的作用:
    - θ=1.0: 不压缩（DenseNet，无 C）
    - θ=0.5: 压缩到一半通道（DenseNet-C，推荐）
    - 压缩可以减少参数和计算量，同时保持性能
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


# ==============================================================================
# 4. 完整 DenseNet 模型
# ==============================================================================
class DenseNet(nn.Module):
    """
    DenseNet-BC 完整架构

    | 模型          | 每一阶段层数      | 增长率 k | 参数量 |
    |--------------|-----------------|---------|--------|
    | DenseNet-121 | [6, 12, 24, 16] | 32      | ~8M    |
    | DenseNet-169 | [6, 12, 32, 32] | 32      | ~14M   |
    | DenseNet-201 | [6, 12, 48, 32] | 32      | ~20M   |
    | DenseNet-264 | [6, 12, 64, 48] | 32      | ~33M   |

    DenseNet-201 (20M) ≈ ResNet-101 (45M) 性能，参数减少 56%！
    """

    def __init__(self, growth_rate: int = 32, block_config: tuple = (6, 12, 24, 16),
                 num_init_features: int = 64, bn_size: int = 4,
                 compression: float = 0.5, num_classes: int = 1000):
        super().__init__()

        # === stem: 初始卷积 + 池化 ===
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # === 4 个密集块 + 过渡层 ===
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 密集块
            block = _DenseBlock(
                num_layers=num_layers,
                in_channels=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            # 最后一块后不加过渡层
            if i != len(block_config) - 1:
                # 过渡层: 通道压缩
                num_out_features = int(num_features * compression)
                trans = _TransitionLayer(num_features, num_out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_out_features

        # === 最终 BN + ReLU ===
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.relu = nn.ReLU(inplace=True)

        # === 分类头 ===
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = self.relu(features)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DenseNet 架构演示")
    print("=" * 60)

    # 各版本参数对比
    configs = {
        "DenseNet-121": (6, 12, 24, 16),
        "DenseNet-169": (6, 12, 32, 32),
        "DenseNet-201": (6, 12, 48, 32),
    }

    for name, cfg in configs.items():
        model = DenseNet(growth_rate=32, block_config=cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15s}: {params/1e6:5.1f}M 参数")

    # 逐层形状
    print("\n--- DenseNet-121 逐层形状 ---")
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                     num_init_features=64)
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        for name, module in model.features.named_children():
            x = module(x)
            print(f"{name:20s}: {list(x.shape)}")

    # 核心概念演示
    print("\n--- DenseNet 核心概念 ---")
    print("Growth Rate (k):")
    print("  - 每层只输出 k 个新特征图")
    print("  - 历史信息通过拼接完整保留")
    print("  - k 很小 (12-32)，最大化参数效率")

    print("\nDenseNet vs ResNet 连接方式:")
    print("  ResNet:   y = F(x) + x     → 加法（信息混合）")
    print("  DenseNet: y = concat(x, F(x)) → 拼接（信息完整保留）")

    print("\n参数效率:")
    print("  DenseNet-201: 20M ≈ ResNet-101: 45M")
    print("  → 相同性能，参数减少 56%")

    print("\n--- DenseNet 的局限性 ---")
    print("  1. 训练显存消耗大（需要保存所有层的中间激活值）")
    print("  2. 推理速度慢于 ResNet（拼接操作 + 大通道数）")
    print("  3. 高分辨率输入下计算开销急剧增长")

```
