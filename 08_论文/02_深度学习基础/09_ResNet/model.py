"""
ResNet: Deep Residual Learning
===============================
论文: "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016 Best Paper)
核心贡献: 残差连接（y = F(x) + x）解决了深层网络的退化问题，使训练152层网络成为可能。
         CV 历史上引用量最高的论文（~210K+）。
架构: ResNet-18/34/50/101/152，BasicBlock（2层）和 Bottleneck（3层）
代码结构:
  1. BasicBlock - ResNet-18/34 的两层残差块
  2. Bottleneck - ResNet-50/101/152 的三层瓶颈残差块
  3. ResNet - 完整模型（支持 18/34/50/101/152）

残差学习的核心洞察:
  如果新增层可以学习恒等映射，深层网络至少不应比浅层差。
  但实践中网络很难学会 H(x)=x。让网络学习残差 F(x)=H(x)-x，
  最优解为恒等映射时只需将 F(x) 推向 0，比学习恒等容易得多。

Bottleneck 设计 (1×1→3×3→1×1):
  输入 256d → 1×1(64) → 3×3(64) → 1×1(256) → +输入
  计算量仅为两层残差块的 ~6%，却保持相同感受野。

与 [[../07_VGG/VGG|VGG]] 的关系: 继承 3×3 卷积设计
与 [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的关系:
  残差连接是 Transformer 架构的核心组件之一
"""

import torch
import torch.nn as nn


# ==============================================================================
# 1. BasicBlock —— ResNet-18/34 的两层残差块
# ==============================================================================
class BasicBlock(nn.Module):
    """
    基本残差块（用于 ResNet-18/34）

    结构: 3×3 conv → BN → ReLU → 3×3 conv → BN → + shortcut → ReLU

    残差公式: y = F(x, {W_i}) + x
    - F(x, {W_i}) = W_2·ReLU(BN(W_1·x)): 需要学习的残差映射
    - x: 恒等映射（identity shortcut）

    当输入输出维度不一致时（如通道数变化、空间分辨率变化），
    需要 1×1 卷积投影 shortcut 来匹配维度:
      y = F(x, {W_i}) + W_s·x
    """

    expansion = 1  # BasicBlock 输出通道 = 输入通道 × 1

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module = None):
        super().__init__()

        # 残差映射 F(x)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Shortcut 连接（恒等映射或投影映射）
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 残差映射
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut（恒等映射或投影）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接: F(x) + x
        out += identity
        out = self.relu(out)

        return out


# ==============================================================================
# 2. Bottleneck —— ResNet-50/101/152 的瓶颈残差块
# ==============================================================================
class Bottleneck(nn.Module):
    """
    瓶颈残差块（用于 ResNet-50/101/152）

    结构: 1×1(降维) → BN → ReLU → 3×3(核心计算) → BN → ReLU → 1×1(升维) → BN → + shortcut

    为什么使用瓶颈设计？
    当网络深度超过 50 层时，两层 BasicBlock 计算量过大。
    瓶颈块通过 1×1 卷积将通道数先降后升，使 3×3 卷积在低维空间计算。

    以 256d 为例:
    - BasicBlock: 256→3×3,256→3×3,256: 256×9×256 + 256×9×256 ≈ 1.18M 参数
    - Bottleneck:  256→1×1,64→3×3,64→1×1,256: 256×64 + 64×576 + 64×256 ≈ 0.074M 参数
    - 参数减少约 94%，计算量降至 ~6%！

    这种 1×1→3×3→1×1 的"三明治"结构来自 [[../08_GoogLeNet/GoogLeNet|GoogLeNet]] 的 1×1 降维思想。
    """

    expansion = 4  # Bottleneck 输出通道 = 输入通道 × 4

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        mid_channels = out_channels // self.expansion  # 瓶颈中间层通道数

        # 1×1 降维
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # 3×3 核心计算（在低维空间进行）
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 1×1 升维（恢复通道数）
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 瓶颈残差映射: 1×1 → 3×3 → 1×1
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Shortcut
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


# ==============================================================================
# 3. 完整 ResNet 模型
# ==============================================================================
class ResNet(nn.Module):
    """
    ResNet 完整架构

    | 模型        | 层数 | 块类型    | 每个阶段的块数  | 参数  | FLOPs  |
    |------------|------|----------|---------------|-------|--------|
    | ResNet-18  | 18   | BasicBlock | [2,2,2,2]    | 11M   | 1.8G   |
    | ResNet-34  | 34   | BasicBlock | [3,4,6,3]    | 22M   | 3.6G   |
    | ResNet-50  | 50   | Bottleneck | [3,4,6,3]    | 26M   | 3.8G   |
    | ResNet-101 | 101  | Bottleneck | [3,4,23,3]   | 45M   | 7.6G   |
    | ResNet-152 | 152  | Bottleneck | [3,8,36,3]   | 60M   | 11.3G  |

    降采样设计: conv3_1、conv4_1、conv5_1 中 stride=2 降采样
    """

    def __init__(self, block, layers: list[int], num_classes: int = 1000):
        """
        block: BasicBlock 或 Bottleneck
        layers: 每个阶段的块数列表，如 [3, 4, 6, 3]
        """
        super().__init__()
        self.in_channels = 64

        # stem: 7×7 卷积 + 3×3 MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个残差阶段
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头: 全局平均池化 → FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """
        构建一个残差阶段

        第一个块可能需要:
        - stride != 1: 空间降采样
        - downsample shortcut: 通道数匹配（通过 1×1 卷积）
        """
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块（可能含降采样）
        layers.append(block(self.in_channels, out_channels * block.expansion,
                            stride, downsample))
        self.in_channels = out_channels * block.expansion

        # 后续块（stride=1，无降采样）
        for _ in range(1, blocks):
            layers.append(block(self.in_channels,
                                out_channels * block.expansion))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ResNet 架构演示")
    print("=" * 60)

    # 测试各种配置
    for name, builder in [("ResNet-18", resnet18), ("ResNet-34", resnet34),
                           ("ResNet-50", resnet50), ("ResNet-101", resnet101),
                           ("ResNet-152", resnet152)]:
        model = builder()
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s}: {params/1e6:5.1f}M 参数")

    # 逐层形状
    print("\n--- ResNet-50 逐层形状 ---")
    model = resnet50()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out = model.relu(model.bn1(model.conv1(x)))
        print(f"Stem:    {out.shape}")
        out = model.maxpool(out)
        print(f"MaxPool: {out.shape}")

        out = model.layer1(out)
        print(f"Layer1:  {out.shape}")
        out = model.layer2(out)
        print(f"Layer2:  {out.shape}")
        out = model.layer3(out)
        print(f"Layer3:  {out.shape}")
        out = model.layer4(out)
        print(f"Layer4:  {out.shape}")

        out = model.avgpool(out)
        print(f"AvgPool: {out.shape}")
        out = model.fc(out.view(out.size(0), -1))
        print(f"FC:      {out.shape}")

    # 瓶颈 vs 基本块 计算量对比
    print("\n--- Bottleneck vs BasicBlock 计算量对比 ---")
    print("BasicBlock (256d → 3×3,256 → 3×3,256):")
    print("  两层层参数量: 256×9×256 + 256×9×256 ≈ 1,179,648")
    print("\nBottleneck (256d → 1×1,64 → 3×3,64 → 1×1,256):")
    print("  三层参数量: 256×64 + 64×576 + 64×256 ≈ 69,632")
    print("  节省约 94% 计算量！")

    print("\n--- ResNet 的核心影响 ---")
    print("  1. 残差连接 → [[../01_Attention_Is_All_You_Need/Attention Is All You Need|Transformer]] 的 Add & Norm")
    print("  2. 恒等捷径 → Gradient Highway（梯度高速公路）")
    print("  3. Bottleneck → [[../10_DenseNet/DenseNet|DenseNet]] 的 1×1 降维")
    print("  4. ResNet 骨干 → 现代 VLA 的标准视觉编码器")
