"""
AlexNet
=======
论文: "ImageNet Classification with Deep Convolutional Neural Networks"
      (Krizhevsky et al., NeurIPS 2012)
核心贡献: 深度学习时代的开篇之作。首次在 ImageNet 上验证了大规模 CNN 的可行性，
         以 15.3% top-5 error 碾压第二名（26.2%），证明了"大数据+大模型+GPU"范式。
架构: 5个卷积层 + 3个全连接层，60M参数，使用ReLU/Dropout/LRN/重叠池化
代码结构:
  1. AlexNet - 完整实现（逐层注释）
  2. LocalResponseNorm - 局部响应归一化

关键设计选择:
  - ReLU 激活: 首次在 CNN 中大规模使用，比 tanh 快 6 倍
  - Dropout: 对抗过拟合（首次大规模验证其效果）
  - LRN: 局部响应归一化（后被 BatchNorm 取代）
  - 重叠池化: 3×3 窗口 stride=2，降低 ~0.4% error
  - 双 GPU 训练: 模型并行（当时 GPU 只有 1.5GB 显存）

与 [[../07_VGG/VGG|VGG]] 的关系: VGG 用统一的小卷积核取代了 AlexNet 的大卷积核
与 [[../09_ResNet/ResNet|ResNet]] 的关系: ResNet 通过残差连接解决了更深网络的退化问题
"""

import torch
import torch.nn as nn


# ==============================================================================
# 1. Local Response Normalization (LRN) — 已被 BatchNorm 取代
# ==============================================================================
class LocalResponseNorm(nn.Module):
    """
    局部响应归一化（LRN）

    公式:
      b^i_{x,y} = a^i_{x,y} / (k + α · Σ (a^j_{x,y})²)^β
      其中 j ∈ [max(0, i-n/2), min(N-1, i+n/2)]

    参数（原论文）: k=2, n=5, α=1e-4, β=0.75

    作用: 让在某一位置激活较强的神经元抑制相邻神经元的激活，
    实现"侧抑制"（lateral inhibition），类似生物神经元。

    历史地位: 在 VGG 中证明 LRN 不提供性能增益，已被 BatchNorm 完全取代。
    """

    def __init__(self, local_size: int = 5, alpha: float = 1e-4,
                 beta: float = 0.75, k: float = 2.0):
        super().__init__()
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        # 对每个位置，在通道维度上取相邻 local_size 个通道做归一化
        # 这里使用简单的实现（PyTorch 有更高效的 nn.LocalResponseNorm）
        # 注意: 原论文的 LRN 沿通道维度，不是空间维度
        n_channel = x.size(1)
        half = self.local_size // 2
        squared = x ** 2
        out = torch.zeros_like(x)

        for i in range(n_channel):
            start = max(0, i - half)
            end = min(n_channel - 1, i + half) + 1
            # 累加相邻通道的平方值
            sum_sq = squared[:, start:end, :, :].sum(dim=1, keepdim=True)
            denom = (self.k + self.alpha * sum_sq) ** self.beta
            out[:, i:i+1, :, :] = x[:, i:i+1, :, :] / denom

        return out


# ==============================================================================
# 2. AlexNet 完整架构
# ==============================================================================
class AlexNet(nn.Module):
    """
    AlexNet 完整架构

    8 层网络（5 卷积 + 3 全连接），约 60M 参数。

    逐层详解:
    - Conv1: 11×11, stride=4, 96核  → 输出 55×55×96
      设计原因: 大核 + 大 stride 是为了匹配当时 GPU 的 1.5GB 显存限制
               （快速降采样减少内存占用）

    - Conv2: 5×5, 256核              → 输出 27×27×256
      设计原因: 5×5 核捕获更大范围的模式

    - Conv3: 3×3, 384核              → 输出 13×13×384
    - Conv4: 3×3, 384核              → 输出 13×13×384
    - Conv5: 3×3, 256核              → 输出 13×13×256
      设计原因: 中间层使用小核捕获局部细节

    - FC6: 4096 + Dropout            → 输出 4096
    - FC7: 4096 + Dropout            → 输出 4096
    - FC8: 1000 (1000 类 ImageNet)   → Softmax 输出

    参数分布: FC6 和 FC7 占约 90% 参数（37.8M + 16.8M = 54.6M / 60M）
    """

    def __init__(self, num_classes: int = 1000, use_lrn: bool = False):
        super().__init__()

        # === 特征提取部分（卷积层） ===

        # Conv1: 224×224×3 → 55×55×96
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),  # AlexNet 首次大规模使用 ReLU
        )
        self.lrn1 = LocalResponseNorm() if use_lrn else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 重叠池化
        # → 27×27×96

        # Conv2: 27×27×96 → 27×27×256
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.lrn2 = LocalResponseNorm() if use_lrn else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # → 13×13×256

        # Conv3: 13×13×256 → 13×13×384（无池化、无 LRN）
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Conv4: 13×13×384 → 13×13×384
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Conv5: 13×13×384 → 13×13×256 → 6×6×256
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # → 6×6×256

        # === 分类部分（全连接层） ===

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),    # Dropout 对抗过拟合
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),    # 两个 FC 层后都用 Dropout (p=0.5)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

        # 权重初始化（原论文: N(0, 0.01)）
        # Conv2/4/5 和全连接层的偏置初始化为 1（防止 ReLU 死神经元）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # Conv2, Conv4, Conv5 偏置初始化为 1
                if m.out_channels in (256, 384):
                    nn.init.constant_(m.bias, 1)
                else:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 1)  # 全连接层偏置初始化为 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取
        x = self.pool1(self.lrn1(self.conv1(x)))
        x = self.pool2(self.lrn2(self.conv2(x)))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool5(self.conv5(x))

        # 展平并分类
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("AlexNet 架构演示")
    print("=" * 60)

    model = AlexNet(num_classes=1000, use_lrn=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} (~60M)")
    # 统计全连接参数占比
    fc_params = sum(p.numel() for name, p in model.named_parameters()
                    if 'classifier' in name)
    print(f"全连接层参数: {fc_params:,} ({fc_params/total_params*100:.1f}%)")
    print("→ 全连接层占了 ~90% 参数，但只做分类，效率低")

    # 模拟输入
    x = torch.randn(2, 3, 224, 224)
    print(f"\n输入: {x.shape}")

    # 逐层查看形状
    print("\n--- 逐层形状 ---")
    with torch.no_grad():
        out = model.conv1(x)
        print(f"Conv1: {out.shape}")
        out = model.pool1(model.lrn1(out))
        print(f"Pool1: {out.shape}")

        out = model.pool2(model.lrn2(model.conv2(out)))
        print(f"Conv2+Pool2: {out.shape}")

        out = model.conv3(out)
        print(f"Conv3: {out.shape}")
        out = model.conv4(out)
        print(f"Conv4: {out.shape}")
        out = model.pool5(model.conv5(out))
        print(f"Conv5+Pool5: {out.shape}")

        out_flat = out.view(2, -1)
        print(f"Flatten: {out_flat.shape}")
        logits = model.classifier(out_flat)
        print(f"FC (logits): {logits.shape}")

    print("\n--- AlexNet 的历史贡献 ---")
    print("  1. ReLU 激活（比 tanh 快 6 倍，解决梯度消失）")
    print("  2. Dropout 正则化（对抗全连接层的过拟合）")
    print("  3. 双 GPU 模型并行训练（突破显存限制）")
    print("  4. '大数据 + 大模型 + GPU'范式的开创者")
    print("  5. 证明了 CNN 在 ImageNet 规模上可行")

    print("\n--- AlexNet 的设计缺陷 ---")
    print("  - Conv1 11×11 核太大、stride=4 → 丢失细粒度信息")
    print("  - 全连接层参数占比过高（90%）")
    print("  - LRN 后来被证明不必要（VGG 消融实验）")
    print("  → 后续 [[../07_VGG/VGG|VGG]] 用 3×3 小卷积核堆叠改进")
    print("  → 后续 [[../09_ResNet/ResNet|ResNet]] 用残差连接训练更深的网络")
