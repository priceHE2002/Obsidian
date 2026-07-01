"""
GoogLeNet (Inception v1)
========================
论文: "Going Deeper with Convolutions" (Szegedy et al., CVPR 2015)
核心贡献: 提出 Inception 模块——在同一层并行使用 1×1、3×3、5×5 卷积和池化，
         让网络自动学习使用哪个尺度。1×1 卷积降维将参数从 60M 降至 5M。
         ILSVRC 2014 分类和定位双向冠军（6.67% top-5 error）。
架构: 22层，Inception 模块 + 全局平均池化 + 辅助分类器，~5M 参数
代码结构:
  1. InceptionModule - 多尺度并行卷积模块（含 1×1 降维）
  2. AuxiliaryClassifier - 辅助分类器（对抗梯度消失）
  3. GoogLeNet - 完整模型

1×1 卷积降维的魔力:
  输入 14×14×256 → 输出 14×14×128 (5×5 卷积):
  - 直接 5×5: 256×128×5×5 = 819,200 次乘法
  - 先 1×1(64) 再 5×5: 256×64+64×128×25 = 221,184 次乘法
  - 减少 73% 计算量！

与 [[../07_VGG/VGG|VGG]] 的对比: VGG 138M 参数 vs GoogLeNet 5M 参数（27倍差异）
与 [[../09_ResNet/ResNet|ResNet]] 的关系: ResNet bottleneck 块继承了 1×1 降维思想
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Inception 模块 —— 多尺度并行卷积
# ==============================================================================
class InceptionModule(nn.Module):
    """
    Inception v1 模块（降维版）

    4 条并行分支:
    分支1: 1×1 卷积                  ← 直接逐点变换
    分支2: 1×1 降维 → 3×3 卷积       ← 小尺度特征
    分支3: 1×1 降维 → 5×5 卷积       ← 大尺度特征
    分支4: 3×3 MaxPool → 1×1 投影    ← 池化分支

    为什么需要多尺度并行？
    图像中目标大小差异极大（一张图中猫可能占 80%，另一张只占 5%），
    单一大小的卷积核无法同时适配。多分支让网络自动选择最佳尺度。

    为什么 1×1 卷积放在 3×3/5×5 之前？
    1×1 卷积在通道维度做线性组合（降维），大幅减少后续大卷积核的计算量。
    这被称为"瓶颈"设计——后被 [[../09_ResNet/ResNet|ResNet]] bottleneck 继承。
    """

    def __init__(self, in_channels: int, out_1x1: int, reduce_3x3: int,
                 out_3x3: int, reduce_5x5: int, out_5x5: int, pool_proj: int):
        super().__init__()

        # 分支1: 1×1 卷积（直接变换）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # 分支2: 1×1 降维 → 3×3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 分支3: 1×1 降维 → 5×5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        # 分支4: 3×3 MaxPool → 1×1 投影
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 在通道维度拼接所有分支输出
        return torch.cat([b1, b2, b3, b4], dim=1)


# ==============================================================================
# 2. 辅助分类器 —— 对抗梯度消失
# ==============================================================================
class AuxiliaryClassifier(nn.Module):
    """
    GoogLeNet 辅助分类器

    放置位置: Inception(4a) 和 Inception(4d) 之后
    训练时总损失: L_total = L_main + 0.3 × L_aux1 + 0.3 × L_aux2
    推理时: 完全丢弃辅助分类器

    三点作用:
    1. 梯度增强: 辅助损失的梯度直接注入网络中部，改善早期层梯度信号
    2. 正则化: 强迫中间层也学到判别特征
    3. 抗梯度消失: 在 BatchNorm 出现之前，这是训练 22 层网络的关键技巧

    注意: 后来 [[../09_ResNet/ResNet|ResNet]] 的残差连接以更优雅的方式解决了梯度问题。
    """

    def __init__(self, in_channels: int, num_classes: int = 1000):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)  # 4a: 14→4, 4d: 14→4
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)  # 辅助分类器 dropout 比主分类器更高 (70%)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==============================================================================
# 3. 完整 GoogLeNet 模型
# ==============================================================================
class GoogLeNet(nn.Module):
    """
    GoogLeNet (Inception v1) 完整架构

    22 层，仅 5M 参数，ILSVRC 2014 冠军（6.67% top-5 error）。

    设计亮点:
    - 用全局平均池化（GAP）替代全连接层: 减少参数，提升泛化（+0.6%）
    - 辅助分类器: 提供中间层梯度信号
    - 固定计算预算 1.5 GFLOPs 的设计目标
    """

    def __init__(self, num_classes: int = 1000, aux_logits: bool = True):
        super().__init__()
        self.aux_logits = aux_logits

        # === 初始卷积层 ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # === Inception 模块序列 ===
        # (in_c, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj)
        inception_cfg = [
            # Stage 3 (28×28)
            (192, 64, 96, 128, 16, 32, 32),    # 3a → 28×28×256
            (256, 128, 128, 192, 32, 96, 64),   # 3b → 28×28×480
            # MaxPool → 14×14
            # Stage 4 (14×14)
            (480, 192, 96, 208, 16, 48, 64),    # 4a → 14×14×512
            (512, 160, 112, 224, 24, 64, 64),   # 4b → 14×14×512
            (512, 128, 128, 256, 24, 64, 64),   # 4c → 14×14×512
            (512, 112, 144, 288, 32, 64, 64),   # 4d → 14×14×528
            (528, 256, 160, 320, 32, 128, 128),  # 4e → 14×14×832
            # MaxPool → 7×7
            # Stage 5 (7×7)
            (832, 256, 160, 320, 32, 128, 128),  # 5a → 7×7×832
            (832, 384, 192, 384, 48, 128, 128),  # 5b → 7×7×1024
        ]

        self.inception3a = InceptionModule(*inception_cfg[0])
        self.inception3b = InceptionModule(*inception_cfg[1])
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(*inception_cfg[2])
        self.inception4b = InceptionModule(*inception_cfg[3])
        self.inception4c = InceptionModule(*inception_cfg[4])
        self.inception4d = InceptionModule(*inception_cfg[5])
        self.inception4e = InceptionModule(*inception_cfg[6])
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(*inception_cfg[7])
        self.inception5b = InceptionModule(*inception_cfg[8])

        # === 辅助分类器 ===
        if aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)   # 4a 后
            self.aux2 = AuxiliaryClassifier(528, num_classes)   # 4d 后

        # === 主分类器（全局平均池化 + Dropout + 线性层） ===
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)  # 比辅助分类器的 0.7 低
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        # 初始卷积
        x = self.conv1(x)
        x = self.conv2(x)

        # Stage 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Stage 4
        x = self.inception4a(x)
        aux1_out = self.aux1(x) if self.aux_logits and self.training else None

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_out = self.aux2(x) if self.aux_logits and self.training else None

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Stage 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 主分类器
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux1_out, aux2_out
        return x


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GoogLeNet (Inception v1) 架构演示")
    print("=" * 60)

    model = GoogLeNet(num_classes=1000, aux_logits=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"GoogLeNet 参数量: {total_params:,} (~5M)")
    print("→ 比 [[../07_VGG/VGG|VGG-16]] (138M) 少 27 倍参数！")

    x = torch.randn(2, 3, 224, 224)
    model.train()
    outputs = model(x)
    print(f"\n输入: {x.shape}")
    if model.aux_logits:
        main, aux1, aux2 = outputs
        print(f"主分类器输出: {main.shape}")
        print(f"辅助分类器1输出: {aux1.shape}")
        print(f"辅助分类器2输出: {aux2.shape}")

    # 推理模式
    model.eval()
    with torch.no_grad():
        out = model(x)
        print(f"\n推理模式输出: {out.shape}")

    # ---- Inception 模块分析 ----
    print("\n--- Inception 模块 3a 通道分析 ---")
    x_test = torch.randn(1, 192, 28, 28)
    inc = InceptionModule(192, 64, 96, 128, 16, 32, 32)
    with torch.no_grad():
        b1 = inc.branch1(x_test)
        b2 = inc.branch2(x_test)
        b3 = inc.branch3(x_test)
        b4 = inc.branch4(x_test)
        out = inc(x_test)
        print(f"分支1 (1×1):          {b1.shape[1]} 通道")
        print(f"分支2 (1×1→3×3):      {b2.shape[1]} 通道")
        print(f"分支3 (1×1→5×5):      {b3.shape[1]} 通道")
        print(f"分支4 (Pool→1×1):     {b4.shape[1]} 通道")
        print(f"拼接后总通道:          {out.shape[1]} (={b1.shape[1]}+{b2.shape[1]}+{b3.shape[1]}+{b4.shape[1]})")

    # ---- 1×1 降维效率分析 ----
    print("\n--- 1×1 卷积降维效率分析 ---")
    H, W = 14, 14
    C_in, C_out = 256, 128
    # 直接 5×5 卷积
    ops_direct = C_in * C_out * 5 * 5 * H * W
    # 1×1 降维(64) + 5×5
    ops_reduce = C_in * 64 * 1 * 1 * H * W + 64 * C_out * 5 * 5 * H * W
    print(f"直接 5×5:      {ops_direct:,} 次乘法")
    print(f"1×1(64)+5×5:   {ops_reduce:,} 次乘法")
    print(f"节省:           {(1-ops_reduce/ops_direct)*100:.0f}% 计算量")

    print("\n--- GoogLeNet 的关键创新 ---")
    print("  1. Inception 模块: 多尺度并行卷积")
    print("  2. 1×1 卷积降维: 大幅减少参数和计算量")
    print("  3. 全局平均池化: 取代全连接层（+0.6% 准确率）")
    print("  4. 辅助分类器: 对抗梯度消失（后被 [[../09_ResNet/ResNet|ResNet]] 残差连接取代）")
