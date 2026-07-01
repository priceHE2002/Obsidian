"""
VGG: Very Deep Convolutional Networks
=====================================
论文: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
      (Simonyan & Zisserman, ICLR 2015)
核心贡献: 证明网络深度对视觉识别至关重要，提出极简设计哲学——全部使用 3×3 卷积
         + 2×2 max pooling。VGG-16/VGG-19 成为视觉特征的标准提取器。
架构: VGG-16: 13个卷积层 + 3个全连接层, 138M参数
代码结构:
  1. VGG16 - 完整实现
  2. VGG19 - 更深的变体

核心设计原则:
  - 全部使用 3×3 卷积（最小可能的感受野）
  - 多个 3×3 堆叠等价于更大的感受野（2×3×3 = 5×5, 3×3×3 = 7×7）
  - 但参数更少（2×9C²=18C² vs 25C²）且非线性更强（更多 ReLU）
  - 通道数逐阶段翻倍: 64→128→256→512→512

与 [[../06_AlexNet/AlexNet|AlexNet]] 的关系: 用统一小核替换了 AlexNet 的大核（11×11→3×3 堆叠）
与 [[../09_ResNet/ResNet|ResNet]] 的关系: ResNet 继承了 VGG 的 3×3 卷积设计
"""

import torch
import torch.nn as nn


# ==============================================================================
# VGG-16 完整架构
# ==============================================================================
class VGG16(nn.Module):
    """
    VGG-16 架构

    5 个卷积阶段 → 3 个全连接层

    逐阶段结构:
    - Stage1: [3×3, 64]×2  + MaxPool → 112×112×64
    - Stage2: [3×3, 128]×2 + MaxPool → 56×56×128
    - Stage3: [3×3, 256]×3 + MaxPool → 28×28×256
    - Stage4: [3×3, 512]×3 + MaxPool → 14×14×512
    - Stage5: [3×3, 512]×3 + MaxPool → 7×7×512

    参数量: ~138M（其中 ~123M 来自全连接层，占 89%）

    小卷积核堆叠的优势:
    2 个 3×3 堆叠 = 5×5 感受野，参数: 2×9C² = 18C² < 25C² (节省 28%)
    3 个 3×3 堆叠 = 7×7 感受野，参数: 3×9C² = 27C² < 49C² (节省 45%)
    且每层有独立的 ReLU → 更多非线性 → 更强的判别能力
    """

    def __init__(self, num_classes: int = 1000, init_weights: bool = True):
        super().__init__()

        # 卷积层配置: (输出通道数, 重复次数)
        # VGG-16: 2,2,3,3,3（共 13 层卷积）
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = self._make_layers(cfg)

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg: list) -> nn.Sequential:
        """根据配置列表构建卷积层"""
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                # MaxPooling: 2×2, stride=2
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # 3×3 卷积，padding=1（保持空间分辨率）
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([
                    conv2d,
                    nn.ReLU(inplace=True),
                ])
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化（2015 年论文未使用此方法，但现代实践推荐）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


class VGG19(nn.Module):
    """
    VGG-19 架构

    相比 VGG-16: Stage3/4/5 各增加一层 3×3 卷积
    卷积层数: 13 → 16, 参数量: 138M → 144M
    性能几乎相同（top-5 error 同为 7.3%） → 16-19 层是当时 CNN 深度的饱和点
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        cfg = [64, 64, 'M', 128, 128, 'M',
               256, 256, 256, 256, 'M',        # VGG-19: 4 层卷积
               512, 512, 512, 512, 'M',        # VGG-19: 4 层卷积
               512, 512, 512, 512, 'M']         # VGG-19: 4 层卷积

        self.features = self._make_layers(cfg)

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, cfg: list) -> nn.Sequential:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==============================================================================
# 小卷积核堆叠感受野分析
# ==============================================================================
def compute_receptive_field(layers: list[tuple[int, int, int]]) -> int:
    """
    计算卷积网络的有效感受野

    参数: [(kernel_size, stride, padding), ...] 列表
    感受野递推公式: RF_{l} = RF_{l-1} + (k_l - 1) × Π_{i=1}^{l-1} s_i
    """
    rf, stride = 1, 1
    for k, s, p in layers:
        rf = rf + (k - 1) * stride
        stride *= s
    return rf


# ==============================================================================
# 演示代码
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VGG 架构演示")
    print("=" * 60)

    # ---- 1. VGG-16 ----
    print("\n--- 1. VGG-16 ---")
    model = VGG16(num_classes=1000)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"VGG-16 参数量: {total_params:,} (~138M)")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        feats = model.features(x)
        print(f"输入: {x.shape}")
        print(f"特征提取后: {feats.shape} (7×7×512)")

        logits = model(x)
        print(f"分类输出: {logits.shape}")

    # ---- 2. 小卷积核堆叠的优势分析 ----
    print("\n--- 2. 小卷积核堆叠 vs 大卷积核 ---")
    # 2 个 3×3 堆叠 = 5×5 感受野
    layers_2x3 = [(3, 1, 1), (3, 1, 1)]
    rf_2x3 = compute_receptive_field([(k, s, p) for k, s, p in layers_2x3])
    print(f"2×3×3 堆叠感受野: {rf_2x3}×{rf_2x3} (等价于 5×5 卷积)")
    # 3 个 3×3 堆叠 = 7×7 感受野
    layers_3x3 = [(3, 1, 1), (3, 1, 1), (3, 1, 1)]
    rf_3x3 = compute_receptive_field([(k, s, p) for k, s, p in layers_3x3])
    print(f"3×3×3 堆叠感受野: {rf_3x3}×{rf_3x3} (等价于 7×7 卷积)")

    # 参数量对比
    C = 256  # 假设输入输出通道数
    params_5x5 = C * C * 5 * 5          # 1 个 5×5
    params_2x3x3 = 2 * C * C * 3 * 3    # 2 个 3×3
    params_7x7 = C * C * 7 * 7          # 1 个 7×7
    params_3x3x3 = 3 * C * C * 3 * 3    # 3 个 3×3
    print(f"\n参数量对比 (C=256 通道):")
    print(f"  5×5 卷积:      {params_5x5:,}")
    print(f"  2×3×3 堆叠:    {params_2x3x3:,} (节省 {(1-params_2x3x3/params_5x5)*100:.0f}%)")
    print(f"  7×7 卷积:      {params_7x7:,}")
    print(f"  3×3×3 堆叠:    {params_3x3x3:,} (节省 {(1-params_3x3x3/params_7x7)*100:.0f}%)")

    # 非线性对比
    print(f"\n非线性层数量:")
    print(f"  5×5 单卷积:    1 个 ReLU")
    print(f"  2×3×3 堆叠:    2 个 ReLU → 更强的判别力")
    print(f"  7×7 单卷积:    1 个 ReLU")
    print(f"  3×3×3 堆叠:    3 个 ReLU → 更强的判别力")

    # ---- 3. VGG-19 ----
    print("\n--- 3. VGG-19 vs VGG-16 ---")
    model19 = VGG19(num_classes=1000)
    params19 = sum(p.numel() for p in model19.parameters())
    print(f"VGG-16 参数量: {total_params:,}")
    print(f"VGG-19 参数量: {params19:,} (+{(params19-total_params)/1e6:.1f}M)")
    print("性能几乎相同 (top-5 error 同为 7.3%)")
    print("→ 16-19 层是 VGG 架构的深度饱和点")

    print("\n--- VGG 的架构遗产 ---")
    print("  1. 全 3×3 卷积 → [[../09_ResNet/ResNet|ResNet]] 完全继承")
    print("  2. 多尺度训练 → 现代数据增强的基础")
    print("  3. VGG-16 预训练权重 → 风格迁移/感知损失的标准特征提取器")
    print("  4. '预训练+微调'范式 → CV 领域的标准做法")
