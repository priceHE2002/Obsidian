---
title: GAN
tags:
  - 基础知识
  - 深度学习
  - 图像算法
  - 面试题
source: "深度学习面试题-图像算法 (1).doc"
created: 2026-07-01
up: "[[00_基础知识/深度学习面试题-图像算法/深度学习面试题-图像算法|深度学习面试题-图像算法]]"
---

# 6. GAN

生成式对抗网络（GAN）由生成器 $G$ 和判别器 $D$ 组成。生成器把随机噪声 $z$ 映射成候选样本 $G(z)$，判别器判断输入是真实样本还是生成样本。判别器希望分清真假，生成器希望骗过判别器，二者构成一个 minimax 对抗过程。

经典 GAN 目标函数：

$$
\min_G \max_D V(D,G)
= \mathbb{E}_{x\sim p_{data}}[\log D(x)]
+ \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
$$

训练时通常交替更新：先固定 $G$ 训练 $D$，再固定 $D$ 训练 $G$。

```python
import torch.nn as nn

# 一个简单的 DCGAN 生成器示例
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),     # 4x4
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),       # 8x8
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),       # 16x16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),         # 32x32
            nn.Tanh()                                     # 输出 [-1, 1]
        )

    def forward(self, z):
        return self.net(z)

z = torch.randn(1, 100, 1, 1)
G = Generator()
fake_img = G(z)
print(f"生成图像形状: {fake_img.shape}")  # [1, 3, 32, 32]
```

GAN 相关的技巧：

![[00_基础知识/深度学习面试题-图像算法/assets/image-38.png]]

## 6.1 生成器

它将一个向量（来自潜在空间，训练过程中对其随机采样）转换为一张候选图像。GAN 常见的诸多问题之一，就是生成器"卡在"看似噪声的生成图像上。

生成器的目标不是直接最小化像素级误差，而是让判别器更难区分生成样本和真实样本。训练不稳定时，生成器可能出现两类问题：

1. 生成结果一直像噪声，说明 $G$ 没学到有效数据分布。
2. 模式崩塌（mode collapse），即 $G$ 只会生成少数几类看起来能骗过 $D$ 的样本。

可尝试的缓解方法包括更稳定的损失（如 WGAN / WGAN-GP）、谱归一化、改进网络结构、调节 $G$ 和 $D$ 的训练步数、加入噪声或使用 label smoothing。Dropout 有时可用，但不是 GAN 稳定训练的通用解。

![[00_基础知识/深度学习面试题-图像算法/assets/image-39.png]]

## 6.2 判别器

判别器模型接收一张候选图像（真实的或合成的）作为输入，并将其划分到这两个类别之一："生成图像"或"来自训练集的真实图像"。

判别器可以看作二分类器，输出 $D(x)$ 表示样本为真的概率。训练早期如果判别器太强，生成器得到的梯度可能很弱；如果判别器太弱，又无法给生成器提供有价值的学习信号。因此 GAN 的关键是维持二者相对平衡。

![[00_基础知识/深度学习面试题-图像算法/assets/image-40.png]]

## 6.3 训练技巧

- 输入规范化到 $[-1,1]$ 之间，生成器最后一层常用 tanh（如果输出图像范围是 $[-1,1]$）。
- 使用 Wasserstein GAN / WGAN-GP 等更稳定的损失函数，缓解 JS 散度在分布不重叠时梯度消失的问题。
- 如果有标签数据的话，尽量使用标签。也有人提出使用反转标签效果很好。另外使用标签平滑，单边标签平滑或者双边标签平滑。
- 使用 mini-batch norm。如果不用 Batch Norm，可以尝试 Instance Norm、Layer Norm、Weight Norm 或 Spectral Norm；其中 Spectral Norm 常用于稳定判别器。
- 判别器中常用 LeakyReLU，生成器中可用 ReLU/LeakyReLU；很多 DCGAN 风格结构会用 stride convolution 替代 pooling，用转置卷积或上采样 + 卷积做生成。
- 优化器尽量选择 Adam，学习率不要设置太大，初始 1e-4 可以参考，另外可以随着训练进行不断缩小学习率。
- 给 $D$ 的网络层增加高斯噪声，相当于是一种正则。

补充：训练 GAN 时不要只看 loss，因为 $G$ 和 $D$ 的 loss 是动态博弈，不一定单调下降。更可靠的评估包括生成样本可视化、FID、IS，以及检查是否出现模式崩塌。
