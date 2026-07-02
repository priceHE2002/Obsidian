---
tags:
  - 代码
  - PyTorch
created: 2026-07-02
---

# VAE (Variational Auto-Encoder) - 代码实现

> 本文档包含 PyTorch 教学实现，可在 Obsidian Execute Code 插件中直接运行代码块。

```python
"""
VAE (Variational Auto-Encoder) — PyTorch 完整实现
===================================================
论文: "Auto-Encoding Variational Bayes"
      (Kingma & Welling, UvA, ICLR 2014)
核心贡献: 重参数化技巧 + 神经 encoder/decoder + ELBO 优化，
        实现端到端的潜变量生成模型训练。

架构: Encoder (MLP: x → μ, logσ²) → 重参数化 (z = μ + σ⊙ε)
      → Decoder (MLP: z → x̂) → ELBO loss (KL项 + 重构项)

与 [[VAE.md|VAE 详解]] 配套阅读。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os


# ==============================================================================
# 1. VAE 完整模型
# ==============================================================================
class VAE(nn.Module):
    """Variational Auto-Encoder (VAE).

    参数化:
        p(z)    = N(0, I)                                    (先验, 固定)
        q_φ(z|x) = N(μ_φ(x), diag(σ²_φ(x)))                  (编码器 / 识别模型)
        p_θ(x|z) = Bernoulli(logits = decoder(z)) 或 Gaussian  (解码器 / 生成模型)

    核心设计决策:
        1. 高斯先验 + 对角协方差后验 → KL 项有解析解，不需要 Monte Carlo 估计
        2. Encoder 输出 log σ² 而非 σ (数值更稳定, softplus 保证正值)
        3. 重参数化: z = μ + σ ⊙ ε, ε ∼ N(0, I) (让采样对 φ 可微)

    Args:
        input_dim:  输入 x 的维度 (MNIST: 784)
        hidden_dim: encoder/decoder 的隐藏层维度 (默认 400)
        latent_dim: 潜变量 z 的维度 (默认 20)
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # ---- Encoder: x → μ(x), log σ²(x) ----
        # 问: 为什么输出 log σ² 而不是 σ?
        # 答: (1) log σ² ∈ ℝ 无约束，σ = exp(0.5·log σ²) 自动为正
        #     (2) 数值稳定性: KL 项中 log σ² 直接可用，减少浮点误差累积
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)        # μ(x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)    # log σ²(x)

        # ---- Decoder: z → x̂ (Bernoulli logits) ----
        # 对于 MNIST (二值化图像)，使用 Bernoulli 解码器
        # 对于连续数据，可替换为 Gaussian 解码器 (需同时输出 μ 和 σ)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)           # 输出 logits

    def encode(self, x):
        """编码: x → μ(x), log σ²(x).

        形状:
            x:      (batch, input_dim)
            返回 μ: (batch, latent_dim)
            返回 logvar: (batch, latent_dim)
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧: z = μ + σ ⊙ ε, ε ∼ N(0, I).

        这是 VAE 最核心的技术创新。为什么需要它?
        - 直接采样 z ∼ N(μ, σ²) 不可导 (采样操作没有梯度)
        - 将随机性移到 ε (独立于 φ 的分布), z = μ + σ⊙ε
        - 梯度可以穿过 μ 和 σ 反向传播

        数学:
            z = μ + σ ⊙ ε, ε ∼ N(0, I), σ = exp(0.5·log σ²)

        Args:
            mu:     (batch, latent_dim) 均值
            logvar: (batch, latent_dim) log 方差
        Returns:
            z: (batch, latent_dim) 采样后的潜变量
        """
        std = torch.exp(0.5 * logvar)           # σ = exp(0.5·log σ²)
        eps = torch.randn_like(std)             # ε ∼ N(0, I)
        return mu + std * eps

    def decode(self, z):
        """解码: z → x̂ (Bernoulli logits).

        Bernoulli 解码器的含义:
            p(x_i=1 | z) = sigmoid(decoder(z)_i)
            log p(x|z) = Σ_i [x_i log p_i + (1-x_i) log(1-p_i)]
        这等价于 BCE 损失 (用 logits 输入时, BCELoss 内部也做 sigmoid)。

        对于连续数据 (如 Frey Face), 应使用 Gaussian 解码器:
            解码器额外输出 log σ², 损失用 MSE 或 NLL。
        """
        h = F.relu(self.fc3(z))
        return self.fc4(h)  # 返回 logits (不经 sigmoid, 留给 BCEWithLogitsLoss)

    def forward(self, x):
        """前向传播: encode → reparameterize → decode.

        Returns:
            x_recon: (batch, input_dim) 重构 logits
            mu:      (batch, latent_dim) 均值
            logvar:  (batch, latent_dim) log 方差
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device='cpu'):
        """从先验 p(z) = N(0, I) 采样, 生成新样本。

        流程:
            z ∼ N(0, I) → decoder(z) → sigmoid 得到概率 → Bernoulli 采样

        Returns:
            samples: (num_samples, input_dim) 生成的样本 (0/1 for Bernoulli)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            logits = self.decode(z)
            return torch.sigmoid(logits)


# ==============================================================================
# 2. ELBO 损失函数
# ==============================================================================
def vae_loss(x, x_recon_logits, mu, logvar):
    """VAE 的 ELBO 损失 (= 负的变分下界, 需 minimize)。

    ELBO 分解 (per datapoint):
        L(θ,φ; x) = E_q[log p_θ(x|z)] - D_KL(q_φ(z|x) || p_θ(z))
                     \________重构项________/   \_______KL正则化项______/

    KL 解析解 (Gaussian 先验 + Gaussian 后验):
        -D_KL = ½ · Σ_j (1 + log σ²_j - μ²_j - σ²_j)

    实现细节:
        1. 重构项用 BCEWithLogitsLoss (数值稳定: 内部 fusion sigmoid+BCE)
        2. KL 项按 latent_dim 求和, batch 求平均
        3. 最终损失 = 重构项 (per-pixel average) + KL 项 (per-latent-dim average)

    Args:
        x:              (batch, input_dim) 原始输入 (二值: 0-1)
        x_recon_logits: (batch, input_dim) 重构 logits (未经 sigmoid)
        mu:             (batch, latent_dim) encoder 输出均值
        logvar:         (batch, latent_dim) encoder 输出 log 方差

    Returns:
        total_loss:     标量, 平均 ELBO 的负数
        recon_loss:     标量, 平均重构误差 (用于日志)
        kl_loss:        标量, 平均 KL 散度 (用于日志)
    """
    batch_size = x.size(0)

    # ---- 重构项: E_q[log p(x|z)] (maximize → minimize negative) ----
    # reduction='sum': 对每个像素求和 → /batch_size 得平均
    # 为什么用 sum 而非 mean? 保持与论文公式一致 (对像素求和)
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon_logits, x, reduction='sum'
    ) / batch_size

    # ---- KL 正则项: D_KL(q(z|x) || p(z)) (minimize) ----
    # 对于 Gaussian prior + Gaussian posterior, 解析解:
    #   D_KL = -½ · Σ_j (1 + log σ²_j - μ²_j - σ²_j)
    # 先对 latent_dim 求和 (/latent_dim), 再对 batch 平均 (/batch_size)
    # → 实际实现: sum over all dims / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / batch_size

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


# ==============================================================================
# 3. 训练 + 评估
# ==============================================================================
def train_epoch(model, dataloader, optimizer, device):
    """单 epoch 训练循环。

    关健设计: L=1 (每个数据点取一个 z), M=batch_size
    论文发现: L=1 + 足够大的 batch_size (≥100) 已足够，
            因为梯度噪声在 batch 维度上被平均了。
    """
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(data.size(0), -1).to(device)  # (B, 784)
        optimizer.zero_grad()

        x_recon, mu, logvar = model(data)
        loss, recon, kl = vae_loss(data, x_recon, mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


@torch.no_grad()
def evaluate(model, dataloader, device):
    """验证集 ELBO 评估。"""
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for data, _ in dataloader:
        data = data.view(data.size(0), -1).to(device)
        x_recon, mu, logvar = model(data)
        loss, recon, kl = vae_loss(data, x_recon, mu, logvar)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n = len(dataloader)
    return total_loss / n, total_recon / n, total_kl / n


# ==============================================================================
# 4. 主训练脚本 (Example: MNIST)
# ==============================================================================
if __name__ == '__main__':
    # ---- 配置 ----
    BATCH_SIZE = 128          # 论文建议 M ≥ 100 (L=1 时)
    LATENT_DIM = 20           # 潜变量维度 (论文测了 3,5,10,20,200)
    HIDDEN_DIM = 400          # encoder/decoder 隐藏层维度
    EPOCHS = 20               # 约 20 epochs 收敛
    LR = 1e-3                 # Adam 学习率

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- 数据加载 ----
    # MNIST: 动态二值化 (像素 ≤ 0.5 → 0, else 1)
    # 注: 这是 VAE 论文的原始做法, 与静态二值化 (固定阈值) 不同
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # 动态二值化
    ])

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # ---- 模型初始化 ----
    model = VAE(
        input_dim=784,   # 28×28
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # 注: 论文使用 Adagrad, Adam 在现代实践中更常用且表现更稳定

    # ---- 训练循环 ----
    for epoch in range(1, EPOCHS + 1):
        train_l, train_r, train_kl = train_epoch(
            model, train_loader, optimizer, device
        )
        test_l, test_r, test_kl = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_l:.2f} (R: {train_r:.2f}, KL: {train_kl:.2f}) | "
              f"Test Loss: {test_l:.2f} (R: {test_r:.2f}, KL: {test_kl:.2f})")

    # ---- 生成样本 ----
    samples = model.sample(64, device=device)
    save_image(samples.view(64, 1, 28, 28), 'vae_samples.png', nrow=8)
    print("Generated samples saved to vae_samples.png")


# ==============================================================================
# 5. Gaussian Decoder 版本 (连续数据, 如 Frey Face)
# ==============================================================================
class VAEGaussian(nn.Module):
    """VAE with Gaussian decoder (for continuous data).

    与 Bernoulli 版本的区别:
        - Decoder 额外输出 log σ²_dec, 用于计算 Gaussian NLL
        - 重构项变为: -log N(x | decoder_μ(z), decoder_σ²(z))
    """

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder (同 Bernoulli 版本)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: z → μ_dec(x̂), log σ²_dec(x̂)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc_mu_dec = nn.Linear(hidden_dim, input_dim)       # μ_dec
        self.fc_logvar_dec = nn.Linear(hidden_dim, input_dim)   # log σ²_dec

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.relu(self.fc3(z))
        mu_dec = torch.sigmoid(self.fc_mu_dec(h))        # 约束在 [0, 1]
        logvar_dec = self.fc_logvar_dec(h)                # log σ²_dec (无约束)
        return mu_dec, logvar_dec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        mu_dec, logvar_dec = self.decode(z)
        return mu_dec, logvar_dec, mu, logvar


def vae_gaussian_loss(x, mu_dec, logvar_dec, mu, logvar):
    """Gaussian decoder 的 ELBO 损失。

    重构项: E_q[-log N(x | μ_dec, σ²_dec)]
           = ½ Σ [(x - μ_dec)²/σ²_dec + log σ²_dec + log 2π]
    等价于加权的 MSE + σ 相关的正则项。"""
    batch_size = x.size(0)

    # 重构项: Gaussian negative log-likelihood
    recon_loss = 0.5 * torch.sum(
        (x - mu_dec).pow(2) / logvar_dec.exp() + logvar_dec +
        torch.log(torch.tensor(2 * torch.pi))
    ) / batch_size

    # KL 项 (同上)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / batch_size

    return recon_loss + kl_loss, recon_loss, kl_loss


# ==============================================================================
# 6. 诊断与调试工具
# ==============================================================================
def compute_elbo_components(model, x, device):
    """分析 ELBO 各分量的数值，用于诊断训练问题。

    常见诊断:
        - KL → 0: 后验坍缩 (decoder 太强, 忽略 z)。尝试 KL annealing
        - KL → +∞: 正常训练 (KL 应从 0 逐渐增大)
        - Recon → NaN: 学习率过大或梯度爆炸。检查 LR, 添加 grad clipping
        - Recon 不下降: decoder 容量不足。增加 hidden_dim 或层数

    Returns:
        dict: {recon, kl, elbo, mu_norm, logvar_mean}
    """
    model.eval()
    with torch.no_grad():
        x = x.view(x.size(0), -1).to(device)
        x_recon, mu, logvar = model(x)
        loss, recon, kl_loss = vae_loss(x, x_recon, mu, logvar)

    return {
        'elbo': -loss.item(),
        'recon': recon.item(),
        'kl': kl_loss.item(),
        'mu_norm': mu.norm(dim=1).mean().item(),
        'logvar_mean': logvar.mean().item(),
        'sigma_mean': logvar.exp().sqrt().mean().item(),
    }


# ==============================================================================
# 7. 概念验证: 重参数化技巧的梯度验证
# ==============================================================================
def verify_reparameterization_grad():
    """验证重参数化技巧的梯度是否正确。

    若直接 z ∼ N(μ, σ²), ∇_μ E[f(z)] 不能用朴素的 Monte Carlo 估计:
        ∇_μ E[f(z)] ≠ E[∇_μ f(z)] (因为 q(z;μ) 也依赖 μ)

    重参数化后: z = μ + σ·ε, ε ∼ N(0, 1), ∇_μ 穿过 μ 反向传播。
    """
    torch.manual_seed(42)
    mu = torch.tensor([1.0], requires_grad=True)
    logvar = torch.tensor([0.0], requires_grad=True)  # σ² = 1

    def naive_loss(mu, logvar):
        """直接采样 → 梯度存在但方差极高。"""
        z = torch.normal(mu, torch.exp(0.5 * logvar))
        return z.pow(2).sum()

    def reparam_loss(mu, logvar):
        """重参数化 → 低方差梯度。"""
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z.pow(2).sum()

    # 验证两种方法都产生可微的梯度
    loss1 = naive_loss(mu, logvar)
    loss1.backward()
    print(f"Naive sampling grad (μ): {mu.grad:.4f}")  # 高方差, 波动大

    mu.grad = None
    logvar.grad = None

    loss2 = reparam_loss(mu, logvar)
    loss2.backward()
    print(f"Reparam trick grad (μ):    {mu.grad:.4f}")  # 更稳定

    mu.grad = None
    logvar.grad = None

    # 解析梯度: E[(μ + ε)²], ∂/∂μ = 2μ = 2.0
    # 重参数化梯度应接近 2.0 (单样本有噪声, 批量取样后可精确匹配)
    print(f"Analytical gradient (μ):   2.0000")

    n_trials = 100000
    grad_sum = 0.0
    for _ in range(n_trials):
        mu_g = torch.tensor([1.0], requires_grad=True)
        logvar_g = torch.tensor([0.0], requires_grad=True)
        eps = torch.randn_like(mu_g)
        z = mu_g + torch.exp(0.5 * logvar_g) * eps
        loss = z.pow(2).sum()
        loss.backward()
        grad_sum += mu_g.grad.item()
    print(f"Monte Carlo estimate (n={n_trials}): {grad_sum/n_trials:.4f} "
          f"(→ 2.0 as n→∞)")
```

---

## 关键设计说明

### 1. 为什么 encoder 输出 `log σ²` 而非 `σ`？

`log σ²` 的值域是 $\mathbb{R}$（无约束），$\sigma = \exp(0.5 \cdot \log \sigma^2)$ 自动为正。如果用 $\sigma$ 则需要 softplus 或 clamp——增加额外约束。且在 KL 项中 $\log \sigma^2$ 直接可用，避免数值误差累积。

### 2. 为什么重构项用 `reduction='sum'`？

与论文公式一致：$\log p(x|z) = \sum_{i=1}^D [x_i \log y_i + (1-x_i)\log(1-y_i)]$ 是对所有像素求和的。（除以 batch_size 得到 per-datapoint 平均。）

### 3. KL 项为什么可能为负？

$D_{KL}$ 本身 $\ge 0$，但实现中是 $-D_{KL}$（取负号加入总损失）。所以代码中的 `kl_loss` 应为**正数**（因为加了负号使损失最小化等价于 KL 最大化）。如果观察到 kl_loss 为负，检查符号。

### 4. VAE 的 ELBO 下界性质

$$\log p(x) = D_{KL}(q(z|x) \parallel p(z|x)) + \mathcal{L}(x) \ge \mathcal{L}(x)$$

训练过程中 $\mathcal{L}(x)$ 增大不保证 $\log p(x)$ 增大——KL gap 可能同时变大。这就是为什么评估 VAE 时常使用 IWAE 或 AIS 估计真实 log 似然。
