---
tags:
  - 论文
  - 归一化
  - 训练加速
  - CNN
  - Internal Covariate Shift
created: 2026-06-30
paper_title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
paper_authors: "Sergey Ioffe, Christian Szegedy"
paper_year: 2015
paper_venue: "ICML 2015"
paper_citations: "~60,000+"
paper_url: "https://arxiv.org/abs/1502.03167"
github: ""
---

# Batch Normalization

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**
*Sergey Ioffe, Christian Szegedy | Google Inc. | ICML 2015 | arXiv 1502.03167*

> Batch Normalization is the single most important training stability technique for deep CNNs. By normalizing each layer's inputs to zero mean and unit variance over each mini-batch, it allows 5-30x higher learning rates, reduces sensitivity to initialization, provides regularization, and enables training with saturating nonlinearities (sigmoid) in deep networks. Its key limitation -- dependence on batch statistics -- makes it unsuitable for RNNs and Transformers, leading to Norm variants like LN, GN, and RMSNorm.

---

## 一、Background/Core Idea

### 1.1 The Internal Covariate Shift Problem

Before Batch Normalization (BN), training deep neural networks faced a fundamental obstacle: as training progresses, the parameters of earlier layers are updated, which shifts the distribution of activations seen by later layers. Ioffe & Szegedy coined the term **Internal Covariate Shift (ICS)** to describe this phenomenon.

Mathematically, let a layer compute $x = Wu + b$ where $u$ is the input from the previous layer. As the parameters of layers below change during training, the distribution of $u$ (and consequently $x$) changes. Later layers must continuously adapt to the shifting input distribution, which:

| Consequence | Explanation | Severity |
|---|---|---|
| Slow training | Need small learning rates to avoid destabilizing the distribution | High |
| Saturation with sigmoid/tanh | When $|x|$ grows large, $g'(x) \to 0$ causing vanishing gradients | Critical |
| Sensitivity to initialization | Poor init pushes activations into saturation regimes | High |
| Need for Dropout | Extra regularization needed because gradients are noisy | Moderate |

The paper formalizes this: "We refer to the change in the distributions of internal nodes of a deep network, in the course of training, as **Internal Covariate Shift**. Eliminating it offers a promise of faster training."

### 1.2 Why Full Whitening Is Infeasible

The ideal solution would be to whiten each layer's inputs: compute the covariance matrix $\text{Cov}[x] = \mathbb{E}_{x \in X}[xx^T] - \mathbb{E}[x]\mathbb{E}[x]^T$ and its inverse square root to produce $\text{Cov}[x]^{-1/2}(x - \mathbb{E}[x])$. However:

1. **Computational cost**: Computing the covariance matrix and its inverse square root for each layer at every update is prohibitively expensive
2. **Gradient complexity**: Backpropagating through the whitening transform requires computing Jacobians $\frac{\partial \text{Norm}(x, X)}{\partial x}$ and $\frac{\partial \text{Norm}(x, X)}{\partial X}$ -- the latter term depends on the entire training set
3. **Non-differentiability**: Full whitening is not easily differentiable in a way that composes with SGD

BN makes two critical simplifications:
- **Per-dimension normalization** instead of full whitening: normalize each scalar feature independently to zero mean and unit variance
- **Mini-batch estimation**: compute statistics from the current mini-batch rather than the full dataset

## 二、Method/Architecture/Technical Contribution

### 2.1 The Batch Normalization Transform

For a mini-batch $\mathcal{B} = \{x_{1 \dots m}\}$ of size $m$, the BN transform consists of four sequential operations:

**Step 1: Mini-batch mean**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**Step 2: Mini-batch variance**
$$\sigma^2_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

**Step 3: Normalize**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$
where $\epsilon$ is a small constant (default $10^{-5}$) for numerical stability.

**Step 4: Scale and shift**
$$y_i = \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma,\beta}(x_i)$$

The learnable parameters $\gamma$ (scale) and $\beta$ (shift) are crucial: they restore the network's representational power. If the normalization removes useful information (e.g., shifting all sigmoid inputs to the linear regime), the network can learn to "undo" the normalization by setting $\gamma = \sqrt{\text{Var}[x]}$ and $\beta = \mathbb{E}[x]$, recovering the identity transform.

### 2.2 Backward Pass: Gradient Flow Through BN

A key design criterion is that BN must be **differentiable**. The paper provides the full chain rule derivation:

$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

$$\frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_\mathcal{B}) \cdot \frac{-1}{2} (\sigma^2_\mathcal{B} + \epsilon)^{-3/2}$$

$$\frac{\partial \ell}{\partial \mu_\mathcal{B}} = \left( \sum_{i=1}^{m} \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \right) + \frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} \cdot \frac{\sum_{i=1}^{m} -2(x_i - \mu_\mathcal{B})}{m}$$

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} + \frac{\partial \ell}{\partial \sigma^2_\mathcal{B}} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \ell}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \ell}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \ell}{\partial y_i}$$

Critical observation: the gradient $\partial \ell / \partial x_i$ depends on ALL examples in the mini-batch through $\mu_\mathcal{B}$ and $\sigma^2_\mathcal{B}$, not just on $x_i$ itself. This batch dependency is fundamental to BN's behavior and also what limits its applicability.

### 2.3 Training vs. Inference Behavior

| Phase | Statistics Used | Key Property |
|---|---|---|
| **Training** | Current mini-batch $\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$ | Stochastic -- normalized value varies per batch |
| **Inference** | Running mean $\mathbb{E}[x]$, running variance $\text{Var}[x]$ accumulated during training | Deterministic -- fixed linear transformation |

During training, BN maintains **running averages** (typically with momentum 0.9 or 0.99):
$$\mathbb{E}[x] \leftarrow (1 - \text{momentum}) \cdot \mathbb{E}[x] + \text{momentum} \cdot \mu_\mathcal{B}$$
$$\text{Var}[x] \leftarrow (1 - \text{momentum}) \cdot \text{Var}[x] + \text{momentum} \cdot \sigma^2_\mathcal{B}$$

At inference time:
$$\text{BN}(x) = \gamma \cdot \frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} + \beta$$

This **training-inference discrepancy** is a fundamental limitation. In contrast, [[Layer Normalization]] and [[RMSNorm]] perform identically at train and test time.

### 2.4 BN in Convolutional Layers

In CNNs, the BN transform must respect the convolutional property -- different spatial locations of the same feature map share the same filter and should be normalized identically. For a feature map of size $p \times q$ with mini-batch size $m$, the effective batch size becomes $m' = m \cdot p \cdot q$. Statistics are computed over:
- All examples in the mini-batch (dimension N)
- All spatial locations (dimensions H, W)

Learnable parameters $\gamma^{(k)}$ and $\beta^{(k)}$ are **per-channel** (one pair per feature map), not per-activation. This preserves the translation equivariance of convolution.

### 2.5 BN Enables Higher Learning Rates -- The Mechanism

The paper demonstrates empirically but does not fully explain theoretically why BN enables 5-30x higher learning rates. Later work (Santurkar et al., 2018, "How Does Batch Normalization Help Optimization?") showed that:

1. BN **smooths the loss landscape**: the Lipschitz constant of the loss becomes smaller, meaning gradients change less abruptly
2. BN prevents the amplification of parameter changes: "small changes to the parameters from amplifying into larger and suboptimal changes in activations in gradients"
3. BN prevents getting stuck in saturated regimes of nonlinearities

The practical effect: with BN, the learning rate can be increased from 0.0015 (Inception baseline) to 0.045 (BN-x30) on ImageNet.

## 三、Experiments and Key Findings

### 3.1 MNIST Sigmoid Network

The first experiment trains a 3-layer fully-connected network (100 hidden units each) with **sigmoid** nonlinearities on MNIST:

| Metric | Without BN | With BN |
|---|---|---|
| Test accuracy after 50K steps | Lower | Higher (~same steps to higher accuracy) |
| Activation distribution stability | Mean/variance shift significantly over training | Distribution much more stable |
| Sigmoid saturation | Activations drift into saturation | Activations kept near zero-mean |

Figure 1(c) in the paper shows the evolution of input distributions to a typical sigmoid: without BN, the {15th, 50th, 85th} percentiles shift dramatically; with BN, they remain stable. This was the direct empirical validation of the internal covariate shift hypothesis.

### 3.2 ImageNet Classification (Inception Architecture)

The main experiment uses a modified Inception network. The paper tests several configurations:

| Model | Learning Rate | Steps to 72.2% | Max Accuracy | Notes |
|---|---|---|---|---|
| **Inception** | 0.0015 | 31.0M steps | 72.2% | Baseline |
| **BN-Baseline** | 0.0015 | 13.3M steps | 72.7% | Same LR, BN alone halves steps |
| **BN-x5** | 0.0075 | 2.1M steps | 73.0% | 5x LR, 14x fewer steps |
| **BN-x30** | 0.045 | 2.7M steps | 74.8% | 30x LR, higher final accuracy |
| **BN-x5-Sigmoid** | 0.0075 | - | 69.8% | Sigmoid instead of ReLU, still trainable |

Key findings:
- **BN-Baseline** (same LR as Inception) needs only 43% of the training steps, proving BN alone accelerates training
- **BN-x5** with 5x LR achieves the same accuracy in just 2.1M steps (14x fewer)
- **BN-x30** achieves the highest max accuracy (74.8%), likely because larger LR helps escape sharp local minima
- **BN-x5-Sigmoid** reaches 69.8% without ReLU -- this was remarkable because training deep sigmoid networks was previously considered nearly impossible

### 3.3 Ensemble and SOTA Results

The ensemble of 6 BN-Inception models achieved:
- **4.9% top-5 validation error** (single-crop: 7.82%)
- **4.82% top-5 test error** on the 100K-image test set
- This **exceeded human-level performance** at the time

The ensemble was particularly notable because BN's regularization effect allowed each individual model to be more diverse.

### 3.4 Additional Findings

1. **Dropout removal**: With BN, the Dropout ratio could be reduced from 0.5 to 0.2 or removed entirely
2. **Sigmoid survival**: BN enabled sigmoid-based deep networks to train, which had been thought to require ReLU
3. **Localization**: BN applied before the nonlinearity ($z = g(\text{BN}(Wu))$) worked better than normalizing layer inputs $u$

## 四、Limitations and Challenges

### 4.1 Fundamental Limitations

| Limitation | Cause | Severity | Mitigation |
|---|---|---|---|
| Small batch instability | Noisy $\mu$ and $\sigma^2$ estimates when $m < 8$ | Critical for large models | [[Group Normalization]] (Wu & He, 2018) |
| Train/inference discrepancy | Different statistics used at train vs eval | Moderate | [[Layer Normalization]] removes this |
| RNN incompatibility | Different time steps need separate statistics | Fundamental | [[Layer Normalization]], [[RMSNorm]] |
| Transformer incompatibility | Variable-length sequences break batch statistics | Fundamental | Pre-LN, [[RMSNorm]] |

### 4.2 The Small-Batch Problem

When batch size is small (e.g., 2 or 4, common in segmentation or video tasks with large inputs), BN's statistics become extremely noisy. The variance estimate has high variance itself, leading to training instability. This is especially problematic in training Diffusion Policy's visual encoder with EMA updates, where BN's running stats become stale.

### 4.3 BN in Distributed Training

Standard BN does not synchronize statistics across GPUs -- each GPU computes statistics on its local mini-batch. This means the effective batch size per BN layer is only the per-GPU batch size. Solutions like `SyncBatchNorm` (torch.nn.SyncBatchNorm) all-gather statistics across all GPUs, but add communication overhead.

## 五、Relationship with Subsequent Work / Impact on the Field

### 5.1 The Normalization Family Tree

BN's success triggered an explosion of normalization methods:

```
Batch Normalization (Ioffe & Szegedy, 2015)
├── Layer Normalization (Ba et al., 2016)       → Transformer standard
│   └── RMSNorm (Zhang & Sennrich, 2019)        → Llama series
│       └── AdaLN (Peebles & Xie, 2023)         → DiT, pi-zero
├── Instance Normalization (Ulyanov et al., 2016) → Style transfer
├── Group Normalization (Wu & He, 2018)          → Mask R-CNN, Diffusion UNet
└── Weight Normalization (Salimans & Kingma, 2016)
```

Each method varies the axis along which statistics are computed:
- **BN**: normalize over (N, H, W), per (C) -- batch and spatial dimensions
- **LN**: normalize over (C, H, W), per (N) -- feature dimension per sample
- **GN**: normalize over (H, W), per (N, G groups of C) -- intermediate
- **IN**: normalize over (H, W), per (N, C) -- per-channel

### 5.2 Theoretical Understanding After BN

BN was initially motivated by "reducing internal covariate shift," but later work (Santurkar et al., 2018) showed that ICS reduction may not be the primary mechanism. Instead, BN smooths the optimization landscape -- making the loss function more Lipschitz-continuous and the gradients better behaved. This deeper theoretical understanding has influenced all subsequent normalization design.

### 5.3 BN in the Age of Vision-Language-Action Models

**Where BN still dominates**: CNN-based vision encoders remain a BN stronghold. In VLA systems:
- **ResNet backbones** (used in many robot learning pipelines) use BN as the default normalization
- **ResNet-50** specifically has BN after every convolutional layer (54 BN layers)
- **Diffusion Policy** paper notes that EMA update is incompatible with BN's running stats, and switched to GroupNorm in its visual encoder

**Where BN has been replaced**: Every Transformer-based VLA component uses [[Layer Normalization]] or [[RMSNorm]]:
- **OpenVLA**: Llama 2 backbone is entirely RMSNorm
- **RT-2**: PaLM-E uses RMSNorm variants  
- **pi-zero**: DiT uses AdaLN (conditional on timestep)
- **FLOWER**: Uses AdaLN as the conditioning interface

## 六、Implications for You / Hardware Compatibility

### 6.1 Practical Decision Guide for Normalization

| Scenario | Default Choice | Reason |
|---|---|---|
| CNN encoder, batch size >= 16 | BatchNorm | Proven, efficient, well-supported |
| CNN encoder, batch size < 16 | GroupNorm | BN stats too noisy, GN is batch-independent |
| Any Transformer | RMSNorm (or LayerNorm) | Sequence-length independent, train=inference |
| Diffusion UNet | GroupNorm | Standard in DDPM/DDIM implementations |
| Large model fine-tuning (8+ GPUs) | SyncBatchNorm | Cross-GPU statistics for accuracy |

### 6.2 PyTorch Implementation Pitfalls

```python
# Correct: BN respects train/eval mode
model.train()  # Uses mini-batch stats
output = model(x)

model.eval()   # Uses running stats
output = model(x)

# Common mistake: torch.no_grad() does NOT disable BN stats update
with torch.no_grad():
    model.train()  # BN WILL update running stats here!
    output = model(x)
```

Key implementation details:
- `nn.BatchNorm2d(num_features, affine=True)`: default learns $\gamma$ and $\beta$
- `track_running_stats=True`: default maintains running mean/var
- `momentum=0.1`: default EMA coefficient for running stats
- In fine-tuning, if the pre-trained BN stats are from a different domain, the running stats will be stale

### 6.3 Why Understanding BN Matters for VLA Research

BN's architecture remains relevant for VLA practitioners:
1. **Feature extraction**: When using pre-trained ResNet backbones, the BN layers are pre-configured with ImageNet running statistics. Domain shift (e.g., robot camera images vs. ImageNet photos) means the BN stats need adaptation.
2. **Normalization thinking**: The fundamental design pattern -- "normalize then affine transform" -- is reused identically in LN, GN, and RMSNorm.
3. **Loss landscape intuition**: BN taught the community that **normalization smooths the optimization landscape**. This insight is why modern VLA training always includes gradient clipping, learning rate warmup, and careful normalization placement.

## PDF

[[Batch Normalization 原文.pdf]]
