---
tags:
  - 论文
  - 归一化
  - 高效架构
  - Transformer组件
  - Llama
created: 2026-06-30
paper_title: "Root Mean Square Layer Normalization"
paper_authors: "Biao Zhang, Rico Sennrich"
paper_year: 2019
paper_venue: "NeurIPS 2019"
paper_citations: "~2,500+"
paper_url: "https://arxiv.org/abs/1910.07467"
github: "https://github.com/bzhangGo/rmsnorm"
---

# RMSNorm

**Root Mean Square Layer Normalization**
*Biao Zhang, Rico Sennrich | University of Edinburgh, University of Zurich | NeurIPS 2019 | arXiv 1910.07467*

> RMSNorm is the LayerNorm simplification that became the default normalization for every major open-source LLM (Llama 1/2/3, Mistral, Qwen, Gemma). By removing the mean-centering step and normalizing only by the Root Mean Square statistic, RMSNorm saves ~7-15% computation while maintaining equivalent or better quality. Its adoption by OpenVLA's Llama 2 backbone makes it the single most relevant normalization method for modern VLA research.

---

## 一、Background/Core Idea

### 1.1 The Computational Cost of LayerNorm

[[Layer Normalization]] normalizes activations using both mean ($\mu$) and variance ($\sigma^2$):

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta, \quad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}$$

Computing $\mu$ and $\sigma$ requires two passes over the input: one to compute $\mu$, and one to compute the deviations $(x_i - \mu)^2$ for $\sigma$. For very deep models (e.g., 32-80 layer Transformers), this overhead accumulates.

The paper challenges the necessity of both LN operations:
1. **Re-centering** (subtracting $\mu$): hypothesized to be unnecessary for Transformer training
2. **Re-scaling** (dividing by $\sigma$): hypothesized to be the essential component

### 1.2 Theoretical Motivation: Re-centering is Dispensable

The paper argues that re-centering invariance -- the property that LN output is invariant to shifting all inputs by a constant -- contributes little to training success:

1. **Residual connections provide implicit centering**: In a Transformer block, $x^{(l+1)} = x^{(l)} + F(x^{(l)})$, the residual connection naturally centers $\mu$ near zero over time
2. **Weight decay controls bias**: L2 regularization on weights pushes activations toward zero mean
3. **The $\beta$ bias parameter learns little gradient**: Empirically, the gradient $\partial \mathcal{L}/\partial \beta$ is often near zero, meaning the re-centering step adds negligible representational capacity

### 1.3 Connection to Weight Normalization

The paper notes a connection to [[Weight Normalization]] (Salimans & Kingma, 2016), which also separates the magnitude and direction of weight vectors. RMSNorm accomplishes a similar effect but through normalization of activations rather than reparameterization of weights. This makes it a drop-in replacement for LayerNorm without changing the underlying layer structure.

## 二、Method/Architecture/Technical Contribution

### 2.1 The RMSNorm Transform

RMSNorm removes the mean-centering and normalizes solely by the Root Mean Square statistic:

$$\overline{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

where:
$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

**Key difference from LayerNorm**: No $\mu$ subtraction, no $\beta$ bias parameter. The only learnable parameter is the gain $\gamma$ (one per feature dimension).

The complete RMSNorm expression:
$$y = f\left(\frac{Wx}{\text{RMS}(a)} \odot g + b\right)$$

where $a = Wx$ are the summed inputs, $g$ is the gain vector (initialized to 1), and $b$ is the bias (initialized to 0), and $f$ is the nonlinearity.

### 2.2 Comparison with LayerNorm

| Property | LayerNorm | RMSNorm |
|---|---|---|
| **Mean computation** | $\mu = \frac{1}{d}\sum x_i$ -- requires one pass | Not needed |
| **Variance computation** | $\sigma^2 = \frac{1}{d}\sum (x_i - \mu)^2$ -- requires second pass | Not needed |
| **RMS computation** | Not used | $\text{RMS} = \sqrt{\frac{1}{d}\sum x_i^2}$ -- single pass |
| **Re-centering** | $x - \mu$ | None |
| **Re-scaling factor** | $\sigma$ (standard deviation) | $\text{RMS}$ (root mean square) |
| **Learnable gain** | $\gamma$ | $\gamma$ (same) |
| **Learnable bias** | $\beta$ | None |
| **Computational cost** | ~2x RMSNorm (for typical implementations) | Lower by 7-64% depending on architecture |

**On the difference between RMS and std**: For a zero-mean signal, $\text{RMS}(x) = \text{std}(x)$. For a signal with non-zero mean, $\text{RMS}(x)^2 = \text{std}(x)^2 + \mu^2$, so RMSNorm also penalizes the mean indirectly.

### 2.3 Invariance Properties

The paper provides a theoretical analysis demonstrating RMSNorm's invariance properties and compares them with other normalization methods:

| Method | Weight matrix re-scaling | Weight matrix re-centering | Weight vector re-scaling | Dataset re-scaling | Dataset re-centering | Single case re-scaling |
|---|---|---|---|---|---|---|
| BatchNorm | Invariant | No | Invariant | Invariant | Invariant | No |
| WeightNorm | Invariant | No | Invariant | No | No | No |
| LayerNorm | Invariant | Invariant | No | Invariant | No | Invariant |
| **RMSNorm** | **Invariant** | **No** | No | Invariant | No | Invariant |
| **pRMSNorm** | **Invariant** | **No** | No | Invariant | No | Invariant |

RMSNorm is invariant to **weight matrix re-scaling** (if $W' = \delta W$, the output is unchanged) but NOT to **weight matrix re-centering** (unlike LayerNorm). This is because RMS has the linearity property $\text{RMS}(\delta a) = \delta \cdot \text{RMS}(a)$, but no centering property.

### 2.4 Gradient Analysis

The paper derives the gradient of the loss with respect to parameters.

For RMSNorm output $\overline{x} = \frac{x}{\text{RMS}(x)} \cdot \gamma$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \overline{x}_i} \cdot \frac{\gamma_i}{\text{RMS}(x)} - \frac{\gamma_i \cdot x_i}{d \cdot \text{RMS}(x)^3} \sum_{j} \frac{\partial \mathcal{L}}{\partial \overline{x}_j} \cdot \overline{x}_j$$

$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial v} \odot \frac{Wx}{\text{RMS}(a)}$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial v}$$

**Key insight about weight matrix gradient**: The gradient $\partial \mathcal{L}/\partial W$ contains a term $\mathbf{R} = \frac{1}{\text{RMS}(a)}(\mathbf{I} - \frac{(Wx)(Wx)^T}{d \cdot \text{RMS}(a)^2})$ which is **negatively correlated** with both input and weight matrix scaling. This acts as an **implicit learning rate adaptor** that dynamically controls the norm of gradients, avoiding large-norm weight matrices and improving convergence stability.

### 2.5 Partial RMSNorm (pRMSNorm)

An extension of the paper: since neurons in a layer often have i.i.d. structure, the RMS can be estimated from only a subset of neurons. pRMSNorm estimates RMS from the first $p\%$ of summed inputs:

$$\text{RMS}(a) = \sqrt{\frac{1}{k} \sum_{i=1}^{k} a_i^2}, \quad k = \lceil d \cdot p \rceil$$

This preserves the re-scaling invariance property because the linearity of RMS still holds. With $p = 6.25\%$, pRMSNorm achieves competitive performance while further reducing computation.

## 三、Experiments and Key Findings

### 3.1 Machine Translation (WMT14 En-De)

The primary experiments use neural machine translation tasks.

**RNNSearch (GRU-based) on WMT14 English-German**:

| Model | newstest2014 BLEU | newstest2017 BLEU | Time per 1K steps | Speedup vs LN |
|---|---|---|---|---|
| Baseline | 21.7 | 23.4 | 399s | - |
| LayerNorm | 22.6 | 23.6 | 665s | baseline |
| RMSNorm | 22.4 | 23.7 | **501s** | **24.7%** |
| pRMSNorm | 22.6 | 23.1 | 493s | 25.9% |

RMSNorm matches or exceeds LayerNorm's BLEU score while being ~25% faster. The speed gap is particularly dramatic for RNNs because LayerNorm in TensorFlow was implemented inefficiently for recurrent architectures.

**Transformer on WMT14 En-De**:

| Model | newstest2014 BLEU | newstest2017 BLEU | Time per 1K steps | Speedup vs LN |
|---|---|---|---|---|
| Baseline | DNC (diverged) | DNC | 210s | - |
| LayerNorm | 26.6 | 27.7 | 248s | baseline |
| RMSNorm | 26.8 | 27.7 | **231s** | **6.9%** |
| pRMSNorm | 26.5 | 27.8 | 225s | 9.3% |

For Transformers, the speedup is smaller (6.9%) because the normalization layer accounts for a smaller fraction of total computation compared to the multi-head attention and FFN layers.

### 3.2 Reading Comprehension (CNN Corpus, Attentive Reader)

| Model | Validation Error Rate | Time per 0.1K steps | Speedup vs LN |
|---|---|---|---|
| Baseline | Highest | 315s | - |
| BatchNorm-LSTM | Lower | 345s | - |
| LayerNorm | ~0.83 | 392s | baseline |
| **RMSNorm** | **~0.83** | **333s** | **15.1%** |
| pRMSNorm | ~0.84 | 330s | 15.8% |

RMSNorm achieves the same error rate as LayerNorm while being 15% faster in this LSTM-based architecture.

### 3.3 Image-Caption Retrieval (Order-Embeddings)

| Model | Caption R@1 | Caption R@5 | Caption R@10 | Image R@1 | Time per 0.1K | Speedup |
|---|---|---|---|---|---|---|
| Baseline | 45.8 | 79.7 | 88.8 | 37.6 | 2.11s | - |
| LayerNorm | 47.9 | 79.5 | 89.2 | 38.4 | 12.02s | baseline |
| **RMSNorm** | **48.7** | **79.7** | **89.5** | **39.0** | **7.12s** | **40.8%** |
| pRMSNorm | 46.8 | 79.8 | 90.3 | 39.0 | 4.34s | 63.9% |

In this Theano-based implementation, RMSNorm is dramatically faster (40.8%) because LayerNorm's Theano implementation was particularly slow for the GRU cell structure.

### 3.4 CIFAR-10 Classification (Convolutional Networks)

| Model | Test Error | Time per Epoch | Speedup vs LN |
|---|---|---|---|
| Baseline | 8.96% | 21s | - |
| BatchNorm | **8.25%** | 38s | - |
| LayerNorm | 10.49% | 39s | baseline |
| **RMSNorm** | **8.83%** | **31s** | **20.5%** |
| pRMSNorm | 10.37% | 30s | 23.1% |

RMSNorm significantly outperforms LayerNorm on CNNs (8.83% vs 10.49% test error) and is 20.5% faster. The paper notes: "Though LayerNorm outperforms Baseline by shortening model convergence, it fails to generalize to the test set." This suggests that removing mean-centering actually helps CNN generalization.

### 3.5 Mean/Standard Deviation Analysis

The paper analyzes the distribution of hidden representations in the RNNSearch model to show WHY RMSNorm works despite not normalizing the mean:

| Model | Mean (ALL) | Std (ALL) | Mean (pos 1) | Std (pos 1) |
|---|---|---|---|---|
| Baseline | -1.60 | 3.04 | -2.60 | 7.35 |
| LayerNorm | -0.51 | 1.51 | -0.43 | 1.19 |
| **RMSNorm** | **-0.73** | **1.50** | **-0.40** | **1.27** |

The baseline has highly variable mean and std across time steps (pos 1 mean = -2.60, overall = -1.60). Both LayerNorm and RMSNorm stabilize std (1.50-1.51), and surprisingly, RMSNorm also stabilizes the mean even though it doesn't explicitly normalize it. This empirically supports the paper's hypothesis that re-centering is unnecessary.

## 四、Limitations and Challenges

### 4.1 Theoretical Limitations

1. **No rigorous proof of re-centering dispensability**: The paper's argument is empirical rather than theoretical. Later work would need to formally characterize when and why re-centering helps or hurts.

2. **Small-scale experiments**: The paper tests on models up to 6-layer Transformers (base setting). The behavior on models with hundreds of billions of parameters -- where the Llama series later proved RMSNorm works -- was not validated in the original paper.

3. **Limited task diversity**: Primarily NLP (translation, reading comprehension, retrieval) with one small CNN experiment. Reinforcement learning, speech, and other modalities were not tested.

4. **No analysis of $\epsilon$ sensitivity**: The numerical stability parameter $\epsilon$ interacts differently with RMS vs std, but the paper does not analyze this.

### 4.2 Practical Tradeoffs

| Consideration | RMSNorm Advantage | RMSNorm Disadvantage |
|---|---|---|
| Computational efficiency | 7-64% faster (architecture dependent) | - |
| Implementation simplicity | No mean or variance computation | - |
| Theoretically justified | Has re-scaling invariance | Lacks re-centering invariance |
| Empirical support | Matches or exceeds LN on tested tasks | Not tested on extremely deep models (original paper) |

## 五、Relationship with Subsequent Work / Impact on the Field

### 5.1 Default Normalization for the Llama Lineage

RMSNorm's adoption by Meta's Llama series made it the de facto standard normalization for open-weight LLMs:

| Model | Normalization | Parameter count | Source |
|---|---|---|---|
| Llama 1 | RMSNorm | 7B-65B | Meta (2023) |
| Llama 2 | RMSNorm | 7B-70B | Meta (2023) |
| Llama 3 | RMSNorm | 8B-405B | Meta (2024) |
| Mistral 7B | RMSNorm | 7B | Mistral AI |
| Qwen 2 | RMSNorm | 0.5B-72B | Alibaba |
| Gemma | RMSNorm | 2B-7B | Google |
| OLMo | RMSNorm | 1B-7B | AI2 |

### 5.2 RMSNorm in VLA Architecture

Every VLA system using a modern LLM backbone inherits RMSNorm:

| VLA System | Backbone | Normalization | Notes |
|---|---|---|---|
| **OpenVLA** | Llama 2 7B | RMSNorm | Pre-LN, 2 RMSNorm per Transformer block |
| **pi-zero** | PaliGemma | RMSNorm (via Gemma) | Gemma uses RMSNorm in all layers |
| **RT-2** | PaLM-E | RMSNorm (via PaLM) | PaLM paper specifies RMSNorm usage |
| **EmbodiedGPT** | LLaMA-Adapter | RMSNorm | Adapter preserves backbone normalization |
| **RoboFlamingo** | MPT (or OpenFlamingo) | RMSNorm (via MPT) | MPT family uses RMSNorm |

In **OpenVLA** specifically, each of the 32 Llama 2 layers contains exactly:
```
x → RMSNorm → Attention → residual → RMSNorm → FFN (SwiGLU) → residual
```
Two RMSNorm operations per layer $= 64$ RMSNorm applications per forward pass.

### 5.3 The Broader Evolution: LN → RMSNorm → AdaLN

```
LayerNorm (2016)
  mu + sigma normalization, learnable gamma + beta

  └─> RMSNorm (2019)                        ──> Llama series, OpenVLA
      Only RMS normalization, learnable gamma only
      7-15% faster than LN

      └─> AdaLN (DiT, 2023)                ──> pi-zero, FLOWER
          Gamma and beta predicted from conditioning signal (t, class label)
          Used when normalization doubles as conditioning interface

          └─> Cross-Attention LN (FLOWER, 2024)
              AdaLN parameters conditioned on both t and visual features
```

## 六、Implications for You / Hardware Compatibility

### 6.1 Practical Recommendations

1. **Replace LN with RMSNorm in all new projects**: RMSNorm is a zero-cost drop-in replacement for LayerNorm in PyTorch. There is no scenario in a Transformer where LN outperforms RMSNorm consistently enough to justify the 7-15% extra computation.

2. **Don't bother with pRMSNorm**: The 1-3% extra speedup from partial RMS estimation is not worth the implementation complexity and slight quality degradation.

3. **LoRA fine-tuning with RMSNorm**: When fine-tuning with LoRA, RMSNorm's $\gamma$ parameters can be frozen without noticeable quality loss. This saves ~0.1-0.2% of trainable parameters.

4. **bf16/fp16 training**: RMSNorm is numerically more stable than LN in low precision because it avoids the subtraction $x - \mu$ which can lose precision when $\mu$ is large.

### 6.2 Implementation in PyTorch

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

**Note**: The official Llama implementation also uses `eps = 1e-5` and places RMSNorm **before** the sublayer (Pre-LN). The nonlinearity in the transformer is **not** applied after RMSNorm.

### 6.3 Computational Cost Breakdown

For a Llama 2 7B model (hidden_size=4096, 32 layers):

| Operation | FLOPs per token per layer | % of total |
|---|---|---|
| RMSNorm (x2) | ~16K | <0.01% |
| Attention (QKV + output) | ~16M | ~20% |
| FFN (SwiGLU with gate) | ~64M | ~80% |

RMSNorm accounts for <0.01% of total compute. The 7-15% speedup claimed in the paper was relative to LayerNorm (not to the full Transformer), and was measured in RNN-heavy implementations where normalization was a larger fraction of compute.

**Conclusion**: In modern Transformers, the primary benefit of RMSNorm over LN is not speed but **simplicity and numerical stability**. Quality is equivalent.

## PDF

[[RMSNorm 原文.pdf]]
