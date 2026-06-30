---
tags:
  - 论文
  - 归一化
  - 训练稳定性
  - 架构组件
  - Transformer
  - RNN
created: 2026-06-30
paper_title: "Layer Normalization"
paper_authors: "Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton"
paper_year: 2016
paper_venue: "arXiv"
paper_citations: "~20,000+"
paper_url: "https://arxiv.org/abs/1607.06450"
---

# Layer Normalization

**Layer Normalization**
*Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton | University of Toronto | arXiv 1607.06450*

> Layer Normalization is the default normalization method for every Transformer-based model in existence. Unlike [[Batch Normalization]], LN normalizes along the feature dimension per individual sample, making it independent of batch size and identical at training and inference time. This property is essential for RNNs handling variable-length sequences and for Transformers processing billions of tokens. Its modern variant [[RMSNorm]] (used in the Llama series) removes the re-centering step for ~10% speedup.

---

## 一、Background/Core Idea

### 1.1 The Limitations of Batch Normalization for Sequence Models

[[Batch Normalization]] revolutionized CNN training but had three fundamental limitations that become critical for RNNs and Transformers:

**1. Batch-size dependency**: BN's statistics ($\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$) are computed over the batch dimension. When batch size is small (common in sequence models with long sequences), these estimates become noisy. In online learning or extremely large distributed models, the mini-batch may contain only 1-2 examples.

**2. Training-inference discrepancy**: BN uses running statistics at inference time, creating two different computational graphs. This makes it harder to deploy and debug.

**3. RNN incompatibility**: In an RNN, the same cell is unrolled across time steps of varying length. BN would need to maintain separate running statistics for each time step, and a test sequence longer than any training sequence would encounter unseen time steps with no statistics. As the paper states: "It is not obvious how to apply [BN] to recurrent neural networks."

Specifically, in an RNN the hidden state at time $t$ is:
$$h_t = f(W_h h_{t-1} + W_x x_t)$$

If we try to apply BN to $W_h h_{t-1}$ at each time step, we need to estimate statistics for each $t$ separately. A test sequence that is longer than all training sequences would have no statistics for its later time steps.

### 1.2 The Core Insight: Normalize Per Sample, Per Layer

LN's key idea is elegantly simple: instead of normalizing over the batch, normalize over the feature dimension for EACH training sample independently. The normalization statistics depend only on the current input at the current time step, not on other examples or other time steps.

This means:
- **No batch dependency**: Works with batch size = 1
- **Training = inference**: Identical computation at both phases
- **Time-step independence**: Each RNN time step normalizes independently
- **Variable-length sequences**: No issue -- each position normalizes its own features

## 二、Method/Architecture/Technical Contribution

### 2.1 The Layer Normalization Transform

For a layer with $H$ hidden units (feature dimension), having summed inputs $a^l$:

$$\mu^l = \frac{1}{H} \sum_{i=1}^{H} a_i^l$$

$$\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_i^l - \mu^l)^2}$$

$$h^l = f\left(\frac{g}{\sigma^l} \odot (a^l - \mu^l) + b\right)$$

Where:
- $g$ (gain) and $b$ (bias) are learnable parameters -- analogous to $\gamma$ and $\beta$ in BN
- $\odot$ denotes element-wise multiplication
- $f(\cdot)$ is the nonlinearity (applied AFTER normalization)

**Key difference from BN**: The normalization is applied to the entire hidden layer vector (all $H$ units) for a single training case, rather than to one feature across all cases in the mini-batch.

### 2.2 Comparison with Batch Normalization

| Property | Batch Normalization | Layer Normalization |
|---|---|---|
| **Normalization axis** | Batch (N) | Feature (H) |
| **Statistics computed over** | N x H x W (across examples) | H (within one example) |
| **Dependence on batch size** | Yes -- small batches hurt | No -- works with batch=1 |
| **Train vs. inference** | Different (running stats) | Identical |
| **RNN applicability** | Poor (time-step statistics) | Natural (per time-step) |
| **CNN effectiveness** | Excellent | Poor (see Section 3.3) |
| **Parameters per layer** | $2 \times C$ (per channel $\gamma, \beta$) | $2 \times H$ (per neuron $\gamma, \beta$) |
| **Extra storage** | Running mean/var (2 buffers) | None |
| **Small batch stability** | Unstable ($m < 8$) | Stable |
| **Variable-length sequences** | Breaks (unseen time steps) | Works naturally |

### 2.3 LN in Recurrent Neural Networks

The paper's primary contribution is making normalization work for RNNs. The LN-RNN formulation:

**Standard RNN cell**:
$$a_t = W_{hh} h_{t-1} + W_{xh} x_t$$

**Layer-normalized RNN cell**:
$$\mu_t = \frac{1}{H} \sum_{i=1}^{H} a_t^{(i)}, \quad \sigma_t = \sqrt{\frac{1}{H} \sum_{i=1}^{H} (a_t^{(i)} - \mu_t)^2}$$
$$h_t = f\left(\frac{g}{\sigma_t} \odot (a_t - \mu_t) + b\right)$$

Each time step $\mu_t$ and $\sigma_t$ are computed independently from the current $a_t$ only -- no statistics are shared across time steps. This prevents the gradient explosion/vanishing that normally plagues RNNs, because $\sigma_t$ adapts to the scale of $a_t$ at each step.

**LN for LSTM**: The paper provides specific LN-LSTM formulations. For Vanilla LSTM:

Standard:
$$\begin{pmatrix} f_t \\ i_t \\ o_t \\ g_t \end{pmatrix} = W_h h_{t-1} + W_x x_t + b$$
$$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t)$$
$$h_t = \sigma(o_t) \odot \tanh(c_t)$$

With LN (applied separately to recurrent weights and input weights):
$$\begin{pmatrix} f_t \\ i_t \\ o_t \\ g_t \end{pmatrix} = \text{LN}(W_h h_{t-1}; \alpha_1, \beta_1) + \text{LN}(W_x x_t; \alpha_2, \beta_2) + b$$
$$h_t = \sigma(o_t) \odot \tanh(\text{LN}(c_t; \alpha_3, \beta_3))$$

The LN is applied to the **summed inputs** before the nonlinearity, similar to BN's placement. LN applied to GRU follows a similar pattern: each linear projection is independently normalized before summation.

### 2.4 Invariance Properties (Theoretical Contribution)

The paper provides a rigorous analysis of invariance properties:

| Method | Weight re-scaling | Weight re-centering | Weight vector re-scaling | Dataset re-scaling | Dataset re-centering | Single case re-scaling |
|---|---|---|---|---|---|---|
| BatchNorm | Invariant | No | Invariant | Invariant | Invariant | No |
| WeightNorm | Invariant | No | Invariant | No | No | No |
| **LayerNorm** | **Invariant** | **Invariant** | No | Invariant | No | **Invariant** |

LN is **invariant to scaling of the entire weight matrix** AND a **shift to all incoming weights**. If $W' = \delta W + \mathbf{1} \gamma^\top$, the LN model output remains unchanged. This is because the $\mu$ and $\sigma$ statistics in LN involve all neurons in the layer, so shifting all weights by a constant vector is absorbed by the normalization.

Most importantly, LN is **invariant to re-scaling of individual training cases** -- the normalization scalar depends only on the current input. This is the property that makes LN robust to varying sequence lengths and input magnitudes.

## 三、Experiments and Key Findings

### 3.1 Image-Sentence Ranking (Order-Embeddings)

The first experiment tests LN on a GRU-based order-embedding model for cross-modal retrieval (MS COCO dataset).

| Model | Caption R@1 | Caption R@5 | Caption R@10 | Image R@1 | Image R@5 | Image R@10 |
|---|---|---|---|---|---|---|
| OE (baseline) | 46.6 | 79.3 | 89.1 | 37.8 | 73.6 | 85.7 |
| OE + LN | **48.5** | **80.6** | **89.8** | **38.9** | **74.3** | **86.3** |

- LN improves all recall metrics
- LN converges in **60% of the training time** compared to baseline
- This was state-of-the-art for RNN embedding models at the time

### 3.2 Attentive Reader (Question Answering)

The reading comprehension task (CNN corpus) directly compares LN with recurrent BN:

| Model | Validation Error |
|---|---|
| LSTM (baseline) | Highest |
| BN-LSTM (Cooijmans et al.) | Lower |
| BN-everywhere | Similar |
| **LN-LSTM** | **Lowest** |

LN-LSTM outperforms both the baseline and the specially-designed recurrent BN. The validation curve shows LN converging faster and to a lower error rate than all alternatives.

### 3.3 Permutation-Invariant MNIST (Feedforward Networks)

This experiment tests LN on feedforward MLPs and contrasts with BN under different batch sizes:

| Condition (batch size) | Baseline | BN | LN |
|---|---|---|---|
| **Large batch (128)** | High NLL | Lower NLL | **Lowest NLL** |
| **Small batch (4)** | Fails to converge | BN unstable (high variance) | **Most stable, fastest convergence** |

The small-batch experiment is critical: at batch size 4, BN's variance estimates are extremely noisy, causing training instability. LN is **completely unaffected** by batch size.

### 3.4 Convolutional Networks (Honest Assessment)

The paper test CNNs and reports honestly: **LN underperforms BN for CNNs**. The test error on CIFAR-10:

| Method | Test Error |
|---|---|
| Baseline (no norm) | 8.96% |
| BatchNorm | **8.25%** |
| LayerNorm | 10.49% |

LN actually **hurts** CNN performance compared to baseline. The paper explains: "With fully connected layers, all the hidden units in a layer tend to make similar contributions to the final prediction... However, the assumption of similar contributions is no longer true for convolutional neural networks." Specifically, boundary neurons (near image edges) have very different statistics from interior neurons, and normalizing them together harms performance.

### 3.5 Additional Experiments

| Task | Model | LN Impact |
|---|---|---|
| Skip-thought vectors | Sentence encoder RNN | Improves all downstream tasks (MR: +2.2%, CR: +0.8%, SUBJ: +0.8%) |
| Handwriting generation | RNN with 500-length sequences | LN reduces NLL from ~0 to -700 over baseline |
| DRAW (MNIST generation) | Recurrent attention model | LN converges 2x faster, better final NLL (82.09 vs 82.36 nats) |

## 四、Limitations and Challenges

### 4.1 Poor Performance on CNNs

As shown in Section 3.4, LN underperforms BN on CNNs. The fundamental issue is that CNNs features have different spatial statistics -- edge detectors near image boundaries behave differently from those at the center. Normalizing them jointly destroys this useful diversity.

### 4.2 Mathematical Understanding Lags BN

BN's effect of smoothing the loss landscape has been rigorously analyzed (Santurkar et al., 2018). For LN, the theoretical understanding is less complete. The paper provides a Fisher information matrix analysis in the supplementary material, but the connection to training dynamics is less direct.

### 4.3 Computational Overhead

LN adds two operations per layer:
1. Computing $\mu$ and $\sigma$ over the feature dimension
2. Applying the affine transform ($g \odot \hat{x} + b$)

While moderate, this overhead became a motivation for [[RMSNorm]], which removes the $\mu$ computation.

## 五、Relationship with Subsequent Work / Impact on the Field

### 5.1 Pre-LN vs. Post-LN: The Transformer Architecture Debate

The original Transformer ([[Attention Is All You Need]], 2017) used **Post-LN**: LN was applied AFTER the residual addition:
$$\text{Output} = \text{LN}(x + \text{Sublayer}(x))$$

Later work discovered that **Pre-LN** (applying LN BEFORE the sublayer) provides more stable training:
$$\text{Output} = x + \text{Sublayer}(\text{LN}(x))$$

| Property | Post-LN | Pre-LN |
|---|---|---|
| Gradient signal | Residual path has LN, blocking clean gradient flow | Residual path is clean (no LN), gradients flow freely |
| Warmup required | Yes (careful warmup needed) | No standard warmup suffices |
| Deep model stability (< 100 layers) | Tends to diverge | Stable |
| Used in | Original Transformer (2017) | GPT-2 onward, all modern LLMs |
| Theoretical analysis | Harder (LN interacts with residual) | Easier (LN is before each block) |

Every modern VLA architecture ([[Llama 2]], Gemma, Qwen, PaliGemma) uses Pre-LN exclusively.

### 5.2 Evolution: LN → RMSNorm → AdaLN

The normalization for Transformers evolved in a clear chain:

1. **LayerNorm (2016)**: Original formulation, mean + variance normalization
2. **[[RMSNorm]] (2019)**: Removes mean-centering, only uses RMS scaling. Saves ~7-15% computation. Used by the Llama series.
3. **AdaLN (2023)**: Adaptive LayerNorm where the scale $\gamma$ and shift $\beta$ are predicted from conditioning signals (timestep $t$, class labels). Used by [[DiT]] (Diffusion Transformers).

In VLA:
- **pi-zero (2024)**: Uses AdaLN for timestep conditioning in the DiT action expert
- **FLOWER (2024)**: Uses AdaLN as the conditioning interface structure (a key innovation)
- **OpenVLA**: Llama 2 backbone uses RMSNorm

## 六、Implications for You / Hardware Compatibility

### 6.1 Practical Guidelines

1. **Default normalization for Transformer = RMSNorm**: In any new Transformer-based VLA model, use RMSNorm (without mean centering) as a drop-in replacement for LN. It saves computation with no quality loss.

2. **Pre-LN by default**: Always use Pre-LN (normalize before sublayer) for Transformer blocks. This is the standard in all post-GPT-2 architectures.

3. **Watch the $\epsilon$ parameter**: The $\epsilon$ in LN prevents division by zero. For fp16/bf16 training, increase $\epsilon$ to $10^{-5}$ (from the default $10^{-6}$) for numerical stability.

4. **LN is NOT always the answer**: For CNN-based vision encoders, use BN (batch >= 16) or GN (batch < 16). LN is for sequence models and Transformers.

### 6.2 PyTorch Implementation

```python
# Standard LayerNorm
ln = nn.LayerNorm(hidden_size)  # elementwise_affine=True by default

# Without learnable parameters
ln_fixed = nn.LayerNorm(hidden_size, elementwise_affine=False)

# RMSNorm (manual implementation)
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 6.3 Understanding LN for VLA Innovation

LN might seem like "just a normalization trick," but it has become an innovation surface:
- **FLOWER's core contribution** is using AdaLN's $\gamma, \beta$ parameters as the conditioning interface
- **DiT's innovation** is conditioning the LN parameters on the diffusion timestep
- **Understanding $\gamma$ and $\beta$** as learnable parameters that can carry conditional information is essential for reading modern VLA papers

## PDF

[[Layer Normalization 原文.pdf]]
