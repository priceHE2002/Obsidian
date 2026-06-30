---
tags:
  - 论文
  - 优化器
  - 训练技巧
  - 基础组件
  - 自适应学习率
created: 2026-06-30
paper_title: "Adam: A Method for Stochastic Optimization"
paper_authors: "Diederik P. Kingma, Jimmy Ba"
paper_year: 2014
paper_venue: "ICLR 2015"
paper_citations: "~200,000+"
paper_url: "https://arxiv.org/abs/1412.6980"
---

# Adam

**Adam: A Method for Stochastic Optimization**
*Diederik P. Kingma, Jimmy Ba | University of Amsterdam, OpenAI / University of Toronto | ICLR 2015 | arXiv 1412.6980*

> Adam is the default optimizer for virtually all deep learning. Combining momentum (accelerated convergence) with per-parameter adaptive learning rates (via gradient second moments), it is robust to hyperparameter choices, handles sparse gradients, and works out of the box for CNNs, RNNs, and Transformers. Its variant AdamW (decoupled weight decay) is the universal optimizer for every VLA system -- OpenVLA, RT-2, Diffusion Policy, pi-zero, and FLOWER all use it.

---

## 一、Background/Core Idea

### 1.1 The Pre-Adam Landscape

Before Adam, gradient-based optimization had two dominant paradigms that addressed different challenges:

| Family | Methods | Strength | Weakness |
|---|---|---|---|
| **Momentum-based** | SGD + Momentum, Nesterov SGD | Accelerated convergence, smooth trajectory | Single global learning rate, poor sparse gradients |
| **Adaptive learning rate** | AdaGrad, RMSProp, AdaDelta | Per-parameter learning rates, handle sparse features | AdaGrad: learning rate monotonically decays to zero; RMSProp: no momentum |

**SGD + Momentum** (Sutskever et al., 2013) maintains a velocity vector that accumulates past gradients:
$$v_t = \mu v_{t-1} + (1 - \mu) g_t$$
$$\theta_t = \theta_{t-1} - \alpha v_t$$

**AdaGrad** (Duchi et al., 2011) accumulates all past squared gradients:
$$G_t = \sum_{i=1}^t g_i^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot g_t$$

AdaGrad works well for sparse features (infrequent large gradients get larger updates) but the learning rate $G_t$ grows monotonically, eventually becoming so small that training effectively stops.

**RMSProp** (Tieleman & Hinton, 2012) fixes the monotonic decay by using an exponential moving average:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{v_t + \epsilon}} \odot g_t$$

But RMSProp lacks momentum and has no bias correction, which causes instability with $\beta_2$ close to 1 (required for sparse gradients).

### 1.2 Adam's Core Insight

Adam (Adaptive Moment Estimation) fuses these two families into a single algorithm that:
1. **Maintains a first moment estimate** $m_t$ (running average of gradients -- like momentum)
2. **Maintains a second moment estimate** $v_t$ (running average of squared gradients -- like RMSProp)
3. **Applies bias correction** to both moments (critical in early training steps)
4. **Offers the effective step size as a trust region**: $|\Delta_t| \lessapprox \alpha$, meaning the learning rate $\alpha$ directly bounds the per-step parameter change

Adam's design principle: "The effective magnitude of the steps taken in parameter space at each timestep are approximately bounded by the stepsize setting $\alpha$. This can be understood as establishing a trust region around the current parameter value."

## 二、Method/Architecture/Technical Contribution

### 2.1 Full Algorithm Derivation

**Algorithm 1: Adam**

**Input**: Learning rate $\alpha = 0.001$, decay rates $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**Initialize**: $m_0 = 0$ (first moment), $v_0 = 0$ (second moment), $t = 0$

For each training step $t$:

1. **Compute gradient**: $g_t = \nabla_\theta f_t(\theta_{t-1})$

2. **Update biased first moment** (momentum):
   $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

3. **Update biased second moment** (adaptive learning rate):
   $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
   (Element-wise square: $g_t^2 = g_t \odot g_t$)

4. **Bias correction**:
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

5. **Parameter update**:
   $$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 2.2 Why Bias Correction Matters

Both $m_t$ and $v_t$ are initialized to zero vectors. Without bias correction:

- In early steps, $m_t \approx (1 - \beta_1) g_t$ is heavily biased toward zero
- $v_t$ with $\beta_2 = 0.999$ is even more biased: after 10 steps, $v_{10} \approx 0.01 \sum_{i} 0.999^{10-i} g_i^2$

The bias correction terms $1 - \beta_1^t$ and $1 - \beta_2^t$ compensate for this initialization bias. For $\beta_2 = 0.999$:
- After 1 step: $1 - 0.999^1 = 0.001$, so $\hat{v}_1 = 1000 \times v_1$ (strong correction)
- After 1000 steps: $1 - 0.999^{1000} \approx 0.632$, moderate correction
- After 7000 steps: $1 - 0.999^{7000} \approx 0.999$, negligible correction

The paper's experiment (Section 6.4) shows that removing bias correction (equivalent to a version of RMSProp with momentum) causes divergence when $\beta_2$ is close to 1, especially in early training.

### 2.3 The Signal-to-Noise Ratio Interpretation

Adam's effective update can be interpreted as:

$$\Delta_t = -\alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

The ratio $\hat{m}_t / \sqrt{\hat{v}_t}$ is a **signal-to-noise ratio (SNR)** approximation:
- When the gradient direction is consistent (high SNR), $\hat{m}_t \approx \pm \sqrt{\hat{v}_t}$, so $|\Delta_t| \approx \alpha$
- When the gradient direction is noisy (low SNR, e.g., near optimum), $\hat{m}_t \ll \sqrt{\hat{v}_t}$, so $|\Delta_t| \ll \alpha$

This means Adam **automatically anneals** the step size as it approaches an optimum -- no learning rate schedule required (though one still helps).

Additionally, the update is **invariant to gradient rescaling**: if all gradients are scaled by $c$, then $\hat{m}_t$ scales by $c$ and $\sqrt{\hat{v}_t}$ also scales by $c$, so the ratio is unchanged.

### 2.4 AdaMax Extension

The paper also proposes AdaMax, a variant using the $\ell_\infty$ norm instead of $\ell_2$:

$$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$$
$$\theta_t = \theta_{t-1} - \frac{\alpha}{1 - \beta_1^t} \cdot \frac{m_t}{u_t}$$

AdaMax simplifies to tracking the maximum (exponentially weighted) absolute gradient. The update bound becomes simpler: $|\Delta_t| \leq \alpha / (1 - \beta_1^t)$. In practice, AdaMax is rarely used compared to standard Adam.

### 2.5 Theoretical Convergence Guarantee

The paper provides a regret bound for convex optimization:

$$R(T) \leq \frac{D^2}{2\alpha(1-\beta_1)} \sum_{i=1}^d \sqrt{T \hat{v}_{T,i}} + \frac{\alpha(1+\beta_1)G_\infty}{(1-\beta_1)\sqrt{1-\beta_2}(1-\gamma)^2} \sum_{i=1}^d \|g_{1:T,i}\|_2 + \cdots$$

where $\gamma = \frac{\beta_1^2}{\sqrt{\beta_2}}$. For sparse data, Adam achieves $O(\log d \sqrt{T})$ regret -- an improvement over $O(\sqrt{d T})$ for non-adaptive methods. This is comparable to the best known result for Adagrad.

## 三、Experiments and Key Findings

### 3.1 Logistic Regression

| Dataset | Task | Adam vs Best Competitor |
|---|---|---|
| MNIST (784-dim pixels) | 10-class logistic regression | Adam matches AdaGrad, significantly outperforms SGD+Nesterov |
| IMDB (10K BoW features) | Sentiment classification | Adam + dropout: fastest convergence, lowest cost |

Adam converges as fast as AdaGrad on sparse features while also benefiting from the momentum-like behavior.

### 3.2 Multi-Layer Neural Networks (MNIST)

2 hidden layers with 1000 hidden units, ReLU activation:

| Condition | Adam vs Others |
|---|---|
| **Deterministic** (no dropout) | Adam converges faster than SFO (quasi-Newton), AdaGrad, SGD+Nesterov |
| **With dropout** | Adam significantly outperforms all competitors |
| **Wall-clock time** | SFO is 5-10x slower per iteration than Adam (which only requires first-order gradients) |

Adam maintained its advantage even with dropout's stochastic regularization.

### 3.3 Convolutional Neural Networks (CIFAR-10)

Architecture: c64-c64-c128-1000 (conv layers + FC)

| Phase | Behavior |
|---|---|
| **First 3 epochs** | Adam and AdaGrad make rapid initial progress |
| **After 45 epochs** | Adam and SGD converge well; AdaGrad plateaus early |

The paper notes that Adam's second moment estimate $\hat{v}_t$ becomes very small after a few epochs (dominated by $\epsilon$), making the approximation less useful for CNNs. However, the first moment (momentum) continues to help.

### 3.4 Bias Correction Ablation (Variational Autoencoder)

This experiment directly tests the importance of bias correction:

| $\beta_1$ | $\beta_2$ | With bias correction | Without bias correction |
|---|---|---|---|
| 0 | 0.99 | Stable | Stable (small bias) |
| 0 | 0.999 | Stable | Slightly unstable |
| 0 | 0.9999 | Stable | **Diverges** |
| 0.9 | 0.99 | Stable | Instability at low $\alpha$ |
| 0.9 | 0.999 | Stable | **Highly unstable** |
| 0.9 | 0.9999 | Stable | **Diverges** |

With $\beta_2 = 0.9999$ and $\alpha$ around 0.001, removing bias correction leads to divergence. This validates the paper's design: bias correction is essential when $\beta_2$ is close to 1.

### 3.5 Language Modeling and Machine Translation

| Task | Dataset | Adam Result |
|---|---|---|
| Language modeling | Penn Treebank | Adam achieves lowest perplexity |
| Machine translation | WMT'14 En-Fr | Adam surpasses Adadelta and SGD |

## 四、Limitations and Challenges

### 4.1 Generalization Gap vs. SGD

A known issue: Adam sometimes generalizes worse than SGD with momentum on some vision tasks. The hypothesis is that Adam's adaptive learning rates may lead to sharper minima (less flat minima = worse generalization).

| Task | Adam | SGD Momentum | Winner |
|---|---|---|---|
| ImageNet classification (ResNet) | 23.5% top-1 err | **22.8%** | SGD |
| CIFAR-10 (Wide ResNet) | 4.2% | **3.9%** | SGD |
| PTB language modeling | **76.4 PPL** | 79.1 PPL | Adam |
| WMT translation | **24.3 BLEU** | 23.8 BLEU | Adam |
| Transformer training | **AdamW** is standard | SGD fails | AdamW |

In practice, Adam/AdamW dominates NLP, generative models, and multimodal systems, while SGD is still competitive in image classification.

### 4.2 Memory Overhead

Adam stores 2 additional values per parameter ($m_t$ and $v_t$):

| Model size | Parameters (bf16) | Adam optimizer states | Total (model + opt) |
|---|---|---|---|
| 7B | 14 GB | 28 GB ($m_t$ + $v_t$ in fp32) | 42 GB |
| 13B | 26 GB | 52 GB | 78 GB |
| 70B | 140 GB | 280 GB | 420 GB |

This memory overhead is a critical constraint. Mitigations:
- **bitsandbytes 8-bit Adam**: Reduces optimizer states to ~7 GB for 7B model
- **Adafactor**: Removes $m_t$ storage (factors it across dimensions)
- **Lion**: Only tracks momentum, no second moment
- **Sophia**: Uses Hessian information, claims 2x fewer steps

### 4.3 Hyperparameter Sensitivity (Practical)

| Hyperparameter | Default | Sensitivity | Common Adjustments |
|---|---|---|---|
| $\alpha$ (learning rate) | 0.001 | High | 2e-5 for LoRA fine-tuning, 1e-4 for training from scratch |
| $\beta_1$ (momentum decay) | 0.9 | Low-Moderate | 0.95 sometimes helps smoother training |
| $\beta_2$ (square decay) | 0.999 | Low-Moderate | 0.995 for noisy gradients, 0.99 for fast adaptation |
| $\epsilon$ (numerical stability) | 1e-8 | Low | 1e-6 for fp16/bf16 training to prevent division by near-zero |
| Weight decay (AdamW) | 0.01 | Moderate | 0.1 for large models, 0.001 for fine-tuning |

## 五、Relationship with Subsequent Work / Impact on the Field

### 5.1 AdamW: The Critical Fix for Transformers

AdamW (Loshchilov & Hutter, 2017, "Decoupled Weight Decay Regularization") identified a subtle bug in Adam's weight decay implementation.

**The problem**: In standard Adam (and SGD), L2 regularization and weight decay are equivalent:
$$\theta_t = \theta_{t-1} - \alpha \cdot (\nabla L + \lambda \theta_{t-1}) = \theta_{t-1}(1 - \alpha \lambda) - \alpha \nabla L$$

But in Adam, the update becomes:
$$\theta_t = \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t + \lambda \theta_{t-1}}{\sqrt{\hat{v}_t} + \epsilon}$$

The weight decay $\lambda \theta_{t-1}$ is **divided by** $\sqrt{\hat{v}_t}$, meaning parameters with large historical gradients (small $\hat{v}_t$) receive weaker regularization. This is incorrect -- weight decay should be uniform.

**AdamW's fix**:
$$\theta_t = \theta_{t-1} - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Weight decay is now **decoupled** from the adaptive learning rate. This seemingly minor change is crucial for Transformers, where different layers (embedding, attention, FFN) have very different gradient scales.

### 5.2 AdamW as the Universal VLA Optimizer

Every major VLA system uses AdamW:

| VLA System | Optimizer | Learning Rate | Weight Decay | Schedule |
|---|---|---|---|---|
| **OpenVLA** | AdamW | 2e-5 | 0.01 | Cosine, 500 warmup steps |
| **Diffusion Policy** | AdamW | 1e-4 | 0.01 or 0.001 | Cosine + warmup |
| **RT-2** | AdamW | 3e-5 | 0.01 | Co-Fine-Tuning schedule |
| **FLOWER** | AdamW | 3e-4 (pretrain), 3e-5 (finetune) | 0.01 | Cosine decay |
| **pi-zero** | AdamW | Various | Various | Through JAX optax |
| **Octo** | AdamW | 3e-4 | 0.01 | Cosine, 3000 warmup steps |

Why AdamW dominates:
1. **Transformer backbone requirement**: Llama 2 is trained with AdamW, so VLA fine-tuning must inherit the same optimizer
2. **Multi-modal gradient diversity**: Image, text, and action gradients have wildly different scales -- adaptive learning rates are essential
3. **SwiGLU activation sensitivity**: [[Llama 2]] uses SwiGLU, which has broader gradient distributions than ReLU, benefiting from adaptive per-parameter control
4. **LoRA compatibility**: LoRA + AdamW is the standard fine-tuning configuration

### 5.3 Why SwiGLU Needs Different Learning Rates Than ReLU

SwiGLU (used in [[Llama 2]], PaLM, Gemini) computes:
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

The Swish activation $\text{Swish}(x) = x \cdot \sigma(x)$ has:
- Non-zero gradients for negative inputs (unlike ReLU which is exactly 0)
- Larger gradient variance than ReLU due to the multiplicative gating structure

This means:
- With SGD's single global LR, SwiGLU networks are harder to tune
- Adam's per-parameter LR automatically adapts: the gating weight $W_2$ may need a different effective LR than the projection weight $W_1$
- LR = 2e-5 for OpenVLA fine-tuning works because Adam's adaptive mechanism handles these per-weight differences

## 六、Implications for You / Hardware Compatibility

### 6.1 Practical VLA Fine-Tuning Configuration

**Recommended default for VLA fine-tuning**:
```python
from torch.optim import AdamW

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,           # Safe starting point for most VLA fine-tuning
    betas=(0.9, 0.999), # Standard defaults
    eps=1e-8,           # Increase to 1e-6 for fp16
    weight_decay=0.01   # Llama 2 default
)

# Schedule: warmup + cosine
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
)
```

### 6.2 Memory Optimization Guide

For a 7B VLA model on consumer GPUs:

| Setting | Model Params | Optimizer State | Gradient | Total VRAM | Feasible GPU |
|---|---|---|---|---|---|
| Full fine-tune (fp32) | 28 GB | 56 GB | 28 GB | 112 GB | A100 80GB |
| Full fine-tune (bf16) | 14 GB | 28 GB | 14 GB | 56 GB | A100, H100 |
| **LoRA + bf16 AdamW** | 14 GB | 28 GB (base) + 0.3 GB (LoRA) | 14 GB | ~48 GB | A100 |
| **LoRA + 8-bit AdamW** | 14 GB | **~7 GB** | 14 GB | ~36 GB | **RTX 4090 24GB** |
| **QLoRA + 4-bit Adam** | ~3.5 GB | ~1 GB | ~3.5 GB | **~8 GB** | **RTX 3090 24GB** |

### 6.3 Hyperparameter Tuning Protocol

**When loss oscillates wildly**:
1. Halve LR (2e-5 -> 1e-5)
2. Increase $\epsilon$ to 1e-6
3. Increase warmup proportion

**When loss converges too slowly**:
1. Double LR (2e-5 -> 5e-5)
2. Check that $\beta_2$ isn't too close to 1 (try 0.99)
3. Increase batch size if possible

**When overfitting**:
1. Increase weight_decay (0.01 -> 0.1)
2. Add gradient clipping (max_norm = 1.0)
3. Reduce LR / increase LoRA dropout

### 6.4 Understanding $\hat{v}_t$ for Debugging

If you see sudden loss spikes during training, inspect the effective step sizes:

```python
# Pseudo-code: monitor the effective step size ratio
for name, param in model.named_parameters():
    if param.grad is not None:
        ratio = m_t[name] / (sqrt(v_t[name]) + eps)
        # If ratio > 10 for any parameter, learning is unstable
```

Large spikes in a specific weight's effective step size often indicate that $\hat{v}_t$ underestimated the gradient magnitude (because the gradient just became much larger than its historical average). This is common when training enters a new region of parameter space (e.g., after a learning rate warmup ends too abruptly).

## PDF

[[Adam 原文.pdf]]
