---
tags:
  - 论文
  - 训练基础设施
  - 优化器
  - 正交化
  - Newton-Schulz
  - 矩阵分解
created: 2026-07-02
paper_title: "Muon: An optimizer for hidden layers in neural networks"
paper_authors: "Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein"
paper_year: 2024
paper_venue: "Blog Post + arXiv:2502.16982 (Scalability by Moonshot AI, 2025)"
paper_citations: "~200+ (blog-based, 2024-2025 快速采用)"
paper_url: "https://kellerjordan.github.io/posts/muon/"
github: "https://github.com/KellerJordan/Muon"
---

# Muon

**Muon: An optimizer for hidden layers in neural networks**
*Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein | 2024 | Blog Post*

> Muon (MomentUm Orthogonalized by Newton-Schulz) 将 SGD with Nesterov 动量产生的 2D 更新矩阵用 Newton-Schulz 迭代近似正交化，使更新方向在白化的同时保持多样性。由个人博客而非学术论文发布，凭 NanoGPT 和 CIFAR-10 速度训练纪录获得关注，作者 Keller Jordan 因此直接入职 OpenAI。Moonshot AI 的 MuonClip 验证了它在 16B MoE 级别训练的可行性。

---

## 一、Background / Core Idea

### 1.1 问题：现有优化器将 2D 权重视为独立标量

AdamW、Lion 等所有主流优化器都**逐元素（element-wise）**处理梯度更新——每个参数 $w_{ij}$ 收到独立的步长调整。但神经网络的 hidden layer 权重本质上是**线性映射** $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，将一个向量空间映射到另一个。把权重矩阵视为一堆独立标量的集合忽略了这个全局几何结构。

Keller Jordan 手动检查发现：SGD-momentum 和 Adam 对 2D 参数产生的更新矩阵**条件数极高（几乎是低秩的）**——所有神经元的更新被少数几个方向主导。

### 1.2 核心洞察：正交化使"稀有方向"的更新增大

Muon 的做法：

1. 正常计算 SGD + Nesterov 动量产生更新矩阵 $G \in \mathbb{R}^{m \times n}$
2. 将 $G$ 近似正交化：找到最近的半正交矩阵 $O = \arg\min_O \{ \|O - G\|_F \}$（即保持 $O^\top O = I$ 或 $OO^\top = I$）
3. 用 $O$ 替代 $G$ 进行参数更新

通过 SVD 视角（$G = USV^\top$），正交化就是**丢弃奇异值 $S$，只保留方向信息 $UV^\top$**。这等价于把所有方向（大奇异值和微小奇异值对应的方向）的统一化更新幅度，让那些"罕见但重要"的方向也得到同等幅度的更新。

### 1.3 与 Shampoo 的理论联系

Bernstein & Newhouse (2024) 发现：去掉 Shampoo 的预条件累积后，其更新退化为 $W_{t+1} = W_t - \eta UV^\top$（即正交梯度）。如果在正交化前加上动量——就得到 Muon。可以将 Muon (momentum=0) 视为一种"无需累积的瞬时 Shampoo"。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 Newton-Schulz 迭代替代 SVD

正交化的精确解是 $UV^\top$（通过 SVD 计算），但 SVD 太慢（$O(mn^2)$）。Muon 使用 **Newton-Schulz 迭代**（一种多项式矩阵迭代），仅需矩阵乘法——天然适合 GPU 的 Tensor Core。

迭代的核心：给定 SVD $G = USV^\top$，每步 NS 迭代产生 $U \varphi^N(S) V^\top$，其中 $\varphi$ 是一个五次多项式。通过选择合适的系数 $(a, b, c)$，使 $\varphi^N(x) \to 1$（在区间 $[0, 1]$ 上），最终逼近 $UV^\top$。

### 2.2 Muon 完整算法

```
Muon 优化器:

输入: 2D 参数 W，学习率 η, 动量 β, Nesterov 标志

for each step:
  1. g_t = ∇L(W)                                    # 计算梯度
  
  2. M_t = β · M_{t-1} + g_t                        # 更新动量
  
  3. if Nesterov:
       G = g_t + β · M_t                            # Nesterov 风格的"前瞻"梯度
     else:
       G = M_t                                       # 普通动量
  
  4. O = NewtonSchulz5(G)                            # 正交化！
  
  5. W = W - η · O                                   # 参数更新
```

其中 `NewtonSchulz5` 的定义：

```python
def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)              # 精心调优的系数
    X = G.bfloat16()
    X /= (X.norm() + eps)                             # 归一化（最奇异值 ≤ 1）
    if G.size(0) > G.size(1):                          # 转置确保 XXᵀ 更小
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A                          # B = b·GGᵀ + c·(GGᵀ)²
        X = a * X + B @ X                              # X = aG + B·G (五次多项式)
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

### 2.3 系数调优：牺牲收敛性换取加速

这是 Muon 的一个巧妙设计选择：

- **收敛多项式**（如 $(a, b, c) = (2, -1.5, 0.5)$）在 $N \to \infty$ 时正确收敛到 $UV^\top$
- **Muon 的五次多项式**：$(a, b, c) = (3.4445, -4.7750, 2.0315)$ **故意不收敛**——在多次迭代后，奇异值不是都变成 1，而是分布在大约 $\text{Uniform}(0.5, 1.5)$

设计理念：最大化 $\varphi'(0) = a$（在 x=0 处的斜率），以便**快速放大小奇异值**。Newton-Schulz 只需 5 步，比精确收敛需要的步数少得多。经验上 $\varepsilon \approx 0.3$（奇异值波动 ±30%）**不损害 loss 曲线**。

### 2.4 关键实现细节

| 细节 | 说明 |
|------|------|
| **仅用于 2D 参数** | hidden layer 权重、卷积核（flatten 最后三维） |
| **AdamW 用于其余参数** | embeddings、bias、LayerNorm、分类头 |
| **Q,K,V 分别应用** | 不要合并为一个 QKV 矩阵 |
| **Nesterov 动量** | 在所有实验中优于普通动量 |
| **bfloat16 稳定** | NS 迭代可在 bfloat16 中运行，无需 fp32 |
| **转置优化** | 确保 $XX^\top$ 是 $m \times m$（$m = \min(d_{\text{out}}, d_{\text{in}})$） |
| **FLOP 开销** | $T \cdot m/B$，其中 $T=5$，$m$ 是 model dim，$B$ 是 batch tokens。典型 LLM 训练 <1% |

### 2.5 Moonshot AI 的两项关键扩展（arXiv:2502.16982）

当 Moonshot AI 将 Muon 扩展到 16B MoE 级别训练 Moonlight 时，发现了两个必需的修改：

**修改 1：添加权重衰减**

原始 Muon 由于 sign-一致的更新幅度，权重不会自然增长。但在长时间训练中，权重仍然可能超过 bfloat16 范围。解决方案：

$$W_t = W_{t-1} - \eta_t \cdot (O_t + \lambda \cdot W_{t-1})$$

其中 $\lambda = 0.1$（与 AdamW 的 weight decay 相同）。

**修改 2：RMS-一致缩放（Per-Parameter Update Scale Adjustment）**

Muon 的正交化输出的 RMS 自然依赖矩阵形状（约 $1/\sqrt{\max(d_{\text{out}}, d_{\text{in}})}$）——大矩阵更新太小，小矩阵更新太大。

修正公式：

$$W_t = W_{t-1} - \eta_t \cdot (0.2 \cdot O_t \cdot \sqrt{\max(d_{\text{out}}, d_{\text{in}})} + \lambda \cdot W_{t-1})$$

缩放因子 $0.2$ 使 RMS 与 AdamW 的经验范围（~0.2-0.4）对齐，实现 AdamW 超参数的即插即用迁移。

### 2.6 MuonClip：处理 QK 注意力爆炸

在训练 Kimi K2（1T 参数 MoE）时，Moonshot AI 发现 Muon 的正交化更新引入了一个新问题：Q、K 投影在训练早期产生的注意力分数可能超过 100。原因：正交化加大了 Q、K 方向的探索自由度。

**MuonClip** 增加了后更新安全检查：

```
if max(QK_scores) > threshold t:
    rescale W_q and W_k proportionally
    η_rescale = t / max_score
    with balancing factor α ≈ 0.5
```

这从源头上控制注意力分数的上限，比 soft-capping 或 QK-Norm 更直接。

---

## 三、Experiments and Key Findings

### 3.1 速度和样本效率

| 任务 | AdamW | **Muon** | 提升 |
|------|:-----:|:--------:|:----:|
| CIFAR-10（94% 准确率，A100-seconds） | 3.3s | **2.6s** | **快 1.27×** |
| NanoGPT speedrun（FineWeb 3.28 val loss） | baseline | **1.35× faster** | 世界纪录 |
| 1.5B Transformer → GPT-2 XL 质量 | 13.3h (8×H100) | **10h** | **快 1.33×** |
| 774M 扩展 | — | **持续加速** | 随规模不退化 |

### 3.2 Moonlight: 生产级 MoE 训练验证（Moonshot AI 2025）

| 基准 | DeepSeek V3-Small (AdamW) | **Moonlight (Muon)** |
|------|:-------------------------:|:--------------------:|
| MMLU | 53.3 | **60.4** |
| HumanEval | 26.8 | **37.2** |
| GSM8K | 31.4 | **45.0** |
| **训练效率** | 1× baseline | **~2×（~52% AdamW FLOPs）** |

Moonlight：16B-total / 2.24B-activated MoE，训练 5.7T tokens。在仅 52% 的 FLOPs 下达到 compute-optimal AdamW 的 loss。

### 3.3 优化器对比（NanoGPT speedrun）

Muon 在样本效率和 wallclock 时间两个维度均超越所有对比优化器（AdamW, Lion, Sophia, Shampoo 等）。

---

## 四、Limitations and Challenges

1. **仅适用于 2D 参数**：embeddings、bias、LayerNorm、分类头仍需 AdamW。需要 Hybrid Optimizer 管理两部分
2. **不适合小批量**：正交化假设批量梯度足够大才能可靠估计"好的"方向。batch_size < 64 时方向估计噪声大
3. **预训练专用**：在微调场景的表现未充分验证（Keller Jordan 自己标注了这一点）
4. **QK 注意力爆炸**：使用 Muon 训练 Transformer 时 Q、K 投影可能产生过大的注意力分数，需要 MuonClip 专门处理
5. **分布式实现复杂**：Newton-Schulz 迭代中的矩阵乘法跨 GPU 分片不如逐元素操作自然。Moonshot AI 使用 ZeRO-1 风格 local gather 解决
6. **社区采纳度低**：截至 2026 年中，大部分社区仍使用 AdamW。Muon 是"有前景但未成标准"的阶段
7. **缺少严格收敛理论**：sign-like 更新 + 非收敛多项式组合的理论保证不完整
8. **权重衰减缺失（原始版本）**：Keller Jordan 原版不含 weight decay，长时间训练有不稳定风险

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 相关工作 | 年份 | 与 Muon 的关系 |
|---------|:----:|---------------|
| **Shampoo** (Gupta et al.) | 2018 | Muon（无动量）等于 Shampoo 去掉累积的极限 |
| **AdamW** (Loshchilov & Hutter) | 2019 | embedding/head/bootstrap 部分仍用 AdamW |
| **Lion** (Chen et al.) | 2023 | 与 Muon 并列的"新型优化器"，均试图超越 AdamW |
| **Orthogonal-SGDM** (Tuddenham et al.) | 2022 | Muon 的前身——正交化 + SVD + 后动量，但被 SGD 超越 |
| **Stochastic Spectral Descent** (Carlson et al.) | 2015a, 2015b, 2016 | 最早的正交化优化器，用随机 SVD 近似 |
| **Moonlight** (Liu et al., Moonshot AI) | 2025 | 首次验证 Muon 的生产级扩展（16B MoE） |
| **MuonClip / Kimi K2** (Moonshot AI) | 2025 | Muon + QK 剪裁，在 1T 参数 Kimi K2 上零训练崩溃 |
| **SOAP** (Vyas et al.) | 2024 | 结合 Shampoo + Adam——与 Muon 不同的优化路线 |

**影响评估**：Muon 是 2024 年底最重要的优化器创新之一。虽然通过博客而非论文发布，但其在 NanoGPT speedrunning 的记录（无人用 AdamW 能打破）提供了一种新颖的信任机制——不需要相信作者，只需相信社区有人能调好 AdamW。Moonshot AI 的生产级验证（Moonlight, Kimi K2）证明 Muon 不仅仅是一个玩具，而是真正可能替代 AdamW 的候选者。

---

## 六、Implications for You / Hardware Compatibility

### 6.1 Muon vs AdamW vs Lion 对比

| 特性 | AdamW | Lion | **Muon** |
|------|:-----:|:----:|:--------:|
| 每参数额外状态 | m+v = 2× | m = 1× | **m = 1×** |
| 更新方式 | 逐元素自适应 | 逐元素符号 | **矩阵级正交化** |
| 计算开销 | baseline | <baseline | **NS迭代 ~<1% FLOPs** |
| 参数适用范围 | 全部 | 全部 | **仅 2D hidden** |
| BF16 稳定 | ✅ | ✅ | ✅ |
| 微调安全性 | ✅ | ⚠️ | ⚠️ (待验证) |
| 社区生态 | 🔥🔥🔥 | 🔥🔥 | 🔥 |

### 6.2 显存对比（7B 模型训练）

| 优化器 | 每参数额外状态 | 7B 额外显存 | 备注 |
|:------|:-------------:|:-----------:|:-----|
| AdamW (bf16) | m+v = 2 × 2B | ~28GB | 基准 |
| Lion (bf16) | m = 2B | ~14GB | — |
| **Muon (bf16)** | **m = 2B** | **~14GB** | 仅用于 hidden 层 ~80% 参数 |

*注：Muon 在 embedding/head/LayerNorm 仍用 AdamW，所以实际优化器显存是 Muon (80%) + AdamW (20%) ≈ 17GB——介于 Lion 和 AdamW 之间。*

### 6.3 使用指南

| 方面 | 建议 |
|------|------|
| **何时用 Muon** | 从零预训练 Transformer（尤其是 hidden layers 权重多且为 2D 的场景），显存/速度都敏感的场合 |
| **何时用 AdamW** | 微调预训练模型、embedding/head/LayerNorm、不确定性高的场景 |
| **Hybrid 配方** | Muon (hidden 权重) + AdamW (embedding, head, bias, LayerNorm) |
| **学习率** | 与 AdamW 相同超参数（得益于 RMS-一致缩放） |
| **权重衰减** | λ=0.1（Moonshot AI 的生产配置） |
| **Q,K,V 分开** | 不要合并为一个 QKV 矩阵，分别应用 Muon |
| **NS 迭代步数** | 默认 steps=5（典型 LLM 训练） |
| **MuonClip** | 训练 Transformer 时建议启动（threshold ~50-100） |

### 6.4 硬件兼容性

| GPU 类型 | 状态 | 备注 |
|:---------|:----:|:-----|
| H100, H200 | ✅ 最佳 | bf16 NS 迭代原生支持 |
| A100 (80GB) | ✅ | bf16 完整支持 |
| A100 (40GB) | ✅ | 显存足够 |
| RTX 4090 (24GB) | ✅ | 消费级 bf16 + 7B 预训练可能需要 GaLore 结合 |
| RTX 3090 (24GB) | ✅ | 同上 |
| RTX 3060 (12GB) | ⚠️ | 需要量化+小模型 |
| M1/M2/M3/M4 Mac (MPS) | ⚠️ | NS 迭代需验证 MPS 性能 |

### 6.5 实践建议

- **从 AdamW 迁移**：Moonshot AI 证明了 RMS-一致缩放使超参数直接迁移。如果有人问你"Muon 的学习率多少"——答案是"和 AdamW 一样"
- **预训练第一，微调其次**：社区对微调场景的 Muon 经验还很少。建议预训练用 Muon，微调用 AdamW/LoRA
- **关注 QK 稳定性**：在 Transformer 训练中监控注意力分数的最大值。如果超过 50-100，考虑引入 MuonClip
- **结合 FSDP/ZeRO**：Newton-Schulz 迭代跨 GPU 分片实现有额外通信。Moonshot AI 的方案是最佳参考

---

## PDF

[[Muon 原文.pdf]]
