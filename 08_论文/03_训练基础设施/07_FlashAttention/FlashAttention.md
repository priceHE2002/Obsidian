---
tags:
  - 论文
  - 训练基础设施
  - 注意力优化
  - 显存优化
  - IO-Aware
created: 2026-06-30
paper_title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
paper_authors: "Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré"
paper_year: 2022
paper_venue: "NeurIPS 2022"
paper_citations: "~5,000+"
paper_url: "https://arxiv.org/abs/2205.14135"
github: "https://github.com/HazyResearch/flash-attention"
---

# FlashAttention

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
*Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré | Stanford (Hazy Research) | NeurIPS 2022 (Oral) | arXiv: 2205.14135*

> 通过 IO-aware 的 tiling 策略，将标准注意力的显存复杂度从 $O(N^2)$ 降低到 $O(N)$，并在不损失精度的前提下实现 2-4 倍训练加速。FlashAttention 已成为所有现代 LLM 训练框架中注意力计算的标配实现，从 GPT-4 到 LLaMA 2 均依赖其加速。

---

## 一、Background / Core Idea

### 1.1 问题：注意力计算的显存瓶颈

标准 Scaled Dot-Product Attention 的计算公式：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N$ 为序列长度，$d$ 为注意力头维度。

**标准实现需要在 HBM（高带宽显存）中存储中间矩阵 $S = QK^\top \in \mathbb{R}^{N \times N}$**：

| 序列长度 $N$ | 注意力矩阵大小（fp16） | 占显存比例（80GB A100） |
|:-----------:|:---------------------:|:----------------------:|
| 1K | 2 MB | 忽略 |
| 8K | 128 MB | 0.2% |
| **32K** | **2 GB** | **2.5%** |
| **64K** | **8 GB** | **10%** |
| **128K** | **32 GB** | **40%** |
| **512K** | **512 GB** | **OOM** |

显然，$O(N^2)$ 显存增长使得长序列训练在标准实现下不可行。

### 1.2 核心洞察：IO-Aware 的层次化计算

论文的核心洞察是**计算速度（FLOPs）已不是瓶颈，显存带宽（HBM ↔ SRAM 的数据移动）才是**：

```
GPU 显存层次（A100-80GB）:
┌─────────────────────────────────────────┐
│  HBM (High Bandwidth Memory): 80GB     │
│  带宽: 2 TB/s                          │
├─────────────────────────────────────────┤
│  SRAM / 共享内存: 192 KB per SM         │
│  (总计 108 SM × 192KB = ~20MB)         │
│  带宽: ~19 TB/s (10× HBM)              │
└─────────────────────────────────────────┘
```

**标准 Attention 的问题**：读取 $Q,K,V$（HBM → SRAM）→ 计算 $S$ → 写回 HBM → 从 HBM 读取 $S$ → softmax → 写回 HBM → 读取计算输出 → 写回 HBM。

$S$ 矩阵 $O(N^2)$ 的多次 HBM 读写是性能瓶颈。FlashAttention 通过 tiling 将所有中间计算保留在 SRAM 中。

### 1.3 与标准 Attention 对比

| 方面 | 标准 Attention | **FlashAttention** |
|:----|:-------------:|:------------------:|
| 精确性 | 精确 | **精确**（无近似） |
| HBM 读写量 | **$O(N^2 + Nd)$** | **$O(N^2d^{-1})$**（实际为 $O(Nd)$ 当 $d$ 固定） |
| 显存复杂度 | $O(N^2)$ | **$O(N)$** |
| 计算复杂度 | $O(N^2d)$ | $O(N^2d)$（相同） |
| 是否近似 | 否 | 否 |
| 壁钟加速 | 1× | **2-4×** |

**FlashAttention 不降低 FLOPs，只减少 HBM 访问**——这是其核心工程突破。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 Tiling 策略：在线 Softmax 算法

FlashAttention 的关键技术是 **Safe Online Softmax**：在不写出完整 $S$ 矩阵的情况下，按块计算并累加 softmax。

标准 Softmax：

$$m(x) = \max_i x_i, \quad p(x)_i = e^{x_i - m(x)}, \quad \text{softmax}(x)_i = \frac{p(x)_i}{\sum p(x)_i}$$

**Online Softmax（按块处理）**：

初始化：$m_0 = -\infty, \quad d_0 = 0, \quad o_0 = \mathbf{0}$

对每个块 $j$：
1. 读取块 $K_j, V_j$ 到 SRAM
2. 计算块 $S_j = QK_j^\top / \sqrt{d}$（在 SRAM 中）
3. 更新 $\tilde{m}_j = \max(S_j)$（行最大值）
4. 更新 $m^{\text{new}} = \max(m_{j-1}, \tilde{m}_j)$
5. 更新 $d^{\text{new}} = e^{m_{j-1} - m^{\text{new}}} \cdot d_{j-1} + \sum e^{S_j - m^{\text{new}}}$
6. 更新 $o^{\text{new}} = e^{m_{j-1} - m^{\text{new}}} \cdot o_{j-1} + e^{S_j - m^{\text{new}}} \cdot V_j$

**关键**：整个过程中 $S_j \in \mathbb{R}^{B_r \times B_c}$ 完全在 SRAM 上，只有 $Q, K, V$ 的分片和最终输出在 HBM 中。

### 2.2 分块大小（Tiling Dimensions）

设 SRAM 大小为 $M$，则分块参数需满足：

$$\text{SRAM 需求} = \underbrace{B_r \cdot d}_{\text{Q 块}} + \underbrace{B_c \cdot d}_{\text{K 块}} + \underbrace{B_c \cdot d}_{\text{V 块}} + \underbrace{2 B_r \cdot B_c}_{\text{S 块}} + \underbrace{4 B_r d}_{\text{输出}} \leq M$$

A100 上典型分块：$B_r = B_c = \min\left(\left\lfloor \sqrt{\frac{M}{4 \cdot \text{sizeof(fp16)}}} \right\rfloor, 64\right)$

典型值：$B_r = B_c = 64$（A100 SRAM=192KB）

### 2.3 精确性证明（Paper Theorem）

论文证明了在线 softmax 算法的**数学等价性**：

> **Theorem 1**: FlashAttention 计算的输出 $\tilde{O} = \text{Attention}(Q, K, V)$ 与标准 attention 的结果在数学上**完全等同**，仅受浮点舍入误差影响（与标准实现相同的误差界）。

这一证明至关重要——它确保了 FlashAttention 不是一种近似方法。

### 2.4 反向传播的 IO 优化

FlashAttention 的反向传播同样采用 tiling 策略，并引入 **重计算（recomputation）**：

| 方法 | 前向 HBM 访问 | 反向额外存储 | 反向计算量 |
|:----|:------------:|:-----------:|:---------:|
| 标准 Attention | $O(N^2 + Nd)$ | $S, P$（$O(N^2)$） | $O(N^2d)$ |
| FlashAttention 前向 | $O(Nd)$ | 无（不存 $S$） | $O(N^2d)$ |
| FlashAttention 反向 | $O(Nd)$ | **无** | $O(N^2d)$ + **重计算开销** |

**反向重计算**：FlashAttention 在反向传播中重新计算（而非读取）$S$ 和 softmax 值 $P$。这在计算量上多了一次前向 FLOPs，但**消除了 $O(N^2)$ 的 HBM 读取**——对于注意力 FLOPsbound 远低于 HBM bound 的现代 GPU 而言，重计算是赚的。

### 2.5 嵌套块稀疏 FlashAttention（补充）

论文探讨了 FlashAttention 如何与块稀疏性（Block-Sparse）结合：

$$\text{BlockSparseFlashAttention: 仅计算非零块的注意力}$$

对预定义稀疏模式（如固定窗口 + 全局 token），可实现 4-8× 额外加速。

---

## 三、Experiments and Key Findings

### 3.1 训练加速

| 模型 | 序列长度 | 标准 Attention（ms/step） | FlashAttention（ms/step） | 加速比 |
|:----|:-------:|:-----------------------:|:------------------------:|:-----:|
| GPT-2 (1.5B) | 1K | — | — | 1.3× |
| GPT-2 (1.5B) | 2K | — | — | 1.5× |
| GPT-2 (1.5B) | 4K | — | — | **2.0×** |
| BERT-large | 512 | — | — | 1.3× |
| **Long-range Arena** | 4K | — | — | **2.0-4.0×** |

FlashAttention 在长序列场景下的加速最为显著。

### 3.2 显存节省

| 模型 | $N$（序列长度） | 标准 Attention | FlashAttention | 节省 |
|:----|:--------------:|:--------------:|:--------------:|:----:|
| GPT-2 1.5B | 1K | OOM (12GB) | 7.9GB | — |
| GPT-2 1.5B | 2K | OOM | 9.8GB | — |
| GPT-2 large | 512 | 16.7GB | 13.3GB | 20% |
| GPT-2 large | 1K | OOM | 14.7GB | — |

更为关键的是，FlashAttention 使得 $N=64K$ 以上的序列训练成为可能——这在此前是根本不可能的。

### 3.3 端到端 BERT 训练

| 配置 | 标准（steps/s） | FlashAttention（steps/s） | 加速 |
|:----|:--------------:|:------------------------:|:----:|
| BERT-base (seq=512) | — | — | 1.3× |
| BERT-large (seq=512) | — | — | 1.2× |
| BERT-large (seq=1024) | — | — | 1.5× |
| BERT-large (seq=2048) | — | — | **2.0×** |

**BERT-large 序列从 512 扩展到 2048 时，FlashAttention 实现 2× 加速**，且保持了与标准 Attention 完全相同的 loss 曲线。

---

## 四、Limitations and Challenges

1. **对 SRAM 大小的依赖**：FlashAttention 的分块大小受 GPU SRAM 限制。A100 (192KB) 充足，但低端 GPU（如 RTX 3060 的 96KB）的分块更小、效率下降
2. **仅加速 Attention 部分**：FlashAttention 只优化注意力计算，不处理 MLP 或 Embedding 的带宽瓶颈。整体加速比取决于 Attention 占总计算的比例（Attention-heavy = 加速大）
3. **CUDA 实现复杂**：FlashAttention 需要手工 CUDA kernel，不支持 TorchScript 或 `torch.compile` 的自动优化
4. **反向重计算增加功耗**：反向多一次前向 FLOPs，增加了 GPU 能耗（虽然时间减少）
5. **与各种 Attention Mask 的兼容性**：Causal mask 已支持，但 Alibi、RoPE + FlashAttention 的融合 kernel 需要额外开发（FlashAttention v2 部分解决）
6. **序列并行限制**：当 $N$ 极大（>128K）时，Single-GPU SRAM 仍不够，需要序列并行（sequence parallelism）配合

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 FlashAttention 的关系 |
|---------|:----:|------------------------|
| **FlashAttention-2** (Dao, 2023) | 2023 | 重新设计 CUDA kernel 调度，减少非矩阵乘操作，2× 进一步加速 |
| **FlashAttention-3** (Dao et al., 2024) | 2024 | 利用 Hopper GPU 的 WGMMA/TMA 指令，利用低精度（FP8），3× 加速 |
| **FlashDecoding** (Dao et al., 2023) | 2023 | FlashAttention 推理优化版本，解决 KV cache 的 batch 维度分裂问题 |
| **xFormers** (Meta) | 2022 | Meta 的注意力优化库，受 FlashAttention 启发实现 fmha |
| **Ring Attention / Sequence Parallelism** | 2023 | 将 FlashAttention 的 tiling 扩展到多 GPU 场景 |
| **Mamba / State Space Models** | 2023 | 受 FlashAttention 效率启发，设计线性的状态空间替代注意力 |
| **FlexAttention** (PyTorch) | 2024 | PyTorch 原生支持 FlashAttention 风格的自定义注意力模式 |
| **DeepSpeed Inference** (Microsoft) | 2023 | 集成 FlashAttention 加速推理 |
| **vLLM** (Kwon et al., 2023) | 2023 | 推理引擎，采用 FlashAttention 风格的 PagedAttention |

**影响评估**：FlashAttention 是整个 LLM 训练基础设施的**关键使能技术**。FlashAttention 的提出使得 Llama 2/3、GPT-4、Gemini、Qwen 等模型的高效长序列训练成为可能。到 2024 年，几乎所有大模型训练框架都默认启用 FlashAttention。

---

## 六、Implications for You / Hardware Compatibility

### GPU 兼容性矩阵

| GPU 架构 | SRAM 大小 | FlashAttention v1 | FlashAttention-2 | FlashAttention-3 | 注意事项 |
|:---------|:---------:|:-----------------:|:----------------:|:----------------:|:---------|
| **A100 (Ampere)** | 192KB/SM | ✅ 完整支持 | ✅ 完整支持 | ❌ | 默认平台 |
| **H100 (Hopper)** | 228KB/SM | ✅ | ✅ | ✅ 最佳 | FA3 利用 FP8 Tensor Core |
| **V100 (Volta)** | 128KB/SM | ⚠️ 支持 | ⚠️ 但效率低 | ❌ | 分块更小导致加速比低 |
| **RTX 4090 (Ada)** | 128KB/SM | ✅ | ✅ | ❌ | 消费级最佳选择 |
| **RTX 3090 (Ampere)** | 128KB/SM | ✅ | ✅ | ❌ | 低于 A100（SM 数少） |
| **RTX 3060 (Ampere)** | 96KB/SM | ⚠️ 可用 | ⚠️ 效率下降 | ❌ | 小 SRAM 约束明显 |
| **H200 (Hopper)** | 228KB/SM | ✅ | ✅ | ✅ | 受益于更大 HBM |

### 对训练实践的影响

- **短序列（<2K）**：FlashAttention 加速有限（~1.3×），因为 MLP 成为瓶颈
- **中序列（2K-8K）**：2-3× 加速，同时显存从 $O(N^2)$ 降为 $O(N)$
- **长序列（8K-128K）**：对 Llama 2 长上下文微调等场景是**必需的**。无 FlashAttention 则 $N>8K$ 的 GPU 显存不可行
- **超长序列（>128K）**：需 FlashAttention + Sequence Parallelism + 其他优化组合

### 实践配置建议

```python
# HuggingFace Transformers 中启用 FlashAttention 的标准方式
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # 仅在 FlashAttention-2 安装后可用
)
```

### 硬件兼容性总结
- ✅ A100/H100：FlashAttention v1/v2 完整支持，FA3 专为 H100 优化
- ✅ RTX 3090/4090：FlashAttention v1/v2 消费级最佳
- ⚠️ V100 / RTX 3060：可用但加速比降低，因 SRAM 较小
- ❌ CPU / Mac MPS：不支持（需要 CUDA + 高带宽 SRAM）

---

## PDF

[[FlashAttention 原文.pdf]]
