---
title: "CounterStruct DL 入门课 · 前置知识"
tags:
  - 深度学习
  - 持续学习
  - 动态稀疏
  - CounterStruct
  - 基础知识
created: 2026-08-21
---

# CounterStruct DL 入门课 · 前置知识

> 这是为真正读懂 [[CounterStruct_v2.1_方案导读_逐章讲解]] 而补的通用深度学习知识，不是方案原文。目标：让你以后能自己看懂方案里的公式和代码。配套正式方案见 [[CounterStruct_v2.1_纯实验方案_正式版]]。

这份文档从 DL 初学者视角出发，把 CounterStruct 会用到的概念逐个补齐。建议按这个顺序读：

$$
\text{神经网络/反向传播}
\rightarrow
\text{Transformer FFN}
\rightarrow
\text{AdamW}
\rightarrow
\text{sparsity + mask}
\rightarrow
\text{continual learning}
\rightarrow
\text{RigL}
\rightarrow
\text{CounterStruct}
$$

## 目录

1. 参数到底是什么（第 1 章）
2. Gradient 与 Backpropagation（第 2 章）
3. Batch、Epoch、Step（第 3 章）
4. Transformer / Attention / FFN（第 4–6 章）
5. Frozen 冻结（第 7 章）
6. Adam / AdamW（第 8–11 章）
7. Dense / Sparsity / Mask（第 12–14 章）
8. Static vs Dynamic Sparsity / Prune / Grow（第 15–16 章）
9. Structured Sparsity / Exact 2:4（第 17–19 章）
10. Continual Learning / Catastrophic Forgetting（第 20–22 章）
11. 四类 CL 路线 / CounterStruct 的位置（第 23–24 章）
12. RigL / SRigL（第 25–27 章）
13. STE（第 28–29 章）
14. Counterfactual / Intervention / Shadow / Horizon（第 30–33 章）
15. Critic / Linear / Ridge Regression（第 34–36 章）
16. Prior / Residual / Credit Assignment / Additivity（第 37–40 章）
17. Johnson Graph / Spearman（第 41–42 章）
18. Seed / SE / Bootstrap（第 43–45 章）
19. Ablation / Calibration / Go-No-Go（第 46–48 章）

---

## 第 1 章：神经网络最基础的东西——参数到底是什么？

先忘掉 Transformer。考虑最简单的一层：

$$
y = Wx + b.
$$

假设输入：

$$
x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix},
$$

权重：

$$
W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}.
$$

那么：

$$
y_1 = w_{11}x_1 + w_{12}x_2 + b_1.
$$

你可以把一个 weight $w_{12}$ 理解成：输入神经元 $x_2$ 到输出神经元 $y_1$ 的一条连接。所以神经网络学习，本质上就是调整大量 $w_i$。

### 1.1 Forward 是什么？

输入数据进去：

$$
x \rightarrow \text{模型} \rightarrow \hat y.
$$

这叫 **forward pass（前向传播）**。

### 1.2 Loss 是什么？

假设正确答案是 $y$，模型预测是 $\hat y$。我们需要一个数字衡量「错得多不多」：

$$
L(\hat y, y).
$$

这就是 **loss**。训练目标就是让 loss 越来越低：

$$
\min_W L(W).
$$

---

## 第 2 章：Gradient 与 Backpropagation

模型有几十亿参数，不可能人工调。所以需要计算：

$$
\frac{\partial L}{\partial w_i}.
$$

这叫 **gradient（梯度）**，告诉你：如果把 $w_i$ 稍微增加一点，loss 会怎么变化？

例如 $g_i = \frac{\partial L}{\partial w_i} > 0$，说明 $w_i \uparrow$ 通常会让 $L \uparrow$。那么为了降低 loss，应该让 $w_i \downarrow$。

最简单的梯度下降：

$$
w_i \leftarrow w_i - \eta g_i.
$$

其中 $\eta$ 叫 **learning rate（学习率）**。

### 2.1 Backward 是什么？

- Forward：$x \rightarrow \hat y \rightarrow L$
- Backward：$L \rightarrow \frac{\partial L}{\partial W}$

所以训练一次大概是：

```text
输入 batch
   ↓
forward
   ↓
loss
   ↓
backward
   ↓
gradient
   ↓
optimizer
   ↓
更新参数
```

这个循环你之后会看到几千、几万次。一次参数更新通常叫 **optimizer step**。

---

## 第 3 章：Batch、Epoch、Step

假设训练集 $10000$ 个样本，batch size 为 $32$。那么一次只拿 $32$ 个样本计算 gradient。大约 $10000/32 \approx 313$ 个 step 扫完一次数据集。

完整扫一次训练集叫 **1 epoch**。所以 epoch ≠ step。例如训练 5 epochs：

$$
313 \times 5 \approx 1565 \ \text{optimizer steps}.
$$

这个概念对后面理解 $20\%,40\%,60\%,80\%$ 的 mutation schedule 很重要。

---

## 第 4 章：Transformer 到底是什么？

Transformer 可以先粗略理解成很多层重复堆叠：

```text
Token Embedding
     ↓
Transformer Block 1
     ↓
Transformer Block 2
     ↓
...
     ↓
Transformer Block L
     ↓
Output
```

一个 Transformer Block 里最重要的两块是 **Attention** 和 **FFN**：

```text
输入
 ↓
Attention
 ↓
FFN
 ↓
输出
```

实际还会有 residual connection、normalization 等，但你现在先抓这两个。

---

## 第 5 章：Attention 和 FFN 分别干什么？

可以用非常粗略但有用的直觉：

- **Attention** 回答：当前 token 应该从别的 token 那里读取什么信息？例如「小明把苹果给小红，因为她饿了」，Attention 帮助模型判断「她」与「小红」的关系。
- **FFN** 回答：当前 token 已经聚合了上下文以后，内部 feature 应该如何非线性变换？

所以可以粗略记：

$$
\text{Attention：token 间交互},
\qquad
\text{FFN：每个 token 内部 feature 变换}.
$$

CounterStruct 主要修改 **FFN**，而不是 Attention。

---

## 第 6 章：Transformer FFN

一个简化 FFN 可以写成：

$$
h \rightarrow W_{\text{up}} \rightarrow \text{activation} \rightarrow W_{\text{down}} \rightarrow h'.
$$

通常 $d_{\text{ffn}} > d_{\text{model}}$。比如 CounterStruct 的 Qwen3-1.7B：

$$
d_{\text{model}} = 2048,
\qquad
d_{\text{ffn}} = 6144.
$$

所以 $W_{\text{up}}$ 会把 $2048$ 维扩展到 $6144$，而 $W_{\text{down}}$ 再把 $6144$ 压回 $2048$。

方案只修改最后 8 层中的 `up_proj` 和 `down_proj`，Attention 与 `gate_proj` 都被冻结。

---

## 第 7 章：Frozen 是什么意思？

Frozen（冻结）意思就是：这些参数参加 forward，但不训练。比如：

```text
Layer 0-19      Frozen
Layer 20-27
    Attention   Frozen
    gate_proj   Frozen
    up_proj     Train
    down_proj   Train
```

所以 CounterStruct 并不是 1.7B 所有参数全都重新训练，它只研究一个固定 target region。这样做有两个好处：

1. 研究问题更可控；
2. shadow branch 可以缓存前面 frozen layers 的输出，降低开销。

---

## 第 8 章：SGD 为什么不够？进入 Adam

最简单 SGD：

$$
w_{t+1} = w_t - \eta g_t.
$$

但是 gradient 会：有噪声、不同参数尺度不同、正负来回震荡。Adam 因此为每个参数维护两个状态。

---

## 第 9 章：Adam 的 $m$ 和 $v$

第一动量（类似「惯性」，最近 gradient 方向的滑动平均）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t.
$$

第二动量（反映最近 gradient 大小平方的滑动平均、波动有多大）：

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2.
$$

Adam 最后类似：

$$
w \leftarrow w - \eta \frac{\hat m}{\sqrt{\hat v} + \epsilon}.
$$

因此真正的 update 不只取决于 $g$，还取决于 $m, v$。

---

## 第 10 章：AdamW 又是什么？

AdamW 在 Adam 基础上额外把 weight decay 与 gradient update 分离，可以粗略写：

$$
w_{t+1} = w_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda w_t.
$$

所以当 CounterStruct 问「如果某个 dormant connection 被激活，下一步会发生什么？」，它不能只看 $g$，还要考虑 $m, v, \eta, \lambda$。这就是方案里 **optimizer-aware** 的含义。

---

## 第 11 章：为什么 newborn connection 的 $m,v$ 很重要？

假设某个 weight 原来是 dormant（$w=0$），现在突然激活。它还没有历史 optimizer state，所以 $m=0, v=0$。这和一个训练很久的 active weight 完全不同。

所以 **old connection ≠ newborn connection**，即便二者当前 gradient 一样。这也是为什么「只看 gradient 大小」可能太简单。

---

## 第 12 章：什么叫 Dense Model？

普通神经网络的一层，几乎所有 weight 都参与计算。例如：

$$
W = \begin{bmatrix} 0.4 & 0.2 & -0.7 & 0.1 \end{bmatrix}.
$$

四个 weight 全 active。这叫 **dense**。

---

## 第 13 章：什么叫 Sparsity？

如果我们故意让很多 weight 为 0：

$$
W = \begin{bmatrix} 0.4 & 0 & -0.7 & 0 \end{bmatrix},
$$

那么 $50\%$ weight 为 0。这就是 **sparsity（稀疏）**。

---

## 第 14 章：Mask 是什么？

我们不一定真的把整个参数 tensor 删除，而是额外保存一个 $M$：

$$
W = [0.4, 0.2, -0.7, 0.1],
\qquad
M = [1, 0, 1, 0].
$$

有效权重：

$$
W_{\text{eff}} = M \odot W = [0.4, 0, -0.7, 0].
$$

这里 $\odot$ 表示逐元素乘法。$M_i=1$ 表示 active，$M_i=0$ 表示 dormant。

---

## 第 15 章：Static Sparsity 与 Dynamic Sparsity

- **Static sparsity**：一开始决定哪些连接保留，之后 mask $M$ 永远不变。
- **Dynamic sparsity**：训练过程中允许 $M$ 发生变化。例如原来 $[1,1,0,0]$ 变成 $[1,0,1,0]$。连接数量还是 2，但连接位置变了。

这叫 **Dynamic Sparse Training（DST）**。CounterStruct 属于 **dynamic topology adaptation** 这一大类。

---

## 第 16 章：Prune 和 Grow

动态稀疏里两个最基础的动作：

- **Prune**：把 active connection 关掉。例如 $M_2: 1 \rightarrow 0$。
- **Grow**：把 dormant connection 激活。例如 $M_3: 0 \rightarrow 1$。

如果总 sparsity 固定，那么往往 **prune 一个 + grow 一个** 同时发生，这就是 **rewiring（重新布线）**。

---

## 第 17 章：为什么要 Structured Sparsity？

如果只是随机 50% weight 为 0：

```text
1 0 1 1 0 0 1 0 ...
```

硬件不一定能很好利用。所以会规定某种规律。比如 **2:4** 表示：每连续四个 weight 里必须正好两个非零。这就叫 **structured sparsity**。

---

## 第 18 章：Exact 2:4

一组 $[w_1,w_2,w_3,w_4]$ 必须满足 $\sum_i M_i = 2$。

合法：$[1,1,0,0]$、$[1,0,1,0]$、$[0,1,0,1]$。

不合法：$[1,1,1,0]$（3 个 active），也不合法：$[1,0,0,0]$（只有 1 个）。

所以叫 **exact 2:4**，不是「大约 50%」，而是每组严格如此。

---

## 第 19 章：为什么 2:4 恰好有 6 个状态？

四个位置选两个：

$$
\binom42 = \frac{4!}{2!\,2!} = 6.
$$

所以：

$$
\mathcal S = \{12, 13, 14, 23, 24, 34\}.
$$

例如 $13$ 就是 $[1,0,1,0]$。这个离散状态空间是理解 CounterStruct 的关键。

---

## 第 20 章：什么是 Continual Learning？

普通机器学习：所有数据一起训练。Continual learning：

$$
T_1 \rightarrow T_2 \rightarrow T_3 \rightarrow \cdots
$$

任务一个接一个来。学习 $T_2$ 时，可能已经不能重新访问 $T_1$ 的数据。

---

## 第 21 章：Catastrophic Forgetting

假设训练完 $T_1$ 准确率 $90\%$，然后学习 $T_2$，重新测试 $T_1$ 变成 $65\%$。下降 $25$ 个百分点，这就是 **灾难性遗忘**。

为什么？因为 $T_2$ 的 gradient 修改了原本对 $T_1$ 有用的参数。

---

## 第 22 章：Stability–Plasticity Dilemma

持续学习的核心矛盾：

- **Stability**：别忘旧任务。
- **Plasticity**：还能学新任务。

如果完全冻结模型，stability 很高但 plasticity 接近 0；如果完全自由更新，plasticity 强但 forgetting 大。所以 continual learning 永远是在找 **稳定性与可塑性的平衡**。

---

## 第 23 章：Continual Learning 的几类经典路线

### 23.1 Replay

保存一部分旧数据，学习新任务时 new data + old examples 一起训练。优点直接，缺点要保存历史数据。

### 23.2 Regularization

不保存旧数据，但告诉模型某些重要参数不要改太多，例如 EWC，大概形式：

$$
L_{\text{new}} + \lambda \sum_i F_i (w_i - w_i^\star)^2.
$$

### 23.3 Parameter isolation

不同任务尽量用不同参数空间，比如 mask、region、subspace。

### 23.4 Adapter / LoRA

Base model 冻结，新增少量可训练参数。例如：

$$
W' = W + BA.
$$

LoRA 就属于这类。

---

## 第 24 章：CounterStruct 的位置

CounterStruct 不是纯 replay（primary setting 明确 **No Historical Samples**），也不是单纯 EWC，也不是 LoRA。它主要走 **fixed-capacity topology adaptation**：不增加 active connection 数量，通过改变连接位置适应新任务。

---

## 第 25 章：RigL 是什么？

理解 CounterStruct 前最重要的 baseline 之一。RigL 的核心思路很直观：

- **Prune**：当前 active weights 中绝对值小的可能不重要，于是删掉（$|w_i|$ small ⇒ prune）。
- **Grow**：当前 dormant weights 中 gradient 大的位置，说明如果激活可能很有用（$|g_i|$ large ⇒ grow）。

所以 RigL 可以粗略记：**small weight prune + large gradient grow**。

---

## 第 26 章：RigL 的核心局限

RigL 的 grow 判断非常「即时」：看到 $|g_i|$ 很大，就说这个 dormant coordinate 值得激活。但它没有真正回答：这条连接激活以后，经过 optimizer 的 8、32、64 个 step，最后还值不值得？

所以 **instantaneous gradient saliency** 与 **future integration value** 不一定是一回事。这就是 CounterStruct 要攻击的核心问题。

---

## 第 27 章：SRigL 是什么？

SRigL 可以先理解成：把 RigL 的动态稀疏思想做成 structured sparsity 版本，更适合作为 exact N:M structured DST 的 baseline。

CounterStruct 方案明确把自己与 RigL/SRigL 的区别写成：

$$
\text{independent coordinate saliency}
\neq
\text{structural-state action value}.
$$

---

## 第 28 章：为什么 dormant weight 也要有 gradient？

正常 masked training 里 $W_{\text{eff}} = M \odot W$，如果 $M_i=0$，这个 weight 对 forward 没贡献，正常 backward 里可能也拿不到我们想要的 dormant gradient signal。

但 dynamic sparsity 恰恰需要知道 dormant 位置里哪个值得长出来。于是需要 **STE**。

---

## 第 29 章：STE 是什么？

Straight-Through Estimator（直通估计器）。方案写：

$$
W_{\text{STE}} = W + \operatorname{stopgrad}(M \odot W - W).
$$

这个表达式的妙处：

- **Forward**：数值上 $W_{\text{STE}} = M \odot W$，所以 forward 仍然看到稀疏模型。
- **Backward**：`stopgrad` 那部分不传播 gradient，所以 backward 看起来更像 $\frac{\partial W_{\text{STE}}}{\partial W} = 1$，于是 dormant coordinate 也能得到 $g_i$。

你可以理解成：前向仍假装它不存在，反向偷偷问——如果它存在，它的 gradient 会怎样？

---

## 第 30 章：什么叫 Counterfactual？

Counterfactual（反事实）：例如真实世界我今天去上课了，反事实问题是「如果我今天没去，会怎样？」

机器学习中：当前 topology 是 $M$，反事实是「如果改成 $M'$，后续 loss 会怎样？」

---

## 第 31 章：什么叫 Intervention？

- **Observation**：看现有系统发生了什么。
- **Intervention**：主动改变一个东西，再看结果。

CounterStruct 的 Keep branch 什么都不变，对比 Transition branch 主动改变 topology。如果其它条件完全一样，那么二者差异更容易归因给 topology intervention。

---

## 第 32 章：Shadow Branch

Shadow（影子分支）：真实模型 checkpoint $C$，复制出 Keep world $C_K$ 和 Transition world $C_T$。两者都训练 8 步，但最后全部丢弃——它们只用于收集实验信号，真实模型不会直接被 shadow training 修改。

---

## 第 33 章：什么叫 Horizon？

Horizon 就是「向未来看多少步」。例如 $H=1$ 只看一步，$H=8$ 看 8 个 optimizer steps，$H=64$ 看 64 steps。

CounterStruct 最核心的问题：$H=1$ 的 score 能不能预测 $H=64$ 的真实结构价值？

---

## 第 34 章：什么叫 Critic？

这个词经常来自 reinforcement learning，你这里不要把它想复杂。Critic 就是 **一个预测「这个 action 有多好」的模型**：

输入 $\phi(a)$，输出 $\hat b(a)$。例如 $\phi(a) = [\text{gradient}, \text{weight}, \text{layer}, \dots]$，Critic 为 $\hat b = \theta^\top \phi(a)$。

---

## 第 35 章：什么是 Linear Regression？

最简单：

$$
y \approx \theta_1 x_1 + \theta_2 x_2 + \cdots.
$$

比如「房价 ≈ 面积 × 系数 + 地段 × 系数 + …」。CounterStruct 里：

$$
\text{structural value} \approx \theta^\top \phi(a).
$$

---

## 第 36 章：Ridge Regression

普通 linear regression 容易在数据少的时候过拟合。Ridge 加 $\lambda\|\theta\|_2^2$，最终类似：

$$
\theta = (X^\top X + \lambda I)^{-1} X^\top y.
$$

CounterStruct 用 **低维 ridge critic** 而不是大 MLP，原因就是 shadow supervision 非常少。

---

## 第 37 章：什么是 Prior？

Prior：在看到当前数据之前已有的经验。CounterStruct 的 $\theta_G^{pre}$ 就是过去任务学到的通用 structural dynamics。

---

## 第 38 章：什么是 Residual？

假设 global critic 预测 $0.4$，但当前 task 实际更喜欢这种 action（$0.7$），差值 $0.3$ 就是当前任务需要额外修正的部分。

所以：

$$
\hat b_t = \theta_G^{pre} + \delta_t,
$$

其中 $\delta_t$ 就是 **task-local residual**。

---

## 第 39 章：Credit Assignment

如果 64 个 action 一起产生 $+0.05$ 的效果，问题：到底哪个 action 有功？这叫 **credit assignment**。

CounterStruct 不假装知道每个 action 的真实 label，它只知道 **bundle-level label**。这一点非常重要。

---

## 第 40 章：Additivity

如果 $Y(A \cup B) = Y(A) + Y(B)$，说明二者效果可加。但神经网络通常有 interaction：

$$
Y(A \cup B) \neq Y(A) + Y(B).
$$

所以 CounterStruct 必须检查：

$$
I_{A,B} = Y_{A \cup B} - Y_A - Y_B.
$$

这就是方案里的 **interaction audit**。

---

## 第 41 章：Johnson Graph

这部分只需要一点组合数学。6 个节点 $12,13,14,23,24,34$，两个 state 如果共享一个 active coordinate 就连边。

例如 $12 \leftrightarrow 13$（共享 $1$），但 $12 \not\leftrightarrow 34$（没有共享 active coordinate）。这就是 $J(4,2)$。

注意：CounterStruct 不声称 Johnson graph 本身新颖，方案明确禁止这个 claim。

---

## 第 42 章：Spearman Correlation

这在机制实验里非常重要。假设 critic 预测 action 排名 `a3 > a1 > a4 > a2`，真实实验结果 `a3 > a1 > a2 > a4`，很接近。Spearman $\rho$ 就比较 **两个排名有多一致**：

- $\rho = 1$：完全一致。
- $\rho = 0$：没有明显关系。
- $\rho = -1$：完全反过来。

---

## 第 43 章：Seed

深度学习有大量随机因素：parameter initialization、batch order、dropout、sampling、GPU numerical variation。所以 $seed=42$ 和 $seed=43$ 可能给不同结果，因此需要 multiple seeds。

---

## 第 44 章：Standard Error

跑 5 次得到 $x_1, \dots, x_5$，先算 $\bar x$，再估计平均值有多不稳定。所以论文经常写 **mean ± SE**。

---

## 第 45 章：Bootstrap

如果很难推导统计分布，就对已有结果反复重采样，比如 10000 次，得到一个 empirical distribution，再取 2.5% 和 97.5% 位置，得到 $95\%$ CI。方案使用 10000 次 hierarchical bootstrap。

---

## 第 46 章：Ablation

Ablation：拆一个模块，看是否还有效。例如 Full = A+B+C，跑 A+C，如果几乎一样，B 可能没必要。所以 ablation 是回答「为什么这个模块存在？」

---

## 第 47 章：Calibration

Calibration 在这个项目不是传统「概率校准」，这里更接近：你预测的 structural value 和真实 intervention outcome 到底有没有对应关系？也就是 **预测是否真的有 empirical meaning**。

---

## 第 48 章：Go / No-Go

科研不是方法写好了所有实验必须硬跑完。CounterStruct 把关键假设预先设成 gate。例如 $\rho < 0.2$ 说明核心假设可能失败，于是 **停止扩大实验规模**。这就是 Go/No-Go。

---

## 小结

等这 8 个概念你真正能自己解释以后，再看 CounterStruct 的公式，难度会下降非常明显：

$$
\boxed{
\text{FFN}
\rightarrow
\text{AdamW}
\rightarrow
\text{Mask}
\rightarrow
\text{2:4}
\rightarrow
\text{RigL}
\rightarrow
\text{STE}
\rightarrow
\text{Shadow Counterfactual}
\rightarrow
\text{Ridge Critic}
}
$$

继续看：[[CounterStruct_v2.1_方案导读_逐章讲解]]。
