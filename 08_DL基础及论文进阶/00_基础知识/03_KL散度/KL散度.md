---
title: KL散度
tags:
  - 基础知识
  - 深度学习
  - KL散度
  - 信息论
  - RLHF
  - 大模型
source: 小红书-1分钟图解：KL散度 (69f16480000000001e00ed81)
created: 2026-07-02
---

# KL散度（Kullback-Leibler Divergence）

> KL散度是信息论中的核心概念，用于衡量两个概率分布之间的差异。在大模型时代，KL散度是 RLHF（基于人类反馈的强化学习）中的关键约束项，防止模型在微调时偏离原始策略过远。

## 一、KL散度概览

![[KL散度_01_概览.jpg]]

**核心直觉：**

- **KL(P‖Q)** 衡量当我们用分布 Q 来近似真实分布 P 时，所损失的信息量。
- KL 散度 ≥ 0，当且仅当 P = Q 时 KL = 0。
- KL 散度不具有对称性：KL(P‖Q) ≠ KL(Q‖P)，它不是真正的"距离"度量。

**为什么 KL 散度在大模型时代如此重要？**

RLHF 训练的优化目标中，除了最大化奖励模型给出的分数，还有一个 KL 惩罚项：

$$\max_{\pi} \ \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} [r(x, y) - \beta \cdot \text{KL}(\pi(\cdot|x) \ \| \ \pi_{\text{ref}}(\cdot|x))]$$

$\beta$ 控制"靠近原始策略"与"追求高奖励"之间的权衡。

## 二、数学定义

![[KL散度_02_定义.jpg]]

对于离散分布：

$$\text{KL}(P \| Q) = \sum_{x} P(x) \cdot \log \frac{P(x)}{Q(x)}$$

对于连续分布：

$$\text{KL}(P \| Q) = \int P(x) \cdot \log \frac{P(x)}{Q(x)} \, dx$$

**分解理解：**

- $P(x)$：权重项，P(x) 大的地方 KL 散度更关注
- $\log \frac{P(x)}{Q(x)}$：差异项，衡量每个点上 P 和 Q 的倍数差异

**含义：** 用 Q 来编码（或近似）来自 P 的数据时，额外需要的平均编码长度。

```python
import torch
import torch.nn.functional as F

def kl_divergence(p_logits, q_logits, temperature=1.0):
    """
    计算两个分布之间的 KL 散度
    
    Args:
        p_logits: 真实分布 P 的 logits
        q_logits: 近似分布 Q 的 logits
        temperature: 温度参数，控制分布的平滑程度
    """
    p = F.softmax(p_logits / temperature, dim=-1)
    q_log_probs = F.log_softmax(q_logits / temperature, dim=-1)
    
    # KL(P||Q) = sum(P(x) * (log P(x) - log Q(x)))
    #        = sum(P(x) * log P(x)) - sum(P(x) * log Q(x))
    kl = F.kl_div(q_log_probs, p, reduction='batchmean', log_target=False)
    return kl

# 示例
p_logits = torch.tensor([[2.0, 1.0, 0.1]])
q_logits = torch.tensor([[0.1, 1.0, 2.0]])

kl_pq = kl_divergence(p_logits, q_logits)  # KL(P||Q)
kl_qp = kl_divergence(q_logits, p_logits)  # KL(Q||P)

print(f"KL(P||Q) = {kl_pq.item():.4f}")
print(f"KL(Q||P) = {kl_qp.item():.4f}")
print(f"不对称性验证: KL(P||Q) != KL(Q||P) ✓")
```

## 三、核心性质与 RLHF 中的角色

![[KL散度_03_性质与RLHF.jpg]]

### 3.1 KL散度的关键性质

| 性质 | 说明 |
|------|------|
| **非负性** | KL(P‖Q) ≥ 0，当且仅当 P = Q 时取 0 |
| **不对称性** | KL(P‖Q) ≠ KL(Q‖P)（因此不是真正的"距离"） |
| **与熵的关系** | KL(P‖Q) = H(P, Q) - H(P)（交叉熵 - 自身熵） |
| **链式法则** | KL 散度满足联合分布的链式分解 |

### 3.2 KL 散度在 RLHF 中的双重角色

**正向 KL（KL(π ‖ π_ref)）：**
- 倾向于让新策略在参考策略概率高的地方也输出高概率
- 模式寻求（mode-seeking）：覆盖参考策略的主要模式

**反向 KL（KL(π_ref ‖ π)）：**
- 倾向于让新策略避免在参考策略概率低的地方输出高概率
- 均值寻求（mean-seeking）：避免生成参考策略不会生成的输出

在 RLHF 实践中，通常使用 **KL(π ‖ π_ref)**（正向 KL），因为：
1. 它是 PPO 训练中最直接的实现方式
2. 通过 token 级别的 KL 惩罚实现，易于逐 token 约束

### 3.3 在 PPO 中的使用

```python
# PPO 训练中 KL 散度的典型用法
def compute_ppo_loss_with_kl(
    current_logits,    # 当前策略
    old_logits,        # 旧策略（用于重要性采样）
    ref_logits,        # 参考策略（SFT 模型）
    advantages,        # 优势函数
    actions,           # 采样的动作
    clip_ratio=0.2,
    kl_coef=0.02       # KL 惩罚系数 β
):
    """
    PPO + KL 约束的简化实现
    
    KL 惩罚项的作用：
    1. 防止策略更新幅度过大
    2. 保持生成文本的流畅性和合理性
    3. 避免奖励黑客（reward hacking）
    """
    # 计算新旧策略的概率比
    current_probs = F.softmax(current_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)
    
    action_probs_current = current_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    action_probs_old = old_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    
    ratio = action_probs_current / (action_probs_old + 1e-8)
    
    # 标准 PPO clipped loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    ppo_loss = -torch.min(surr1, surr2).mean()
    
    # KL 惩罚项：约束当前策略不偏离参考策略太远
    ref_probs = F.softmax(ref_logits, dim=-1)
    kl_penalty = F.kl_div(
        F.log_softmax(current_logits, dim=-1),
        ref_probs,
        reduction='batchmean'
    )
    
    total_loss = ppo_loss + kl_coef * kl_penalty
    return total_loss, kl_penalty.item()
```

## 四、图解计算与实际应用

![[KL散度_04_计算应用.jpg]]

### 4.1 四种典型场景分析

| 场景 | P(x) | Q(x) | P(x)/Q(x) | log(P/Q) | KL含义 |
|------|------|------|-----------|----------|--------|
| **场景一**：P=Q | 两者相等 | → | 1 | 0 | KL=0，无信息损失 |
| **场景二**：Q范围更大 | 集中 | 分散 | < 1 | 负值 | 部分权重处 Q 过散 |
| **场景三**：Q范围更小 | 分散 | 集中 | > 1 | 正值 | 部分权重处 Q 过窄 |
| **场景四**：Q(x)=0, P(x)>0 | 有概率 | 0 | ∞ | +∞ | KL → ∞，严重惩罚 |

### 4.2 权重项的直观感受

以二分类为例（P 为真实标签分布，Q 为预测分布）：

```python
import torch
import torch.nn.functional as F

# === 情况1: P 有确定偏好，Q 也很确定（且一致）===
P1 = torch.tensor([0.9, 0.1])
Q1 = torch.tensor([0.8, 0.2])
kl1 = (P1 * torch.log(P1 / Q1)).sum()
print(f"P={P1.tolist()}, Q={Q1.tolist()}, KL={kl1:.4f}")  # 较小

# === 情况2: P 有确定偏好，Q 不匹配 ===
P2 = torch.tensor([0.9, 0.1])
Q2 = torch.tensor([0.2, 0.8])
kl2 = (P2 * torch.log(P2 / Q2)).sum()
print(f"P={P2.tolist()}, Q={Q2.tolist()}, KL={kl2:.4f}")  # 很大！

# === 情况3: P 比较均匀，Q 也均匀（即使不完全匹配）===
P3 = torch.tensor([0.55, 0.45])
Q3 = torch.tensor([0.45, 0.55])
kl3 = (P3 * torch.log(P3 / Q3)).sum()
print(f"P={P3.tolist()}, Q={Q3.tolist()}, KL={kl3:.4f}")  # 较小

# === 情况4: Q 为 0 的地方 P 不为 0 → KL 爆炸 ===
P4 = torch.tensor([0.6, 0.4])
Q4 = torch.tensor([0.0, 1.0])
kl4 = (P4 * torch.log(P4 / (Q4 + 1e-10))).sum()
print(f"P={P4.tolist()}, Q={Q4.tolist()}, KL={kl4:.4f} (近似无穷)")
```

**关键直觉：**
- 权重 P(x) 决定了 KL 散度关注的区域
- $\log(P/Q)$ 项决定了差异的方向和幅度
- 当 P(x) 大而 Q(x) 小时，KL 受到最大惩罚
- 当 P(x) 小的地方，即使 Q(x) 差异大，对 KL 影响也很小

### 4.3 大模型训练中的实际考量

**RLHF 中的 KL 系数选择：**
- $\beta$ 太大 → 模型几乎不变，学习效果差
- $\beta$ 太小 → 模型过度追求奖励，可能"奖励黑客"（输出无意义的重复或怪异文本）
- 常见做法：动态调整 $\beta$，保持 KL 在一个目标区间内

**DPO 中的隐式 KL（无需显式计算）：**

DPO 回避了显式的奖励建模和 PPO 训练，但最优策略仍等价于 KL 约束下的奖励最大化：

$$\pi^* = \arg\max_{\pi} \ \mathbb{E}_{x \sim D, y \sim \pi} [r_{\phi}(x, y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$$

DPO 的优雅之处在于：通过构造 pairwise 偏好数据，直接学习最优策略，而不需要训练单独的奖励模型或做 PPO。

## 来源

- 图片来自小红书笔记：[1分钟图解：KL散度](https://www.xiaohongshu.com/explore/69f16480000000001e00ed81)
- 话题标签：`#大模型` `#强化学习算法` `#RLHF`
