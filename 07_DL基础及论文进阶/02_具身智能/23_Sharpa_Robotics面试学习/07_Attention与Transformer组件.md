---
title: "Attention与Transformer组件"
created: 2026-10-28
tags:
  - 面试/Sharpa
  - 具身智能
source_conversation: "6ae06c53-5b60-83e8-a1a2-78acf1bc1002"
answer_provenance: "本次重建，非原答案逐字恢复"
---

# Attention 与 Transformer 组件

> 【通用清单 S4】【本次重建答案】不是 Sharpa 专属面试原题。代码是便于理解的 PyTorch 教学实现；无 KV cache、并行通信和高性能内核优化。需安装 PyTorch 后运行。

## Q21【通用清单 S4】手写 MHA，并说明完整张量流程

### 思路分析

每个 token 通过 Q 表达“查什么”，K 表达“可按什么匹配”，V 承载被汇总的内容。先计算 query-key 相似度，再按 key 维归一化，最后对 V 加权。

### 严格推导

输入 $X\in\mathbb R^{B\times L\times D}$，H 个头，d=D/H。投影、拆头后 Q/K/V 为 [B,H,L,d]：

$$S=\frac{QK^T}{\sqrt d}+M,\qquad P=\operatorname{softmax}_{\text{key}}(S),\qquad O=PV.$$

每个 query 对应一行，softmax 沿最后一维 key 位置归一化。合头得到 [B,L,D]，再经输出投影。缩放依据是：独立、均值为 0、方差约 1 的 q、k 分量，其点积方差约 d；除以 √d 把量级保持在稳定范围。该假设解释尺度选择，不要求训练后分量严格独立。

因果 mask 对 j>i 的位置置负无穷，禁止看未来。padding mask 还需屏蔽无效 key；若一整行全为负无穷，朴素 softmax 将产生 NaN，应保证有效 key 或单独处理。下方代码只示范无 padding 的自注意力。

~~~python
import math
import torch
from torch import nn

class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, causal=False):
        b, length, dim = x.shape
        qkv = self.qkv(x).reshape(b, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if causal:
            blocked = torch.ones(length, length, device=x.device,
                                 dtype=torch.bool).triu(1)
            scores = scores.masked_fill(blocked, float('-inf'))
        probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().reshape(b, length, dim)
        return self.out(y)
~~~

【学习扩展：数值例】单头、d=1，query=1，两个 key 为 0、ln 3，value 为 2、6；分数 [0,ln 3]，权重 [1/4,3/4]，输出 $2/4+18/4=5$。改变 K 改变权重；改变 V 改变被聚合的信息。真实模型中的 Q/K/V 是学习的投影，不是手工语义标签。

Attention 分数计算时间 O(BHL²d)，显式分数存储 O(BHL²)；线性投影另有 O(BLD²)。FlashAttention 的目标之一是避免把整个注意力矩阵写入高带宽显存，不能据此说全注意力的算术复杂度已变为线性。

### 结论

$$\boxed{\text{MHA}=\text{投影拆头}\to QK^T/\sqrt d\to\text{mask/softmax}\to PV\to\text{合头投影}.}$$

## Q22【通用清单 S4】GQA 与 MHA、MQA 有什么不同？

### 思路分析

GQA 减少 KV 头数，让多个 query 头共享一组 K/V，主要影响解码的 KV cache 与带宽。参见 [GQA 原论文](https://arxiv.org/abs/2305.13245)。

### 严格推导

设 query 头数 Hq、KV 头数 Hkv，Hq 是 Hkv 的整数倍。Q=[B,Hq,Lq,d]，K/V=[B,Hkv,Lk,d]。每 Hq/Hkv 个 query 头对应同一个 KV 头。MHA：Hkv=Hq；MQA：Hkv=1；中间取值为 GQA。

~~~python

def grouped_attention(q, k, v):
    # mask 由调用方按实际 Lq、Lk、cache 位置构造；这里不含因果 mask。
    hq, hkv = q.shape[1], k.shape[1]
    if hq % hkv or k.shape != v.shape:
        raise ValueError('invalid head grouping')
    repeat = hq // hkv
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    weights = torch.softmax((q @ k.transpose(-1, -2)).float()
                            / math.sqrt(q.shape[-1]), dim=-1).to(v.dtype)
    return weights @ v
~~~

这里显式复制 K/V 只演示映射关系；高效实现应让内核共享访问，避免真的把缓存扩成 Hq 个头。每层 KV cache 约有 $2BLH_{kv}d$ 个元素（2 是 K、V），乘元素字节数和层数得到总量。相比同 Hq 的 MHA，理论缓存元素数比例 Hkv/Hq；整体延迟和显存并不必然同比下降。

### 结论

$$\boxed{\text{GQA 共享 KV 头，保留较多 query 头；节省主要体现在 KV cache 和访问带宽。}}$$

## Q23【通用清单 S4】LayerNorm 怎样写？

### 思路分析

对每个 token 的隐藏维归一化，与 batch 中其他样本无关。此处规范化最后一维 D，不沿 batch/序列维计算。

### 严格推导

$$\mu=\frac1D\sum_i x_i,\quad\sigma^2=\frac1D\sum_i(x_i-\mu)^2,\quad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.$$

方差分母为 D，而不是 D−1。

~~~python

def layer_norm(x, weight, bias, eps=1e-5):
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = ((xf - mean) ** 2).mean(dim=-1, keepdim=True)
    y = ((xf - mean) * torch.rsqrt(var + eps)).to(x.dtype)
    return y * weight + bias
~~~

可学习的 γ、β 是 D 维；数值统计以 float32 累积，再按需要转换。训练和推理都使用当前样本的统计，不维护 BatchNorm 式 running mean/variance。

### 结论

$$\boxed{\text{LayerNorm：隐藏维减均值、除标准差，再仿射变换。}}$$

## Q24【通用清单 S4】RMSNorm 与 LayerNorm 有何区别？

### 思路分析

RMSNorm 只按均方根缩放，标准形式不减均值、通常不加 bias。参见 [RMSNorm 原论文](https://arxiv.org/abs/1910.07467)。

### 严格推导

$$y_i=\gamma_i\frac{x_i}{\sqrt{D^{-1}\sum_jx_j^2+\epsilon}}.$$

~~~python

def rms_norm(x, weight, eps=1e-6):
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return y.to(x.dtype) * weight
~~~

【学习扩展】忽略 ε、取 γ=1、β=0，对 x=(1,3)，LayerNorm 输出 (−1,1)，RMSNorm 输出 (1/√5,3/√5)，后者均值不为 0。RMSNorm 不具有任意平移不变性；含 ε 时缩放不变性也不是所有尺度下严格相等。

### 结论

$$\boxed{\text{LN 去中心并缩放；RMSNorm 仅按均方根缩放。}}$$

## Q25【通用清单 S4】RoPE 为什么能体现相对位置？

### 思路分析

在二维通道对上按位置旋转 Q、K；同一平面上的旋转复合把绝对位置差变成相对角度。参见 [RoFormer 原论文](https://arxiv.org/abs/2104.09864)。

### 严格推导

令 $R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}$。位置 m 的 query 与位置 n 的 key 分别变为 $R(m\theta)q$ 和 $R(n\theta)k$，则

$$[R(m\theta)q]^T[R(n\theta)k]=q^TR((n-m)\theta)k.$$

由 $R(\alpha)^T=R(-\alpha)$ 及旋转复合律可得，相对位移 n−m 进入点积。不同通道对采用不同频率。这里采用相邻通道配对；某些实现采用前后半维配对，两种权重/布局约定不能随意混用。

~~~python

def rope_adjacent(x, positions, base=10000.0):
    # x: [B,H,L,D]，positions: [L]；D 为偶数。
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError('head dimension must be even')
    inv_freq = base ** (-torch.arange(0, dim, 2, device=x.device,
                                     dtype=torch.float32) / dim)
    angle = positions.to(x.device).float()[:, None] * inv_freq[None, :]
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    result = torch.stack((even * cos - odd * sin,
                          even * sin + odd * cos), dim=-1)
    return result.flatten(-2).to(x.dtype)
~~~

通常对 Q/K 应用 RoPE，V 不必旋转。带 cache 的解码要使用实际位置偏移；不能每次从 0 重新编码。数学表达允许计算更远位置，不等于模型已在长上下文下验证可靠。

### 结论

$$\boxed{\text{RoPE 将 Q/K 旋转，利用旋转复合使点积依赖相对位置。}}$$

## Q26【通用清单 S4】MoE 怎样路由？总参数与激活参数有什么区别？

### 思路分析

router 为每个 token 选择少数专家。模型可拥有大量总参数，但每个 token 只经过其中部分专家。参见 [稀疏门控 MoE 原论文](https://arxiv.org/abs/1701.06538)。

### 严格推导

设专家 $E_1,\ldots,E_M$，路由得分 $s=W_rx$，选择集合 $I=\operatorname{TopK}(s,k)$，一种常见归一化方式为

$$g_i=\frac{e^{s_i}}{\sum_{j\in I}e^{s_j}},\quad y=\sum_{i\in I}g_i E_i(x).$$

这是在选中 logits 上归一化的约定，其他实现可能先全体 softmax 再 top-k，需明确。下方伪代码保留 token 派发的含义：

~~~text
对所有 token 计算路由 logits
选择每个 token 的 k 个专家及权重
按专家汇集 token（可涉及跨卡通信）
每个专家批量计算收到的 token
将专家输出按权重合并回原 token 位置
训练时按具体方法处理负载均衡、容量和路由稳定性
~~~

总参数包含所有专家；激活参数包含每 token 经过的专家和共享层。不能说“只激活 2 个专家，因此只需存 2 个专家”：分布式训练/推理仍需管理全部模型权重。专家失衡会形成通信和算力热点；容量限制可能丢 token 或重路由，必须说清实现。负载均衡辅助损失是常见选择，不能说所有 MoE 都必须采用同一种辅助损失。

### 结论

$$\boxed{\text{MoE 用稀疏条件计算增加容量，收益受路由、负载与跨卡通信影响。}}$$


---
返回 [[00_Sharpa面试学习导航]] · 来源 [[01_题目来源与原图]]
