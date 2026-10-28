---
title: "PPO_GRPO与交叉熵"
created: 2026-10-28
tags:
  - 面试/Sharpa
  - 具身智能
source_conversation: "6ae06c53-5b60-83e8-a1a2-78acf1bc1002"
answer_provenance: "本次重建，非原答案逐字恢复"
---

# PPO、GRPO 与交叉熵

> 【通用清单 S4】【本次重建答案】这里把原帖的主题扩为学习题，不声称截图给过以下完整问法或代码。

## Q27【通用清单 S4】交叉熵是什么，为什么不能先 softmax 再交给 logits 接口？

### 思路分析

交叉熵度量目标分布下的负对数预测概率。对 one-hot 标签等于正确类别的负对数概率；实现应直接使用 logits 的稳定 log-sum-exp。

### 严格推导

目标分布 p、预测分布 q：

$$H(p,q)=-\sum_i p_i\log q_i.$$

若标签为类别 y、logits 为 z，则

$$\ell=-\log\frac{e^{z_y}}{\sum_je^{z_j}}=-z_y+\log\sum_je^{z_j}.$$

令 $m=\max_jz_j$，稳定实现：

$$\log\sum_je^{z_j}=m+\log\sum_je^{z_j-m}.$$

对 logits 求导：$\partial\ell/\partial z_j=\operatorname{softmax}(z)_j-\mathbf1[j=y]$。先 softmax 再交给期待 logits 的交叉熵，会被再次当 logits 归一化，改变目标。

~~~python

def cross_entropy_logits(logits, target):
    # logits: [N,C]，target: [N]，不含 ignore_index 的简化实现。
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -log_probs.gather(-1, target[:, None]).mean()
~~~

语言模型必须正确右移：位置 t 的 logits 预测 token t+1；padding、提示词或无监督区域按任务 mask。损失可除以有效 token 数，但若所有 mask 都为 0，要先跳过该批次，不能把 NaN 悄悄吞掉。二分类单 logit 通常用 BCEWithLogitsLoss；多标签和互斥多分类不是同一问题。

【学习扩展】z=(0,ln 3)、y=1，则 q=(1/4,3/4)，损失 −ln(3/4)≈0.28768。若 q_y 越接近 1，损失越接近 0。

### 结论

$$\boxed{\text{交叉熵从 logits 稳定计算，先明确类别、维度、标签位移与 mask。}}$$

## Q28【通用清单 S4】PPO 的目标与 clipping 是什么？

### 思路分析

用旧策略采样轨迹，再限制新策略对采样动作概率的过度改变。PPO-Clip 是策略目标的裁剪形式，参见 [PPO 原论文](https://arxiv.org/abs/1707.06347)。

### 严格推导

令优势 $\hat A_t$ 固定、概率比为

$$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
=\exp(\log\pi_\theta-\log\pi_{old}).$$

最大化：

$$L^{clip}=\mathbb E_t\left[\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].$$

当优势为正，增加该动作概率有利，但超过上边界后不再从裁剪项获得额外收益；当优势为负，减小概率有利，但低于下边界后该方向也被截断。并非把 r 本身强制限制在区间内，PPO clipping 也不保证真实 KL 一定小于某阈值。

~~~python

def ppo_actor_loss(new_logp, old_logp, advantage, clip_eps=0.2):
    ratio = (new_logp - old_logp.detach()).exp()
    advantage = advantage.detach()
    first = ratio * advantage
    second = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
    return -torch.minimum(first, second).mean()
~~~

此代码只含 actor 裁剪损失。完整实现还需价值函数损失、熵项（符号随最小化/最大化约定）、有效步 mask、梯度裁剪、采样和更新循环。

优势可用 GAE：

$$\delta_t=r_t^{env}+\gamma m_tV(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.$$

此处用 $r_t^{env}$ 表示环境奖励，避免与概率比 r_t 混淆。真正终止时 bootstrap mask $m_t=0$；时间截断通常仍应根据终止观测 bootstrap，递推连续性 mask $c_t$ 则必须阻断跨 episode 传播。实现中要区分 terminated/truncated，不能直接拿自动 reset 后的观测当终止观测。

【学习扩展】$\epsilon=0.2$、A=1、r=1.5，目标 min(1.5,1.2)=1.2；A=−1、r=0.5，目标 min(−0.5,−0.8)=−0.8。两例展示裁剪对“过度有利更新”的抑制，而不是所有方向都不更新。

### 结论

$$\boxed{\text{PPO-Clip 对优势加权概率比取保守裁剪目标，old policy 和优势在本轮更新中固定。}}$$

## Q29【通用清单 S4】GRPO 怎样估计优势，和 PPO 有何关系？

### 思路分析

同一 prompt 采样一组回答，用组内相对奖励形成优势，减少对独立 critic 的依赖。这里讲 DeepSeekMath 的 outcome-supervision 基本形式，参见 [原论文第 4.1 节](https://arxiv.org/html/2402.03300v3#S4.SS1)。不同后续变体的归一化、KL 与长度处理不能混称为唯一标准实现。

### 严格推导

对 prompt q，从旧策略采样 G 个回答 $o_i$，奖励 $R_i$。组内：

$$\bar R=\frac1G\sum_iR_i,\quad s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\quad
A_i=\frac{R_i-\bar R}{s_R+\varepsilon}.$$

结果奖励场景可将 A_i 用于该回答的各有效 token。对 token t 定义新旧概率比 $\rho_{i,t}$，一种基本目标写作：

$$J=\frac1G\sum_i\frac1{T_i}\sum_{t=1}^{T_i}
\left[\min\{\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i\}
-\beta d_{KL,i,t}\right].$$

这明确采用“先回答内平均、再组内平均”的约定，不能不加说明改成全组 token 平均。KL 参考策略与 rollout 的旧策略作用不同。常见 token 级非负 KL 估计形式：令 $u=\pi_{ref}(o_{i,t}\mid h)/\pi_\theta(o_{i,t}\mid h)$，则 $d_{KL}=u-\log u-1$；其采样分布、重要性权重与梯度实现须按具体算法核对，不能仅凭逐 token 数值就声称精确计算了全词表 KL。

~~~python

def group_advantages(rewards, eps=1e-6):
    # [batch_of_prompts, group_size]；同一行必须对应同一个 prompt。
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    return ((rewards - mean) / (std + eps)).detach()
~~~

【学习扩展】奖励 [0,0,1,1] 的均值 0.5、总体标准差 0.5，忽略 ε 后优势 [−1,−1,1,1]。若组内奖励全相同，则该组相对奖励信号为 0；G=1 也无法产生组内相对信号。应监控零方差组、有效回答比例、奖励投机和长度偏差。

### 结论

$$\boxed{\text{GRPO 用同题多回答的组内奖励构造优势，保留 PPO 式裁剪思想；并非无需任何评价信号。}}$$

## 【学习扩展】RL 与机器人项目怎样联系？

PPO 可用于交互式控制策略；语言模型式 GRPO 需有可比较的采样组和可用奖励，不能仅因“用了 Agent”就套 GRPO。若用世界模型生成 rollout，必须单独评估模型偏差和真实环境迁移。训练成本包括采样、奖励计算、策略更新和验证，不能只报反向传播时间。


---
返回 [[00_Sharpa面试学习导航]] · 来源 [[01_题目来源与原图]]
