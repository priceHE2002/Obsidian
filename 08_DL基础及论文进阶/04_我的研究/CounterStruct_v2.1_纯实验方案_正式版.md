# CounterStruct v2.1 纯实验方案（正式版）

**论文标题：**

> **CounterStruct: Horizon-Calibrated Structural Action Values for Fixed-Capacity Continual Learning**

> 一句话主线：在 exact-2:4 topology graph 上，用少量真实 shadow interventions 学习 integration-horizon structural action value，再把 task-localized structural preference 压缩进固定大小的 structural-state memory，在固定 active connectivity capacity 下做保守的 topology adaptation。

---

## 目录

本方案共 70 节（第 0–69 章），按十三大部分组织：

1. **第一部分 · 定位与研究边界**（第 0–3 章）：版本定位 / 核心研究问题 / 贡献边界 / 与近邻方法的区别
2. **第二部分 · 模型与结构定义**（第 4–10 章）：模型范围 / 2:4 Topology Graph / 初始拓扑 / 状态 / Replay / 训练 / Mutation Schedule
3. **第三部分 · 结构动作的测量与监督**（第 11–16 章）：Probe / Cheap Prior / 特征 / Shadow Bundles / 计算优化 / Horizon Label
4. **第四部分 · Horizon Critic 与结构状态记忆**（第 17–19 章）：Global+Task-Local Critic / Memory R / Task-End 固化
5. **第五部分 · 决策与执行**（第 20–26 章）：历史损害 / 最终 Utility / 预筛分 / 组搜索 / Edge 预算 / 执行 / 伪代码
6. **第六部分 · Benchmark 与实验设计**（第 27–40 章）：四套 Benchmark / 实验 A–F / 基线 / 评估矩阵 / 通用能力
7. **第七部分 · 核心机制实验**（第 41–46 章）：Temporal Calibration / Multi-Horizon / 聚合信用 / 记忆校准 / Macro Bridge
8. **第八部分 · Ablations**（第 47–52 章）
9. **第九部分 · 行为指标与资源账本**（第 53–57 章）：行为指标 / 记忆账本 / Replay 对照 / 统计 / 效应阈值
10. **第十部分 · Go/No-Go 门禁**（第 58 章）
11. **第十一部分 · 执行协议**（第 59–61 章）：开发协议 / 冻结协议 / 执行优先级
12. **第十二部分 · 可复现与硬件**（第 62–64 章）：运行工件 / Mutation 日志 / 硬件验证
13. **第十三部分 · 论文主图与成败定义**（第 65–69 章）：五张主图 / 成功模式 / 失败解读 / Checklist / 项目定义

---

# 第一部分 · 定位与研究边界

## 0. 版本定位

v2.1 保留 v2.0 的主创新方向，但针对外部严格审阅暴露出的四个风险做方法级降险：

1. aggregate bundle supervision 的 credit assignment 可能被 action interaction 污染；
2. 单一跨任务 critic 会让「当前任务监督」与「历史记忆写入」之间出现解释歧义；
3. 每 event 4 条 bundle equation 对低维 critic 的早期冷启动仍偏弱；
4. six-state complete transition 中的 two-swap 并非主创新，却会引入额外结构风险。

因此 v2.1 的原则是：

$$
\boxed{
\text{保留 horizon action-value 主线}
+
\text{减少非必要动作复杂度}
+
\text{强化 task-local supervision}
+
\text{直接审计 interaction}
}
$$

v1.0 的 one-step AdamW + Fisher-style historical memory 继续作为强 predecessor baseline：

> **LFS — Local-Fisher Swap**

v2.1 不把新增的统计审计包装成新的「模块」；论文主创新仍然只有：

> **用稀疏真实反事实 intervention 学习 integration-horizon structural action value，并把 task-localized structural preference 压缩进固定大小的 structural-state memory。**

---

## 1. 核心研究问题

CounterStruct v2.1 围绕四个可证伪问题展开：

1. **Exact-2:4 rewiring 是否应该在合法 topology-state graph 上作为结构动作建模，而不是独立 prune/grow saliency？**
2. **结构动作在 newborn connection 经过 integration horizon 后的价值，能否由少量真实 counterfactual shadow interventions 学到，并优于 one-step saliency？**
3. **这种 aggregate intervention supervision 是否在局部近似可加，即 bundle-level label 是否足以学习可推广的 per-action ranking？**
4. **每个任务在其仍为 current task 时学到的 structural preference，能否压缩进固定大小的 historical state memory，并在未来预测 past-task structural damage？**

主证据链：

$$
\boxed{
\text{2:4 topology graph}
\rightarrow
\text{shadow-intervention horizon critic}
\rightarrow
\text{interaction audit}
\rightarrow
\text{task-localized state consolidation}
\rightarrow
\text{continual behavior}
}
$$

---

## 2. 核心贡献边界

**允许的主贡献：**

1. 将 exact contiguous 2:4 group 表示为 6 个合法 structural states，并在其 one-edge-replacement 邻接图上定义原子结构动作；
2. 用少量真实 keep-vs-transition shadow integration rollouts 在线监督轻量 integration-horizon critic；
3. 使用 **global transferable prior + task-local residual critic**，使当前任务的结构价值在写入 historical memory 前由该任务自己的 shadow labels 校准；
4. 用 task-count-independent structural-state memory 记录「各历史任务在其作为 current task 时估计的 state-relative cost」的任务平均；
5. 用 temporal calibration、micro-bundle interaction audit 与 macro-horizon bridge 分别检验 horizon validity、aggregate credit validity 与最终 continual-behavior 转化。

**禁止声明：**

- 首次提出 dynamic sparsity；
- 首次提出 look-ahead pruning；
- 首次提出 continual-learning regularization；
- 首次提出 Fisher / EWC memory compression；
- six-state / Johnson graph 本身是 novelty；
- sparse-training speedup；
- 单凭 8B 实验宣称 scaling law；
- horizon critic 是 exact future accuracy predictor；
- structural-state memory 等价于真实历史 loss landscape；
- bundle additivity 是数学保证。

最终 novelty sweep 若发现已有工作实现相同的：

$$
\boxed{
\text{N:M topology graph}
+
\text{real shadow-intervention horizon critic}
+
\text{task-local continual structural-state memory}
}
$$

则必须重新收窄 claim。

---

## 3. 与最近邻方法的明确区别

### 3.1 SRigL / RigL

SRigL 类方法核心仍是：

- prune：weight saliency；
- grow：gradient saliency；
- 周期性 topology evolution。

CounterStruct 的区别必须由实验直接证明：

$$
\boxed{
\text{independent coordinate saliency}
\neq
\text{structural-state action value}
}
$$

### 3.2 LookAhead / RL Compression / Shadow Evaluation

ICLR 2020 的 **Lookahead: A Far-Sighted Alternative of Magnitude-based Pruning** 主要将单层 magnitude distortion 推广到 multi-layer distortion；它不是在线 shadow rollout action critic。

更广义地，AMC 等 RL compression 工作会学习 layer-level compression policy / value function，但其 action 通常是 layer sparsity ratio，reward 是完整压缩模型的 validation performance，不是 continual DST 中的 per-topology-edge online intervention supervision。

因此 CounterStruct 不宣称：

- 首次「看未来」；
- 首次学习 pruning/compression action value；
- 首次使用 critic。

真正需要实验支持的区别是：

> 在固定 N:M topology graph 上，使用少量真实 keep-vs-transition **weight-level structural interventions** 在线学习 integration-horizon ranking，并把该结构价值继续用于 history-example-free continual topology adaptation。

### 3.2A SMET / Newborn Cold-Start

2026 年 SMET 指出 LLM dynamic sparse training 中 newly regrown parameters 存在 Adam cold-start / optimization-instability 问题，并通过 optimizer warm-up 与 density-aware LR scaling 处理。

CounterStruct 与其关系：

- SMET 主要回答 **newborn parameter 如何稳定优化**；
- CounterStruct 主要回答 **哪个 structural transition 值得发生**；
- CounterStruct 的 H=8 shadow rollout 会自然观察 newborn integration dynamics，但不能据此声称首次发现 cold-start。

因此若实现中出现明显 newborn instability，应把 SMET-style warm-up 作为独立 optimizer-control ablation，而不是把稳定化效果误归因于 horizon critic。

### 3.3 EWC / Fisher / MAS / SI

v2.1 主方法不维护 per-coordinate Fisher precision。

v1.0 的 $P,H$ 方案作为 LFS baseline 保留，用于回答：

> 新效果是否只是 Fisher-style regularization 与 DST 的组合？

### 3.4 OSFT / PaRSP

OSFT / PaRSP 主要通过 subspace / region protection 减少历史任务干扰。

CounterStruct 的差异不是「保护参数」，而是：

> 在固定 active connectivity capacity 下，学习 topology state transition 的 action value。

### 3.5 Bi-Level DST / 2:4 Mask Optimization

已有工作已经将 dynamic sparse training 表述为 bi-level optimization；2:4 mask 的离散/局部优化本身也不是空白。

因此 CounterStruct 禁止把以下内容单独列为 novelty：

- 「DST 是 weight/topology 双层优化」；
- 「2:4 group 只有有限合法 masks」；
- 「穷举 2:4 legal masks」。

六状态表示的作用是提供 **action-value learning 与 structural-state memory 的天然离散载体**，而不是单独构成主要贡献。

---

# 第二部分 · 模型与结构定义

## 4. 模型与结构范围

### 4.1 Primary Backbone

$$
\boxed{\text{Qwen3-1.7B}}
$$

关键配置：

$$
d_{model}=2048,\qquad d_{ffn}=6144,\qquad L=28.
$$

修改最后 8 层：

```text
Layer 0–19   Frozen
Layer 20–27  Attention   Frozen
             gate_proj   Frozen
             up_proj     CounterStruct
             down_proj   CounterStruct
```

candidate coordinates：

$$
8\times2\times2048\times6144
=
\boxed{201,326,592}.
$$

active coordinates：

$$
\boxed{100,663,296}.
$$

### 4.2 Scale Backbone

$$
\boxed{\text{Qwen3-8B}}
$$

$$
d_{model}=4096,\qquad d_{ffn}=12288,\qquad L=36.
$$

只修改最后 2 层 up/down projection：

$$
2\times2\times4096\times12288
=
\boxed{201,326,592}
$$

candidate coordinates，与 1.7B primary structural search space 完全相同。

8B 不参与 CounterStruct hyperparameter development。

### 4.3 Optional Backbone

Llama-3.2-1B-Instruct 只作为资源允许时的 cross-family secondary replication，不再作为 mandatory full-comparison stream。

---

## 5. Exact Contiguous 2:4 Topology Graph

对每个 contiguous 4-group：

$$
q=[w_1,w_2,w_3,w_4],
\qquad
\sum_{i\in q}M_i=2.
$$

定义 6 个合法 topology states：

$$
\boxed{
\mathcal S_q=
\{12,13,14,23,24,34\}.
}
$$

v2.1 不再把 6 个 states 视为 complete action graph。

定义 topology graph：

$$
\boxed{
\mathcal G_{2:4}=J(4,2)
}
$$

即 Johnson graph：若两个 state 恰好共享一个 active coordinate，则相邻。

因此原子 structural action 固定为：

$$
\boxed{
a=(s_q\rightarrow s'_q),
\qquad
|s_q\cap s'_q|=1.
}
$$

每个 state 恰有 4 个 one-swap neighbors。

每个原子动作：

$$
\boxed{
c(a)=1
}
$$

即只替换一条 active edge。

互补 state（例如 $12\rightarrow34$）需要两次独立 one-swap transition，中间必须经过真实 integration window；不允许把 two-swap 作为单个原子 action。

这样做的目的不是声称 Johnson graph 新颖，而是：

1. 去掉不必要的 two-swap 赌注；
2. 使 aggregate shadow bundle 中每个 action 的 turnover cost 完全一致；
3. 将论文新颖性集中在 **horizon action-value learning**，而不是「可一次改更多边」。

---

## 6. Initial Topology

所有 structural methods 共用：

$$
\boxed{M_0=\text{top-2-of-4 magnitude mask}}.
$$

规则：

1. 每 4-group 保留 pretrained magnitude 最大的两个 weights；
2. dormant tensor values 置 0；
3. active cooldown 初始化 $C=2$；
4. dormant cooldown 初始化 $C=0$。

---

## 7. Persistent State

CounterStruct v2.1 主方法维护：

$$
\boxed{
W,M,C,m,v,R,
A_G,b_G,
A_t,b_t
}
$$

其中：

- $W$：dense storage target weights；
- $M$：exact-2:4 mask；
- $C$：cooldown；
- $m,v$：AdamW moments；
- $R$：structural-state historical memory；
- $A_G,b_G$：跨任务 global horizon-critic sufficient statistics；
- $A_t,b_t$：当前任务 task-local residual critic sufficient statistics。

主方法不维护 $P,H$。

critic 参数维度固定为低维常数，与模型参数量和 task 数均无关。

每个新 task：

$$
A_t\leftarrow\lambda_t I,
\qquad
b_t\leftarrow0.
$$

task-local state 在 task 结束后合并进 global sufficient statistics，然后清空。

---

## 8. Replay Policy

Primary setting：

$$
\boxed{\text{No Historical Samples}}
$$

学习 $T_{t+1}$ 时不访问历史 task training examples。

不保存：

- historical raw samples；
- historical logits；
- per-task checkpoints；
- per-task Fisher tensors；
- historical gradients。

未来任务只携带固定-size model/optimizer/structural state。

论文不得把该设置包装成「比 replay 更省内存」。

正确叙事为：

> history-example-free and task-count-independent persistent state.

Replay baseline 单独报告，并额外加入 memory-matched replay secondary comparison。

---

## 9. Normal Training

有效权重：

$$
W_{eff}=M\odot W.
$$

训练阶段仍使用 dense masked Linear。

Backward 后：

$$
\nabla W\leftarrow M\odot\nabla W.
$$

Dormant invariant：

$$
M_i=0\Rightarrow w_i=m_i=v_i=0.
$$

---

## 10. Mutation Schedule

每 task 固定 4 次 mutation：

$$
\boxed{20\%,40\%,60\%,80\%}.
$$

最后 20% 作为 final newborn integration window。

Cooldown：

$$
C_i\in\{0,1,2\}.
$$

只有所有待 prune coordinates 都满足 $C=2$ 的 structural transition 才合法。

# 第三部分 · 结构动作的测量与监督

## 11. Current Dense Structural Probe

每 mutation event：

$$
|B_{probe}|=32.
$$

使用 STE：

$$
W_{STE}
=
W+\operatorname{stopgrad}(M\odot W-W).
$$

Forward 仍使用 sparse effective weights，但 backward 获取 active + dormant signed gradients：

$$
g_i=\frac{\partial L_{cur}}{\partial w_i}.
$$

Probe：

- current-task only；
- 不进行真实 optimizer step；
- 不写 optimizer state；
- probe batch 不作为紧随其后的 real training batch。

---

## 12. Cheap Analytic Action Prior

v2.1 保留 one-step AdamW counterfactual，但它只作为 critic feature 与 shadow stratification prior。

对于 topology-graph 邻接 transition：

$$
a:s\rightarrow s',
$$

固定：

$$
P(a)=s\setminus s'=\{p\},
\qquad
G(a)=s'\setminus s=\{g\}.
$$

对 active prune coordinate $p$ 计算 keep-world hypothetical AdamW increment：

$$
\delta_p^{keep}.
$$

对 newborn $g$ 从：

$$
w_g=m_g=v_g=0
$$

计算：

$$
\delta_g^{new}.
$$

定义：

$$
\boxed{
B^{(1)}(a)
=
g_p(w_p+\delta_p^{keep})
-
g_g\delta_g^{new}.
}
$$

只允许解释为：

> optimizer-aware one-step action prior.

它既不是最终 utility，也不承担 long-horizon claim。

---

## 13. Structural Action Feature Vector

对每个 legal one-swap action 构造固定：

$$
\boxed{\phi(a)\in\mathbb R^{10}}.
$$

连续 action features：

1. $B^{(1)}(a)$；
2. $|w_p|$；
3. $|g_p|$；
4. $|g_g|$；
5. $\log\frac{|g_g|+\epsilon}{|g_p|+\epsilon}$；
6. $|\delta_p^{keep}|$；
7. $|\delta_g^{new}|$。

结构/context features：

8. normalized layer depth；
9. matrix type（`up_proj` / `down_proj`）；
10. task progress $r\in\{0.2,0.4,0.6,0.8\}$。

删除 v2.0 中的 minimum prune cooldown feature，因为 legal action 已要求 prune coordinate mature，该 feature 在候选集中近似常数。

### 13.1 跨事件稳定的 feature 语义

v2.1 不再使用会改变原始数值坐标系的 event-level median/MAD z-score 直接做跨事件 pooling。

对每个连续 feature $x$，在当前 mutation event 的全部 legal actions 中计算 mid-rank percentile：

$$
F_e(x(a))
=
\frac{\operatorname{midrank}_e(x(a))-0.5}{N_e}.
$$

使用：

$$
\boxed{
\tilde x_e(a)=2F_e(x(a))-1\in[-1,1].
}
$$

因此跨 event 的语义固定为：

> 「该 action 在当前候选集合中的相对位置」。

layer depth / matrix type / task progress 使用固定编码，不做 event-dependent rescaling。

该设计主动把 critic 定位为 **within-event action ranking model**，而不是跨 task 的绝对 loss predictor。

---

## 14. Counterfactual Shadow Integration Bundles

这是 v2.1 的核心在线监督机制。

每个 mutation event，在 real topology mutation 前构造：

$$
\boxed{J_{shadow}=8}
$$

个 shadow bundles。

每 bundle：

$$
\boxed{K_{shadow}=64}
$$

个 mutually group-disjoint one-swap actions。

### 14.1 Bundle stratification

按 preliminary score：

$$
S_{prior}(a)
$$

划分为 8 个 octile strata。

每个 stratum 构造一个 bundle，并尽量平衡：

- layer；
- `up_proj/down_proj`；
- group location。

禁止不同 action 共享 exact-2:4 group。

### 14.2 Branches

所有 branches 从完全相同 checkpoint 开始。

#### Keep Branch

不修改 topology。

#### Transition Branch $j$

执行 bundle $B_j$ 中全部 one-swap transitions。

所有 branches 共享：

- current-task shadow integration batches；
- optimizer global step；
- scheduler state；
- RNG state；
- cached frozen-prefix hidden states；
- held-out evaluation batch。

Shadow integration horizon：

$$
\boxed{H_{shadow}=8}.
$$

Shadow steps 只更新 branch clone，不写真实 model / optimizer。

因此一条 TRACE stream 最多产生：

$$
8\text{ tasks}\times4\text{ events/task}\times8
=
\boxed{256}
$$

条 aggregate intervention equations，而不是 v2.0 的 128 条。

第一个 task 在四个 event 后可得到 32 条 task-local equations，对 10 维 residual critic 提供更稳的冷启动监督。

---

## 15. Shadow Rollout Compute Optimization

由于 target region 之前的层全部 frozen：

1. shadow examples 的 frozen prefix 只执行一次；
2. cache target-region input hidden states；
3. keep / 8 transition branches 复用同一 prefix cache；
4. 9 个 branches 顺序执行，不同时保留 9 份完整模型；
5. 完成当前 event 后立即释放 branch-specific optimizer/model buffers。

必须真实报告：

- normal-step latency；
- one shadow step latency；
- one full mutation event overhead；
- full stream wall-clock；
- peak HBM。

目标：

$$
\boxed{\text{shadow overhead}\le30\%}
$$

相对 LFS full stream。

若实现后：

$$
30\%<\text{overhead}\le40\%
$$

仍可继续实验，但论文不得使用「low-overhead」措辞。

若：

$$
\boxed{\text{overhead}>40\%}
$$

则进入 engineering no-go：先优化 cached-prefix / branch-state handling，再启动大规模 seeds。

---

## 16. Horizon Label

在相同 held-out current-task batch 上：

$$
L_K^{(H)}
=
\text{keep branch loss after }H_{shadow},
$$

$$
L_{B_j}^{(H)}
=
\text{transition bundle loss after }H_{shadow}.
$$

定义 bundle relative benefit：

$$
\Delta_j^{(H)}
=
\frac{L_K^{(H)}-L_{B_j}^{(H)}}
{|L_K^{(H)}|+\epsilon}.
$$

CounterStruct **不把 bundle outcome 平均后伪装成单 action 的真实 label**。

定义 bundle-level design vector：

$$
\boxed{
x_j
=
\sum_{a\in B_j}\phi(a).
}
$$

定义 observed bundle target：

$$
\boxed{
y_j=\Delta_j^{(H)}.
}
$$

Critic 使用的局部可加模型为：

$$
\boxed{
y_j
\approx
\theta^\top x_j
=
\sum_{a\in B_j}\theta^\top\phi(a).
}
$$

因此每个 shadow bundle 只提供一条 **aggregate intervention equation**；算法从未观测单个 action 的 ground-truth label。

为数值稳定，可同时将 $x_j,y_j$ 除以固定 $K_{shadow}=64$，该变换不改变 ridge 解的含义。

该 local-additivity assumption 不是默认真理，必须由后续 $K=64\rightarrow256$ locality calibration 检验。

---

# 第四部分 · Horizon Critic 与结构状态记忆

## 17. Global-Prior + Task-Local Horizon Critic

v2.1 不再用单一跨任务 ridge 参数直接承担所有任务的 action-value prediction。

定义：

$$
\boxed{
\hat b_{t,H}(a)
=
(\theta_G^{pre}+\delta_t)^\top\phi(a).
}
$$

其中：

- $\theta_G^{pre}$：进入 task $t$ 前冻结的 global structural-dynamics prior；
- $\delta_t$：仅由 task $t$ 的 shadow intervention labels 学到的 task-local residual。

### 17.1 Global prior

维护：

$$
A_G
=
\lambda_G I
+
\sum_{\tau<t}\sum_j x_{\tau j}x_{\tau j}^\top,
$$

$$
b_G
=
\lambda_G\theta_0
+
\sum_{\tau<t}\sum_j x_{\tau j}y_{\tau j}.
$$

$$
\boxed{
\theta_G^{pre}=A_G^{-1}b_G.
}
$$

固定：

$$
\lambda_G=1.
$$

初始 prior：

- $B^{(1)}$ feature coefficient = 1；
- 其它 coefficient = 0。

### 17.2 Task-local residual

task $t$ 开始：

$$
A_t=\lambda_t I,
\qquad
b_t=0,
\qquad
\lambda_t=1.
$$

对当前 task 的每条 aggregate equation：

$$
(x_j,y_j),
$$

使用 global prior residual：

$$
r_j
=
y_j-(\theta_G^{pre})^\top x_j.
$$

累积：

$$
A_t
\leftarrow
A_t+x_jx_j^\top,
$$

$$
b_t
\leftarrow
b_t+x_jr_j.
$$

$$
\boxed{
\delta_t=A_t^{-1}b_t.
}
$$

因此当前任务的 critic prediction 明确包含当前任务 shadow supervision。

### 17.3 Task-end merge

在 structural-state memory 写入完成后，才把 task $t$ 的原始 aggregate equations 合并进 global statistics：

$$
A_G\leftarrow A_G+\sum_jx_jx_j^\top,
$$

$$
b_G\leftarrow b_G+\sum_jx_jy_j.
$$

然后：

$$
A_t,b_t\text{ reset}.
$$

这避免把「未来 task 的 critic」反向解释成对过去 task 的 supervision。

### 17.4 Conservative uncertainty proxy

定义：

$$
\boxed{
\sigma_t(a)
=
\sqrt{
\phi(a)^\top
\left(
A_G^{-1}+A_t^{-1}
\right)
\phi(a)
}.
}
$$

它只解释为 ridge leverage / coverage proxy，不是 Bayesian posterior CI。

---

## 18. Structural-State Historical Memory

v2.1 不保存 coordinate-level $P,H$。

对每个 exact-2:4 group $q$，维护：

$$
\boxed{
R_q\in\mathbb R^6
}
$$

对应 6 个合法 topology states。

$R_q(s)$ 表示：

> 过去任务对 topology state $s$ 的平均相对结构代价。

初始化：

$$
\boxed{R_q(s)=0.}
$$

---

## 19. Task-End Structural-State Consolidation

每个 task 结束、在该 task 数据仍可访问时，使用：

$$
\boxed{|B_{state}|=64}
$$

current-task examples 获取 state-transition features。

对每个 group 当前 state $s_t$，只评价 topology graph 的 4 个 one-swap neighbors：

$$
s\in\mathcal N_{\mathcal G}(s_t).
$$

使用 **该 task 的 localized critic**：

$$
\hat b_{t,H}(s_t\rightarrow s)
=
(\theta_G^{pre}+\delta_t)^\top\phi(s_t\rightarrow s).
$$

定义：

$$
C_{t,q}(s_t)=0,
$$

对 one-swap neighbor：

$$
\boxed{
C_{t,q}(s)
=
-\operatorname{clip}
\left(
\hat b_{t,H}(s_t\rightarrow s),
-c_{max},c_{max}
\right).
}
$$

对与当前 state 不相邻的唯一 complement state，不让 critic 对未训练过的 two-swap transition 外推。

利用 $J(4,2)$ 的结构：current state 与其 complement 共享同一组 4 个中间邻居。将 complement cost 定义为 graph-harmonic completion：

$$
\boxed{
C_{t,q}(s_{comp})
=
\frac14
\sum_{u\in\mathcal N_{\mathcal G}(s_t)}
C_{t,q}(u).
}
$$

这等价于在未观测 complement node 上采用 unweighted graph-Laplacian minimum-energy harmonic extension。

该值只用于补全 6-state memory potential，不被解释为真实 two-step rollout value；real mutation 始终只允许 one-swap atomic action。

固定 clipping scale：

$$
c_{max}
=
5\left(
\operatorname{median}(|\hat b_{t,H}|)
+\epsilon_C
\right).
$$

历史 memory：

$$
\boxed{
R_q^{(t)}(s)
=
\frac{t-1}{t}R_q^{(t-1)}(s)
+
\frac1tC_{t,q}(s).
}
$$

### 19.1 正确解释

$R$ 不是「未来时刻重新预测过去任务 loss」。

它保存的是：

> **每个历史 task 在它仍是 current task、其 shadow labels 仍可获得时形成的 state-relative preference estimate 的等任务权平均。**

因此 $1/t$ 衰减是刻意对应：

$$
\boxed{\text{average-over-tasks continual objective}}
$$

而不是遗忘 bug。

但这种 mean-memory 可能弱化 worst-case / earliest-task protection，因此 Long-CL 必须额外报告 early-task retention，不能只看平均 Forgetting。

---

# 第五部分 · 决策与执行

## 20. Historical Structural Damage

mutation event 中当前 state：

$$
s\rightarrow s'.
$$

定义：

$$
\boxed{
D_R(a)
=
R_q(s')-R_q(s).
}
$$

若：

$$
D_R>0,
$$

表示历史 structural-state memory 认为新 state 比当前 state 更不利。

允许：

$$
D_R<0.
$$

---

## 21. Final Horizon-Calibrated Structural Utility

Critic current-task value：

$$
\hat b_{t,H}(a).
$$

historical state damage：

$$
D_R(a).
$$

uncertainty：

$$
\sigma_t(a).
$$

定义：

$$
\boxed{
U_{CS}(a)
=
\hat b_{t,H}(a)
-
D_R(a)
-
\beta\sigma_t(a).
}
$$

固定：

$$
\boxed{\beta=1.}
$$

解释：

> conservative horizon-calibrated structural action value；其中 uncertainty 项是 leverage proxy，而非严格 Bayesian posterior confidence interval。

禁止解释为 exact long-term return。

Admission：

$$
\boxed{U_{CS}(a)>0.}
$$

---

## 22. Preliminary Score for Shadow Stratification

在 critic 当次 event 更新前，用：

$$
\boxed{
S_{prior}(a)
=
\widetilde{B^{(1)}(a)}
-D_R(a)
}
$$

仅用于构造 shadow quantile strata。

实际 topology selection 必须使用更新后的：

$$
U_{CS}.
$$

---

## 23. Group-Wise Topology-Graph Search

对 group $q$ 当前 state $s_q$，只枚举 Johnson-graph 邻居：

$$
\boxed{
\mathcal A_q
=
\{
s_q\rightarrow s':
s'\in\mathcal N_{\mathcal G}(s_q)
\}.
}
$$

每组固定最多：

$$
\boxed{4}
$$

个 one-swap actions。

过滤：

- prune coordinate 必须 mature；
- transition 后 exact 2:4 必须成立。

选择：

$$
a_q^*
=
\arg\max_{a\in\mathcal A_q}
U_{CS}(a).
$$

若最大值：

$$
\le0,
$$

该 group 不 mutation。

---

## 24. Edge-Turnover Budget

所有原子 action 都只替换 1 条 active edge，因此 budget 直接定义为：

$$
\boxed{
B_{edge}
=
\left\lfloor
\rho_{edge}N_{active}
\right\rfloor.
}
$$

Development-only：

$$
\boxed{
\rho_{edge}\in\{0.5\%,1.0\%\}.
}
$$

每个被选 action 消耗 1 个 edge-replacement unit。

候选 group 按：

$$
\boxed{U_{CS}(a_q^*)}
$$

排序，在：

$$
U_{CS}>0
$$

且 edge budget 内依次 admission。

所有 structural baselines 使用 matched replaced-active-edge budget。

同时报告：

- selected group fraction；
- actual replaced active-edge fraction；
- positive-U candidate fraction。

---

## 25. Topology Transition

对所有被 deactivate 的 coordinates：

$$
M_i\leftarrow0,
\quad
w_i\leftarrow0,
\quad
m_i\leftarrow0,
\quad
v_i\leftarrow0,
\quad
C_i\leftarrow0.
$$

对所有 newborn coordinates：

$$
M_i\leftarrow1,
\quad
w_i\leftarrow0,
\quad
m_i\leftarrow0,
\quad
v_i\leftarrow0,
\quad
C_i\leftarrow0.
$$

保持：

$$
R_q\text{ unchanged during immediate transition}.
$$

$R$ 只在 task-end consolidation 更新。

---

## 26. CounterStruct v2.1 Pseudocode

```text
Input:
    pretrained model
    tasks T1 ... TT
    exact contiguous 2:4 topology graph G_2:4

Persistent state:
    W, M
    AdamW m, v
    cooldown C
    structural memory R[group, 6]
    global critic A_G, b_G

Initialize:
    M0 = top-2-of-4 magnitude
    dormant W/m/v = 0
    R = 0
    initialize global one-step prior

For each task t:

    freeze theta_G_pre from A_G, b_G
    initialize task-local residual A_t = I, b_t = 0
    task_equation_buffer = []

    train CURRENT TASK ONLY
    mutation events at 20/40/60/80%

    for normal optimizer steps:
        normal dense-masked training

        if mutation event:

            update cooldown

            # 1. current structural probe
            compute STE signed gradients

            # 2. enumerate 4 Johnson-graph neighbors/group
            compute one-step B1
            compute fixed 10-D relative-rank features phi
            compute historical D_R

            # 3. eight shadow intervention equations
            build 8 stratified bundles, K=64
            run one keep + eight transition branches for H=8
            obtain aggregate x_j, y_j

            # 4. current-task residual update
            residual_j = y_j - theta_G_pre^T x_j
            update A_t, b_t
            delta_t = inverse(A_t) b_t

            # 5. score real actions
            b_hat = (theta_G_pre + delta_t)^T phi
            sigma = leverage(A_G, A_t, phi)
            U = b_hat - D_R - beta * sigma

            select best positive one-swap action/group
            admit under matched edge-turnover budget
            execute real transitions
            reset changed W/m/v/C

            save x_j, y_j into task_equation_buffer
            release shadow buffers

    # task-end state-memory consolidation
    use current-task data only
    use theta_G_pre + delta_t
    estimate current task state-relative costs C_t
    R = ((t-1)/t) * R + (1/t) * C_t

    # only after memory write:
    merge task_equation_buffer into A_G, b_G

    discard task-local residual and current-task probe data
    evaluate task matrix
```

# 第六部分 · Benchmark 与实验设计

## 27. Primary Benchmark — TRACE-8 Order-1

固定顺序：

```text
C-STANCE
→ FOMC
→ MeetingBank
→ Py150
→ ScienceQA
→ NumGLUE-cm
→ NumGLUE-ds
→ 20Minuten
```

每 task：5000 training examples。

Task epochs：

$$
\boxed{[5,3,7,5,3,5,5,7]}.
$$

其它训练配置继承 v1.0：

- AdamW；
- LR $1\times10^{-5}$；
- cosine；
- effective batch 32；
- max prompt 1024；
- max answer 512；
- Qwen `enable_thinking=False`。

---

## 28. Benchmark — TRACE Order-2

使用与 v1.0 一致的预注册 Order-2。

所有 CounterStruct-specific config zero-retuning。

主要回答 task-order sensitivity。

---

## 29. Benchmark — Seq-GLUE-7

顺序：

```text
CoLA → SST-2 → MRPC → QQP → QNLI → RTE → MNLI
```

one epoch/task。

CounterStruct-specific config 全部从 TRACE dev freeze 后迁移。

---

## 30. Benchmark — Long-CL-15

继续使用公开 15-task composition 与预注册 Order-4。

每 task 最多 1000 unique training examples。

Long-CL 同时检查两类 failure：

### A. structural freezing

$$
\boxed{
f_t^{U>0},
\quad
\rho_{edge,t},
\quad
AG_t
}
$$

若后半程 positive-U 与 edge turnover 同时塌缩并伴随 AG collapse，则判定 freezing。

### B. mean-memory dilution / early-task erosion

额外必须报告：

$$
\boxed{
ACC_{T_1}(t),
\quad
ACC_{\text{first-5}}(t),
\quad
F_{\text{early}}(t)
}
$$

其中：

$$
F_{\text{early}}
$$

为最早 5 个任务的平均 forgetting。

原因：

> $R$ 使用等任务权 running mean，理论上对 average-over-tasks objective 合理，但可能牺牲 worst-case early-task protection。

CounterStruct-specific config 全部 zero-retuning。

---

## 31. Experiment A — Primary Structural Evidence

$$
\boxed{\text{Qwen3-1.7B}\times\text{TRACE-8 Order-1}}
$$

Mandatory structural methods：

1. Dense Regional FT；
2. Static Exact-2:4；
3. SRigL-style Exact-2:4；
4. IPGH；
5. **LFS — Local-Fisher Swap**（v1.0 core）；
6. **CounterStruct v2.1**。

Seeds：

- Static / SRigL / IPGH / LFS / CounterStruct：

$$
\boxed{5\text{ seeds}:42,43,44,45,46}
$$

- Dense Regional FT：3 seeds。

Primary structural contrasts：

1. CounterStruct vs LFS；
2. CounterStruct vs SRigL；
3. LFS vs IPGH。

第一个 contrast 是新颖性最重要的 contrast。

---

## 32. Experiment B — Recent Continual-Learning Comparators

同一 Qwen3-1.7B × TRACE Order-1 common protocol。

Mandatory external suite 缩减为：

- Naive FT；
- LoRA；
- O-LoRA；
- Meta-UCF；
- OSFT；
- PaRSP；
- Any-SSR。

全部 3 seeds：42/43/44。

GORP / TreeLoRA 不再 mandatory，资源允许时放 appendix。

原因：

> v2.1 主贡献是 structural action-value mechanism，不用 baseline 数量替代机制证据。

仍必须进行 official-code / replay / task-ID / persistent-state / contamination audit。

---

## 33. Experiment C — Benchmark Generalization

$$
\boxed{\text{Qwen3-1.7B}\times\text{Seq-GLUE-7}}
$$

Required：

- SRigL；
- LFS；
- CounterStruct；
- O-LoRA；
- Meta-UCF；
- OSFT；
- PaRSP。

3 seeds。

---

## 34. Experiment D — Long-Horizon

$$
\boxed{\text{Qwen3-1.7B}\times\text{Long-CL-15 Order-4}}
$$

Required：

- Static；
- SRigL；
- LFS；
- CounterStruct；
- O-LoRA；
- Meta-UCF；
- OSFT。

3 seeds。

---

## 35. Experiment E — Task-Order Robustness

$$
\boxed{\text{Qwen3-1.7B}\times\text{TRACE Order-2}}
$$

Required：

- Static；
- SRigL；
- LFS；
- CounterStruct。

3 seeds。

zero retuning。

---

## 36. Experiment F — 8B Scale-Direction Replication

$$
\boxed{\text{Qwen3-8B}\times\text{TRACE Order-1}}
$$

v2.1 将 8B 从 single-seed descriptive experiment 提升为 mandatory multi-seed structural replication。

Required：

- Static；
- SRigL；
- LFS；
- CounterStruct。

Seeds：

$$
\boxed{42,43,44}.
$$

全部 zero retuning。

只允许声明：

> mechanism effect direction and multi-seed replication on an 8B backbone.

仍禁止 scaling law。

---

## 37. LFS — Local-Fisher Swap Baseline

LFS 完整保留 v1.0 核心：

- exact 2:4 one-swap actions；
- one-step AdamW counterfactual；
- squared-gradient historical $P,H$；
- $U=B_{cur}-D_{hist}$；
- same mutation schedule；
- same current probe；
- same edge-turnover budget。

LFS 的作用：

$$
\boxed{
\text{证明 v2.1 的增益不是 v1.0 那种 Taylor + Fisher + DST 组合即可解释。}
}
$$

---

## 38. Memory Comparator — EWC-DR-style Structural Protection

Secondary ablation：

在 LFS historical importance 中加入 EWC-DR-style importance estimator，保持其余 structural action protocol 不变。

目的：

> 排除「v2.1 只是因为原 Fisher estimator 太差」这一解释。

该实验只在 Qwen3-1.7B TRACE seed42/43/44 上运行。

---

## 39. Main Evaluation Matrix

每完成 task $i$，评价全部 tasks $j$：

$$
A_{i,j}.
$$

报告：

- Final ACC；
- Forgetting；
- AG；
- BWT；
- FWT。

Primary stability-plasticity requirement：

$$
\boxed{
AG_{CS}\ge AG_{SRigL}-1.0\text{ point}.
}
$$

---

## 40. General Capability

继续使用：

- MMLU；
- BBH；
- TyDiQA；
- PIQA；
- BoolQ。

结构方法必须分解：

$$
W_{dense}\rightarrow M_0\rightarrow M_T.
$$

报告：

$$
\Delta GA_{init},
\quad
\Delta GA_{stream},
\quad
\Delta GA_{total}.
$$

不允许隐藏 initial 2:4 pruning cost。

---

# 第七部分 · 核心机制实验

## 41. Temporal Calibration — 核心机制实验

这是 v2.1 的第一主机制实验。

固定 checkpoints：

- TRACE $T_4@60\%$；
- TRACE $T_7@60\%$。

Seeds：

$$
\boxed{42,43,44}.
$$

构造 held-out mutation bundles：

- 10 predicted-value deciles；
- 3 independent families/decile；
- primary $K=64$；
- secondary $K=256$。

所有 calibration bundles 与 online 8-bundle shadow-training equations 完全 disjoint。

正文 Figure 2 必须同时展示：

$$
\rho_{RigL}(H),
\quad
\rho_{1step}(H),
\quad
\rho_{critic}(H).
$$

---

## 42. Multi-Horizon Realized Value

从相同 checkpoint 为每个 calibration bundle clone keep / transition branches。

所有 branches 使用同一 integration sequence。

最长运行：

$$
\boxed{H_{max}=64\text{ real optimizer steps}}.
$$

在：

$$
\boxed{H\in\{1,8,32,64\}}
$$

分别评价相同 held-out current-task batch。

定义：

$$
U_{real}^{(H)}
=
\frac{L_K^{(H)}-L_B^{(H)}}{|L_K^{(H)}|+\epsilon}.
$$

比较三种 predictor：

1. RigL / magnitude-gradient heuristic；
2. analytic one-step $B^{(1)}$；
3. CounterStruct horizon critic $\hat b_H$。

---

## 43. Primary Temporal-Validity Metrics

对每个 horizon：

$$
\rho_{critic}(H)
=
\operatorname{Spearman}(
\widehat U_{critic},
U_{real}^{(H)}
).
$$

同样计算：

$$
\rho_{1step}(H),
\quad
\rho_{RigL}(H).
$$

Primary target：

$$
\boxed{\rho_{critic}(64)}.
$$

预注册成功标准：

$$
\boxed{
\rho_{critic}(64)\ge0.30
}
$$

且 95% bootstrap CI lower bound > 0。

同时要求相对 one-step prior：

$$
\boxed{
\rho_{critic}(64)-\rho_{1step}(64)\ge0.10
}
$$

作为 practical target；若 CI 不支持该差异，不得宣称 horizon critic 显著优于 one-step surrogate。

---

## 44. Locality / Aggregate-Credit Calibration

online critic 使用 $K=64$ aggregate intervention equations 学习 action ranking，因此必须分别检验：

1. **规模 locality**；
2. **action interaction / credit assignment**。

### 44.1 Bundle-scale locality

比较：

$$
K\in\{64,256\}.
$$

报告：

- $\rho(K,H)$；
- predicted decile vs realized value；
- $P(U_{real}>0\mid \widehat U>0)$；
- $K=64\rightarrow256$ correlation degradation。

如果 $K=64$ 本身无 predictive association，则 aggregate critic 主线失败。

### 44.2 Direct Micro-Bundle Interaction Audit

这是 v2.1 新增的 mandatory credit-assignment audit，不作为 online critic training data。

单个 weight-level action 在 1.7B 模型上的 realized loss effect 可能低于 numerical noise，因此 primary audit 不把「singleton 必须可测」作为前提，而使用更可测量的 **micro-bundle factorial intervention**。

在 T4/T7、seeds 42/43/44：

每个 checkpoint × seed 选择：

$$
\boxed{16\text{ matched micro-bundle pairs}}
$$

每对 $(A,B)$ 满足：

- $|A|=|B|=8$；
- A、B 内部以及 A/B 之间全部 group-disjoint；
- 来自相同 predicted-value decile；
- 尽量匹配 layer / matrix composition；
- 不与 online shadow-training bundles 重叠。

从同一 checkpoint 分支：

1. Keep；
2. A-only；
3. B-only；
4. $A\cup B$-joint。

primary：

$$
H=8.
$$

另对每 checkpoint × seed 的 4 对执行：

$$
H=64
$$

stress test。

定义 realized benefits：

$$
Y_A,\quad Y_B,\quad Y_{A\cup B}.
$$

二阶 interaction：

$$
\boxed{
I_{A,B}^{(H)}
=
Y_{A\cup B}^{(H)}
-
Y_A^{(H)}
-
Y_B^{(H)}.
}
$$

normalized interaction ratio：

$$
\boxed{
\Gamma(H)
=
\operatorname{median}_{(A,B)}
\frac{
|I_{A,B}^{(H)}|
}{
|Y_A^{(H)}|+|Y_B^{(H)}|+\epsilon
}.
}
$$

同时报告：

- median / P90 $\Gamma$；
- interaction sign；
- additive model 是否导致 micro-bundle ranking reversal。

预注册解释：

$$
\Gamma(8)\le0.25
$$

视为 strong local-additivity support；

$$
0.25<\Gamma(8)\le0.50
$$

视为 approximate / noisy support；

若：

$$
\boxed{\Gamma(8)>0.50}
$$

或 joint interaction 大量导致 ranking reversal，则 per-action additive critic 的机制解释失败。

### 44.3 Optional Singleton Audit

只有在 P0 noise-floor test 证明单 action H=8 effect 可稳定测量时，才额外运行 singleton $a$、$b$、$\{a,b\}$ audit。

若 singleton signal 低于 noise floor，不把「测不出 singleton」解释成 additivity failure；primary 结论仍由 K=8 micro-bundle factorial audit 给出。

该设计比只看 $K=64\rightarrow256$ 更直接，也避免强行用不可测的单-coordinate loss change 做统计结论。

---

## 45. Two-Stage Structural-Memory Calibration

v2.1 将 structural-memory 证据拆成：

$$
\boxed{
\text{write-time fidelity}
\rightarrow
\text{future historical fidelity}
}
$$

避免只在未来事后审计 $R$。

### 45.1 Write-Time Fidelity

在 T4/T7 checkpoint 对当前 task 使用 held-out current-task bundles，不进入 online critic training。

预测：

$$
\widehat C_t(B)
=
-\sum_{a\in B}
\hat b_{t,H}(a).
$$

实际 current-task structural cost 由同 checkpoint keep / transition branch 的 H=8 relative loss 得到：

$$
C_{t,real}^{(8)}(B).
$$

计算：

$$
\boxed{
\rho_{write}
=
\operatorname{Spearman}
(
\widehat C_t,
C_{t,real}^{(8)}
).
}
$$

这直接验证：

> 被写入 $R$ 的 task-specific preference 在该 task 仍为 current task 时是否有 empirical grounding。

### 45.2 Future Historical Fidelity

在相同 T4/T7 checkpoints，使用 evaluation-only past-task batches。

预测 historical damage：

$$
\widehat D_R(B)
=
\sum_{a\in B}
[R(s'_a)-R(s_a)].
$$

实际：

$$
D_{past,real}^{(64)}
=
\sum_{\tau<t}
\frac{
L_{\tau,B}^{(64)}-L_{\tau,K}^{(64)}
}{
|L_{\tau,K}^{(64)}|+\epsilon
}.
$$

计算：

$$
\boxed{
\rho_R
=
\operatorname{Spearman}
(
\widehat D_R,
D_{past,real}^{(64)}
).
}
$$

Past-task data 只用于 calibration evaluation，永不用于：

- critic training；
- $R$ update；
- hyperparameter selection；
- topology decision。

主文必须明确：

> $R$ 保存的是 historical task-local structural preference estimates，而不是对 past loss landscape 的精确重建。

---

## 46. Macro-Horizon Bridge

DeepSeek 指出的核心风险必须通过 benchmark-level ablation 回答：

> 即使 horizon critic 能预测 8/64-step loss，它是否真的改善 task-end continual behavior？

因此必须比较：

- CounterStruct full；
- CounterStruct w/o Horizon Critic；
- LFS。

若 full method 在 task-end ACC / Forgetting 上不优于 w/o critic，则不得把 temporal calibration 作为最终 continual-learning improvement 的机制解释。

# 第八部分 · Ablations

## 47. Ablation A1 — w/o Horizon Critic

使用：

$$
\hat b_H(a)\leftarrow \widetilde{B^{(1)}(a)}.
$$

保留 structural-state memory。

回答：

> shadow horizon learning 是否必要？

---

## 48. Ablation A2 — w/o Structural-State Memory

设置：

$$
R=0.
$$

保留 horizon critic。

回答：

> historical structural preference 是否必要？

---

## 49. Ablation A3 — Fisher Memory

用 LFS $P,H$ historical penalty 替换 $R$，保留 horizon critic current value。

回答：

> state-level memory 是否优于 per-coordinate quadratic protection？

---

## 50. Ablation A4 — w/o Task-Local Residual

设置：

$$
\delta_t=0.
$$

即所有 task 只使用：

$$
\hat b_H(a)=\theta_G^\top\phi(a).
$$

保留：

- 8 shadow bundles/event；
- structural-state memory；
- uncertainty admission。

回答：

> 当前任务自己的 shadow supervision 是否只需要更新 global critic，还是 task-local calibration 对当前决策与 memory write 确实必要？

如果 full 与 global-only 无差异，则 task-local residual 不能作为必要机制解释。

---

## 51. Ablation A5 — w/o Uncertainty Penalty

$$
\beta=0.
$$

比较主方法：

$$
\beta=1.
$$

回答：

> conservative admission 是否减少错误 structural mutations？

---

## 52. Ablation A6 — One-Step Shadow Horizon

$$
H_{shadow}=1.
$$

与主方法：

$$
H_{shadow}=8
$$

比较。

这是直接针对「one-step self-consistency」批评的 ablation。

---

# 第九部分 · 行为指标与资源账本

## 53. Structural Behavior Metrics

每 mutation event 记录：

- legal action count；
- positive $U$ action fraction；
- selected groups；
- edge-turnover ratio；
- critic global prediction；
- task-local residual magnitude；
- critic uncertainty；
- historical $D_R$ distribution；
- shadow keep loss；
- 8 shadow bundle realized benefits；
- global/local critic residuals；
- topology Jaccard。

每 task 记录：

- number of accumulated global equations；
- number of task-local equations；
- $\|\delta_t\|_2$；
- write-time $\rho_{write}$（calibration checkpoints）；
- $R$ state dispersion。

Long-CL 按 task index 绘制：

- $\operatorname{Var}_s(R_t)$；
- positive-U fraction；
- edge-turnover ratio；
- critic uncertainty；
- $ACC_{T_1}(t)$；
- $ACC_{\text{first-5}}(t)$；
- $F_{\text{early}}(t)$；
- overall AG / Forgetting。

---

## 54. Memory Accounting

候选 coordinates：

$$
N=201,326,592.
$$

2:4 groups：

$$
G=N/4
=
\boxed{50,331,648}.
$$

Structural-state memory：

$$
R\in\mathbb R^{G\times6}.
$$

若 BF16：

$$
50,331,648\times6\times2
=
603,979,776\text{ bytes}.
$$

即约：

$$
\boxed{0.604\text{ GB}\approx0.563\text{ GiB}.}
$$

相比 LFS 的 FP32 $P,H$：

$$
\approx1.61\text{ GB}.
$$

主方法 historical structural memory 约降低：

$$
\boxed{2.67\times}.
$$

但论文不宣称其「小于 replay」。

必须报告：

- persistent bytes；
- growth/task；
- replay bytes；
- shadow temporary buffers；
- peak HBM。

---

## 55. Memory-Matched Replay Secondary Baseline

为直接回应「固定内存不等于小内存」：

除常规 ER-64/task 外，加入 secondary：

> **ER-MemoryMatched**

允许 historical exemplars 的实际 serialized/tokenized bytes 不超过 CounterStruct persistent structural memory bytes。

该 baseline 明确属于不同约束条件：它允许保存历史样本。

目的不是要求 CounterStruct 打败 memory-matched replay，而是透明展示：

$$
\text{history-data-free constraint}
\quad vs\quad
\text{sample-storage allowed}
$$

的 Pareto tradeoff。

---

## 56. Statistical Protocol

Primary structural seed set：

$$
\boxed{\{42,43,44,45,46\}}.
$$

External / broad / Order-2 / Long-CL：

$$
\boxed{\{42,43,44\}}.
$$

8B structural replication：

$$
\boxed{\{42,43,44\}}.
$$

主表：

$$
\mathrm{mean}\pm SE.
$$

Primary paired contrasts：

1. CounterStruct vs LFS；
2. CounterStruct vs SRigL。

paired hierarchical bootstrap：10,000 次，95% CI。

不把 tasks 当作 iid replicates。

---

## 57. Practical Effect Thresholds

除统计方向外，预注册 practical thresholds。

CounterStruct vs LFS / SRigL：

至少满足一个：

$$
\boxed{
\Delta ACC\ge1.0\text{ point}
}
$$

或：

$$
\boxed{
\Delta Forgetting\le-1.0\text{ point}
}
$$

同时：

$$
AG_{CS}\ge AG_{SRigL}-1.0.
$$

若只出现 <1 point 且 CI 大量跨 0 的改善，结果定义为「weak effect」，不包装成 strong method win。

---

# 第十部分 · Go/No-Go 门禁

## 58. Go / No-Go

### G0 — Correctness

必须全部通过：

- exact 2:4 legality 100%；
- 6 state encodings tested；
- Johnson-graph 4-neighbor mapping tested；
- dormant W/m/v = 0；
- shadow branch isolation；
- optimizer / scheduler / RNG restore exact；
- cached-prefix forward parity；
- global/local critic merge unit test。

**Deterministic shadow replay test：**

在 deterministic debug config 下，同 checkpoint、同 bundle、同 batches 连续重复两次 branch rollout：

- 若 kernel 支持 deterministic path，要求 loss trajectory bitwise identical；
- 否则预声明 tolerance，并要求 relative loss difference $\le10^{-6}$。

性能正式 run 可使用更快 kernel，但必须先量化 nondeterministic noise floor。

### G1 — Dynamic Topology Works

SRigL 相对 Static 至少显示 plasticity trend。

### G2 — LFS Baseline Works

LFS 必须实现 v1.0 intended behavior；若 LFS 不优于 IPGH/SRigL，先检查 complete-action premise。

### G3 — Shadow Signal Measurable

$K=64,H=8,J=8$ 的 shadow realized effects 必须：

- 超过 numerical / nondeterministic noise floor；
- 在 repeated branches 中可复现；
- 至少覆盖正负 outcome，而不是全部接近 0。

### G4 — Horizon Critic Valid

$$
\boxed{
\rho_{critic}(64)\ge0.30
}
$$

且 bootstrap 95% CI lower bound > 0。

若：

$$
\rho_{critic}(64)<0.20
$$

则 horizon-predictive 主 claim 失败。

### G5 — Critic Beats One-Step

practical target：

$$
\boxed{
\rho_{critic}(64)-\rho_{1step}(64)\ge0.10.
}
$$

若没有稳定改善，不得把 horizon critic 作为主要算法贡献。

### G6 — Aggregate Credit Valid

micro-bundle interaction audit：

strong target：

$$
\Gamma(8)\le0.25.
$$

若：

$$
\boxed{\Gamma(8)>0.50}
$$

或大量 micro-bundle ranking reversal，则 additive per-action critic interpretation 失败。

### G7 — Structural Memory Valid

同时要求：

1. write-time $\rho_{write}>0$ 且方向稳定；
2. future $\rho_R>0$；
3. full CounterStruct 相对 w/o $R$ 在 Forgetting 上有一致改善趋势。

若只有 $\rho_R>0$ 而 $\rho_{write}\approx0$，不得把 $R$ 描述成经当前任务监督的 structural preference memory。

### G8 — Full Method Beats LFS

5-seed TRACE Order-1 中达到 practical effect threshold，且 AG non-inferiority 成立。

这是最重要的 method novelty gate。

### G9 — Order / Benchmark Transfer

TRACE Order-2 与 Seq-GLUE 至少保持主要 effect direction。

### G10 — Long-Horizon Memory Behavior

Long-CL 同时检查：

#### freezing

不得同时出现：

- positive-U fraction $\rightarrow0$；
- edge turnover $\rightarrow0$；
- AG collapse。

#### early-task erosion

若 overall average 尚可但：

$$
ACC_{T_1}(t)
$$

或 first-5-task retention 持续系统性恶化，则记录：

> mean-memory dilution limitation.

不得只用平均 Forgetting 隐藏 early-task failure。

### G11 — 8B Replication

3 seeds 上 CounterStruct vs LFS/SRigL 的主要 effect direction 与 1.7B 一致。

### G12 — Resource Bound

- $R$ memory $\le0.7$ GB；
- persistent state/task growth = 0；
- shadow overhead target $\le30\%$；
- hard engineering ceiling = 40%。

---

# 第十一部分 · 执行协议

## 59. Development Protocol

唯一 method development stream：

$$
\boxed{
\text{Qwen3-1.7B}
\times
\text{TRACE Order-1}
\times
\text{seed 42}
}
$$

允许开发：

1. $\rho_{edge}\in\{0.5\%,1.0\%\}$；
2. 一次 $H_{shadow}\in\{4,8\}$ pilot，仅用于 signal/noise 与 runtime；
3. branch caching / deterministic correctness engineering。

固定不搜索：

- $J_{shadow}=8$；
- $K_{shadow}=64$；
- feature dimension $d=10$；
- percentile/rank feature transform；
- $\lambda_G=1$；
- $\lambda_t=1$；
- $\beta=1$；
- one-swap Johnson-graph action space；
- $R$ running task mean。

micro-bundle interaction audit：

- 不能用于调 feature；
- 不能用于选择 $\lambda$；
- 不能反向选择 K；
- 只用于 validate / falsify aggregate-credit assumption。

若必须修改上述核心公式或 action graph，版本继续升级，禁止复用 frozen final-test 结果。

---

## 60. Freeze Protocol

Final multi-seed runs 前冻结：

- model / dataset revisions；
- task orders；
- train/eval/test split；
- target layers；
- exact contiguous 2:4 grouping；
- 6-state encoding；
- Johnson-graph neighbor mapping；
- one-swap-only atomic action rule；
- 10-D feature definition；
- event-percentile feature transform；
- $J=8,K=64,H=8$；
- global prior formula；
- task-local residual formula；
- global merge timing；
- ridge lambdas；
- uncertainty beta；
- $R$ write rule；
- $R$ equal-task running mean；
- edge-turnover budget；
- all primary baselines；
- primary statistical contrasts；
- temporal calibration checkpoints/horizons；
- micro-bundle interaction audit protocol；
- write-time / future-memory calibration；
- practical effect thresholds；
- claim boundaries。

---

## 61. Execution Priority

### P0 — Mechanism / Novelty Gate

1. exact 6-state + Johnson-graph unit tests；
2. Static seed42；
3. SRigL seed42；
4. IPGH seed42；
5. LFS seed42；
6. cached-prefix shadow branch；
7. deterministic branch replay test；
8. $K=64,H=8,J=8$ noise-floor pilot；
9. global-prior + task-local residual critic toy test；
10. CounterStruct seed42 partial TRACE；
11. temporal pilot：critic vs one-step at H=64；
12. micro-bundle interaction pilot；
13. task-local memory-write pilot。

**若 G3/G4/G5/G6 中任一明显失败，不启动完整 external / 8B。**

### P1 — Primary Paper Evidence

14. TRACE Order-1 structural 5 seeds；
15. held-out temporal calibration 3 seeds；
16. micro-bundle interaction audit 3 seeds；
17. write-time + future state-memory calibration；
18. A1/A2/A3/A4 core ablations；
19. macro-horizon bridge；
20. general-capability decomposition；
21. profiling / memory accounting。

### P2 — Robustness

22. TRACE Order-2 3 seeds；
23. Long-CL-15 3 seeds，含 early-task retention；
24. Seq-GLUE 3 seeds。

### P3 — Recent Baselines

25. OSFT；
26. Meta-UCF；
27. PaRSP；
28. Any-SSR；
29. O-LoRA；
30. LoRA / Naive FT。

### P4 — Scale

31. Qwen3-8B Static/SRigL/LFS/CounterStruct 3 seeds；
32. real 2:4 runtime validation。

### P5 — Appendix

33. EWC-DR structural comparator；
34. no uncertainty；
35. H=1 shadow；
36. memory-matched replay；
37. optional Llama cross-family replication；
38. optional GORP / TreeLoRA。

# 第十二部分 · 可复现与硬件

## 62. Required Run Artifacts

```text
config.yaml
environment.txt
git_commit.txt
model_revision.txt
dataset_revision.txt
seed_manifest.yaml
claim_boundary.yaml

metrics/
    task_matrix.json
    aggregate_metrics.json
    general_ability.json
    early_task_retention.json

structure/
    mask_task_*.pt
    state_id_task_*.pt
    mutation_events.jsonl
    topology_similarity.json

critic/
    global_A_task_*.pt
    global_b_task_*.pt
    local_A_event_*.pt
    local_b_event_*.pt
    global_coefficients.jsonl
    local_residual_coefficients.jsonl
    critic_uncertainty.jsonl
    feature_transform_manifest.yaml

shadow/
    bundle_manifest.jsonl
    keep_losses.parquet
    transition_losses.parquet
    labels.parquet
    deterministic_replay_check.json
    cached_prefix_parity.json

history/
    R_task_*.pt
    R_summary.jsonl
    write_time_calibration.json

calibration/
    temporal_bundles.jsonl
    realized_h1.parquet
    realized_h8.parquet
    realized_h32.parquet
    realized_h64.parquet
    rho_by_horizon.json
    rho_by_bundle_size.json
    microbundle_interaction_h8.parquet
    microbundle_interaction_h64.parquet
    interaction_gamma.json
    historical_damage_calibration.json

profiling/
    normal_step_latency.json
    shadow_step_latency.json
    mutation_latency.json
    peak_hbm.json
    persistent_state_bytes.json
```

---

## 63. Mutation Log Schema

每 selected action 至少记录：

```yaml
model:
benchmark:
task_id:
event_id:
layer:
matrix:
group_id:
current_state:
target_state:
johnson_neighbor: true

B1_prior:
critic_global:
critic_local_residual:
critic_total:
critic_sigma:
historical_D_R:
U_CS:

selected:
rank:
```

每 event 额外记录：

```yaml
num_legal_actions:
num_positive_actions:
selected_groups:
replaced_edges:
edge_turnover_ratio:

shadow_bundle_count: 8
shadow_horizon: 8
shadow_overhead_seconds:

global_equation_count:
task_local_equation_count:
local_residual_norm:
critic_residual_summary:
```

calibration run 额外记录：

```yaml
rho_critic_h1:
rho_critic_h8:
rho_critic_h32:
rho_critic_h64:
rho_write:
rho_R:
interaction_gamma_h8:
interaction_gamma_h64:
```

---

## 64. Hardware Validation

最终 learned exact-2:4 topology 必须转换到真实 semi-structured sparse representation。

至少 benchmark：

Qwen3-1.7B：

$$
6144\times2048,
\quad
2048\times6144.
$$

Qwen3-8B：

$$
12288\times4096,
\quad
4096\times12288.
$$

记录：

- exact 2:4 legality；
- conversion success；
- dense-masked parity；
- sparse latency；
- dense latency；
- throughput；
- p10/p50/p90；
- memory。

训练阶段仍明确为 dense masked training，不宣称 sparse-training acceleration。

---

# 第十三部分 · 论文主图与成败定义

## 65. 论文主图预注册

如果结果支持，正文优先放以下 5 张图。

### Figure 1 — Method

6-state Johnson topology graph → 8 shadow bundles → global + task-local horizon critic → task-local state-memory write → conservative action selection。

### Figure 2 — Temporal Predictivity + Credit Validity

主 panel：

$$
\rho(H),
\qquad
H=1,8,32,64
$$

三条曲线：

- RigL；
- one-step；
- CounterStruct critic。

副 panel：

$$
\Gamma(8),\Gamma(64)
$$

以及 micro-bundle factorial interaction scatter。

### Figure 3 — Macro-Horizon Bridge

CounterStruct full / w/o horizon critic / w/o task-local residual / LFS：

- ACC；
- Forgetting；
- AG。

这张图负责把：

$$
\text{predictive critic}
$$

连接到：

$$
\text{continual benefit}.
$$

### Figure 4 — Structural Memory Through Time

Long-CL 同时画：

- positive-U；
- edge turnover；
- $R$ dispersion；
- AG；
- $ACC_{T_1}$；
- first-5-task forgetting。

### Figure 5 — Performance–Memory–Compute Pareto

包含：

- CounterStruct；
- LFS；
- SRigL；
- OSFT；
- Meta-UCF；
- PaRSP；
- replay baselines。

---

## 66. 论文成功模式

理想证据组合：

1. one-step score 对 H=64 的相关明显衰减；
2. horizon critic 在 H=64 仍保持中等相关；
3. critic 相对 one-step 的 $\Delta\rho$ 在 3 seeds 一致；
4. micro-bundle interaction audit 显示 H=8 局部 interaction 足够小，aggregate credit 有直接实验证据；
5. full CounterStruct 在 final ACC / Forgetting 上达到 practical threshold 并优于 LFS；
6. global+task-local critic 优于 global-only；
7. $\rho_{write}>0$ 且 $\rho_R>0$，说明 memory write 与 future damage 两阶段都可校准；
8. Long-CL 不出现 freezing，也不出现被平均指标隐藏的 severe early-task erosion；
9. 8B 三 seed 保持主要 effect direction。

这时论文真正的新知识应表述为：

> **即时 saliency 不足以评价结构可塑性。对于固定容量的 N:M topology graph，可以用少量真实 aggregate counterfactual interventions 学到具有 integration-horizon predictivity 的结构动作值；当局部 interaction 可控时，这些 action values 能经 task-local calibration 压缩为跨任务 structural-state preference，并改善 continual topology adaptation。**

这个表述比「六状态搜索」或「更复杂 pruning score」更不可约。

---

## 67. Failure Interpretation

### Failure A — critic 只能预测 H=8，不能预测 H=64

结论：

> horizon critic 只具有短期局部有效性。

不得建立 long-horizon mechanism claim。

### Failure B — critic calibration 强，但 final CL 不优于 LFS

结论：

> structural action predictivity does not translate into continual benefit.

方法论文接受概率显著下降，但可保留机制分析价值。

### Failure C — aggregate interaction 太强

若：

$$
\Gamma(8)>0.50
$$

或 micro-bundle ranking reversal 大量出现：

> aggregate bundle equations 不能可靠分解为 per-action additive value。

不得继续把 $\theta^\top\phi(a)$ 解释成经过验证的 per-action structural value。

### Failure D — task-local residual 无帮助

若 global-only 与 full 等价：

> cross-task critic prior 已足够，task-local residual 不是必要机制。

保留实现可简化，不把 local residual 写成贡献。

### Failure E — structural memory write/future calibration 失败

若：

$$
\rho_{write}\approx0
$$

则 memory 写入本身缺乏 empirical grounding。

若：

$$
\rho_{write}>0,\quad \rho_R\approx0
$$

则说明 task-local preference 随后续训练失去 historical predictive value。

两种情况必须区分。

### Failure F — early-task erosion

如果 overall ACC / mean Forgetting 尚可，但 earliest tasks 系统性恶化：

> equal-task mean structural memory trades worst-case retention for bounded plasticity.

必须报告，不得被平均指标掩盖。

### Failure G — 8B 方向反转

明确记录 scale limitation。

---

## 68. 最终 Checklist

### Core Novelty

- [ ] 6 legal topology states/group
- [ ] Johnson $J(4,2)$ one-swap action graph
- [ ] no atomic two-swap
- [ ] v1.0 one-step/Fisher reduced to LFS
- [ ] 8 aggregate shadow bundles/event
- [ ] H=8 horizon supervision
- [ ] global transferable prior
- [ ] task-local residual critic
- [ ] bounded structural-state memory R
- [ ] no P/H in full method

### Correctness

- [ ] exact 2:4 after every graph transition
- [ ] all 4 graph neighbors/state tested
- [ ] branch isolation
- [ ] optimizer/scheduler/RNG restore
- [ ] deterministic shadow replay check
- [ ] prefix-cache parity
- [ ] dormant W/m/v zero
- [ ] global/local critic merge test

### Primary Evidence

- [ ] TRACE structural 5 seeds
- [ ] CounterStruct vs LFS
- [ ] CounterStruct vs SRigL
- [ ] practical effect threshold
- [ ] AG non-inferiority
- [ ] macro-horizon bridge

### Temporal / Credit Calibration

- [ ] T4@60%
- [ ] T7@60%
- [ ] H=1/8/32/64
- [ ] K=64 primary
- [ ] K=256 secondary
- [ ] critic vs one-step
- [ ] critic vs RigL
- [ ] rho_critic(64) with bootstrap CI
- [ ] direct micro-bundle interaction audit
- [ ] interaction Gamma reported

### Historical Memory

- [ ] task-local write uses current-task critic
- [ ] rho_write
- [ ] rho_R
- [ ] full vs w/o R
- [ ] full vs Fisher memory
- [ ] early-task retention reported
- [ ] R persistent bytes ≤0.7GB
- [ ] growth/task = 0

### Robustness

- [ ] TRACE Order-2
- [ ] Seq-GLUE
- [ ] Long-CL-15
- [ ] Qwen3-8B 3 seeds
- [ ] zero retuning

### External Comparators

- [ ] OSFT
- [ ] Meta-UCF
- [ ] PaRSP
- [ ] Any-SSR
- [ ] O-LoRA
- [ ] LoRA
- [ ] Naive FT
- [ ] official-code audit
- [ ] replay audit
- [ ] task-ID/router audit
- [ ] contamination audit

### Resource Honesty

- [ ] no sparse-training speed claim
- [ ] 8-bundle shadow overhead reported
- [ ] peak HBM
- [ ] persistent-state bytes
- [ ] replay bytes
- [ ] memory-matched replay secondary baseline

### Claim Boundary

- [ ] no “first dynamic sparsity”
- [ ] no “first lookahead”
- [ ] no “Johnson graph is novelty”
- [ ] no “bundle additivity is guaranteed”
- [ ] no “R is true past loss landscape”
- [ ] no exact future utility claim
- [ ] no 8B scaling-law claim
- [ ] final 2026 novelty sweep before submission

---

## 69. 最终项目定义

CounterStruct v2.1 的核心不是：

> 「给 prune/grow 一个更复杂的 score。」

也不是：

> 「因为 2:4 只有六个 state，所以枚举它们。」

最终统一主线为：

$$
\boxed{
\begin{aligned}
&\text{在 exact-2:4 topology graph 上，}\\
&\text{用少量真实 shadow interventions 学习 integration-horizon action ranking，}\\
&\text{直接审计 aggregate credit 的 interaction 假设，}\\
&\text{再用 task-localized structural preferences 构造 bounded continual memory，}\\
&\text{在固定 active connectivity capacity 下做保守 topology adaptation。}
\end{aligned}
}
$$

论文标题、abstract、method、Figure 2、macro-horizon ablation 与 claim boundary 必须全部围绕：

$$
\boxed{
\textbf{horizon-calibrated structural action value}
}
$$

这一不可约核心，而不是围绕 dynamic sparsity、six-state enumeration 或 Fisher replacement。
