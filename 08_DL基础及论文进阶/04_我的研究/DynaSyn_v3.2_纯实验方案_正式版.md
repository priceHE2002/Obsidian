# DynaSyn v3.2 纯实验方案（正式版）

## 1. 项目定义

**DynaSyn: Counterfactual Structural Plasticity for Fixed-Capacity Continual Learning in Large Language Models**

DynaSyn 在固定 active connectivity capacity 下，对 Transformer FFN 的 connectivity graph 进行动态 exact contiguous 2:4 重构。

模型状态：

$$
(W_t,M_t)\rightarrow(W_{t+1},M_{t+1}),
$$

其中：

$$
M_t\in\mathcal M_{2:4},
\qquad
\|M_t\|_0=C,\quad \forall t.
$$

整个实验不增加 active connection 数，只允许 connectivity redistribution。

## 1.1 实验主问题与主证据链

v3.2 的核心实验不以“动态稀疏本身”为主要贡献，而围绕两个必须被直接检验的机制问题展开：

1. **完整结构动作的反事实评分是否优于分离式 prune / grow heuristic？** 主要通过 DynaSyn-v3.2、DynaSyn-v2.2-NR 与 SRigL-style Exact-2:4 的同预算比较回答。
2. **反事实结构效用是否具有可测量的 empirical predictive value？** 主要通过 multi-scale mutation bundle calibration 回答，并显式测量这种预测能力随 intervention scale 增大而退化的 locality range。

在上述主线之外，v3.2 进一步检验：

- fixed active connectivity capacity 下能否在 no-replay continual learning 中改善 stability，同时不显著牺牲 acquisition；
- historical quadratic memory 随任务数增长后，是否出现 precision accumulation 导致的 structural freezing；
- 主要效应是否跨 benchmark、task order、backbone 保持方向一致；
- learned exact-2:4 topology 是否能转换为真实 semi-structured sparse representation，并在真实矩阵 shape 上通过 runtime validation。

主要机制链固定为：

$$
\boxed{
\text{complete-action counterfactual scoring}
\rightarrow
\text{local empirical calibration}
\rightarrow
\text{fixed-capacity continual behavior}
}
$$

允许的核心解释：

$$
\boxed{
U_{p\rightarrow g}
\text{ 是 optimizer-aware local counterfactual loss surrogate}
}
$$

禁止把实验解释为：

- 首次提出 dynamic sparsity；
- exact future accuracy gain 或 guaranteed loss decrease；
- sparse-training speedup；
- 相同比例参数更新的 scaling law；
- 仅凭单个 Qwen3-8B seed 宣称方法“scales to 8B”。

---

# 2. 硬件与模型

## 2.1 Primary Backbone

$$
\boxed{\text{Qwen3-1.7B}}
$$

关键配置：

$$
d_{\mathrm{model}}=2048,
\qquad
d_{\mathrm{ffn}}=6144,
\qquad
L=28.
$$

修改范围：

```text
Layer 0–19   Frozen
Layer 20–27  Attention   Frozen
             gate_proj   Frozen
             up_proj     DynaSyn
             down_proj   DynaSyn
```

每层目标矩阵：

$$
W_{\mathrm{up}}
\in
\mathbb R^{6144\times2048},
$$

$$
W_{\mathrm{down}}
\in
\mathbb R^{2048\times6144}.
$$

8 层总 candidate coordinates：

$$
8\times2\times2048\times6144
=
\boxed{201,326,592}.
$$

Exact 2:4 下 active coordinates：

$$
\boxed{100,663,296}.
$$

---

## 2.2 Secondary Backbone

$$
\boxed{\text{Llama-3.2-1B-Instruct}}
$$

关键配置：

$$
d_{\mathrm{model}}=2048,
\qquad
d_{\mathrm{ffn}}=8192,
\qquad
L=16.
$$

修改最后 6 层：

```text
Layer 0–9    Frozen
Layer 10–15  Attention   Frozen
             gate_proj   Frozen
             up_proj     DynaSyn
             down_proj   DynaSyn
```

总 candidate coordinates：

$$
6\times2\times2048\times8192
=
\boxed{201,326,592}.
$$

Exact 2:4 下 active coordinates：

$$
\boxed{100,663,296}.
$$

---

## 2.3 Scale-Validation Backbone

$$
\boxed{\text{Qwen3-8B}}
$$

关键配置：

$$
d_{\mathrm{model}}=4096,
\qquad
d_{\mathrm{ffn}}=12288,
\qquad
L=36.
$$

只修改最后 2 层：

```text
Layer 0–33   Frozen
Layer 34–35  Attention   Frozen
             gate_proj   Frozen
             up_proj     DynaSyn
             down_proj   DynaSyn
```

每层目标矩阵：

$$
W_{\mathrm{up}}
\in
\mathbb R^{12288\times4096},
$$

$$
W_{\mathrm{down}}
\in
\mathbb R^{4096\times12288}.
$$

2 层总 candidate coordinates：

$$
2\times2\times4096\times12288
=
\boxed{201,326,592}.
$$

Exact 2:4 下 active coordinates：

$$
\boxed{100,663,296}.
$$

因此三个 structural backbones 均保持完全相同的：

$$
\boxed{201,326,592\text{ candidate coordinates}}
$$

与：

$$
\boxed{100,663,296\text{ active coordinates}}.
$$

Qwen3-8B 只用于 mechanism scale-direction validation，不参与 DynaSyn hyperparameter development。

---

# 3. Exact Contiguous 2:4 Connectivity

对 PyTorch Linear：

$$
W\in
\mathbb R^{\mathrm{out\_features}\times\mathrm{in\_features}},
$$

固定沿：

$$
\boxed{\mathrm{weight.shape[-1]}}
$$

即 K / `in_features` dimension 分组。

实现必须等价于：

```python
assert W.ndim == 2
out_features, in_features = W.shape
assert in_features % 4 == 0

W_grouped = W.reshape(
    out_features,
    in_features // 4,
    4
)
```

每个 group：

$$
q=[w_1,w_2,w_3,w_4]
$$

始终满足：

$$
\boxed{
\sum_{i\in q}M_i=2.
}
$$

设：

$$
A(q)=\{a,b\}
$$

为两个 active coordinates，

$$
D(q)=\{c,d\}
$$

为两个 dormant coordinates。

则一个 group 的合法 one-swap action 最多为：

$$
a\rightarrow c,
\quad
a\rightarrow d,
\quad
b\rightarrow c,
\quad
b\rightarrow d.
$$

因此：

$$
\boxed{
|\mathcal N_1(M_q)|\le4.
}
$$

每次 mutation event 都穷举 group 内所有合法 one-swap actions。

---

# 4. Initial Topology

加载完整 pretrained checkpoint 后，对每个 contiguous 4-group：

1. 计算四个 pretrained weights 的 $|w|$；
2. 保留 magnitude 最大的两个；
3. 其余两个设 `mask=0`；
4. dormant weight tensor value 置 0；
5. active cooldown state 初始化为 $C=2$；
6. dormant cooldown state 初始化为 $C=0$。

因此：

$$
M_0
=
\text{top-2-of-4 magnitude mask}.
$$

所有 structural methods 使用完全相同的：

$$
\boxed{M_0}.
$$

Historical state 初始化：

$$
\boxed{
P=0,\qquad H=0.
}
$$

---

# 5. Persistent State

DynaSyn v3.2 维护：

$$
\boxed{
W,M,P,H,C,m,v
}
$$

其中：

- $W$：dense storage weight tensor；
- $M$：exact-2:4 binary mask；
- $P$：historical quadratic precision；
- $H$：historical precision-weighted center statistic；
- $C$：rewiring-event cooldown；
- $m,v$：AdamW first/second moments。

Historical preferred center：

$$
\boxed{
\mu_i
=
\frac{H_i}{P_i+\epsilon_P}.
}
$$

默认：

$$
\boxed{
\epsilon_P=10^{-12}.
}
$$

---

# 6. Replay Protocol

## 6.1 Primary Setting

主实验：

$$
\boxed{\text{No Replay}}
$$

学习 task $T_{t+1}$ 时，不访问：

$$
T_1,\dots,T_t
$$

的 historical training samples。

不保存：

- replay exemplars；
- historical logits；
- historical gradients；
- 每任务 Fisher tensor；
- 每任务 checkpoint。

Task $T_t$ 结束后，仅使用当前 task 数据完成一次 historical consolidation。

consolidation 完成后删除该 task 的 consolidation examples。

未来任务只携带：

$$
\boxed{
W,M,P,H,C,\text{AdamW state}.
}
$$

---

## 6.2 Replay Complementarity

Secondary experiment：

$$
\boxed{
\text{DynaSyn-v3.2 + 64 exemplars/task}
}
$$

当前训练 batch 中 replay 比例约：

$$
10\%.
$$

该实验不替代 replay-free 主结果。

---

# 7. Normal Training Step

有效权重：

$$
W_{\mathrm{eff}}
=
M\odot W.
$$

Forward：

$$
y=W_{\mathrm{eff}}x.
$$

训练阶段仍使用 dense masked Linear：

$$
\boxed{\text{dense forward/backward compute}}.
$$

Backward 后：

$$
\boxed{
\nabla W
\leftarrow
M\odot\nabla W.
}
$$

Dormant coordinates 无 parameter-update eligibility。

使用标准 AdamW。

Normal step invariant：

$$
M_i=0
\Rightarrow
w_i=0,
$$

$$
M_i=0
\Rightarrow
m_i=v_i=0.
$$

---

# 8. Mutation Schedule

每个 task：

$$
\boxed{
K_{\mathrm{rewire}}=4.
}
$$

若总 optimizer steps 为：

$$
T_{\mathrm{steps}},
$$

则 mutation event：

$$
\boxed{
r_k
=
\left\lfloor
\frac{k}{5}T_{\mathrm{steps}}
\right\rfloor,
\qquad
k=1,2,3,4.
}
$$

即约：

$$
\boxed{
20\%,40\%,60\%,80\%.
}
$$

不执行 task-final mutation。

最后 20% optimizer steps 保留为最后一批 newborn connections 的 integration window。

---

# 9. Cooldown

定义：

$$
\boxed{
C_i\in\{0,1,2\}.
}
$$

含义：

- $C=0$：newborn；
- $C=1$：跨过一个 rewiring interval；
- $C=2$：mature。

每次 mutation event 开始时，对 active coordinates：

$$
\boxed{
C
\leftarrow
\min(C+1,2).
}
$$

Prune eligibility：

$$
\boxed{
C_p=2.
}
$$

Grow 后：

$$
C_g\leftarrow0.
$$

---

# 10. Current-Task Dense Counterfactual Probe

每个 mutation event 使用当前 task：

$$
\boxed{
|B_{\mathrm{probe}}|=32.
}
$$

固定拆为：

$$
4\times8
$$

microbatches。

构造 STE weight：

$$
\boxed{
W_{\mathrm{STE}}
=
W+
\operatorname{stopgrad}
(
M\odot W-W
).
}
$$

Forward 仍看到：

$$
M\odot W,
$$

但 backward 能得到 active 与 dormant candidate coordinates 的 signed gradients。

对 probe microbatches 求平均 signed gradient：

$$
\boxed{
g_i
=
\frac{\partial L_{\mathrm{current}}}
{\partial w_i}.
}
$$

Probe 规则：

1. current-task only；
2. no optimizer step；
3. no real Adam state write；
4. no parameter update；
5. 保留 gradient sign；
6. probe batch 不作为紧随其后的 normal optimizer batch；
7. probe 结束释放 dense gradient buffer。

---

# 11. Historical Dense Consolidation Probe

Task $T_t$ 结束后，从当前 task 取：

$$
\boxed{
|B_t^{\mathrm{hist}}|=64.
}
$$

固定拆成：

$$
8
$$

个不重叠 microbatches：

$$
B_1,\dots,B_8,
\qquad
|B_b|=8.
$$

同样使用 STE：

$$
W_{\mathrm{STE}}
=
W+
\operatorname{stopgrad}
(
M\odot W-W
).
$$

对 microbatch $b$：

$$
g_b^{(t)}
=
\nabla_W L_t(B_b).
$$

定义 candidate-coordinate historical sensitivity：

$$
\boxed{
F_i^{(t)}
=
\frac18
\sum_{b=1}^{8}
\left(
g_{b,i}^{(t)}
\right)^2.
}
$$

平方为 element-wise square。

禁止使用：

$$
\left(
\frac18\sum_{b=1}^{8}g_b
\right)^2
$$

替代：

$$
\frac18\sum_{b=1}^{8}g_b^2.
$$

Historical sensitivity 必须覆盖：

$$
\boxed{\text{active + dormant coordinates}}.
$$

---

# 12. Historical Quadratic Memory Update

Task $t$ 结束时：

$$
\boxed{
P_i^{(t)}
=
P_i^{(t-1)}
+
F_i^{(t)}.
}
$$

定义 task-end effective coordinate：

$$
\boxed{
w_i^{(t)}
=
(M\odot W)_i.
}
$$

然后：

$$
\boxed{
H_i^{(t)}
=
H_i^{(t-1)}
+
F_i^{(t)}
w_i^{(t)}.
}
$$

Historical center：

$$
\boxed{
\mu_i^{(t)}
=
\frac{H_i^{(t)}}
{P_i^{(t)}+\epsilon_P}.
}
$$

---

# 13. Historical Quadratic Compression

对 coordinate $i$，过去 task 的 diagonal local approximation：

$$
Q_{\tau,i}(w)
=
\frac12
F_i^{(\tau)}
\left(
w-w_i^{(\tau)}
\right)^2.
$$

历史累计：

$$
Q_{\mathrm{hist},i}(w)
=
\frac12
\sum_{\tau<t}
F_i^{(\tau)}
\left(
w-w_i^{(\tau)}
\right)^2.
$$

定义：

$$
P_i
=
\sum_{\tau<t}
F_i^{(\tau)},
$$

$$
H_i
=
\sum_{\tau<t}
F_i^{(\tau)}w_i^{(\tau)},
$$

$$
\mu_i
=
\frac{H_i}
{P_i+\epsilon_P}.
$$

则：

$$
\boxed{
Q_{\mathrm{hist},i}(w)
=
\frac12
P_i
(w-\mu_i)^2
+
C_i^{\mathrm{const}}.
}
$$

后续 mutation comparison 中常数项抵消。

---

# 14. Historical State Lifecycle

v3.2 中：

$$
P,H
$$

定义为 coordinate-level historical loss memory。

因此：

$$
\boxed{
\text{prune/grow 不 reset }P,H.
}
$$

Topology transition 只 reset：

$$
\boxed{
w,m,v,C.
}
$$

Dormant coordinate 允许：

$$
\boxed{
P_i>0
}
$$

或：

$$
\boxed{
H_i\neq0.
}
$$

---

# 15. Counterfactual Structural Action

每个合法 action：

$$
\boxed{
a=(p\rightarrow g)
}
$$

其中：

- $p$：mature active coordinate；
- $g$：同一 exact-2:4 group 的 dormant coordinate。

比较：

### Keep World

Topology 不改变。

$p$ 继续 active，$g$ 继续 dormant。

### Swap World

执行：

$$
p\rightarrow0,
$$

并：

$$
g\rightarrow\text{active}.
$$

然后比较两个 world 的 one-step optimizer-aware local surrogate。

---

# 16. Hypothetical AdamW Update

设当前真实 optimizer global step 为：

$$
s.
$$

对 coordinate $i$：

$$
m_i^+
=
\beta_1m_i
+
(1-\beta_1)g_i,
$$

$$
v_i^+
=
\beta_2v_i
+
(1-\beta_2)g_i^2.
$$

Bias-corrected：

$$
\hat m_i^+
=
\frac{m_i^+}
{1-\beta_1^{s+1}},
$$

$$
\hat v_i^+
=
\frac{v_i^+}
{1-\beta_2^{s+1}}.
$$

Hypothetical AdamW increment：

$$
\boxed{
\delta_i
=
-\eta_s
\frac{\hat m_i^+}
{\sqrt{\hat v_i^+}+\epsilon_{\mathrm{Adam}}}
-
\eta_s\lambda_{\mathrm{wd}}w_i.
}
$$

该计算只用于 scoring：

$$
\boxed{\text{禁止写入真实 optimizer state}}.
$$

---

# 17. Keep-World Active Update

对 prune candidate $p$，使用当前：

$$
w_p,m_p,v_p
$$

与 probe gradient：

$$
g_p
$$

计算：

$$
\boxed{
\delta_p^{\mathrm{keep}}.
}
$$

Keep world 下一步 active value：

$$
\boxed{
w_p^{K}
=
w_p+\delta_p^{\mathrm{keep}}.
}
$$

Dormant $g$：

$$
\boxed{
w_g^K=0.
}
$$

---

# 18. Swap-World Newborn Update

执行 topology swap 后：

$$
w_p^S=0.
$$

对于 newborn $g$，transition state 为：

$$
w_g=0,
\qquad
m_g=0,
\qquad
v_g=0.
$$

但 AdamW global step 仍为：

$$
s+1.
$$

使用：

$$
g_g
$$

计算：

$$
\boxed{
\delta_g^{\mathrm{new}}.
}
$$

Swap world：

$$
\boxed{
w_g^S
=
\delta_g^{\mathrm{new}}.
}
$$

---

# 19. Predicted Current-Task Benefit

Keep world：

$$
L_{\mathrm{keep}}
\approx
L_{\mathrm{cur}}
+
g_p
\delta_p^{\mathrm{keep}}.
$$

Swap world：

$$
L_{\mathrm{swap}}
\approx
L_{\mathrm{cur}}
-
g_pw_p
+
g_g
\delta_g^{\mathrm{new}}.
$$

因此定义 swap 相对 keep 的 predicted current-task benefit：

$$
\boxed{
B_{\mathrm{cur}}(p\rightarrow g)
=
g_p
\left(
w_p+\delta_p^{\mathrm{keep}}
\right)
-
g_g
\delta_g^{\mathrm{new}}.
}
$$

若：

$$
B_{\mathrm{cur}}>0,
$$

则在当前 one-step local surrogate 下，swap 对 current task 优于 keep。

---

# 20. Predicted Historical Damage

## 20.1 Prune-Side Historical Change

Keep world：

$$
w_p^K
=
w_p+\delta_p^{\mathrm{keep}}.
$$

Swap world：

$$
w_p^S=0.
$$

因此：

$$
\boxed{
D_p
=
\frac12P_p
\left[
\mu_p^2
-
\left(
w_p+\delta_p^{\mathrm{keep}}-\mu_p
\right)^2
\right].
}
$$

---

## 20.2 Grow-Side Historical Change

Keep world：

$$
w_g^K=0.
$$

Swap world：

$$
w_g^S
=
\delta_g^{\mathrm{new}}.
$$

因此：

$$
\boxed{
D_g
=
\frac12P_g
\left[
\left(
\delta_g^{\mathrm{new}}-\mu_g
\right)^2
-
\mu_g^2
\right].
}
$$

---

## 20.3 Total Historical Damage

$$
\boxed{
D_{\mathrm{hist}}(p\rightarrow g)
=
D_p+D_g.
}
$$

允许：

$$
D_{\mathrm{hist}}<0.
$$

此时 local historical surrogate 预测该 mutation 可能改善历史任务局部状态。

---

# 21. Unified Counterfactual Mutation Utility

定义：

$$
\boxed{
U_{p\rightarrow g}
=
B_{\mathrm{cur}}(p\rightarrow g)
-
D_{\mathrm{hist}}(p\rightarrow g).
}
$$

解释限定为：

$$
\boxed{
\text{optimizer-aware local counterfactual loss surrogate}.
}
$$

禁止解释为：

- exact future accuracy gain；
- exact expected utility；
- economic utility；
- guaranteed loss decrease。

Admission threshold：

$$
\boxed{
U_{p\rightarrow g}>0.
}
$$

只表示：

> 在当前冻结的 local surrogate 下，预测 swap 优于 keep。

---

# 22. Group-Wise Exhaustive Search

对每个 exact-2:4 group：

$$
\mathcal A_q
=
\{
(p,g):
p\in Active(q),
g\in Dormant(q),
C_p=2
\}.
$$

最多：

$$
|\mathcal A_q|=4.
$$

对所有 action 计算：

$$
U_{p\rightarrow g}.
$$

选：

$$
\boxed{
a_q^*
=
\arg\max_{a\in\mathcal A_q}
U_a.
}
$$

并定义：

$$
\boxed{
U_q^*
=
U(a_q^*).
}
$$

无合法 mature prune candidate：

$$
U_q^*=-\infty.
$$

---

# 23. Global Mutation Admission

最大 mutation budget：

$$
\boxed{
\rho_{\max}.
}
$$

Development-only 搜索：

$$
\boxed{
\rho_{\max}
\in
\{0.5\%,1\%,2\%\}.
}
$$

候选 groups：

$$
\boxed{
\mathcal C
=
\{
q:
U_q^*>0
\}.
}
$$

最终：

$$
\boxed{
\mathcal Q'
=
\operatorname{TopU}
\left(
\mathcal C,
\left\lfloor
\rho_{\max}|\mathcal Q|
\right\rfloor
\right).
}
$$

Realized mutation ratio：

$$
\boxed{
\rho^{\mathrm{realized}}
=
\frac{|\mathcal Q'|}
{|\mathcal Q|}
\le
\rho_{\max}.
}
$$

---

# 24. Topology Transition

## 24.1 Prune

对 $p$：

$$
M_p\leftarrow0,
$$

$$
w_p\leftarrow0,
$$

$$
m_p\leftarrow0,
$$

$$
v_p\leftarrow0,
$$

$$
C_p\leftarrow0.
$$

保持：

$$
\boxed{
P_p,H_p\text{ unchanged}.
}
$$

---

## 24.2 Grow

对 $g$：

$$
M_g\leftarrow1,
$$

$$
w_g\leftarrow0,
$$

$$
m_g\leftarrow0,
$$

$$
v_g\leftarrow0,
$$

$$
C_g\leftarrow0.
$$

保持：

$$
\boxed{
P_g,H_g\text{ unchanged}.
}
$$

---

# 25. Grow Initialization

Primary：

$$
\boxed{
w_g=0.
}
$$

Gradient-imprinted initialization 只作为 appendix ablation。

---

# 26. DynaSyn v3.2 Pseudocode

```text
Input:
    pretrained model
    tasks T1 ... TT
    exact contiguous 2:4 groups Q
    K_rewire = 4

State:
    W, M
    AdamW m, v
    historical P, H
    cooldown C

Initialize:
    build top-2-of-4 M0
    dormant W = 0
    P = 0
    H = 0
    active C = 2
    dormant C = 0

For each task t:

    train with CURRENT TASK ONLY

    compute mutation events:
        20%, 40%, 60%, 80%

    for optimizer step s:

        W_eff = M * W

        loss = answer-only SFT loss

        dense backward

        grad(W) = M * grad(W)

        AdamW.step()

        if s is mutation event:

            C(active) = min(C(active)+1, 2)

            # Current-task probe
            sample 32 current-task examples
            split 4 x 8
            use STE
            obtain signed dense g

            for every exact-2:4 group q:

                enumerate all legal mature p -> g actions

                for each action:

                    compute hypothetical
                        delta_p_keep

                    compute hypothetical
                        delta_g_new

                    B_cur =
                        g_p * (w_p + delta_p_keep)
                        - g_g * delta_g_new

                    mu_p = H_p / (P_p + eps_P)
                    mu_g = H_g / (P_g + eps_P)

                    D_p =
                        0.5 * P_p *
                        [mu_p^2
                         - (w_p + delta_p_keep - mu_p)^2]

                    D_g =
                        0.5 * P_g *
                        [(delta_g_new - mu_g)^2
                         - mu_g^2]

                    D_hist = D_p + D_g

                    U = B_cur - D_hist

                choose highest-U legal action

            retain groups with U_best > 0

            select global top-U groups,
                capped by rho_max

            execute swaps

            reset W/m/v/C
                on transitioned coordinates

            DO NOT reset P/H

            assert exact 2:4 legality
            assert dormant W/m/v == 0

            release probe gradients

    # task-end consolidation
    sample 64 current-task examples
    split into 8 x 8

    use STE gradients

    F = mean(g_b ** 2)

    W_eff = M * W

    P = P + F
    H = H + F * W_eff

    discard consolidation examples

    evaluate all benchmark tasks
```

---

# 27. Benchmark A — Original TRACE-8

任务：

$$
\boxed{
\begin{aligned}
\text{C-STANCE}
\rightarrow&
\text{FOMC}
\rightarrow
\text{MeetingBank}\\
\rightarrow&
\text{Py150}
\rightarrow
\text{ScienceQA}\\
\rightarrow&
\text{NumGLUE-cm}
\rightarrow
\text{NumGLUE-ds}\\
\rightarrow&
\text{20Minuten}.
\end{aligned}
}
$$

每 task 使用完整训练集：

$$
\boxed{5000\text{ training examples/task}}.
$$

数据 split：

- train；
- validation/eval；
- final test。

所有 development decision 只允许访问 validation/eval。

Final test 只在 full configuration freeze 后运行。

---

# 28. TRACE Training Configuration

默认：

$$
\boxed{
\text{LR}=1\times10^{-5}.
}
$$

Task epochs：

$$
\boxed{
[5,3,7,5,3,5,5,7].
}
$$

依次对应：

```text
C-STANCE    5
FOMC        3
MeetingBank 7
Py150       5
ScienceQA   3
NumGLUE-cm  5
NumGLUE-ds  5
20Minuten   7
```

其余：

- optimizer：AdamW；
- scheduler：cosine；
- warmup：0；
- weight decay：0；
- max prompt length：1024；
- max answer length：512；
- effective batch：32。

Qwen：

```python
enable_thinking=False
```

所有方法保持相同：

- tokenizer；
- chat template；
- data order；
- answer-only loss；
- evaluation script。

---

# 29. Benchmark B — Seq-GLUE-7

任务顺序：

$$
\boxed{
\begin{aligned}
\text{CoLA}
\rightarrow
\text{SST-2}
\rightarrow
\text{MRPC}
\rightarrow
\text{QQP}\\
\rightarrow
\text{QNLI}
\rightarrow
\text{RTE}
\rightarrow
\text{MNLI}.
\end{aligned}
}
$$

统一转换为 instruction-output format。

默认：

- one epoch/task；
- AdamW；
- $\beta_1=0.9$；
- $\beta_2=0.98$；
- learning rate $=3\times10^{-5}$；
- effective batch $=64$；
- max sequence $=512$；
- weight decay $=0.01$。

---

# 29A. Benchmark C — Long-CL-15 Long-Horizon Stress Test

Long-horizon 实验采用公开的 15-task continual-learning composition：

- CL benchmark：Yelp、Amazon、DBpedia、Yahoo、AG News；
- GLUE：MNLI、QQP、RTE、SST-2；
- SuperGLUE：WiC、CB、COPA、MultiRC、BoolQ；
- IMDb。

该 15-task protocol 的 task-level evaluation metric 统一为 accuracy；所有 task score 在聚合前转换到 0–100 score scale。

Primary long-horizon order 固定为公开 Order-4：

$$
\boxed{
\begin{aligned}
\text{MNLI}\rightarrow\text{CB}\rightarrow\text{WiC}\rightarrow\text{COPA}\rightarrow\text{QQP}
\rightarrow\text{BoolQ}\rightarrow\text{RTE}\rightarrow\text{IMDb}\\
\rightarrow\text{Yelp}\rightarrow\text{Amazon}\rightarrow\text{SST-2}\rightarrow\text{DBpedia}
\rightarrow\text{AG News}\rightarrow\text{MultiRC}\rightarrow\text{Yahoo}.
\end{aligned}
}
$$

该 composition 与三种长序列 order 均已有公开 continual-learning 文献协议；v3.2 的 primary long-horizon stream 使用 Order-4，不根据实验结果选择 order。

数据预算：

- 每 task 目标为最多 1000 个**不重复** training examples；
- 若官方 train split 少于 1000 个 unique examples，则使用全部 unique train examples，禁止通过重复采样凑满 1000；
- validation 优先使用公开 benchmark / official split；若复现代码明确采用 per-class holdout，则按该 protocol 固定；
- 所有实际 sample count、dataset revision、sampling RNG 写入 `longcl_manifest.yaml`；
- final test examples 不参与 development、mutation scoring、historical consolidation 或 baseline tuning。

统一转换为 instruction-output format。

Long-CL-15 的 common backbone training configuration 固定为：

- one epoch/task；
- AdamW；
- $\beta_1=0.9$；
- $\beta_2=0.98$；
- learning rate $=3\times10^{-5}$；
- effective batch $=64$；
- max sequence $=512$；
- weight decay $=0.01$。

DynaSyn-specific mutation schedule、$\rho_{\max}$、cooldown、probe budget、historical consolidation budget 与 utility formula 全部继承 TRACE Order-1 development 后冻结的配置。

**Long-CL-15 上禁止重新调 DynaSyn-specific hyperparameter。**

Long-horizon 的核心附加诊断按 task index $t$ 报告：

$$
\boxed{
\operatorname{median}(P_t),\quad
P_{90}(P_t),\quad
\rho^{\mathrm{realized}}_t,\quad
f^{U>0}_t,\quad
AG_t
}
$$

其中：

$$
f^{U>0}_t
=
\frac{\#\{q:U_q^*>0\}}
{\#\{q:\text{legal mature action exists}\}}.
$$

该实验直接检查：

$$
\boxed{
P\text{ 的累计是否导致 }f^{U>0}_t\downarrow
\text{ 与 }\rho^{\mathrm{realized}}_t\downarrow
}
$$

以及这种下降是否伴随 acquisition collapse。

Order-5 / Order-6 作为资源允许时的 secondary extension；不得根据 Order-4 结果反向选择更有利的长序列顺序。若启动 extension，顺序在 freeze 前固定为：

### Order-5

$$
\boxed{
\begin{aligned}
\text{MultiRC}\rightarrow\text{BoolQ}\rightarrow\text{WiC}\rightarrow\text{MNLI}\rightarrow\text{CB}\rightarrow\text{COPA}\rightarrow\text{QQP}\rightarrow\text{RTE}\\
\rightarrow\text{IMDb}\rightarrow\text{SST-2}\rightarrow\text{DBpedia}\rightarrow\text{AG News}\rightarrow\text{Yelp}\rightarrow\text{Amazon}\rightarrow\text{Yahoo}.
\end{aligned}
}
$$

### Order-6

$$
\boxed{
\begin{aligned}
\text{Yelp}\rightarrow\text{Amazon}\rightarrow\text{MNLI}\rightarrow\text{CB}\rightarrow\text{COPA}\rightarrow\text{QQP}\rightarrow\text{RTE}\rightarrow\text{IMDb}\\
\rightarrow\text{SST-2}\rightarrow\text{DBpedia}\rightarrow\text{AG News}\rightarrow\text{Yahoo}\rightarrow\text{MultiRC}\rightarrow\text{BoolQ}\rightarrow\text{WiC}.
\end{aligned}
}
$$

Order-5 / Order-6 仅用于 secondary order extension，不进入 DynaSyn hyperparameter selection。

---

# 29B. Benchmark D — TRACE Order-2 Robustness

第二 TRACE 顺序固定使用公开 Order-2：

$$
\boxed{
\begin{aligned}
\text{NumGLUE-cm}
\rightarrow\text{NumGLUE-ds}
\rightarrow\text{FOMC}
\rightarrow\text{20Minuten}\\
\rightarrow\text{C-STANCE}
\rightarrow\text{Py150}
\rightarrow\text{MeetingBank}
\rightarrow\text{ScienceQA}.
\end{aligned}
}
$$

数据、prompt、evaluation metric 与 Original TRACE-8 完全一致。

每个 task 的 epoch 数按 **task identity** 继承 Section 28，而不是按 position 继承：

```text
NumGLUE-cm  5
NumGLUE-ds  5
FOMC        3
20Minuten   7
C-STANCE    5
Py150       5
MeetingBank 7
ScienceQA   3
```

Order-2 只运行核心 structural methods：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2。

固定：

$$
\boxed{3\text{ seeds}:42,43,44}.
$$

禁止在 Order-2 上重新选择 target layers、$\rho_{\max}$、probe size、cooldown 或 utility formula。

定义 order-specific effect：

$$
\Delta^{(o)}_{\mathrm{v3.2-SRigL}}
=
Metric^{(o)}_{\mathrm{v3.2}}
-
Metric^{(o)}_{\mathrm{SRigL}},
\qquad o\in\{1,2\}.
$$

同时报告：

$$
\boxed{
\Delta_{\mathrm{order-avg}}
=
\frac12
\left(
\Delta^{(1)}+\Delta^{(2)}
\right)
}
$$

与：

$$
\boxed{
\Delta_{\mathrm{worst-order}}
=
\min\left(\Delta^{(1)},\Delta^{(2)}\right)
}
$$

其中 metric 方向在计算 worst-order 前统一转换为“越大越好”的 signed effect；例如 Forgetting 使用负号转换。

---

# 30. Main Experiment Matrix

## Experiment A — Primary Full Comparison

$$
\boxed{
\text{Qwen3-1.7B}
\times
\text{TRACE-8 Order-1}
}
$$

Structural methods：

- Dense Regional FT；
- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2。

External continual-learning methods：

- Naive FT；
- LoRA；
- O-LoRA；
- Meta-UCF；
- **OSFT (Orthogonal Subspace Fine-Tuning)**；
- GORP；
- TreeLoRA；
- Any-SSR；
- PaRSP。

Seed policy：

- Static / SRigL / DynaSyn-v2.2-NR / DynaSyn-v3.2：
  $$
  \boxed{5\text{ seeds}:42,43,44,45,46}
  $$
- Dense Regional FT 与所有 external baselines：
  $$
  \boxed{3\text{ seeds}:42,43,44}
  $$

DynaSyn 与 external baseline 的 paired contrast 只使用公共 seed subset：

$$
\boxed{\{42,43,44\}}.
$$

该实验是唯一 full-comparison stream，也是 fixed-capacity structural claim 的 primary performance stream。

---

## Experiment B — Benchmark Generalization

$$
\boxed{
\text{Qwen3-1.7B}
\times
\text{Seq-GLUE-7}
}
$$

至少运行：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2；
- LoRA；
- O-LoRA；
- Meta-UCF；
- OSFT；
- Any-SSR；
- PaRSP。

全部：

$$
\boxed{3\text{ seeds}:42,43,44}.
$$

GORP / TreeLoRA 在该 benchmark 上作为 secondary extension，不用于 benchmark-generalization gate。

---

## Experiment C — Long-Horizon Continual Learning

$$
\boxed{
\text{Qwen3-1.7B}
\times
\text{Long-CL-15 Order-4}
}
$$

Required methods：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2；
- O-LoRA；
- Meta-UCF；
- OSFT。

固定：

$$
\boxed{3\text{ seeds}:42,43,44}.
$$

该实验不用于重新开发 DynaSyn，只检验：

- stability 随 task horizon 增长是否维持；
- historical precision accumulation 是否导致 structural freezing；
- constant-size $(P,H)$ memory 是否在 15-task stream 中保持可用 plasticity。

---

## Experiment D — Backbone Generalization

$$
\boxed{
\text{Llama-3.2-1B-Instruct}
\times
\text{TRACE-8 Order-1}
}
$$

至少运行：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2；
- O-LoRA；
- Meta-UCF；
- OSFT；
- Any-SSR。

全部：

$$
\boxed{3\text{ seeds}:42,43,44}.
$$

---

## Experiment E — TRACE Task-Order Robustness

$$
\boxed{
\text{Qwen3-1.7B}
\times
\text{TRACE-8 Order-2}
}
$$

只运行：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2。

固定：

$$
\boxed{3\text{ seeds}:42,43,44}.
$$

该实验是 zero-retuning order transfer test。

---

## Experiment F — 8B Mechanism Scale-Direction Validation

$$
\boxed{
\text{Qwen3-8B}
\times
\text{TRACE-8 Order-1}
}
$$

只运行：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v3.2。

固定：

$$
\boxed{\text{seed}=42}.
$$

使用完整 TRACE-8 task stream，不缩短 task 数。

除 architecture-induced target-layer mapping 外，全部继承已经 freeze 的 Qwen3-1.7B DynaSyn 配置：

- learning rate；
- mutation schedule；
- $\rho_{\max}$；
- cooldown；
- current probe budget；
- historical consolidation budget；
- utility formula；
- admission rule。

Experiment F：

$$
\boxed{\text{禁止重新调参}}.
$$

该实验只控制 absolute structural search-space size 与 active capacity，与 1.7B / Llama structural setup 保持同量级 candidate coordinates。

它**不**控制 target parameters 占整个模型参数的比例，也不是 matched-fraction scaling study。

该实验唯一回答：

$$
\boxed{
\text{同一 counterfactual structural mechanism 在更大 backbone 上是否保持 effect direction}
}.
$$

禁止据此单独宣称：

- scaling law；
- 8B SOTA continual learning；
- full-model 8B structural adaptation；
- 相同比例参数更新下的可扩展性。

---

# 31. Structural Baselines

## B0 — Dense Regional FT

只训练与 DynaSyn 相同的 target FFN matrices，但不施加 sparsity。

角色：

$$
\boxed{\text{100% capacity reference}}.
$$

---

## B1 — Static Exact-2:4

保持：

$$
\boxed{
M_t=M_0.
}
$$

只更新 active weights。

---

## B2 — SRigL-style Exact-2:4

共享：

- same $M_0$；
- same mutation schedule；
- same mutation budget；
- same current-task dense probe；
- same zero initialization；
- same optimizer transition reset。

Prune：

$$
\arg\min |w|.
$$

Grow：

$$
\arg\max |g|.
$$

---

## B3 — DynaSyn-v2.2-NR

使用旧版 heuristic：

- active survival score；
- dormant grow gradient；
- score normalization；
- heuristic $R_q$；
- cooldown。

但：

$$
\boxed{\text{No Replay}}
$$

以保证与 v3.2 的 replay policy 一致。

---

# 32. External Continual-Learning Baselines

## 32.1 Required Recent Comparator Suite

Experiment A 必须包含：

- Naive FT；
- LoRA；
- O-LoRA；
- Meta-UCF；
- **OSFT (Orthogonal Subspace Fine-Tuning)**；
- GORP；
- TreeLoRA；
- Any-SSR；
- PaRSP。

其中：

- **OSFT** 是 fixed-parameter-count / constrained full fine-tuning 的 primary recent contrast；
- Meta-UCF 是 memory-constant task-conditioned adapter contrast；
- Any-SSR 是 TRACE / large-model continual-learning contrast；
- PaRSP 是 sparse-region / rehearsal-free contrast；
- GORP 与 TreeLoRA 提供 continual optimization / adapter-structure contrasts；
- O-LoRA 提供 long-sequence orthogonal-subspace protocol anchor。

OSFT 的 official implementation audit 至少记录：

- adaptive SVD 分解对象；
- protected / trainable subspace 的 rank 或 energy rule；
- subspace recomputation frequency；
- gradient projection / update constraint；
- persistent state；
- 是否随 task 数增长保存 task-specific state；
- common-backbone port 后的 trainable / actually updated parameters。

任何方法是否属于 replay-free，不根据论文标题或口头描述决定，而根据实际运行代码审计：

$$
\boxed{
\text{只要训练时访问 historical task samples，即 Replay}=1.
}
$$

---

## 32.2 Common-Protocol Porting

Experiment A 中所有 external baselines 必须在：

$$
\boxed{
\text{Qwen3-1.7B}\times\text{TRACE-8 Order-1}
}
$$

上重新运行，不直接复制论文 reported number。

统一：

- pretrained checkpoint revision；
- task order；
- 5000 training examples/task；
- train / validation / final-test split；
- tokenizer；
- chat template；
- answer-only evaluation target；
- evaluation scripts；
- final test freeze rule；
- seeds 42/43/44。

方法自身不可删除的机制保持原样，例如：

- adapter generation；
- low-rank projection；
- hierarchical adapter organization；
- analytic router；
- null-space / orthogonal-subspace projection；
- adaptive SVD；
- task-region mapping。

禁止为了“看起来更公平”而把 external method 改造成 DynaSyn 的 parameterization。

Long-CL-15 中也必须重新运行 selected external baselines，不复制其 T5 / 7B published numbers；long-horizon published setting 只用于固定 task composition、order 与 data-preparation reference。

---

## 32.3 Official-Code Reproduction Audit

每个 recent baseline 在 common-backbone port 前必须保存：

```text
baseline_name/
    paper_reference.txt
    repository_url.txt
    repository_commit.txt
    original_config.yaml
    ported_config.yaml
    code_diff.patch
    reproduction_notes.md
    replay_audit.json
    persistent_state_audit.json
    task_id_router_audit.json
```

执行顺序：

1. 优先使用官方实现或作者公开实现；
2. 若截至 freeze date 无公开实现，则允许 paper-faithful reimplementation，但必须单独标记 `reimplemented_from_paper=true`；
3. 先验证实现能够在其支持的 reference setup 正常训练与评价；
4. 再 port 到统一 backbone / dataset protocol；
5. 记录所有 architecture adaptation；
6. 不允许 silent modification。

OSFT 优先使用其公开 `mini_trainer` implementation；必须 pin repository commit 与 OSFT-specific config。

若 paper 与 released code 的默认值不一致，以实际使用值为准，并在 artifact 中记录。

---

## 32.4 Baseline Development Budget

external baseline 只允许使用：

$$
\boxed{\text{development seed}=42}
$$

与 validation/eval split。

每个方法：

- paper / official-code default 必须作为第一配置；
- 最多允许额外 2 个邻近配置；
- 最多只搜索 1 个主要 method-specific hyperparameter；
- 不访问 final test；
- freeze 后不得重新选择 baseline hyperparameter。

因此每个 recent baseline 的 development candidates：

$$
\boxed{\le3}.
$$

OSFT 的主要可开发 hyperparameter 只允许从其 official config 中选择一个最关键的 subspace/rank control；不得同时搜索多个 SVD / rank / layer knobs。

---

## 32.5 Fairness / Resource Reporting

所有 baseline 必须报告：

- trainable parameters；
- actually updated parameters；
- replay memory；
- historical-task sample access；
- auxiliary anchor / covariance / calibration data access；
- persistent method state；
- persistent-state growth per task；
- number of task-specific adapters / regions / protected subspaces；
- inference 是否需要 oracle task ID；
- 是否使用 learned router；
- train wall-clock；
- peak HBM；
- inference latency（若 method 改变 inference graph）。

不同方法不强行 parameter-match。

公平性通过同时报告：

$$
\boxed{
\text{performance}
+
\text{memory}
+
\text{parameter growth}
+
\text{wall-clock}
}
$$

进行 Pareto comparison。

Replay-based strong baseline 单独成表，不与 replay-free primary claim 混合。

---

## 32.6 Evaluation-Data Contamination Audit

TRACE-8 / Seq-GLUE-7 / backbone-generalization stream 中，以下 benchmark 作为 general-capability evaluation：

- MMLU；
- BBH；
- TyDiQA；
- PIQA；
- BoolQ。

在这些 stream 中，任何 external method 的：

- training data；
- replay data；
- anchor data；
- covariance estimation data；
- router fitting data；
- hyperparameter-tuning data；

都不得包含对应 general-capability evaluation examples。

若某方法原始官方配置使用其中某个 benchmark 作为 auxiliary anchor / calibration data，则 common-protocol run 必须：

1. 在 freeze 前换成预先固定的 disjoint auxiliary set；
2. 记录替换原因与 config diff；
3. 不得使用 final general-capability score 选择 auxiliary set。

若无法做到严格 disjoint，则该 benchmark 对该方法标记：

$$
\boxed{\text{CONTAMINATED / NOT COMPARABLE}}
$$

并从跨方法 $\Delta GA$ aggregate 中排除。

### Long-CL-15 特殊规则

Long-CL-15 本身包含 BoolQ，因此：

$$
\boxed{
\text{Long-CL-15 不使用 BoolQ 作为 general-capability hold-out}
}
$$

v3.2 的 primary general-capability decomposition 仍以 TRACE-8 Order-1 为准。

Long-CL-15 的核心目标是 long-horizon continual behavior 与 structural-freezing diagnosis；不得把其 task-stream BoolQ score 同时当作独立 general-capability retention evidence。

---

# 33. Evaluation Matrix

每完成 task $i$，评价全部 tasks $j$。

定义：

$$
\boxed{
A_{i,j}
=
\text{checkpoint after task }i
\text{ on task }j.
}
$$

因此得到完整：

$$
(T+1)\times T
$$

evaluation matrix。

包括：

$$
M_0
$$

以及：

$$
M_1,\dots,M_T.
$$

---

# 34. Final Average Accuracy

$$
\boxed{
ACC_{\mathrm{final}}
=
\frac1T
\sum_{j=1}^{T}
A_{T,j}.
}
$$

---

# 35. Forgetting

$$
\boxed{
F
=
\frac1{T-1}
\sum_{j=1}^{T-1}
\left[
\max_{i=j,\dots,T}A_{i,j}
-
A_{T,j}
\right].
}
$$

越低越好。

---

# 36. Acquisition Gain

对 task $j$：

$$
\boxed{
AG_j
=
A_{j,j}
-
A_{j-1,j}.
}
$$

平均：

$$
\boxed{
AG
=
\frac1T
\sum_{j=1}^{T}
AG_j.
}
$$

---

# 37. Backward Transfer

$$
\boxed{
BWT
=
\frac1{T-1}
\sum_{j=1}^{T-1}
\left(
A_{T,j}
-
A_{j,j}
\right).
}
$$

---

# 38. Forward Transfer

对 future task $j$：

$$
\boxed{
FWT_j
=
A_{j-1,j}
-
A_{0,j}.
}
$$

平均：

$$
\boxed{
FWT
=
\frac1{T-1}
\sum_{j=2}^{T}
FWT_j.
}
$$

---

# 39. Plasticity Non-Inferiority

预注册：

$$
\boxed{
\delta_{AG}=1.0
}
$$

0–100 score point。

要求：

$$
\boxed{
AG_{\mathrm{v3.2}}
\ge
AG_{\mathrm{SRigL}}
-
1.0.
}
$$

同时报告：

$$
AG_{\mathrm{v3.2}}
-
AG_{\mathrm{v2.2}}.
$$

---

# 40. General Capability Retention

至少评价：

- MMLU；
- BBH；
- TyDiQA；
- PIQA；
- BoolQ。

对 structural methods 固定检查三个 reference states：

### Dense Pretrained

原始 dense pretrained checkpoint：

$$
\boxed{W_{\mathrm{dense}}}.
$$

### Sparse Initialization

完成 top-2-of-4 initialization、尚未学习任何 continual task：

$$
\boxed{M_0}.
$$

### Final Continual State

完整 task stream 后：

$$
\boxed{M_T}.
$$

对 benchmark $k$，定义：

$$
S_{\mathrm{dense},k}
=
Score(W_{\mathrm{dense}},k),
$$

$$
S_{0,k}
=
Score(M_0,k),
$$

$$
S_{T,k}
=
Score(M_T,k).
$$

---

## 40.1 Initialization Cost

$$
\boxed{
\Delta GA_{\mathrm{init},k}
=
S_{0,k}-S_{\mathrm{dense},k}.
}
$$

用于回答：

$$
\boxed{
\text{exact-2:4 初始化本身损失了多少 general capability}
}.
$$

---

## 40.2 Continual-Stream Retention

$$
\boxed{
\Delta GA_{\mathrm{stream},k}
=
S_{T,k}-S_{0,k}.
}
$$

用于回答：

$$
\boxed{
\text{continual learning 过程在既定 sparse initialization 上又改变了多少能力}
}.
$$

---

## 40.3 End-to-End Retention

$$
\boxed{
\Delta GA_{\mathrm{total},k}
=
S_{T,k}-S_{\mathrm{dense},k}.
}
$$

并满足：

$$
\boxed{
\Delta GA_{\mathrm{total},k}
=
\Delta GA_{\mathrm{init},k}
+
\Delta GA_{\mathrm{stream},k}.
}
$$

对五个 benchmark 分别平均：

$$
\boxed{
\Delta GA_{x}
=
\frac15\sum_k\Delta GA_{x,k},
\qquad
x\in\{\mathrm{init},\mathrm{stream},\mathrm{total}\}.
}
$$

Experiment A 的 structural methods 额外在：

$$
\boxed{M_4}
$$

执行一次 mandatory mid-stream general-capability evaluation。

external non-sparse methods 以同一个：

$$
W_{\mathrm{dense}}
$$

作为起点，因此其：

$$
\Delta GA_{\mathrm{init}}=0
$$

按定义报告，仅比较 stream / total degradation。

所有 general-capability evaluation 使用固定 evaluator revision、固定 prompt/few-shot policy 与固定 decoding config；这些设置在 final stream 前冻结。

---

# 41. Counterfactual Mutation Calibration

固定在 TRACE primary stream：

$$
\boxed{
T_4\text{ 的 }60\%\text{ mutation event}
}
$$

以及：

$$
\boxed{
T_7\text{ 的 }60\%\text{ mutation event}.
}
$$

Seeds：

$$
\boxed{42,43,44}.
$$

Calibration 不用于 DynaSyn hyperparameter selection。

它只检验：

$$
\boxed{
U\text{ 是否具有 empirical predictive value，以及这种 predictive value 的 locality range}
}.
$$

---

# 42. Multi-Scale Mutation Bundle Construction

先计算 checkpoint 上所有合法 mutation actions 的：

$$
U.
$$

按 $U$ 分为：

$$
\boxed{10\text{ deciles}}.
$$

对每个 decile 构造：

$$
\boxed{5\text{ matched bundle families}}.
$$

每个 family 先采样一个长度为：

$$
\boxed{4096}
$$

的 ordered action list：

$$
(a_1,a_2,\dots,a_{4096}),
$$

要求：

$$
\boxed{
\text{4096 actions 之间不共享 exact-2:4 group}
}.
$$

定义四个 nested bundle sizes：

$$
\boxed{
K\in\{64,256,1024,4096\}.
}
$$

其中：

$$
B_K
=
\{a_1,\dots,a_K\}.
$$

因此同一个 family 的四个 bundle：

- 来自同一 checkpoint；
- 来自同一 $U$ decile；
- 共享相同 ordered action prefix；
- 只改变 intervention scale。

每个 checkpoint：

$$
10\times5\times4
=
\boxed{200}
$$

mutation observations。

三个 seeds、两个 checkpoints：

$$
3\times2\times200
=
\boxed{1200}
$$

mutation observations。

control branch 在同一 bundle family 的四个 $K$ 之间共享，因此 control observations：

$$
3\times2\times10\times5
=
\boxed{300}.
$$

Primary locality sizes：

$$
\boxed{K=64,256}.
$$

Stress-test sizes：

$$
\boxed{K=1024,4096}.
$$

---

# 43. Calibration Control Branch

对每个 bundle family 从完全相同 checkpoint clone。

### Control

不执行 topology mutation。

在固定 current integration batch 上进行：

$$
\boxed{1}
$$

个真实 AdamW step。

### Mutation-64

执行：

$$
B_{64}
$$

后在同一个 current integration batch 上进行 1 个真实 AdamW step。

### Mutation-256

执行：

$$
B_{256}
$$

后进行同样 1 step。

### Mutation-1024

执行：

$$
B_{1024}
$$

后进行同样 1 step。

### Mutation-4096

执行：

$$
B_{4096}
$$

后进行同样 1 step。

五个 branches 必须共享：

- checkpoint；
- integration batch；
- optimizer global step；
- scheduler state；
- RNG state before branch-specific mutation；
- evaluation data。

除 topology intervention 外，不允许 branch-specific 差异。

---

# 44. Calibration Evaluation Data

评价：

### Current Task

固定 held-out current-task eval batch。

### Past Tasks

对每一个：

$$
\tau<t
$$

使用固定 held-out past-task evaluation batch。

Past-task samples：

$$
\boxed{\text{evaluation only}}.
$$

禁止用于：

- optimizer；
- mutation scoring；
- hyperparameter tuning；
- historical consolidation；
- bundle construction。

所有 $K$ 使用完全相同 evaluation examples。

---

# 45. Realized Mutation Utility and Locality Calibration

定义：

$$
\boxed{
J
=
L_{\mathrm{current}}
+
\sum_{\tau<t}L_\tau.
}
$$

对 bundle size $K$：

$$
\boxed{
U_{\mathrm{real}}^{(K)}
=
J_{\mathrm{control}}
-
J_{\mathrm{mutation},K}.
}
$$

预测：

$$
\boxed{
\widehat U_{\mathrm{bundle}}^{(K)}
=
\sum_{a\in B_K}U_a.
}
$$

对每一个：

$$
K\in\{64,256,1024,4096\}
$$

分别计算：

$$
\boxed{
\rho_U(K)
=
\operatorname{Spearman}
\left(
\widehat U_{\mathrm{bundle}}^{(K)},
U_{\mathrm{real}}^{(K)}
\right).
}
$$

报告：

- $\rho_U(64)$；
- $\rho_U(256)$；
- $\rho_U(1024)$；
- $\rho_U(4096)$；
- hierarchical bootstrap 95% CI；
- 每个 $K$ 的 predicted-$U$ decile vs realized-$U$ reliability curve；
- $\rho_U(K)$ vs $K$ locality-degradation curve。

另外对每个 $K$ 报告：

$$
\boxed{
P(
U_{\mathrm{real}}^{(K)}>0
\mid
\widehat U^{(K)}>0
)
}
$$

与：

$$
\boxed{
P(
U_{\mathrm{real}}^{(K)}>0
\mid
\widehat U^{(K)}<0
).
}
$$

Primary empirical-predictive claim 只由：

$$
\boxed{K=64,256}
$$

支持。

Calibration CI 使用 10,000 次 hierarchical bootstrap：

1. resample seeds；
2. 保留两个预注册 checkpoints；
3. 在每个 seed × checkpoint × decile 内 resample matched bundle families；
4. 对每个 $K$ 重新计算 pooled Spearman $\rho_U(K)$。

$K=1024,4096$ 只检验 local surrogate 随 intervention scale 增大时的 degradation，不要求与小 bundle 保持相同相关强度。

---

# 46. Current-Benefit Calibration

Primary component-calibration bundle size：

$$
\boxed{K=256}.
$$

Secondary sensitivity：

$$
\boxed{K=64}.
$$

匹配：

- layer；
- matrix；
- historical-damage decile；
- $|w_p|$ decile；
- checkpoint；
- seed。

比较：

$$
\text{high-}B_{\mathrm{cur}}
$$

与：

$$
\text{low-}B_{\mathrm{cur}}.
$$

固定：

- bundle size；
- integration batch；
- optimizer step；
- mutation checkpoint；
- total mutation count。

评价：

$$
\boxed{
\Delta L_{\mathrm{current}}
=
L_{\mathrm{current}}^{\mathrm{mutation}}
-
L_{\mathrm{current}}^{\mathrm{control}}.
}
$$

预期方向：

$$
\boxed{
B_{\mathrm{cur}}\uparrow
\Rightarrow
\Delta L_{\mathrm{current}}\downarrow
}
$$

只作为 empirical calibration hypothesis，不作为数学保证。

---

# 47. Historical-Damage Calibration

Primary component-calibration bundle size：

$$
\boxed{K=256}.
$$

Secondary sensitivity：

$$
\boxed{K=64}.
$$

匹配：

- layer；
- matrix；
- $B_{\mathrm{cur}}$ decile；
- mutation count；
- checkpoint；
- seed。

比较：

$$
\text{high-}D_{\mathrm{hist}}
$$

与：

$$
\text{low-}D_{\mathrm{hist}}.
$$

评价：

$$
\boxed{
\Delta L_{\mathrm{past}}
=
\sum_{\tau<t}
L_\tau^{\mathrm{mutation}}
-
\sum_{\tau<t}
L_\tau^{\mathrm{control}}.
}
$$

预期方向：

$$
\boxed{
D_{\mathrm{hist}}\uparrow
\Rightarrow
\Delta L_{\mathrm{past}}\uparrow
}
$$

只作为 empirical calibration hypothesis，不作为数学保证。

---

# 48. Ablation A1 — w/o Historical Memory

设置：

$$
\boxed{
P=H=0.
}
$$

于是：

$$
D_{\mathrm{hist}}=0.
$$

Mutation utility：

$$
\boxed{
U=B_{\mathrm{cur}}.
}
$$

---

# 49. Ablation A2 — Active-Only Historical Memory

Task-end historical probe 后：

$$
\boxed{
F_i=0
\quad
\text{for dormant coordinates}.
}
$$

只保留 active-coordinate historical state。

用于比较：

$$
\text{Full Candidate Memory}
$$

vs

$$
\text{Active-Only Memory}.
$$

---

# 50. Ablation A3 — Precision-Only Historical Memory

设置：

$$
\boxed{
H=0.
}
$$

于是：

$$
\mu=0.
$$

Historical surrogate 退化为：

$$
\boxed{
Q_{\mathrm{hist}}
=
\frac12Pw^2.
}
$$

用于比较：

$$
(P,H)
$$

与：

$$
P\text{-only}.
$$

---

# 51. Ablation A4 — v2.2 Heuristic

完整运行：

$$
\boxed{
\text{DynaSyn-v2.2-NR}.
}
$$

用于比较：

$$
\text{heuristic independent prune/grow ranking}
$$

与：

$$
\text{counterfactual complete-action scoring}.
$$

---

# 52. Ablation A5 — w/o Cooldown

所有 active coordinates：

$$
\boxed{
C=2.
}
$$

即所有 active positions 立即 prune eligible。

---

# 53. Ablation A6 — Gradient-Imprinted Newborn

Secondary：

$$
\boxed{
w_g
=
-\eta_{\mathrm{init}}g_g.
}
$$

只作为 newborn initialization ablation。

主方法仍使用：

$$
w_g=0.
$$

---

# 54. Statistical Protocol

## 54.1 Seed Tiers

Primary structural seed set：

$$
\boxed{
S_{\mathrm{struct}}=\{42,43,44,45,46\}.
}
$$

用于：

- Static Exact-2:4；
- SRigL-style Exact-2:4；
- DynaSyn-v2.2-NR；
- DynaSyn-v3.2；
- TRACE Order-1 primary structural contrasts。

External / broad-comparison seed set：

$$
\boxed{
S_{\mathrm{ext}}=\{42,43,44\}.
}
$$

用于：

- Dense Regional FT；
- external recent baselines；
- Seq-GLUE-7；
- Long-CL-15；
- Llama backbone replication；
- TRACE Order-2 robustness。

Calibration seed set：

$$
\boxed{
S_{\mathrm{cal}}=\{42,43,44\}.
}
$$

Qwen3-8B mechanism scale-direction validation：

$$
\boxed{\text{seed}=42}.
$$

该单 seed 只作 descriptive directional replication，不参与 multi-seed significance claim。

---

## 54.2 Pairing Rules

所有可配对方法尽可能共享：

- initialization；
- data order；
- task order；
- $M_0$；
- evaluation samples；
- development/final split；
- random-seed mapping。

Structural 5-seed summary 使用全部 $S_{\mathrm{struct}}$。

DynaSyn 与 external baseline 的 pairwise contrast 必须限制在：

$$
\boxed{
S_{\mathrm{struct}}\cap S_{\mathrm{ext}}
=
\{42,43,44\}.
}
$$

禁止把 DynaSyn 的 5-seed mean 与某 external method 的 3-seed mean 直接当成 paired effect。

---

## 54.3 Reporting

主表报告：

$$
\boxed{
\mathrm{mean}\pm SE.
}
$$

同时报告 paired effect size：

$$
\boxed{
\Delta_{A-B}=Metric_A-Metric_B.
}
$$

每个 primary structural contrast 额外报告 5 个 seed-wise paired differences。

不以单个 seed 的最好结果作为正文数字。

不把 task-level observations 当作相互独立的重复实验。

---

# 55. Paired Bootstrap and Pre-Registered Contrasts

## 55.1 Primary Structural Contrasts

预注册 primary structural contrasts：

1. DynaSyn-v3.2 vs SRigL-style Exact-2:4；
2. DynaSyn-v3.2 vs DynaSyn-v2.2-NR。

均使用：

$$
\boxed{5\text{ paired seeds}}.
$$

Primary uncertainty analysis 使用 paired hierarchical bootstrap：

1. resample seeds with replacement；
2. 在 sampled seed 内对 task-indexed metric components 做 paired resampling；
3. 保持同一 bootstrap draw 中 method pairing 不变；
4. 重新计算 aggregate metric difference。

重复：

$$
\boxed{10,000}
$$

次。

报告：

$$
\boxed{95\%\text{ confidence interval}.}
$$

由于 continual tasks 具有顺序依赖，task-resampling CI 解释为 aggregate robustness interval，而不是把每个 task 当作 iid independent experiment。

---

## 55.2 Primary External Contrasts

固定 external contrasts：

1. DynaSyn-v3.2 vs **OSFT**；
2. DynaSyn-v3.2 vs Meta-UCF。

使用公共：

$$
\boxed{3\text{ paired seeds}:42,43,44}.
$$

Secondary recent-method contrasts：

- v3.2 vs Any-SSR；
- v3.2 vs PaRSP；
- v3.2 vs GORP；
- v3.2 vs TreeLoRA；
- v3.2 vs O-LoRA。

全部报告 effect + 95% CI，但禁止在 final test 后根据“谁最强”重新定义 comparator hierarchy。

---

## 55.3 Order-Robustness Statistics

TRACE Order-1 与 Order-2 分别计算 effect，并报告：

- order-specific effect；
- order-average effect；
- worst-order signed effect；
- 3-seed paired CI on Order-2。

Order-2 不参与任何 hyperparameter selection。

---

## 55.4 Long-Horizon Statistics

Long-CL-15 报告：

- final ACC；
- Forgetting；
- BWT；
- AG；
- task-indexed $P$ statistics；
- task-indexed positive-$U$ fraction；
- task-indexed realized mutation ratio。

额外计算：

$$
\boxed{
\rho_{P,\rho}
=
\operatorname{Spearman}
\left(
\operatorname{median}(P_t),
\rho_t^{\mathrm{realized}}
\right)
}
$$

与：

$$
\boxed{
\rho_{P,U+}
=
\operatorname{Spearman}
\left(
\operatorname{median}(P_t),
f_t^{U>0}
\right).
}
$$

这两个量只作为 structural-freezing diagnosis，不作为因果证明。

---

# 56. Main Result Tables

## 56.1 TRACE Order-1 Continual-Learning Performance

| Method | Seeds | Replay | Task-ID / Router | ACC ↑ | Forgetting ↓ | AG ↑ | BWT ↑ | FWT ↑ | ΔGA_stream ↑ | ΔGA_total ↑ |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Naive FT | 3 | | | | | | | | | |
| Dense Regional FT | 3 | 0 | none | | | | | | | |
| LoRA | 3 | | | | | | | | | |
| O-LoRA | 3 | | | | | | | | | |
| Meta-UCF | 3 | | | | | | | | | |
| **OSFT** | 3 | | | | | | | | | |
| GORP | 3 | | | | | | | | | |
| TreeLoRA | 3 | | | | | | | | | |
| Any-SSR | 3 | | | | | | | | | |
| PaRSP | 3 | | | | | | | | | |
| Static Exact-2:4 | 5 | 0 | none | | | | | | | |
| SRigL-style Exact-2:4 | 5 | 0 | none | | | | | | | |
| DynaSyn-v2.2-NR | 5 | 0 | none | | | | | | | |
| **DynaSyn-v3.2** | **5** | **0** | **none** | | | | | | | |

`Replay`、`Task-ID / Router` 对 external baseline 由 code audit 后填写，不预先假定。

DynaSyn-v3.2 与 external method 的显式 effect table 使用 seeds 42/43/44 重新聚合。

---

## 56.2 General-Capability Decomposition

| Method | ΔGA_init ↑ | ΔGA_stream ↑ | ΔGA_total ↑ |
|---|---:|---:|---:|
| Static Exact-2:4 | | | |
| SRigL-style Exact-2:4 | | | |
| DynaSyn-v2.2-NR | | | |
| **DynaSyn-v3.2** | | | |
| LoRA | 0 | | |
| Meta-UCF | 0 | | |
| OSFT | 0 | | |
| Any-SSR | 0 | | |
| PaRSP | | | |

会改变 base-network effective initialization 的方法按实际 initialization state 计算，不强行置 $\Delta GA_{\mathrm{init}}=0$。

Replay-based methods 单独报告。

---

## 56.3 TRACE Order Robustness

| Method | Order-1 ACC | Order-2 ACC | Order-1 Forgetting | Order-2 Forgetting | Order-Avg Effect vs SRigL | Worst-Order Effect vs SRigL |
|---|---:|---:|---:|---:|---:|---:|
| Static Exact-2:4 | | | | | | |
| SRigL-style Exact-2:4 | | | | | 0 | 0 |
| DynaSyn-v2.2-NR | | | | | | |
| **DynaSyn-v3.2** | | | | | | |

---

## 56.4 Long-CL-15

| Method | ACC ↑ | Forgetting ↓ | AG ↑ | BWT ↑ | Median ρ_realized | Final positive-U fraction | Persistent-state Growth / Task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static Exact-2:4 | | | | | 0 | 0 | 0 |
| SRigL-style Exact-2:4 | | | | | | | 0 |
| DynaSyn-v2.2-NR | | | | | | | 0 |
| **DynaSyn-v3.2** | | | | | | | 0 |
| O-LoRA | | | | | | | |
| Meta-UCF | | | | | | | |
| OSFT | | | | | | | |

DynaSyn 的 `Persistent-state Growth / Task = 0` 仅指 $(P,H)$ 等固定-size persistent tensors 的**张量尺寸**不随 task 数增加；数值内容会持续更新。

---

# 57. Structural Behavior Metrics

每个 mutation event 保存：

$$
\rho^{\mathrm{realized}}.
$$

保存：

- candidate group count；
- mature group count；
- $U>0$ group count；
- selected group count；
- per-layer mutation rate；
- per-matrix mutation rate；
- mean / median $U$；
- $B_{\mathrm{cur}}$ distribution；
- $D_{\mathrm{hist}}$ distribution；
- fraction $D_{\mathrm{hist}}<0$；
- task-end topology snapshot。

Task-end 额外保存：

$$
\boxed{
\operatorname{median}(P),
\quad
P_{90}(P),
\quad
P_{99}(P)
}
$$

以及：

$$
\boxed{
f^{U>0}
=
\frac{N_{U>0}}
{N_{\mathrm{mature\ legal}}}.
}
$$

Long-CL-15 必须绘制随 task index 的：

- $\operatorname{median}(P_t)$；
- $P_{90}(P_t)$；
- $f_t^{U>0}$；
- $\rho_t^{\mathrm{realized}}$；
- $AG_t$；
- Forgetting contribution；
- topology Jaccard to $M_0$ 与上一 task。

若出现：

$$
P_t\uparrow,
\qquad
f_t^{U>0}\rightarrow0,
\qquad
\rho_t^{\mathrm{realized}}\rightarrow0,
$$

则记录为 historical-precision-induced structural freezing signal；不得仅以低 forgetting 将其解释为成功，因为可能同时伴随 plasticity collapse。

---

# 58. Topology Similarity

Task-end edge sets：

$$
E_1,\dots,E_T.
$$

Jaccard：

$$
\boxed{
J(E_i,E_j)
=
\frac{|E_i\cap E_j|}
{|E_i\cup E_j|}.
}
$$

按：

- layer；
- `up_proj`；
- `down_proj`

分别分析。

---

# 59. Memory Accounting

三个 structural backbones 的 candidate coordinates 均为：

$$
\boxed{
N=201,326,592.
}
$$

Exact-2:4 active coordinates：

$$
\boxed{
N_{\mathrm{active}}=100,663,296.
}
$$

若：

$$
P,H
$$

都使用 FP32：

$$
2\times
201,326,592
\times4
$$

bytes。

约：

$$
\boxed{
1.61\text{ GB}
\approx
1.50\text{ GiB}.
}
$$

因此 Qwen3-8B mechanism scale-direction validation 通过只修改最后 2 层，将 DynaSyn candidate-level persistent historical memory 保持在与 1.7B primary experiment 相同量级。

必须单独报告：

- base-model weights；
- target dense storage weights；
- mask；
- cooldown；
- $P$；
- $H$；
- Adam first moment；
- Adam second moment；
- gradient/probe temporary buffers；
- activation memory；
- peak HBM。

对所有 external baseline 额外报告：

$$
\boxed{
\text{persistent bytes after task }t
}
$$

以及：

$$
\boxed{
\frac{\Delta\text{persistent bytes}}{\Delta t}
}
$$

用于区分 constant-memory 与 per-task-growing methods。

---

# 60. Training-Time and Resource Accounting

所有主方法记录：

1. total wall-clock / full stream；
2. normal step latency；
3. mutation event latency；
4. current probe latency；
5. task-end consolidation latency；
6. evaluation time 单列；
7. peak HBM；
8. persistent state at stream end；
9. persistent-state growth / task；
10. inference latency。

Structural-method 正文表：

| Method | Train Time / Stream | Peak HBM | Updated Params | Persistent State | Inference Latency |
|---|---:|---:|---:|---:|---:|
| Static | | | | | |
| SRigL | | | | | |
| v2.2 | | | | | |
| **v3.2** | | | | | |

External-method resource table：

| Method | Trainable Params | Updated Params | Persistent State @T | Growth / Task | Replay Bytes | Task-ID / Router | Train Time | Peak HBM |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| LoRA | | | | | | | | |
| O-LoRA | | | | | | | | |
| Meta-UCF | | | | | | | | |
| OSFT | | | | | | | | |
| GORP | | | | | | | | |
| TreeLoRA | | | | | | | | |
| Any-SSR | | | | | | | | |
| PaRSP | | | | | | | | |

不把 dense masked training 描述成 sparse-training speedup。

DynaSyn efficiency claim 只允许建立在：

- fixed active connectivity capacity；
- deployable exact 2:4 structure；
- measured final sparse inference/runtime；
- transparent training overhead accounting。

之上。

---

# 61. Hardware Validation

最终 learned exact-2:4 topology 至少对以下真实矩阵 shape 做 runtime validation。

## Qwen3-1.7B

$$
6144\times2048,
$$

$$
2048\times6144.
$$

## Llama-3.2-1B

$$
8192\times2048,
$$

$$
2048\times8192.
$$

## Qwen3-8B

$$
12288\times4096,
$$

$$
4096\times12288.
$$

流程：

```text
learned exact-2:4 W
    ↓
validate exact 2:4
    ↓
confirm dormant values = 0
    ↓
convert to semi-structured sparse representation
    ↓
sparse forward
    ↓
dense-masked forward parity
    ↓
real-shape GEMM benchmark
```

至少记录：

- conversion success；
- numerical parity；
- median latency；
- p10 / p90 latency；
- throughput；
- memory；
- dense-vs-sparse speed ratio。

benchmark 必须包含 warm-up，并固定：

- dtype；
- batch size；
- sequence-independent GEMM shape；
- device；
- software stack；
- clock/power policy（若可控）。

训练阶段仍按 Section 7 报告为 dense masked forward/backward。

---

# 62. Exact-2:4 Invariants

任何 mutation 后：

$$
\boxed{
\forall q,
\qquad
\sum_{i\in q}M_i=2.
}
$$

分组维度固定：

$$
\boxed{
\text{weight last / K dimension}.
}
$$

---

# 63. Dormant-State Invariants

任何 normal step / mutation 后：

$$
\boxed{
M_i=0
\Rightarrow
w_i=0.
}
$$

$$
\boxed{
M_i=0
\Rightarrow
m_i=v_i=0.
}
$$

但：

$$
\boxed{
M_i=0
\not\Rightarrow
P_i=H_i=0.
}
$$

---

# 64. Numerical Assertions

每个 mutation event 后：

```text
exact_2_4_legality == 100%

max(abs(W[dormant])) == 0

max(abs(exp_avg[dormant])) == 0

max(abs(exp_avg_sq[dormant])) == 0

isfinite(P).all()

isfinite(H).all()

P.min() >= 0
```

并检查：

```text
historical P/H unchanged by prune/grow transition
```

---

# 65. Counterfactual Unit Test

构造 toy exact-2:4 Linear。

随机初始化：

- $W$；
- $m$；
- $v$；
- $P$；
- $H$；
- signed probe gradients。

枚举 4 个合法 swaps。

分别：

1. 用 vectorized v3.2 scoring 计算 $U$；
2. clone keep branch；
3. clone swap branch；
4. 显式执行 hypothetical one-step parameter states；
5. 根据同一公式直接计算 loss surrogate difference。

FP32 relative error：

$$
\boxed{
<10^{-5}.
}
$$

---

# 66. Historical Compression Unit Test

随机生成：

$$
F^{(1)},F^{(2)},F^{(3)}
$$

以及：

$$
w^{(1)},w^{(2)},w^{(3)}.
$$

显式：

$$
Q_{\mathrm{explicit}}(w)
=
\frac12
\sum_\tau
F^{(\tau)}
(w-w^{(\tau)})^2.
$$

Compressed：

$$
Q_{\mathrm{compressed}}(w)
=
\frac12
P(w-\mu)^2+C.
$$

对于随机：

$$
w_a,w_b
$$

要求：

$$
\boxed{
Q_{\mathrm{explicit}}(w_a)
-
Q_{\mathrm{explicit}}(w_b)
\approx
Q_{\mathrm{compressed}}(w_a)
-
Q_{\mathrm{compressed}}(w_b).
}
$$

FP32 tolerance 内成立。

---

# 67. STE Probe Unit Test

验证：

### Forward

$$
W_{\mathrm{STE}}
$$

与：

$$
M\odot W
$$

输出一致。

### Backward

Dormant coordinates：

$$
\boxed{
\nabla w_i\neq0
}
$$

可被读取。

### Isolation

Probe 前后：

- model weights 不变；
- Adam state 不变；
- scheduler state 不变；
- gradient scaler state 不变。

---

# 68. Development Protocol

## 68.1 DynaSyn Development

唯一 DynaSyn development stream：

$$
\boxed{
\text{Qwen3-1.7B}\times\text{TRACE-8 Order-1}\times\text{seed }42
}
$$

只访问 validation/eval split。

允许搜索：

$$
\boxed{
\rho_{\max}
\in
\{0.5\%,1\%,2\%\}.
}
$$

默认 current probe：

$$
\boxed{32}.
$$

若显存/噪声需要验证，只允许：

$$
16
$$

vs

$$
32
$$

一次开发比较。

除此之外不进行大规模搜索。

**Seq-GLUE-7、Long-CL-15、TRACE Order-2、Llama backbone、Qwen3-8B 均不是 DynaSyn development stream。**

---

## 68.2 External Baseline Development

所有 recent baselines：

- 仅 seed 42；
- 仅 validation/eval；
- official default + 最多 2 个邻近配置；
- 只允许搜索 1 个主要 method-specific hyperparameter；
- 最多 3 个 configs/method；
- 不访问 final test。

OSFT 必须先跑 official / reference configuration，再允许一个主要 subspace-control hyperparameter 的最多 2 个邻近值。

所有 baseline development logs 在 freeze 前保存。

---

## 68.3 Calibration Pilot

在 final three-seed calibration 前，只允许使用：

$$
\boxed{
\text{seed }42,
\quad
T_4@60\%,
\quad
K\in\{64,256\}
}
$$

做 pilot。

pilot 只检查：

- branch cloning correctness；
- sign convention；
- evaluation variance；
- $\rho_U$ 是否完全失效。

禁止根据 pilot outcome 修改 $U$ 的公式。

如果需要修改公式，则版本升级，不得继续称为 frozen v3.2。

---

## 68.4 Zero-Retuning Transfer Tests

以下实验只允许 compatibility debugging 与 numerical correctness testing，不允许 method-level retuning：

1. Seq-GLUE-7；
2. Long-CL-15 Order-4；
3. TRACE Order-2；
4. Llama-3.2-1B-Instruct × TRACE；
5. Qwen3-8B × TRACE。

允许 architecture-induced changes 仅限：

- tensor shape；
- target-layer index mapping；
- tokenizer/chat-template compatibility；
- distributed execution configuration；
- batch decomposition needed to satisfy memory limits while preserving effective batch。

禁止根据这些 transfer-test 的 performance 修改：

- $U$ formula；
- $\rho_{\max}$；
- cooldown；
- probe size；
- consolidation size；
- mutation schedule。

---

# 69. Freeze Protocol

完成 development 后冻结：

- model revisions：Qwen3-1.7B / Llama-3.2-1B / Qwen3-8B；
- dataset revisions；
- TRACE Order-1 / Order-2；
- Long-CL-15 task composition 与 Order-4；
- Long-CL-15 sample-selection rule 与 RNG；
- tokenizer；
- prompt template；
- target layers；
- optimizer；
- scheduler；
- learning rate；
- mutation schedule；
- $\rho_{\max}$；
- cooldown；
- probe budget；
- consolidation budget；
- all formulas；
- structural baselines；
- external baseline list；
- **OSFT repository commit 与 port config**；
- external baseline repository commits；
- external baseline port patches；
- external baseline hyperparameters；
- structural 5-seed set；
- external 3-seed set；
- primary statistical contrasts；
- general-capability evaluator revision；
- general-capability prompt / few-shot policy；
- calibration checkpoints；
- calibration decile construction；
- calibration bundle family count；
- $K\in\{64,256,1024,4096\}$；
- calibration sampling RNG seed；
- Qwen3-8B target layers 与 mechanism-scale-direction-only interpretation。

保存：

```text
frozen_config.yaml
git_commit.txt
environment.txt
baseline_manifest.yaml
baseline_commits/
baseline_patches/
eval_manifest.yaml
calibration_manifest.yaml
trace_order_manifest.yaml
longcl_manifest.yaml
seed_manifest.yaml
claim_boundary.yaml
```

`claim_boundary.yaml` 至少写死：

```yaml
primary_mechanism_claim:
  - complete_action_counterfactual_scoring
  - empirical_locality_calibration
primary_system_claim:
  - fixed_active_connectivity_capacity
  - replay_free_primary_setting
forbidden_claims:
  - sparse_training_speedup
  - matched_fraction_scaling_law
  - scales_to_8b_from_single_seed
```

冻结后才能运行 full final seed sets。

任何 freeze 后的方法级公式变化都要求创建新版本，不得用 v3.2 final-test result 反向调参。

---

# 70. Go / No-Go

## G0 — Evaluation Ready

要求：

- tokenizer/template 正确；
- deterministic eval；
- task metrics 正确；
- Qwen thinking 关闭；
- dense / $M_0$ / $M_T$ general-capability evaluator 一致。

---

## G1 — Static Exact-2:4 Correct

要求：

$$
\boxed{
2{:}4\text{ legality}=100\%.
}
$$

Dormant invariants 全部通过。

---

## G2 — Dynamic Topology Has Plasticity

Validation 比较：

$$
\boxed{
AG_{\mathrm{SRigL}}
>
AG_{\mathrm{Static}}.
}
$$

若没有稳定趋势，优先检查：

- probe；
- mutation budget；
- target layers；
- training budget；
- implementation correctness。

---

## G3 — v3.2 Counterfactual Operator

Development seed 比较：

$$
\boxed{
\text{v3.2 vs v2.2-NR}.
}
$$

要求至少在 ACC / Forgetting 之一出现稳定改善，同时 AG 无明显下降。

若 v3.2 完全无改善，则停止扩展 v3.2 大规模实验，保留 failure analysis。

---

## G4 — Local Counterfactual Calibration

Development pilot 至少要求：

$$
\boxed{
\rho_U(64)>0
}
$$

或：

$$
\boxed{
\rho_U(256)>0.
}
$$

Final empirical-predictive claim 只有在 pooled final calibration 中：

$$
\boxed{
K\in\{64,256\}
}
$$

呈现稳定 positive association 时成立。

若 $K=64,256$ 都接近 0，则记录 calibration failure，不把 $U$ 解释为具有 empirical predictive value。

$K=1024,4096$ 允许因 higher-order interaction 导致 correlation degradation。

---

## G5 — Replay-Free Stability

No-replay 下：

$$
F_{\mathrm{v3.2}}
<
F_{\mathrm{SRigL}}
$$

且：

$$
AG_{\mathrm{v3.2}}
\ge
AG_{\mathrm{SRigL}}-1.
$$

---

## G6 — Benchmark Generalization

TRACE Order-1 与 Seq-GLUE 中：

$$
\boxed{
\operatorname{sign}
\left(
\Delta_{\mathrm{v3.2-v2.2}}
\right)
}
$$

总体方向一致。

---

## G7 — Backbone Generalization

Qwen3-1.7B 与 Llama-3.2-1B 中：

$$
\boxed{
\operatorname{sign}
\left(
\Delta_{\mathrm{v3.2-SRigL}}
\right)
}
$$

总体方向一致。

---

## G8 — Recent-Baseline Competitiveness

在 Qwen3-1.7B × TRACE-8 Order-1 中重点检查：

- **OSFT**；
- Meta-UCF；
- Any-SSR；
- PaRSP。

若 DynaSyn-v3.2 在：

- ACC；
- Forgetting；
- resource / persistent-state profile；

三个维度上被全部 primary recent contrasts 严格支配，则只保留 structural-plasticity / calibrated-local-surrogate 的窄机制结论。

---

## G9 — 8B Mechanism Scale Direction

Qwen3-8B seed42 中比较：

$$
\boxed{
\text{v3.2 vs SRigL}.
}
$$

要求至少：

- ACC 或 Forgetting 改善方向与 1.7B 一致；
- AG 不出现明显 collapse；
- exact-2:4 与 runtime validation 全部通过。

即使满足，也只记录为：

$$
\boxed{
\text{mechanism scale-direction consistency}
}
$$

而不是 scaling law。

若方向反转，则明确记录 scale limitation。

---

## G10 — General-Capability Decomposition Complete

必须同时得到：

$$
\boxed{
\Delta GA_{\mathrm{init}},
\quad
\Delta GA_{\mathrm{stream}},
\quad
\Delta GA_{\mathrm{total}}.
}
$$

禁止只用 $M_T-M_0$ 隐藏 initial 2:4 pruning cost。

---

## G11 — TRACE Order Robustness

Order-2 中，v3.2 vs SRigL 的 ACC / Forgetting 至少一个主要稳定性指标保持与 Order-1 相同改善方向，且 AG 不出现明显 collapse。

若 Order-2 完全反转，则：

$$
\boxed{
\text{记录 task-order sensitivity}
}
$$

不得只报告 Order-1。

---

## G12 — Long-Horizon Plasticity

Long-CL-15 中必须同时检查：

1. final ACC / Forgetting；
2. AG 随 task index 的趋势；
3. $\operatorname{median}(P_t)$；
4. $f_t^{U>0}$；
5. $\rho_t^{\mathrm{realized}}$。

若后半程出现：

$$
\boxed{
f_t^{U>0}\approx0}
$$

且：

$$
\boxed{\rho_t^{\mathrm{realized}}\approx0}
$$

并伴随 AG collapse，则判定为 structural freezing limitation；低 forgetting 不能抵消该结论。

---

# 71. Execution Priority and Order

## 71.1 P0 — Correctness Gate

1. Qwen3-1.7B TRACE evaluation pipeline；
2. dense pretrained general-capability baseline；
3. exact 2:4 grouping unit test；
4. Static Exact-2:4 seed42；
5. $M_0$ general-capability evaluation；
6. STE current probe；
7. SRigL-style Exact-2:4 seed42；
8. historical $P,H$ implementation；
9. historical compression unit test；
10. counterfactual scoring unit test；
11. DynaSyn-v2.2-NR seed42；
12. DynaSyn-v3.2 seed42；
13. G0–G5 validation；
14. calibration pilot $K=64,256$。

P0 未通过时，不启动大规模 external / 8B runs。

---

## 71.2 P1 — Primary Evidence

15. freeze DynaSyn method config；
16. TRACE Order-1 Static / SRigL / v2.2 / v3.2 seeds 42–46；
17. final multi-scale calibration seeds 42/43/44；
18. dense / $M_0$ / $M_4$ / $M_T$ general-capability evaluation；
19. OSFT official-code audit + seed42 development；
20. Meta-UCF / Any-SSR / PaRSP audit + seed42 development；
21. OSFT / Meta-UCF / Any-SSR / PaRSP final seeds 42/43/44；
22. primary resource accounting。

---

## 71.3 P2 — Robustness Evidence

23. TRACE Order-2 core structural methods seeds 42/43/44；
24. Long-CL-15 Order-4 core structural methods seeds 42/43/44；
25. Long-CL-15 O-LoRA / Meta-UCF / OSFT seeds 42/43/44；
26. Seq-GLUE-7 structural + selected external methods seeds 42/43/44；
27. G6 / G11 / G12 evaluation；
28. Long-horizon structural-freezing diagnostics。

---

## 71.4 P3 — Breadth and Attribution

29. GORP / TreeLoRA official-code audit and TRACE final seeds；
30. Llama-3.2-1B TRACE core replication seeds 42/43/44；
31. A1/A2/A3/A4 primary ablations；
32. A5/A6 secondary ablations；
33. optional Long-CL Order-5 / Order-6 extension；
34. replay complementarity。

---

## 71.5 P4 — Mechanism Scale Direction and Hardware

35. Qwen3-8B architecture compatibility tests；
36. Qwen3-8B TRACE Static / SRigL / v3.2 seed42 with zero retuning；
37. three-shape-family hardware runtime validation；
38. final persistent-state / wall-clock / Pareto plots；
39. final artifact integrity audit。

---

# 72. Required Artifacts Per Run

每个 run 保存：

```text
config.yaml
environment.txt
git_commit.txt
model_revision.txt
dataset_revision.txt
benchmark_manifest.yaml
task_order_manifest.yaml
seed_manifest.yaml

metrics/
    task_matrix.json
    aggregate_metrics.json
    general_ability.json
    general_ability_reference_states.json
    seedwise_effects.json

topology/
    mask_task_*.pt
    mutation_events.jsonl
    topology_similarity.json

historical/
    P_task_*.pt
    H_task_*.pt
    precision_summary_task_*.json

profiling/
    wallclock.json
    peak_hbm.json
    event_latency.json
    persistent_state_bytes.json
    inference_latency.json

calibration/
    predicted_actions.parquet
    bundle_families.jsonl
    branch_manifest.jsonl
    realized_effects.parquet
    rho_by_bundle_size.json
    reliability_by_bundle_size.json

baseline_audit/
    baseline_name.txt
    repository_commit.txt
    original_config.yaml
    ported_config.yaml
    code_diff.patch
    replay_audit.json
    auxiliary_data_audit.json
    contamination_audit.json
    task_id_router_audit.json
    persistent_state_audit.json
```

对于 structural methods，`baseline_audit/` 可只保留本仓库 implementation manifest。

对于 Long-CL-15 额外保存：

```text
long_horizon/
    longcl_manifest.yaml
    actual_sample_counts.json
    order_id.txt
    p_by_task.json
    positive_u_fraction_by_task.json
    realized_mutation_ratio_by_task.json
    acquisition_by_task.json
    structural_freezing_diagnostics.json
```

对于 TRACE Order-2 额外保存：

```text
order_robustness/
    trace_order_2.yaml
    inherited_hparams.yaml
    no_retuning_assertion.txt
    order_specific_effects.json
```

对于 Experiment F 额外保存：

```text
scale_validation/
    inherited_config.yaml
    architecture_mapping.yaml
    no_retuning_assertion.txt
    claim_boundary.yaml
```

---

# 73. Mutation Event Log Schema

每个 mutation event 至少记录：

```yaml
model_id:
model_scale:
benchmark_id:
task_order_id:
task_id:
event_id:
optimizer_step:
rho_max:
num_groups:
num_mature_groups:
num_positive_u_groups:
num_selected_groups:
realized_mutation_ratio:
positive_u_fraction:

median_p:
p90_p:
p99_p:

mean_u:
median_u:
mean_b_cur:
mean_d_hist:
fraction_negative_d_hist:

per_layer_mutations:
per_matrix_mutations:
```

每个 selected group 记录：

```yaml
layer:
matrix:
group_index:
prune_index:
grow_index:

w_p:
g_p:
g_g:

P_p:
H_p:
mu_p:

P_g:
H_g:
mu_g:

delta_p_keep:
delta_g_new:

B_cur:
D_p:
D_g:
D_hist:
U:
```

---

# 74. Final Checklist

## Model / Structure

- [ ] Qwen3-1.7B revision 固定
- [ ] Llama-3.2-1B revision 固定
- [ ] Qwen3-8B revision 固定
- [ ] 1.7B target layers = 20–27
- [ ] Llama target layers = 10–15
- [ ] 8B target layers = 34–35
- [ ] exact 2:4 沿 K dimension
- [ ] 三个 structural backbones candidate count = 201,326,592
- [ ] 三个 structural backbones active count = 100,663,296
- [ ] Static / SRigL / v2.2 / v3.2 共用对应 backbone 的 $M_0$

## Core Mechanism

- [ ] primary mechanism = complete-action counterfactual scoring
- [ ] primary calibration = empirical locality calibration
- [ ] 不把 dynamic sparsity 本身作为唯一 novelty
- [ ] 不把 $U$ 解释为 exact expected gain
- [ ] 不宣称 sparse-training speedup
- [ ] 不从 8B single seed 推导 scaling law

## Training

- [ ] no replay 主设置
- [ ] answer-only loss
- [ ] Qwen `enable_thinking=False`
- [ ] normal dormant gradient update eligibility = 0
- [ ] AdamW state reset 正确
- [ ] global optimizer step 不被 newborn reset
- [ ] cross-benchmark / cross-order / cross-backbone zero retuning
- [ ] Qwen3-8B mechanism scale-direction validation zero retuning

## Historical Memory

- [ ] task-end 64 examples
- [ ] 8×8 microbatch Fisher-style estimator
- [ ] active + dormant 都计算 $F$
- [ ] $P=P+F$
- [ ] $H=H+FW_{\mathrm{eff}}$
- [ ] prune/grow 不 reset $P,H$
- [ ] historical compression unit test 通过
- [ ] Long-CL 保存 median / p90 / p99 $P$

## Mutation

- [ ] 20/40/60/80%
- [ ] current probe = 32
- [ ] signed gradient
- [ ] enumerate all legal group actions
- [ ] hypothetical optimizer 不写真实 state
- [ ] compute $B_{\mathrm{cur}}$
- [ ] compute $D_p$
- [ ] compute $D_g$
- [ ] compute $D_{\mathrm{hist}}$
- [ ] compute $U$
- [ ] only $U>0$ candidate
- [ ] top-$U$ under $\rho_{\max}$
- [ ] exact 2:4 legality after transition

## External Baselines

- [ ] O-LoRA
- [ ] Meta-UCF
- [ ] **OSFT**
- [ ] GORP
- [ ] TreeLoRA
- [ ] Any-SSR
- [ ] PaRSP
- [ ] OSFT official `mini_trainer` implementation audited
- [ ] official repository commit pinned
- [ ] common-backbone port diff saved
- [ ] replay behavior audited from code
- [ ] persistent-state growth audited
- [ ] auxiliary anchor / covariance / calibration data audited
- [ ] general-capability contamination audit complete
- [ ] task-ID / router requirement audited
- [ ] baseline development <= 3 configs/method
- [ ] final-test hyperparameters frozen

## Evaluation Streams

- [ ] TRACE-8 Order-1 full stream
- [ ] TRACE-8 Order-2 core structural stream
- [ ] Seq-GLUE-7
- [ ] Long-CL-15 Order-4
- [ ] optional Long-CL Order-5 / Order-6 only after freeze
- [ ] Qwen3-1.7B structural methods 5 seeds = 42–46
- [ ] external baselines 3 seeds = 42–44
- [ ] Llama core replication 3 seeds
- [ ] Qwen3-8B full TRACE seed42
- [ ] full task×checkpoint matrix
- [ ] ACC
- [ ] Forgetting
- [ ] AG
- [ ] BWT
- [ ] FWT
- [ ] mean ± SE
- [ ] paired bootstrap 95% CI
- [ ] seed-wise structural effects reported
- [ ] primary contrasts frozen before final test

## Order Robustness

- [ ] TRACE Order-2 sequence fixed
- [ ] task epochs inherited by task identity
- [ ] Order-2 no retuning
- [ ] order-specific effects
- [ ] order-average effect
- [ ] worst-order signed effect

## Long-Horizon

- [ ] 15-task composition fixed
- [ ] Order-4 fixed before final runs
- [ ] unique training sample selection audited
- [ ] actual sample count per task saved
- [ ] no duplicate oversampling to reach 1000
- [ ] Long-CL no DynaSyn retuning
- [ ] positive-$U$ fraction by task
- [ ] realized mutation ratio by task
- [ ] $P$ statistics by task
- [ ] AG by task
- [ ] structural-freezing diagnosis completed
- [ ] BoolQ not reused as independent general-capability hold-out on Long-CL

## General Capability

- [ ] dense pretrained evaluated
- [ ] $M_0$ evaluated
- [ ] Experiment A $M_4$ evaluated
- [ ] $M_T$ evaluated
- [ ] $\Delta GA_{\mathrm{init}}$
- [ ] $\Delta GA_{\mathrm{stream}}$
- [ ] $\Delta GA_{\mathrm{total}}$
- [ ] same evaluator revision / prompts / decoding

## Calibration

- [ ] T4 60% checkpoint
- [ ] T7 60% checkpoint
- [ ] 10 $U$ deciles
- [ ] 5 matched bundle families/decile
- [ ] bundle sizes 64 / 256 / 1024 / 4096
- [ ] nested-prefix construction
- [ ] no shared group within bundle family
- [ ] same checkpoint branches
- [ ] same integration batch
- [ ] current + past held-out evaluation
- [ ] predicted bundle $U$
- [ ] realized bundle $U$
- [ ] $\rho_U(64)$
- [ ] $\rho_U(256)$
- [ ] $\rho_U(1024)$
- [ ] $\rho_U(4096)$
- [ ] bootstrap CI
- [ ] reliability curves
- [ ] locality-degradation curve

## Ablations

- [ ] w/o Historical Memory
- [ ] Active-Only Historical Memory
- [ ] Precision-Only Historical Memory
- [ ] DynaSyn-v2.2-NR
- [ ] w/o Cooldown
- [ ] optional gradient-imprint initialization

## Resource / Hardware

- [ ] P/H bytes reported
- [ ] Adam state bytes reported
- [ ] persistent-state growth / task reported
- [ ] train wall-clock reported
- [ ] peak HBM reported
- [ ] inference latency reported
- [ ] Qwen3-1.7B real-shape sparse kernel benchmark
- [ ] Llama-3.2-1B real-shape sparse kernel benchmark
- [ ] Qwen3-8B real-shape sparse kernel benchmark
- [ ] dense-vs-sparse parity
- [ ] dense-vs-sparse speed ratio

## Reproducibility

- [ ] frozen_config.yaml
- [ ] baseline_manifest.yaml
- [ ] calibration_manifest.yaml
- [ ] trace_order_manifest.yaml
- [ ] longcl_manifest.yaml
- [ ] seed_manifest.yaml
- [ ] claim_boundary.yaml
- [ ] git commit
- [ ] environment versions
- [ ] fixed random seeds
- [ ] external baseline commits pinned
- [ ] code diffs saved
- [ ] final test only after freeze
- [ ] final test 不反向修改配置
