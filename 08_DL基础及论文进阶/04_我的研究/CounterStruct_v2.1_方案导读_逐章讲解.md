---
title: "CounterStruct v2.1 方案导读 · 逐章讲解"
tags:
  - CounterStruct
  - 持续学习
  - 动态稀疏
  - 结构可塑性
  - 论文导读
created: 2026-08-21
updated: 2026-08-22
---

# CounterStruct v2.1 方案导读 · 逐章讲解

> 严格沿着 [[CounterStruct_v2.1_纯实验方案_正式版]] 的逻辑，从 DL 初学者视角逐章解释。方案主线是：在 exact-2:4 topology graph 上，通过真实 shadow intervention 学习 integration-horizon structural action value，再用 task-localized state memory 做持续学习。

你可以先把整个问题理解成一个非常具体的问题：

> **一个大语言模型在不断学习新任务时，能不能在「不增加连接数量」的情况下，主动把一部分旧连接换成更适合当前任务的新连接，而且换之前先做一点「小规模试验」，判断这次换线到底值不值得？**

前置概念没补齐的，先看 [[CounterStruct_DL入门课_前置知识]]。

## 目录

- 第 0–3 章：版本定位 / 核心问题 / 边界 / Related Work
- 第 4–10 章：模型范围 / Topology Graph / 初始拓扑 / 状态 / Replay / 训练 / Mutation Schedule
- 第 11–16 章：Probe / Cheap Prior / 特征 / Shadow Bundles / 计算优化 / Horizon Label
- 第 17–25 章：Critic / Memory R / 最终 Utility / 执行 Transition
- 第 26 章：完整伪代码
- 第 27–36 章：Benchmark / Baselines / Generalization
- 第 37–45 章：LFS / Comparator / 评估矩阵 / 机制实验
- 第 46–52 章：Macro Bridge / Ablations
- 第 53–61 章：行为指标 / Memory 账本 / 统计 / Go-No-Go / 协议
- 第 62–69 章：可复现 / 五张图 / 成功与失败定义

---

## 方案第 0 章：版本定位

标题：**CounterStruct: Horizon-Calibrated Structural Action Values for Fixed-Capacity Continual Learning**。

三个关键词：

- **Horizon-Calibrated**：不是只看 $t+1$，而是关心 $t+H$。
- **Structural Action Value**：不是给单个 weight 算 saliency，而是给 topology transition 算价值。
- **Fixed-Capacity**：active connection 总数固定，只允许重新分配。

v2.1 明确把 v1.0 的 one-step AdamW + Fisher-style memory 降级为 LFS baseline，新的主创新集中在 horizon action-value learning 与 task-local state memory。

---

## 第 1 章：核心研究问题

方案实际问四个问题：

- **Q1 结构变化应该怎么表示？** 传统是 prune p + grow g，CounterStruct 是 state $s \rightarrow$ state $s'$。
- **Q2 短期 gradient 能不能代表未来价值？** 它怀疑 instant saliency 不足以预测 integration-horizon value。
- **Q3 bundle supervision 能不能反推出 action ranking？** 真实监督是 64-action bundle，最后需要 per-action ranking，所以必须验证局部可加性。
- **Q4 当前任务学到的结构偏好能不能压缩成历史记忆？** $T_t$ 学习时形成 $C_t$，之后 $T_{t+1}, T_{t+2}$ 还能不能靠它保护 $T_t$？

---

## 第 2 章：Contribution Boundary

这章主要是在保护论文不乱吹。允许说：用 sparse real shadow interventions 学 horizon structural value。

不能说：我们首次做 dynamic sparsity、Johnson graph 是创新、Critic 精确预测未来 accuracy、$R$ 就是真实 past loss landscape。这一章对你以后写论文特别重要。

---

## 第 3 章：Related Work 边界

这一章不是算法实现，而是在回答 reviewer「你跟别人有什么不同？」

- **RigL / SRigL**：它们是 coordinate saliency，CounterStruct 是 state-transition value。
- **LookAhead / RL Compression**：不能说第一次考虑未来，别人早就考虑过 future information。真正差别是 fixed N:M topology graph 上进行 online weight-level shadow intervention supervision。
- **SMET**：更关注 newborn weight 怎样优化稳定；CounterStruct 关注哪个 newborn transition 值得发生。若实现中出现明显 newborn instability，应把 SMET-style warm-up 作为独立 ablation，不能把稳定化效果误归因给 horizon critic。
- **EWC / Fisher**：CounterStruct full 不使用 $P, H$；LFS 保留它作为 predecessor。
- **OSFT / PaRSP**：它们靠 subspace / region protection 减少历史干扰；CounterStruct 不「保护参数」，而是学习 topology state transition 的 action value。
- **Bi-Level DST / 2:4 mask 优化**：「DST 是双层优化」「穷举 2:4 合法 mask」都已有工作做过，禁止单独当 novelty——六状态表示的价值在于给 action-value learning 和 structural memory 提供天然离散载体。

---

## 第 4 章：模型范围

Primary 是 **Qwen3-1.7B**，只改最后 8 层 FFN 的 `up_proj`、`down_proj`。

总 candidate $201,326,592$，active $100,663,296$。

Scale backbone 是 **Qwen3-8B**，只改最后 2 层，但特意让 candidate coordinate 数仍为 $201,326,592$——这相当于控制 structural search-space size。

---

## 第 5 章：2:4 Topology Graph

每组 $q = [w_1,w_2,w_3,w_4]$，状态 $\{12,13,14,23,24,34\}$。只允许 one-swap：$12 \rightarrow 13$ 合法，$12 \rightarrow 34$ 不允许一步完成。

这样每个 action 的结构成本都相同 $c(a)=1$。这是 v2.1 相比 v2.0 的重要简化。

---

## 第 6 章：Initial Topology

所有 structural methods 都使用相同 $M_0$。规则：每个 4-group 保留 pretrained magnitude 最大的两个；dormant 数值置 0；active cooldown 初始化为 $C=2$（一开始就允许被 prune），dormant cooldown 初始化 $C=0$。

例如 pretrained $[0.9, 0.2, -0.7, 0.1]$，absolute magnitude $[0.9, 0.2, 0.7, 0.1]$，保留 $w_1, w_3$，所以 $M = [1,0,1,0]$。这样所有方法从同一个 starting topology 开始。

---

## 第 7 章：Persistent State

CounterStruct 真正长期保存：$W, M, C, m, v, R, A_G, b_G, A_t, b_t$。可以记成三组：

- 模型状态：$W, M$。
- optimizer 状态：$m, v$。
- CounterStruct 状态：$C, R, \text{critic statistics}$。

---

## 第 8 章：Replay Policy

禁止：保存旧训练样本、旧 logits、每 task checkpoint、旧 gradient、per-task Fisher。

所以 future task 只能靠固定-size state。正确 claim 是 **history-example-free**，而不是 memory-free。

---

## 第 9 章：Normal Training

仍然 $W_{\text{eff}} = M \odot W$，而且训练本身是 **dense masked training**，不是 sparse training acceleration。这点后面 systems reviewer 很可能会问。

---

## 第 10 章：Mutation Schedule

每 task 在 $20\%,40\%,60\%,80\%$ 四次结构更新。为什么不是一直改？因为 topology change 本身有成本，而且 newborn connection 需要 integration 时间。Cooldown 也防止刚出生就被重新删除。

---

## 第 11 章：Current Dense Structural Probe

每次 mutation 用 $|B_{\text{probe}}| = 32$ 个样本，用 STE 获得 $g_i$（包括 dormant gradient）。这些 gradient 不直接更新模型，它们只是 **measurement**。

---

## 第 12 章：Cheap Analytic Prior

对 $p \rightarrow g$ 计算假想 AdamW update $\delta_p^{keep}, \delta_g^{new}$，然后：

$$
B^{(1)} = g_p(w_p + \delta_p^{keep}) - g_g \delta_g^{new}.
$$

直觉：第一项是保留旧连接 $p$ 下一步大概贡献多少，第二项是 newborn $g$ 下一步大概贡献多少，比较二者。

但注意 $B^{(1)} \neq U_{CS}$，它只是 feature。

---

## 第 13 章：10-D Action Features

Critic 不直接看所有模型状态，它只看一个 action summary $\phi(a) \in \mathbb R^{10}$。包括 $B^{(1)}, |w_p|, |g_p|, |g_g|, \dots$，以及 layer depth、matrix type、task progress。这是非常典型的 **hand-designed low-dimensional feature** 设计。

### 第 13.1 节：为什么用 percentile？

不同 task/event 的 gradient 量级可能完全不同（event A 的 $g=0.001$，event B 的 $g=0.1$），直接 pooling 很危险。于是改成「这个 action 在当前候选集合里排第几百分位」，映射到 $[-1,1]$。这样 critic 更像 **ranking model**，而不是绝对 loss predictor。

---

## 第 14 章：Shadow Bundles

每 event 8 个 bundles，每 bundle 64 个 mutually group-disjoint one-swap actions。所以一条 TRACE（v2.0 是每 event 4 条 bundle、共 128 条；v2.1 翻倍为 8 条，给 10 维 critic 更稳的冷启动监督）：

$$
8 \text{ tasks} \times 4 \text{ events} \times 8 = 256
$$

条 aggregate equations。

为什么分 8 个 strata？如果只随机抽 action，很可能大部分都是普通 action。所以先按 preliminary score 分 8 个档，从很差一直到很好，这样 critic 能看到 **更宽的 outcome range**。

---

## 第 15 章：Shadow Compute Optimization

如果每 event 真复制 9 个完整模型并同时运行会很贵。但前面的层被冻结，所以先算一次 frozen prefix hidden states 并 cache 下来，之后 Keep 和 8 个 Transition branches 只从 target region 继续跑。

目标 overhead ≤ 30%，超过 40% 则暂停大规模实验，先优化工程。

---

## 第 16 章：Horizon Label

Keep 得到 $L_K^{(8)}$，Transition 得到 $L_B^{(8)}$，label：

$$
y = \frac{L_K^{(8)} - L_B^{(8)}}{|L_K^{(8)}| + \epsilon}.
$$

如果 $y > 0$，说明 topology transition 更好。

---

## 第 17 章：Global Prior + Task-local Critic

这是算法最重要的一章之一。完整 predictor：

$$
\hat b_{t,H}(a) = (\theta_G^{pre} + \delta_t)^\top \phi(a).
$$

- **Global**：过去任务留下 $\theta_G$，代表通用 structural dynamics。它用 ridge 形式维护（充分统计 $A_G, b_G$，$\lambda_G=1$），初始 prior 即 one-step 假设（$B^{(1)}$ 系数 = 1、其余 = 0），所以冷启动时 critic 自动退化为 one-step score。
- **Local**：当前任务自己产生 shadow labels，先算 global residual $r_j = y_j - (\theta_G^{pre})^\top x_j$，再拟合 $\delta_t$。所以 $\delta_t$ 只学「当前 task 与过去通用规律不同的地方」。

为什么 task end 才 merge？先用 $\theta_G^{pre} + \delta_t$ 写当前任务 memory，然后才把当前 task 数据合并进 $A_G, b_G$。因果顺序更清楚：

```text
过去任务
 ↓
global prior
 ↓
当前 task shadow data
 ↓
local residual
 ↓
当前 task preference
 ↓
写入 R
 ↓
再变成未来任务的 global knowledge
```

---

## 第 18 章：Structural-State Memory $R$

每个 group $R_q \in \mathbb R^6$，例如 $R_q = [0.1, 0.6, 0.3, 0.8, 0.4, 0.2]$，对应六个 state。这个东西表达：历史任务总体来说对六种状态分别有多大结构代价。

---

## 第 19 章：Task-End Consolidation

task 结束时当前 task 数据马上要消失，所以这是最后机会。利用 $\theta_G^{pre} + \delta_t$ 评估当前 task 对 4 个邻居 state 的偏好，得到 $C_t$，然后更新：

$$
R^{(t)} = \frac{t-1}{t} R^{(t-1)} + \frac1t C_t.
$$

为什么 $\frac1t$？因为它想维护 **所有历史任务的等权平均**，而不是越老的任务权重越高。但坏处是很早的任务最终只占平均值很小比例，所以 Long-CL 必须检查 early-task erosion。

一个细节：critic 只评价当前 state 的 4 个 one-swap 邻居，不评价互补 state（如 $12$ 的互补 $34$）。为了让 6 维 $R$ 完整，方案把互补 state 的代价定义为那 4 个邻居代价的平均（graph-harmonic completion），这只用于补全 memory，真实 mutation 仍只允许 one-swap。

---

## 第 20 章：Historical Damage

假设 $R(12) = 0.2$，$R(13) = 0.7$，考虑 $12 \rightarrow 13$，那么：

$$
D_R = 0.7 - 0.2 = 0.5.
$$

意味着历史任务总体认为 13 比 12 更差。

---

## 第 21 章：最终 Utility

全文最核心公式：

$$
U_{CS}(a) = \hat b_{t,H}(a) - D_R(a) - \beta \sigma_t(a),
\qquad
\beta = 1.
$$

其中不确定性项：

$$
\sigma_t(a)
=
\sqrt{
\phi(a)^\top
\left(
A_G^{-1}+A_t^{-1}
\right)
\phi(a)
}.
$$

你以后写代码时可以直接把它记成：

```text
score =
    future_current_task_value
    - historical_damage
    - uncertainty_penalty
```

翻译成人话：**最终分数 = 当前任务未来收益 − 历史任务风险 − 我有多不确定**。这就是整篇论文最值得记住的公式。

顺带说一下 $\sigma_t$ 的直觉：它就是前置知识第 36 章那个 ridge 回归解 $(X^\top X+\lambda I)^{-1}$ 的「杠杆值」。某个 action 的 feature 方向如果落在已有 shadow 数据覆盖很少的区域，$\phi^\top(A_G^{-1}+A_t^{-1})\phi$ 就大，说明 critic 对这类 action 没什么把握。方案明确说它只是 leverage proxy，不是 Bayesian posterior CI。

---

## 第 22 章：Preliminary Score

Shadow sampling 发生在 critic 更新之前，所以还没有当次最新的 critic，因此需要 $S_{\text{prior}}$ 先粗略分层。但是 $S_{\text{prior}}$ 永远不能用于最终 topology selection，最终必须用 $U_{CS}$。

---

## 第 23 章：Group-Wise Search

每个 group 当前 state 只有 4 个邻居，算 $U_1, U_2, U_3, U_4$，取 $a_q^\star = \arg\max U$。如果 $\max U \le 0$，这个 group **不改**。

---

## 第 24 章：Edge Budget

不能无限 rewiring。定义：

$$
B_{\text{edge}} = \lfloor \rho_{\text{edge}} N_{\text{active}} \rfloor.
$$

例如 $\rho = 1\%$，那每 event 最多替换 1% active edges。所有 structural baseline 使用相同 budget，这是公平比较的重要条件。

---

## 第 25 章：真正执行 Transition

被删：$M=0, w=0, m=0, v=0, C=0$。新生：$M=1, w=0, m=0, v=0, C=0$。这一步会真的改变真实模型。

注意 $R$ 不会马上改，历史 memory 只在 **task end** 更新。

---

## 第 26 章：完整伪代码

这一章你以后写工程时最重要。整个方法就是：

```text
for task:

    冻结 global critic
    初始化 local residual

    正常训练

    到 mutation event:
        1. STE probe
        2. 枚举 topology actions
        3. 做 8 个 shadow bundle
        4. 跑 H=8
        5. 更新 local critic
        6. 算 U
        7. 选 mutation
        8. 真正改变 topology

    task end:
        用 localized critic 形成 C_t
        写入 R
        再把当前 task shadow data merge 到 global critic
```

如果这段逻辑你能自己复述，方法部分基本已经入门了。

---

## 第 27–30 章：四套 Benchmark

- **TRACE-8**：Primary，8 tasks（C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten），每 task 5000 样本，epochs 依次 $[5,3,7,5,3,5,5,7]$，AdamW、LR $1\times10^{-5}$、cosine、effective batch 32，负责主要方法证据。
- **TRACE Order-2**：同样任务换顺序，回答「是不是只对一个 task order 有效？」
- **Seq-GLUE-7**：不同 benchmark family，回答「是不是只对 TRACE 有效？」
- **Long-CL-15**：15 tasks，回答「长期学习会不会 freezing / early erosion？」

方案特别要求同时检查 structural freezing 和最早任务 retention。

---

## 第 31 章：Primary Structural Evidence

最重要 structural baselines：Dense Regional FT、Static 2:4、SRigL、IPGH、LFS、CounterStruct。

Primary seeds 为 5（42–46；Dense Regional FT 只跑 3 seeds）。三组 primary contrast：

1. **CounterStruct vs LFS**——最重要的新颖性 contrast；
2. CounterStruct vs SRigL；
3. LFS vs IPGH——验证 complete-action 前提本身是否成立。

为什么 CounterStruct vs LFS 最关键？因为 LFS 已经包含旧版 one-step + Fisher 思路，如果 CounterStruct 不能超过它，horizon critic 的新颖性很难站住。IPGH 则是专门构造的 matched independent prune/grow heuristic baseline，用来排除「只是 heuristic 组合方式不同」的解释。

---

## 第 32 章：External Baselines

这部分回答「和 CL 社区其他路线比怎么样？」mandatory suite 是 Naive FT、LoRA、O-LoRA、Meta-UCF、OSFT、PaRSP、Any-SSR，全部 3 seeds。这里不要求它们参数形式和 CounterStruct 一样，重点是使用统一 backbone / task protocol 公平比较，且必须做 official-code / replay / task-ID / contamination audit。GORP / TreeLoRA 降为可选 appendix——方案明确说「不用 baseline 数量替代机制证据」。

---

## 第 33–36 章：Generalization / Scale

- Seq-GLUE：benchmark transfer。
- Long-CL：long horizon。
- Order-2：order robustness。
- Qwen3-8B：scale-direction replication，3 seeds。

8B 只能说「方向在更大 backbone 仍成立」，不能说「已证明 scaling law」。

---

## 第 37 章：LFS Baseline

LFS = v1.0 核心，包括 one-step AdamW + $P,H$ + Fisher-style memory。它是非常重要的「自我对照」。

---

## 第 38 章：EWC-DR-style Comparator

这是进一步回答：会不会只是 LFS 的 Fisher estimator 太差，所以 CounterStruct 才赢？所以换更强的 importance estimation 做 secondary comparator。

---

## 第 39 章：Evaluation Matrix

训练到 task $i$ 后，把目前见过的所有 task 都评一次，得到 $A_{i,j}$。例如 $A_{5,2}$ 表示「学完第 5 个任务后，在第 2 个任务上的成绩」。这张矩阵是 continual learning 的基础。

---

## 第 40 章：General Capability

除了 CL benchmark，还要检查模型原本通用能力是不是被 2:4 pruning 直接毁掉了。所以分解 $W_{\text{dense}} \rightarrow M_0 \rightarrow M_T$，从而区分初始 pruning damage 和 continual-training damage。

---

## 第 41–43 章：Temporal Calibration

这几章是论文最核心的机制实验。固定 checkpoint（$T_4@60\%$ 和 $T_7@60\%$，seeds 42/43/44），构造 held-out calibration bundles：按 predicted-value 分 10 个 decile、每 decile 3 个独立 family，primary $K=64$、secondary $K=256$，且全部与 online shadow-training equations 完全 disjoint。然后真实跑 $H=1,8,32,64$，比较三种 predictor：

$$
\rho_{\text{RigL}}(H),
\qquad
\rho_{\text{1step}}(H),
\qquad
\rho_{\text{critic}}(H).
$$

目标：

$$
\rho_{\text{critic}}(64) \ge 0.30,
\qquad
\rho_{\text{critic}}(64) - \rho_{\text{1step}}(64) \ge 0.10.
$$

反过来，若 $\rho_{\text{critic}}(64) < 0.20$，则 horizon-predictive 主 claim 直接失败（对应 G4 门禁）。这其实是在真正验证论文标题里的 **Horizon-Calibrated** 有没有成立。

---

## 第 44 章：Aggregate Credit Validity

这是第二个生死实验。因为 critic 从 64-action bundle 学习，所以必须验证 action interaction。

Micro-bundle audit 在每个 checkpoint × seed 选 16 对匹配的 micro-bundle 对 $(A,B)$，每对 $|A|=|B|=8$、内部及彼此全部 group-disjoint。从同一 checkpoint 分 4 个 branch：Keep、A-only、B-only、$A \cup B$，primary 跑 $H=8$，另取 4 对跑 $H=64$ 做 stress test。算：

$$
I_{A,B} = Y_{A \cup B} - Y_A - Y_B.
$$

再定义 normalized $\Gamma$（interaction 相对 $|Y_A|+|Y_B|$ 的中位数比例）。预注册解释分三档：$\Gamma(8) \le 0.25$ 视为 strong local-additivity support；$0.25 < \Gamma(8) \le 0.50$ 视为 approximate / noisy support；$\Gamma(8) > 0.50$ 则 **additive critic interpretation 失败**。

---

## 第 45 章：Structural Memory Calibration

这章分两阶段：

- **45.1 Write-time fidelity**：当前 task 还没消失时，critic 认为某 state 好不好 vs 实际 shadow branch 结果相关吗？得到 $\rho_{\text{write}}$。
- **45.2 Future historical fidelity**：以后某时刻，$R$ 预测过去任务会不会受损 vs 真正拿 past-task evaluation data 去测的 damage 相关吗？得到 $\rho_R$。

这是非常漂亮的两阶段证据：

```text
当前任务写入时是否靠谱？
           ↓
过了几个任务以后仍然靠谱吗？
```

---

## 第 46 章：Macro-Horizon Bridge

即使 $\rho = 0.5$ 也可能最终 ACC 没改善。所以必须回答「局部 action predictivity 是否真的转化成 continual learning 改善？」比较 Full、w/o Critic、LFS。如果 full 不赢，不能说 critic 是最终 CL improvement 的原因。

---

## 第 47–52 章：Ablations

建议你以后按一个问题来记：

- **A1 w/o Horizon Critic**：看未来真的有必要吗？
- **A2 w/o $R$**：历史 structural memory 有必要吗？
- **A3 Fisher Memory**：state-level memory 真比旧 Fisher memory 好吗？
- **A4 w/o Task-Local Residual**（$\delta_t = 0$）：每个 task 自己校准 critic 有必要吗？
- **A5 w/o uncertainty**（$\beta = 0$）：保守 admission 有用吗？
- **A6 $H=1$**：8-step shadow 相比 one-step shadow 真有价值吗？

---

## 第 53 章：Structural Behavior Metrics

不只看 ACC，还记录算法自己内部发生什么：positive-U、edge turnover、$\|\delta_t\|$、critic uncertainty、topology Jaccard。

为什么？因为如果结果坏了，需要知道坏在 critic、坏在历史 memory、还是根本没人愿意 mutation？

---

## 第 54 章：Memory Accounting

一共有 $50,331,648$ 个 2:4 groups，每 group 6 个 state values，BF16 下：

$$
50,331,648 \times 6 \times 2 \ \text{bytes} = 0.604\ \text{GB} \approx 0.563\ \text{GiB}.
$$

比 LFS 的 $P,H$（1.61 GB）小约 2.67×。但 0.6GB 仍然不是「免费」。

---

## 第 55 章：Memory-Matched Replay

这是一个非常公平的问题：你用了 0.6GB structural memory，那如果把同样 0.6GB 给 replay 方法存旧样本会怎样？所以允许 replay 保存 $\le$ CounterStruct memory bytes 的数据，目的是展示 Pareto tradeoff，不是强行要求 CounterStruct 一定赢。

---

## 第 56 章：Statistics

Primary structural 用 5 seeds，其他很多 3 seeds。Primary 的 CounterStruct vs LFS、CounterStruct vs SRigL 用 10000 次 hierarchical bootstrap。这部分是在降低「只是随机 seed 碰巧好」这种可能性。

---

## 第 57 章：Practical Effect Threshold

统计显著不等于实际重要。所以规定至少 $\Delta ACC \ge 1.0$ 或 $\Delta Forgetting \le -1.0$，否则即使方向稍好，也只能写 **weak effect**。

---

## 第 58 章：Go / No-Go

这是整个执行计划最重要的章节之一。方案把关键假设预先设成 13 道门禁（G0–G12），分三档：

**正确性与前提（G0–G2）**：G0 是正确性（exact 2:4 合法、branch 隔离、optimizer/RNG 恢复、deterministic shadow replay 复现等）；G1 是 dynamic topology 本身要 work（SRigL 相对 Static 有 plasticity trend）；G2 是 LFS baseline 要实现 v1.0 预期行为。

**核心机制生死线（G3–G8）**——最关键的六道：

- **G3** Shadow signal 能不能测出来（超过 noise floor、可复现、有正负 outcome）？
- **G4** Critic 对 $H=64$ 有没有预测性（$\rho \ge 0.30$ 且 CI>0；$<0.20$ 则失败）？
- **G5** Critic 是否比 one-step 更好（$\Delta\rho \ge 0.10$）？
- **G6** Bundle interaction 是否够小（$\Gamma(8) \le 0.50$，理想 $\le 0.25$）？
- **G7** Memory 是否真的有用（$\rho_{write}>0$ 且 $\rho_R>0$）？
- **G8** Full 是否打败 LFS（达 practical threshold 且 AG 不劣）？

**迁移与资源（G9–G12）**：G9 是 Order-2/Seq-GLUE 方向一致；G10 是 Long-CL 不 freezing、不 hidden early-task erosion；G11 是 8B 三 seed 方向一致；G12 是资源上限（$R\le0.7$GB、shadow overhead ≤30%、硬上限 40%）。

如果 G3/G4/G5/G6 掉了，不值得继续烧几百 GPU-hours 跑外围实验。

---

## 第 59 章：Development Protocol

只能用 Qwen3-1.7B × TRACE Order-1 × seed42 开发方法。允许调少量东西，但不能看完所有 final seed 再反过来改算法。这是防止 **test-set overfitting / research overfitting**。

---

## 第 60 章：Freeze Protocol

当方法定下来之后，feature、H、K、critic、$R$、task order、baseline、threshold 全部冻结。之后多 seed 只是在 **验证预先确定的方法**，而不是继续开发。

---

## 第 61 章：Execution Priority

整个项目按 $P0 \rightarrow P1 \rightarrow P2 \rightarrow P3 \rightarrow P4 \rightarrow P5$：

- **P0** 最重要，只验证核心科学假设有没有信号（branch correctness、shadow noise floor、critic vs one-step、interaction、memory write）。
- **P1** 如果 P0 活着，建主论文证据。
- **P2** robustness。
- **P3** external baselines。
- **P4** 8B 规模复现。
- **P5** appendix（EWC-DR comparator、no uncertainty、H=1 shadow、memory-matched replay、可选 Llama/GORP/TreeLoRA）。

方案明确规定如果 G3/G4/G5/G6 明显失败，就不要启动完整 external / 8B。

---

## 第 62–64 章：Artifacts / Logs / Hardware

这些章节不是理论，是为了 **reproducibility**。例如每次 run 保存 config.yaml、seed、git commit、model revision、metrics、mask、critic state、shadow labels、$R$、profiling，这样几个月之后还能回答「当时这个结果到底怎么跑出来的？」

---

## 第 65 章：论文五张主图

- **Figure 1** 方法图：2:4 topology → shadow → critic → $R$ → action。
- **Figure 2** 最重要机制图 $\rho(H)$：Critic 真的比 instantaneous saliency 看得远吗？
- **Figure 3** Macro bridge：看得远真的让 continual learning 更好吗？
- **Figure 4** Long-CL：学久了会 freezing 或忘掉最早任务吗？
- **Figure 5** Performance–Memory–Compute Pareto：性能更好需要付多少 memory / compute？

---

## 第 66 章：论文怎样才算成功？

最理想不是所有 benchmark SOTA，而是形成一条完整因果证据链：

$$
\boxed{\text{one-step 不够}}
\downarrow
\boxed{\text{critic 确实能预测更长 horizon}}
\downarrow
\boxed{\text{bundle interaction 足够小}}
\downarrow
\boxed{\text{task-local preference 可以写进 } R}
\downarrow
\boxed{\text{full method 最终比 LFS 更好}}
\downarrow
\boxed{\text{Long-CL / Order-2 / 8B 仍保持}}
$$

---

## 第 67 章：失败也要提前定义

这是这份方案很成熟的一点。例如：

- Critic 只预测 H=8 → 不能说 long-horizon structural value。
- Critic 很准但 CL 不提升 → predictivity 没转化成 continual benefit。
- Interaction 太强 → bundle 不能分解成 per-action value。
- $R$ 写入失败 → structural memory 没有 empirical grounding。

方案都已经提前写清楚。

---

## 第 68–69 章：最终 Checklist 与项目定义

项目真正的统一主线是 **horizon-calibrated structural action value**，不是 dynamic sparsity、不是「2:4 有六个状态」、也不是 Fisher 替换。

而是：

$$
\boxed{
\begin{aligned}
&\text{先对少量真实 topology changes 做平行实验，}\\
&\text{从它们真正训练一段时间后的结果学习 structural value，}\\
&\text{再用这个 value 指导当前 task 的 rewiring，}\\
&\text{并把当前 task 的结构偏好压缩成历史 memory，}\\
&\text{以后学习新任务时同时考虑当前收益与历史风险。}
\end{aligned}}
$$

这就是 CounterStruct v2.1 最准确的初学者理解。

---

## 总结

### 如果你只想记住 5 句话

1. **模型永远保持 exact 2:4，也就是每四条潜在连接只开两条。**
2. **它不只是问「哪个 weight gradient 大」，而是问「如果换这条连接并继续训练一段时间，这次结构变化最后值不值」。**
3. **它用少量真实 shadow branches 做平行宇宙实验，再训练一个很小的 critic 去预测所有候选 structural actions。**
4. **它用 $R$ 保存过去任务对六种局部 topology states 的结构偏好，防止新任务为了自己破坏旧任务。**
5. **最终换不换连接由 $U = \text{当前长期收益} - \text{历史损害} - \text{不确定性}$ 决定。**

### 四个生死线

| 生死线 | 如果失败意味着什么 |
|---|---|
| Shadow signal 测不出来 | 根本没有足够 supervision |
| Critic 不比 one-step 强 | 核心 novelty 失败 |
| Bundle interaction 太强 | 不能从 bundle 学 per-action value |
| Full 方法不比 LFS 强 | critic 机制没有转化成最终 CL 收益 |

（这四条分别对应方案 G3 / G5 / G6 / G8；完整门禁共 G0–G12，见上面第 58 章。）

这个研究并不是「代码全部写完就一定有论文」，而是 **先跑 P0，验证核心科学假设是否真的存在**。如果不存在，就应该停止。

### 整篇论文脑图

```text
连续学习新任务
        │
        ▼
固定 exact 2:4 连接数量
        │
        ▼
现在要不要把旧连接 p 换成新连接 g？
        │
        ├──────── cheap one-step prior
        │
        ▼
做少量 shadow 平行实验
Keep vs Transition
        │
        ▼
真实训练 8 steps
        │
        ▼
得到 bundle-level outcome
        │
        ▼
训练 Horizon Critic
global prior + task-local residual
        │
        ▼
预测每个 structural action
未来一段时间后的收益
        │
        ├──────── 历史 structural memory R
        │                    │
        │                    ▼
        │              过去任务是否讨厌这个 state？
        │
        ▼
U = 当前收益 - 历史风险 - 不确定性
        │
        ▼
U > 0 ?
   ┌────┴────┐
   │         │
  Yes        No
   │         │
换连接       不换
   │
   ▼
继续训练
   │
   ▼
task end：把当前任务的结构偏好写入 R
   │
   ▼
学习下一个任务
```

### 知识树

```text
神经网络参数 W
│
├── Forward / Loss
├── Gradient / Backward
└── Optimizer
      └── AdamW
          ├── m
          └── v

Transformer
│
├── Attention
└── FFN
    ├── up_proj
    └── down_proj

Sparsity
│
├── Mask M
├── Active / Dormant
├── Prune / Grow
└── Structured 2:4
      └── 6 legal states

Continual Learning
│
├── Catastrophic Forgetting
├── Stability / Plasticity
├── Replay
├── Regularization
└── Parameter isolation

Dynamic Sparse Training
│
└── RigL / SRigL
    ├── magnitude prune
    └── gradient grow
          │
          ▼
核心问题：
instant gradient
真的等于 future structural value 吗？
          │
          ▼
CounterStruct
│
├── STE probe
├── one-step prior
├── shadow intervention
├── H=8 rollout
├── bundle label
├── ridge critic
│   ├── global prior
│   └── task-local residual
├── interaction audit
├── structural memory R
├── uncertainty
└── U_CS
      │
      ▼
topology mutation
```

前置概念见：[[CounterStruct_DL入门课_前置知识]] · 正式方案见：[[CounterStruct_v2.1_纯实验方案_正式版]]
