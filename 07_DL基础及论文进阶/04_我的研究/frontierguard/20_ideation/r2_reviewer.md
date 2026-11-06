# r2-reviewer｜敌意审稿 + 定位重构：21 死于对象层，唯一活路是"配对律"

**判词**：21 的骨架（部分识别 + 证书宽度 + 分配错配）不是证据不足，是**对象没定义**。要上 3.95，必须把对象从"认证组合成功"换成"**组合的可测量性**"，并给出一条指数→多项式的测量律。

## 1 六位审稿人（原话式）

**ICLR-R1 理论**："T1 是 Fréchet(1951) 在乘积空间上的推论，'边际不决定联合'是 75 年前常识。T2 只有上界没有匹配下界，H²(R+1)² 不是速率是断言。T4 的 n∝v^{1/3} 是可分凸目标上的 Lagrange 乘子，属经典最优实验设计。更严重：§3 自认五条命题待证——ICLR 不接受待证定理。"（4，拒）

**ICLR-R2 实验**："四技能、26 runs、一张卡，自称 skill library。E1 每计划 200 条审 95% 覆盖率，二项标准误 1.54pp，**结构上分不清 93% 与 97%**，核心覆盖率主张功效不足。主臂 scripted，LLM 臂 480 eps。"（3）

**ICLR-R3 意义**："信了全部定理，周一我做什么？证书宽 H(R+1)ε、联合探测每查询 H 倍成本，却没有'abstain 还是探测'的判据。"（3）

**CoRL-R1 严谨**："q_i 是什么？若是孤立成功率，β 不是修正项而是研究对象；若是 plan-induced，它不可能从孤立 rollout 得到，则 T1 反驳的'marginal-only 估计器'不存在。**两种读法下 T1 都是稻草人。** C-LGA 指数 v^{1/2}q^{−2/3} 与 T4 的 v^{1/3}q^{−2/3} 不一致：方法与定理互不相干。"（3，拒）

**CoRL-R2 机器人**："真贡献是复合环境（跨技能状态延续、per-skill 判定、入口态重放），却只有一段；S7 若 p≈0.3，组合缺口被单技能支配，'库级'塌回单技能。"（3）

**CoRL-R3 影响**："跑过两技能链的人都猜到边际校准不决定组合；H=3 的 8pp 缺口是常识不是发现。"（2）

**一票否决 = CoRL-R1 的对象攻击**：加实验救不了，一段话可验，且同时杀死 T1/T2/T4。ICLR 侧另有流程否决：**待证定理**。

## 2 五种叙事（S 上限 / 最强攻击 / 预驳）

**A–D 的共同病根**：都在问"给定边际还能说什么"（对已有数据的识别推论），不问"**该测什么、测多少**"（测量设计）。E 换的是问句。

| 叙事 | 对象 | 上限 | 最强攻击 | 预驳 |
|---|---|---|---|---|
| A 校准不组合 | Fréchet 推论 | 3.55 | 1950s 教科书 | 改成"测量不可行"才活→即 E |
| B 证书方法 | plan 级 PAC 证书 | 3.65 | conformal 套壳 | 组合诱导偏移破坏可交换性；仍是 CP 标准扩展 |
| C 证书分配 | n∝v^{1/3}≠Neyman | 3.70 | Lagrange 练习 | 方向反转真，但属凸性推论 |
| D 认证独立统计问题 | 匹配界+相变+验证 | 3.78 | 匹配界=自造预算划分上的 Hoeffding/Le Cam | 成本结构是真设计空间，仍答不出"对象是什么" |
| **E 配对律** | 计划的**测量维度** | **3.92** | "Markov 组合+Assouad 计数，K² 就是自由参数个数" | 见下 |

**E**：设 r_{ij}(μ)=技能 j 从技能 i 诱导的入口分布 μ 进入的成功率（**接口响应核**）。**T1 识别**：全部 K^H 个计划的成功由 K 个边际 + K² 个接口核确定。**T2 匹配界**：H=2 时排序到 margin Δ 需且仅需 Θ(K²/Δ²)；一般 H 为 O(K²H²/Δ²)，逐计划 Θ(K^H/Δ²)——**指数变多项式**；K^{H−2}>H² 即配对更省（H=4 时 K=4 打平、K≥5 占优；H=3 交叉点 K≈9）。**T3 相变**：|S(π)−∏q_i|≤H·L·κ ⇒ **κ*=Δ/(2HL)**；κ≤κ* 时乘积/边际校准 minimax 最优，κ>κ* 时乘积形可证错 ≥Δ/2。**预驳**：①"successor features/GPI 而已"→SF 假设奖励线性、动力学不变，我们给的是测量预算的 minimax 率与相变（1606.05312 已核）；②"Markov 组合已知"→已知的是表示，未知的是**要测多少对、何时可以不测**；③"4 技能 toy"→4 技能正处交叉点左侧，故只用于测 κ，律由 K=20–100 合成扫描验证——是设计不是缺陷。

**E 相对 21 的三处质变**：①对象——从"给 S(π) 发证书"（派生量）改为"计划泛函的测量维度"（可证伪的率）；②不可能性——从"联合≠乘积"改为"少测一对就排不对序"，与 ECE 无关；③实验——4 技能从"证据"降为"测 κ 的实例"，律由合成扫描承担，正好绕开"4 技能是 toy"。**S 归因**：N +0.55／TQ +0.10／Exp +0.15／Ins +0.65／C +0.05／Rep −0.05 ⇒ **ΔS≈+0.28–0.32 → 3.88–3.92**。**W8 生死步**（1 周、0 租卡）：16 个有序对（含自配对）×300 eps 测 r_{ij} 与 κ；若 κ 的 95% CI 全在 κ* 左侧则 E 死，退 D（3.78）。

## 3 第一页重构

**Title**：*Pairs, Not Products: The K² Measurement Law and Interface Phase Transition of Skill Composition*

**Abstract（148 词）**：Skill-library papers report per-skill success and predict multi-skill plans by multiplying them. We show this practice has a measurable cost. Plan success over K skills and horizon H is a functional on K^H plans, yet its sufficient statistic is the K² pairwise interface responses—the success of skill j entered from the state distribution that skill i induces. We prove Θ(K²/Δ²) measurements are necessary and sufficient to rank plans to margin Δ at H=2, and O(K²H²/Δ²) in general, versus Θ(K^H/Δ²) plan-wise: the exponential plan space collapses to a polynomial measurement budget. We then locate the boundary: with κ the interface deviation and L its Lipschitz constant, κ*=Δ/(2HL). Below it the product of marginals is minimax-optimal and per-skill calibration suffices; above it the product form is provably wrong by Δ/2. A synthetic sweep over K=20–100, H=2–5 confirms the predicted slopes, and four dexterous skills measure κ and straddle the boundary.

**三贡献**：①识别——计划成功泛函的充分统计量是 K² 接口核，不是 K^H 计划；②匹配界——Θ(K²/Δ²)（H=2）/O(K²H²/Δ²)，逐计划 Θ(K^H/Δ²)；③相变——κ*=Δ/(2HL)，把"校准可组合吗"化归为可测标量，附测量协议与 4 技能落点。

**Fig.1**：(a) log-log 测量预算 vs K：逐计划（斜率 H）/配对律（斜率 2）/下界；(b) (κ, Δ/(2HL)) 相图，4 技能为点，颜色=实测 |S−∏q|。

**差异表**：P2P 规模预训练、默认可组合｜ADEPT 预/后训练、per-skill｜OmniReset reset 多样性、per-task｜SGS/SER 采样调度、无计划对象｜ReMix 数据配比、无决策对象｜Decision calibration 单预测器、无组合。

## 4 单会 >50%？

**边际可行，但不是"过 50% 的方案"。** E 点估计 S 3.88–3.92，exec 0.87–0.89（砍掉 26-run 网格、W8 一周可判生死，执行风险显著低于 21）。CoRL：σ(1.15(3.90−3.66))=57%×0.88≈**50%**（区间 46–54%）；ICLR：σ(1.15×0.16)=55%×0.88≈**48%**。单会是掷硬币；写成 >50% 即自欺。**最诚实替代**：CoRL 2027 主投（W31，8 月出结果）→ 未中按审稿改投 ICLR 2028（W46），**级联 55–58%**；若 W8 测得 κ<κ*，转 NeurIPS D&B 发"接口测量卡"，3.75 档。

## 5 核验（arXiv API，2026-11-02 实查）

2606.26428 P2P｜2608.19182 ADEPT｜2603.15789 OmniReset｜2610.12465 SGS｜2511.21011 SER｜2107.05719｜2510.23471｜1606.05312 SF｜1905.10681｜2401.03360。ReMix=PMLR v270:145–164（会议，无 arXiv，未编 ID）。Fréchet 1951／Hoeffding 1942／Manski 1990／Assouad 1983 按经典期刊引。
