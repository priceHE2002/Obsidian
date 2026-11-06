# 21 · Round 1 汇总：从“校准测量”到“组合成功的部分识别”

**一句话结论**：10 个视角中唯一同时满足“新形式化对象 + 纯仿真 + 45 周/¥5,000 + 复用现有 4 技能资产”的是 lens_08 的 **CSC（Certified Skill Composition）**：把技能库的组合成功重述为**部分识别**问题，让证书宽度成为可设计变量，并推出“证书最优分配 ≠ 成功率最优分配”。CD-SNR（lens_03）作相变定理、DIL（lens_01）作可识别性证书、CAPS（lens_04）作 propose-certify 算法、DLA（lens_02）的 σ̂ 三义修复并入。估计 S 3.60→**3.96（3.90–4.02）**，质量 57–60%，exec 0.85–0.90，**CoRL 期望点估计 ≈51%（S 3.90–4.02、exec 0.85–0.90 下约 ±7pp）、ICLR ≈50%**；**只在 W6/W10 两个 gate 都过时成立**（条件概率 ≈60%），否则走 fallback。lens_05 未到，未纳入。

## 1. 候选排序（能否把 S 从 3.60 抬到 ≥3.95）

| # | 候选（lens） | ΔS／S | 增量成本 | exec | 主要风险 | 结论 |
|---|---|---|---|---|---|---|
| 1 | CSC 部分识别+证书（08 A） | +0.29~0.39／3.89–3.99 | 70–100 GPU-h，¥600–900，12–16 周 | 0.85–0.90 | 需“同 episode 状态延续”，W4 前必须验证；证书易宽 | **主线** |
| 2 | CD-SNR 相变+接口偏差（03 A） | +0.33~0.40／3.93–4.00 | 42 GPU-h，¥300，5 周，0 新训练 | 0.82–0.85 | 相变若不可见需双通道解释 | 并入主线（T5+§4） |
| 3 | CAPS 边际不可能性+证书（04 A） | +0.375／3.98 | 120 GPU-h，¥900，10–12 周 | 0.88–0.90 | BAI/回放成本“现成”、召回饱和 | 算法层（propose-certify） |
| 4 | DIL 决策可识别性（01 A） | +0.33／3.93 | 25 GPU-h，¥176，18 周 | 0.88 | 古典设计套壳、增益仅 2× | 证书层（“测不出”判据） |
| 5 | Recovery Value 单调律（09 A） | +0.375／3.98 | 3 GPU-h，¥260，5 周 | 0.90 | 单调律可能不成立（P≈35% 滚小） | 只作 §5 推论 |
| 6 | DLA 漂移受限分配（02 A） | +0.36／3.96（ICLR） | 24 人日，¥149 | 0.85 | 需先做 0-GPU 生死步 | σ̂ 修复+指数预测并入 §3 |
| 7 | 库级标定律（06 A） | +0.33／3.93 | 386 GPU-h，¥3,965，14 周 | 0.80–0.85 | 4–6 技能不足以立法；预算吃满 | 只取“共享 trunk 干涉” |
| 8 | 可移植契约（07 A） | +0.25–0.30 | 60 GPU-h，¥640，+40 人日 | 0.70 | 3 形态不足以立律 | 不作 |
| 9 | 测量+负结果（10 B／lead_seed B） | +0.16~0.25 | 0 新训练，3 周 | 0.90 | 天花板=poster | **fallback** |
| 10 | venue 策略（10） | +5pp only | 0 | — | 不改变贡献类型 | 只用于 framing |

判据：①只留一个核心形式化对象（避免“理论拼盘”）；②W1–W10 内必须完成可证伪的“生死步”，把执行风险压低；③主实验继续复用 4 技能（S1/S2/S7/S8）与 fixed-K 协议。07/06 单独都到不了 3.95，5 号定律方向存疑，均不入主线。

## 1.5 S 与概率重建（诚实账）

| 维度（权重） | 当前 | 新线 | Δ | 依据 |
|---|---|---|---|---|
| N 新颖性（.25） | 3.25 | 3.90 | +0.65 | 新对象“组合成功部分识别+证书驱动设计”，非加实验 |
| TQ 技术质量（.20） | 3.85 | 4.00 | +0.15 | 定理有闭式常数、覆盖率审计、合成 oracle 真值 |
| Exp 实验（.20） | 3.65 | 3.75 | +0.10 | 4 技能+2 长程+840 agent eps，规模仍小是硬伤 |
| Insight（.20） | 3.45 | 4.05 | +0.60 | 两条反直觉定理（校准只值窄带；证书最优≠成功率最优） |
| Clarity（.10） | 3.80 | 3.85 | +0.05 | 一个对象、一条主线，比 v3.2 清楚 |
| Repro（.05） | 4.40 | 4.35 | −0.05 | 去掉真机相关承诺；开源 benchmark+证书脚本 |

加权 ΔS≈+0.36 → **S≈3.96**。按 judge 模型 CoRL：质量 σ(1.15(S−3.66))=σ(0.345)≈**58.5%**，exec 0.87 → 期望 **≈51%**；ICLR（阈值 3.74）：σ(1.15×0.22)=σ(0.253)≈56.3%，exec 0.88 → **≈50%**。**执行风险折扣**：W6/W10 双 gate 各约 0.8/0.75 通过率 ⇒ 联合条件概率 ≈0.60；把“gate 失败后退回 v3.2 测量版（≈38%）”计入，**无条件期望 ≈0.60×51%+0.40×42%≈47–48%**。因此口径如实写：**“点估计 50–51%，跨线但不承诺；要稳过 50% 需要 W6/W10 都过早 gate 且 exec≥0.87”**。

## 2. 新 thesis 与就近工作差异

**Thesis**：per-skill 边际校准既不充分也不必要于组合决策；组合成功率应被当作**部分识别量**——用 sharp Fréchet 界给出可计算的最坏情形，用联合采样收紧，用证书宽度驱动“采哪个技能、测哪个计划”；由此得到两条与主流相反的可检验结论：**校准在 ρ=Δ/D 的窄带之外对决策无价值**；**证书最优分配 n∝v^{1/3} 与成功率最优 Neyman n∝√(wv) 不同**。

| 工作 | 他们优化/主张什么 | 本文差异（一句话） |
|---|---|---|
| Play2Perfect 2606.26428 | 单策略规模+真机，33× 样本效率 | 不竞争规模；我们问“更大库的组合成功能否被认证” |
| ADEPT 2608.19182 | 大规模预训练+视觉触觉后训练 | 目标仍是 per-skill 成功率；无 plan 级识别 |
| OmniReset 2603.15789 | reset 分布多样性 | 与我们正交；不提供联合识别 |
| SGS 2610.12465 | success-band 采样、2^20 envs | 分配信号是 success band；无 plan/证书对象 |
| SER 2511.21011 | 错峰 reset | 纯系统层；本文作 baseline 的非平稳控制 |
| LGA（本组 v3.2） | LP×σ̂ 技能级分配 | 本文把它降为采样器；σ̂ 改作证书宽度，不加乐观 bonus |
| ReMix（CoRL'24 oral, PMLR v270:145–164） | 数据配比优化（IL） | 无决策对象、无组合；本文是“估计→决策”而非“配比” |
| Decision calibration（Zhao 2107.05719；Kiyani 2510.23471） | 单预测器/部分校准预测的决策校准 | 我们把对象扩到 H 步×重试×跨技能依赖，并证明 marginal-only 不可能性；新增接口偏差 β |
| Conformal risk control 2208.02814 / CP-RL 2311.01457 | 单风险/单策略覆盖 | 我们给 plan 级、跨技能依赖的 PAC 证书；conformal 只用于 shift 项 |

**“不是 incremental”三句（模拟审稿人 + 反驳）**：①“conformal/shield 套壳”→ 贡献是 T1 的**识别不可能性**与 T4 的**分配错配**，conformal 只在 shift 项被用到，且我们同时报 assumption-free 界；②“边际校准早有人做（Zhao/Kiyani）”→ 他们做单预测器决策，没有 H 步×重试×跨技能依赖，且我们证“边际 ECE=0 仍不可识别 plan”；③“BAI/回放证书现成（lens_04 风险）”→ 我们把它降为执行层算法，主贡献是“证书宽度是可设计变量”及由此改写的采样/分配目标。

## 3. 形式化与 5 条命题（状态：待证明，W2–W4 完成；不声称已证）

**设定**：技能 g∈[K]，边际成功率 p_g(x)；计划 π=(g_1..g_H)，每步最多 R 次重试，则步成功 q_i=1−(1−p_i)^{R+1}，计划成功 S(π)=E[∏q_i]。边际数据只给 q̂_i 与置信区间；独立假设下的点估计 S_ind=∏q̂_i。

**假设**：A1 计划内事件可交换/静态（技能成功率在执行期内不漂移）；A2 重试同分布且不改变后续入口（R≤2 时近似成立，W6 检验）；A3 边际估计有界误差（Hoeffding）；A4 计划集有限（|Π|≤10²）且 planner 规则固定；A5 训练/组合入口分布偏移由 β 项显式建模（不假设可忽略）。以上任一条被违反而结论方向反转时，按 §9 的负结果分支报告。

- **T1（sharp 界 + 边际不可能性）**：对任意 H 个事件的边际 q_i，联合成功的 sharp 界为 L=max(0,Σq_i−(H−1))、U=min_i q_i（Fréchet；两端可由 copula 构造达到）。推论：当 Σq_i≤H−1 时 L=0，**任何只用 marginals 的估计器都不能认证 S>0**，与 ECE 是否为 0 无关。数值：H=3,q=0.7 ⇒ 乘积 0.343 vs L=0.10（差 24.3pp）；q=0.6 ⇒ L=0。**证明思路**：事件交的 Fréchet–Hoeffding 界+构造性下界。
- **T2（重试放大 + 证书样本复杂度）**：per-skill 误差 ε 传播为 |Ŝ−S|≤H(R+1)ε（v3.2 Prop 3a 降为推论）；要认证 L>0 需总误差 <m（m=Σq_i−(H−1) 的余量），即每技能 n≳2H²(R+1)²σ²/m²。**证明思路**：q 对 p 的 Lipschitz 常数 (R+1)+union bound；这是“证书要多少样本”的闭式，而非装饰性上界。
- **T3（依赖感知收紧与覆盖率）**：对被查询计划做 m 次**联合 rollout**，可直接估 S(π)（方差≤S(1−S)/m）并把界收窄到观测共失败质量允许的范围；用 latent-class/copula 的收紧标为 model-conditional，必须在 held-out 计划上过 misspecification 检验。**预测**：覆盖率审计中 marginal-only 证书在正依赖下覆盖率 <1−α（如 80% vs 95%）；联合采样后回到 ≥1−α。**证明思路**：Hoeffding/β-覆盖+构造反例。
- **T4（证书最优分配 ≠ 成功率最优）**：最小化证书宽度（Σ√(v_i/n_i) 型）给 **n_i∝v_i^{1/3}**；最小化乘积界的宽度给 n_i∝(v_i/q_i²)^{1/3}；而成功率方差的 Neyman 解为 n_i∝√(w_i v_i)。三者在 v 异质/低 p 区显著不同，且前两者对 p 的单调方向可与 ECE 最优相反（对应 judge 复算的 ECE 比 1.000：不是实验没做好，是口径错了）。**证明思路**：Lagrange 乘子+Cauchy–Schwarz。
- **T5（决策相变：校准何时有用）**：ρ=Δ/D，Δ=计划间 margin，D=√(2Σc_i²v_i/n_i)。若 Δ≥D√(2log(2|Π|/δ)) 则 P(误选)≤δ、regret=0；若 Δ≤cD 则 Le Cam 两点下界给 E[regret]≥c′Δ；**只有 ρ∈[c, √(2log(2|Π|/δ))] 的窄带里，校准/分配才影响决策**。推论（自证伪）：若现有 512k CRN 下 ρ≈20，则该基准原理上测不出校准效应，H2 的阈值叙事应撤销——这把“负结果”变成定理。**证明思路**：union bound+Le Cam 两点法。
  另：接口偏差 β_s=p_s^{comp}−p_s^{iso} 给出不随 n 消失的通道，|S−Ŝ|≤Σc_s(ε_s+|β_s|)；须在组合入口分布上重估 p（配额采样），这是 §4 的“接口覆盖配额”。

**可证伪预测（预注册）**：①真实 L1/L2 上 `|S_ind−S_joint|≥8pp`（<3pp 则 T1 的工程意义消失，转负结果）；②marginal-only 证书在正依赖计划的覆盖率 <90%（联合采样后 ≥95%）；③C-LGA 与 Neyman 的分配份额在 ≥50% 技能上差 >20%，且证书宽度缩 ≥15%；④当 ρ≥3 时重采样不改变 planner 选择（≥99%），即该基准“测不出校准”；⑤加入 β 项后残差 <1pp，无 β 时残差随 H 单调增。

**理论状态声明**：T1 用 Fréchet–Hoeffding+构造 copula（证明思路完整）；T2 由 Lipschitz+union bound；T3 由 Hoeffding+β-覆盖；T4 由 Lagrange；T5 由 Le Cam 两点法——均在 W2–W4 形式化，**本文不声称已证**；若 T5 的窄带在实验区间不可见，则按“β/漂移主导”的双通道诊断树报告（仍有故事）。

## 4. 算法与复杂度

**C-LGA（训练期分配）**：分数 s_g=LP_g×√(v_g)/(q_g²)^{1/3} 型证书权重（实现为指数 1/3 的 Neyman 变体），Level-2 技能内仍用 z-score；guards：n_g<30 冻结、σ̂>q90 冻结 50 更新、单技能>40%N 降温。**propose-certify（执行期）**：LLM/scripted 出候选集 Π̂（k≤10）→ 对每个计划做 m 次联合 rollout → 得 [L,U] 与 PAC 证书 → 淘汰 U<maxL 者 → 对宽度最大者追加 m 次 → 直到最优计划 L 超过其余 U（停时规则），否则 abstain/retry。**接口**：技能返回 (p̂, CI, ECE_dec, β̂, 证书配方)，planner 消费区间与证书而非点估计。**复杂度**：训练成本不变 O(B)；证书测试 O(|Π|·m·H) 步，占总采样 ≤5%；证书分配每 10 个更新一次，O(K)。**与用户三目标的映射**：①“大规模 RL 技能训练”= 4 技能各 500M/125M 步 fixed-K 训练（保留）；②“技能库级分配”= C-LGA 的证书口径分配 vs uniform/SGS/Neyman（G1a/G1b）；③“LLM agent 组合”= propose-certify + 证书盾的 scripted/LLM 双臂。三者共用同一 4 技能资产，不新增训练协议。

## 5. 实验矩阵与预算（≤¥4,800）

| 组 | 内容 | runs/规模 | GPU-h | ¥ |
|---|---|---|---|---|
| E0 | 合成 oracle（精确 copula，验证 T1/T4/T5、覆盖率） | CPU 10^4 库 | 0 | 0 |
| E1 | 覆盖率审计（5 计划×200 联合×2 任务×8 seeds） | 16k eps | 5 | 40 |
| E2 | 缺口分解（复用日志+200/plan） | — | 10 | 80 |
| E3 | 接口偏差 β（入口态重放） | 500 eps | 4 | 30 |
| E4 | 盾 A/B（random/conformal-only/cert/cert+shield×100 配对） | 400 eps | 2 | 300（LLM） |
| E5 | 偏移压力（3 档×2 任务×50） | 300 eps | 10 | 80+230（LLM） |
| G0 | 工程集成/调试（租卡） | — | 24 | 168 |
| G1a | 分配替换主格 N=8192：uniform/SGS-band/C-LGA×8 seeds + LGA-success 2 runs | 26 runs | 247.3 | 1,731 |
| G1b | 同上 N=2048 | 24 runs | 75.4 | 528 |
| 合计 | 租卡 377.6 h → ×1.12 缓冲=422.9 h | — | — | **¥3,660（余 ¥1,140）** |

**算式**：h/run=B(N)/15,000/3600+0.25，B=K·N·L（K=1221,L=50）：N=8192⇒9.26+0.25=**9.51h**；N=2048⇒2.90+0.25=**3.14h**；26×9.51+24×3.14=247.3+75.4=322.6h，加 E1–E5+G0=55h ⇒ **377.6h**；×1.12 缓冲=422.9h×¥7=**¥2,960**，+电 220+存储 80+LLM 400=**¥3,660（余 ¥1,140）**。**长程任务与 agent**：2 个 L 任务（L1=双件装配、L2=紧配合+扰动），**技能间同 episode 状态延续、不 reset**（W4 验证的红线）；双臂 = scripted planner（无 LLM、可复现）+ LLM planner（Qwen2.5-7B 本地 vLLM 4-bit 主扫，2 个 API 条件用 API 复现），每臂 2 任务×40 eps×4 API 条件≈320 eps，加 pilot 160 eps=480 eps。本地 120 GPU-h 免费（合成、日志重放、agent 评测）；LLM 预算 ¥400（480 episodes，硬顶 ¥500）。**削减链（任何情景逐级执行直到 ≤5,000）**：C1 砍 G1b（−75.4h，−¥591）→¥3,069；C2 砍 LGA-success 2 runs（−19h，−¥149）→¥2,920；C3 砍 E4/E5（−12h 租卡−¥530 LLM ≈−¥624）→¥2,296；C4 seeds 8→6（−57.1h，≈−¥448）→¥1,848。**FLOOR**：G1a 3 方法×4 seeds+G0 = ¥1,583。**单价 +50%**：C1 即闭合 **¥4,254**；**双因素 ×2.25**：C1 ¥6,031→C2 ¥5,696→**C3 ¥4,954 ✓**（C4 ¥3,947 为余量）。

## 6. v3.2 组件去留

| v3.2 组件 | 处置 |
|---|---|
| 4 技能 S1/S2/S7/S8、fixed-K K=1221 | **保留**（主协议） |
| fixed-B 协议 B / 6 技能 gate / G3–G5 | **删除**（省预算，换证书实验） |
| LGA（LP×σ̂ 乐观 bonus） | 改造为 C-LGA；σ̂ 统一为“估计量不确定性”，只进证书宽度（修三义） |
| H2（ECE −20% 阈值） | 降为支持证据；主表改 decision-weighted ECE 与覆盖率 |
| CRN 16 cell×1,000×8 seeds | **保留并升级**为 1,000/plan 的联合 rollout 集 |
| M1–M8（fixed-K、CRN、head、TOST、SGS 判据、预算再平衡…） | M1/M3/M4/M5/M8 保留；M2/M6/M7 删除或改写 |
| Prop 3 / Neyman | 分别降为 T2 推论 / T4 特例；新增 T1/T3/T5 |
| ReMix 引用 | 补进 related work（PMLR v270:145–164），并明确“配比≠决策” |
| 真机 / 6 技能扩展 | 不恢复 |

## 7. 排期（W1–W26 + ICLR W46；CoRL W31 可选）

W1–2 修 harness：**技能间同 episode 状态延续（不 reset）**+统一 API+入口态日志（W4 验证失败则降级并写明 reset 偏差）；W2–4 合成 oracle 与 T1/T4 数值/T5 相图（CPU）；W4–6 证书实现+PAC 覆盖率；**W6 gate-1**（生死步，1 周）：在既有 L1/L2 日志上算 `gap=|S_ind−S_joint|` 与 ρ；判据 `gap≥8pp 且 ρ≤3` → 走主线；`gap<3pp` → 转负结果版（lens_10 B，仍有 S≈3.9）；`ρ≥5` → 先改 Π 设计（加近 margin 计划）再判。W7–9 E1–E3；W10–16 G1a/G1b；W13–18 E4/E5+β 重估；W17–20 agent 2×2（scripted+LLM）；**W20 gate-2**：证书盾 undetected failure 降 ≥15% 且 C-LGA 同预算 certified success +≥3pp，否则砍盾/C-LGA 保测量；W21–24 分析/图/消融；W24 决策日志；W26 冻结；W27–30 写作；**W31 CoRL 2027 先投（可选）**；W32–45 改稿；W46 ICLR 2028。

## 8. venue

**主投 ICLR 2028**（形式化对象+反例+受控实验正合 ML 口径；lens_10 的一手证据：ICLR'25 的校准/决策理论论文均无机器人、无规模）。CoRL 2027 只作“先投再改”期权（机器人版，理论压附录）——**需 Lead 拍板是否接受双投风险**。NeurIPS 2027 main/D&B 与 ICML 2027 弃（时间与 ≥10 任务要求）。

## 9. 风险与负结果分支

| 风险 | 触发 | Plan B |
|---|---|---|
| R1 技能间不能状态延续 | W4 验证失败 | 显式建模 reset 偏差，改“分段组合”benchmark；证书仍成立，故事降级 |
| R2 组合缺口 <3pp | W6 gate-1 | 负结果版：校准确实可组合（正结果）；改 CoRL/NeurIPS D&B |
| R3 证书太宽/vacuous | W8 | 改为“可认证性测量”：给 bound–样本曲线，主张变“多数技能库不可认证” |
| R4 C-LGA 无增益 | W12 | 保 T4 的合成验证+真实负结果，分配贡献改为“证书口径与成功率口径不可混用” |
| R5 定理被判 trivial | 审稿 | 强调 T1 的反例（ECE=0 而 regret Θ(1)）与 T4 的方向反转，定理只作骨架 |
| R6 LLM 方差大 | W17 | scripted planner 为主臂，LLM 为实例 |
| R7 排期滑 | W16 | 依次砍 E4、E5、G1b；W26 前保 E0–E3+G1a |

## 10. 需 Lead 拍板

1. **双投策略**：CoRL 2027（W31）先投、ICLR 2028（W46）再投是否接受（一稿多投边界/robot 社区风险）？若否，直接只投 ICLR。
2. **技能集**：保留 S1/S2/S7/S8（S7 难学但提供低 p 区）还是换 S3 提升 E1 覆盖率实验的可测性？
3. **主格规模**：3 方法×8 seeds+2 LGA-success runs（¥3,660，余 ¥1,140）是否按此锁定；是否把结余用于补 second-N（+¥528）或 lens_09 的 Recovery Value 小节（+¥260）？
4. **是否把 lens_09 的 Recovery Value 并入 §5**（+¥260、3 GPU-h，但会增“第二个对象”风险）。
5. 题目定稿：*Calibration Does Not Compose: Partial Identification and Certified Allocation for Skill Libraries*（备选：*Certified Skill Composition*）。

## 附录：已核实引用（arXiv API，2026-11-02）

2107.05719（Calibrating Predictions to Decisions）；2510.23471（Robust Decision Making with Partially Calibrated Forecasts）；2504.15615（Dimension-Free Decision Calibration）；2208.02814（Conformal Risk Control）；2311.01457（Conformal Policy Learning for Sensorimotor Control）；2306.02551（Conformal Predictive Safety Filter）；1708.08611（Safe RL via Shielding）；2402.06529（Introspective Planning）；1609.05140（Option-Critic）；2106.13105（Option Keyboard）；2401.03360（Generative Skill Chaining）；2306.10193（Conformal Language Modeling）；1904.01685（Measuring Calibration in DL）；1405.3316（Non-stationary bandits）；2306.00196（Restless Bandits）；1812.06162（Large-Batch Training）；2506.03404（On-policy Parallelized Data Collection）；2307.14430（Skill-it!）；2305.14975（Just Ask for Calibration）；2302.09664（Semantic Uncertainty）；2107.07511（Gentle Intro Conformal）；2304.15004（Emergent Abilities Mirage）；2402.01817（LLM-Modulo）；2305.16291（Voyager）；2204.01691（SayCan）；2210.03629（ReAct）；2305.15334（Gorilla）；2307.16789（ToolLLM）；2002.06673（Performative Prediction）；1902.10730（Degenerate Feedback Loops）。
