# 22 · 红队 Round 1：尽最大努力杀死 `21_synthesis_v1`

**审计对象**：`21_synthesis_v1.md`（135 行）。**角色**：反对派审稿人，不是合著者。**结论先行**：骨架（部分识别 + 证书宽度 + 分配错配）可以活，但**按现状活不到 50%**；它的 51% 是 S 重估的算术产物，不是可辩护的点估计。

---

## 0. 判词与概率（Q5/Q7）

我的独立打分（rubric：N .25 / TQ .20 / Exp .20 / Insight .20 / C .10 / Rep .05）：

| 维度 | 21 的重估 | 我给 | 理由 |
|---|---|---|---|
| N | 3.90（+0.65） | **3.35**（+0.10） | T1 是 Fréchet–Hoeffding；T4 是 Lagrange 练习；T5 是 Le Cam 两点法；新对象只有"混合成本下的证书分配"若被形式化才成立 |
| TQ | 4.00（+0.15） | **3.40** | T2 无下界、缺 δ/Π；T5 常数错 2×；T4 与 C-LGA 指数不一致；ρ≈20 无算式 |
| Exp | 3.75（+0.10） | **3.25** | 4 技能、2 任务、26 runs；主臂 scripted；E1 仅 16k eps；外部基线被删 |
| Insight | 4.05（+0.60） | **3.85** | 相变 + 口径错配是真洞察；但"校准不组合"是 Fréchet 的已知推论 |
| Clarity | 3.85 | **3.55** | 连贯但数字过密，读者抓不住一条主线 |
| Repro | 4.35 | **4.25** | scripted 主臂、CRN、预注册都好；扣分在 LLM/API 不确定性与"阈值事后改"观感 |

加权：.25×3.35+.20×3.40+.20×3.25+.20×3.85+.10×3.55+.05×4.25=**3.51**。

| 口径 | 3.51（现状） | 3.61（最小补丁） | 3.75（三件套全做） | 21 声称 |
|---|---|---|---|---|
| CoRL（c=3.66，exec .82） | 38%（30–46） | 40%（32–48） | 44%（36–52） | **51%** |
| ICLR（c=3.74，exec .84） | 37%（29–45） | 39%（31–47） | 43%（35–51） | **50%** |
| 级联 | 48%（40–56） | 50%（42–58） | 55%（47–63） | 60% |

**Q5 直接回答：不能 >50%。** 差 8–13pp；binding 是 N 与 Exp。21 的 3.96 需要 N+0.65、Ins+0.60；我只接受 N+0.2、Ins+0.4。即使到 3.75，>50% 仍需 S≈3.95 且 exec≈0.86——在 4 技能/纯仿真/1 人约束下不可达。**建议目标改为"ICLR 主投、级联 50–55%"，并如实告诉用户。**

**Q7 一句话**：把 T1 降为**引理**，把 T2 做成**上下界匹配的认证样本复杂度**，并让 T4 只负责**评测期**证书分配、与训练期 C-LGA **彻底解耦**；其他不动。

---

## 1. 理论真伪/新颖性（Q1）

**审稿人原话**："Fréchet–Hoeffding is a 1950s result; partial identification is a standard econometrics tool; sharpness of these bounds is textbook. What is the new theorem?"（Fréchet 1951；Hoeffding 1942；Rüschendorf 1982；Manski 1990）——这一击**成立**。T1 的 L=max(0,Σq_i−(H−1))、U=min q_i 及 copula 可达性没有一条是新的；H=3,q=0.7→0.343 vs 0.10 只是"联合可远低于乘积"的教科书例子。

**比 novelty 更致命：21 没定义"边际"。** 组合中技能 i 的成功率依赖前缀诱导的入口分布 p_g(x)；孤立评测边际 ≠ 组合入口边际。T1 的 q_i 若指 plan-induced，则**不可能只从孤立 rollout 得到**（必须联合 rollout），"marginal-only 估计器"是空靶；若指孤立边际，则 β 是系统误差而非"修正项"。**必须把 q_i(π) 写成 plan-induced，β 升为一阶项。**

**能接受的最强版本（三件套，缺一不可）**：
1. **T1 降为 Lemma**，只建立"识别间隙存在"，不占贡献；
2. **认证样本复杂度上下界**：直接联合采样估 S(π) 需 n=Θ(log(1/δ)/w²)（Hoeffding + Le Cam/Assouad 下界）；边际复用证书需 n=Θ(H²(R+1)²σ²log(|Π|/δ)/m²)。21 的 T2 **漏 log(2|Π|/δ) 且只有上界**；两者的差就是"复用 vs 直采"的成本结构——这是可实验检验的新对象；
3. **证书宽度作为 design variable**：预算 B 下选 (边际估计数 n_g，计划联合采样数 m_π)，min max_π 宽度 s.t. 成本；证明"全边际/全联合"是两端、最优在内点。

**T4 复核通过但脱节**：min Σ√(v_i/n_i)⇒n_i∝v_i^{1/3} ✓；乘积界 Σ(S/q_i)(R+1)√(v_i/n_i)⇒n_i∝(v_i/q_i²)^{1/3} ✓；Neyman n_i∝√(w_i v_i) ✓；方向反转 ✓（p→0：证书权重∝p^{−1/3}，ECE 权重∝p^{1/3}）。**但 C-LGA 的 s_g=LP_g·v_g^{1/2}q_g^{−2/3} 与 T4 的 v^{1/3}q^{−2/3} 指数不一致，且 LP（训练期学习进度）与 v（评测期方差）是两回事；T4 证评测采样分配，C-LGA 做训练环境分配——主算法没有定理支撑。** 修法：T4 只用于 propose-certify；训练期分配保留 LGA 作启发式，正文写"无定理保证"。

**T5 常数错 2 倍**：要 P(误选)≤δ 需每计划误差≤Δ/2，union 后 **Δ≥2D√(log(2|Π|/δ))**，不是 D√(2log(...))。Le Cam：KL=ρ²、TV≤ρ/√2 ⇒ E[regret]≥(Δ/2)(1−ρ/√2)；c=1→c′=0.146，c=0.5→0.323；21 只写 c′ 无值，不可验。

### 1.5 四种叙事，只有一种能活

| 框架 | 审稿人一击 | 结论 |
|---|---|---|
| A "我们证明校准不足以组合" | "Fréchet 的直接推论，1950s 就有" | 死 |
| B "我们提出证书方法" | "set-valued/conformal 已有；只是 CP 用于技能库" | 半死 |
| C "我们给证书分配定理" | "Lagrange / 最优实验设计；n∝v^{1/3} 是标准结果" | 半死 |
| D "组合成功的认证是独立统计问题：最优样本分配、样本复杂度、'校准何时无关'的相变，与边际校准不同；给匹配上下界 + 相变 + 受控验证" | 需要真下界与混合成本模型 | **唯一可活** |

21 现在 A/B/C/D 混讲，读者判断不了贡献是哪条；**必须砍成一个**。

---

## 2. ρ 数字与自证伪推论（Q2 续）

用 21 自己的数字：H=3、q≈0.7、c=S/q≈0.49、v≈0.23，则 D=√(2Σc²v/n)：

| n/plan | D | Δ=1pp | 2.5pp | 5pp | 10pp |
|---|---|---|---|---|---|
| 200 | .041 | .24 | .61 | 1.22 | 2.44 |
| 1,000 | .018 | .55 | 1.37 | 2.75 | 5.50 |
| 4,000 | .009 | 1.10 | 2.75 | 5.50 | 11.0 |

**要 ρ≈20 需 D≈0.0012 ⇒ n≈2.3×10⁵/plan**，与"512k rollouts、16 cell×1,000、8 seeds"（每 cell 1,000）差两个数量级。**"ρ≈20"没有算式支撑**；据它撤 H2 叙事是错误推断：H2 测"分配→边际校准"，T5 说"校准→决策只在窄带有效"，两者可同时成立（ECE 改善但决策不变）。21 把两件事混成一件。

**正确表述**：ρ 依计划对而异；L1/L2 若用分离良好的计划对，ρ>3 ⇒ agent 必 null；要让 H3 有信号，**必须构造成对近 margin 计划**（同技能集、不同顺序/重试预算，使 Δ 与 D 同量级），这是 W6 的实打实工作，不是"加几个 plan"。

---

## 3. 工程红线（Q3）

同 episode 状态延续 = **新建复合环境**：共享场景、per-skill reward、per-skill 成功判定、入口态分布、状态机；MJX 里还要对齐 ctrl_dt/物理步，否则入口态漂移。拆解：复合 env 10–14 人日、统一 API/planner 接口 3–4、CRN 评测 harness 3、入口态日志与 β 估计 2 → **harness 16–21 人日（3.5–4.5 周）**，21 写 W1–2 偏乐观。

**W4 gate 判定不成立**："在既有 L1/L2 日志上算 gap"——L1/L2 是分开 reset 的技能评测，没有"技能间延续"的日志。改为：W1–3 建复合 env 跑 200 条 → 判据"复合 env 可跑 + 入口态 KL<0.1 nats"；否则触发 fallback。

**更便宜的 fallback（建议直接作主协议）**：**segmented composition**（技能间 reset，但入口态从上一技能终态采样/重放）+ 显式 β_s 估计 + 证书。这样"组合"仍成立（计划级成功），β 变成可测科学量（接口偏差），而非失败补丁；同 episode 延续降为 W2–W4 探针，不进主表。

**1 人排期**：技能 39 + arm 12 + harness 18 + 实验/分析 20 + 写作 5 ≈ **94 人日 ≈19 周**；45 周够，但 W26 冻结要求 harness 不滑过 W6；滑 2 周就砍 LLM 臂（保 scripted）。

---

## 4. 实验规模与被判 toy 的风险（Q4）

4 技能/2 任务/26 runs 撑不起"大规模 RL 技能库"：P2P 是 24,576 envs×7 天，我方 4×500M=2B steps、N=8192。审稿人原话："This is a controlled study of four skills, not a skill library."

可辩护写法：①主张改为"controlled study at library-relevant depth"；②加 **E0 合成 scaling**（10²–10⁴ 库、技能数 20/50/100，精确 copula，CPU）——直接测 T1/T4/T5 的 ground truth，**0 GPU**；③报总步数 2B 与 N=8192；④写进 limitation。另：21 删了 G3–G5（P2P-style/OmniReset-style/鲁棒性），审稿人必问外部基线，至少保留一个 scaled P2P-style（v3.2 估 ¥239–361）。

**钱按我重算≈¥850（不是 1,056）**，优先级：P0 **G6 CRN 32h/¥224（必须，否则 H2 无效）**；P1 **LLM ¥641**；P2 **PLR-style 基线 +8 runs/¥532**（比 LGA-success 2 runs 值，21 把 PLR 删了）；P3 合成 scaling（0 GPU）；P4 第二 N 已有 G1b；P5 第 5/6 技能（¥400+、+2 周）不做；P6 Recovery Value ¥260 可选。

---

## 5. 预算/算术（Q6，独立复算）

逐行：E1 5+E2 10+E3 4+E4 2+E5 10+G0 24=**55h**（Lead 的 65.6h 找不到对应行，疑重复计 10.6h）；G1a 26×9.51=247.3h；G1b 24×3.14=75.4h；合计 377.7h；×1.12=422.9；×7=¥2,960；+220+80+400=**¥3,660**（表内自洽）。

**四处不闭合**：
1. **G6 CRN 漏行**：§6 说保留并升级 CRN，预算无 32h/¥224。补后 409.6h→458.8h→¥3,211→+750=**¥3,961**。
2. **LLM 混口径**：表内 E4 ¥300+E5 ¥230=530，预算取 400，§6 又写硬顶 500，v3.2 重算 641 → 少计 **¥241**。
3. **FLOOR 错**：¥1,583 隐含 LLM 200。修正=12×9.51=114.1h+G0 24+G6 32=170.1h→190.5h→¥1,334+750=**¥2,084**。
4. **削减链混基数**：C3 用 530 砍、基数用 400；"双因素 C3 即闭合 ¥4,954"在修正基数下不成立：7,976→C1 6,646→C2 6,311→C3 5,570→**C4 4,221**，必须 C4（seeds 8→6，MDE 8.4pp，H1 阈值同步改）。单价+50% 修正后 C1=**¥4,872**（21 写 4,254，乐观 618）。

**结论**：基准真值 ≈**¥4,153（对 5,000 余 ¥847）**，不是 3,660/3,744。注：21 的"余 ¥1,140"是与自设 **≤4,800** 比（4,800−3,660），Lead 的 1,056 是 4,800−3,744（3,744=3,660+84，疑把 E4/E5 的 12 GPU-h 重复计一次）。

### 5b. W6 gate 情景树（21 的 3.96 没乘 gate 概率）

| W6 结果 | P | 路径 | S |
|---|---|---|---|
| gap≥8pp 且 ρ≤3 | .45 | 主线 | 3.75 |
| gap 3–8pp | .25 | 效应减弱，负结果框架 | 3.55 |
| gap<3pp | .15 | 负结果版（校准确实可组合） | 3.45 |
| ρ≥5 | .15 | 改计划对，+2 周 | 3.60 |

期望 S=.45×3.75+.25×3.55+.15×3.45+.15×3.60=**3.63** → CoRL ≈40%，与我的点估计一致；21 的 3.96 是"全过 gate 且 N/Ins 全兑现"的上尾，不是期望。

---

## 5c. 审稿人 attack 面（按命中率排序）

1. "Four skills is not a library; the two tasks are yours." → E0 合成 scaling（20/50/100 技能）+ limitation + 报 2B steps。
2. "No external baseline (P2P/OmniReset)." → 恢复 scaled P2P-style（¥239–361）。
3. "PLR is the obvious allocation baseline; why removed?" → 恢复 PLR-style（删 LGA-success 2 runs）。
4. "Your own theorem says the agent experiment is null unless ρ∈[1,3]; are your plans in that band?" → 报 ρ 分布 + 构造近 margin 计划 + H3 改相变检验。
5. "Joint certificates cost H× per query; what is the real overhead?" → 报 O(|Π|mH) 与 ≤5% 预算 + cost–benefit/abstain 规则。
6. "β is a fudge factor for a failed assumption." → E3 估 |β|、检验 |β| 随 H 单调、把 β 升为贡献（接口偏差可测）。
7. "2pp ECE improvement is negligible." → decision-weighted + ρ 分层 + 明确不主张决策收益。
8. "Threshold was 30% then 20%: goalpost moving." → 论文写预注册修正记录与时间戳，两阈值都报。
9. "LLM results are prompt/version dependent." → scripted 主臂 + API 快照 + 温度 0 + 缓存 + 版本串。
10. "No real robot; is this robotics?" → 投 ICLR；CoRL 版写明 sim-only 是设计选择，引 Playground/OmniReset 纯仿真先例。

---

## 6. 模拟三审 + 最小补丁清单

**R1（理论）**：Borderline→Reject。理由：T1 教科书、T2 无下界、T4 与算法脱节、T5 常数缺失。**R2（实验）**：Borderline。理由：4 技能像 toy、外部基线缺失、agent 主臂 scripted、E1 功效未报。**R3（意义）**：Weak Reject。理由：若校准不组合，实践者该做什么？证书成本高，缺 cost–benefit 规则。**AC**：Borderline（poster），除非三件套落地 + 合成 scaling + agent LLM 臂有信号。

**CRITICAL**：C1 理论三件套（T2 匹配下界 + T5 常数 + T4/C-LGA 解耦）；C2 对象（q_i plan-induced；β 一阶化；主协议 segmented）；C3 叙事（H2 与 H3 逻辑分开；删 ρ≈20；报 ρ 分布）；C4 预算（补 G6、LLM 641、FLOOR/双因素重算，C4 触发）。
**MAJOR**：M1 恢复 PLR-style（删 LGA-success 2 runs 换它）；M2 E0 合成 20–100 技能 scaling；M3 摘要不把 4 技能写成 "library"；M4 会期改**顺序投**——CoRL 六月底投、8–9 月决定，未中再投 ICLR（九月截稿）；**21 的"双投风险"是伪问题**，真风险是 CoRL 通知晚于 ICLR 截稿；M5 概率改"ICLR 主投、CoRL 期权、级联 50–55%"。**MINOR**：m1 经典引文按期刊写（Fréchet 1951 / Hoeffding 1942 / Rüschendorf 1982 / Manski 1990，**勿编 arXiv ID**）；m2 E1 报 16k eps 功效；m3 C-LGA 改名"启发式证书权重"。

### 6b. 三件套落地的时间账（W6 前）

| 动作 | 人日 | 窗口 |
|---|---|---|
| T2 匹配下界 + δ/Π 常数 | 3–5 | W2–W3 |
| 混合成本证书分配形式化（边际复用 vs 联合直采） | 5–7 | W3–W5 |
| E0 合成 scaling（20/50/100 技能） | 3 | W2–W3 |
| segmented composition 主协议 | 3 | W3–W4 |
| 报告/图/预注册修订 | 2 | W5–W6 |

合计 16–20 人日；砍掉 21 原计划里"T5 相图 + 冗余 plan 设计"的约 10 人日 → 净增 6–10 人日，**W6 gate 仍可保住**。不做则 TQ 停在 3.40、S≤3.60（CoRL ≈39%）。

---

## 7. 文献核验（arXiv API，2026-11-02 实查）

2107.05719 Zhao et al. *Calibrating Predictions to Decisions*；2510.23471 Kiyani & Hassani *Robust Decision Making with Partially Calibrated Forecasts*；2504.15615 Tang et al. *Dimension-Free Decision Calibration*；2208.02814 Angelopoulos & Bates *Conformal Risk Control*；2311.01457 Huang et al. *Conformal Policy Learning for Sensorimotor Control*；2306.02551 Strawn & Ayanian *Conformal Predictive Safety Filter*；1708.08611 Alshiekh et al. *Safe RL via Shielding*。经典四条无 arXiv，按期刊引。
