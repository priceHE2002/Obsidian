# 99B · 灵巧手技能库 + LLM Agent 组合 —— 最终实验方案（决策层）

**项目**：DexSkillBench / LGA（Library-level Learning-Progress Allocation）
**版本**：v1.0（2026-11-02）· 对应正文 v3.2（`15_proposal_dexterous_v3.md`，940 行）· 终审 `16_acceptance_dexterous_final.md`
**主投**：CoRL 2027（预计 2027-06 截稿，W31）**兜底**：ICLR 2028（预计 2027-09 下旬，W46–47）
**诚实结论**：方案质量已到位，但**单会中稿期望 ≈38%，级联 ≈45%（质量口径 ≈47% / ≈60%），达不到 >50% 的期望口径**；>50% 在"1 人、45 周、纯仿真、¥5,000"约束下算术不可达。详见 §13。

---

## 0. 执行摘要（一页）

| 项 | 结论 |
|---|---|
| 课题 | 大规模 RL 训灵巧技能 → 每技能封装为 **带校准成功率的 JSON 函数** → LLM agent 组合成长程任务 |
| 用户约束 | 纯仿真（不用 SO-101 真机）；贡献并重（分配方法 + 技能库 + agent 评测）；1 人 45 周；租卡 ≤¥5,000；本地 4070 Ti Super 16GB/32GB |
| 核心科学问题 | 固定交互/算力预算下，**技能库级数据分配**是否通过 **per-skill 校准** 影响 **LLM 组合成功率**？ |
| 方法 | **LGA**：技能级 LP 导数 × 估计量不确定性（σ̂）加权 → 技能内 config 桶分配 → Hedge + 3 guards → learned calibration head；输出校准区间 |
| 理论 | Prop 1（校准误差沿 H 步、R 次重试放大：`≤2H(R+1)ε`=24ε）；Prop 2（Neyman 分配形式与 26–40% **全-M 异质上界**，**不是** H2 阈值）；Composition corollary（n≈1.1e5/skill，不 binding） |
| 基准 | **DexSkillBench**：4 核心技能（S1 Reorient-Any / S2 Rotate-Z / S7 Peg-Insert 3mm / S8 Cap-Twist）+ 2 长程任务；6 技能为 W12 gate 加分；固定-seed **CRN**（16 cell×1,000 held-out）；开源 harness |
| 主协议 | **fixed-K** K=1221（N=8192 → B=500.1M；N=2048 → B=125M）；fixed-B 112.5M 为次要；8 seeds；4 方法主格（uniform/SER/SGS/LGA）+ PLR-style |
| 统计 | H1a 平均 ≥6pp（MDE 5.9，power 0.81）、H1b min-skill 非劣 δ=5pp、H1c utility-weighted ≥5pp；H2 配对 CRN decision-weighted ECE 相对 ↓**≥20%**；H3 剂量-反应（0/2/5/10pp 噪声）斜率 |
| 预算 | **基准 ¥4,809（余 ¥191）**；单价 +50% 经 C1–C3 闭合 ¥4,995；双因素经 C1–C5 → FLOOR **¥4,872**（强制 4-cell CRN）；硬上限 ¥5,000 |
| 排期 | W1（2026-11-02）→ W5 G0 → W8 G1 → W12 G2 → W24 决策 → W26 冻结 → W31 投 CoRL；W32–W45 ICLR 兜底 |
| 概率（judge 终审） | v3.1：CoRL 35%（质量 47%）/ ICLR 31%（42%）/ 级联 42%（59%）；v3.2 补 G1–G3 后 ≈**38% / 33% / 45%**（质量 ≈47%/42%/60%）；**单会 <50%** |
| 结论 | **投 CoRL 2027 + 三个补丁已落地；按 35–38% 做计划，不要按 50%** |

---

## 1. 目标与硬约束

### 1.1 用户目标（原文要点）
1. 做灵巧手**大量精细操作**的大规模强化学习；
2. 把每个精细操作**封装成 LLM agent 可调用的函数**；
3. 实现 **agent 驱动的灵巧手精细操作机器人系统**（参考 Play2Perfect）；
4. 目标会议不限于 ICRA——用户明确放开："不投ICRA，投其他的也可以哈，ICML/ICLR/NeurIPS/CoRL"（m00864）。

### 1.2 硬约束（不可突破）
- **纯仿真**：不用 SO-101 真机（用户明确："SO101 真机就不用了…做纯仿真就行"）。
- **1 人全职 45 周**（2026-11-02 → 约 2027-09）。
- **租卡 ≤¥5,000**（含缓冲，硬上限）；本地 4070 Ti Super 16GB / 32GB RAM / 2TB SSD 免费算力。
- 贡献并重：分配方法（FG/LGA）、可调用技能库、agent 组合评测——不能只做其中一条。

### 1.3 目标演变（记录）
- 原方向（FrontierGuard-RL）→ 用户 2026-11-02 改为"灵巧手大规模 RL + 技能库 + LLM agent"；旧课题 00–99 仅作方法与流程参考，不再作为依据。
- 投稿：ICRA 2028（sim-only 折价）→ 用户放开后改为 **CoRL 2027 主投 / ICLR 2028 兜底**。

---

## 2. 关键决策与理由

| # | 决策 | 理由 |
|---|---|---|
| D1 | 纯仿真、不做真机 | 用户明确要求；省去真机 5–8 人周与 ¥0 租卡，但 sim-only 需在 framing/limitation 里补（judge 已计入折价） |
| D2 | 从"系统论文"改为"**受控因果链**论文" | Play2Perfect + ADEPT 已占住 pretrain→post-train；正面比规模必输。改为测"分配→校准→组合"的因果链 |
| D3 | **CoRL 2027 主投** | 机器人问题 + 受控实验 + 开源基准最匹配；8 页够放方法/理论/实验；ICLR 更看重 ML 抽象，作兜底 |
| D4 | 4 核心技能 +（W12 gate 后）6 技能 | 1 人 16 周工程只够 4 技能（4 技能≈60–72 人日）；6 技能是加分目标；砍 stretch/G3/G4/G5 换 2 周缓冲 |
| D5 | fixed-K 主协议 | 避免"用吞吐赢论文"；fixed-B 下 N=8192 的 K 只有 305，比较的是 cold-start 不是分配。fixed-B 作次要协议 |
| D6 | 校准只有 **CRN** 才可测 | 250/cell 二项 SE≈3.2pp，30% 相对阈值不可测；16 cell×1,000×8 seeds 配对后 MDE≈1.5–2.0pp |
| D7 | H2 阈值 −30% → **−20% 相对** | judge 复算：frontier v∈0.21–0.25 时 Neyman 比 1.000；仅样本集中通道上限 29.3%<30%，30% 不可达 |
| D8 | 预算按固定公式逐行复算 + 削减链 | 两轮独立审计（Lead + judge）抓出 G6/LLM 少算；基准 ¥4,809、C3 ¥4,995、FLOOR ¥4,872 全 ≤5,000 |
| D9 | CoRL 被拒后**级联 ICLR**，不双投 | 双投违规；同一主表两套 framing；W24 六档决策（D10）避免"一刀切" |

---

## 3. 定位与近邻（与 v3.2 §9 一致）

| 工作 | 它做了什么 | 我们与它的关系 |
|---|---|---|
| **Play2Perfect**（2606.26428） | 22-DoF Sharpa + KUKA，Isaac Sim，24,576 envs×7 天 play 预训练，SAPG，真机 0.5mm 插入 60% | 规模差两个数量级；我们不比吞吐，只比 fixed-K 下的**分配效率**与**校准测量**；真机数字作外部锚 |
| **ADEPT**（2608.19182，CoRL 2026） | 大规模 RL 灵巧预训练 + post-train，视触觉 sim2real | 同为"先训后调"，但它是更强单技能 + 真机路线；我们是技能库级分配 + 校准 |
| **OmniReset**（2603.15789） | 多样 reset、单奖励、随算力扩展的多任务 | reset 多样性与我们正交；它是任务级，我们是技能库/config 桶级 |
| **SGS**（2610.12465，CoRL 2026） | success-band 三档加权 reset，2^20 envs | **最直接威胁**；我们是"LP 导数×σ̂、技能库级"，判据/输出不同；不称其为特例，只称三档单调近似 |
| **SER**（2511.21011） | staggered resets 增加并行数据多样性 | 调 reset 时序，不分配经验；作基线 |
| **Being-0**（2503.12533） | FM + 模块化技能库 + VLM connector | 系统路线；我们是受控测量，不拼工程规模 |

**研究缺口（我们的卖点）**：现有技能库工作假设"每技能成功率已知"，但**校准误差如何随分配变化、又如何传播到长程组合**没人系统测过；大规模 RL 只在单任务内做采样，技能库级 allocation 与 calibration 的耦合是空白。

---

## 4. 研究问题与假设

### 4.1 RQ
- **RQ1（分配）**：固定预算下，技能库级 allocation（LGA）是否比 uniform/SER/SGS/PLR 式在库级与技能级成功率上更好？
- **RQ2（校准）**：分配是否改善 per-skill 校准（decision-weighted frontier ECE）？机制是"样本集中"还是"非平稳/EMA 滞后"？
- **RQ3（组合）**：校准误差是否因果地影响 LLM agent 的长程组合成功率（剂量-反应 + 重试）？

### 4.2 假设（主检验族 = {H1a,H1b,H1c,H2,H3}）
- **H1a**：LGA 的平均技能成功率 ≥ uniform +6pp（8 seeds，MDE 5.9pp，power 0.81）。
- **H1b（非劣）**：LGA 的 min-skill 成功率不劣于 uniform 超过 δ=5pp（防"饿死难技能"刷高平均）。
- **H1c**：utility-weighted 成功率 ≥ uniform +5pp。
- **H2**：LGA 的 held-out **decision-weighted frontier ECE** 相对 uniform 降低 **≥20%**（配对 CRN，95% CI 排除 0）。
- **H3**：向 p̂ 注噪 0/2/5/10pp，组合成功率**单调下降**（交互项斜率显著，power 0.82）。
- **H4–H9（探索性）**：技能数 4 vs 6、API 粒度、planner 2×2、head vs EMA、跨任务泛化、失败模式。
- **可检验预测 TP1–TP6**（斜率/margin 翻转/分配替代/技能数/剂量/head），失败含义预先写死。

### 4.3 负结果分支
- **N1**：H1 不成立、H2 成立 → "分配主要影响校准而非库级成功率"，投 CoRL 校准版。
- **N2**：H1/H2 都成立但 H3 不成立 → "校准不等于被 agent 用上"，收为测量论文。
- **N3**：H2 只达 10–20% → 降级为方向性证据，不写 30% 叙事。
- **N4**：H1 不成立、H2 也不成立 → ICLR 理论版（W32–W45 扩 6 技能 + 负结果）。

---

## 5. 方法：LGA（两级分配 + 校准输出）

### 5.1 总体
每个技能 g（LEAP 16-DoF 或 Franka+LEAP 臂式）有 goal-conditioned 共享策略；技能库总预算 B 在技能间、技能内 config 桶间两级分配。每步：
1. **技能级**：`u_g = (p_fast − p_slow)(1 − p_slow) + λ σ̂_g`（LP 导数 × 估计量不确定性）→ Hedge 更新权重；
2. **技能内**：对 config 桶做 z-score 归一化 LP，按权重分配；
3. **预算归一**：按实测每步算力 c_g 归一化（`Σ c_g b_g = C`）；
4. **输出**：每技能的成功率点估计 + 校准区间（learned calibration head + β 后验/isotonic）。

### 5.2 关键设计
- **σ̂（估计量不确定性）**：3 条滞后 EMA（5/20/100 次更新）的离散度；**不是** model epistemic uncertainty。方向与 bandit 相反：**给"不确定到无法判断"的桶降温**（数据源不是决策对象）。
- **Guards**：G-1 n<30 或 σ̂>q90 冻结权重；G-2 权重 TV>0.2 冻结 50 步；G-3 熵 <0.5logM 或单技能 >40% 时温度 ×1.2。
- **校准 head**：在训练日志上拟合（0 GPU）；EMA age–误差回归作机制诊断；两者都不做 → H2 降级。
- **技能 API（JSON）**：`{"skill": "s7_peg_insert", "params": {...}, "p_success": 0.62, "ci": [0.55,0.69], "calibration": "frontier-v3", "tolerance_mm": 3}`；agent 用 p̂/CI 做 plan selection 与重试。

### 5.3 Agent 层
- **P0 random**（下界）、**P1 ReAct+反思**、**P2 plan-then-execute**、**P3 无 p̂**（消融，强制不看校准）、**O1 oracle**（真成功率）。
- 600 base episodes + 240 剂量 episodes = 840；固定模型快照/prompt/温度；3 次重复；LLM 成本 ¥641（硬顶 ¥700）。

---

## 6. 理论（正文 0.75–1 页 + 附录 2 页）

- **Prop 1（校准误差传播）**：每技能校准误差 ε，H 步、每技能 R 次重试 → 计划选择损失 `≤ 2H(R+1)ε`。H=4、R=2、ε=1pp → 24pp。**只作方向/斜率预测，不是幅度预测**（v3.1 漏 (R+1)，v3.2 已修）。
- **Corollary（rollout 下界）**：选对计划需每技能 `n ≳ 2H²(R+1)² log(2S/δ)/m²`；H=4,R=2,S=6,δ=.05,m=.12 → ≈1.1e5/skill；B=500M 下不 binding，只作定性结论。
- **Prop 2（Neyman 分配）**：L2 目标 n*∝√(w_b v_b)；L1/ECE 目标 n*∝(w_b√v_b)^{2/3}；uniform/最优比可差 O(M)；**26–40% 是全-M 双异质例的上界，不是 H2 阈值**；H2 阈值改用 −20%。
- **Prop 3（组合损失）**：`≤2Hε(1+ε)^{H−1}`（no-retry 版）+ margin 条件。
- **引理 4a/4b**：LGA 分数与 Neyman 解方向一致；局部灵敏度 `ECE(α)/ECE* ≤ 1+(3/8)max δ_b²`，用于 W6 预测阈值。
- **明确不做**：PPO 收敛证明、conformal 包装、多形态迁移进主线。

---

## 7. 基准：DexSkillBench

### 7.1 环境与技能
- **仿真栈**：MuJoCo Playground（MJX, warp）为主；本地 4070 Ti Super 可跑 N=1024；租 A100 跑 N=8192。
- **手**：LEAP 16-DoF（开源）；Aero 附录。
- **臂**：Franka（单臂 + LEAP），手内技能 mask 臂；**控制统一 20Hz**（手内 20Hz、臂式 20Hz；400Hz 物理求解器仅作 S7 探针，不进主表）。
- **4 核心技能**：
  - S1 **Reorient-Any**：任意 6D 姿态重定向（目标姿态随机）；
  - S2 **Rotate-Z**：绕 z 轴连续旋转（角度随机）；
  - S7 **Peg-Insert**：1–3mm 间隙（主表 **3mm**，W4 探针 1/3/5mm；5mm 回退；后置横向 2mm）；
  - S8 **Cap-Twist**：旋盖 2 圈（**无螺纹**，避免接触求解器不稳）。
- **6 技能 gate（W12）**：加 S3 In-hand-Translate、S4 Finger-Gait；S5/S6 stretch 不进 CoRL。
- **2 长程任务**：L1 = S1→S7→S8；L2 = S2→S7→S8（3 步、2 次重试）；L3/L4 移 ICLR。

### 7.2 评测集与协议
- **CRN 校准集**：每个 (skill×bucket) cell 固定 **1,000 个 held-out 初始状态**（固定 seed，W6 冻结；4 技能=16 cell；6 技能=24 cell），所有方法/种子共用。
- **主协议 fixed-K**：K=1221，L=50；N=8192 → B=500.1M；N=2048 → B=125M；8 seeds；主格 4 方法（uniform/SER/SGS/LGA）+ PLR-style（G2 消融 9 变体×2 seeds）。
- **次要 fixed-B**：B=112.5M（N≤2048 对照），标注"不是主结论"。
- **主指标**：库级平均成功率、min-skill、utility-weighted、decision-weighted frontier ECE、组合成功率、规划效率。
- **统计**：8 seeds、混合效应模型、Holm（主族 5 端点）、配对 CRN、TOST（非劣）、bootstrap CI；预注册 W30 写死。

### 7.3 Run 矩阵（最终）
| 项 | 数量 | GPU-h |
|---|---|---|
| 租卡 G1a N=8192（4 方法×8 seeds） | 32 | 304.3 |
| 租卡 G1b N=2048 | 32 | 100.5 |
| 租卡 G2 消融（9 变体×2 seeds） | 18 | 56.5 |
| 租卡 G6 CRN（4 方法×16 cell×1,000×8 seeds=512k） | 1 批 | 32.0 |
| **租卡合计** | **83 runs** | **493.3** |
| 本地（Tier-0/Tier-1/pilot/校准 smoke） | 31 runs | 132.0（免费） |
| Agent（base+剂量） | 840 episodes | —（LLM ¥641） |

---

## 8. 预算（¥5,000 硬上限，可逐行复算）

- **基准**：`493.3×1.12 = 552.5 GPU-h × ¥7/h = ¥3,868` + 电 ¥220 + 存储 ¥80 + LLM ¥641 = **¥4,809（余 ¥191）**。
- **单价 +50%（¥10.5/h）**：¥6,742 → **C1 砍 G2 → ¥6,078 → C2 G1a 8→6 seeds → ¥5,183 → C3 CRN 32→16h → ¥4,995 ✓**；C4（安全余量）G1b 8→6 → ¥4,699。
- **双因素（×2.25）**：¥9,643 → C1–C5 → **FLOOR ¥4,872**（3 方法×5 seeds×2 N + G2 8 runs + **4-cell 小 CRN 8h**；保 32h CRN 会 ¥5,296 破线，禁止）。
- **硬上限规则**：¥5,000 不得突破；W4/W10/W16 预算审计；余 <¥100 冻结非主实验；LLM 预警 ¥600/硬顶 ¥700；Plan C = 2×4090 或开源权重模型。

---

## 9. 排期（45 周，CoRL 口径）

| 周 | 里程碑 |
|---|---|
| W1–W4 | 环境/资产/arm+LEAP 集成；W4 只做 **Aloha peg 代理 sanity**（≥20%）+ arm 可跑；**G0 顺延 W5** |
| W5 | **G0**：4 技能 MVP + S7 三容差探针（1/3/5mm）+ 本机/租卡吞吐实测 |
| W6 | 预注册冻结（H1–H3、阈值、D10 六档）；CRN 状态集生成 |
| W7–W9 | 本地 scaling 试跑；head 拟合；agent dry-run；**W8 G1**：4 技能可训、A100 有效吞吐 ≥10k steps/s |
| W10–W16 | 主格 G1a/G1b；**W12 G2**：LGA−uniform 预读 ≥3pp 否则转 N1/N2；超前 ≥2 周才加 S3/S4 |
| W17–W22 | 6 技能（若 gate 通过）/消融/长程 agent 2×2 + 剂量 |
| W20 | 中期审计：H1a 趋势、预算、排期 |
| W24 | **D10 六档决策**（a–f）：定 CoRL 叙事 |
| W26 | **实验冻结**（主表/图/统计全定） |
| W27–W31 | 写作、内审、视频/artifact；**W31 投 CoRL 2027**（≈2027-06-07，日期待核实） |
| W32–W45 | 若被拒 → 按审稿意见改投 **ICLR 2028**（≈2027-09 下旬，待核实）；扩 6 技能/理论版/负结果 |

**Go/No-Go**：W5 G0（技能可训+吞吐）→ W8 G1（4 技能本地可训）→ W12 G2（分配信号）→ W20（H1 趋势）→ W24（叙事决策）→ W26（冻结）。

---

## 10. 风险与负结果（top）

| 风险 | 概率 | 影响 | 触发动作 |
|---|---|---|---|
| H2 阈值（20%）不达标 | 0.45–0.55 | 高 | D10(c) 改校准/benchmark 版；W6 用引理 4b 预测阈值 |
| H1a <6pp | 0.40 | 高 | D10(b)/(c)，H1 降 secondary |
| W5/W8 gate 滑期 | 0.45 | 高 | Plan C：纯手内 4 技能 / Aloha 代理；期望 −5pp |
| CRN/LLM 预算低估 | 0.40 | 中 | 已修 G2 + C4；W16 复核 |
| A100 涨价/排队 | 0.30 | 中 | 2×4090 Plan C（¥2.5/h，N=2048） |
| 单人中断 | 0.25 | 高 | W16/W24 各留 1 周缓冲 |
| Sim-only 折价 | 1.00 | 中 | framing/limitation/外部锚（已计入） |
| CoRL 日期未公布 | 0.20 | 高 | W1/W20 核实；ICLR 兜底 |

---

## 11. 投稿策略与概率（judge 终审口径）

| 口径 | CoRL 2027 | ICLR 2028 | 级联（任一） |
|---|---|---|---|
| v3.1 期望（judge 终审） | 35%（80% CI 26–44） | 31% | 42% |
| v3.1 质量（完美执行） | 47%（43–50） | 42% | 59% |
| **v3.2 补 G1–G3 后估计** | **≈38%（28–46）** | **≈33%** | **≈45%** |
| v3.2 质量口径估计 | ≈47–49% | ≈42–44% | ≈60–61% |

**为什么达不到 >50%（诚实）**：单会质量上限 ≈47%（v3.1）→ 乘 exec 0.75 = 35%；补丁后 ≈38%。要到 50% 需质量 ≥59% 且 exec ≥0.85，即"6 技能跑完 + 真机代理证据 + 当期无同向强竞争"，在 1 人/45 周/纯仿真/¥5,000 下不现实。**不承诺 >50%；按 35–38% 做计划，级联按 42–45% 做期望管理。**

**改会/降级路径**：CoRL 2027 主投 → 被拒后 ICLR 2028（W32–W45）；NeurIPS 2027 D&B 仅在 W20 显示 benchmark 成熟时作机会项，不分兵；**不投 ICRA/RA-L**（sim-only 折价）；workshop 不降级（除非 W24 全失败）。

---

## 12. 审计闭环（v1 → v3.2）

| 阶段 | 文件 | 主要问题 | 修复 |
|---|---|---|---|
| v1 | `11_proposal_dexterous_v1.md`（844 行） | B/K 五处不一致；预算 ¥4,849 的 P0/FLOOR 矛盾；S7 1–2mm 被 Aloha 8mm 事实证伪；缺传播界；LLM 成本低估 | critic 批改 → v2 |
| critic | `12_critique_dexterous_v1.md`（330 行，C1–C9） | 协议/预算/功效/S7/S8/排期/理论/LLM 九项 | v2 修 |
| v2 | `13_proposal_dexterous_v2.md`（714 行） | 预算情景表与附录 C 冲突；校准预测器循环；H1 可被饿死难技能刷高；pilot 时点矛盾 | Lead 审计 → critic 修 |
| judge v2 | `14_acceptance_dexterous_v2.md`（297 行） | CoRL 33%（质量 43%）；M1–M8 必改（(R+1)、Neyman、CRN、head、阈值、剂量、4 技能排期、S7 探针） | writer v3 |
| v3.1 | `15_proposal_dexterous_v3.md`（930 行） | judge 终审 G1/G2/G3：H2 −30% 不可达、CRN/LLM 少算、W4 过载；§11 CI 行错位、§7 GPU-h 列和不闭合、D10 缺两档 | v3.2 补丁 |
| **v3.2** | **`15_proposal_dexterous_v3.md`（940 行）** | 以上全部落地：H2→20%；CRN 512k/32h/¥224；LLM ¥641；基准 **¥4,809**；C1–C4/FLOOR ¥4,872；G0→W5；D10 六档 | **终稿** |
| 终审 | `16_acceptance_dexterous_final.md`（67 行） | 判 "35–38% 的 CoRL 论文"；三补丁后 ≈38%；单会 >50% 算术不可达 | 采纳，不再追 50% |

**Lead 独立复核**：arXiv API 核实全部引用（SAPG 2407.20230、PLR 2010.03934、RED 2110.02439、SayCan 2204.01691、ReAct 2210.03629、CaP 2209.07753、RoboCodeX 2402.16117、GRAPE 2411.19309、SGS 2610.12465、P2P 2606.26428、OmniReset 2603.15789、ADEPT 2608.19182、Being-0 2503.12533、SER 2511.21011、Playground 2502.08844、LeapCube 相关），预算脚本 `budget_v3.py` 实跑复算，v3.2 自评 38/33/45 与 judge 35/31/42 偏差 ≤3pp。

---

## 13. 若要逼近 >50%，还差什么（记录，不承诺）

1. **质量**：CoRL 质量上限 47%（v3.1）→ 需 +12pp 质量分，即一次"真机制创新"：6 技能全跑 + H2 机制（head/age）成立 + H3 强剂量效应；单人 45 周无空间。
2. **执行**：exec 0.75 → 0.85：需 6 技能 60→90 人日工程 + 额外 2–3 个月；与 W31 截稿冲突。
3. **外部**：CoRL 2027 当期无同向强竞争（P2P/ADEPT follow-up 不占位）。
4. **替代**：接受级联期望 ≈45% + ICLR 兜底；或加真机证据（用户明确不做）；或把目标改为"质量口径 >50%"（CoRL 质量 47%、级联质量 60%——但仍非 >50% 的 CoRL 单会）。
5. **务实建议**：按 CoRL 35–38% 计划 + ICLR 兜底；把"6 技能 + 真机"写进 future work，而不是硬塞进 45 周。

---

## 14. 交付物与文件索引

| 文件 | 内容 |
|---|---|
| `15_proposal_dexterous_v3.md` | **终稿正文**（940 行，v3.2；含 §0–§15 + 附录 A–F、预算脚本 `budget_v3.py`） |
| `16_acceptance_dexterous_final.md` | judge 终审（35/31/42；G1–G3；D10 六档建议） |
| `14_acceptance_dexterous_v2.md` | v2 审判（M1–M8） |
| `12_critique_dexterous_v1.md` | v1 批改（C1–C9） |
| `13_proposal_dexterous_v2.md` | v2 正文（714 行） |
| `11_proposal_dexterous_v1.md` | v1 初稿（844 行） |
| `17_theory_bound_dexterous.md` | 理论备忘录（266 行：Prop 1/2/3、Neyman、Hedge、TP 表） |
| `10_BRIEF_DEXTEROUS.md` / `10b_VENUE_STRATEGY.md` | 任务书 / 投稿策略 |
| `budget_v3.py` | 预算脚本（BASE ¥4,809；C1–C4；FLOOR ¥4,872） |

---

## 15. 立即行动（W1–W6）

1. **W1**：建 repo（`dexs-bench/` `dexs-train/` `dexs-calib/` `dexs-agent/` `repro/`）；拉 MuJoCo Playground/LEAP/Franka 模型；确认本地 4070 显存上限（N=1024 目标）。
2. **W2–W4**：arm+LEAP 集成；S1/S2 手内技能 MVP；S7 Aloha 代理 sanity；**W4 末**：锁定租卡价格（A100 ≤¥7/h，否则 C1）。
3. **W5**：**G0**（4 技能可训 + S7 三容差探针 + 吞吐 ≥10k steps/s）；W6 预注册冻结。
4. **W6–W8**：本地 scaling 试跑；head 拟合；agent dry-run 100 episodes；**W8 G1**。
5. **随时**：预算日志 `budget_log.md`；W4/W10/W16 审计；CoRL 2027 日期 W1/W20 两次核实。

---

*本文件为决策层文档；实验细节以 `15_proposal_dexterous_v3.md`（v3.2）为准；概率以 `16_acceptance_dexterous_final.md` 为审计锚。*
