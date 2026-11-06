# Round 4 场地策略：日历、接受率、三 framing 与单会概率

**结论先行**：四会没有任一场能达到单会 >50%。最接近 CoRL 2027，点估 42%（80% CI 36–48）。推荐 **CoRL 2027 主投 + ICLR 2028 备投**；NeurIPS 2027（main 与 E&D）与 ICML 2027/2028 因日历出局。任何 ">50%" 只能在级联（55–58%）或 TMLR（50%，45–55）口径下说，不能写成单会。

## 1 日历（2027/2028 未发布，按 +1 年推断，标"估"）

| 场次 | 截稿 AoE | 通知 | 会议 | 与冻结 W24–W26 / 45 周 |
|---|---|---|---|---|
| CoRL 2027 | 摘要 5/24 估；全文 5/27 估 | 9/3 估 | 11 月估 | 冻结后 3.5 周，W27–30 写作 ✓ |
| NeurIPS 2027 main | 5/3 估；5/5 估 | 9/23 估 | 12 月估 | 撞 W26 冻结 ✗ |
| NeurIPS 2027 E&D | 与 main 相同（官方 identical） | 9/23 估 | 12 月 | 同上 ✗ |
| ICLR 2028 | 9/17 估；9/24 估 | 12/15 估 | 2028-04 估 | W46–47，超窗 1–2 周 ✓ 最晚 |
| ICML 2027 | ≈2027-01（W9） | ≈5 月 | — | 太早 ✗ |
| ICML 2028 | ≈2028-01 | ≈5 月 | 2028-07 | 窗口外 ✗ |

实测锚点：CoRL 2026 = 摘要 5/25、全文 5/28、通知 9/4、会 11/9–11（corl.org）；ICLR 2027 = 摘要 9/18、全文 9/25、决定 12/16、会 4/26–28（iclr.cc/Conferences/2027/Dates）；NeurIPS 2026 = 摘要 5/4、全文 5/6、通知 9/24（neurips.cc/Conferences/2026/Dates）。ICLR 2028 官方页 404（未发布）。**最晚可投 = ICLR 2028**。

## 2 接受率与偏好（2025–26）

| 场次 | 投稿 | 接受 | 接受率 | 来源 |
|---|---|---|---|---|
| NeurIPS 2025 main | 21,575 | — | 24.5% | 官方 blog 2025-09-30 |
| NeurIPS 2024 D&B | 1,820 | — | 25.3% | 官方 D&B blog |
| ICLR 2026 | 19,525 | 5,355 | 27.4% | 官方 blog 2026-03-31 |
| CoRL 2026 | ≈2,660* | 687 | ≈26%* | OpenReview 计 |
| CoRL 2025 | ≈1,155* | 263 | ≈23%* | PMLR v305 + OpenReview |
| ICML 2026 | 未公布 | — | ≈25–27%（估/待核） | 未找到官方统计 |

*OpenReview 编号上限推断，官方未公布，应视为区间。

偏好：四会基础率 24–27%，"挑软柿子"不成立。①**CoRL 是唯一在 CFP 写明真机鼓励的会**："Authors are encouraged to report real-robot experiments or provide convincing evidence that simulation experiments are transferable to real robots" → 纯仿真折价有据；但 sim-only 可中（JaxRobotarium CoRL'25 poster、SimplerEnv CoRL'24），测量/基准类基本只到 poster。②**NeurIPS** 2026 新增贡献类型 "Negative Results"，E&D 官方 scope 明写 evaluation 是独立研究对象、"Submissions need not introduce a new model or outperform prior work" → framing B 内容归属最正；但 E&D 与 main 时间线完全相同（官方 identical），撞冻结，不可投。③**ICLR** 无真机明文要求、理论最友好、有 discussion 期，27.4%；但对"机器人味"有折价（估计 −0.1~0.15 S）。

## 3 三 framing 对照

| | A 测量理论 | B 基准与协议 | C 认证修复回路 |
|---|---|---|---|
| 卖点 | K²U 层-核 + 测量律 Θ(K²U/Δ²) + 相变 κ*=Δ/(2H) + 重试反演 q=1−(1−p)^{R+1}（实测指数 0.97/1.98/3.03）+ β 墙 | DexSkillBench 协议 + 开源 harness + 负结果（校准不组合） | 测量→定位→修复→认证闭环，LLM agent 组合收益 |
| 最合场次 | CoRL（theory 明列）/ICLR | NeurIPS E&D（内容最正，日历出局） | CoRL（应用叙事） |
| 最可能拒因 | 4 技能撑不起普适律；E1 12 有序对 vs K²=16；无真机 | 4 技能/单模拟器规模不足；main track 负结果难 | 4 技能 +3.8pp 不显著、基线弱 |
| 需补 | T1–T3 证完 + K²U 匹配下界 + 4 自对 + 定理↔实验表 | ≥8–10 技能/2 模拟器 + 外部基线 + 全开源 | 强基线×3 + 跨 LLM backbone |
| Δp（CoRL/ICLR） | +2~3 / +3~5 | 0（poster 级）/ 条件合但不可投 | +3~5，但稀释理论，总 S 不升 |

## 4 单会 >50%：诚实判断

- **都达不到**：CoRL 2027 42%（36–48）> ICLR 2028 40%（34–45）> NeurIPS/ICML（日历出局）。
- 阈值：p=σ(1.15(S−3.66))×exec；exec=0.85 → S≈3.97，exec=0.88 → S≈3.90。现状 3.71–3.76、Round 4 修复后 3.80–3.85，差 0.05–0.15 S。
- 要补（≤3 条，ROI 序）：①T1–T3 证完 + K²U 匹配下界 + 定理↔实验表（4–6 周、¥0、+2~3pp）；②成本-精度实验（接口表 vs 逐计划，N=10/100、K=4/20；2 周、¥200、+1.5~2pp）；③4 自对 1,200 eps + 6 技能负结果 + W6 三阈值硬过（2 周、¥400、+1~1.5pp）。全做 → CoRL 46–48%，仍 <50%；再 +4–6 周做分离定理 headline（S≈3.94）才摸到 ≈50%，超窗 → 应转 ICLR 2028 或 TMLR（50%）。

## 5 推荐路径

1. **不放弃 CoRL 2027 主投**：全文截稿在冻结后 3.5 周，W27–30 写作正好；sim-only 折价用"接口级可证伪预测 + 负结果 + 开源 harness"抵消。framing 取 **A-core + B-shell**（CoRL CFP 明列 theory 与 benchmarks，审稿人懂 dexterous/skill library）。
2. **不改投 NeurIPS**：E&D 内容最合 B，但 timeline 与 main 相同（比 CoRL 早约 4 周）撞冻结；9 月通知与 ICLR 截稿重合 → 拒后零改稿窗口。**不改投 ICML**：2027 太早、2028 窗口外。
3. **备投 ICLR 2028**（W46–47，超 1–2 周）：CoRL 9 月出结果 → 3 周改稿 + ICLR discussion，是唯一真实级联（55–58%）。
4. 风险：CoRL 2027 若沿 2026 的 687 篇规模继续扩张，poster 门槛或降但 oral 更凶；新增 rebuttal + reciprocal reviewing（1 人约 1 周成本）；W6 fallback（S≈3.60）→ CoRL 37%，直接转 TMLR。

注：接受率除标注外均来自各会官方 blog；CoRL 投稿数为 OpenReview 编号推断（待核）；2027/2028 日期为按 2026/2027 实际日期推断（估计）。
