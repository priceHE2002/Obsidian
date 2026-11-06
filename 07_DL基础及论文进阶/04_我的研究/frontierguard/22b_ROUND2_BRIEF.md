# 22b · Round 2 发散任务书（给 8 个独立 subagent）

## 背景（读文件前先看这一段）
- 项目：`/Users/heyuhang/Desktop/frontierguard/`。目标：一份能冲 ICLR 2028 / CoRL 2027 的论文方案，**裁判（judge）判定中稿点估计 >50%** 才停。
- 当前方案：`21_synthesis_v1.md`（CSC：技能库组合成功的部分识别 + 证书宽度设计变量 + 证书最优分配）。Round 1 红队 `22_redteam_v1.md` 判定：**S=3.51（自评 3.96），CoRL 38% / ICLR 37% / 级联 48%，不能 >50%**。
- 红队结论：骨架可活，但①T1 是 Fréchet–Hoeffding 教科书结论、未定义 plan-induced 边际；②T2 只有上界、缺匹配下界；③T4 与训练期 C-LGA 指数不一致、主算法无定理；④ρ≈20 无算式；⑤4 技能/26 runs 被判 toy、外部基线被删；⑥预算漏 G6/LLM，真值 ≈¥4,153（余 ¥847）。
- 红队认为唯一能活的叙事 D：**"组合成功的认证是独立统计问题——最优样本分配、样本复杂度、校准何时无关的相变；给匹配上下界 + 相变 + 受控验证"**；三件套全做也只到 S≈3.75（级联 55%）。
- 硬约束：纯仿真、1 人 45 周、租卡 ≤¥5,000、本地 4070 Ti Super 16GB/32GB；4 技能（S1 Reorient-Any、S2 Rotate-Z、S7 Peg-Insert、S8 Cap-Twist）+ 2 长程任务；fixed-K K=1221/B=500M；主协议可改 segmented composition（技能间 reset + 入口态重放 + β 一阶化）；允许改核心对象、允许加 1–2 个技能、允许用合成 oracle（CPU 免费）做 20–100 技能 scaling。**不许真机、不许编造引用**。

## Round 2 的唯一目标
提出**一个**能比 21 更强的核心贡献（或对 21 做一次质变级强化），使理论新颖性 N 与实验严谨性 Exp 同时上一个台阶，让 critic/judge 有理由给 S≥3.95（exec≥0.85 时单会 >50%）。**不要拼盘**：每个提案只能有一个中心对象，其余都是它的定理/实验/附录。

## 你必须回答的问题（按你的视角选 2–3 个，全部给数字/算式）
1. 这个核心对象相对 21 的**增量到底新在哪**？审稿人最强一击是什么、如何预驳？
2. 能写出哪 1–2 条**非教科书**的定理？陈述、假设、证明思路、常数；若做不到，诚实说"只能做到 X"。
3. 实验上如何把"4 技能像 toy"抬到可辩护（合成 20–100 技能 scaling？外部 baseline？第 5 技能？2B steps 叙事？）。
4. 执行成本（人日/GPU-h/¥/周）与 W6/W10 gate 判据；exec 风险 0.85 是否可信。
5. 对 S 的诚实增量（N/TQ/Exp/Insight 各多少），以及 CoRL/ICLR 两口径点估计。
6. 如果这条路也到不了 4.0，你认为**最诚实的替代目标**是什么（例如"ICLR 主投、级联 50–55%"或改投 D&B）。

## 输出格式（写入 `/Users/heyuhang/Desktop/frontierguard/20_ideation/r2_<你的编号>.md`，中文 ≤1,200 字）
十个字段：标题 / 一句话 thesis / 核心形式化对象 / 2–3 条定理（陈述+思路）/ 与 21 的差异（≥3 条）/ 实验最小包（含成本）/ 预算与排期 / 对 S 与两口径概率的增量 / 最强审稿人攻击与预驳 / 最大风险与 fallback。引用用 arXiv API 核实，标注"已核实/经典期刊"。

## 可用素材
- 必读：`21_synthesis_v1.md`、`22_redteam_v1.md`、`20_ideation/lens_08_verification.md`、`lens_03_composition.md`、`lens_04_agent.md`、`lens_02_exploration.md`、`lens_01_information.md`、`lens_05_measurement.md`、`lens_06_scaling.md`、`lens_09_new_object.md`、`lens_10_venue.md`、`20_IDEATION_BRIEF.md`、`15_proposal_dexterous_v3.md`（v3.2 资产表/协议）、`16_acceptance_dexterous_final.md`（judge 口径）、`17_theory_bound_dexterous.md`（理论模板）。
- 关键数字：judge 映射 CoRL 质量≈σ(1.15(S−3.66))、ICLR≈σ(1.15(S−3.74))；期望=质量×exec；当前 S=3.60、质量 47%、exec 0.75。
- 时间：今天是 2026-11-02；ICLR 2028 截稿 ≈2027-09（W46），CoRL 2027 ≈2027-06（W31，先投后改）。
