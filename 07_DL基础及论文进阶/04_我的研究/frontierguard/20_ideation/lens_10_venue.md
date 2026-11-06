# lens_10｜投稿策略：高录用论文原型逆向工程

**结论先行**：推荐"B 的底稿 + A 的包装"——CoRL 2027 交可复现协议，ICLR 2028 交分离定理，同一批实验不重跑。venue 本身只值 +5pp，杠杆在"结论形态"而非会场。

## 一、一手核过的证据（2026-11）
| 论文 | 录用 | 出处 |
|---|---|---|
| ReMix：数据配比优化 | CoRL'24 **oral** | PMLR v270:145–164 |
| PoliFormer / Cross-Embodied Scaling | CoRL'24 oral | v270:408–432 / 496–512 |
| Unpacking Failure Modes（纯测量） | CoRL'24 **poster** | v270:689–723 |
| QA-Calibration of LM Confidence | ICLR'25 poster | OpenReview D2hhkU5O48 |
| Reconciling Model Multiplicity（决策校准） | ICLR'25 poster | OpenReview uy4EavBEwl |

待核实：Skill-it!(2307.14430)、Playground(2502.08844)、Being-0(2503.12533) 页面无 venue 注记。

读数：CoRL'24 约 30 篇 oral 几无例外含真机或超大规模，测量类只到 poster；ReMix 是 oral ⇒ "数据/预算分配"已被占位，LGA 正撞此处（N=3.25 的来源）。

## 二、五种原型 × 适配
P1 分离/不可能性定理+构造反例+受控验证（QA-Calib、Reconciling）：★★★★★，CPU 可跑，唯一同时满足"1 人/纯仿真/¥5k"且天花板高的。P2 测量/负结果+协议（Agia）：★★★★，天花板=poster/D&B。P3 分配优化+规模化（ReMix）：★★，需规模且赛道已满。P4 基准/基础设施（Playground）：★★★，4 技能太少（需≥10 任务）。P5 系统+真机（P2P/ADEPT）：★，不可行。

## 三、候选 A（激进）Calibration Does Not Compose
**thesis**：per-skill 边际校准对组合决策既非必要也非充分；对象应是计划排序泛函的决策校准误差 CCE=sup_{π,π'}|P(π≻π')−Φ(Δ̂/D̂)|，存在 ρ=Δ/D 相位边界。
**缺口**：decision calibration 只覆盖单预测器；组合（H 步、重试、接口偏移 β）版本空白，且 β 不随 n 消失。
**三定理**：T1 分离——ECE≤ε 而 regret Θ(1)（构造 p_A=1−p_B）；T2 充分条件；T3 分配错配——ECE 最优 n∝(w√v)^{2/3} vs 决策最优 n∝√v，frontier 带内差 5.5×。
**预测**：①4 方法 8 seeds，ECE 与组合成功率 |ρ|<0.3；②合成反例 ECE=0、regret≈0.4；③Δ/D>√(2log(2N/δ)) 时损失骤降；④接口偏移解释残差。
**最小包**：合成套件（CPU 1.5 周）+复用 CRN 日志（0 GPU）+12 runs×N=2048（38 GPU-h/¥270）+agent 300 ep（¥250）≈¥600；主网格 493→120 GPU-h，省 10 周。
**最小改动量**：删 G1b（32 runs/100h）、G2 18→8 runs、G1a 8→6 seeds，新增合成套件与 2 个复核 run；代码复用 ≥80%。
**区别**：P2P/ADEPT=规模系统；OmniReset/SGS=reset 分布；SER=reset 时机；LGA=估计器级分配；ReMix=IL 配比（无决策对象）；CCA(lead_seed)/CD-SNR(lens_03) 是同族更强定理 → A 的差异是用可证伪负结果钉住前提，不再叠一个上界。
**非 incremental**：否证"per-skill 校准即可"这一默认前提，反例为 ECE=0 而 regret Θ(1)。
**ΔS**：N+0.70 TQ+0.20 Exp+0.05 Ins+0.55 C−0.05 Rep−0.10 = **+0.335，S≈3.94**。归因：N 涨在新形式化对象而非新任务；Ins 涨在反例与相位边界；Rep 降因无真机。
**venue**：CoRL 2027 机器人版先投 → ICLR 2028 主投。
**风险/rebuttal**：①"把 2021 决策校准搬到机器人"→组合版 scaling 不同（H、重试）+相位边界可预注册；②"反例太玩具"→反例是证明，实例只验预测；③"4 技能"→定理不依赖技能数。

## 四、候选 B（稳健）DexSkillBench-C
决策加权校准协议（CRN+decision-weighted ECE+剂量-反应）为交付物，负结果为主结论；沿用 83 runs/¥4,809，仅重写 §1/§4 并预注册，0 GPU、+3 周。
ΔS：+0.25/+0.20/+0.10/+0.15/+0.05/+0.05=**+0.155，S≈3.76**；CoRL 42%，级联 46%。
本形态本质增量，天花板受此封顶。风险："another benchmark + ECE 表"即拒；rebuttal 只能靠预注册+CRN 可复用性，故需在摘要首句就把"负结果"写成主张而非副产品。

## 五、日历与专门判断
ICML'27（≈2027-01）无结果，弃；NeurIPS'27 main（≈2027-05）早 4 周且 9 月出结果撞 ICLR 截稿，弃；NeurIPS'27 D&B 同截稿、需 ≥10 任务，只作机会项；**CoRL'27（≈2027-06，8 月出结果，无 rebuttal）→ ICLR'28（≈2027-09 下旬，有讨论期）**，5–6 周窗口成立（日期待核实）。故"一套实验、两套 framing"：W20–W26 先写 ICLR 理论附录，二次改稿压到 ≤3 周。

**判断一：转"决策校准理论+受控实验"后 ICLR/NeurIPS 概率更高吗？是，但仅 +3~6pp。** ICLR'25 收录的两篇校准/决策理论均无机器人、无规模（一手核实），ML 会审"一般性主张+可证伪形式化"，不审硬件。前提是 ML-native：主张须关于估计器与决策规则，机器人只是实例；若仍以 benchmark 为骨架，会被判"domain-specific engineering with a theorem"。

**判断二：CoRL 因无真机/规模小而折价更多吗？是，多 3~5pp。** 折价落在 N 与 C 两维（≈−0.15~−0.25 S），对"现象类"主张折价小于"能力类"（Agia 纯测量仍中 poster）。ICLR 对"机器人味"折价 −0.1~−0.15 S，framing 得当可消到 −0.05，且不折 scale。

**算术诚实**：A 的 CoRL 质量 58%×0.78≈45%、ICLR 51.7%×0.85≈44%、级联 ~50%（强相关，级联增益有限）。venue 只值 +5pp；破 50% 需 S≥4.05——靠定理落地，不靠换会场。
