# Lens 08 — 可验证性 / 安全性 / 风险控制

**判断：有条件能。** 但"conformal + shield"本身是套壳；只有把问题重述为**组合成功的部分识别**，让理论改变采样与执行决策，才可能到 3.95。与 lens 03 的 CD-SNR 是同一组合缺口的不同形式化（它换决策对象，我换识别对象），**不要同时主推**。

## 候选 A：Certified Skill Composition（CSC）——组合成功的部分识别 + 证书驱动采集
- **thesis**：per-skill 校准不能证明 plan 成功；组合成功是部分识别问题，证书宽度是可设计变量。我们给 sharp bound、依赖感知收紧、PAC 证书、执行盾与证书最优分配。
- **缺口**：现有"校准每技能再相乘"隐含独立。同一组 marginals 下，H=3、p=0.7 的 plan 成功可在 [0.10,0.70] 内；乘积 0.343 无保证，Fréchet 下界低 24pp；p=0.6 时下界为 0，纯校准证书 vacuous。LLM planner 的 argmax 恰好落在这一带。
- **形式化**：`C(π,x)=[L,U]`，`P(∀π∈Π:S(π,x)∈C)≥1−α`；`Δ=Δ_dep+Δ_retry+Δ_sel+Δ_shift`。边际-only 不可能性：任何只依赖 marginals 的有效下界 `≤max(0,1−Σ(1−q_i))`。收紧：对 LLM 实际查询的 plan 做联合 rollout，latent-class/copula 估依赖（标为 model-conditional），assumption-free 界才是无条件保证；形成"查询→证书→补采样"主动认证环。分配定理：Fréchet 宽度 `Σ√(v_i/n_i)` 最小化给 `n_i∝v_i^{1/3}`；乘积界给 `n_i∝(v_i/q_i²)^{1/3}`，与 success-optimal Neyman `n∝√(wv)` 不同——**证书最优 ≠ 成功率最优**。
- **可证伪**：①L1/L2 frontier 上 `|乘积−真实联合|≥8pp`；<3pp 则组合缺口可忽略，转负结果。②边际-only 乘积证书在正依赖下覆盖率 <1−α（如 80% vs 95%）；若已覆盖，依赖修正不必要。③同 abort rate=20% 下，证书盾把 undetected failure 降 ≥30%（vs random block）、≥15%（vs conformal-only）；否则盾是装饰。④C-LGA 同预算 certified success +≥3pp、平均成功率代价 ≤1pp；若平坦，分配对证书不重要。
- **最小包**：E1 覆盖率审计（5 plan×200 联合 rollout×2 任务×8 seed≈16k eps，~5 GPU-h）；E2 缺口分解（复用 agent 日志+200/plan，~10 GPU-h）；E3 盾 A/B（4 条件×100 配对=400 eps，LLM≈¥300）；E4 偏移压力（3 档×2 任务×50=300 eps，~10 GPU-h，¥230）；E5 C-LGA（离线重加权+2 seed×2 方法确认重训=4 runs，~24 GPU-h，¥170）。合计 **12–16 周、70–100 GPU-h、增量 ¥600–900**；从 G1b/G2/6 技能 gate 砍。依赖：现有 4 技能/L1/L2/CRN/LLM harness；把 L1/L2 钉死为**同一 episode 状态延续**（若现有 harness 在技能间 reset，W4 先修），并加联合 rollout logger；否则组合故事塌缩。
- **区别**：P2P/ADEPT：规模+真机，我们不比；OmniReset：reset 多样性；SGS：success-band 分配；SER：reset 时序；LGA：只最大化成功率/校准，C-LGA 加联合 plan 采样与证书目标。CP-RL 式 per-policy conformal（如 2311.01457）与 safety filter（2306.02551）给 per-step/per-policy 覆盖，不给 plan 级、跨技能依赖保证；conformal risk control（2208.02814）/anytime-valid（2602.04364）只管单风险/序贯有效性，不解决联合识别；shielding（1708.08611）假设已知不安全集，我们是未知成功概率下的证据性弃权。
- **不是 incremental**：审稿人原话："conformal 套壳/标准工具拼装。" 反驳：贡献不是 conformal，而是①组合成功的 sharp 部分识别界与边际-only 不可能性；②证书驱动采集（查询分布决定采样）这一新的实验设计问题；③证书最优分配定理。conformal 只用于 shift 项，且我们明确报告 assumption-free 界。
- **S 归因**（推断权重 N.25/TQ.20/Exp.20/Ins.20/C.10/Rep.05）：乐观 N+0.75、TQ+0.20、Exp+0.20、Ins+0.55、C+0.10、Rep+0.05 → **+0.39，S≈3.99**；保守 N+0.55、TQ+0.15、Exp+0.15、Ins+0.45、C+0.05、Rep+0.02 → **+0.29，S≈3.89**。命中 3.95 约 **50–55%**，取决于 N/Insight 能否拿上限。
- **venue**：ICLR 2028 / NeurIPS 2027（部分识别+实验设计更贴 ML）；CoRL 2027 需把状态延续 benchmark 与盾放前面，理论压附录。
- **风险/rebuttal**：Fréchet 界"vacuous"——是，我们量化 marginal-only 的代价并给联合数据收缩曲线；copula 任意——assumption-free 界不依赖它，模型只收紧，且用 held-out plan 做 misspecification 检验；盾=fallback——我们测 undetected failure 与 abort tradeoff，和 random/conformal-only 同 abort rate 比；4 技能/sim-only——开源状态延续 benchmark + P2P/ADEPT 外部锚；**纯仿真让 ground-truth coverage audit 可行且便宜**，这是内部效度优势，不是折价。

## 候选 B：Runtime Assurance Contracts——LLM 技能调用的"声明成功 vs 证据"
- **thesis**：LLM agent 的 silent failure 是任务级、非安全级的；给每个技能调用加前置/后置条件契约与 monitor，planner 在证据不足时 abstain/retry，风险预算约束 undetected failure。
- **缺口**：shielding 防已知不安全动作，不防"agent 以为成功"；conformal 给边际覆盖，不给任务级证据。我们测 **belief–evidence gap**（agent 的 plan 成功估计/自报完成 vs 后置条件），并给 risk–coverage frontier。契约 `c=(pre,post,p,ci,monitor)`，执行 iff `monitor=pass ∧ L≥τ`；`P(undetected)≤(1−se)P(fail)+α_mon`，se 由 held-out 标注估下界。
- **可证伪**：①gap≥15%（否则 silent failure 不是一等问题，负结果）；②contract+abstain 在同覆盖率下降 undetected failure ≥25%；③gap 随步数/技能数单调上升。
- **最小包**：复用 840 episodes 日志 + 120 条双人标注（κ≥0.8）+ 200 配对干预；LLM≈¥600，0 新训练，~10 GPU-h，4–6 周；需 postcondition monitor（现有 env 可读插入深度/横向误差/盖角）。
- **区别**：vs shielding——任务级 silent failure，不是安全约束；vs conformal——证据来自 postcondition，不是分布覆盖；vs Foresight/Rewind-IL（2606.23085/2604.16683）——我们做 agent 级契约与 abstention，不做新检测器；vs introspective planning（2402.06529）——它让 planner 表达不确定性，但不验证 postcondition。
- **不是 incremental**：审稿人："monitor+abstain 是 runtime verification 老套路。" 反驳：新的是在 LLM-orchestrated dexterous 任务上测 belief–evidence gap，并把契约/弃权接到风险预算与 planner；是测量+系统贡献，不是新 monitor。
- **S 归因**：N+0.35、TQ+0.10、Exp+0.25、Ins+0.40、C+0.10、Rep+0.05 → **+0.25，S≈3.85**；上限 3.90。
- **venue**：CoRL 2027（系统+测量）；ICLR 风险大。
- **风险/rebuttal**：novelty 低——用 gap 的规模与 dose-response 撑；monitor 误报——报 sensitivity/specificity 与保守上界；4 技能——同一限制，诚实写。

## 结论
有略高于 50% 的路（约 50–55%），但只在 A 的"部分识别+证书驱动采集"版本；它把理论变成算法（采样分配/主动认证），直接回应 judge 的"theory decorative"。B 是稳健 fallback（S≈3.85–3.90），不足以单独到 3.95。建议 A 为主、B 为 W8 gate 降级；W8 先测缺口与符号，若 `|gap|<3pp` 或盾无增益，转 B。与 lens 03 合并时只保留一个 §4 核心，另一个压成推论/附录，避免"两个理论拼盘"。

**Refs（已核）**：2208.02814, 2311.01457, 2306.02551, 1708.08611, 2606.00090, 2606.23085, 2604.16683, 2602.04364, 2402.06529.
