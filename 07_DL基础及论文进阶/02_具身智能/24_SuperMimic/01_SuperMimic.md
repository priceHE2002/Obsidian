---
tags:
  - 论文
  - 具身智能
  - 人形控制
  - 世界模型
  - MBRL
  - CoRL2026
created: 2026-11-08
paper_title: "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model"
paper_authors: "Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath (UC Berkeley)"
paper_year: 2026
paper_venue: "CoRL 2026 Oral"
paper_url: "https://arxiv.org/abs/2610.28480"
github: "https://supermimic.github.io"
---

# SuperMimic

**SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**
*UC Berkeley（Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, Koushil Sreenath）| CoRL 2026 Oral | arXiv 2610.28480*

> **"算力不该是瓶颈。"** 人形全身控制这两年走的是「跟踪越来越大的动作库」这条路，但每加一批动作就要多加一批 GPU——SONIC 是 128 张 L40 训 7 天，约 21,000 GPU-hours。SuperMimic 用**一个可微世界模型**把这份算力砍到 1 张 H100 × 3 小时：GPU-hours 少 1000 倍以上，129,514 个动作的失败数从 179 降到 25，9.8 小时双卡后零失败，策略直接上 Unitree G1。

---

## 一、问题：动作跟踪的算力墙

物理动作模仿（physics-based motion imitation）是当前人形全身控制的主路线：给一段参考动作，训一个 RL 策略去跟踪它。要覆盖更多动作，无非两种做法：

1. **每段动作训一个专家，再蒸馏合并**成一个通用控制器（BeyondMimic 一类）；
2. **直接训一个通才策略**，吃下整个动作库（SONIC 一类）。

两条路都随动作数增长而变贵：蒸馏有损，通才要堆集群。SONIC 的账很直白——128 张 L40 × 168 小时 ≈ 21,000 GPU-hours。

作者指出的浪费点在于：**所有参考动作共享同一套人形刚体动力学**，而 model-free PPO 每次都要从带噪的仿真 rollout 里重新估计「这个动作会导致什么后果」。共享结构没有被显式建模，于是「扩展动作库」就只能等于「扩展算力」。

## 二、方法：一个世界模型 + 一段短想象

### 2.1 训练循环（三段交替，一个 loop）

每一轮做三件事，循环往复（论文 Figure 2 / Algorithm A.1）：

| 阶段 | 做什么 | 关键细节 |
|---|---|---|
| ① 采数据 | 当前策略在仿真器里跑，transition 进固定大小 replay buffer（新的覆盖最旧的） | 执行的是加噪动作 $a_t+\epsilon_t$，噪声只为拓宽世界模型的拟合分布，可选 |
| ② 拟合世界模型 | 从 buffer 采 $k=8$ 步的自回归窗口，最小化预测误差 | 世界模型是**确定性**的，条件在每 episode 的 sim context $\xi$ 上（摩擦、标定偏移、躯干质心） |
| ③ 改策略 | 在世界模型上跑 $H=32$ 步 imagined rollout，BPTT 反传 | 世界模型**冻结**；critic $V_\psi$ 在同一阶段训练，用来 bootstrap $H$ 步之后的 return |

因为动力学不依赖「是哪个策略采的数据」，旧策略的 transition 依然可以用来拟合世界模型——这让 buffer 能持续复用，学习与控制一起进步，而不是分阶段。

### 2.2 世界模型

$$\hat{s}_{t+1}=f_\omega(\hat{s}_t,a_t,\xi),\quad \hat{r}_{t+1}=r_\omega(\hat{s}_t,a_t,g_t),\quad \bar{\hat{o}}_t=g_\omega(\hat{s}_t,\xi),\quad \hat{d}_t=d_\omega(\hat{s}_t,g_t)$$

- 预测的是**无噪声观测** $\bar{o}_t$；传感器噪声 $\eta_t$ 在世界模型之外重新加回去，策略看到的是 $\bar{\hat{o}}_t+\eta_t$。
- 终止是软的：仿真器给硬标签 $d_t\in\{0,1\}$，模型输出终止概率 $\hat{d}_t\in[0,1]$，这样存活权重在想象 rollout 里可微。
- 损失是四项之和：

$$\mathcal{L}_{WM}(\omega)=\mathbb{E}_{\mathcal{B}}\sum_{j=1}^{k}\Big(\|\hat{s}_j-s_j\|^2+(\hat{r}_j-r_j)^2+\|\bar{\hat{o}}_j-\bar{o}_j\|^2+\mathrm{BCE}(\hat{d}_j,d_j)\Big)$$

其中解析可算的头（比如某些观测分量）不加损失项。

关键立场：**世界模型不是「可微仿真器」**。仿真器本身不求导（MuJoCo Warp 的接触是非光滑的），SuperMimic 只是拿它记下来的 transition 拟合一个可微代理，梯度从这个代理走。

### 2.3 策略更新：BPTT 代替 likelihood ratio

想象 rollout 的目标：

$$\hat{J}(\theta)=\mathbb{E}_{s_0\sim\rho_0,\pi_\theta}\Big[\sum_{t=0}^{H-1}\gamma^t\hat{c}_t\hat{r}_{t+1}+\gamma^H\hat{c}_H V_\psi(\hat{s}_H,g_H,\xi)\Big]$$

- 起点 $s_0$ 从 replay buffer 采（Dyna 风格），带当时的 $\xi$。
- 世界模型可微，所以 $\nabla_\theta\hat{J}$ 可以直接沿 $H$ 步反传——**一阶解析梯度**。
- 对比 PPO：PPO 把动力学当黑盒，用 $\nabla_\theta\log\pi_\theta$ 乘 advantage 做似然比估计；BPTT 直接对每个动作求 return 的导数，短 horizon 下方差低得多。论文的实测是：达到 PPO 在世界模型上的最好 return，**imagined steps 少约 64 倍**。

## 三、结果

### 3.1 与 SONIC 的对比（129,514 个基准动作）

| 指标 | SuperMimic | SONIC（released checkpoint） |
|---|---|---|
| 训练算力 | 1× H100，3 h ≈ **3 GPU-h** | 128× L40，168 h ≈ **21,000 GPU-h** |
| GPU-hours | **>1000× 更少**（实际约 7200×） | 基准 |
| 归一化 return | 3 h 超过 SONIC 的 0.90 | 0.90 |
| 失败动作数 | **25**（3 h 后）；**0**（9.8 h / 2 GPU） | **179**（含全部 44 个 flip） |
| 墙钟时间 | >50× 更短（约 56×） | — |

数据侧还有个容易忽略的点：SuperMimic 只用了 BONES-SEED 的公开子集 **264.56 小时**，是 SONIC 611 小时的 2.3 倍少；硬件上 H100 相对 L40 的峰值优势最多 5.5×。作者把这两项都摊开算过：**2.3× 的数据差和 5.5× 的峰值差解释不了 1000× 的 GPU-hours 差**。

### 3.2 三个关键性质（消融结论）

1. **预测误差随数据规模稳定**：动作从 1 小时加到 264 小时（264×），世界模型误差不退化——这是「一个模型吃下整个动作库」成立的前提。
2. **rollout 比仿真器快 10× 以上**：想象比真跑仿真便宜，才有吞吐优势。
3. **解析梯度省样本**：同样 return，imagined steps 少约 64×。

### 3.3 真机与下游接口

- 单个策略直接部署到 **Unitree G1**，覆盖多样甚至杂技类动作，无需按动作调参。
- 初步实验：在同一世界模型上训一个**速度指令编码器**（用它自己的任务 reward），跟踪精度超过蒸馏基线——说明这个世界模型不只能训跟踪器，还能训下游接口。

## 四、锐评

**范式层面**：这篇的真正贡献不是「快」，而是把「扩展动作库」的成本结构改了。原来 scaling 的斜率由 GPU 数决定，现在由「世界模型能不能拟合住新动作」决定。后者是可研究的（误差随数据规模的变化可以画曲线），前者只能烧钱。论文标题里的 "making it practical to measure motion-tracking scaling laws" 是认真的。

**与同类工作的位置**：

- 对 **BeyondMimic / SONIC**：它们代表「model-free + 大规模」，SuperMimic 是「model-based + 单卡」；关键差别在梯度来源（采样估计 vs 解析）。
- 对 **Dreamer / TD-MPC**：同属 model-based，但 Dreamer 在潜空间做 actor-critic、TD-MPC 做短程规划，二者仍用 model-free 的方式更新策略；SuperMimic 是直接在学到的动力学上反传。
- 对 **SHAC / SuperTrack / SuperDyno**：同样是「可微代理 + 解析梯度」，但那些工作多在角色动画/四足规模，SuperMimic 把它推到了万级动作库 + 人形真机。
- 注意：以上归类来自论文 Related Work 与我的领域对照；与 SONIC 的数字对比是论文给的，其余不给具体数字。

**局限（按重要性）**：

1. **世界模型是从刚体仿真拟合的**，接触非光滑（脚落地、手抓取）处的误差会直接进策略梯度；真机只在 G1 上验证，跨本体（不同质量分布/执行器）没给。
2. **"3 小时"的账要看清**：它包含采集 + 拟合 + 策略更新的完整 loop，但不含世界模型之外的工程调参；论文自己也在附录里把 checkpoint 评估排除在外。端到端复现成本要按自己的仿真吞吐重算。
3. **软终止是近似**：用终止概率替代硬终止，让梯度可微，但也意味着世界模型可能「学会不终止」来骗高 return；论文没给这个失效模式的分析。
4. **仍是跟踪**：动作库之外的新技能（比如带物体交互的操作）不在覆盖范围，和 VLA 的语义泛化是两条线。

## 五、对 WAM / VLA 研究的启示

1. **世界模型可以当「梯度加速器」而不是「规划器」**。现在 WAM 论文大多把世界模型用于想象/预测（Motus、FastWAM、ReactWorld），SuperMimic 证明了另一条用法：只拿它提供可微梯度，不要求它生成得好看。对动作头训练，这可能比生成式预训练更直接。
2. **scaling law 的对照组**。动作数据 1→264 小时误差稳定，意味着可以设计「数据量 → 控制能力」的受控实验。具身智能现在最缺的就是这种可重复的度量。
3. **共享动力学的 amortize 思路可以迁移**：人形、四足、机械臂共享的是「刚体 + 接触」这套动力学，一个世界模型 + 少量本体适配，可能替掉 per-robot 的 RL 管线。

## 六、关联

- [[01_Motus]]：Motus 把世界模型当统一预测骨干；SuperMimic 把它当可微环境。同一件东西的两种用法，值得对照读。
- [[01_FastWAM]]：FastWAM 的结论是「推理时不需要想象」，SuperMimic 进一步说明「训练时想象也比真跑仿真划算」。
- [[01_World Models Survey]]：综述里「组合预测误差」是世界模型的根本瓶颈；SuperMimic 用 $k=8$ 自回归拟合 + 短 horizon（$H=32$）+ critic bootstrap 正是对这个瓶颈的工程回应。
- [[2026-11-08-论文推荐]]：当日日报的补录条目。
