---
tags:
  - 论文
  - X光
  - VLA
  - 具身智能
  - 空间记忆
  - 长程操作
created: 2026-11-16
paper_title: "STVLA: Spatial-aware and Temporal-memory for Vision-Language-Action Models"
paper_authors: "Zhuo Han, Yeqiang Wang, Xinran Wang, Hongwei Han, Yi Xiong, Zunian Wan, Jianbo Mao, Huan Peng, Yongcheng Liu, Wei Shan, Maoqing Yao"
paper_year: 2026
paper_venue: "arXiv preprint (cs.RO)"
paper_url: "https://arxiv.org/abs/2611.16589"
arxiv_id: "2611.16589"
method: "STVLA"
code_url: "https://stvla.github.io/"
---

# STVLA_X光

**STVLA: Spatial-aware and Temporal-memory for Vision-Language-Action Models**
*Zhuo Han, Yeqiang Wang, Xinran Wang, … Maoqing Yao（AgiBot 智元机器人） | arXiv 2026*

## 论文说了什么

**问题**：VLA 策略大多只看当前观测。要它做多步任务，缺两样东西——**空间结构**（物体在哪、被什么挡住）与**任务历史**（刚才做到哪一步）。现有两条补法各自残缺：把历史帧的视觉 token 当记忆（MemoryVLA 这类），遇到遮挡和视角变化就失效；只用当前场景的空间表示，又记不住「场景是怎么变的」。失败表现很具体：反复抓取、空抓、忘记已放好的物体。

**视角**：既然几何基础模型已经能给出与视角无关的场景结构，那就**把几何 token 也当成一种记忆模态**——历史库里每个 entry 同时存视觉 token 与空间 token，检索时一起查、一起用。一致性不靠外观相似，而靠几何对应。

**结果**：LIBERO 97.8%、LIBERO-Mem 41.0%、MIKASA-Robo 45.2%、SimplerEnv Bridge 74.0%，四个基准均为对比方法最高；实机三任务均值 76.7%，比 π0.5（35.6）、Spatial Forcing（36.1）、MemoryVLA（46.1）高出一大截。

## 方法拆解

1. **空间 token 化**：冻结 1B 的 VGGT-Ω，输入 512 px 头部相机窗口（前序帧作上下文），取当前帧 1024 个 patch 特征（宽度 2048）。一个可训练压缩器用**学习 query 的池化交叉注意力**把它们聚成紧凑潜变量，再过**变分瓶颈** $q_{t,i}=\mathcal{N}(\mu_{t,i},\mathrm{diag}(e^{\ell_{t,i}}))$，重参数采样投影到语言模型宽度，得到 $N_s$ 个空间 token。辅助目标（仅训练时）：
$$\mathcal{L}_{recon}=\frac{1}{N_s d_\omega}\sum_i\|\hat\omega_{t,i}-\bar\omega_{t,i}\|_2^2,\qquad \mathcal{L}_{KL}=\frac{1}{N_s d_z}\sum_i D_{KL}\big(q_{t,i}\,\|\,\mathcal{N}(0,I)\big)$$
目标 $\bar\omega$ 是冻结几何特征的固定平均池化。推理时用后验均值代替采样。
2. **有界记忆库**：$X_t=[V_t;S_t]$（视觉 token 拼空间 token），库内只保留本 episode 更早的步，$H_t=(X_k)_{k\in I_t}$，$|I_t|\le L_m$；超出容量丢最旧。无历史时当前 token 直通。
3. **解析式检索打分**：对第 $j$ 条记忆，用 token 均值 $h_j$ 与当前均值 $\bar x_t$ 定义相关性 $r_j$ 与非冗余度 $u_j$，
$$r_j=\tfrac12\big(1+\cos(h_j,\bar x_t)\big),\qquad u_j=\frac{\|h_j-\bar x_t\|_2}{\|h_j\|_2+\|\bar x_t\|_2},\qquad s_j=r_j u_j\in[0,1]$$
两者相乘 = 「既像当前场景、又不重复当前场景」。分数标准化后作为**加性偏置**塞进 token 级注意力：
$$\tilde s_j=\frac{s_j-\mu_s}{\tau_r\max(\sigma_s,\epsilon)},\qquad \alpha_{j,n}=\mathrm{softmax}_{j,n}\!\left(\frac{q^\top k_{j,n}}{\sqrt d}+\beta\tilde s_j\right)$$
$\tau_r=1$、$\beta$ 可调（默认 0.5），**不引入任何新参数**。
4. **门控融合 + 分段掩码**：$L$ 层交叉注意力取出 $M_t$，学习到的逐元素门与当前 token 融合，视觉与空间 token 保持两组分别进策略前缀。block-causal 掩码把前缀分成「视觉」「空间」「指令 + 状态」三段，动作 token 可读全部前缀、段内双向，空间段被屏蔽在原始前缀之外——既保留原 VLM 的注意力模式，又让空间信息直达动作生成。
5. **动作头**：沿用 backbone 的 flow-matching，$x_t^\tau=\tau\epsilon+(1-\tau)a_t^\star$，
$$\mathcal{L}_{act}=\mathbb{E}\Big[\tfrac{1}{d_a}\big\|v_\theta(x_t^\tau,\tau,c_t)-(\epsilon-a_t^\star)\big\|_2^2\Big]$$
该损失同时反传进空间压缩器。
6. **训练**：8×H100，batch 16/GPU，输入 224×224；实机任务用 237 条示教训 20k 步，每方法 20 trials × 3 个目标高度。

## 结果

| 基准 | STVLA | 最强对照 |
|---|---|---|
| LIBERO（五套均值） | **97.8** | π0.5 96.6 / MemoryVLA 96.5 |
| LIBERO-Mem（十个记忆任务） | **41.0** | π0.5 38.0 |
| MIKASA-Robo（五个记忆密集任务） | **45.2** | MemoryVLA 41.2 / π0.5 32.4 |
| SimplerEnv Bridge（四任务均值） | **74.0** | MemoryVLA 71.9 |

SimplerEnv 的 Stack Cube：54.2% vs 37.5%（MemoryVLA）。

实机（Table V，SR %）：

| 方法 | 叠方块 | 物体整理 | 手机装盒 | 均值 |
|---|---|---|---|---|
| π0.5 | 41.7 | 55.0 | 10.0 | 35.6 |
| Spatial Forcing | 58.3 | 40.0 | 10.0 | 36.1 |
| MemoryVLA | 88.3 | 35.0 | 15.0 | 46.1 |
| **STVLA** | **100.0** | **85.0** | **45.0** | **76.7** |

消融（实机叠方块）：去掉记忆 100→95.0（失败变成执行犹豫）；去掉空间分支 →96.7（失败变成空抓与放置错位，两种失败模式互补）；检索偏置 $\beta=0$ 时 80.0%，$\beta=0.5$ 时 100%，$\beta=1.0$ 时 81.7%——收益不随偏置单调增长；VGGT-Ω 输入帧 $K=1/2/3/4/5$ 对应 88.3/93.3/85.0/100.0/86.7，$K=4$ 最优。

## 锐评

- **对照一：π0.5（同代 VLA 底座）**。LIBERO 只差 1.2 点（97.8 vs 96.6），实机却差 41.1 点（76.7 vs 35.6）。这说明 LIBERO 这类仿真基准已经饱和到测不出差异，而「空间 + 历史」的收益只在真实多步、存在遮挡与状态变化的场景里才显形。把 LIBERO 当主要证据的 VLA 论文，现在都该被这个差距提醒。
- **对照二：MemoryVLA（外观记忆）**。LIBERO-Mem 只赢 3 点（41.0 vs 38.0），实机赢 30.6 点，其中手机装盒 45.0 vs 15.0。差异来源清楚：外观 token 在遮挡与视角变化下会被误检索，几何 token 不会。这条对照直接把「记忆该记什么」这个问题摆上台面——记外观不如记结构。
- **对照三：Spatial Forcing（有空间、无记忆）**。实机均值 36.1，几乎等于 π0.5 的 35.6；叠方块 58.3 略好，物体整理 40.0 反而更差。说明**空间信息单独用不够**，必须有「可检索的历史」把它串起来；消融里 w/o spatial（96.7）与 w/o memory（95.0）的不同失败模式也印证二者互补。
- **真实局限**：作者自述依赖冻结 3D 基础模型的特征质量，特征不好时空间 token 也差；记忆窗口很短，显著更长的任务上收益会掉。此外检索打分是特征空间的代理，作者明说它「不是任务信息增益的直接估计」；$\beta$ 与 $K$ 都是非单调超参（$K=5$ 比 $K=4$ 掉 13.3 点），换平台重调的成本不低。评测全在 AgiBot 自家平台 + 桌面任务，跨本体、跨场景的结论未给。

## 借鉴意义

- **记忆用检索而非拼接**：把历史放外部库、用解析分数挑少量 entry，避免把长上下文硬塞进 prefix；这条对任何有历史依赖的策略都成立，而且不用训练新模块。
- **双模态记忆共享一个 entry**：视觉与几何 token 在同一时间步绑定、同一次 cross-attention 里检索，天然保证「同一时刻的外观与结构」对齐；拆成两个 bank 会引入错配。
- **零参数相关性 × 非冗余打分**：$s_j=r_ju_j$ 可以原样搬到视频世界模型的记忆检索、检索增强生成的多模态库里。
- **变分瓶颈压 token**：几何特征维度高（每帧 1024×2048），用 VAE 式压缩 + 重建监督保住几何信息，是在固定 token 预算下扩容的实用手段。
- **基准饱和警告**：LIBERO 上的 1–2 点差异没有判别力，真实任务是唯一可信的证据来源。

## 关联

- [[2026-11-16-论文推荐]]：当日日报条目。
- [[09_OnlyAct_X光|OnlyAct]]：同样从冻结的 V-JEPA 2 抽表示喂给轻量策略头，路线是「大表示 + 小适配」；STVLA 把这套用在几何 + 记忆上。
- [[12_A2WM_X光|A²WM]]：A²WM 用局部动作灵敏度给 QP 解、不重训策略；STVLA 用几何记忆给动作条件、不重训 backbone——都是「冻结大模型 + 外挂可解模块」的思路。
- [[14_Topo-4DGS_X光|Topo-4DGS]]：一个在表示层记录场景的生灭，一个在策略层检索场景的历史；若把 4DGS 的时变表示接进 STVLA 的记忆库，是「几何记忆」的一条自然延伸。
