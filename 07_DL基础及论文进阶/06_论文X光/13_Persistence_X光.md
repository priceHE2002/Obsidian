---
tags:
  - 论文
  - X光
  - 世界模型
  - 评测
  - 表征学习
created: 2026-11-15
paper_title: "Prediction or Persistence? What Future-Prediction Scores Miss About World Models"
paper_authors: "Animesh Maheshwari"
paper_year: 2026
paper_venue: "CoRL 2026 Workshop \"Do Robots Need World Models?\""
paper_url: "https://arxiv.org/abs/2611.15082"
arxiv_id: "2611.15082"
method: "Persistence Control"
code_url: ""
---

# Persistence_X光

**Prediction or Persistence? What Future-Prediction Scores Miss About World Models**
*Animesh Maheshwari（Deccan AI） | CoRL 2026 Workshop "Do Robots Need World Models?"*

## 论文说了什么

**问题**：机器人学习里选世界模型——从哪个预训练、拿哪个做规划、用哪个当评测器——主要看「未来预测分数」。但这个数字是三件事的和：编码器对场景的表示能力、**近未来与刚过去有多像**、预测器真正的贡献。第三项才是「预测」，前两项在什么都不预测时照样贡献分数。于是「一个根本不预测的模型也能高分」，「一个有害的预测器也能高于随机」。

**视角**：不需要新模型，只需要一个**配对对照**。把同一个模型的预测器换成恒等映射 $g^{\text{pers}}_\theta(z,W)=z$（「哪个候选最像刚刚看到的」），编码器、预处理、能量函数、试验集全部不变，两份模型的 top-1 之差就是预测器的净贡献 $\delta_\theta$。一次额外前向的代价，把「分数」变成「可归因的贡献」。

**结果**：在 27,000 次检索试验（704 个视频、25 个格子、9 个模型变体同一批试验）上，分数与贡献**分离**：V-JEPA 2.1 ViT-g 的预测器贡献 +0.291，同族的 V-JEPA 2 ViT-H / ViT-L 只有 +0.041 / +0.028——分数都在 0.42 以上，贡献差 7–10 倍。DINO-WM 得分 0.136（高于随机 0.091），但对照 0.315，贡献 **−0.179**：去掉预测器反而更好。

## 方法拆解

1. **检索探针**：一条试验 $\tau=(V,t,A,W)$——看 $A$ 秒锚点，从同一视频的 11 个 $W$ 秒窗口里挑真实后续（chance $=1/11=0.0909$）。干扰项从锚点起点与真值终点向外铺，不与锚点重叠，且每个模型用**自己训练时的能量函数**排序（能量只在它定义的嵌入空间里有意义）。
2. **persistence 对照**：$\delta_\theta=\frac{1}{|T|}\sum_\tau[\mathbb{1}\{\hat\jmath_\tau=j^\star_\tau\}-\mathbb{1}\{\hat\jmath^{\text{pers}}_\tau=j^\star_\tau\}]$。编码器/能量/试验固定，差值只能归给预测器；$\delta_\theta<0$ 意味着「不如不预测」。
3. **数据与包络**：1,094 段家用第一视角视频（479 小时，无动作流）；扫描 $A\in\{1..10\}$ s、$W\in\{1,1.5,..,A\}$ s 共 100 格，把输入预算与时域分开；冻结预测器只能在其位置预算内查询（V-JEPA 2.1 ViT-g 上界 8 s）。跨模型数字取 27,000 条试验的固定交集。
4. **统计**：所有区间为 704 个视频上的 95% cluster bootstrap（B=10,000）。

## 结果

V-JEPA 2.1 ViT-g 的 0.736 被完整拆开（Table 1）：

| 层级 | top-1 | 增量归因 |
|---|---|---|
| 随机 (1/11) | 0.091 | — |
| persistence，V-JEPA 2 ViT-L 编码器 | 0.396 | +0.305 视频连续性 + 表示 |
| persistence，V-JEPA 2.1 ViT-g 编码器 | 0.444 | +0.048 更好的编码器（仍是零预测） |
| V-JEPA 2.1 ViT-g 全模型 | 0.736 | +0.291 预测器 |
| 完美检索 | 1.000 | +0.264 未解决 |

Table 2（同一批 27,000 试验）：V-JEPA 2 ViT-H 0.437（$\delta=+0.0408\,[+0.0326,+0.0491]$）、ViT-L 0.424（$\delta=+0.0284$）、共享对照 0.396；DINO-WM 0.136 vs 0.315（$\delta=-0.179$）；LeWM 0.201 vs 0.208（$\delta=-0.0066$）。按 $W$ 拆分更刺眼：$W=1$ s 时 persistence-L 拿到 0.500，**高于两个 V-JEPA 2 全模型**（0.448 / 0.452）——短时域上「不预测」直接赢。

## 锐评

- **对照一：[[09_OnlyAct_X光|OnlyAct]] 这类「拿 V-JEPA 2 特征抽动作」的工作**。它们的前提是预训练嵌入里有可用的运动信息，本文恰好量化了这个前提：编码器单独就有 0.396（vs 随机 0.091），所以「特征好用」是真的；但同时说明**下游涨点不能记在预测器账上**——除非你能报出 $\delta_\theta$。这是对整条 JEPA-for-control 路线最便宜的一次体检。
- **对照二：DINO-WM / LeWM 这类动作条件模型**。它们的分数在本文探针下低于自己的 persistence 对照，但作者明确标注是域外测量（PushT 动作空间 vs 被动家用视频），只报方向、不报幅度。这种「知道自己测的是什么、把不可解释的部分标成 illustrative」的写法，比结论本身更值得抄。
- **对照三：常规 world-model 基准（PSNR/FVD/下游成功率）**。那些指标把编码器、连续性与预测混在一起，无法回答「预测器有没有用」；本文的对照只需要一次前向，成本低到可以写进任何评测脚本。反过来说，它也没解决「下游有没有用」——作者自己把这一步留给 planning/control 评测。
- **真实局限**（作者自陈）：11 选 1 的检索只考**时间定位**，比生成容易得多，且不检验预测内容；语料是单一家庭域；V-JEPA 2.1 只有一个 checkpoint 可评，没有 scaling 曲线；被动视频不含动作，对动作条件模型先天不公。

## 借鉴意义

- **「把模块替换成恒等」做归因**：任何声称某模块带来增益的工作，都可以照这个模板做一次零成本消融——不需要重训，只需换一个 forward。
- **报分数时同时报 persistence**：世界模型的分数只有和「复制当前帧」的基线一起看才有意义；这条对视频预测、latent planning、甚至 VLA 的 future-token 辅助损失都适用。
- **拆解表格的写法**：chance → persistence → +encoder → +predictor → perfect，四级分解让「还剩多少没解决」（0.264）和「哪一步贡献多少」同时可见。

## 关联

- [[2026-11-15-论文推荐]]：当日日报条目。
- [[12_A2WM_X光|A²WM]]：那篇默认「预训练策略给出可用标称行为」，这篇提供了检验「预训练表示到底贡献了什么」的工具。
- [[09_OnlyAct_X光|OnlyAct]]：同吃 V-JEPA 2 特征，一个做动作表示抽取，一个做贡献归因；合起来看才知道涨点该记在谁头上。
- [[11_K-PWM_X光|K-PWM]] / [[10_WM-PPO_X光|WM-PPO]]：两篇都靠世界模型的预测质量论证收益，按本文标准应补一条 persistence 对照。
