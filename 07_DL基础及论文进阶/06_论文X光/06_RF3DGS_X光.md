---
tags:
  - 论文
  - X光
  - 3DGS
  - 逆渲染
created: 2026-11-11
paper_title: "RF3DGS: Reflection-Free 3D Gaussian Splatting with Self-Priors"
paper_authors: "Qiang Wen, Guotao Meng, Junhong Ruan, Hongyu Liu, Chenyang Lei, Qifeng Chen"
paper_year: 2026
paper_venue: "arXiv 2611.13772"
paper_url: "https://arxiv.org/abs/2611.13772"
arxiv_id: "2611.13772"
method: "RF3DGS"
code_url: ""
---

# RF3DGS_X光

**RF3DGS: Reflection-Free 3D Gaussian Splatting with Self-Priors**
*Qiang Wen, Guotao Meng, Junhong Ruan, Hongyu Liu, Chenyang Lei, Qifeng Chen | arXiv 2611.13772*

## 论文说了什么

**问题**：隔着玻璃拍展品（博物馆、文物）时，反射会叠在透射画面上。2D 去反射方法逐帧处理，多视角之间不一致；直接把带反射的图喂给 3DGS，反射会被烘进高斯，重建出来的场景自带「鬼影」。

**视角**：只用一条常见的光度先验——同一个 3D 点在不同视角里，**最小像素强度**最接近无反射的透射信息（因为反射随视角变化，透射相对稳定）。难点是怎么把它落到 3DGS 的多视角优化里。

**结果**：摘要称在合成与真实场景都达到 SOTA，并建立了第一个基于 Unity 的 3D 去反射基准；具体指标 `[摘要未给]`。

## 方法拆解

1. **粗深度**：先用几何先验优化一个 vanilla 3DGS，得到每个视角粗略的无反射深度图。
2. **3D warping 造伪标签**：用这些深度把辅助视角的像素 warp 到参考视角，按光度先验（取 min）聚合出干净的透射伪标签。
3. **ATR 损失**：Adaptive Twin Rendering——同时渲染「伪标签监督的分支」和「原始退化图分支」，前者提供去反射信号，后者保住高频细节，避免过度平滑。
4. 整体是自监督，不需要成对的「有反射/无反射」数据。

## 锐评

- 与两条路线对比：一类是 2D 去反射网络（逐帧、无多视角约束），另一类是直接 3DGS 重建（反射烘进几何）。RF3DGS 的区别是把多视角一致性做成伪标签，属于「用几何换监督」。注意：摘要没点名具体基线，实际对比对象 `[摘要未给]`。
- 真实局限：光度先验依赖「反射随视角变化、透射稳定」，如果反射面与相机几乎共面（弱视差）或透射本身有视角相关效应（折射、镀膜），先验会失效；伪标签的质量完全取决于粗深度，深度错的地方会污染监督。摘要没有给失败案例的定量分析。

## 借鉴意义

- 对 3DGS 资产管线：玻璃展柜、橱窗、车窗这类「隔层拍摄」场景可以直接套这套自监督流程，不需要额外采集无反射参考。
- 对 world model：多视角一致性作为免费监督信号，可以迁移到「动态物体去除」「光照分离」等重建前的预处理。

## 关联

- [[2026-11-11-论文推荐]]：当日日报条目。
- [[2026-11-08-论文推荐]]：同属 3DGS 工程化（那天的 CrossPressor / BudgetSplat 管压缩，这篇管去反射）。
