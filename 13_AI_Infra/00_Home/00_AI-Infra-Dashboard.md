---
tags: [ai-infra, dashboard, miniserve]
status: active
start: 2027-01-01
---

# AI Infra Notebook Dashboard

> [!abstract] 核心目标
> 2027 年上半年形成一套可以支撑 AI Infra 实习面试的完整能力链：
> **LLM Inference → Scheduling → KV Memory → GPU Kernels → Profiling → Benchmark → Open Source → Interview**。

## 旗舰项目

[[13_AI_Infra/13_MiniServe/00_MiniServe-Architecture|MiniServe: A GPU-Efficient LLM Inference Runtime]]

核心能力：

- [[13_AI_Infra/07_Scheduling/01_Continuous-Batching|Continuous Batching]]
- [[13_AI_Infra/06_KV-Cache/02_Paged-KV-Cache|Paged KV Cache]]
- [[13_AI_Infra/06_KV-Cache/03_Prefix-Cache|Prefix Cache]]
- [[13_AI_Infra/03_Triton/01_Triton-Learning-Path|Triton Kernels]]
- [[13_AI_Infra/11_Performance/01_Benchmark-and-Profiling|Benchmark & Profiling]]

## 2027 上半年路线

- [[13_AI_Infra/14_2027-Plan/2027-H1-Master-Roadmap|2027 H1 Master Roadmap]]
- [[13_AI_Infra/14_2027-Plan/2027-01_January|2027-01：Baseline Runtime]]
- [[13_AI_Infra/14_2027-Plan/2027-02_February|2027-02：KV Cache + Scheduler]]
- [[13_AI_Infra/14_2027-Plan/2027-03_March|2027-03：Paged KV + Prefix Cache]]
- [[13_AI_Infra/14_2027-Plan/2027-04_April|2027-04：Triton + GPU Profiling]]
- [[13_AI_Infra/14_2027-Plan/2027-05_May|2027-05：Benchmark + Engineering]]
- [[13_AI_Infra/14_2027-Plan/2027-06_June|2027-06：Open Source + Interview]]

## 三个硬 Deadline

| Deadline | 验收标准 |
|---|---|
| 2027-01-31 | Baseline Runtime Done |
| 2027-03-31 | Resume-ready MiniServe |
| 2027-05-31 | Interview-ready MiniServe |

## 每周固定动作

- [ ] 编码 8h+
- [ ] Debug / Profile 4h+
- [ ] 系统或论文阅读 3h
- [ ] Benchmark 2h
- [ ] README / 笔记 2h
- [ ] 完成一份 [[13_AI_Infra/15_Templates/Weekly-Review|Weekly Review]]

## 学习任何 Infra 概念时，只问四个问题

1. 它解决什么问题？
2. 原来的瓶颈是什么？
3. 它怎么解决？
4. 它带来了什么新的 trade-off？

模板：[[13_AI_Infra/15_Templates/Concept-Note-Template|Concept Note Template]]
