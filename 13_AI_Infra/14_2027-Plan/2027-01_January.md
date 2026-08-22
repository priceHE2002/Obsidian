---
tags: [2027, january, miniserve]
status: planned
---
# 2027-01 — Baseline Runtime

## 月目标

真正理解 LLM inference execution path，并建立可复现 baseline。

## Week 1 — LLM Inference

- [ ] Transformer forward mental model
- [ ] Attention / QKV
- [ ] KV Cache
- [ ] Prefill
- [ ] Decode
- [ ] Sampling
- [ ] Tensor shapes
- [ ] 完成 [[13_AI_Infra/05_LLM-Inference/01_LLM-Inference-Mental-Model|LLM Inference Mental Model]]

## Week 2 — PyTorch + GPU

- [ ] CUDA async execution
- [ ] synchronization
- [ ] CUDA Event timing
- [ ] VRAM / bandwidth / FLOPS 基础
- [ ] 正确 benchmark GPU latency

## Week 3 — Naive Runtime

- [ ] 建 repo
- [ ] load model
- [ ] tokenize
- [ ] prefill
- [ ] decode loop
- [ ] sampling

## Week 4 — Benchmark Harness

- [ ] TTFT
- [ ] TPOT
- [ ] throughput
- [ ] P50/P95
- [ ] peak GPU memory
- [ ] JSON/CSV results

## 1 月验收

- [ ] 能独立解释 Prefill / Decode
- [ ] 能写 autoregressive decode loop
- [ ] baseline 可复现
- [ ] Benchmark Harness 可复用
