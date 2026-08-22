---
tags: [miniserve, roadmap]
---
# MiniServe V1–V4 Plan

## V1 — Naive Runtime

目标：先把 execution path 跑通。

- [ ] load model
- [ ] tokenizer
- [ ] prefill
- [ ] autoregressive decode loop
- [ ] sampling
- [ ] baseline benchmark

## V2 — Serving Runtime

目标：从单请求脚本升级成在线 serving runtime。

- [ ] Request abstraction
- [ ] Request Manager
- [ ] WAITING/PREFILL/DECODING/FINISHED
- [ ] Scheduler
- [ ] Continuous Batching
- [ ] Streaming

## V3 — Memory-aware Runtime

目标：显存管理真正独立出来。

- [ ] KV Block Pool
- [ ] Block Allocator
- [ ] per-request block table
- [ ] Paged KV Cache
- [ ] Prefix Cache
- [ ] ref_count

## V4 — GPU-optimized Runtime

目标：profile 真实热点并替换 kernel。

- [ ] Triton RMSNorm
- [ ] Triton RoPE or KV Cache Store
- [ ] Nsight Systems
- [ ] Nsight Compute
- [ ] end-to-end ablation

## 完成定义

### Resume-ready

V1 + V2 + Paged KV + benchmark。

### Interview-ready

再加：Prefix Cache + Triton + Nsight + 与 vLLM 对比 + 清晰 limitation。
