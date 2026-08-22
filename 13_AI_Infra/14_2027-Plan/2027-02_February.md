---
tags: [2027, february, scheduler, kv-cache]
status: planned
---
# 2027-02 — KV Cache + Scheduler

## Week 5 — KV Cache

- [ ] 实现 KV Cache abstraction
- [ ] append / get / reset
- [ ] no-cache vs cache benchmark
- [ ] 记录 decode latency 与 VRAM

## Week 6 — Multi-request Engine

- [ ] `Request`
- [ ] `RequestManager`
- [ ] WAITING/PREFILL/DECODING/FINISHED
- [ ] async request queue

## Week 7 — Continuous Batching

- [ ] 每轮重新构建 running set
- [ ] 完成请求立即退出
- [ ] waiting request 立即补入
- [ ] concurrency benchmark

## Week 8 — Scheduler

- [ ] FCFS baseline
- [ ] `max_num_seqs`
- [ ] `max_num_batched_tokens`
- [ ] admission control
- [ ] P50/P95 latency

## 2 月验收

- [ ] KV Cache
- [ ] Request abstraction
- [ ] Scheduler
- [ ] Continuous Batching
- [ ] 多并发 benchmark
