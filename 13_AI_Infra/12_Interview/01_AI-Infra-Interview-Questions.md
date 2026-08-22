---
tags: [interview, ai-infra, miniserve]
---
# AI Infra Interview Question Bank

## LLM Inference

- [ ] Prefill 与 Decode 有什么本质差异？
- [ ] TTFT 与 TPOT 分别主要受哪些因素影响？
- [ ] 为什么需要 KV Cache？
- [ ] KV Cache 大小如何估算？

## Scheduling

- [ ] 为什么需要 Continuous Batching？
- [ ] Batch 越大越好吗？
- [ ] 如何做 admission control？
- [ ] 如何避免 starvation？
- [ ] 为什么需要 Chunked Prefill？

## Paged KV

- [ ] 为什么不用每请求连续大块 KV？
- [ ] Block size 过大/过小分别有什么问题？
- [ ] Block table lookup 有什么代价？
- [ ] Request 结束后如何安全回收？
- [ ] Prefix sharing 如何做 ref counting？

## GPU

- [ ] Warp 是什么？
- [ ] Memory coalescing 为什么重要？
- [ ] Occupancy 是什么？高 occupancy 一定更快吗？
- [ ] Kernel fusion 为什么可能加速，也为什么可能变慢？
- [ ] Decode 为什么常常更受 memory traffic 影响？

## Performance

- [ ] GPU timing 为什么要 synchronize？
- [ ] 如何设计公平 benchmark？
- [ ] MiniServe 比 vLLM 慢怎么办？
- [ ] 如何证明某项优化真的有效？

## 项目深挖

- [ ] MiniServe 最难的 bug 是什么？
- [ ] 最失败的一次优化是什么？
- [ ] 如果再给两周，你会优化哪里？
- [ ] 当前架构的最大限制是什么？
