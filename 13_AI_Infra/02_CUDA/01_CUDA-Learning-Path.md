---
tags: [cuda, roadmap]
---
# CUDA Learning Path

## P0：够 MiniServe 用的 CUDA

- [ ] Host / Device
- [ ] Kernel launch
- [ ] Grid / Block / Thread
- [ ] Warp
- [ ] Global / Shared / Register memory
- [ ] Synchronization
- [ ] CUDA Event timing
- [ ] Async execution
- [ ] Coalesced memory access

## P1：进入性能工程

- [ ] Occupancy
- [ ] Register pressure
- [ ] Shared-memory bank conflict
- [ ] Vectorized load/store
- [ ] CUDA Streams
- [ ] Pinned memory
- [ ] Kernel fusion

## P2：有时间再做

- [ ] CUDA Graph
- [ ] CUTLASS / CuTe
- [ ] Tensor Core
- [ ] CUDA C++ extension for PyTorch

## MiniServe 建议落地任务

先不要手写完整 Attention。优先：

1. RMSNorm
2. RoPE
3. KV Cache Store

其中 KV Cache Store 与 [[13_AI_Infra/06_KV-Cache/02_Paged-KV-Cache|Paged KV]] 的架构关联最强。
