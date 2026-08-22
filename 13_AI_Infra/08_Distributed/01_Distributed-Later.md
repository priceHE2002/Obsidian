---
tags: [distributed, later]
status: deferred
---
# Distributed Systems — Later

## 为什么暂时后置？

单卡阶段先建立完整的 inference mental model。否则容易把 DDP、TP、PP、NCCL、RDMA 学成孤立名词。

## 等 MiniServe 单卡完成后再补

- [ ] Data Parallel
- [ ] Tensor Parallel
- [ ] Pipeline Parallel
- [ ] NCCL Collectives
- [ ] AllReduce / AllGather / ReduceScatter
- [ ] Communication-Compute Overlap
- [ ] RDMA / RoCE / InfiniBand

## 最小补强实验

短租 2×GPU，做：

1. DDP baseline
2. gradient bucket
3. async all-reduce
4. compute/communication overlap

目标是形成一个额外面试故事，而不是重写完整 distributed runtime。
