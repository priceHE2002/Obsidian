---
tags: [triton, gpu-kernel]
---
# Triton Learning Path

## 目标

不是“会写 Triton 语法”，而是用 Triton 验证 GPU 性能推理：

> profile → hypothesize → optimize → benchmark

## 推荐顺序

- [ ] Vector Add
- [ ] Fused Softmax
- [ ] RMSNorm
- [ ] RoPE
- [ ] KV Cache Store

## 每个 Kernel 必须有三份东西

1. Correctness test
2. Benchmark against PyTorch baseline
3. Profiling note

## 记录模板

| Item | Result |
|---|---|
| Shape |  |
| Dtype |  |
| PyTorch latency |  |
| Triton latency |  |
| Speedup |  |
| Bottleneck |  |
| Why faster/slower |  |

## 设计原则

即使 Triton 比 PyTorch 慢，也不要“删掉失败结果”。面试中，能解释慢在哪里，往往比只展示一个漂亮 speedup 更有价值。
