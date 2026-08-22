---
tags: [pytorch, cuda, fundamentals]
---
# PyTorch GPU Execution

## 核心 Mental Model

PyTorch 的很多 CUDA 调用是异步的。Python 端返回，并不代表 GPU 已经执行结束。

因此这种测时可能错误：

```python
start = time.time()
y = model(x)
end = time.time()
```

正确思路：

- `torch.cuda.synchronize()`
- 或 CUDA Events

## 必须理解

- CPU launch GPU kernel
- CUDA stream
- synchronization point
- tensor device movement
- allocation / caching allocator
- warmup

## Benchmark 基本纪律

- warmup
- fixed dtype
- fixed shapes
- synchronize
- repeat
- report median / percentile
- record GPU model and software versions

关联：[[13_AI_Infra/11_Performance/01_Benchmark-and-Profiling|Benchmark & Profiling]]
