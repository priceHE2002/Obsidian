---
tags: [interview, pitch, miniserve]
---
# MiniServe 60-Second Pitch

## 60 秒版本

MiniServe 是一个单 GPU LLM inference runtime。我把系统拆成 control plane 与 execution plane：control plane 由 request manager、scheduler 和 KV block manager 组成，负责 iteration-level continuous batching、admission control 和 paged KV memory allocation；execution plane 由 model runner 与 PyTorch/Triton kernels 组成，根据 scheduler 产生的 batch metadata、block table 和 slot mapping 执行 Prefill/Decode。KV Cache 使用固定大小 physical blocks，实现逻辑连续、物理离散的动态存储，并进一步支持 prefix sharing。整个系统用 TTFT、TPOT、throughput、tail latency 和 GPU memory utilization 进行端到端 benchmark，并与 HuggingFace / vLLM 做对照。

## 之后主动引导面试官进入三条线

1. Scheduling：Continuous Batching / Chunked Prefill
2. Memory：Paged KV / Prefix Cache / Allocator
3. GPU：Triton / Nsight / Memory Access

目标不是背这段，而是最终能自然地根据面试官兴趣展开。
