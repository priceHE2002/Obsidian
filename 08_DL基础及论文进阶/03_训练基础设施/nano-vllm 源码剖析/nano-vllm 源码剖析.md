---
tags:
  - 训练基础设施
  - LLM推理
  - vLLM
  - 源码剖析
created: 2026-07-07
---

# nano-vllm 源码剖析

> nano-vllm 是一个用 **~1200 行 Python** 从零实现的轻量级 vLLM 推理引擎。它完整覆盖了 vLLM 的核心技术栈：PagedAttention、连续批处理（Continuous Batching）、前缀缓存（Prefix Caching）、张量并行（Tensor Parallelism）、CUDA Graph、Triton Kernel 等。整个项目仅 16 个源文件，是学习 LLM 推理系统的绝佳教材。

## 项目概览

```
nanovllm/                    ← 核心包（~1200 行）
├── __init__.py              ← 导出 LLM, SamplingParams
├── config.py                ← 全局配置（700行）
├── llm.py                   ← 主入口 LLM 类（80行）
├── sampling_params.py       ← 采样参数（270行）
├── engine/                  ← 推理引擎核心
│   ├── llm_engine.py        ← LLMEngine：主循环 + generate() 方法
│   ├── model_runner.py      ← ModelRunner：GPU 推理 + CUDA Graph + TP
│   ├── scheduler.py         ← Scheduler：调度器 + 连续批处理
│   ├── sequence.py          ← Sequence：序列状态管理
│   └── block_manager.py     ← BlockManager：PagedAttention KV-cache 管理
├── models/
│   └── qwen3.py             ← Qwen3 完整模型实现（~220行）
├── layers/                  ← 模型底层组件
│   ├── attention.py         ← 注意力：Triton store kernel + FlashAttention
│   ├── linear.py            ← 线性层：Column/Row/QKV/Merged Parallel
│   ├── embed_head.py        ← Vocab Parallel Embedding + LM Head
│   ├── layernorm.py         ← RMSNorm（带 fused residual add）
│   ├── rotary_embedding.py  ← RoPE 旋转位置编码
│   ├── activation.py        ← SiLU Gate 激活
│   └── sampler.py           ← 采样器（temperature sampling）
└── utils/
    ├── context.py            ← 全局上下文（Prefill/Decode 参数传递）
    └── loader.py             ← HuggingFace 权重加载器
```

## 核心设计理念

1. **PagedAttention**：KV-Cache 按块管理（block_size=256 tokens），支持前缀共享
2. **Continuous Batching**：Prefill 和 Decode 可以在同一批中处理
3. **全局 Context**：通过 `set_context` / `get_context` 在模型各层之间传递 Prefill/Decode 参数，避免修改每层 forward 签名
4. **Tensor Parallel**：通过 `torch.distributed` + SharedMemory 实现多 GPU 张量并行
5. **CUDA Graph**：对 Decode 阶段进行 CUDA Graph 捕获，减少 CPU-GPU 同步开销
6. **Torch Compile**：对关键小算子（RMSNorm、激活、采样）用 `@torch.compile` 加速

## 目录

- [[00_总览与架构/总览与架构]]
- [[01_入口与配置/入口与配置]]
- [[02_调度引擎/调度引擎总览]]
- [[03_模型执行引擎/模型执行引擎总览]]
- [[04_模型层实现/模型层实现总览]]
- [[05_工具与辅助/工具与辅助]]
- [[06_性能优化专题/性能优化专题总览]]
- [[07_端到端流程/端到端流程]]
