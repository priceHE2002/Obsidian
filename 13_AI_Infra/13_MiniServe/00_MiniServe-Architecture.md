---
tags: [miniserve, architecture, ai-infra]
---
# MiniServe Architecture

## 一句话定义

MiniServe 是一个**架构完整但规模受控**的单 GPU LLM inference runtime，用来系统性研究：

$$
\text{Scheduling}+\text{Memory Management}+\text{GPU Kernel Optimization}
$$

## 四层架构

```text
┌─────────────────────────────────┐
│ Serving Layer                   │
│ API / Request / Streaming       │
├─────────────────────────────────┤
│ Control Layer                   │
│ Scheduler / Continuous Batching │
├─────────────────────────────────┤
│ Memory Layer                    │
│ Paged KV / Prefix Cache         │
├─────────────────────────────────┤
│ Execution Layer                 │
│ PyTorch / Triton / CUDA / GPU   │
└─────────────────────────────────┘
          ↑ Benchmark / Metrics ↑
```

## Control Plane

负责 **Policy**：

- Request state
- Scheduler
- Admission control
- Token budget
- Block allocation
- Prefix reuse

## Execution Plane

负责 **Execution**：

- Tensor preparation
- Model forward
- KV read/write
- Triton/CUDA kernels
- Logits
- Sampling

## 核心对象

### Request

```text
WAITING → PREFILL → DECODING → FINISHED
```

建议字段：

```text
request_id
prompt_tokens
output_tokens
max_new_tokens
state
block_table
arrival_time
first_token_time
finish_time
```

### SchedulerBatch

Scheduler 不直接调用模型，而是输出一份 execution plan：

```text
requests
input_tokens
positions
block_tables
slot_mapping
num_prefill_tokens
num_decode_tokens
```

### BlockManager

负责 physical KV block 生命周期。

### ModelRunner

只负责 GPU execution，不决定调度策略。

## 一次请求的执行路径

```text
HTTP Request
   ↓
Tokenizer
   ↓
Request(WAITING)
   ↓
Scheduler admission
   ↓
Allocate KV blocks
   ↓
PREFILL
   ↓
First token → TTFT
   ↓
DECODING
   ↓
Continuous batching each iteration
   ↓
EOS / max_new_tokens
   ↓
FINISHED
   ↓
Free KV blocks
```

## 三个核心 Feature 如何协同

### Continuous Batching

请求不断进入/退出，要求显存能动态 allocate/free。

### Paged KV

用 block-based allocation 让上述动态生命周期可管理。

### Triton Kernels

最终根据 scheduler 产生的 batch metadata 与 block layout 做高效 GPU execution。

因此项目真正的系统关系是：

```text
Scheduler → Memory Layout → Kernel
```

## V1 约束

只支持：

- single GPU
- one model family
- FP16/BF16
- basic sampling

暂时不追求：

- TP/PP
- multi-node
- quantization matrix
- speculative decoding
- production-grade compatibility

关联：[[13_AI_Infra/13_MiniServe/01_MiniServe-V1-V4|V1–V4 Plan]]
