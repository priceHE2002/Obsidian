---
tags: [miniserve, benchmark]
---
# MiniServe Benchmark Plan

## Baselines

- HuggingFace naive generation
- MiniServe variants
- vLLM

## Main Metrics

| Metric | Meaning |
|---|---|
| TTFT | arrival → first token |
| TPOT | time per output token |
| Throughput | tokens/s |
| P50/P95 | tail behavior |
| Peak VRAM | memory efficiency |
| GPU Utilization | execution efficiency |

## Ablation Matrix

| Variant | KV | CB | Paged KV | Prefix | Triton |
|---|---:|---:|---:|---:|---:|
| V0 |  |  |  |  |  |
| V1 | ✓ |  |  |  |  |
| V2 | ✓ | ✓ |  |  |  |
| V3 | ✓ | ✓ | ✓ |  |  |
| V4 | ✓ | ✓ | ✓ | ✓ |  |
| V5 | ✓ | ✓ | ✓ | ✓ | ✓ |

## Scaling

Concurrency：1 / 2 / 4 / 8 / 16 / 32

Prompt：128 / 512 / 1024 / 2048

Output：64 / 128 / 256 / 512

## Paged KV Specific

- block size sensitivity
- fragmentation
- allocator overhead
- OOM behavior under concurrency

## Prefix Cache Specific

Hit ratio：0 / 25 / 50 / 75 / 100%

## 最终必须回答

1. 哪个 feature 贡献了什么？
2. 哪个 workload 下效果最明显？
3. 哪个 workload 下效果变差？
4. MiniServe 与 vLLM 的 gap 在哪里？
