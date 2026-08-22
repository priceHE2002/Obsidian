---
tags: [paged-kv, memory-management, miniserve]
---
# Paged KV Cache

## 核心问题

如果每个请求按 `max_seq_len` 提前预留连续 KV 空间，会导致显存预留浪费，并使动态进入/退出的请求难以灵活管理。

## 核心设计

把 KV Cache 划分成固定大小的 physical blocks；每个 sequence 维护自己的 logical→physical `block_table`。

```text
Logical blocks:   [0] [1] [2]
                    |   |   |
Block table:        7   2   9
                    |   |   |
Physical pool: ... [7] ... [2] ... [9] ...
```

## 地址转换

若 block size 为 $B$，token 位置 $t$：

$$
\text{logical\_block}=\left\lfloor\frac{t}{B}\right\rfloor
$$

$$
\text{offset}=t\bmod B
$$

物理 block 由：

```text
physical_block = block_table[logical_block]
```

得到。

## 建议的数据结构

```cpp
struct Sequence {
    std::vector<int> block_table;
    int num_tokens;
};
```

```cpp
class BlockAllocator {
public:
    int allocate();
    void free(int block_id);
};
```

## 必须做的实验

- Contiguous KV vs Paged KV
- 不同 block size
- 不同 concurrency
- 不同 sequence length
- VRAM utilization
- internal fragmentation

## 面试必须讲 trade-off

收益：

- 减少预留浪费
- 动态扩展
- 便于 block 级共享

代价：

- block table lookup
- metadata 管理
- kernel 访存更复杂
- block size 选择存在 trade-off

关联：[[13_AI_Infra/06_KV-Cache/03_Prefix-Cache|Prefix Cache]]、[[13_AI_Infra/07_Scheduling/01_Continuous-Batching|Continuous Batching]]
