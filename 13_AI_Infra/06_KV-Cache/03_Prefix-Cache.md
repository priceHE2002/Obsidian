---
tags: [prefix-cache, kv-cache]
---
# Prefix Cache

## 场景

多个请求共享系统提示词或长前缀时，对相同 prefix 重复做 Prefill 和保存 KV 是浪费。

## Block-level Sharing

以 block 为共享单位：

```text
Prefix Hash → Cached Block IDs → request block_table
```

## 需要的元数据

```cpp
struct KVBlock {
    int block_id;
    int ref_count;
};
```

当 `ref_count > 0` 时 block 不能被回收。

## 基础流程

1. 对完整 block 的 token prefix 做 hash
2. 查询 cache
3. Hit：复用 physical block
4. Miss：正常 Prefill 并登记
5. Request 结束时递减引用计数

## 实验

Prefix hit ratio：0% / 25% / 50% / 75% / 100%

记录：

- TTFT
- compute saved
- VRAM
- cache metadata overhead

## 面试扩展

- Hash collision 怎么处理？
- Partial block 怎么办？
- Cache eviction policy？
- 多租户场景的安全边界？
