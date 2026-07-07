---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - 前缀缓存
created: 2026-07-07
up: "[[性能优化专题总览|性能优化专题总览]]"
---

# Prefix Caching 详解

前缀缓存（Prefix Caching / Automatic Prefix Caching）是 LLM 推理最重要的优化之一。它的核心思想是：当多个请求共享相同的前缀（如相同的 system prompt），这些前缀的 KV-Cache 只需计算一次，后续请求直接复用。

## 核心机制回顾

nano-vllm 的前缀缓存实现融合在 `BlockManager` 中，核心流程为：

```
1. can_allocate(): 遍历 Block，计算链式哈希，匹配已有缓存
2. allocate(): 复用命中的 Block（ref_count++），分配新 Block
3. hash_blocks(): Prefill 完成后给 Block 写入哈希
4. deallocate(): ref_count--，ref_count==0 时真正回收
```

完整代码分析见 [[../02_调度引擎/02.3_块管理与前缀缓存/块管理与前缀缓存]]。

## 链式哈希的必要性

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

考虑两个请求：
- A: "Translate to French: Hello, how are you?"
- B: "Translate to German: Hello, how are you?"

如果进行简单哈希（只用 Block 内的 token_ids），Block 2 的 "Hello, how are you?" 哈希相同，可能错误匹配。链式哈希将前一个 Block 的哈希作为前缀纳入计算，确保了 B 的 Block 2 的哈希与 A 的 Block 2 不同（因为 Block 1 的哈希不同）。

## 缓存生命周期

```
[序列到达] → can_allocate(seq)
    ├─ 逐 Block 计算哈希 → 查找 hash_to_block_id
    ├─ 双重验证（哈希 + token_ids 对比）
    └─ 返回 num_cached_blocks

[Prefill 中] → allocate(seq, num_cached_blocks)
    ├─ 缓存 Block：ref_count++ 或 ref_count=1
    └─ 新 Block：_allocate_block()

[Prefill 后] → hash_blocks(seq)
    └─ 新写满的 Block → compute_hash → 写入 hash_to_block_id

[序列完成] → deallocate(seq)
    └─ 每个 Block ref_count--
        └─ ref_count==0 → _deallocate_block → 回收
```

## 为什么缓存粒度是 Block？

nano-vllm 选择以 Block（256 tokens）为单位进行缓存匹配，而非单个 token 或整个序列：

**优于 token 级缓存**：token 级缓存的匹配开销太大（每个 token 都要查哈希表），且 KV-Cache 存储本身就是 block 粒度的
**优于序列级缓存**：序列级缓存只在完全匹配时有用，部分前缀匹配（如不同的 user message 共享相同的 system prompt）会失效

Block 级缓存在精细度和效率之间取得了平衡。256 token 的 block 大小意味着：
- 每个 Block 的哈希只需计算和查找一次
- 即使只共享部分 Block，也能获得缓存收益
- 内存对齐友好（256 整除、与 attention kernel 对齐）

## 验证机制

```python
block_id = self.hash_to_block_id.get(h, -1)
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    break
```

双层验证：哈希查找 + token_ids 逐元素比较。xxhash 的碰撞概率极低（~2^-64），但 token_ids 比较是 O(block_size) 的安全网。一旦某层验证失败，`break` 停止继续匹配——因为链式哈希的特性，后续 Block 必然也不会匹配。

## 缓存失效

nano-vllm **不会主动 evict 缓存**。Block 的 `hash_to_block_id` 记录一直保留，直到 Block 被 `deallocate` 且 `ref_count==0` 时才移除。这意味着：

- 即使所有请求都已完成，KV-Cache 的哈希记录仍然保留
- 新请求到达时可以匹配到之前的缓存
- 当显存不足时，通过抢占机制（`preempt` → `deallocate`）释放 Block，自然导致缓存条目被清理

这是一个"自然淘汰"策略——缓存随 Block 分配/释放自然产生和消亡，不需要独立的 LRU 或 TTL 策略。对于推理引擎而言，这比显式的缓存淘汰更简单且更有效，因为 KV-Cache 的大小本身就是由显存限制自然管理的。

## 实际收益

前缀缓存在以下场景收益最大：
1. **多个请求共享 system prompt**：system prompt 的 KV-Cache 只计算一次
2. **Few-shot prompting**：示例部分的 KV-Cache 在多个推理之间共享
3. **Beam search / parallel sampling**：同一 prompt 的多个采样分支共享 prompt 部分的 KV-Cache
4. **Chunked Prefill 中断恢复**：被中断的 Prefill 不丢失已计算的 KV-Cache

在基准测试中（256 个随机请求、无共享前缀），前缀缓存几乎无收益。但在实际应用中（如 chatbot 的 system prompt 共享），它可以节省 30-50% 的 Prefill 计算。
