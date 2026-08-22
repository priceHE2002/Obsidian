---
tags: [sglang, source-reading]
---
# SGLang Source Reading

## 阅读时机

建议 2027 年 5 月以后，在 MiniServe 已有：

- scheduler
- paged KV
- prefix cache
- benchmark

之后再读。

## 重点问题

- 它如何组织请求调度？
- Prefix / Radix 类复用思想如何落地？
- Prefill 与 Decode 如何协调？
- Runtime 与 frontend 如何解耦？

## 最有价值的输出

不要写源码摘要。写：

> MiniServe 当前实现与 SGLang 在同一问题上的设计差异，以及如果继续演进我会怎么改。

## Vault 内已有资料

- [[08_DL基础及论文进阶/00_基础知识/05_SGLang推理框架/SGLang推理框架|SGLang推理框架]] — 已有笔记，含 RadixAttention 与 vLLM 对比
