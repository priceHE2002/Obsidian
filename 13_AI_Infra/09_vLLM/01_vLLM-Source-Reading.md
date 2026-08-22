---
tags: [vllm, source-reading]
---
# vLLM Source Reading

## 原则

不要从 repo 第一行读到最后一行。每次只围绕自己已经实现的问题去读成熟系统。

## 推荐顺序

### Phase 1：Request / Sequence

回答：成熟系统如何表示一个请求的状态？

### Phase 2：Scheduler

回答：每个 iteration 如何决定执行哪些 request / token？

### Phase 3：KV Cache / Block Manager

回答：逻辑 sequence 与 physical KV blocks 如何关联？

### Phase 4：Model Runner

回答：Scheduler metadata 怎样转换成 GPU inputs？

## 每次源码阅读都写三列

| MiniServe | vLLM | 为什么不同 |
|---|---|---|
|  |  |  |

## 禁止行为

不要为了“看懂 vLLM”而看 vLLM。先自己遇到问题，再去看成熟实现如何解决。

## Vault 内已有资料

- [[08_DL基础及论文进阶/03_训练基础设施/nano-vllm 源码剖析/nano-vllm 源码剖析|nano-vllm 源码剖析]] — 1200 行的迷你 vLLM，先读它再读 vLLM 本体，性价比最高
