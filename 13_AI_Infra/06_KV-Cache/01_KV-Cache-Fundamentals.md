---
tags: [kv-cache, llm-inference]
---
# KV Cache Fundamentals

## 为什么需要 KV Cache？

自回归生成中，历史 token 的 K/V 不需要在每一步重新计算。保存后，每一步只为新 token 计算新的 K/V，再与历史 K/V 做 attention。

## 粗略显存估算

若：

- 层数 $L$
- KV heads $H$
- head dimension $D$
- sequence length $S$
- 每元素字节数 $B$

则单序列 KV Cache 近似：

$$
\text{KV Memory}\approx 2LHDSB
$$

其中 2 来自 K 与 V。

## 为什么在线 Serving 难？

模型权重相对固定，但不同请求的 KV Cache 会随着生成动态增长、结束并释放。在线服务因此需要真正的动态显存管理。

下一步：[[13_AI_Infra/06_KV-Cache/02_Paged-KV-Cache|Paged KV Cache]]
