---
title: SGLang推理框架
tags:
  - 基础知识
  - 深度学习
  - SGLang
  - LLM推理
  - RadixAttention
  - KVCache
  - 大模型部署
  - 推理优化
  - vLLM对比
  - AI基础设施
  - 机器学习系统
source: 小红书-SGLang高性能推理框架全拆解 (6a3a5a23000000000603533a)
created: 2026-07-06
---

# SGLang 高性能推理框架

> SGLang 是 UC Berkeley LMSYS 团队开源的 LLM Serving 框架，通过 RadixAttention（KV Cache 共享）、Compressed FSM（约束解码加速）和 Frontend DSL（gen/select/fork）三大核心技术，在共享前缀场景下实现 Prefill 计算量降低 96%。

## 一、概览：SGLang 是什么

![[01_SGLang_概览与三大核心.jpg]]

SGLang 是一个面向 LLM 推理的完整技术栈，包含两层设计：

- **Frontend DSL**：提供 `gen` / `select` / `fork` 三个编程原语，让多轮对话和分支推理自动享受底层 KV Cache 共享
- **Runtime 引擎**：以 RadixAttention 和 Cache-Aware Scheduling 为核心的推理运行时，自动管理 GPU 内存和调度策略

核心卖点就三件事：

1. **RadixAttention**：用 Radix Tree 管理所有请求的 KV Cache，自动发现并复用公共前缀——在线客服场景下 System Prompt 被重复 Prefill 的问题直接消失，实测 Prefill 计算量能降 96%（从需要 26 张 A100 降到 1 张）
2. **Compressed FSM**：把 JSON Schema 等约束编译成压缩有限状态机，确定性 token 全部跳过不走 GPU，结构化生成的额外开销从 30% 降到 5% 以内
3. **Frontend DSL**：提供 `gen` / `select` / `fork` 三个编程原语，多轮对话和分支推理自动享受底层 KV Cache 共享，不需要开发者手动拼接 HTTP 请求

## 二、三大核心卖点

![[02_三大核心卖点总览.jpg]]

### 卖点 1：Prefill 降 96%

在 1000 QPS 在线客服场景，所有请求共享同一个 System Prompt（4K token）。没有 KV Cache 共享时，1000 个请求各自 Prefill System Prompt，需要约 26 张 A100。SGLang 的 RadixAttention 自动发现公共前缀并复用 KV Cache，只需 1 张 A100。

> **96% 的 Prefill 节省意味着什么？** 从 26 张到 1 张 GPU，不只是一个数量级的成本降低，更关键的是让"共享前缀"场景从 GPU 成本不可行变得完全可行。实际部署中，少用 25 张 A100 ≈ 每月节省约 $30,000+ 云费用。

```
# 传统方式：每次请求都独立 Prefill
System Prompt (4K tokens)：每次 Prefill → 1000 次独立计算

# SGLang 方式：所有请求共享 System Prompt 的 KV Cache
System Prompt (4K tokens)：一次 Prefill → 1000 次直接复用
```

### 卖点 2：结构化解码从 30% 开销降到 5%

JSON/函数调用等结构化输出场景，传统 token-level masking 开销约 30%。SGLang 的 Compressed FSM 将约束编译成压缩有限状态机，确定性 token 跳过 GPU 计算，额外开销降至 5% 以内。

### 卖点 3：编程模型简化 LLM 调用

三个原语替代复杂 HTTP 编排：

| 原语 | 作用 | 场景 |
|------|------|------|
| `gen()` | 生成一段文本（支持 stop/regex/temperature） | 基本对话 |
| `select()` | 从候选项中选一个（计算 log P） | 分类、路由 |
| `fork(n)` | 复制 n 份状态并行展开 | Tree-of-Thought、Best-of-N |

## 三、Prefill vs Decode 的本质区别

![[03_Prefill与Decode本质区别.jpg]]

理解 LLM 推理的两阶段是理解所有优化技术的前提：

### Prefill 阶段

计算所有输入 token 的注意力——一次性计算 `Q × K^T`：

$$Q \times K^T = [n \times d] \times [d \times n]$$

- 输入：n 个 token 并行处理
- 计算特征：**Compute-bound**，GPU 算力满负载
- 产出：n 个 token 的 KV Cache + 1 个 token 的 logits

### Decode 阶段

逐个 token 生成，每次只计算当前 token 对历史的注意力：

$$q \times K_{cache}^T = [1 \times d] \times [d \times t]$$

- 输入：1 个 token
- 计算特征：**Memory-bound**，GPU 99% 时间在等待 HBM 访存
- 关键指标：算数强度（Arithmetic Intensity）≈ 1 FLOP/Byte

> 以 A100 为例，HBM 带宽 2TB/s，峰值算力 312 TFLOPS。Decode 阶段每次只需约 200 FLOP/Byte，远达不到算力上限。

**关键洞察：** Prefill 是 Compute-bound（算力密集），Decode 是 Memory-bound（显存带宽密集）。两种负载混在一个 batch 里会互相干扰——这就是 Chunked Prefill 的动机。

## 四、KV Cache 计算详解

![[04_KV_Cache计算详解.jpg]]

### KV Cache 大小公式

$$\text{KV\_Size} = 2 \times n\_\text{layers} \times n\_\text{kv\_heads} \times d\_\text{head} \times \text{seq\_len} \times \text{dtype\_size}$$

### 典型模型单 token KV 开销（FP16）

| 模型 | KV 大小 / token | 配置说明 |
|------|----------------|---------|
| LLaMA-7B | 0.5 MB | 32 layers × 32 heads（MHA） |
| LLaMA-70B | 0.3 MB | 80 layers × 8 KV heads（GQA） |
| Mixtral-8×7B | 0.1 MB | 32 layers × 8 KV heads（GQA） |

### 实际内存压力计算

一个 2×A100（80GB）部署 LLaMA-70B 场景，同时处理 100 个请求，平均 seq_len=4096：

$$100 \times 4096 \times 0.3\text{ MB} = 122.9\text{ GB}$$

KV Cache 占了 90%+ 的显存，而模型权重本身才约 140GB。

**内存压力本质：** 增大 batch size → KV Cache 占满显存 → 即使 GPU 算力空闲也无法处理更多请求。RadixAttention 通过共享 KV Cache 直接解决这个瓶颈。

## 五、Radix Tree：KV Cache 管理的基石

![[05_Radix_Tree基础与构建.jpg]]

### 为什么是 Radix Tree？

Radix Tree（基数树 / 压缩 Trie）是 Trie 的空间优化版本。对于 LLM token 序列：

- **公共前缀合并**：相邻节点如果只有一个子节点则合并（Path Compression）
- **空间最优**：只存储分支点，不存储线性链上的中间节点
- **匹配效率**：查找复杂度 O(匹配长度)，而非 O(树深度)

### Radix Tree vs 普通 Trie

```
普通 Trie（每个 token 一个节点）     Radix Tree（压缩路径）
[10] → [20] → [30] → [40] → [50]    [10,20,30,40,50]（单节点存储整条链）
```

对 LLM 推理的意义：System Prompt 通常包含 4K+ token 的连续前缀，Radix Tree 将其压缩为一个逻辑节点，大幅降低元数据存储开销。

### 三个核心操作

- **match_prefix**：给定 token 序列，找到最长公共前缀
- **insert**：插入新 KV 到对应节点
- **split**：必要时分裂节点

## 六、RadixAttention 内部结构

![[06_RadixAttention内部结构与KV_Block.jpg]]

### RadixTreeNode 数据结构

每个节点存储：

| 字段 | 含义 |
|------|------|
| `token_ids` | 该节点代表的 token 序列 |
| `kv_block_indices` | 指向 GPU HBM 中 KV Block 的索引 |
| `children` | 子节点引用 |
| `ref_count` | 引用计数（Copy-on-Write 核心） |
| `last_access_time` | LRU 驱逐依据 |

### KV Block 管理

KV Cache 以固定大小的 Block（如 16 token/block）为单位管理：

```
Block 大小 = 2 × n_layers × d_model × block_size × dtype_size
例如 LLaMA-7B 每个 Block = 2 × 32 × 4096 × 16 × 2B = 8MB
```

### 两层存储架构

- **CPU 内存**：存储完整 Radix Tree 结构 + 全部 KV 数据（经济）
- **GPU HBM**：只保留活跃请求的 KV Block（速度优先）

新请求到达时，通过 Radix Tree 定位共享前缀，将对应 KV Block 从 CPU 加载到 GPU——一次 Host-to-Device 拷贝，替代整个 Prefill 计算。

## 七、Radix Tree 的匹配与插入

![[07_Radix_Tree匹配与插入操作.jpg]]

### match_prefix 操作

```
输入：[10, 20, 30, 40, 50] token 序列
  ↓
从 root 出发，沿 Radix Tree 逐节点匹配
  ↓
输出：匹配长度（如 [10,20,30] 匹配，[40] 起为新分支）
复杂度：O(匹配深度)
```

### insert + split 操作

```
已有路径：[10,20,30,40,50] → Leaf_A
新序列： [10,20,30,60,70]

① match_prefix 找到公共前缀 [10,20,30]
② split Leaf_A，将 [40,50] 独立为新节点 Leaf_B
③ 为 [60,70] 创建 Leaf_C
④ ref_count 更新：Leaf_A ref+1, Leaf_C ref+1

结果：
  [10,20,30]  (ref=2, 两个子分支共享)
  ├── [40,50] → Leaf_B (ref=1)
  └── [60,70] → Leaf_C (ref=1)
```

### 引用计数与 KV Cache 回收

`ref_count` 机制是实现安全共享的关键：

- `ref_count++`：一个新请求匹配到该前缀
- `ref_count--`：一个请求完成，释放对前缀的引用
- `ref_count == 0`：该 Block 加入 Free List，可被新请求覆盖

> 这正是 Copy-on-Write 在 KV Cache 管理中的实现：同一个 KV Block 可以被多个请求同时读取，只有当某个分支需要修改时才真正复制。

## 八、Cache-Aware Scheduling

![[08_CacheAware_Scheduling与LPM调度.jpg]]

### 为什么需要 Cache-Aware 调度？

即使有了 Radix Tree 管理 KV Cache 共享，如果调度策略不好会出现"缓存刚存进去就被新请求挤出去"的恶性循环。

### LPM（Longest Prefix Match）调度

SGLang 的调度策略：**优先调度与现有缓存匹配最长的请求**。

```
Waiting Queue:
  Req X: 匹配 30 tokens  ✅ 优先调度
  Req Z: 匹配 50 tokens  ✅ 最先调度！
  Req V: 匹配 10 tokens  ❌ 最后调度
```

### 效果：命中率显著提升

| 场景 | FCFS 命中率 | LPM 命中率 |
|------|-----------|----------|
| 100% 共享前缀 | 87.5% | 94.6% |
| 独立短请求 | 100% | 100% |

LPM 调度 + Radix Tree 的组合，让分支推理（如 Best-of-16 采样）的 KV Cache 命中率达到 93.75%。

## 九、Static vs Continuous Batching

![[09_Static_vs_Continuous_Batching.jpg]]

### Static Batching 的问题

所有请求必须在同一时刻一起完成。短请求被长请求拖住（"padding 等待"），GPU 利用率仅约 29.2%。

### Continuous Batching 机制

每个 iteration 动态调整 batch 组成，新请求可以随时插入：

```
Iter 1: [A-Prefill][B-Decode][C-Decode]
Iter 2: [A-Prefill][B-Decode][C-Decode]
Iter 3: [D-Prefill][B-Decode][C-Decode]  ← D 随时加入
Iter 4: [D-Decode][E-Prefill][B-Decode]  ← C 完成退出，E 加入
Iter 5: [D-Decode][E-Prefill][B-Decode]
```

**三阶段流水线：** Prefill（Compute-bound）→ Decode（Memory-bound）→ KV Cache 回收。通过 Continuous Batching，三个阶段的资源交替使用，GPU 利用率大幅提升。

## 十、Chunked Prefill

![[10_Chunked_Prefill技术.jpg]]

### 核心问题

Prefill 是 Compute-bound，Decode 是 Memory-bound。长 Prefill 和 Decode 混在一个 iteration 中时：

- Prefill 占用 GPU 算力 → Decode 等待，TTFT（Time to First Token）尖峰
- 典型表现：Decode 的 TBT（Time Between Tokens）从 5ms 飙到 40ms

### Chunked Prefill 方案

将长 Prefill 切成 2048-4096 token 的 chunk，与 Decode 交替执行：

```
无 Chunked Prefill:
  [==== Prefill 4096 ====][Decode][Decode][Decode]
   Decode 等待 40ms

Chunked Prefill:
  [Chunk 512][Decode][Chunk 512][Decode][Chunk 512][Decode]...
   10ms       5ms     10ms      5ms     10ms      5ms
   Decode TBT 从 40ms 降到 15ms
```

### Chunk Size 选取

- **太小**（~64）：kernel launch 开销占比过高
- **太大**（~8192）：Decode TBT 仍然受影响
- **A100 推荐**：2048-4096
- **H100 推荐**：4096-8192

**设计哲学：** Chunk 大小和 Decode batch 大小交替，让 Compute 和 Memory 资源交替使用，避免任何一种成为瓶颈。

## 十一、Paged Attention + Copy-on-Write

![[11_Paged_Attention与Copy_on_Write.jpg]]

### Paged Attention 原理

将 KV Cache 分页为固定大小的 Block（类比操作系统的虚拟内存分页），通过 Page Table 映射逻辑位置到物理 Block：

```
逻辑序列：[K0][K1][K2][K3][K4][K5][K6][K7]
Page Table: [P0][P1][P2][P3][P4][P5][P6][P7]
物理 Block:  P1, P5, P2, P0, P7, ...
```

**三大优势：**

- **无碎片**：Block 固定大小，不存在外部碎片
- **零拷贝共享**：多请求共享同一前缀只需复制 Page Table，不复制数据
- **按需分配**：只分配实际使用的 Block

### Copy-on-Write 机制

```
父请求               子请求（fork）
KV Blocks: [P0, P1, P2]    shared Page Table: [P0, P1, P2]
                           ref_count = 2

子请求需要写入 P1 时：
  ① 分配新 Block P1'
  ② 复制 P1 → P1'  （真正的 Copy）
  ③ 更新子请求的 Page Table：P1 → P1'
  ④ ref_count 原 P1 减 1
```

> RadixAttention 和 Paged Attention + CoW 的组合：Radix Tree 管理宏观的 KV Cache 共享拓扑（哪些请求共享哪些前缀），Paged Attention 管理微观的物理内存分配。两者配合，让 KV Cache 的管理从"每请求独立"变为"全系统共享"。

## 十二、CFSM + Jump-Forward 约束解码

![[12_CFSM与Jump_Forward解码.jpg]]

### 对比三种约束解码方案

| 方案 | 原理 | 额外开销 |
|------|------|---------|
| Token-Level Masking | 在 logits 层面设置 -inf mask | ~30% |
| DFA FSM | 确定性有限状态机引导 | ~10-20% |
| **CFSM** | Compressed FSM + Jump-Forward | **~5%** |

### CFSM 的两阶段加速

**Jump-Forward（Prefill 阶段）：**

对于确定性 token 序列（如 JSON 中的 `{ "name": "`），编译为一条压缩路径，用一次 Prefill 直接跳过：

```
Token-Level 方式：每个 token 独立 Decode
{ → " → n → a → m → e → " → : → "     （9 次 Decode）

Jump-Forward：一次 Prefill 全部跳过
{"name":"   → 1 次 Prefill                     （1 次 Prefill，Compute-bound 高效处理）
```

**Compressed FSM（Decode 阶段）：**

将 FSM 状态图压缩，减少状态数。解码时只有分支点（Branching）走 GPU Decode，确定性路径全部跳过。典型场景下节省 40-50% 的 Decode 调用。

### Schema Cache

对于重复的 JSON Schema，CFSM 的编译结果可以被缓存（Schema Cache），避免每次请求重新编译。

## 十三、Frontend DSL：gen / select / fork

![[13_DSL编程原语_gen_select_fork.jpg]]

### 三个核心原语

```python
import sglang as sgl

# s 是一个 StreamState，背后自动管理 KV Cache
s = "The capital of France is"

# gen: 生成文本（支持 stop/regex/temperature）
s += sgl.gen("name", stop="\n", temperature=0.7)
result = s["name"]  # "Paris"

# select: 从候选项中计算 log P 选择
s += sgl.select("sentiment", choices=["positive", "negative", "neutral"])

# fork: 复制 n 份状态，各自独立展开
forks = s.fork(3)  # 创建 3 个分支
for i, f in enumerate(forks):
    f += sgl.gen(f"branch_{i}")
```

### fork 的核心价值：无需重复 Prefill

```
原始方式：每个分支独立发送 HTTP 请求
  Prefill(共享前缀) + Decode(分支1)  ← 浪费
  Prefill(共享前缀) + Decode(分支2)  ← 浪费
  Prefill(共享前缀) + Decode(分支3)  ← 浪费

SGLang fork 方式：
  一次 Prefill(共享前缀) → KV Cache 共享
    ├── Decode(分支1)
    ├── Decode(分支2)
    └── Decode(分支3)
```

**典型场景：**
- Tree-of-Thought：每次展开 forking 多个推理路径，共享前缀自动复用
- Best-of-N 采样：fork N 个分支，各自生成完整回答，选择最优
- 多轮对话：每个用户共享 System Prompt 前缀

## 十四、性能优化与 Tensor Parallelism

![[14_性能优化与Tensor_Parallelism.jpg]]

### 级联注意力优化

将 Prefill 和 Decode 的计算节奏协调起来，减少两种负载互相等待的间隙：

```
无优化:   [Prefill 5ms] [gap] [Decode 1ms] [gap] [Decode 5ms]
优化后:   一次 Prefill (3ms) + 级联 Decode (7ms)
总耗时:   12ms（vs 原来的 5+1+5 = 11ms 但有大量间隙浪费）
吞吐提升: ~2.1x
```

### Tensor Parallelism (TP)

SGLang 使用 TP + DP 的组合（vLLM 使用 TP + PP + DP）：

```
TP 方案：每张 GPU 持有完整的模型层权重切片，QKV 投影并行计算

GPU0: Q₀ K₀ V₀    GPU1: Q₁ K₁ V₁    GPU2: Q₂ K₂ V₂    GPU3: Q₃ K₃ V₃
        ↓                    ↓                   ↓                   ↓
       Attention           Attention          Attention          Attention
        ↓                    ↓                   ↓                   ↓
    AllReduce (NCCL over NVLink)
```

**TP 的权衡：**

- NVLink 带宽足够时，TP 的通信开销小于 PP 的流水线气泡
- 单机 8 卡内 TP 是最优策略；跨机仍需 DP/PP
- SGLang 优先 TP，因为它追求低延迟场景（共享前缀多，减少 Prefill 后其余请求都很快）

### 预热机制

SGLang 使用预热（Precept）平滑负载峰值：

$$\text{Precept}(X) = \min\left(1, \frac{X}{\text{capacity}}\right)$$

当请求量超过当前容量时，不接受全部新请求而是平滑接入，避免 KV Cache 被瞬间冲垮。

## 十五、SGLang vs vLLM 对比总结

![[15_SGLang_vs_vLLM对比总结.jpg]]

### 技术路线对比

| 维度 | SGLang | vLLM |
|------|--------|------|
| KV Cache 管理 | **Radix Tree + LPM 调度** | Hash Table |
| 约束解码 | **CFSM + Jump-Forward（5%）** | Outlines（10-30%） |
| 编程模型 | **gen/select/fork DSL** | 无 |
| 分布式 | **TP + DP（低延迟优先）** | TP + PP + DP（更成熟） |

### 适用场景：选 SGLang

- ✅ 多轮对话系统（共享 System Prompt）
- ✅ API 批处理服务（大量请求共享前缀）
- ✅ 结构化输出（JSON / SQL / 正则）
- ✅ Tree-of-Thought / Best-of-N（fork 原语）
- ✅ 在线客服 / RAG（前缀高度重复）

### 适用场景：选 vLLM

- ✅ 独立短请求（无共享前缀）
- ✅ 多机分布式部署（PP 更成熟）
- ✅ 已有 vLLM 生产管线（生态成熟度）
- ✅ 需多种模型架构支持（vLLM 支持更广）

### 性能提升幅度

| 技术 | 提升幅度 |
|------|---------|
| RadixAttention | Prefill 10-25x 加速 |
| Continuous Batching | 2-5x vs Static Batching |
| CFSM Jump-Forward | 1.5-2x 结构化解码加速 |
| Cascade Attention | 1.5-3x 吞吐提升 |
| **综合** | **SGLang vs vLLM 1.5-5x** |

## 关键公式速查

### KV Cache 大小

$$\text{KV\_Size} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{dtype\_size}$$

### Attention 计算复杂度

$$\text{Prefill: } QK^T = [n \times d] \times [d \times n] \quad \text{(Compute-bound)}$$
$$\text{Decode: } qK_{\text{cache}}^T = [1 \times d] \times [d \times t] \quad \text{(Memory-bound)}$$

### Block 大小

$$\text{Block\_Size} = 2 \times n_{\text{layers}} \times d_{\text{model}} \times \text{block\_size} \times \text{dtype\_size}$$

## 来源

- 图片来自小红书笔记：[SGLang高性能推理框架全拆解](https://www.xiaohongshu.com/explore/6a3a5a23000000000603533a)
- 作者：Vincent | AIGC
- 话题标签：#SGLang #LLM推理 #RadixAttention #KVCache #大模型部署 #推理优化 #vLLM对比 #AI基础设施 #机器学习系统
