---
tags:
  - 论文
  - 训练基础设施
  - 分布式训练
  - 数据并行
  - PyTorch
created: 2026-06-30
paper_title: "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"
paper_authors: "Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison et al."
paper_year: 2023
paper_venue: "VLDB 2023"
paper_citations: "~1,200+"
paper_url: "https://arxiv.org/abs/2304.11277"
github: "https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp"
---

# FSDP

**PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**
*Yanli Zhao, Andrew Gu, Rohan Varma et al. | PyTorch Team (Meta) | VLDB 2023 | arXiv: 2304.11277*

> PyTorch 原生的 ZeRO-3 实现，将模型参数、梯度和优化器状态分片到所有 GPU 上，使超大模型训练不再依赖第三方库（如 DeepSpeed）。FSDP 已成为 HuggingFace Transformers 生态中分布式训练的事实标准后端。

---

## 一、Background / Core Idea

### 1.1 问题：DDP 的显存墙

分布式数据并行（DDP）是最直观的并行策略：每张 GPU 维护完整模型副本，独立计算梯度，通过 AllReduce 同步梯度。

```
DDP 显存分布（单卡）:
┌─────────────────────────────────────┐
│       模型参数:    7B × 2B = 14GB      │
│       梯度:       7B × 2B = 14GB      │
│       优化器状态:  7B × 4B × 2 = 56GB │  (Adam: fp32 mom + var)
│       激活值:      序列依赖             │
│       总需求:      >84GB (7B 模型)     │
└─────────────────────────────────────┘
```

- 7B 模型 DDP 训练需要 **每卡 >84GB**（Adam 优化器）
- 175B 模型需要 **每卡 >2TB**——完全不可行
- 激活值（activation memory）在大序列下进一步推高需求

### 1.2 核心洞察：ZeRO 与分片思想

ZeRO（Zero Redundancy Optimizer, Rajbhandari et al., 2020）提出革命性思路：

> 每个张量只存储一个分片（shard），而非全量副本。全局通信在需要时执行。

| 方案 | 模型参数 | 梯度 | 优化器状态 | 通信量 | 显存节省 |
|:----|:--------:|:----:|:---------:|:-----:|:--------:|
| DDP | 全量 | 全量 | 全量 | 1× AllReduce | 1× |
| **ZeRO-1** | 全量 | 全量 | **分片** | 1× AllReduce | **4× (Adam)** |
| **ZeRO-2** | 全量 | **分片** | **分片** | 1× AllReduce | **8× (Adam)** |
| **ZeRO-3 (FSDP)** | **分片** | **分片** | **分片** | **all-gather + reduce-scatter** | **N× (按 GPU 数)** |

### 1.3 ZeRO-3（FSDP）的核心机制

ZeRO-3 将所有三个状态（参数、梯度、优化器）分片，代价是在前向/反向传播中额外增加 all-gather 通信：

```
前向: all-gather 收集完整参数 → 计算 → 丢弃其他分片
反向: all-gather 收集完整参数 → 计算梯度 → reduce-scatter 聚合梯度
```

- **与 DDP 的比较**：FSDP 增加 $2\times$ all-gather 通信（前向一次、反向一次），但 reduce-scatter 替代了 DDP 的 all-reduce，综合通信量约为 DDP 的 1.5 倍
- **显存节省**：$\text{ratio} \approx N$（GPU 数），175B 模型在 64 GPU 上单卡显存从 2TB 降至约 32GB

### 1.4 FSDP vs DeepSpeed ZeRO-3

| 特性 | PyTorch FSDP | DeepSpeed ZeRO-3 |
|:----|:------------:|:----------------:|
| 开发团队 | Meta PyTorch | Microsoft DeepSpeed |
| API 风格 | 原生 PyTorch（`wrap()` / `auto_wrap()`） | 配置文件 + `model_engine` |
| 自动分层 | `auto_wrap_policy`（按 Transformer block） | `stage3_param_persistence_threshold` |
| 通信后端 | NCCL（PyTorch 原生） | NCCL（有自定义 kernel） |
| CPU offload | 支持 | 支持（更成熟） |
| 混合精度 | `@torch.cuda.amp` + FSDP | 内置 `bf16` mode |
| HuggingFace 集成 | **官方推荐** | 需额外插件 |
| overlap 通信 | `forward_prefetch` / `backward_prefetch` | `overlap_comm` |
| 推理支持 | FSDP inference mode | ZeRO-Inference |

**FSDP 的核心优势是作为 PyTorch 原生组件**——零外部依赖，与 `torch.compile`、DTensor、Distributed Checkpoint 深度融合。

---

## 二、Method / Architecture / Technical Contribution

### 2.1 FSDP 分片配置

| 参数 | 类型 | 说明 | 显存节省 ↑ | 通信量 ↓ |
|:----|:----|:----|:---------:|:--------:|
| `sharding_strategy` | `FULL_SHARD`, `HYBRID_SHARD`, `NO_SHARD` | 控制分片粒度 | 高 | 低 |
| `cpu_offload` | bool | 优化器状态卸载到 CPU | 极高 | 极高延迟 |
| `auto_wrap_policy` | callable | 自动包裹子模块 | 灵活性 | 易用性 |
| `limit_all_gathers` | int | 限制并行 all-gather 数量 | 显存抖动控制 | — |

**三种分片策略对比：**

| 策略 | 参数分片 | 梯度分片 | 通信模式 | 典型用途 |
|:----|:-------:|:--------:|:--------:|:--------|
| `FULL_SHARD` | ✅ | ✅ | AllGather + ReduceScatter | 模型大于单卡内存（推荐） |
| `HYBRID_SHARD` | 节点内分片 | 节点内分片 | 节点内 AllGather + 节点间 AllReduce | 跨节点延迟高时 |
| `NO_SHARD` | ❌ | ❌ | AllReduce | 等同 DDP |
| `SHARD_GRAD_OP` (ZeRO-2) | ❌ | ✅ | AllGather + ReduceScatter | 混合方案 |

### 2.2 前向/反向执行流

```
FSDP 包装层（FullyShardedDataParallel）执行流程：

前向传播（每个 FSDP unit）:
[train] partition = all_gather(all_param_shards)
[train] output = module.forward(partition)
[train] discard collected shards (except own)

反向传播（每个 FSDP unit）:
[grad] all_gather(all_param_shards)     # 重新收集参数
[grad] grad_output = autograd.grad(module, input)
[grad] reduce_scatter(grad)             # 聚合 + 分片
[grad] optimizer.step(own_shard)        # 仅更新分片
```

### 2.3 混合精度策略（Mixed Precision）

FSDP 支持三态混合精度，参数维护在指定精度：

| 方案 | 模型参数 | 梯度 | 通信 | 最佳场景 |
|:----|:--------:|:----:|:----:|:--------|
| fp16 | fp16 | fp16 | fp16 | V100（Tensor Core fp16） |
| bf16 | bf16 | bf16 | bf16 | A100/H100（bf16 Tensor Core） |
| fp32 | fp32 | fp32 | fp32 | 精度极度敏感任务 |
| **fp16_bf16** | fp32（主） | fp16/bf16 | fp16/bf16 | 默认推荐 |

**关键设计**：参数保持 fp32 的主副本用于更新，但在前向/反向中被 cast 到更低精度用于计算。通信默认使用 bf16（A100/H100）以减少带宽。

### 2.4 通信重叠优化（Overlap Strategies）

FSDP 论文对通信重叠做了系统实验：

| 策略 | 通信时间 | 计算时间 | 总时间 | 吞吐 Loss |
|:----|:-------:|:--------:|:------:|:---------:|
| 无重叠 | 35% | 100% | 135% | 35% |
| `backward_prefetch` | 35% | 100% | 115% | 15% |
| `forward_prefetch` | 35% | 100% | 110% | 10% |
| **两者全开** | 35% | 100% | **105%** | **5%** |

- `backward_prefetch`：在反向传播前预取下一层的参数
- `forward_prefetch`：在前向传播前预取后续层的参数
- `limit_all_gathers`：限制未完成 all-gather 数量，防止显存峰值

### 2.5 Auto Wrap Policy（自动包裹策略）

FSDP 不要求手动包裹每个子模块，通过 `auto_wrap_policy` 自动决定分片边界：

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 按 Transformer Block 边界自动包裹
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer, GPT2Block}
)
```

**理论依据**：Transformer 的 block 结构天然适合做分片边界——每个 block 的前向/反向是一个独立的计算单元，block 间的通信可以通过 prefetch 完美隐藏。

### 2.6 与 `torch.compile` 的协同

FSDP + `torch.compile` 在 PyTorch 2.0+ 中需要特殊处理：

- **问题**：`torch.compile` 默认会推断整个计算图，但与 FSDP 的预取逻辑冲突
- **解决**：使用 `torch.distributed._composable.fsdp` API（FSDP2），与 `torch.compile` 兼容性更好
- **结果**：FSDP2 + `torch.compile` 使 A100 上 7B 模型训练吞吐提升约 15-30%

---

## 三、Experiments and Key Findings

### 3.1 可扩展性（Scaling Efficiency）

论文在 256-512 块 A100 上测试不同模型规模的扩展效率：

| 模型 | 参数量 | GPU 数 | MFU | 每 GPU TFLOPs | 线性扩展效率 |
|:----|:-----:|:------:|:---:|:-------------:|:------------:|
| GPT-2 Large | 774M | 256 | 48% | 148 | 85% |
| GPT-2 XL | 1.5B | 256 | 45% | 138 | 83% |
| GPT-3 13B | 13B | 256 | 39% | 120 | 80% |
| **GPT-3 175B** | **175B** | **512** | **40-45%** | **123-138** | **~91%**（模型内扩展） |
| OPT-30B | 30B | 256 | 37% | 114 | — |
| T5-11B | 11B | 128 | 42% | 130 | — |

**关键观察**：FSDP 在超大规模下扩展效率达 80-90%，当模型规模相对于 GPU 集群足够大时，通信/计算比降低，效率提升。

### 3.2 FSDP vs DeepSpeed ZeRO-3 性能对比

| 模型 | GPU 数 | FSDP (ms/step) | DeepSpeed (ms/step) | FSDP 吞吐相对 |
|:----|:------:|:--------------:|:-------------------:|:------------:|
| GPT-2 1.5B | 8 × A100 | **580** | 620 | **+7%** |
| GPT-2 1.5B | 16 × A100 | **310** | 335 | **+8%** |
| OPT-13B | 8 × A100 | **2100** | 2350 | **+12%** |
| OPT-13B | 16 × A100 | **1120** | 1180 | **+5%** |

**FSDP 通常比 DeepSpeed ZeRO-3 快 5-12%**，但在需要额外高级功能（如 offload 到 NVMe）时，DeepSpeed 更成熟。

### 3.3 显存实测

| 配置 | 模型 | GPU 数 | DDP 显存 | FSDP 显存 | 节省 |
|:----|:----|:------:|:--------:|:---------:|:----:|
| fp32, no act ckpt | GPT-2 1.5B | 8 | OOM | 8.2GB | >90% |
| fp16, act ckpt | GPT-3 13B | 8 | OOM | 12.4GB | >90% |
| bf16, act ckpt | Llama 7B | 4 | 56GB | 18.6GB | 67% |
| bf16, act ckpt + CPU offload | Llama 13B | 4 | OOM | 8.9GB (GPU) | >90% |

---

## 四、Limitations and Challenges

1. **通信开销无法忽略**：FSDP 的理论通信量是 DDP 的 1.5 倍，在节点间带宽受限时（如 100GbE vs NVLink 600GB/s），通信会成为瓶颈
2. **计算/通信重叠的敏感性**：overlap 策略需要精心调参，`prefetch` 批次和 `limit_all_gathers` 参数不当会导致显存峰值或通信等待
3. **CPU offload 的延迟代价**：参数在 GPU↔CPU 间的传输增加数倍延迟，仅适合极低吞吐要求的实验
4. **与 `torch.compile` 的兼容性**：FSDP1（torch 2.0 前）与编译后端严重冲突，即使 FSDP2 也仍有边缘情况
5. **小模型效率不佳**：对于模型尺寸小于 GPU 显存的情况，FSDP 的分片只会增加通信开销而无法利
6. **手动调优复杂**：auto_wrap_policy + 分片策略 + 混合精度 + prefetch 策略的组合空间大，缺乏自动化

---

## 五、Relationship with Subsequent Work / Impact on the Field

| 后续工作 | 年份 | 与 FSDP 的关系 |
|---------|:----:|---------------|
| **DeepSpeed ZeRO-3** (Rajbhandari et al.) | 2020 | 并行开创性工作，FSDP 的设计很大程度上借鉴了 ZeRO |
| **FSDP2** (PyTorch 2.1+) | 2023 | 基于 DTensor 重写，支持 `torch.compile` |
| **DDP** (PyTorch) | 2019 | FSDP 的直接前身，为显存效率牺牲通信 |
| **HSDP** (Hybrid Sharding) | 2023 | 节点内 FSDP + 节点间 DDP 的混合策略 |
| **fully_sharded** (HuggingFace) | 2023 | HuggingFace Trainer 默认集成 FSDP |
| **Tensor Parallelism** (Megatron-LM) | 2022 | 与 FSDP 互补，处理单层过大的问题 |
| **Pipeline Parallelism** (GPipe/1F1B) | 2019 | 与 FSDP 可组合使用，用于超大模型分布式训练 |

**影响评估**：FSDP 将 ZeRO-3 的能力以原生 PyTorch API 的形式带给全社区。截至 2024 年，HuggingFace Trainer 的 FSDP 集成是开源社区训练 7B-70B 模型的首选方案。几乎每个开源模型（Llama、Mistral、Qwen、Gemma）的分布式训练代码都包含 FSDP 配置。

---

## 六、Implications for You / Hardware Compatibility

### GPU 显存估算（FSDP 训练 7B-70B 模型）

| 配置 | 模型 | GPU 数 | 每卡训练显存 | 每秒 Token 数（每卡） | 推荐 GPU |
|:----|:----|:------:|:-----------:|:-------------------:|:---------|
| bf16 FSDP (FULL_SHARD) | Llama 7B | 8 | ~12-16GB | ~2500 | ✅ A100 40GB / RTX 6000 Ada |
| bf16 FSDP (FULL_SHARD) | Llama 7B | 4 | ~22GB | ~2000 | ✅ A100 40GB |
| bf16 FSDP (HYBRID_SHARD) | Llama 13B | 16 | ~16-20GB | ~1200 | ✅ A100 40GB × 16 |
| bf16 FSDP (FULL_SHARD) | Llama 70B | 64 | ~32GB | ~500 | ⚠️ A100 80GB 集群 |
| bf16 + CPU offload | Llama 70B | 8 | ~12GB (GPU) | ~50 | ⚠️ GPU 足够但极慢 |
| fp32 FSDP | GPT-3 175B | 512 | ~32GB | ~150 | ❌ 仅 A100/H100 集群 |

### 对分布式训练实践的指导

- **优先 FULL_SHARD**：除非跨节点通信慢（<200 GbE），否则 FULL_SHARD 是最节约显存的选项
- **HYBRID_SHARD 适合慢速网络**：节点内 NVLink 做 FSDP + 节点间 AllReduce 做 DDP，减少跨节点通信量
- **开启两种 prefetch**：`backward_prefetch` + `forward_prefetch` 通常可恢复大部分通信开销（总吞吐损失降至 5%）
- **避免 CPU offload**：除非在消费级 GPU 上训练超过显存的大模型，否则 CPU offload 的延迟不可接受
- **auto_wrap_policy 必须正确**：错误的分片策略会导致跨 layer 的通信顺序化，严重降低并行度

### 与相关技术的硬件兼容性

| 技术组合 | 硬件要求 | 兼容性 |
|:--------|:--------|:------:|
| FSDP + Deepspeed ZeRO-3 混合使用 | 不可行（冲突） | ❌ |
| FSDP + Tensor Parallelism | 需要 NVSwitch 或高带宽互联 | ⚠️ 仅 A100/H100 集群 |
| FSDP + Activation Checkpointing | 无额外硬件需求 | ✅ 所有 GPU |
| FSDP + FlashAttention | 无额外需求 | ✅ 所有 GPU |
| FSDP2 + `torch.compile` | A100/H100（CUDA 12+） | ⚠️ 100% 兼容（但特定模块可能自动降级） |
| FSDP + DDP 混合训练 | 需要同一通信后端（NCCL） | ⚠️ 仅 HYBRID_SHARD |

### 硬件兼容性总结
- ✅ FSDP FULL_SHARD 7B：4-8 块 A100 40GB / RTX 6000 Ada
- ✅ FSDP FULL_SHARD 13B：8-16 块 A100 40GB
- ⚠️ FSDP FULL_SHARD 70B：64 块 A100 80GB 集群
- ⚠️ FSDP + CPU offload 70B：可用 8 块 A100 但极慢
- ❌ FSDP fp32 训练 175B：仅超大规模 H100 集群

---

## PDF

[[FSDP 原文.pdf]]
