---
tags:
  - 训练基础设施
  - LLM推理
  - nano-vllm
  - LLMEngine
created: 2026-07-07
up: "[[模型执行引擎总览|模型执行引擎总览]]"
---

# LLMEngine 主循环

`LLMEngine` 是整个推理引擎的中枢控制器。它负责初始化所有子系统、管理主循环、以及对外提供 `generate()` 接口。

## 初始化流程 `__init__()`

```python
def __init__(self, model, **kwargs):
    # 1. 过滤 kwargs，构建 Config
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(model, **config_kwargs)

    # 2. 设置全局 block_size
    Sequence.block_size = config.kvcache_block_size

    # 3. 启动 Tensor Parallel Worker 进程
    self.ps = []
    self.events = []
    ctx = mp.get_context("spawn")
    for i in range(1, config.tensor_parallel_size):
        event = ctx.Event()
        process = ctx.Process(target=ModelRunner, args=(config, i, event))
        process.start()
        self.ps.append(process)
        self.events.append(event)

    # 4. 在主进程中创建 ModelRunner (rank 0)
    self.model_runner = ModelRunner(config, 0, self.events)

    # 5. 加载 tokenizer 并设置 EOS
    self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
    config.eos = self.tokenizer.eos_token_id

    # 6. 创建调度器
    self.scheduler = Scheduler(config)

    # 7. 注册退出清理
    atexit.register(self.exit)
```

### 关键细节

**`ctx = mp.get_context("spawn")`**：在 CUDA 环境中必须使用 `spawn` 而非默认的 `fork`。这是因为 CUDA 上下文不能被 fork——`fork` 会复制父进程的 CUDA 上下文到子进程，导致 GPU 资源冲突。`spawn` 创建全新的 Python 解释器进程，每个进程独立初始化 CUDA。

**Worker 进程的 rank > 0**：主进程（rank 0）同时运行 LLMEngine 和 ModelRunner，而 Worker 进程（rank 1~N）只运行 ModelRunner 的 `loop()` 方法，等待主进程通过 SharedMemory 发指令。

**`atexit.register(self.exit)`**：确保即使异常退出也能清理进程和 CUDA 资源。

## step() — 单步执行

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

每一步的执行顺序：调度 → GPU 推理 → 后处理 → 返回完成的序列。

`num_tokens` 的正负号有一个巧妙用途：Prefill 时返回正数（本轮处理的 token 数），Decode 时返回负数（-完成的序列数）。在 `generate()` 中据此判断本轮是 Prefill 还是 Decode：

```python
if num_tokens > 0:
    prefill_throughput = num_tokens / (perf_counter() - t)   # tokens/s
else:
    decode_throughput = -num_tokens / (perf_counter() - t)   # sequences/s
```

## generate() — 完整的生成循环

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    pbar = tqdm(total=len(prompts), desc="Generating", ...)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)

    # 批量添加请求
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()        # 执行一步
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
        pbar.set_postfix({"Prefill": f"{int(prefill_throughput)}tok/s",
                          "Decode": f"{int(decode_throughput)}tok/s"})
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            pbar.update(1)                      # 每完成一个序列更新进度条

    pbar.close()
    # 按 seq_id 排序保证输出顺序
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids} for token_ids in outputs]
    return outputs
```

设计亮点：
- **先批量添加再循环 step**：所有请求一次性加入，调度器统一管理，无需等一个完成再加下一个
- **tqdm 进度条显示实时吞吐量**：直观展示 Prefill 和 Decode 的 tokens/s
- **按 seq_id 排序输出**：保证结果顺序与输入一致
- **返回 dict 格式**：同时包含 `text`（解码后文本）和 `token_ids`（原始 token）
