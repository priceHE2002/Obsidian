---
tags: [miniserve, repo, engineering]
---
# MiniServe Repo Structure

```text
miniserve/
├── api/
│   ├── server.py
│   └── protocol.py
├── engine/
│   ├── engine.py
│   ├── request.py
│   └── request_manager.py
├── scheduler/
│   ├── scheduler.py
│   ├── policy.py
│   └── batch.py
├── memory/
│   ├── block.py
│   ├── allocator.py
│   ├── block_manager.py
│   ├── kv_cache.py
│   └── prefix_cache.py
├── model/
│   ├── runner.py
│   ├── loader.py
│   └── attention.py
├── kernels/
│   ├── torch/
│   ├── triton/
│   │   ├── rmsnorm.py
│   │   ├── rope.py
│   │   └── kv_cache.py
│   └── cuda/
├── sampling/
│   └── sampler.py
├── benchmark/
│   ├── latency.py
│   ├── throughput.py
│   ├── memory.py
│   └── workloads.py
├── metrics/
│   └── collector.py
├── tests/
├── examples/
├── README.md
└── pyproject.toml
```

## 依赖方向

```text
API → Engine → Scheduler → ModelRunner
                ↓          ↑
            MemoryManager ─┘
```

原则：底层不要反向依赖上层。

## 工程要求

- [ ] type hints
- [ ] unit tests
- [ ] deterministic benchmark seeds where applicable
- [ ] config-driven experiments
- [ ] structured JSON/CSV result output
- [ ] README architecture diagram
- [ ] Dockerfile（后期）
- [ ] CI（后期）
