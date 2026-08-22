---
tags: [ai-infra, moc]
---
# AI Infra Knowledge Map

## 底座

[[13_AI_Infra/04_PyTorch/01_PyTorch-GPU-Execution|PyTorch GPU Execution]] → [[13_AI_Infra/05_LLM-Inference/01_LLM-Inference-Mental-Model|LLM Inference]] → [[13_AI_Infra/01_GPU/01_GPU-Mental-Model|GPU Mental Model]]

## 推理系统主线

[[13_AI_Infra/05_LLM-Inference/02_Prefill-vs-Decode|Prefill vs Decode]] → [[13_AI_Infra/06_KV-Cache/01_KV-Cache-Fundamentals|KV Cache]] → [[13_AI_Infra/07_Scheduling/01_Continuous-Batching|Continuous Batching]] → [[13_AI_Infra/06_KV-Cache/02_Paged-KV-Cache|Paged KV]] → [[13_AI_Infra/07_Scheduling/02_Chunked-Prefill|Chunked Prefill]]

## GPU 优化主线

[[13_AI_Infra/02_CUDA/01_CUDA-Learning-Path|CUDA]] → [[13_AI_Infra/03_Triton/01_Triton-Learning-Path|Triton]] → [[13_AI_Infra/11_Performance/02_Nsight-Checklist|Nsight]] → [[13_AI_Infra/11_Performance/01_Benchmark-and-Profiling|Performance Engineering]]

## 源码阅读

[[13_AI_Infra/09_vLLM/01_vLLM-Source-Reading|vLLM]] → [[13_AI_Infra/10_SGLang/01_SGLang-Source-Reading|SGLang]]

## 后置能力

[[13_AI_Infra/08_Distributed/01_Distributed-Later|Distributed Systems]]

## 面试输出

[[13_AI_Infra/12_Interview/01_AI-Infra-Interview-Questions|Question Bank]] + [[13_AI_Infra/12_Interview/02_MiniServe-60s-Pitch|60s Pitch]] + [[13_AI_Infra/13_MiniServe/04_Resume-and-Story|Resume & Project Story]]

## Vault 交叉资源

- [[08_DL基础及论文进阶/03_训练基础设施/nano-vllm 源码剖析/nano-vllm 源码剖析|nano-vllm 源码剖析]] — MiniServe 每做一个模块，对照它的实现
- [[08_DL基础及论文进阶/03_训练基础设施/07_FlashAttention/FlashAttention|FlashAttention 笔记]] — GPU 优化主线第 4 月的论文背景
- [[08_DL基础及论文进阶/00_基础知识/05_SGLang推理框架/SGLang推理框架|SGLang推理框架]] — 5 月源码阅读阶段的参考资料
