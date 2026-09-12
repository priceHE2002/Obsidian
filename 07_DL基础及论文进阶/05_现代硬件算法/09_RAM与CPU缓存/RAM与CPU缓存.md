---
tags:
  - 现代硬件算法
  - RAM与CPU缓存
created: 2026-09-11
source: https://en.algorithmica.org/hpc/cpu-cache/
original_title: RAM & CPU Caches
---

# RAM 与 CPU 缓存

在[[外部存储.md|上一章]]中，我们从理论角度研究了计算机存储器，并使用[[外部存储模型.md|外部存储模型]]来估计访存受限算法的性能。

虽然外部存储模型对于涉及 HDD 和网络存储的计算大致准确——在这些场景下，对内存中数值进行算术运算的开销相比外部 I/O 操作可以忽略不计——但对于缓存层级中更低的层级而言，它过于粗略，因为在这些层级中，这两类操作的开销已经相当接近。

要对内存中的算法进行更精细的优化，就必须开始考虑 CPU 缓存系统的大量具体细节。与其去研读那些充满枯燥规格说明和理论上限的 Intel 文档，我们将通过运行大量小型基准测试程序、采用与实际代码中常见的访问模式相似的访问方式，来以实验手段估计这些参数。

<!--

At this level, we can no longer simply ignore either all arithmetic or memory operations. To perform more fine-grained optimization of realistic programs, we need to know the cost of memory accesses on real systems and in real units — in cycles and nanoseconds — along with many other intricacies of the RAM and CPU cache system.

-->

### 实验环境

与之前一样，我将所有实验运行在 Ryzen 7 4700U 上，这是一颗 "Zen 2" CPU，其主要与缓存相关的规格如下：

- 8 个物理核心（无超线程），主频 2GHz（加速模式可达 4.1GHz——[[如何获得准确结果.md|我们在实验中将其禁用]]）；
- 256K 的 8 路组相联 L1 数据缓存，即每核 32K；
- 4M 的 8 路组相联 L2 缓存，即每核 512K；
- 8M 的 16 路组相联 L3 缓存，在 8 个核心之间[[内存共享.md|共享]]；
- 16GB（2×8G）DDR4 RAM，频率 2667MHz。

你可以在 Linux 上运行 `dmidecode -t cache` 或 `lshw -class memory`，或在 Windows 上安装 [CPU-Z](https://en.wikipedia.org/wiki/CPU-Z) 来与自己的硬件进行对比。你也可以在 [WikiChip](https://en.wikichip.org/wiki/amd/ryzen_7/4700u) 和 [7-CPU](https://www.7-cpu.com/cpu/Zen2.html) 上找到关于该 CPU 的更多细节。并非所有结论都能推广到现有的每一种 CPU 平台。

<!--

Although the CPU can be clocked at 4.1GHz in boost mode, we will perform most experiments at 2GHz to reduce noise — so keep in mind that in realistic applications the numbers can be multiplied by 2.

-->

由于[[如何获得准确结果.md|难以阻止编译器优化掉未使用的值]]，本文中的代码片段为了叙述清晰做了适度简化。如果你想自行复现，请查阅[代码仓库](https://github.com/sslotin/amh-code/tree/main/cpu-cache)。

### 致谢

本章的灵感来自 Igor Ostrovsky 的 "[Gallery of Processor Cache Effects](http://igoro.com/archive/gallery-of-processor-cache-effects/)" 以及 Ulrich Drepper 的 "[What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)"，这两篇文章都可以作为良好的延伸阅读。

<!--

### Recall: CPU Caches

If you jumped to this page straight from Google or just forgot what [we've been doing](../), here is a brief summary of how memory operations work in CPUs:

The last few points may be a bit hand-wavy, but don't worry: they will become clear as we go along with the experiments and demonstrate it all in action.

## Summary and Lessons Learned

Excluding TLB, our experiments suggest the following:

| Type | Size | Latency | Bandwidth |
|:-----|:-----|---------|-----------|
| L1   | 32K  | 2ns     | $\infty$  |
| L2   | 512K | 10ns    | 50G/s     |
| L3   | 4M   | 50ns    | 35G/s     |
| RAM  | GB   | 100ns   | 8G/s      |

There are more thorough [measurements for Zen 2](https://www.7-cpu.com/cpu/Zen2.html).

We can learn valuable lessons from our experiments. There are two types of memory-bound algorithms. Loops or data structures.

**Latency-constrained.** For the purpose of designing algorithms, a more important characteristic is the **bandwidth-latency product** which basically tells how many cache lines you can request while waiting for the first one without queueing up. It is around 5 or more on most systems. CPUs can detect simple patterns such as linear iteration forward or backward.

**Bandwidth-constrained.** We started the previous section with how it is not relevant which algorithm is used to determine cache eviction. In most practical cases, this is really the case.

But in some cases the specifics start to matter. In set-associative cache, there may be a problem when we are only working with data cells that all map to the same cache line. When is this the case? When we are considering memory locations that are all have the same remainder modulo some large power of two.

Unfortunately, this happens quite often, as we programmers love using powers of two for our algorithms and data structures.

Fortunately, this is easy to fix: just don't use powers of two. Not necessarily for the algorithm, but at least for the memory layout.

More fundamental [academic paper](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1993/CSD-93-767.pdf) by Rafael Saavedra and Alan Smith.

-->
