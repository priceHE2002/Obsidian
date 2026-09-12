---
tags:
  - 现代硬件算法
  - SIMD并行
created: 2026-09-11
source: https://en.algorithmica.org/hpc/simd/auto-vectorization/
original_title: Auto-Vectorization and SPMD
---

# 自动向量化与 SPMD

SIMD 并行最常用于*易并行*（embarrassingly parallel）的计算：即对其所做的事无非是对数组所有元素施加某个逐元素函数，再把结果写回某处。在这种情形下，你甚至不需要了解 SIMD 的工作原理：编译器完全有能力自行优化这类循环——你只需知道存在这样的优化，并且它通常能带来 5~10 倍的加速。

什么都不做、单纯依赖自动向量化，实际上是最流行的使用 SIMD 的方式。事实上，在许多情况下，甚至建议坚持使用普通的标量代码，以换取简洁性与可维护性。

但即便对于那些看起来很容易向量化的循环，也常常因为某些技术细节而未被优化。[[契约式编程.md|如同许多其他情形一样]]，编译器可能需要程序员提供一些额外信息，因为程序员对问题的了解可能比静态分析所能推断的更多。

### 潜在问题

考虑我们[[内建函数与向量类型.md#SIMD 寄存器|最初使用]]的 "a + b" 例子：

```c++
void sum(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}
```

让我们站在编译器的角度思考，当这个循环被向量化时，可能出什么错。

**数组大小。** 如果数组大小事先未知，那么它可能小到根本不值得做向量化。即便数组足够大，我们仍需要插入一次额外的检查，用以处理循环中剩余的、需用标量方式处理的部分，而这会付出一条分支的代价。

为了消除这些运行时检查，请使用编译期常量作为数组大小，并且最好将数组补齐到 SIMD 块大小的最近整数倍。

**内存别名。** 即便数组大小问题不存在，向量化这个循环在技术上也不总是正确的。例如，数组 `a` 与 `c` 可能以这样的方式相交：它们的起点仅相差一个位置——因为谁知道呢，也许程序员本就想通过这种方式用卷积计算斐波那契数列。在这种情况下，SIMD 块中的数据会相互重叠，观察到的行为将不同于标量情形。

当编译器无法证明该函数可能被用于相交数组时，它必须生成两个实现变体——一个向量化的、一个"安全"的——并插入运行时检查以在二者之间选择。为了避免这些检查，我们可以通过添加 `__restrict__` 关键字告诉编译器不存在内存别名：

```cpp
void add(int * __restrict__ a, const int * __restrict__ b, int n) {
    for (int i = 0; i < n; i++)
        a[i] += b[i];
}
```

另一种 SIMD 特有的方式是"忽略向量依赖"编译指示。它是一种通用的、向编译器声明循环各次迭代之间不存在依赖的方法：

```c++
#pragma GCC ivdep
for (int i = 0; i < n; i++)
    // ...
```

**对齐。** 编译器同样对数组的对齐情况一无所知，因而要么在启动向量化段之前处理数组开头的一些元素，要么通过使用[[数据搬移.md|非对齐内存访问]]而潜在地损失一些性能。

为了帮助编译器消除这种边界情况，我们可以对静态数组使用 `alignas` 说明符，并使用 `std::assume_aligned` 函数标记指针已对齐。

**检查向量化是否发生。** 无论哪种情况，检查编译器是否如你所愿地向量化了循环都很有用。你可以[[编译的各个阶段.md|将其编译到汇编]]并寻找以 "v" 开头的指令块，或者添加 `-fopt-info-vec-optimized` 编译器标志，让编译器指出自动向量化发生在何处以及使用了何种 SIMD 宽度。若把 `optimized` 换成 `missed` 或 `all`，你还能获得关于其他位置为何未发生向量化的一些解释。

告知编译器我们的确切意图还有[许多其他方式](https://software.intel.com/sites/default/files/m/4/8/8/2/a/31848-CompilerAutovectorizationGuide.pdf)，但在特别复杂的情况下——例如循环内部存在大量分支或函数调用——退到更低一层抽象、手动做向量化会更简单。

### SPMD

在自动向量化与手动使用 SIMD 内建函数之间，存在一种巧妙的折中："单程序多数据"（single program, multiple data，SPMD）。这是一种计算模型，程序员编写看似普通串行程序的代码，但它实际上在硬件上并行执行。

编程体验大体相同，并且仍然存在一个根本限制：计算必须是数据并行的；但 SPMD 确保了向量化必定发生，而不论编译器与目标 CPU 架构如何。它还允许计算跨多个核心自动并行化，在某些情况下甚至卸载到其他类型的并行硬件上。

一些现代语言（[Julia](https://docs.julialang.org/en/v1/base/base/#Base.SimdLoop.@simd)）、多进程 API（[OpenMP](https://www.openmp.org/spec-html/5.0/openmpsu42.html)）以及专用编译器（Intel 的 [ISPC](https://ispc.github.io/)）都支持 SPMD，但它在 GPU 编程领域取得了最大的成功，因为那里的问题与硬件都是大规模并行的。

我们将在第 2 部分更深入地探讨这种计算模型。

<!-- This approach is especially popular with [game developers](https://twitter.com/pbrubaker/status/1537041398037303296) because they need to support many platforms and have reliable performance, and also because it resembles the way graphics programming is done. -->
