---
tags:
  - 现代硬件算法
  - 算法案例
created: 2026-09-11
source: https://en.algorithmica.org/hpc/algorithms/argmin/
original_title: Argmin with SIMD
---

# 用 SIMD 求 Argmin

计算一个数组的*最小值*是[[归约.md|极易向量化]]的，因为它与任何其他归约并无区别：在 AVX2 中，你只需使用方便的 `_mm256_min_epi32` 内建函数作为内部操作即可。它在一个周期内就能算出两个 8 元素向量的最小值——甚至比标量情形更快，后者至少需要一次比较和一次条件移动。

求出该最小元素的*索引*（*argmin*）则要困难得多，但它仍然可以被非常高效地向量化。在本节中，我们设计一种算法，使其计算 argmin 的速度（几乎）与计算最小值一样快，并且比朴素标量方法快约 15 倍。

### 标量基线

在我们的基准测试中，我们创建一个由随机 32 位整数组成的数组，然后反复尝试在它们之中找出最小值的索引（若最小值不唯一，则取第一个）：

```c++
const int N = (1 << 16);
alignas(32) int a[N];

for (int i = 0; i < N; i++)
    a[i] = rand();
```

为了便于说明，我们假设 $N$ 是 2 的幂，并且为 $N=2^{13}$ 运行全部实验，这样[[内存带宽.md|内存带宽]]就不会成为问题。

要在标量情形下实现 argmin，我们只需要维护索引而不是最小值：

```c++
int argmin(int *a, int n) {
    int k = 0;

    for (int i = 0; i < n; i++)
        if (a[i] < a[k])
            k = i;
    
    return k;
}
```

它的运行速度约为 1.5 GFLOPS——即平均每秒处理 $1.5 \cdot 10^9$ 个值，或者说每个周期约 0.75 个值（CPU 主频为 2GHz）。

我们把它与 `std::min_element` 做个比较：

```c++
int argmin(int *a, int n) {
    int k = std::min_element(a, a + n) - a;
    return k;
}
```

<!--

https://github.com/llvm-mirror/libcxx/blob/78d6a7767ed57b50122a161b91f59f19c9bd0d19/include/algorithm#L2489

https://github.com/gcc-mirror/gcc/blob/16e2427f50c208dfe07d07f18009969502c25dc8/libstdc%2B%2B-v3/include/bits/stl_algo.h#L5606

```nasm
lea	r8, 24[rdx]	# __first,
mov	r11d, DWORD PTR [rax]
cmp	DWORD PTR 12[rdx], r11d
cmovl	rax, r10	# __result,, __result, __first
```

```nasm
cmp	eax, r12d	# prephitmp_103, _108	
jle	.L36	#,	
mov	eax, r12d	# prephitmp_103, _108	
mov	ecx, ebp	# k, ivtmp.29	
.L36:	
lea	rdx, 4[r8]	# ivtmp.29,	
```

```nasm
cmp	eax, r12d	# prephitmp_103, _108	
jle	.L36	#,	
mov	eax, r12d	# prephitmp_103, _108	
mov	ecx, ebp	# k, ivtmp.29	
.L36:	
lea	rdx, 4[r8]	# ivtmp.29,	
```

-->

GCC 给出的版本只有约 0.28 GFLOPS——显然，编译器没能穿透所有这些抽象。这又一次提醒我们：永远不要使用 STL。

### 索引向量

向量化标量实现的难点在于，相邻迭代之间存在依赖关系。当我们优化[[归约.md|数组求和]]时，遇到了同样的问题，而我们是通过把数组切分为 8 片来解决的，每一片代表索引模 8 余数相同的一个子集。我们可以在这里套用同样的技巧，只是我们还必须考虑数组的索引。

当我们把连续的元素及其索引都放进向量中时，就可以用[[无分支编程.md|谓词]]并行地处理它们：

```c++
typedef __m256i reg;

int argmin(int *a, int n) {
    // indices on the current iteration
    reg cur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // the current minimum for each slice
    reg min = _mm256_set1_epi32(INT_MAX);
    // its index (argmin) for each slice
    reg idx = _mm256_setzero_si256();

    for (int i = 0; i < n; i += 8) {
        // load a new SIMD block
        reg x = _mm256_load_si256((reg*) &a[i]);
        // find the slices where the minimum is updated
        reg mask = _mm256_cmpgt_epi32(min, x);
        // update the indices
        idx = _mm256_blendv_epi8(idx, cur, mask);
        // update the minimum (can also similarly use a "blend" here, but min is faster)
        min = _mm256_min_epi32(x, min);
        // update the current indices
        const reg eight = _mm256_set1_epi32(8);
        cur = _mm256_add_epi32(cur, eight);       // 
        // can also use a "blend" here, but min is faster
    }

    // find the argmin in the "min" register and return its real index

    int min_arr[8], idx_arr[8];
    
    _mm256_storeu_si256((reg*) min_arr, min);
    _mm256_storeu_si256((reg*) idx_arr, idx);

    int k = 0, m = min_arr[0];

    for (int i = 1; i < 8; i++)
        if (min_arr[i] < m)
            m = min_arr[k = i];

    return idx_arr[k];
}
```

它的运行速度约为 8–8.5 GFLOPS。迭代之间仍然存在一些相互依赖，因此我们可以通过每次迭代处理超过 8 个元素、并利用[[归约.md#归约|指令级并行]]来优化它。

这会在很大程度上提升性能，但还不足以追上计算最小值的速度（约 24 GFLOPS），因为还存在另一个瓶颈。在每次迭代中，我们需要一个融合加载的比较、一个融合加载的最小值、一次混合以及一次加法——也就是一共 4 条指令来处理 8 个元素。由于该 CPU（Zen 2）的解码宽度为 4，即使我们设法消除了所有其他瓶颈，性能仍然会被限制在 8 × 2 = 16 GFLOPS。

相反，我们将切换到另一种方法，它每个元素所需的指令更少。

### 分支并不可怕

当我们运行标量版本时，多久会更新一次最小值？

直觉告诉我们，如果所有值都是各自独立随机抽取的，那么下一个元素小于之前所有元素这一事件应该并不频繁。更精确地说，它等于已处理元素个数的倒数。因此，`a[i] < a[k]` 条件被满足的期望次数等于调和级数之和：

$$
\frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \ldots + \frac{1}{n} = O(\ln(n))
$$

所以对于一百个元素的数组，最小值大约更新 5 次；一千个元素的数组约 7 次；而一百万个元素的数组只有 14 次——这与所有"是否为新最小值"检查的总次数相比，其实一点都不大。

编译器大概无法自行推断出这一点，因此让我们[[情境化优化.md|显式地提供]]这一信息：

```c++
int argmin(int *a, int n) {
    int k = 0;

    for (int i = 0; i < n; i++)
        if (a[i] < a[k]) [[unlikely]]
            k = i;
    
    return k;
}
```

编译器[[机器码布局.md|优化了机器码布局]]，现在 CPU 能够以约 2 GFLOPS 的速度执行该循环——相比未加提示的循环的 1.5 GFLOPS，这是一个不大但可观的提升。

思路如下：如果在整个计算过程中我们只在十几次左右更新最小值，我们就可以抛弃所有向量混合和索引更新，只需维护最小值并定期检查它是否发生了变化。在这个检查内部，我们可以使用任意慢的更新 argmin 的方法，因为它只会被调用几次。

要用 SIMD 实现它，我们在每次迭代中只需要做一件事：一次向量加载、一次比较，以及一次测试是否为零：

```c++
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;
    
    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 8) {
        reg y = _mm256_load_si256((reg*) &a[i]); 
        reg mask = _mm256_cmpgt_epi32(p, y);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            for (int j = i; j < i + 8; j++)
                if (a[j] < min)
                    min = a[idx = j];
            p = _mm256_set1_epi32(min);
        }
    }
    
    return idx;
}
```

它已经有约 8.5 GFLOPS 的表现，但现在该循环受限于 `testz` 指令，而它只有一个的吞吐。解决办法是加载两个连续的 SIMD 块并对它们取最小值，这样 `testz` 实际上一次就能处理 16 个元素：

```c++
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;
    
    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 16) {
        reg y1 = _mm256_load_si256((reg*) &a[i]);
        reg y2 = _mm256_load_si256((reg*) &a[i + 8]);
        reg y = _mm256_min_epi32(y1, y2);
        reg mask = _mm256_cmpgt_epi32(p, y);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            for (int j = i; j < i + 16; j++)
                if (a[j] < min)
                    min = a[idx = j];
            p = _mm256_set1_epi32(min);
        }
    }
    
    return idx;
}
```

这个版本运行在约 10 GFLOPS。要消除其他障碍，我们可以做两件事：

- 将块大小增加到 32 个元素，以允许更多的指令级并行。
- 优化局部 argmin：与其精确计算它的位置，不如只保存该块的索引，最后再回过头来一次性找到它。这样我们只需在每次正向检查时计算最小值并将其广播到一个向量中，这更简单也快得多。

实现了这两项优化后，性能提升到了惊人的约 22 GFLOPS：

```c++
int argmin(int *a, int n) {
    int min = INT_MAX, idx = 0;
    
    reg p = _mm256_set1_epi32(min);

    for (int i = 0; i < n; i += 32) {
        reg y1 = _mm256_load_si256((reg*) &a[i]);
        reg y2 = _mm256_load_si256((reg*) &a[i + 8]);
        reg y3 = _mm256_load_si256((reg*) &a[i + 16]);
        reg y4 = _mm256_load_si256((reg*) &a[i + 24]);
        y1 = _mm256_min_epi32(y1, y2);
        y3 = _mm256_min_epi32(y3, y4);
        y1 = _mm256_min_epi32(y1, y3);
        reg mask = _mm256_cmpgt_epi32(p, y1);
        if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
            idx = i;
            for (int j = i; j < i + 32; j++)
                min = (a[j] < min ? a[j] : min);
            p = _mm256_set1_epi32(min);
        }
    }

    for (int i = idx; i < idx + 31; i++)
        if (a[i] == min)
            return i;
    
    return idx + 31;
}
```

这几乎已经达到了上限，因为仅计算最小值本身也只能运行在约 24–25 GFLOPS。

所有这些"喜欢分支"的 SIMD 实现唯一的问题是，它们都依赖于最小值极少被更新的假设。这对于随机输入分布是成立的，但在最坏情况下不成立。如果我们用递减数列填充数组，最后这个实现的性能会下降到约 2.7 GFLOPS——几乎慢了 10 倍（尽管仍然比标量代码快，因为我们只在每个块上计算最小值）。

一种修复方法是做类似快速排序的随机化算法所做的事：自己打乱输入并以随机顺序遍历数组。这可以让你避免这种最坏情况的惩罚，但由于与随机数生成和[[预取.md|内存]]相关的问题，实现起来很棘手。还有一个更简单的办法。

### 先求最小值，再求索引

我们知道如何[[归约.md|快速计算数组的最小值]]，也知道如何[[掩码与混合.md#掩码|在数组中快速查找元素]]——那么我们为什么不干脆分别算出最小值再找出它呢？

```c++
int argmin(int *a, int n) {
    int needle = min(a, n);
    int idx = find(a, n, needle);
    return idx;
}
```

如果我们最优地实现这两个子程序（参见相关链接文章），对于随机数组性能约为 18 GFLOPS，对于递减数组约为 12 GFLOPS——这是合理的，因为我们预计分别要读取数组 1.5 次和 2 次。这本身并不算太差——至少我们避免了 10 倍的最坏情况性能惩罚——但问题在于，当数组更大、性能受限于[[内存带宽.md|内存带宽]]而非计算时，这种受惩罚的性能也会延续到更大的数组上。

幸运的是，我们已经知道如何修复它。我们可以把数组切分为固定大小 $B$ 的块，并在这些块上计算最小值，同时维护全局最小值。当某个新块上的最小值低于全局最小值时，我们就更新它，并记住当前全局最小值所在的块编号。在遍历完整个数组之后，我们只需要回到那个块，扫描其中的 $B$ 个元素来找到 argmin。

这样我们只处理了 $(N + B)$ 个元素，既不必牺牲一半也不必牺牲三分之一的性能：

```c++
const int B = 256;

// returns the minimum and its first block
pair<int, int> approx_argmin(int *a, int n) {
    int res = INT_MAX, idx = 0;
    for (int i = 0; i < n; i += B) {
        int val = min(a + i, B);
        if (val < res) {
            res = val;
            idx = i;
        }
    }
    return {res, idx};
}

int argmin(int *a, int n) {
    auto [needle, base] = approx_argmin(a, n);
    int idx = find(a + base, B, needle);
    return base + idx;
}
```

最终实现对随机数组和递减数组的结果分别约为 22 和约 19 GFLOPS。

完整的实现，包括 `min()` 和 `find()`，大约有 100 行。如果你想看，可以[点击这里](https://github.com/sslotin/amh-code/blob/main/argmin/combined.cc)，不过它离生产级还差得很远。

### 总结

下面是全部实现结果的汇总：

```
algorithm    rand   decr   reason for the performance difference
-----------  -----  -----  -------------------------------------------------------------
std          0.28   0.28   
scalar       1.54   1.89   efficient branch prediction
+ hinted     1.95   0.75   wrong hint
index        8.17   8.12
simd         8.51   1.65   scalar-based argmin on each iteration
+ ilp        10.22  1.74   ^ same
+ optimized  22.44  2.70   ^ same, but faster because there are less inter-dependencies
min+find     18.21  12.92  find() has to scan the entire array
+ blocked    22.23  19.29  we still have an optional horizontal minimum every B elements
```

对这些结果要持保留态度：测量值[[如何获得准确结果.md|相当嘈杂]]，它们只针对两种输入分布、特定的数组大小（$N=2^{13}$，即 L1 缓存的大小）、特定的架构（Zen 2）以及特定且略显过时的编译器（GCC 9.3）完成——编译器优化对于基准测试代码的小改动也非常脆弱。

还有一些次要的优化点，但潜在提升小于 10%，所以我懒得去做了。也许有一天我会鼓起勇气，把这个算法优化到理论极限，处理不能被块大小整除的数组尺寸以及对齐内存的情况，然后在多种架构上正确地重新运行基准测试，附带 p 值之类的东西。如果有人在我之前做到了，请[回我一声](http://sereja.me/)。

### 致谢

第一个基于索引的 SIMD 算法由 Wojciech Muła 于 2018 年[最初设计](http://0x80.pl/notesen/2018-10-03-simd-index-of-min.html)。

感谢 Zach Wegner[指出](https://twitter.com/zwegner/status/1491520929138151425)，当使用内建函数手动实现时，Muła 算法的性能会有所提升（我原本使用的是[[内建函数与向量类型.md#指令参考|GCC 向量类型]]）。

<!--

Thanks to Alexander Monakov for [being meticulous](https://twitter.com/_monoid/status/1491827976438231049) and pushing me to investigate the STL version.

-->

在发布之后，我发现 [BQN](https://mlochbaum.github.io/BQN/) 的创造者 [Marshall Lochbaum](https://www.aplwiki.com/wiki/Marshall_Lochbaum)，在 2019 年从事 Dyalog APL 工作时，设计了一种[非常相似的算法](https://forums.dyalog.com/viewtopic.php?f=13&t=1579&sid=e2cbd69817a17a6e7b1f76c677b1f69e#p6239)。请多多关注数组编程语言的世界！
