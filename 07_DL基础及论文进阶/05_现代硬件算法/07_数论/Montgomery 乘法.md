---
tags:
  - 现代硬件算法
  - 数论
created: 2026-09-11
source: https://en.algorithmica.org/hpc/number-theory/montgomery/
original_title: Montgomery Multiplication
---

# Montgomery 乘法

毫不意外，[[模运算.md|模运算]] 中有很大一部分计算常常耗费在求模运算上，它的速度与 [[整数除法.md|通用整数除法]] 一样慢，通常根据操作数大小需要 15–20 个周期。

应对这一麻烦的最佳方法是彻底避免求模，用 [[无分支编程.md|谓词（predication）]] 来延迟或替换它，例如在计算模和时即可做到：

```cpp
const int M = 1e9 + 7;

// input: array of n integers in the [0, M) range
// output: sum modulo M
int slow_sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++)
        s = (s + a[i]) % M;
    return s;
}

int fast_sum(int *a, int n) {
    int s = 0;
    for (int i = 0; i < n; i++) {
        s += a[i]; // s < 2 * M
        s = (s >= M ? s - M : s); // will be replaced with cmov
    }
    return s;
}

int faster_sum(int *a, int n) {
    long long s = 0; // 64-bit integer to handle overflow
    for (int i = 0; i < n; i++)
        s += a[i]; // will be vectorized
    return s % M;
}
```

然而，有时你面对的只是一连串模乘法，除了使用需要常量模与某些预计算的 [[整数除法.md|整数除法技巧]] 外，并无好办法逃避计算除法的余数。

但还有一种专为模运算设计的技术，称为 *Montgomery 乘法*（Montgomery multiplication）。

### Montgomery 空间

Montgomery 乘法首先将乘数变换到 *Montgomery 空间*（Montgomery space），在该空间中模乘法可以廉价地完成，待需要其实际值时再变换回来。与通用整数除法方法不同，Montgomery 乘法仅执行一次模约简时并不高效，只有在存在一连串模运算时才值得使用。

该空间由模 $n$ 和一个与 $n$ 互素的、满足 $r \ge n$ 的正整数 $r$ 定义。算法涉及模 $r$ 与除以 $r$ 的操作，因此在实践中 $r$ 取为 $2^{32}$ 或 $2^{64}$，从而这两个操作可分别通过右移与按位与完成。

<!-- Therefore $n$ needs to be an odd number so that every power of $2$ will be coprime to $n$. And if it is not, we can make it odd (?). -->

**定义。** 数 $x$ 在 Montgomery 空间中的*代表元*（representative）$\bar x$ 定义为

$$
\bar{x} = x \cdot r \bmod n
$$

计算这一变换涉及一次乘法与一次取模——而这正是我们原本想优化的昂贵操作——因此，只有当数字进出 Montgomery 空间的变换开销物有所值时，我们才使用该方法，而不用于通用模乘法。

<!-- Note that the transformation is actually such a multiplication that we want to optimize, so it is still an expensive operation. However, we will only need to transform a number into the space once, perform as many operations as we want efficiently in that space and at the end transform the final result back, which should be profitable if we are doing lots of operations modulo $n$. -->

在 Montgomery 空间内部，加法、减法与相等性检查照常进行：

$$
x \cdot r + y \cdot r \equiv (x + y) \cdot r \bmod n
$$

但乘法并非如此。将 Montgomery 空间中的乘法记为 $*$，"普通"乘法记为 $\cdot$，我们期望的结果是：

$$
\bar{x} * \bar{y} = \overline{x \cdot y} = (x \cdot y) \cdot r \bmod n
$$

但 Montgomery 空间中的普通乘法给出：

$$
\bar{x} \cdot \bar{y} = (x \cdot y) \cdot r \cdot r \bmod n
$$

因此，Montgomery 空间中的乘法定义为

$$
\bar{x} * \bar{y} = \bar{x} \cdot \bar{y} \cdot r^{-1} \bmod n
$$

这意味着，在 Montgomery 空间中将两个普通数相乘后，我们需要将结果乘以 $r^{-1}$ 并取模来*约简*（reduce）——而这一特定操作有一种高效的方法。

### Montgomery 约简

假设 $r=2^{32}$，模 $n$ 为 32 位，而需要约简的数 $x$ 为 64 位（两个 32 位数之积）。我们的目标是计算 $y = x \cdot r^{-1} \bmod n$。

由于 $r$ 与 $n$ 互素，可知存在 $[0, n)$ 范围内的两个数 $r^{-1}$ 与 $n^\prime$ 满足

$$
r \cdot r^{-1} + n \cdot n^\prime = 1
$$

二者均可例如通过 [[扩展欧几里得算法.md|扩展欧几里得算法]] 计算。

利用这一恒等式，可将 $r \cdot r^{-1}$ 表示为 $(1 - n \cdot n^\prime)$，并将 $x \cdot r^{-1}$ 写为

$$
\begin{aligned}
x \cdot r^{-1} &= x \cdot r \cdot r^{-1} / r
\\             &= x \cdot (1 - n \cdot n^{\prime}) / r
\\             &= (x - x \cdot n \cdot n^{\prime}    ) / r
\\             &\equiv (x - x \cdot n \cdot n^{\prime} + k \cdot r \cdot n) / r &\pmod n &\;\;\text{(for any integer $k$)}
\\             &\equiv (x - (x \cdot n^{\prime} - k \cdot r) \cdot n) / r &\pmod n
\end{aligned}
$$

现在，若取 $k = \lfloor x \cdot n^\prime / r \rfloor$（$x \cdot n^\prime$ 乘积的高 64 位），它将被消去，而 $(k \cdot r - x \cdot n^{\prime})$ 将恰好等于 $x \cdot n^{\prime} \bmod r$（$x \cdot n^\prime$ 的低 32 位），于是：

$$
x \cdot r^{-1} \equiv (x - x \cdot n^{\prime} \bmod r \cdot n) / r
$$

算法本身只是直接求解这个等式，执行两次乘法以计算 $q = x \cdot n^{\prime} \bmod r$ 与 $m = q \cdot n$，再从 $x$ 中减去它，并将结果右移以除以 $r$。

唯一还要处理的是结果可能不在 $[0, n)$ 范围内；但由于

$$
x < n \cdot n < r \cdot n \implies x / r < n
$$

且

$$
m = q \cdot n < r \cdot n \implies m / r < n
$$

可保证

$$
-n < (x - m) / r < n
$$

因此，我们只需检查结果是否为负，若是则加上 $n$，得到如下算法：

```c++
typedef __uint32_t u32;
typedef __uint64_t u64;

const u32 n = 1e9 + 7, nr = inverse(n, 1ull << 32);

u32 reduce(u64 x) {
    u32 q = u32(x) * nr;      // q = x * n' mod r
    u64 m = (u64) q * n;      // m = q * n
    u32 y = (x - m) >> 32;    // y = (x - m) / r
    return x < m ? y + n : y; // if y < 0, add n to make it be in the [0, n) range
}
```

最后一次检查相对廉价，但仍位于关键路径上。如果我们能接受结果落在 $[0, 2 \cdot n - 2]$ 而非 $[0, n)$ 范围内，便可移除它，无条件地将 $n$ 加到结果上：

```c++
u32 reduce(u64 x) {
    u32 q = u32(x) * nr;
    u64 m = (u64) q * n;
    u32 y = (x - m) >> 32;
    return y + n
}
```

我们还可以在计算图中将 `>> 32` 操作提前一步，计算 $\lfloor x / r \rfloor - \lfloor m / r \rfloor$ 而非 $(x - m) / r$。这之所以正确，是因为 $x$ 与 $m$ 的低 32 位相同，毕竟

$$
m = x \cdot n^\prime \cdot n \equiv x \pmod r
$$

但我们为何要主动选择做两次右移而非一次？这样做有益，因为对于 `((u64) q * n) >> 32`，我们需要做一次 32×32 乘法并取结果的高 32 位（x86 的 `mul` 指令 [[整数.md#整数类型|本就会]] 将其写入一个独立的寄存器，因此不花费额外代价），而另一次右移 `x >> 32` 不在关键路径上。

```c++
u32 reduce(u64 x) {
    u32 q = u32(x) * nr;
    u32 m = ((u64) q * n) >> 32;
    return (x >> 32) + n - m;
}
```

Montgomery 乘法相对于其他模约简方法的一大优势在于，它不需要很大的数据类型：只需一次 $r \times r$ 乘法来提取结果的高、低 $r$ 位，[大多数硬件对此有特殊支持](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=7395,7392,7269,4868,7269,7269,1820,1835,6385,5051,4909,4918,5051,7269,6423,7410,150,2138,1829,1944,3009,1029,7077,519,5183,4462,4490,1944,5055,5012,5055&techs=AVX,AVX2&text=mul) 也使其易于推广到 [[SIMD并行.md|SIMD]] 与更大的数据类型：

```c++
typedef __uint128_t u128;

u64 reduce(u128 x) const {
    u64 q = u64(x) * nr;
    u64 m = ((u128) q * n) >> 64;
    return (x >> 64) + n - m;
}
```

注意，通用整数除法技巧无法实现 128×64 的取模：编译器会[回退](https://godbolt.org/z/fbEE4v4qr) 到调用一个缓慢的[大数算术库函数](https://github.com/llvm-mirror/compiler-rt/blob/69445f095c22aac2388f939bedebf224a6efcdaf/lib/builtins/udivmodti4.c#L22) 来支持它。

### 更快的逆元与变换

Montgomery 乘法本身很快，但它需要一些预计算：

- 求 $n$ 模 $r$ 的逆以计算 $n^\prime$，
- 将数字变换*到* Montgomery 空间，
- 将数字变换*出* Montgomery 空间。

最后一个操作已由我们刚实现的 `reduce` 过程高效完成，但前两个可稍作优化。

**计算逆元** $n^\prime = n^{-1} \bmod r$ 可以比扩展欧几里得算法更快：利用 $r$ 是 2 的幂这一事实，并使用如下恒等式：

$$
a \cdot x \equiv 1 \bmod 2^k
\implies
a \cdot x \cdot (2 - a \cdot x)
\equiv
1 \bmod 2^{2k}
$$

证明：

$$
\begin{aligned}
a \cdot x \cdot (2 - a \cdot x)
   &= 2 \cdot a \cdot x - (a \cdot x)^2
\\ &= 2 \cdot (1 + m \cdot 2^k) - (1 + m \cdot 2^k)^2
\\ &= 2 + 2 \cdot m \cdot 2^k - 1 - 2 \cdot m \cdot 2^k - m^2 \cdot 2^{2k}
\\ &= 1 - m^2 \cdot 2^{2k}
\\ &\equiv 1 \bmod 2^{2k}.
\end{aligned}
$$

我们可以从 $x = 1$（即 $a$ 模 $2^1$ 的逆）出发，将该恒等式恰好应用 $\log_2 r$ 次，每次使逆元的位数翻倍——这有点类似于 [[牛顿法.md|牛顿法]]（Newton's method）。

**变换** 一个数到 Montgomery 空间，可以通过将其乘以 $r$ 再按 [[整数除法.md|通常方式]] 取模完成，但我们也可以利用如下关系：

$$
\bar{x} = x \cdot r \bmod n = x * r^2
$$

将一个数变换到该空间不过是乘以 $r^2$。因此我们可以预计算 $r^2 \bmod n$，转而执行一次乘法与约简——这实际是否更快还不一定，因为将数乘以 $r=2^{k}$ 可用左移实现，而乘以 $r^2 \bmod n$ 则不能。

### 完整实现

将所有内容封装进一个 `constexpr` 结构会很方便：

```c++
struct Montgomery {
    u32 n, nr;
    
    constexpr Montgomery(u32 n) : n(n), nr(1) {
        // log(2^32) = 5
        for (int i = 0; i < 5; i++)
            nr *= 2 - n * nr;
    }

    u32 reduce(u64 x) const {
        u32 q = u32(x) * nr;
        u32 m = ((u64) q * n) >> 32;
        return (x >> 32) + n - m;
        // returns a number in the [0, 2 * n - 2] range
        // (add a "x < n ? x : x - n" type of check if you need a proper modulo)
    }

    u32 multiply(u32 x, u32 y) const {
        return reduce((u64) x * y);
    }

    u32 transform(u32 x) const {
        return (u64(x) << 32) % n;
        // can also be implemented as multiply(x, r^2 mod n)
    }
};
```

为测试其性能，我们可以将 Montgomery 乘法接入 [[快速幂.md|快速幂]]：

```c++
constexpr Montgomery space(M);

int inverse(int _a) {
    u64 a = space.transform(_a);
    u64 r = space.transform(1);
    
    #pragma GCC unroll(30)
    for (int l = 0; l < 30; l++) {
        if ( (M - 2) >> l & 1 )
            r = space.multiply(r, a);
        a = space.multiply(a, a);
    }

    return space.reduce(r);
}
```

使用编译器生成的快速取模技巧的普通快速幂每次 `inverse` 调用约需 170ns，而本实现约需 166ns；若省略 `transform` 与 `reduce`，则降至约 158ns（一个合理的用例是 `inverse` 作为更大规模模运算的子过程）。改进虽小，但对于 SIMD 应用与更大的数据类型，Montgomery 乘法优势会明显得多。

**练习。** 实现高效的*模* [[矩阵乘法.md|矩阵乘法]]（matix multiplication）。
