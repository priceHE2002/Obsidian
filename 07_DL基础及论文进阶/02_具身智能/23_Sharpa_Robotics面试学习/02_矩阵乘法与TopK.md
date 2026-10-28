---
title: "矩阵乘法与TopK"
created: 2026-10-28
tags:
  - 面试/Sharpa
  - 具身智能
source_conversation: "6ae06c53-5b60-83e8-a1a2-78acf1bc1002"
answer_provenance: "本次重建，非原答案逐字恢复"
---

# 矩阵乘法与 Top-K

> 【本次重建答案】原题来自 S1、S3；推导、代码和变式为本次学习整理。代码采用 Python 标准库，缓存讨论以 C/C++ 连续行主序数据为前提。

## Q01【截图题目 S1/S3】矩阵乘法为什么 ijk 访问不友好，怎样改成 ikj？

### 思路分析

先以定义写对结果，再根据存储布局调整循环。设 A 为 m×n、B 为 n×p，p 对应截图中的 k，避免与循环下标混用。结果 C 为 m×p：

$$C_{ij}=\sum_{r=0}^{n-1}A_{ir}B_{rj}.$$

行主序存储时，同一行相邻元素连续，固定列遍历不连续。优化的关键是让最内层 j 连续访问 B 的一行和 C 的一行。

### 严格推导

以元素占 w 字节的连续 B 为例：

$$\operatorname{addr}(B_{rj})=B_0+(rp+j)w.$$

ijk（此处写作 i-j-r）最内层 r 每增加 1，地址增加 pw；ikj（i-r-j）最内层 j 每增加 1，地址增加 w。后者更有利于空间局部性，还可把 A[i,r] 保存在寄存器中复用。

固定 i,j 时，两种次序都把 r=0,…,n−1 的乘积累加到同一 C[i,j]，因此在精确算术中结果相同。浮点实现仍可能受编译器重排、融合乘加等影响，不能一概要求任意优化后逐位相同。

~~~python
def matmul_ijk(a, b):
    m, n = len(a), len(a[0])
    assert len(b) == n and n > 0
    p = len(b[0])
    assert all(len(row) == n for row in a)
    assert all(len(row) == p for row in b)
    c = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for r in range(n):
                c[i][j] += a[i][r] * b[r][j]
    return c


def matmul_ikj(a, b):
    m, n = len(a), len(a[0])
    assert len(b) == n and n > 0
    p = len(b[0])
    assert all(len(row) == n for row in a)
    assert all(len(row) == p for row in b)
    c = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for r in range(n):
            value = a[i][r]
            for j in range(p):
                c[i][j] += value * b[r][j]
    return c
~~~

接口前提：a、b 是非空规则矩阵；若支持空矩阵应另定返回规则。Python 的列表套列表不等于 C 的单块连续二维数组；上述 Python 代码演示循环与正确性，不据此声称实际跑得比 NumPy/BLAS 快。

时间复杂度均为 Θ(mnp)，输出空间 Θ(mp)，除输出外辅助空间 Θ(1)。改变循环顺序改善访存方式，不降低乘加次数的渐进量级。

### 结论

$$\boxed{\text{行主序下，ikj 使 B、C 的内层访问连续；复杂度仍为 }\Theta(mnp).}$$

### 【学习扩展】转置、分块与 408 口径

转置 B 后可对 Bᵀ[j] 连续做点积，付出 Θ(np) 预处理和额外存储；适合 B 重复使用。分块让 A/B/C 小块留在缓存，块尺寸须结合缓存容量、相联度与数据类型，不能机械选固定值。SIMD、线程并行和 BLAS 是进一步工程方法，需实测。

408 中 Cache 行大小按数据块口径；Tag、有效位等不计入题给 Cache 行大小。若行大小 64 B、每个 double 为 8 B，则一行装 8 个 double；实际缺失数还受起始对齐、映射、冲突、替换和写策略影响，不能仅凭连续访问断言全命中。

## Q02【截图题目 S1】大数组取最大的 10 个元素，怎样选择算法？

### 思路分析

先问：需要有序结果吗？能改输入吗？是否流式？重复值按元素还是按不同数值计数？令输入长 N、K=10，下述保留重复元素，默认输出降序。

### 严格推导

| 方法 | 时间 | 额外空间 | 适用条件 |
|---|---|---|---|
| 完整排序后截取 | O(N log N) | 取决于排序实现 | 最省实现成本，同时需要全序 |
| 大小 K 的最小堆 | O(N log K)，结果排序再 O(K log K) | O(K) | 流式、不能保存全部数据 |
| 随机 Quickselect | 期望 O(N)，最坏 O(N²) | 原地迭代可 O(1) | 可改输入，只要求分离 Top-K |
| Quickselect 后排序 K 项 | 期望 O(N+K log K) | 取决于实现 | 离线，要求 Top-K 有序 |
| 有序长度 K 数组 | O(NK) | O(K) | K 很小，可作为简单候选并实测 |

最小堆的根是“当前保留的 K 个数中最小者”，不是全局最大者。

不变式：处理前 t 个输入后，堆保存其中最大的 min(t,K) 个元素。初始为空满足。若堆未满，加入当前元素；若满且 x≤堆顶，已有 K 个保留值不小于 x，无需替换；若 x>堆顶，删除最小保留值再插入 x，仍留下最大的 K 个。因此归纳可得最终结果正确。

~~~python
import heapq

def topk_largest(values, k):
    if k < 0:
        raise ValueError('k must be nonnegative')
    if k == 0:
        return []
    heap = []
    for x in values:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)
~~~

若 K>N，返回全部 N 个元素；若只要不同数值，必须额外设计去重逻辑。若输入含 NaN，需先定义它的排序语义。

### 结论

$$\boxed{\text{流式 Top-K 用大小 K 的最小堆；离线可考虑选择算法；先明确输出要求。}}$$

### 【学习扩展】面试官说“能不能更简单”怎么办？

K=10 时可维护长度至多 10 的降序数组，扫描插入 O(K)，总 O(NK)，固定 K 下关于 N 为线性；堆是 O(N log K)，K 固定时也关于 N 为线性，但常数不同。deque 本身不提供任意 Top-K 所需的顺序维护，不能只说容器名字而不解释不变式。单调队列通常用于滑动窗口最值，是另一问题。

如果问“第 K 大”而非“前 K 大”，选择算法只需返回一个秩统计量；若要前 K 个有序，还需排序选出部分。不能把 Quickselect 的期望线性写成无条件最坏线性。


---
返回 [[00_Sharpa面试学习导航]] · 来源 [[01_题目来源与原图]]
