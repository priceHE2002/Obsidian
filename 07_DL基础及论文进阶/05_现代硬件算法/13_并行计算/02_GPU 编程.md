---
tags:
  - 现代硬件算法
  - 并行计算
created: 2026-09-11
source: https://en.algorithmica.org/hpc/parallel/_index.en/
original_title: GPU Programming
---

# GPU 编程

这是一个以 HTML 形式渲染的 Jupyter 笔记本。如果你想在这里直接做练习，可以在 [Colab]() 中打开它，或者[下载]()后在本地编辑。在前一种情况下，你需要完成一个小任务：安装 CUDA 以及它的 Python 绑定 PyCuda。在基于 Debian 的机器上，下面这些命令大概就足够了：
* `apt-get install nvidia-cuda-dev nvidia-cuda-toolkit`
* `pip install pycuda`

前置知识：基本的 Python 与 C 知识、基础算法，以及关于计算机大致工作原理的了解。

## 摩尔定律的微妙之处（Subtleties of the Moore's law）

下面这张图大致描绘了 CPU 领域正在发生的变化：

<img width='600px' src='https://www.karlrupp.net/wp-content/uploads/2015/06/35years.png'>

**摩尔定律**指出，微处理器中的晶体管数量大约每两年翻一番。这大致意味着性能也翻一番。

你可以看到，大约在 2005 年，设计思路发生了一次转变。

这些核心或多或少是相互独立的。

现代 GPU 出现于 21 世纪初。它们利用了自身所运行的特定领域。

一个核心的速度存在物理极限。

一个确凿的极限是：光速。你至少需要电磁波（光也属于电磁波）从主板一端传到另一端所需的时间。

其中一些核心具有

Google Colab 上默认可免费使用的 GPU 相当[强劲](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)。作者也不知道 Google 为何这么做，但这很棒。

## 为何要多处理（Why multiprocessing?）

时钟频率——例如，Intel Core i7 可以拥有……这就给出了一个上界

有两种类型的

## 通用 GPU（General-purpose GPU）

曾有一段时间，对冲基金从游戏公司高薪聘请计算机图形学人才，看中的正是他们的计算能力。

有好几种。

这就像 Windows 与 Linux 的关系。

我们将采用 CUDA，因为它更为普及，尤其是在那些无人关心这件事的领域，比如深度学习。

## 异构计算（Heterogeneous computing）

CUDA 编程涉及让代码在两个不同的平台上并发运行：一个带有一个或多个 CPU 的主机（host）系统，以及一个或多个 GPU 的设备（device）。

## 与 CPU 的差异

### 线程（Threads）

CPU 上的线程通常是重量级实体。操作系统必须在 CPU 执行通道之间换入换出线程，以提供多线程能力。因此上下文切换既缓慢又昂贵。

相比之下，GPU 上的线程极为轻量。在典型的系统中，成千上万个线程排队等待工作——以每 32 个线程为一个*束（warp）*。如果 GPU 必须等待某一个线程束，它会立即开始在另一个线程束上执行。由于所有活跃线程都分配有独立的寄存器，在 GPU 线程之间切换时，无需交换寄存器或其他状态。资源会一直分配给每个线程，直到它执行完毕。

简而言之，CPU 核心被设计为同时最小化一两个线程的延迟，而 GPU 被设计为处理大量并发的轻量级线程，以最大化吞吐。

### 内存（Memory）

主机系统与设备各自拥有相互独立、彼此专属的物理内存。由于主机内存与设备内存之间隔着 PCI Express（PCIe）总线，主机内存中的内容有时必须跨总线传送到设备内存，或反向传送，正如"CUDA 设备运行什么？"一节所述。

如果你这样思考，可以轻易丢掉 98% 的性能。

## 安装 PyCUDA

CUDA 可用于多种语言。

不错的文档可在此处找到：https://documen.tician.de/pycuda/index.html

如果你在使用 Colab，请前往 Runtime -> Change runtime type -> Hardware accelerator，并将其设置为 "GPU"。

```python
# you may want to clear the output of this cell after installation
from IPython.display import clear_output
 
# this might take a while
!pip install pycuda

clear_output()
```

```python
import numpy as np

from pycuda.compiler import SourceModule
import pycuda.driver as drv
import pycuda.autoinit
```

## 基础（The basics）

让我们从一个简单的例子开始，然后再深入。

## 内核（Kernels）

就跟 C 或 C++ 一样，只是你要使用一些自定义的內建函数与限定符（specifier）。

CUDA 几乎就像普通的 C，只是你可以将某些函数指定为在设备上运行。根据实现方式，工作流如下：

你需要把你的计算机看作一台异构机器：既有主机数据，也有设备数据。

* 你将输入数据移动到设备内存。
* 你在设备上运行某些计算。
* 你把这些数据取回。

事实上，内核的运行是并发的——你的程序在内核运行完成之前并不会阻塞。较新的设备甚至能以这种方式并发运行多个内核，并等待它们的结果。

## 著名的 $A + B$ 问题

为了测试以及与主机协调，我们将使用 **NumPy** 包。如果你没有安装它，请执行：`pip install numpy`。

NumPy 是 Python 中用于线性代数与数组操作的包。它由 C 编写，非常高效，但只运行在 CPU 上，因此我们将以它为基准进行对照。

```python
# lets generate our test data: two float arrays filled with something random
a = numpy.random.randn(100).astype('float32')
b = numpy.random.randn(100).astype('float32')
# the type needs to be specified in this case, because randn's default type is float64, but CUDA knows nothing about it

# we need to create space where kernel should write its answers to
dest = numpy.zeros_like(a)

# this is the kernel itself
mod = SourceModule("""
    __global__ void add(float *dest, float *a, float *b) {
        const int i = threadIdx.x;
        dest[i] = a[i] + b[i];
    }
""")

# you need to specify the source code, and PyCUDA will compile it
add_kernel = mod.get_function("add")

add_kernel(
    drv.Out(dest),  # specifies that this memory should be accessible for writing
    drv.In(a),  # specifies this should be accessible for reading
    drv.In(b),
    block=(100,1,1)  # we'll talk about it in a minute
)

assert np.allclose(dest, a + b), 'WA'  # checks that these are equal
print('OK')
```

      File "<ipython-input-27-afc857479fe4>", line 19
        %%time
        ^
    SyntaxError: invalid syntax

### 内存管理（Memory management）

在 CUDA C API 中，你需要显式地分配内存。所以这实际上相当不错。

还有 `drv.InOut` 函数，它使内存既可读取也可写入，但本教程不会用到它，因为我们也需要在过程中检验代码。

这里的大部分操作都是内存操作，因此在此处衡量性能毫无意义。别担心，我们很快就会进入更复杂的例子。

GPU 有非常特定的操作。不过，就 NVIDIA GPU 而言，管理它相当简单：这些卡具有*计算能力（compute capabilities）*（1.0、1.1、1.2、1.3、2.0 等），在能力等级 $x$ 中添加的所有特性在更高版本中同样可用。这些可以在运行时或编译时检查。

你可以在这篇维基百科文章中查看差异：https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications

## 同步（Synchronization）

**归约（Reduction）**是任意的数组级操作。

假设有如下的问题：

## 动态规划（Dynamic programming）

考虑如下递推式：

```python
## Problem: dynamic programming
```

## 工作量 vs. 延迟（Work vs. Latency）

我们现在实际上同时考虑工作复杂度与步骤复杂度。

有些任务，尤其是密码学中的任务，无法被并行化。但有些可以。

## 以 $O(\log n)$ 时间对数组求和（Summing arrays in $O(\log n)$ time）

假设我们想对含有 $n$ 个元素的数组执行某种结合律（即 $A*(B*C) = (A*B)*C$）运算。比如，把它们求和。

通常，我们会用一个简单的循环来做：

```c++
float s = 0;
for (int i = 0; i < n; i++) {
     s += a[i]; 
}
```

它的计算图长这样：

<img width='400px' src='https://www.elemarjr.com/wp-content/uploads/2018/03/sequential_sum.png'>

这在工作复杂度上是最优的，但在步骤复杂度上不是：它是 $O(n)$。我们可能想要一种在工作复杂度上稍差、但可以并行化的方案。

让我们试试这种分治思路：

<img width='400px' src='https://www.elemarjr.com/wp-content/uploads/2018/03/parallel_sum.png'>

现在它的工作复杂度仍是 $O(n)$（你实际上需要同样数量的加法），但它的步骤复杂度是 $O(\log n)$。

当你自顶向下展开递归时，你会看到：为了求得每个所需的值，

<img width='400px' src='http://i.stack.imgur.com/Uehc3.png'>

## 归约小数组（Reducing small arrays）

```python
a = numpy.random.randn(2048).astype('float32')

mod = SourceModule("""
    __global__ void sum(float *dest, float *a, float *b) {
        const int i = threadIdx.x;
        // for l from 0 to logn:
        //   __sync_threads()
        //   if the thread is active
        //     sum two elements into where they belong
        // a[0] should containt the needed sum
    }
""")

sum_kernel = mod.get_function("sum")

add_kernel(
    drv.InOut(a),
    block=(1024,1,1)
)

assert np.allclose(dest, a + b), 'WA'  # checks that these are equal
print('OK')
```

## 束与线程块（Warps and thread blocks）

线程以 32 个为一组进行捆绑。组内所有线程必须要么都在等待，要么都在执行相同的操作。这是架构上的困难所致。

<img width='300px' src='https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Block-thread.svg/1920px-Block-thread.svg.png'>

你其实可以用 2D 与 3D 索引做同样的事情——挺奇怪的，对吧？

## 原子操作（Atomics）

## 归约大数组（Reducing big arrays）

## 归约超大数组（Reducing very big arrays）

现在，事情变得困难了。是时候讲清楚 GPU 并行究竟是如何工作的了。

```python

```

## 稠密矩阵乘法（Dense Matrix multiplication）

让我们进入第一个真正用得上 GPU 的例子：矩阵乘法。

## 排序（Sorting）

我们最后（也最难）的任务是实现排序。

你可能已经注意到，我们大多数时候都主张使用分治方法。

确实如此。它们管用。但我们还无法得到一个现成可用的算法。

```python
# we'll use a deep learning library for benchmarking because I'm not familiar with anything else 
import torch

a = torch.randn(10**8)
b = a.cuda()
```

```python
# this should run for ~15 secs
%time c = torch.sort(a)
%time c = torch.sort(b)
```

    CPU times: user 15.2 s, sys: 177 µs, total: 15.2 s
    Wall time: 15.2 s
    CPU times: user 274 ms, sys: 237 ms, total: 511 ms
    Wall time: 511 ms

所以，有 30 倍的加速。这样我们就知道需要与之竞争的目标了。

```python
b.sort()
```

    (tensor([-5.4567, -5.3551, -5.3288,  ...,  5.3529,  5.4484,  5.4486],
            device='cuda:0'),
     tensor([55083205,  8383169, 73705953,  ..., 79814161, 50474932, 27805828],
            device='cuda:0'))

排序算法有两种：数据驱动的。

第二种可以用排序网络（sorting networks）来表示和分析。下面是我们将要使用的那个，它被称为双调排序（bitonic sort）。

<img src='https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/BitonicSort.svg/1686px-BitonicSort.svg.png'>

它有 $O(\log n)$ 个阶段，总共包含 $1 + 2 + 3 + \ldots + \log n = O(\log^2 n$ 个无法并行化、且涉及数组每个元素的比较块。因此，总体而言它具有 $O(n \log^ n)$ 的工作量复杂度，但步骤复杂度为 $O(\log^2 n)$，这相当理想。

它实际上并不难实现。为了说清楚，下面是一个缓慢的递归 Python 实现：

```python
def bitonic_sort(a, up=False):
    if len(a) <= 1:
        return a
    else: 
        l = bitonic_sort(x[:len(a) // 2], True)
        r = bitonic_sort(x[len(a) // 2:], False)
        return bitonic_merge(first + second, up)

def bitonic_merge(a, up): 
    # assume input a is bitonic, and sorted list is returned 
    if len(a) == 1:
        return a
    else:
        bitonic_compare(a, up)
        l = bitonic_merge(a[:len(a) // 2], up)
        r = bitonic_merge(a[len(a) // 2:], up)
        return l + r

def bitonic_compare(a, up):
    dist = len(a) // 2
    for i in range(dist):  
        if (a[i] > a[i + dist]) == up:
            a[i], a[i + dist] = a[i + dist], x[i]  # this is how swap is done in Python
```

```python
bitonic_sort([57, 179, 42, 17, 300, 111])
```

    [300, 179, 111, 57, 42, 17]

```python
a = np.random.randn(10**8).astype('float32')
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-27-58a927c14aae> in <module>()
    ----> 1 a = np.random.randn(10**8).astype('float32')
    

    NameError: name 'np' is not defined

## 为何选择 CUDA（Why CUDA）

其中大部分仍然适用。

再次强调，GPU 编程非常特定化。

SSE 与张量核心（tensor cores）。

## 内核（Kernels）

就跟 C 或 C++ 一样，只是你要使用一些自定义的內建函数与限定符。

CUDA 几乎就像普通的 C，只是你可以将某些函数指定为在设备上运行。根据实现方式，工作流如下：

你需要把你的计算机看作一台异构机器：既有主机数据，也有设备数据。

* 你将输入数据移动到设备内存。
* 你在设备上运行某些计算。
* 你把这些数据取回。

事实上，内核的运行是并发的——你的程序在内核运行完成之前并不会阻塞。较新的设备甚至能以这种方式并发运行多个内核，并等待它们的结果。

你需要理解的关于 GPU 的一点是：它们对其应用领域极度专门化。

为此需要相应的内建函数（intrinsics）。

如今，大量价值来自加密货币与深度学习。后者依赖于两种特定操作：用于线性层的矩阵乘法，以及用于计算机视觉中卷积层的卷积。

首先，它们引入了"乘加（multiply-accumulate）"操作（例如 `x += y * z`），每个 GPU 时钟周期执行一次。

Google 使用张量处理单元（Tensor Processing Units）。没有人真正知道它们如何工作（这是他们出租而非出售的专有硬件）。

每个张量核心对大小为 4x4 的小矩阵执行操作。每个张量核心每个 GPU 时钟可以执行 1 次矩阵乘加操作。它将两个 fp16 的 4x4 矩阵相乘，并把乘积 fp32 矩阵（大小：4x4）加到累加器（同样为 fp32 的 4x4 矩阵）上。

每个时钟周期能完成大量工作。

不过，对于深度学习而言，你其实并不需要比这更精确的精度。

它被称为混合精度（mixed precision），因为输入矩阵是 fp16，而乘法结果与累加器是 fp32 矩阵。

也许，更恰当的名字应该是"4x4 矩阵核心"，但 NVIDIA 的市场团队决定使用"张量核心（tensor cores）"这一叫法。

所以，你看，这并不完全是公平的比较。

<img width='500px' src='https://static.seekingalpha.com/uploads/2018/8/11/275308-15340093003448672_origin.png'>

*<center>你需要把这张图再稍微延伸一点点：去年 11 月，NVIDIA 的股价在比特币崩盘后下跌了 30%，所以我不会那么乐观</center>*

一直到 int4（16 个取值，你没有听错）

你需要了解大量这类专门化的知识，才能写出高效的代码。因此，从零开始编写库是个坏主意。

总之，出于教学与娱乐的目的，今天我们将重新发明轮子，亲手实现一个矩阵乘法。

## 归约一个数组（Reducing an array）

这似乎很简单：你只需要。

当你执行 `s += x` 时，实际发生了什么？这并非单一操作。实际上，发生了四件事：

1. 将 $x$ 读入寄存器
2. 将 $s$ 读入寄存器
3. 计算 $s + x$
4. 把它写回 $s$ 最初所在的位置

两个线程可能以交错的方式执行它。假设线程 A 取到了 $s$，但一纳秒之后线程 B 会向此处写入，而线程 A 并不知道这一点，于是会用未改变的值覆盖写回。

注意：用原子操作（atomics）来做这件事

对于小的数据类型，原子操作是在硬件层面实现的，比这快得多。

`std::atomic` 被引入以处理多线程上下文中的原子操作。在多线程环境中，当两个线程操作同一个变量时，你必须格外小心，以避免竞态条件。

## 内存类型（Memory types）

如果各种类型的设备内存相互竞速，结果会是这样：

寄存器大小（= 机器字宽）为 32 位，但它们也具备 64 位能力（否则无法拥有超过 4GB 的内存）。

* 第 1 名：**寄存器内存（Register memory）**
  <br> 这是只有写入它的那个线程才可见的数据。它仅存在于该线程的生命周期内。
* 第 2 名：**共享内存（Shared Memory）**
  <br> 在线程块内的所有线程之间共享。仅存在于该块的生命周期内。这种内存允许线程之间进行通信（数据共享）。这就是为什么你应当
* 第 3 名：**常量内存（Constant Memory）**
  <br> 
* 第 4 名：纹理内存（Texture Memory）
* 并列最后一名：局部内存（Local Memory）与全局内存（Global Memory）

你现在需要关心的是寄存器

目前，你需要关心不同

访问全局内存需要数百（个周期）。

## 问题：稠密矩阵乘法（Problem: dense matrix multiplication）

其中许多实际上是稀疏的。你可以用社交网络图或网页图来做文章。

很酷。但让我们略微让人失望一下：
