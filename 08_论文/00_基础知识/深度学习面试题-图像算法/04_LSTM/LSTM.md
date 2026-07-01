---
title: LSTM
tags:
  - 基础知识
  - 深度学习
  - 图像算法
  - 面试题
source: "深度学习面试题-图像算法 (1).doc"
created: 2026-07-01
up: "[[00_基础知识/深度学习面试题-图像算法/深度学习面试题-图像算法|深度学习面试题-图像算法]]"
---

# 4. LSTM

长短期记忆网络（Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的 RNN 存在的长期依赖问题而专门设计出来的，所有的 RNN 都具有一种重复神经网络模块的链式形式。

三个门（遗忘门、输入门、输出门），两个状态（细胞状态 $C_t$、隐藏状态 $h_t$）。面试时可以用一句话概括：LSTM 用门控机制控制"忘掉什么、写入什么、输出什么"，让梯度可以沿细胞状态更稳定地传播。

![[00_基础知识/深度学习面试题-图像算法/assets/image-29.png]]

### 遗忘门

![[00_基础知识/深度学习面试题-图像算法/assets/image-30.png]]

- 作用对象：细胞状态。
- 作用：将细胞状态中的信息选择性的遗忘。
- $f_t$ 和 $C_{t-1}$ 做逐元素乘法，$f_t$ 越接近 0 表示越应该遗忘，越接近 1 表示越应该保留。

### 输入门

![[00_基础知识/深度学习面试题-图像算法/assets/image-31.png]]

- 作用对象：细胞状态。
- 作用：将新的信息选择性的记录到细胞状态中。

操作步骤：

- 步骤一：sigmoid 层称"输入门层"，决定什么值我们将要更新。
- 步骤二：tanh 层创建一个新的候选值向量加入到状态中。

### 输出门

![[00_基础知识/深度学习面试题-图像算法/assets/image-32.png]]

- 作用对象：隐层 $h_t$。
- 作用：确定输出什么值。

操作步骤：

- 步骤一：通过 sigmoid 层来确定细胞状态的哪个部分将输出。
- 步骤二：把细胞状态通过 tanh 进行处理，并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。

## 4.1 LSTM 结构推导，为什么比 RNN 好？

普通 RNN 的状态更新通常是：

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

长序列反向传播时，梯度要反复乘以 $W_h$ 和激活函数导数，容易消失或爆炸。

LSTM 的核心改进是引入细胞状态：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

其中 $f_t$ 控制保留多少旧记忆，$i_t$ 控制写入多少新信息。由于 $C_t$ 中有一条近似线性的加法路径，梯度可以更直接地沿 $C_t$ 传播，因此比普通 RNN 更适合学习长期依赖。需要注意：LSTM 能缓解梯度消失，但不是绝对避免，仍可能需要梯度裁剪、归一化和合理初始化。

## 4.2 为什么 LSTM 模型中既存在 Sigmoid 又存在 Tanh 两种激活函数？

- Sigmoid 用在各种 gate 上，产生 $0 \sim 1$ 之间的门控系数，适合表示"通过多少信息"。
- Tanh 用在候选记忆和输出上，把内容压到 $(-1,1)$，适合表示带符号的信息强弱。

简单记忆：Sigmoid 决定"开关比例"，Tanh 生成"具体内容"。

```python
import torch.nn as nn

# LSTM 示例
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2,
               batch_first=True, bidirectional=True)
x = torch.randn(4, 5, 10)   # [batch, seq_len, input_size]
out, (h_n, c_n) = lstm(x)    # out: [4,5,40] (双向*20), h_n: [4,4,20]
print(f"输出形状: {out.shape}, 隐状态: {h_n.shape}, 细胞状态: {c_n.shape}")
```

## 4.3 LSTM 中为什么经常是两层双向 LSTM？

有些任务的当前位置标签需要同时依赖前文和后文，例如分词、命名实体识别、语音帧标注等。双向 LSTM 可以同时获得左侧上下文和右侧上下文，两层堆叠则增强表达能力。但如果是在线预测或生成任务，未来信息不可用，就不能使用双向结构。

## 4.4 RNN 扩展改进

RNN 的主要扩展方向有两类：一类是改变信息流方向，例如双向 RNN/LSTM；另一类是把 CNN、注意力机制或门控结构结合进来，提高特征提取能力和长程依赖建模能力。

### 4.4.1 Bidirectional RNNs

将两层 RNN 叠加在一起，当前时刻输出（第 $t$ 步的输出）不仅仅与之前序列有关，还与之后序列有关。例如：为了预测一个语句中的缺失词语，就需要该词汇的上下文信息。Bidirectional RNNs 是一个相对较简单的 RNN 变体，是由两个 RNN 上下叠加在一起组成的，输出由前向 RNN 和后向 RNN 共同决定。

### 4.4.2 CNN-LSTMs

该模型中，CNN 用于提取对象特征，LSTM 用于预测。CNN 由于卷积特性，其能够快速而且准确地捕捉对象特征。LSTM 的优点：能够捕捉数据间的长时依赖性。

### 4.4.3 Bidirectional LSTMs

有两层 LSTM：一层处理过去的训练信息，另一层处理将来的训练信息。

通过前向 LSTM 获得前向隐藏状态，后向 LSTM 获得后向隐藏状态，当前隐藏状态是前向隐藏状态与后向隐藏状态的组合。

### 4.4.4 GRU

（2014 年提出）是一般 RNN 的变型版本，其主要是从以下两个方面进行改进：

1. 以语句为例，序列中不同单词处的数据对当前隐藏层状态的影响不同，越前面的影响越小，即每个之前状态对当前的影响进行了距离加权，距离越远，权值越小。

2. 在产生误差 error 时，其可能是由之前某一个或者几个单词共同造成，所以应当对对应的单词 weight 进行更新。GRU 首先根据当前输入单词向量 word vector 以及前一个隐藏层状态 hidden state 计算出 update gate 和 reset gate。再根据 reset gate、当前 word vector 以及前一个 hidden state 计算新的记忆单元内容（new memory content）。当 reset gate 为 1 的时候，new memory content 忽略之前所有 memory content，最终的 memory 是由之前的 hidden state 与 new memory content 一起决定。

![[00_基础知识/深度学习面试题-图像算法/assets/image-33.jpeg]]

## 4.5 LSTM、RNN、GRU 区别？

![[00_基础知识/深度学习面试题-图像算法/assets/image-34.png]]

与 LSTM 相比，GRU 将输入门和遗忘门合并为更新门，并用重置门控制历史状态参与候选状态的程度。GRU 没有显式的细胞状态，参数更少、训练更快；LSTM 表达能力更强、更细致，但计算更重。实践中二者效果常接近，数据量小或追求速度时可以优先尝试 GRU。

```python
import torch.nn as nn

# GRU 示例
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
x = torch.randn(4, 5, 10)   # [batch, seq_len, input_size]
out, h_n = gru(x)            # out: [4,5,20], h_n: [1,4,20]
print(f"GRU 输出形状: {out.shape}, 参数更少")
```

## 4.6 LSTM 是如何实现长短期记忆功能的？

LSTM 通过细胞状态 $C_t$ 保存长期记忆，通过隐藏状态 $h_t$ 输出当前时刻对外可见的信息：

1. 遗忘门决定从旧记忆 $C_{t-1}$ 中保留多少：

$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$

2. 输入门决定当前输入写入多少，新候选记忆由 tanh 生成：

$$
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
$$

$$
\tilde{C}_t = \tanh(W_C[h_{t-1}, x_t] + b_C)
$$

3. 更新细胞状态：

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

4. 输出门决定从细胞状态中暴露多少到隐藏状态：

$$
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

其中 $\odot$ 表示逐元素乘法。长期记忆主要保存在 $C_t$ 中，短期输出主要体现在 $h_t$ 中。

## 4.7 LSTM 的原理、公式、手推梯度反向传播

LSTM 的前向公式可以统一写成：

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1}, x_t] + b_f) \\
i_t &= \sigma(W_i[h_{t-1}, x_t] + b_i) \\
\tilde{C}_t &= \tanh(W_C[h_{t-1}, x_t] + b_C) \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \\
o_t &= \sigma(W_o[h_{t-1}, x_t] + b_o) \\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

反向传播的关键不是死记所有矩阵求导，而是看清梯度流向：

1. 损失对 $h_t$ 的梯度会先传到输出门 $o_t$ 和细胞状态 $C_t$。
2. $C_t$ 的梯度会沿两条路传播：一条传到 $C_{t-1}$，系数是 $f_t$；另一条传到 $f_t$、$i_t$ 和 $\tilde{C}_t$。
3. 各个门的梯度再经过 sigmoid/tanh 的导数，传回对应的权重矩阵 $W_f, W_i, W_C, W_o$。
4. 由于 $\partial C_t / \partial C_{t-1} = f_t$，当遗忘门接近 1 时，梯度可以较稳定地向前传播，这就是 LSTM 缓解长期依赖问题的核心。

面试中如果要求"手推"，建议先写出上面的前向公式，再围绕 $C_t$ 的加法路径说明梯度为什么比普通 RNN 更稳定。
