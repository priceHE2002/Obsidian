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

# 4.LSTM

长短期记忆网络（Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的RNN存在的长期依赖问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。

三个门（遗忘门，输入门，输出门），两个状态（Ct,ht）

![[00_基础知识/深度学习面试题-图像算法/assets/image-29.png]]

遗忘门

![[00_基础知识/深度学习面试题-图像算法/assets/image-30.png]]

作用对象：细胞状态 。

作用：将细胞状态中的信息选择性的遗忘。

Ft和Ct-1做点积操作，Ft确保Ct-1有哪些东西需要被遗忘调

输入层门

![[00_基础知识/深度学习面试题-图像算法/assets/image-31.png]]

作用对象：细胞状态

作用：将新的信息选择性的记录到细胞状态中。

操作步骤：

步骤一:sigmoid 层称 “输入门层” 决定什么值我们将要更新

步骤二，tanh 层创建一个新的候选值向量加入到状态中

输出层门

![[00_基础知识/深度学习面试题-图像算法/assets/image-32.png]]

作用对象：隐层ht 作用：确定输出什么值。

操作步骤：

步骤一:通过sigmoid 层来确定细胞状态的哪个部分将输出。

步骤二:把细胞状态通过 tanh 进行处理，并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。

## 4.1 LSTM结构推导，为什么比RNN好？

推导forget gate，input gate，cell state， hidden information等的变化；因为LSTM有进有出且当前的cell informaton是通过input gate控制之后叠加的，RNN是叠乘，因此LSTM可以防止梯度消失或者爆炸。

## 4.2为什么LSTM模型中既存在sigmoid又存在tanh两种激活函数，而不是选择统一一种sigmoid或者tanh？

sigmoid用在了各种gate上，产生0~1之间的值，一般只有sigmoid最直接了；

tanh用在了状态和输出上，是对数据的处理，这个用其他激活函数或许也可以。

## 4.3 LSTM中为什么经常是两层双向LSTM？

有些时候预测需要由前面若干输入和后面若干输入共同决定，这样会更加准确。

## 4.4 RNN扩展改进

### 4.4.1 Bidirectional RNNs

将两层RNNs叠加在一起，当前时刻输出(第t步的输出)不仅仅与之前序列有关，还与之后序列有关。例如：为了预测一个语句中的缺失词语，就需要该词汇的上下文信息。Bidirectional RNNs是一个相对较简单的RNNs，是由两个RNNs上下叠加在一起组成的。输出由前向RNNs和后向RNNs共同决定。

### 4.4.2 CNN-LSTMs

该模型中，CNN用于提取对象特征，LSTMs用于预测。CNN由于卷积特性，其能够快速而且准确地捕捉对象特征。LSTMs的优点：能够捕捉数据间的长时依赖性。

### 4.4.3 Bidirectional LSTMs

有两层LSTMs。 一层处理过去的训练信息，另一层处理将来的训练信息。

通过前向LSTMs获得前向隐藏状态，后向LSTMs获得后向隐藏状态，当前隐藏状态是前向隐藏状态与后向隐藏状态的组合。

### 4.4.4 GRU

（14年提出）是一般的RNNs的变型版本，其主要是从以下两个方面进行改进。

1.以语句为例，序列中不同单词处的数据对当前隐藏层状态的影响不同，越前面的影响越小，即每个之前状态对当前的影响进行了距离加权，距离越远，权值越小。

2.在产生误差error时，其可能是由之前某一个或者几个单词共同造成，所以应当对对应的单词weight进行更新。GRUs的结构如下图所示。GRUs首先根据当前输入单词向量word vector以及前一个隐藏层状态hidden state计算出update gate和reset gate。再根据reset gate、当前word vector以及前一个hidden state计算新的记忆单元内容(new memory content)。当reset gate为1的时候，new memory content忽略之前所有memory content，最终的memory是由之前的hidden state与new memory content一起决定。

![[00_基础知识/深度学习面试题-图像算法/assets/image-33.jpeg]]

## 4.5 LSTM、RNN、GRU区别？

![[00_基础知识/深度学习面试题-图像算法/assets/image-34.png]]

与LSTM相比，GRU内部少了一个”门控“，参数比LSTM少，但是却也能够达到与LSTM相当的功能。考虑到硬件的计算能力和时间成本，因而很多时候我们也就会选择更加实用的GRU。

## 4.6 LSTM是如何实现长短期记忆功能的？

## 4.7 LSTM的原理、写LSTM的公式、手推LSTM的梯度反向传播
