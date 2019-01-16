# TextClassification-Keras

这个代码仓库使用 **Keras** 框架实现了多种用于**文本分类**的**深度学习模型**，其中包含的模型有：**FastText**, **TextCNN**, **TextRNN**, **TextBiRNN**, **TextAttBiRNN**, **HAN**, **RCNN**, **RCNNVariant** 等等。除了模型实现，还附带了简化的应用程序。

- [English documents](README.md)
- [中文文档](README-ZH.md)

## 向导

1. [环境](#环境)
2. [使用说明](#使用说明)
3. [模型](#模型)
    1. [FastText](#1-fasttext)
    2. [TextCNN](#2-textcnn)
    3. [TextRNN](#3-textrnn)
    4. [TextBiRNN](#4-textbirnn)
    5. [TextAttBiRNN](#5-textattbirnn)
    6. [HAN](#6-han)
    7. [RCNN](#7-rcnn)
    8. [RCNNVariant](#8-rcnnvariant)
    999. [未完待续……](#未完待续)
4. [引用](#引用)

## 环境

- Python 3.6
- NumPy 1.15.2
- Keras 2.2.0
- Tensorflow 1.8.0

## 使用说明

代码部分都位于目录 `/model` 下，每种模型有相应的目录，该目录下放置了模型代码和应用代码。

例如：FastText 的模型代码和应用代码都位于 `/model/FastText` 下，模型部分是 `fast_text.py`，应用部分是 `main.py`。

## 模型

### 1 FastText

FastText 在论文 [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf) 中被提出。

#### 1.1 论文的描述

<p align="center">
	<img src="image/FastText.png">
</p>

1.	Using a look-up table, **bags of ngram** covert to **word representations**.
2.	Word representations are **averaged** into a text representation, which is a hidden variable.
3.	Text representation is in turn fed to a **linear classifier**.
4.	Use the **softmax** function to compute the probability distribution over the predefined classes.

#### 1.2 此处的实现

FastText 的网络结构：

<p align="center">
	<img src="image/FastText_network_structure.png">
</p>

### 2 TextCNN

TextCNN 在论文 [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181) 中被提出。

#### 2.1 论文的描述

<p align="center">
	<img src="image/TextCNN.png">
</p>

1. Represent sentence with **static and non-static channels**.
2. **Convolve** with multiple filter widths and feature maps.
3. Use **max-over-time pooling**.
4. Use **fully connected layer** with **dropout** and **softmax** ouput.

#### 2.2 此处的实现

TextCNN 的网络结构：

<p align="center">
	<img src="image/TextCNN_network_structure.png">
</p>

### 3 TextRNN

TextRNN 在论文 [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf) 中有被提到，但并不是这篇论文提出的。

#### 3.1 论文的描述

<p align="center">
	<img src="image/TextRNN.png">
</p>

#### 3.2 此处的实现

TextRNN 的网络结构：

<p align="center">
	<img src="image/TextRNN_network_structure.png">
</p>

### 4 TextBiRNN

TextBiRNN 是基于 TextRNN 的改进版本，将网络结构中的 RNN 层改进成了双向（Bidirectional）的 RNN 层，希望不仅能考虑正向编码的信息，也能考虑反向编码的信息。暂时没有找到相关的论文。

TextBiRNN 的网络结构：

<p align="center">
	<img src="image/TextBiRNN_network_structure.png">
</p>

### 5 TextAttBiRNN

TextAttBiRNN 是基于 TextBiRNN 的改进版本，引入了注意力机制（Attention）。对于双向 RNN 编码得到的表征向量，模型能够通过注意力机制，关注与决策最相关的信息。其中注意力机制最先在论文 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf) 中被提出，而此处对于注意力机制的实现参照了论文 [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf)。

#### 5.1 论文的描述

<p align="center">
	<img src="image/FeedForwardAttention.png">
</p>

In the paper [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf), the **feed forward attention** is simplified as follows,

<p align="center">
	<img src="image/FeedForwardAttetion_fomular.png">
</p>

Function `a`, a learnable function, is recognized as a **feed forward network**. In this formulation, attention can be seen as producing a fixed-length embedding `c` of the input sequence by computing an **adaptive weighted average** of the state sequence `h`.

#### 5.2 此处的实现

Attention 的实现不做介绍，请直接查阅源代码。

TextAttBiRNN 的网络结构：

<p align="center">
	<img src="image/TextAttBiRNN_network_structure.png">
</p>

### 6 HAN

HAN 在论文 [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174) 中被提出。

#### 6.1 论文的描述

<p align="center">
	<img src="image/HAN.png">
</p>

1. **Word Encoder**. Encoding by **bidirectional GRU**, an annotation for a given word is obtained by concatenating the forward hidden state and backward hidden state, which summarizes the information of the whole sentence centered around word in current time step.
2. **Word Attention**. By a one-layer **MLP** and softmax function, it is enable to calculate normalized importance weights over the previous word annotations. Then, compute the sentence vector as a **weighted sum** of the word annotations based on the weights.
3. **Sentence Encoder**. In a similar way with word encoder, use a **bidirectional GRU** to encode the sentences to get an annotation for a sentence.
4. **Sentence Attention**. Similar with word attention, use a one-layer **MLP** and softmax function to get the weights over sentence annotations. Then, calculate a **weighted sum** of the sentence annotations based on the weights to get the document vector.
5. **Document Classification**. Use the **softmax** function to calculate the probability of all classes.

#### 6.2 此处的实现

此处的 Attention 的实现使用了 FeedForwardAttention 的实现方式，与 TextAttBiRNN 中的 Attention 相同。

HAN 的网络结构：

<p align="center">
	<img src="image/HAN_network_structure.png">
</p>

此处使用了 TimeDistributed 包装器，希望 Embedding、Bidirectional RNN 和 Attention 层的参数能够在时间步维度上共享。

### 7 RCNN

RCNN 在论文 [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552) 中被提出。

#### 7.1 论文的描述

<p align="center">
	<img src="image/RCNN.png">
</p>

1. **Word Representation Learning**. RCNN uses a recurrent structure, which is a **bi-directional recurrent neural network**, to capture the contexts. Then, combine the word and its context to present the word. And apply a **linear transformation** together with the `tanh` activation fucntion to the representation.
2. **Text Representation Learning**. When all of the representations of words are calculated, it applys a element-wise **max-pooling** layer in order to capture the most important information throughout the entire text. Finally, do the **linear transformation** and apply the **softmax** function.

#### 7.2 此处的实现

RCNN 的网络结构：

<p align="center">
	<img src="image/RCNN_network_structure.png">
</p>

### 8 RCNNVariant

RCNNVariant 是基于 RCNN 的改进版本，做了以下几点改进。暂时没有找到相关的论文。

1. 三输入改成了**单输入**，移除了左右上下文的输入。
2. 使用**双向的 LSTM/GRU** 取代传统 RNN 进行编码。
3. 使用**多通道的 CNN** 进行语义向量的表征。
4. 使用 **ReLU 激活层**取代 Tanh 激活层。
5. 同时使用 **AveragePooling** 和 **MaxPooling** 进行池化。

RCNNVariant 的网络结构：

<p align="center">
	<img src="image/RCNNVariant_network_structure.png">
</p>

### 未完待续……

## 引用

1. [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
2. [Keras Example IMDB FastText](https://github.com/keras-team/keras/blob/master/examples/imdb_fasttext.py)
3. [Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)
4. [Keras Example IMDB CNN](https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py)
5. [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
6. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
7. [Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems](https://arxiv.org/pdf/1512.08756.pdf)
8. [cbaziotis's Attention](https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d)
9. [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)
10. [Richard's HAN](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/)
11. [Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)
12. [airalcorn2's RCNN](https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier)