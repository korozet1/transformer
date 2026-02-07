# ⚡ Transformer-PyTorch-Deep-Dive (源码深度解析)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)
![Architecture](https://img.shields.io/badge/Architecture-Encoder--Decoder-success)
![Status](https://img.shields.io/badge/Status-Research_&_Education-purple)

> **"What I cannot create, I do not understand."** — Richard Feynman

## 📖 项目背景 (Introduction)

本项目是 Google 经典论文 **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** 的 **PyTorch 原生复现与源码级深度解析**。

不同于直接调用 `torch.nn.Transformer` API，本项目**手动构建了 Transformer 的每一个组件**（Embedding, Positional Encoding, Multi-Head Attention, LayerNorm, FeedForward），旨在从底层 Tensor 变换的角度，彻底解构大语言模型（LLM）的基石。

代码核心逻辑参考自 **黑马程序员 (Heima Programmer)** 课程实现，在此基础上，我进行了**架构重构、详细中文注释注入以及数学原理对应**，使其更适合作为深度学习进阶的学习资料。

---

## 📐 架构与源码深度映射 (Architecture & Code Mapping)

本项目将 Transformer 的抽象架构图拆解为具体的代码实现。下图展示了论文中的经典架构，下表详细说明了每个模块对应的源码文件位置。

<div align="center">
  <img src="https://machinelearningmastery.com/wp-content/uploads/2018/10/The-Transformer-Model-Architecture.png" alt="Transformer Architecture" width="500" />
  <br>
  <em>Figure 1: The Transformer Architecture (Vaswani et al., 2017)</em>
</div>

<br>

### 📂 源码文件深度拆解 (Source Code Manifest)

为了方便从零构建 Transformer，本项目将架构拆解为 9 个独立的 Python 模块。以下是每个文件的详细说明及对应的架构位置：

#### 1. `dm01_input.py` —— 输入层 (The Foundation)
* **架构位置**: 对应图中最底部的 **Input Embedding** 和 **Positional Encoding**（粉色区域）。
* **核心类**:
    * `Embeddings`: 将输入的 Token ID 转换为 512 维的向量 ($d_{model}$)。
    * `PositionEncoding`: 实现 $\sin / \cos$ 频率位置编码。
* **关键点**: 这里实现了 `x = x + pe(x)`，即位置信息是**相加**到词向量上的，而不是拼接。

#### 2. `dm02_encoder_element.py` —— 核心组件库 (The Bricks)
* **架构位置**: 对应图中橙色的 **Multi-Head Attention** 模块。
* **核心类**:
    * `MultiHeadAttention`: 实现了 Transformer 的灵魂。包含 $Q, K, V$ 的线性变换、多头切分 (Split Heads)、缩放点积 (Scaled Dot-Product) 和最终拼接。
    * `attention()`: 独立的函数，计算 $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。
    * `subsequent_mask`: 生成**上三角矩阵**掩码，用于 Decoder 屏蔽未来信息。

#### 3. `dm03_encoder_sublayer.py` —— 连接组件 (The Glue)
* **架构位置**: 对应图中的 **Add & Norm**（黄色方块）。
* **核心类**:
    * `SublayerConnection`: 封装了 **残差连接 (Residual Connection)** 和 **层归一化 (LayerNorm)**。
    * `LayerNorm`: 实现了 $y = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$。
* **作用**: 这是构建深层网络不退化的关键，所有的 Attention 和 FeedForward 层都包裹在这个组件中。

#### 4. `dm04_encoder_layer.py` —— 编码器层 (Encoder Layer)
* **架构位置**: 对应图中左侧灰色的 **Encoder Block**（其中的一层）。
* **核心类**:
    * `EncoderLayer`: 组装了两个子层：
        1.  Self-Attention (自注意力)
        2.  Feed Forward (前馈网络)
* **逻辑**: 输入 -> Attention -> Add&Norm -> FeedForward -> Add&Norm -> 输出。

#### 5. `dm05_encoder.py` —— 编码器堆叠 (The Encoder)
* **架构位置**: 对应图中左侧的 **Nx**（堆叠 6 层）。
* **核心类**:
    * `Encoder`: 负责将 `EncoderLayer` 克隆 $N$ 次（默认 6 层）。
* **作用**: 提取输入序列的深层语义特征，最终输出 **Context Vector**（也就是 K 和 V），供解码器使用。

#### 6. `dm06_decoder_layer.py` —— 解码器层 (Decoder Layer)
* **架构位置**: 对应图中右侧灰色的 **Decoder Block**（其中的一层）。
* **核心类**:
    * `DecoderLayer`: **这是最复杂的模块**，它比 Encoder 多了一个子层。
        1.  **Masked Self-Attention**: 只能看过去的词。
        2.  **Cross-Attention (Src-Attn)**: **重点！** 这里 $Q$ 来自解码器，$K, V$ 来自编码器的输出。
        3.  Feed Forward。

#### 7. `dm07_decoder.py` —— 解码器堆叠 (The Decoder)
* **架构位置**: 对应图中右侧的 **Nx**（堆叠 6 层）。
* **核心类**:
    * `Decoder`: 负责将 `DecoderLayer` 克隆 $N$ 次。
* **作用**: 一步步生成预测结果。它接收 Encoder 的输出和已经生成的 Token，预测下一个 Token。

#### 8. `dm08_output.py` —— 输出生成器 (The Generator)
* **架构位置**: 对应图中右上角的 **Linear** 和 **Softmax**。
* **核心类**:
    * `Generator`: 一个全连接层，将 Decoder 的输出维度（512维）映射回词表大小（如 10000维），然后接 `log_softmax`。
* **作用**: 输出每个词的概率，概率最大的就是预测结果。

#### 9. `dm09_transformer.py` —— 总装车间 (Assembly)
* **架构位置**: 包含了整张图。
* **核心类**:
    * `EncoderDecoder`: 整个模型的容器。
    * `make_model()`: 工厂函数，负责实例化上述所有组件，初始化参数（如 Xavier 初始化），并构建完整的 Transformer 对象。

---

## 🧠 核心技术深度解析 (Technical Deep Dive)

### 1. 位置编码 (Positional Encoding)
由于 Transformer 完全基于注意力机制，不具备 RNN 的时序归纳偏置，因此必须显式注入位置信息。本项目实现了论文中的**正弦/余弦频率编码**：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

### 2. 缩放点积注意力 (Scaled Dot-Product Attention)
这是 Transformer 的核心引擎。为了防止 $d_k$ (维度) 过大导致点积结果推向 Softmax 的饱和区（梯度消失），引入了缩放因子 $\sqrt{d_k}$。

$$Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

### 3. 多头注意力 (Multi-Head Attention)
通过将模型投影到不同的子空间（Subspaces），让模型能够同时关注不同位置的特征信息。

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$

---

## 🛠️ 快速开始 (Quick Start)

### 环境依赖
* Python 3.8+
* PyTorch 1.10+
* NumPy

### 运行完整模型测试
执行主程序，观察 Tensor 在 Encoder 和 Decoder 中的流动与形状变化：

```bash
python dm09_transformer.py
```

**预期输出**:
```text
my_transformer: EncoderDecoder(...)
result是Transformer的输出结果: tensor(...)
result的(形状): torch.Size([2, 4, 1000]) 
# [Batch_Size, Seq_Len, Vocab_Size]
```

---

## 📝 个人研读笔记 (Study Notes)

在复现过程中，我对以下几个架构细节有了更深的理解：

1.  **Why LayerNorm?** Transformer 选择了 LayerNorm 而非 BatchNorm，因为 NLP 数据的 Seq_Len 往往不一致，BN 在变长序列上表现不佳。代码在每个 `SublayerConnection` 中都使用了 LN (`dm03_encoder_sublayer.py`)。

2.  **Decoder 的 "Shifted Right"** 训练时 Decoder 的输入需要向右移动一位（即 `<Start>` 符号起始），这是为了配合 Mask 机制，确保模型是在“预测”下一个词，而不是“看到”了下一个词。

3.  **Cross-Attention 的交互逻辑** 在 `dm06_decoder_layer.py` 中，可以清晰看到 Decoder 的 `Q` 来自自身，而 `K, V` 来自 Encoder 的输出。这本质上是一个**Query-Retrieval**（查询-检索）过程。

---

## 🤝 致谢 & 引用

* **Original Paper**: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
* **Code Reference**: 感谢 **黑马程序员** 提供的基础教学代码，为本项目的深度分析提供了良好的起步框架。

---

**Author**: [korozet1](https://github.com/korozet1)  
**Role**: CS Graduate Student & AI Researcher  
**Focus**: Computer Vision (YOLO) & NLP (Transformer/LLM)
