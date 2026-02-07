# ⚡ Transformer-PyTorch-Deep-Dive (源码深度解析)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)
![Architecture](https://img.shields.io/badge/Architecture-Encoder--Decoder-success)
![Status](https://img.shields.io/badge/Status-Research_&_Education-purple)

> **"What I cannot create, I do not understand."** — Richard Feynman

## 📖 项目背景 (Introduction)

本项目是 Google 经典论文 **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** 的 **PyTorch 原生复现与源码级深度解析**。

不同于直接调用 `torch.nn.Transformer` API，本项目**手动构建了 Transformer 的每一个组件**（Embedding, Positional Encoding, Multi-Head Attention, LayerNorm, FeedForward），旨在从底层 Tensor 变换的角度，彻底解构大语言模型（LLM）的基石。

代码核心逻辑基于 **黑马程序员** 课程实现，在此基础上，我进行了**架构重构、详细注释注入以及数学原理对应**，使其更适合作为深度学习进阶的学习资料。

---

## 🏗️ 项目结构 (Directory Structure)

```text
Transformer-Deep-Dive/
├── dm01_input.py           # [Input] 词嵌入(Embedding) + 位置编码(PE)
├── dm02_encoder_element.py # [Core] 多头注意力(MHA) + 掩码生成(Mask) + 缩放点积
├── dm03_encoder_sublayer.py# [Block] 残差连接(Residual) + 层归一化(LayerNorm)
├── dm04_encoder_layer.py   # [Layer] 单层 Encoder 组装
├── dm05_encoder.py         # [Module] 完整 Encoder 堆叠 (N=6)
├── dm06_decoder_layer.py   # [Layer] 单层 Decoder (含 Cross-Attention)
├── dm07_decoder.py         # [Module] 完整 Decoder 堆叠 (N=6)
├── dm08_output.py          # [Output] 线性层 + Softmax 生成概率
└── dm09_transformer.py     # [Main]  Transformer 整体架构组装与测试
```

---

## 🧠 核心技术深度解析 (Technical Deep Dive)

### 1. 位置编码 (Positional Encoding)
由于 Transformer 完全基于注意力机制，不具备 RNN 的时序归纳偏置，因此必须显式注入位置信息。本项目实现了论文中的**正弦/余弦频率编码**：

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

* **代码对应**: `dm01_input.py` -> `PositionEncoding` 类
* **关键实现**: 使用 `div_term` 计算频率衰减，通过 `register_buffer` 将 PE 注册为非参数常量。

### 2. 缩放点积注意力 (Scaled Dot-Product Attention)
这是 Transformer 的核心引擎。为了防止 $d_k$ (维度) 过大导致点积结果推向 Softmax 的饱和区（梯度消失），引入了缩放因子 $\sqrt{d_k}$。

$$Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

* **代码对应**: `dm02_encoder_element.py` -> `attention()` 函数
* **Mask 机制**: 在计算 Softmax 之前，使用 `masked_fill(mask == 0, -1e9)` 将需要掩盖的位置（如 Padding 或 Decoder 的未来信息）置为负无穷。

### 3. 多头注意力 (Multi-Head Attention)
通过将模型投影到不同的子空间（Subspaces），让模型能够同时关注不同位置的特征信息。

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$
$$\text{where } head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

* **代码对应**: `dm02_encoder_element.py` -> `MultiHeadAttention` 类
* **实现细节**: 采用了 `view` 和 `transpose` 操作实现 Heads 的并行计算，而非循环实现，极大提升了训练效率。

### 4. 掩码策略 (Masking Strategy)
本项目实现了两种关键掩码：
* **Padding Mask**: 用于处理变长序列，忽略 `<pad>` token。
* **Subsequent Mask (Look-ahead Mask)**: 用于 Decoder 训练阶段。通过**上三角矩阵**强制模型只能利用当前及之前的 Token 进行预测，保持自回归（Auto-regressive）特性。

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
**Profile**: CS Graduate Student | CV & NLP Researcher
