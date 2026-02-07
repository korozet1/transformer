"""
案例:
    演示输入部分的 嵌入层 和 位置编码介绍.

总结(回顾):
    输入部分由2部分组成, 分别是:
        词嵌入层(Word Embedding)
        位置编码(Positional Encoding)
"""

# 导包
import torch
import torch.nn as nn   # neural network: 神经网络
import numpy as np
import math
import matplotlib.pyplot as plt

# todo 1.定义函数, 实现: 输入部分之 -> 词嵌入层
class Embeddings(nn.Module):   # 叫Embeddings的目的是为了和Python的类名做区分, 实际开发写: Embedding
    # 1. 初始化函数
    # 参1: 词汇表大小(去重后单词的总个数), 参2: 词嵌入的维度
    def __init__(self, vocab_size, d_model):
        # 1.1 初始化父类信息.
        super().__init__()
        # 1.2 定义变量, 接收: 词汇表大小, 词嵌入的维度
        self.vocab_size = vocab_size
        self.d_model = d_model

        # '欢迎来广州' -> {0: '欢迎', 1: '来', 2: '广州'} ->  把0(单词索引)转成 [值1, 值2, 值3, 值4...] 词向量形式
        # 1.3 定义词嵌入层, 将单词索引映射为词向量.
        self.embed = nn.Embedding(vocab_size, d_model)

    # 2. 定义前向传播方法
    def forward(self, x):
        # 将输入的单词索引映射为词向量, 并乘以 根号d_model进行缩放.
        # 缩放的目的: 为了平衡梯度, 避免梯度消失或梯度爆炸.
        return self.embed(x) * math.sqrt(self.d_model)

# todo 2.测试Embeddings(词嵌入层)
def use_embedding():
    # 1. 定义变量, 记录: 词表大小(1000), 词嵌入维度(512).
    vocab_size, d_model = 1000, 512
    # 2. 实例化Embeddings类.
    my_embed = Embeddings(vocab_size, d_model)
    # 3. 创建张量, 包含2个句子, 每个句子4个词.
    x = torch.tensor([
        # ['我', '爱', '吃', '猪脚饭'],   # 单词
        # ['你', '爱', '吃', '螺蛳粉']
        [100, 2, 421, 600],             # 单词索引
        [500, 888, 3, 615]
    ])
    # 4. 计算嵌入结果.
    result = my_embed(x)
    # 5. 打印结果.
    print(f'result: {result.shape}, {result}')  # [2, 4, 512]


# todo 3. 定义函数, 实现: 输入部分之 -> 位置编码层
class PositionEncoding(nn.Module):
    # 1. 初始化函数.
    # 参1: 词向量的维度(512), 参2: 随机失活概率. 参3: 最大句子长度.
    def __init__(self, d_model, dropout, max_len=60):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 定义dropout层, 防止过拟合.
        self.dropout = nn.Dropout(p=dropout)
        # 1.3 定义pe, 用于保存位置编码结果. 形状: [max_len, d_model] -> [60, 512]
        pe = torch.zeros(max_len, d_model)
        # 1.4 定义一个位置列向量, 从0 到 max_len - 1
        # 形状改变:             [60]        ->  [60, 1]
        position = torch.arange(0, max_len).unsqueeze(1)    # 形状: [60, 1]

        # 1.5 定义1个转换(变化矩阵), 本质是公式里的: 1 / 10000^(2i/d_model)
        # 公式推导: 10000^(2i/d_model) = e^((2i/d_model) * ln(10000))
        # 1/上述内容, 所以求倒数: e^((2i/d_model) * -ln(10000)) -> e^(2i * -ln(10000)/d_model)

        # torch.arange(0, d_model, 2) -> [0, 2, 4, 6, 8.....510]  偶数维度
        # [0, 2, 4, 6, 8.....510] + 1 -> [1, 3, 5, 7, 9.....511]  奇数维度
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 形状: [1, 256]

        # 1.6 计算三角函数里面的值.
        # position形状: [max_len, 1] -> [60, 1]
        # div_term形状: [1, 256]
        # position * div_term = [60, 256]
        position_value = position * div_term

        # 1.7 进行pe的赋值, 偶数位置使用 正弦函数(sin)
        # pe形状: [60, 512], position_value形状: [60, 256]
        pe[:, 0::2] = torch.sin(position_value)    # 形状: [60, 256]

        # 1.8 进行pe的赋值, 奇数位置使用 余弦函数(cos)
        pe[:, 1::2] = torch.cos(position_value)    # 形状: [60, 256]

        # 1.9 将pe进行升维, 增加1个批次维度.
        pe = pe.unsqueeze(0)    #  [1, 60, 512]

        # 1.10 将pe注册到模型的缓冲区, 利用它, 但是不更新它的参数.
        # 回顾: sin(α + β) = sin(α)cos(β) + cos(α)sin(β), cos(α + β) = cos(α)cos(β) - sin(α)sin(β)
        # 带入: sin(5) = sin(3 + 2) = ....
        # pe会作为模型的一部分, 在模型保存, 加载的时候会被处理, 而在模型训练时它的值不会被优化器更新(因为: 位置编码是固定规则)
        self.register_buffer('pe', pe)


    # 2. 前向传播.
    def forward(self, x):
        # 1. 这段代码是位置编码的核心逻辑, 负责把 '词向量' 和 '位置编码' 融合(相加)
        # 参数x: 词向量, 形状为: [batch_size, seq_len, d_model] -> [1, 60, 512]
        # self.pe 的形状: [1, max_len, d_model] -> 假设: [1, 1000, 512]
        x = x + self.pe[:, :x.size(1)]     # [1, 60, 512] + [1, 60, 512]


        # 2. 随机失活, 不改变形状, 形状还是: [batch_size, seq_len, d_model] -> [1, 60, 512]
        return self.dropout(x)


# todo 4. 测试 Positional_Encoding(位置编码层)
def use_position():
    # 1. 定义词汇表大小 和 词嵌入维度.
    vocab_size = 1000
    d_model = 512

    # 2. 实例化Embeddings层.
    my_embed = Embeddings(vocab_size, d_model)      # [1000, 512]
    # 3. 创建输入张量, 形状: [2, 4] -> 2个句子, 每个句子4个单词
    x = torch.tensor([
        [100, 2, 421, 600],  # 单词索引
        [500, 888, 3, 615]
    ])
    # 4. 计算词嵌入结果.
    embed_x = my_embed(x)   # 形状: [2, 4, 512] -> [batch_size, seq_len, d_model]

    # 5. 实例化PositionEncoding层.
    my_position = PositionEncoding(d_model, dropout=0.1)
    # 6. 计算位置编码结果.
    position_x = my_position(embed_x)   # [2, 4, 512]

    # 7. 返回结果.
    return position_x


# todo 5. 测试plot_Positional(可视化位置编码)
def plot_position():
    # 1. 实例化位置编码器.
    # 参1: 词嵌入维度(维度小方便画图看规律), 参2: 随机失活概率, 参3: 最大序列长度.
    my_position = PositionEncoding(20, dropout=0, max_len=100)

    # 2. 生成全0的输入, 观察位置编码的模式.
    # (1, 100, 20) -> 批次大小, 句子长度, 词嵌入维度.
    y = my_position(torch.zeros(1, 100, 20))

    # 3. 打印输出形状.
    print(f'y.shape: {y.shape}')    # [1, 100, 20]

    # 4. 设置图表大小.
    plt.figure(figsize=(20, 15))
    # 5. 绘制位置编码第4到第7列, 100个词的[4, 5, 6, 7]列
    plt.plot(np.arange(100), y[0, :, 4:8].detach().numpy())
    # 6. 添加图例.
    plt.legend([f'dim {p}' for p in [4, 5, 6, 7]])
    # 7. 显示图表.
    plt.show()


if __name__ == '__main__':
    # 1. 测试词嵌入.
    # use_embedding()

    # 2. 测试位置编码.
    # result = use_position()
    # print(f'result: {result.shape}, {result}')

    # 3. 测试位置编码的可视化.
    plot_position()