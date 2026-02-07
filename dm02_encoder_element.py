"""
案例:
    演示 编码器中各个部分的功能.

掩码:
    遮掩住词向量中的某些数据值, 目的: 模型预测信息时, 未来的信息不能被提前利用.

上三角矩阵, 下三角矩阵如何查看:
    总结:
        上三角矩阵: 对角线下边全是0, 上边有值.
        下三角矩阵: 对角线上边全是0, 下边有值.

        nn.triu(m, k)   m表示1个矩阵, k表示对角线的起始位置(默认值: 0)
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from dm01_input import *


# todo 1. 测试产生上三角矩阵.
def dm01_test_triu():
    # 测试产生上三角矩阵
    print(np.triu([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4],
                   [5, 5, 5, 5, 5]], k=1))
    print(np.triu([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4],
                   [5, 5, 5, 5, 5]], k=0))  # 默认
    print(np.triu([[1, 1, 1, 1, 1],
                   [2, 2, 2, 2, 2],
                   [3, 3, 3, 3, 3],
                   [4, 4, 4, 4, 4],
                   [5, 5, 5, 5, 5]], k=-1))


# todo 2. 测试产生下三角矩阵
def dm02_test_triu(size):
    # 1. 生成上三角矩阵, m表示1个矩阵, k表示对角线的起始位置(默认值: 0)
    temp = np.triu(m=np.ones((1, size, size)), k=0).astype('uint8')
    # print(temp)

    # 2. 从上三角矩阵, 转成下三角矩阵.
    return torch.from_numpy(1 - temp)


# todo 3. 掩码张量的可视化
def dm03_test_mask():
    # 简单说(总结): 纵向选'当前位置', 横向看'能看到哪些位置' 0(黄色) -> 遮挡,  1 -> 不遮挡
    plt.figure(figsize=(5, 5))
    plt.imshow(dm02_test_triu(20)[0])   # [1, 20, 20] -> [20, 20]
    plt.show()


# todo 4. 定义函数, 进行注意力的计算.
# 参数: query/key/value: 注意力的三个核心输入, 形状通常是: [batch_size, seq_len, d_model]
# mask: 掩码张量(一般是 Decoder解码器中的未来位置要被掩码)
# dropout: 随机失活概率
def attention(query, key, value, mask=None, dropout=None):
    # 1. 求查询张量的特征维度 d_k
    d_k = query.size()[-1]      # 词向量维度.

    # 2. 计算原始注意力分数, 模拟: Q * K^T / sqrt(d_k)
    # key.transpose(-2, -1), 维度从 [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
    # query @ key.transpose(-2, -1) -> [batch_size, seq_len, d_model] @ [batch_size, d_model, seq_len] -> [batch_size, seq_len, seq_len]
    # 除以 sqrt(d_k) -> 缩放操作, 避免内积过大, 导致梯度消失...
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 3. 掩码处理(可选), 遮挡不需要关注的位置.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. 计算注意力权重 和 归一化.
    # dim=-1: 最后1维 -> seq_len_k
    p_attn = F.softmax(scores, dim=-1)

    # 5. 随机失活(可选)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 6. 计算最终注意力输出, 权重加权求和.
    # torch.matmul(p_attn, value):  用注意力权重p_attn 对 value加权求和.
    # p_attn: 注意力权重(用于可视化 或者 调试)
    return torch.matmul(p_attn, value), p_attn


# todo 5. 测试 注意力计算.
def use_attiention():
    # 1. 获取位置编码后的结果. 形状: [2, 4, 512]
    position_x = use_position()  # 调用 dm01_input # dm_position()函数

    # 因为是自注意力机制, 所以: Q= K = V
    query = key = value = position_x    # 形状: [2, 4, 512]

    # ------------------------- 思路1: 没有掩码 -------------------------
    # 2. 没有掩码: 调用 attention()函数即可.
    result1, p_attn = attention(query, key, value)
    # 打印注意力输出的形状
    print(f'result1.shape: {result1.shape}')    # [2, 4, 512]
    # 打印注意力权重 p_attn
    print(f'p_attn: {p_attn.shape}, {p_attn}')  # [2, 4, 4], 每个词和其他词的注意力权重.
    print("-" * 30)


    # ------------------------- 思路2: 有掩码 -------------------------
    # 3. 构建1个全0的掩码张量.
    mask = torch.zeros(2, 4, 4)  # 因为: 注意力权重为: [2, 4, 4]

    # 4. 传入mask(掩码张量), 计算结果.
    result2, p_attn2 = attention(query, key, value, mask)

    # 5. 打印结果.
    print(f'result2.shape: {result2.shape}')       # [2, 4, 512]
    # 打印注意力权重 p_attn
    print(f'p_attn2: {p_attn2.shape}, {p_attn2}')  # [2, 4, 4], 每个词和其他词的注意力权重.


# todo 6. 定义1个克隆函数 -> 克隆指定模块N次, 返回包含N个相同模块的列表.
# 参1: 要被克隆的模块, 例如: Linear(), Conv2d()...
# 参2: 克隆的次数.
def clones(module, N):
    # copy.deepcopy(): 深拷贝, 每个模块拥有独立的参数.
    # nn.ModuleList(): 模块列表, 可以动态增加和删除模块.
    # 应用场景: Transformer中多处需要相同的子层, 例如: 多头注意力机制的多个线性变换层, 编码器和解码器堆叠多个相同结构的层...
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# todo 7. 定义函数, 实现: 多头注意力层 -> 核心思想: 把词向量维度拆分为多个头, 并行计算小维度的注意力, 增强模型表达能力.
class MultiHeadAttention(nn.Module):
    # 1. 初始化函数.
    # 参1: 词向量的维度.   参2: 多头个数.   参3: 随机失活概率.
    def __init__(self, embed_dim, head, dropout_p = 0.1):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 确保能整除, 即: 分头成功.
        assert embed_dim % head == 0
        # 3. 计算每个头的词嵌入维度
        self.d_k = embed_dim // head        # 例如: 512/8 = 64
        self.head = head                    # 例如: 8
        # 4. 定义4个线性层, 前3个 -> 用于Q, K, V的投影,  最后1个 -> 用于输出(多头注意力结果)的投影.
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        # 5. 定义随机失活层.
        self.dropout = nn.Dropout(dropout_p)
        # 6. 保存注意力权重, 用于可视化或者分析.
        self.atten = None


    # 2. 前向传播函数 -> 实现多头注意力计算流程.
    def forward(self, query, key, value, mask=None):
        # 1. 判断是否需要掩码.
        # query, key, value的形状: [batch_size, seq_len, d_model] -> [2, 4, 512]
        # mask的形状: [batch_size, seq_len, seq_len] -> [1, batch_size, seq_len, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(0)

        # 2. 获取batch_size(批量大小)
        self.batch = query.size(0)

        # 3. 线性变换: Q, K, V -> 多头注意力计算.
        # mode(x):   通过线性层将 输入 投影到 embed_dim维度, 即: 512维
        # view(...): 将投影结果重塑为: [batch_size, seq_len, head, d_k] -> [2, 4, 8, 64]
        # transpose(1, 2): 转置, 形状: [batch_size, seq_len, head, d_k] -> [batch_size, head, seq_len, d_k] -> [2, 8, 4, 64]
        query, key, value = [
            model(x).view(self.batch, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]

        # 4. 多头注意力计算.
        # x的形状:     [batch_size, head, seq_len, d_k] -> [2, 8, 4, 64]
        # atten的形状: [batch_size, head, seq_len, seq_len] -> [2, 8, 4, 4]
        x, self.atten = attention(query, key, value, mask, self.dropout)

        # 5. 将多头注意力结果 -> 合并.
        # x的形状: [batch_size, seq_len, head * d_k] -> [2, 4, 512]
        atten_x = x.transpose(1, 2).contiguous().view(self.batch, -1, self.head * self.d_k)

        # 6. 通过最后1个线性层处理, 返回结果.
        return self.linears[-1](atten_x)


# todo 8. 测试编码器 -> 多头注意力机制.
def use_multihead():
    # 1. 创建多头注意力层 对象
    my_attention = MultiHeadAttention(512, 8)
    # 2. 获取位置编码处理后的结果(即: 输入部分的代码)
    position_x = use_position()
    # 因为是自注意力机制, 所以: Q= K = V
    query = key = value = position_x

    # 3. 创建掩码张量, 形状为: [batch_size, seq_len, seq_len]
    mask = torch.zeros(8, 4, 4)
    # 4. 将数据送给模型 -> 得到多头注意力结果.
    result = my_attention(query, key, value, mask)
    # 5. 打印输出形状, 要与输入的形状一致. 即: [2, 4, 512]
    print(f'(多头注意力层)result.shape: {result.shape}')

    # 优化: 返回多头注意力结果, 后续可以直接用.
    return result


# todo 9. 初始化前馈全连接层(它包含2个全连接层), 它是Transformer的一个基础模块, 作用: 对注意力层的输出进一步加工(强化特征的)
class FeedForward(nn.Module):
    # 1. 初始化函数.
    # 参1: 输入数据的维度(词向量的维度),
    # 参2: (前馈神经网络)第1层全连接层的输出维度, 即: 你要把词向量 映射到的 维度, 该值一般比 d_model要大
    # 参3: 随机失活概率
    def __init__(self, d_model, d_ff, dropout_p = 0.1):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 定义第1个全连接层, 输入维度为: d_model, 输出维度为: d_ff
        self.linear1 = nn.Linear(d_model, d_ff) # 例如: 512 -> 2048维
        # 1.3 定义第2个全连接层, 输入维度为: d_ff, 输出维度为: d_model
        self.linear2 = nn.Linear(d_ff, d_model) # 例如: 2048 -> 512维
        # 1.4 创建随机失活层.
        self.dropout = nn.Dropout(dropout_p)

    # 2. 前向传播函数.
    def forward(self, x):
        # 1. 通过第1个全连接层, 映射到 d_ff维.
        x = self.linear1(x)
        # 2. 通过激活函数(ReLu会把负值变为0, 保留正数, 增加了非线性功能), 在结合 随机失活.
        x = self.dropout(F.relu(x))
        # 3. 通过第2个全连接层, 映射到 d_model维.
        x = self.linear2(x)
        # 4. 返回结果.
        return x



# todo 10. 测试前馈全连接层, 先通过多头注意力层拿到输出, 再把这个输出喂给前馈全连接层 -> ...
def use_ff():
    # 1. 获取多头注意力结果.
    attn_x = use_multihead()    # [2, 4, 512]
    # 2. 创建前馈全连接层对象.
    my_ff = FeedForward(512, 2048)
    # 3. 通过前馈全连接层, 获取结果.
    result = my_ff(attn_x)
    # 4. 打印输出形状
    print(f'(前馈全连接层)result.shape: {result.shape}')
    # 5. 反馈结果.
    return result


# todo 11. 定义1个规范化层, 初始化规范化层(Layer Normalization), 用于对张量进行标准化处理, 让模型训练更稳定.
class LayerNorm(nn.Module):
    # 1. 初始化函数.
    # 参1: 词嵌入维度(假设: 512),   参2: 小常数, 防止分母变为零
    def __init__(self, features, eps = 1e-6):
        # 1.1 初始化父类成员.
        super().__init__()

        # 回顾线性公式: y = kx + b ->  ax + b
        # 1.2 定义可学习的缩放系数 a, 初始化值 1, 形状: [features]
        # 作用: 对标准化后的数据进行缩放.
        self.a = nn.Parameter(torch.ones(features))
        # 1.3 定义可学习的偏置(平移)系数 b, 初始化值 0, 形状: [features]
        self.b = nn.Parameter(torch.zeros(features))
        # 1.4 定义小常数, 防止分母变为零.
        self.eps = eps

    # 2. 前向传播函数, 对传入的张量x进行标准化处理.
    def forward(self, x):
        # x的形状: [batch_size, seq_len, d_model] -> [2, 4, 512], 处理后, 维度一致.
        # 1. 计算最后1维(词向量的维度)的均值.
        # 参1: -1表示最后1维, 参数: keepdim=True表示保持维度不变.
        x_mean = x.mean(-1, keepdim = True) # 即: [2, 4, 512] -> [2, 4, 1]

        # 2. 计算最后一维(词向量维度)的标准差
        x_std = x.std(-1, keepdim = True)   # 即: [2, 4, 512] -> [2, 4, 1]

        # 3. 进行标准化处理.
        # 回顾线性公式: y = kx + b ->  ax + b
        return self.a * (x - x_mean) / (x_std + self.eps) + self.b


# todo 12. 测试规范化层, 思路: 先通过多头注意力层得到输出 -> 在通过前馈全连接层得到输出 -> 通过规范化层进行规范化处理.
def user_layernorm():
    # 1. 获取前馈全连接层结果.
    ff_x = use_ff()
    # 2. 实例化规范化层对象.
    my_layernorm = LayerNorm(512)
    # 3. 将前馈全连接层结果, 传给规范化层 -> 获取规范化结果.
    result = my_layernorm(ff_x)
    # 4. 打印输出形状, 要与输入的形状一致. 即: [2, 4, 512]
    print(f'(规范化层)result.shape: {result.shape}')


# 面试题: BN(Batch Normalization) 和 LN(Layer Normalization)的区别.




# 调用函数
if __name__ == '__main__':
    # 1. 测试上三角矩阵
    # dm01_test_triu()

    # 2. 测试下三角矩阵
    # print(dm02_test_triu(5))

    # 3. 测试掩码张量的可视化
    # dm03_test_mask()

    # 4. 测试编码器 -> 注意力计算.
    # use_attiention()

    # 5. 测试编码器 -> 多头注意力机制.
    # use_multihead()

    # 6. 测试前馈全连接层.
    # use_ff()

    # 7. 测试规范化层.
    user_layernorm()