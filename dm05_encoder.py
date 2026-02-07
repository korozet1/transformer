"""
案例:
    演示 编码器的代码实现.

总结:
    1. 编码器默认有6个编码器层.
    2. 每个编码器层默认有2个子层
        多头注意力层
        前馈全连接层
    3. 还有 残差连接 + 层规范化
"""

# 导包
from dm04_encoder_layer import *


# todo 1. 定义编码器类.
class Encoder(nn.Module):
    # 1. 初始化函数
    # 参1: 单个编码器层对象,  参2: 编码器层数量
    def __init__(self, layer, N):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 克隆N个 编码器层对象.
        self.layers = clones(layer, N)
        # 1.3 定义最终的规范化层.
        self.norm = LayerNorm(layer.d_model)    # 512

    # 2. 前向传播函数
    # 参1: x输入张量, 维度: [batch_size, seq_len, d_model] -> [2, 4, 512]
    # 参2: mask掩码张量, 维度: [batch_size, seq_len, seq_len] -> [8, 4, 4], 8个头
    def forward(self, x, mask):
        # 2.1 依次通过N个编码器层.
        for layer in self.layers:
            # x通过每一个 编码器层的处理.
            x = layer(x, mask)

        # 2.2 最终规范化. 提升模型的稳定性.
        return self.norm(x)


# todo 2. 测试编码器对象.
def use_encoder():
    # 1. 获取数据(词向量 + 位置编码)
    x = use_position()

    # 2. 构建单个编码器层(作为基础单元)
    # 2.1 多头注意力层的对象
    multi_head = MultiHeadAttention(512, 8)  # 多头注意力层
    # 2.2 前馈全连接层的对象
    feed_forward = FeedForward(512, 2048)  # 前馈全连接层
    # 2.3 把 多头注意力层和 前馈全连接层 组合起来 -> 编码器层对象
    encoder_layer = EncoderLayer(512, multi_head, feed_forward)

    # 3. 实例化编码器(堆叠 3 个编码器层), 论文默认是: 6个
    encoder = Encoder(encoder_layer, 3)
    # 4. 构建掩码张量.
    mask = torch.zeros(8, 4, 4)
    # 5. 执行编码过程.
    encoder_output = encoder(x, mask)

    # 6. 打印结果
    print(f'编码器输出形状: {encoder_output.shape}')  # [2, 4, 512]
    # 7. 可选, 观察输入和输出结果.
    print(f'输入示例: \n{x[0, 0, :5]}')     # 第1个句子的第1个词的前5个词向量.
    print(f'输出示例: \n{encoder_output[0, 0, :5]}')

    # 8. 返回编码器层处理后的结果, 作为解码器的输入, 继续往后执行.
    return encoder_output



use_encoder()