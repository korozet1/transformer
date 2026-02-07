"""
案例:
    演示Transformer框架中的 编码器的 两个子层代码实现.

    编码器子层1:
        多头注意力层(Multi-Head Attention) + 残差连接(Add) + 层规范化(Norm)

    编码器子层2:
        前向传播层(Feed Forward) + 残差连接(Add) + 层规范化(Norm)
"""

# 导包
from dm01_input import *
from dm02_encoder_element import *


# todo 1. 初始化子层连接结构, 核心是: 残差连接 ＋ 层规范化, 让模型训练更稳定, 避免梯度消失和梯度爆炸
class SublayerConnection(nn.Module):
    # 1. 初始化函数
    # 参1: 输入的维度(词向量的维度)
    # 参2: 随机失活的概率
    def __init__(self, d_model, dropout=0.1):
        # 1. 初始化父类
        super().__init__()
        # 2. 定义规范化层(LayerNorm)
        self.norm = LayerNorm(d_model)
        # 3. 定义随机失活层(Dropout)
        self.dropout = nn.Dropout(dropout)

    # 2. 前向传播函数
    # 参1: x -> 输入张量, 形状一般是: [batch_size, seq_len, d_model] -> [2, 4, 512]
    # 参2: sublayer -> 子层对象, 例如: 多头注意力层(Multi-Head Attention)的对象, 前馈全连接层(Feed Forward)的对象
    def forward(self, x, sublayer):
        # 核心逻辑: 两种常见的实现方式
        # 方式1(大多场景用这个): 先子层处理, 再残差连接, 层规范化
        #            随机失活      层规划范    子层处理        残差连接
        my_result = self.dropout(self.norm(sublayer(x))) + x

        # 方式2: 先层规范化, 再子层处理, 残差连接
        # 感兴趣自己写, 代码一样, 就顺序不同.

        # 返回结果
        return my_result


# todo 2. 测试子层连接结构.
def use_sublayer():
    # 1. 准备输入数据(词向量 + 位置编码)
    x = use_position()
    # 2. 创建子层连接对象
    sublayer_conn = SublayerConnection(512)

    # 3. 定义子层对象 -> 传入该对象, 充当: 子层处理, 可以是: 多头注意力层(Multi-Head Attention), 前馈全连接层(Feed Forward)
    # 思路1: 采用函数嵌套(闭包写法)实现, 定义子层函数 -> 一个可调用的对象, 接收x, 返回处理结果.
    def sublayer(x):
        # 3.1 创建 多头注意力层(Multi-Head Attention)对象
        multi_attn = MultiHeadAttention(embed_dim=512, head=8)
        # 3.2 计算 注意力, 并返回.
        # 因为是自注意力机制, 所以: Q=K=V=x
        return multi_attn(x, x, x)

    # 4. 通过子层连接结构 处理输入.
    # result = sublayer_conn(x, sublayer)

    # 思路2: 用匿名函数实现.
    # 多头注意力
    # result = sublayer_conn(x, lambda x: MultiHeadAttention(embed_dim=512, head=8)(x, x, x))
    # 前馈全连接层
    result = sublayer_conn(x, lambda x: FeedForward(512, 2048)(x))

    # 5. 打印子层处理结果.
    print(f'result.shape: {result.shape}')  # [2, 4, 512]


# todo 3. 主函数, 程序入口
if __name__ == '__main__':
    use_sublayer()