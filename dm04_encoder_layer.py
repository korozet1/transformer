"""
案例:
    自定义代码, 模拟: 编码器层
流程:
    词向量 + 位置编码 -> 多头注意力 -> 残差连接 + 层规范化 -> 前馈神经网络 -> 残差连接 + 层规范化
"""

# 导包
# from dm01_input import *                # 输入部分(词嵌入层 + 位置编码)
# from dm02_encoder_element import *      # 编码器的组件(多头注意力层, 前馈神经网络, 规范化层, 掩码...)
from dm03_encoder_sublayer import *     # 编码器的子层...

# 需求: 所谓的 编码器层就是把刚才的两个自称 合二为一 串起来.

# todo 1. 初始化编码器层.
class EncoderLayer(nn.Module):
    # 1. 初始化函数
    # 参1: 词嵌入维度,  参2: 多头注意力对象,  参3: 前馈神经网络对象, 参4: 随机失活概率
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        # 1.1 初始化父类成员.
        super().__init__()

        # 1.2 保存子层实例.
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 1.3 克隆两个 子层连接结构.
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)


    # 2. 前向传播函数. 完成: 自注意力层 + 前馈神经网络层的特征处理动作, 每层都带 残差连接 和 规范化
    def forward(self, x, mask):
        # 1. 第1层 子层连接: 自注意力层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 2. 第2层 子层连接: 前馈神经网络层
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))

        # 3. 返回结果.
        return x


# todo 2. 测试编码器层的流程.
def use_encoder_layer():
    # 1. 准备数据(词嵌入 + 位置编码)
    x = use_position()
    # 2. 实例化子层组件.
    multi_head = MultiHeadAttention(embed_dim=512, head=8)
    ff = FeedForward(d_model=512,  d_ff=2048)

    # 3. 创建编码器层.
    encoder_layer = EncoderLayer(d_model=512, self_attn=multi_head, feed_forward=ff)

    # 4. 构建掩码张量.  形状: [batch_size, seq_len, seq_len]
    mask = torch.zeros(8, 4, 4)

    # 5. 执行编码器层的 -> 前馈传播
    output = encoder_layer(x, mask)
    # 6. 验证输出的维度.
    print(f'编码器层输出形状: {output.shape}')
    # 7. 可选: 打印部分输出内容, 观察特征该变化. 即: 第1句话, 前2个词, 前5个维度(词向量), 回想: [2, 4, 512]
    print(f'编码器层输出内容: \n{output[:1, :2, :5]}')


if __name__ == '__main__':
    use_encoder_layer()