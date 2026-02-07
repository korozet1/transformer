"""
案例:
    演示解码器层的代码实现.
"""

# 导包
from dm05_encoder import *          # 编码器


# todo 1. 定义解码器层 -> Masked Multi-Head Attention, Feed Forward, Multi-Head Attention...
class DecoderLayer(nn.Module):
    # 1. 初始化函数.
    # 参1 d_model: 词向量维度
    # 参2 self_attn: 自注意力机制, 处理 解码器输入序列内部关系(解码器的输入)
    # 参3 src_attn: 源序列(编码器-解码器)注意力机制, 处理 关联编码器的输出 和 解码器的输入序列之间的关系
    # 参4 feed_forward: 前馈全连接层, 强化特征(解码器的输出)
    # 参5 dropout: 随机失活概率.
    def __init__(self, d_model, self_attn, src_attn , feed_forward, dropout=0.1):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 定义属性.
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        # 1.3 定义3个子层的连接结构.
        self.layers = clones(SublayerConnection(d_model, dropout), 3)

    # 2. 解码器层的 前向传播.
    # 参1 x: 解码器的输入序列(词嵌入 + 位置编码)
    # 参2 encoder_output: 编码器的输出序列(词嵌入 + 源序列位置编码)
    # 参3 source_mask: 源序列的填充掩码, 用于 编码器-解码器 注意力.
    # 参4 target_mask: 目标序列的填充掩码, 用于 自注意力.
    def forward(self, x, encoder_output, source_mask, target_mask):
        # 1. 经过第1个子层 -> 多头自注意力机制.
        x = self.layers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 2. 经过第2个子层 -> 多头注意力机制.              Q  != K,      K = V
        # 查询来自 解码器的输入序列, K,V来自编码器的输出序列.
        x = self.layers[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, source_mask))
        # 3. 经过第3个子层 -> 前馈全连接层.
        # x = self.layers[2](x, lambda x: self.feed_forward(x))
        x = self.layers[2](x, self.feed_forward)
        # 4. 返回结果.
        return x


# todo 2. 测试解码器层.
def use_decoder_layer():
    # 1. 定义解码器的输入 -> Q
    y = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])

    # 2. 将值传入到Embedding层(词嵌入层)
    # 2.1 创建词嵌入层对象.
    my_embed = Embeddings(1000, 512)
    # 2.2 将上述的y(输入) -> 词向量
    embed_y = my_embed(y)
    print("词嵌入层:", embed_y.shape)

    # 3. 将embed_y(词嵌入处理结果) -> 位置编码层
    my_position = PositionEncoding(512, 0.1)
    position_y = my_position(embed_y)
    print("位置编码层:", position_y.shape)

    # 4. 实例化 多头注意力机制.
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)   # 深层拷贝 -> 确保参数不会共享.
    src_attn = copy.deepcopy(multi_attn)

    # 5. 实例化 前馈全连接层.
    ff = FeedForward(512, 2048)

    # 6. 获取编码器的输出结果.
    encoder_output = use_encoder()

    # 7. 定义mask: source_mask 和 target_mask真实作用不一样(要解决你真正的业务来判断), 这里我就随便举例了.
    source_mask = torch.zeros(8, 4, 4)
    target_mask = torch.zeros(8, 4, 4)

    # 8. 实例化 解码器层.
    my_decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)

    # 9. 将 数据 传给 解码器层, 获取其输出结果.
    result = my_decoder_layer(position_y, encoder_output, source_mask, target_mask)

    # 10. 打印结果.
    print("解码器层:", result.shape)

if __name__ == '__main__':
    use_decoder_layer()