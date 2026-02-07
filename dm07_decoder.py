"""
案例:
    演示 解码器 代码实现.
"""

# 导包
from dm06_decoder_layer import *

# todo 1. 定义解码器: 把多个解码器层(层数: 6)进行堆叠, 最后加一层规范化, 让输出更稳定.
class Decoder(nn.Module):
    # 1. 初始化函数
    # 参1: layer -> 单个解码器层(DecoderLayer类)的对象, 要被复制N次
    # 参2: 解码层的堆叠数量
    def __init__(self, layer, N):
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 克隆N个解码器层.
        self.layers = clones(layer, N)
        # 1.3 定义最终的规划范层.
        self.norm = LayerNorm(layer.d_model)

    # 2. 定义前向传播函数.
    # 参1 x: 解码器的输入序列(词嵌入 + 位置编码)
    # 参2 encoder_output: 编码器的输出序列(词嵌入 + 源序列位置编码)
    # 参3 source_mask: 源序列的填充掩码, 用于 编码器-解码器 注意力.
    # 参4 target_mask: 目标序列的填充掩码, 用于 自注意力.
    def forward(self, x, encoder_output, source_mask, target_mask):
        # 1. 数据x依次经过 多个 解码器层的处理即可.
        for layer in self.layers:       # layer: 就表示某一个具体的解码器层.
            x = layer(x, encoder_output, source_mask, target_mask)

        # 2. 全局规范化, 把所有层的输出再统一标准化.
        return self.norm(x)


# todo 2. 测试解码器, 从输入到输出走一遍流程, 验证每层的维度是否正常.
def use_decoder():
    # 1. 定义解码器的输入 -> Q
    y = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
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
    self_attn = copy.deepcopy(multi_attn)  # 深层拷贝 -> 确保参数不会共享.
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

    # 9. 堆6层解码器层 -> 组成解码器.
    my_decoder = Decoder(my_decoder_layer, 6)

    # 10. 跑一遍解码流程: 输入 -> 6层解码器层 -> 输出
    result = my_decoder(position_y, encoder_output, source_mask, target_mask)

    # 11. 打印结果.
    print(f'最终解码器的输出结果(维度): {result.shape}')
    print(f'最终解码器的输出结果: {result}')

    # 12. 返回结果.
    return result

if __name__ == '__main__':
    # 测试: 解码器.
    use_decoder()