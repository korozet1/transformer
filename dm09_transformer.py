"""
案例:
    演示完成的 Transformer框架.
"""
# 导包
from dm08_output import *


# todo 1. 定义整体的Transformer框架模型, 初始化Transformer模型的主要组件.
class EncoderDecoder(nn.Module):        # 编码器-解码器模型
    # 1. 初始化函数.
    def __init__(self, source_embed, encoder, target_embed, decoder, generator):
        """
        函数功能: 模型的初始化函数, 初始化父类成员, 定义子层等...
        :param source_embed: 编码器的输入信息(词向量 + 位置编码)
        :param encoder: 编码器模块
        :param target_embed: 解码器的输入信息(词向量 + 位置编码)
        :param decoder: 解码器
        :param generator: 输出层, 将解码器输出 转成 词汇表概率分布.
        """
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 定义属性, 记录上述的参数.
        self.source_embed = source_embed
        self.encoder = encoder
        self.target_embed = target_embed
        self.decoder = decoder
        self.generator = generator

    # 2. 前向传播函数.
    def forward(self, source_x, target_y, source_mask, target_mask):
        """
        函数功能: 前向传播
        :param source_x: 编码器的输入, 即: [batch_size, seq_len] -> 例如: [2, 4]
        :param target_y: 解码器的输入, 即: [batch_size, seq_len] -> 例如: [2, 4]
        :param source_mask: 代表padding-mask, 防止填充的padding值影响注意力计算结果, 例如: [8, 4, 4]
        :param target_mask: 代表sentence-mask, 防止未来信息被提前利用.
        :return: 模型预测的词汇表概率分布,  形状: [batch_size, seq_len, vocab_size]
        """

        # 1. 得到编码器的输出结果.
        encoder_result = self.encode(source_x, source_mask)
        # 2. 得到解码器的输出结果.
        decoder_result = self.decode(target_y, encoder_result, source_mask, target_mask)
        # 3. 将解码器的结果 经过 输出层输出.
        output = self.generator(decoder_result)
        return output

    # 3. 得到编码器的输出结果, 编码器的前向传播过程.
    def encode(self, source_x, source_mask):
        """
        编码器的前向传播过程
        :param source_x: 输入, 形状: [batch_size, seq_len]  -> [2, 4]
        :param source_mask: 形状: [batch_size, seq_len, seq_len] -> [8, 4, 4]
        :return: encoder_output: [batch_size, seq_len, d_model] -> [2, 4, 512]
        """
        # 1.1 对source_x进行处理
        embed_x = self.source_embed(source_x)
        # 1.2 通过编码器处理, 输出结果为: [batch_size, seq_len, d_model], 形状: [2, 4, 512]
        encoder_output = self.encoder(embed_x, source_mask)
        # 1.3 返回编码器输出结果.
        return encoder_output

    # 4. 得到解码器的输出结果, 解码器的前向传播过程.
    def decode(self, target_y, encoder_output, source_mask, target_mask):
        """
        解码器的前向传播过程
        :param target_y: 解码器的输入序列(词嵌入 + 位置编码)
        :param encoder_output: 编码器的输出序列(词嵌入 + 源序列位置编码)
        :param source_mask: 源序列的填充掩码, 用于 编码器-解码器 注意力.
        :param target_mask: 目标序列的填充掩码, 用于 自注意力.
        :return:
        """

        # 1. 将target_y进行词嵌入, 形状: [batch_size, seq_len, d_model] -> [2, 4, 512]
        embed_y = self.target_embed(target_y)
        # 2. 通过解码器处理(刚才的数据), 并结合 输出部分 输出.
        return self.decoder(embed_y, encoder_output, source_mask, target_mask)


# todo 2. 测试...
def make_model():
    # 1. 定义深度拷贝工具函数, 用于 复用模块.
    c = copy.deepcopy

    # ------------ 编码器层 ------------
    # 2. 编码器 输入嵌入层
    source_embed = Embeddings(vocab_size=1000, d_model=512)
    # 3. 编码器 位置编码层
    source_position = PositionEncoding(d_model=512, dropout=0.1)

    # 4. 多头注意力机制实例化
    self_attn = MultiHeadAttention(embed_dim=512, head=8)

    # 5. 定义变量, 记录: 层归一化维度(要与特征维度保持一致)
    size = 512
    # 6. 前馈全连接层实例化.
    ff = FeedForward(d_model=size, d_ff=2048)
    # 7. 随机失活概率 -> 防止过拟合
    dropout_p = 0.2

    # 8.单个编码器层实例化.
    # 包括: 多头注意力 + 前馈网络 + 残差连接 + 层规范化
    my_encoder_layer = EncoderLayer(size, self_attn, ff, dropout_p)
    # 9. 编码器模块实例化.
    encoder = Encoder(my_encoder_layer, 6)


    # ------------ 解码器层 ------------
    # 1. 解码器的输入嵌入层. (复用编码器嵌入层)
    # target_embed = Embeddings(vocab_size=1000, d_model=512)
    target_embed = c(source_embed)      # 深拷贝

    # 2. 解码器的位置编码层. (复用位置编码层)
    target_position = c(source_position)

    # 3. 解码器自注意力机制(处理目标序列内部关系)
    self_attn1 = c(self_attn)
    # 4. 解码器-编码器 注意力机制(处理目标序列与源序列关系)
    score_attn1 = c(self_attn)
    # 5. 解码器的前馈神经网络
    ff1 = c(ff)

    # 6. 单个解码器层的实例化.
    my_decoder_layer = DecoderLayer(size, self_attn1, score_attn1, ff1, dropout_p)

    # 7. 解码器主题构建
    decoder = Decoder(my_decoder_layer, 6)

    # 8. 输出层实例化.
    generator = Generator(size, 1000)       # 处理后: [2, 4, 1000]
    # 9. 组装完整Transformer模型.
    # 编码器输入处理: 词嵌入 + 位置编码
    # 解码器输入处理: 词嵌入 + 位置编码
    my_transformer = EncoderDecoder(
        nn.Sequential(source_embed, source_position),         # 编码器输入处理链
        encoder,
        nn.Sequential(target_embed, target_position),        # 解码器输入处理链
        decoder,
        generator
    )
    # 10. 打印模型结构
    print(f'my_transformer: {my_transformer}')


    # --------------- 模型前向测试 ---------------
    # 1. 构建测试输入数据.
    # 源序列, 2个句子, 每个句子4个词(token)
    source_x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    # 目标序列, 2个句子, 每个句子4个词(token)
    target_y = torch.LongTensor([[3, 8, 6, 4], [2, 1, 6, 6]])

    # 2. 初始化掩码(实际开发中药根据序列长度动态生成)
    source_mask = torch.zeros(8, 4, 4)
    target_mask = c(source_mask)

    # 3. 执行前向传播.
    result = my_transformer(source_x, target_y, source_mask, target_mask)
    print(f'result是Transformer的输出结果: {result}')
    print(f'result的(形状): {result.shape}')


# 测试代码
if __name__ == '__main__':
    make_model()
