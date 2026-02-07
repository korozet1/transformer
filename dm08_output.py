"""
案例:
    演示Transformer的 输出部分.
"""

# 导包
from dm07_decoder import *


# todo 1. 定义输出部分, 把解码器的特征转成最终预测结果(比如: 翻译后的词)
class Generator(nn.Module):
    # 1. 初始化函数.
    # 参1 d_model: 词向量维度, 例如: 512
    # 参2 vocab_size: 词典大小, 例如: 1000
    def __init__(self, d_model, vocab_size):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 定义线性层, 封装: 输入: 词向量维度, 输出: 词典大小
        # 例如: 输入[2, 4, 512] -> 输出[2, 4, 1000]  假设词典大小为1000
        self.linear = nn.Linear(d_model, vocab_size)

    # 2. 前向传播函数.
    def forward(self, x):
        # 2.1 线性层, 输出: [2, 4, 1000]
        x = self.linear(x)
        # 2.2 log_softmax(): 将分数转成 对数概率分布.
        # -1表示对最后1维(词汇表维度)做计算, 确保 概率和为 1
        return F.log_softmax(x, dim=-1)


# todo 2. 测试输出结果, 从解码器到最终预测的完成流程.
def use_generator():
    # 功能: 获取解码器的特征 -> 通过生成器转成目标词汇表的概率分布 -> 验证输出维度 概率分布格式.
    # 1. 获取解码器的输出结果.
    result = use_decoder()
    # 2. 初始化 输出生成器, 将512维的解码器特征, 转成1000维的概率分布.
    generator = Generator(512, 5000)
    # 3. 通过生成器, 获取概率分布.
    output = generator(result)
    # 4. 验证输出
    # 形状为: [batch_size, seq_len, vocab_size], 每个位置的数值是对数概率, 可以通过 exp()转成概率.
    print(f'output 模型最终输出结果: {output.shape}, {output}')     # [2, 4, 1000]

    # 5. 验证概率和为1, 取 第1个样本, 第1个词的概率总和 -> 无限接近1
    # 5.1 提取第1个样本, 第1个词的 对数概率, 形状: [1000]
    log_probs = output[1, 3]
    # 5.2 把对数概率 -> 转成普通的概率.
    probs = torch.exp(log_probs)
    # print(f'第1个样本, 第1个词的概率: {probs}')
    # 5.3 验证概率和为1
    print(f'第1个词的概率总和为: {torch.sum(probs)}')


# todo 3. 测试
if __name__ == '__main__':
    use_generator()