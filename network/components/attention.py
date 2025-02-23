import torch
import torch.nn as nn
import torch.nn.functional as F

from network.components.weight_init_tool import weight_init


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim, heads):
        """
        注意力机制模块的实现
        :param input_dim: 输入注意力机制模块的 query key 和 value 的最后一维维度数, 三者要保持一致
        :param output_dim: 经过注意力机制模块后输出 query 最后一维维度, output_dim 不一定等于 input_dim
        :param heads: 多头注意力的头数
        """
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads

        self.query_linear = nn.Linear(input_dim, output_dim * heads)
        self.key_linear = nn.Linear(input_dim, output_dim * heads)
        self.value_linear = nn.Linear(input_dim, output_dim * heads)
        self.out_linear = nn.Linear(output_dim * heads, output_dim)

        self.query_linear.apply(weight_init)
        self.key_linear.apply(weight_init)
        self.value_linear.apply(weight_init)
        self.out_linear.apply(weight_init)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, seq_len_q, input_dim)
        :param key: (batch_size, seq_len_k, input_dim)
        :param value: (batch_size, seq_len_v, input_dim)
        :param mask: Optional, (batch_size, seq_len_q, seq_len_k)
        :return: output: (batch_size, seq_len_q, output_dim)
        """
        # 将输入的 query key 和 value 投影到多头空间
        Q = self.query_linear(query)  # (batch_size, seq_len_q, output_dim * heads)
        K = self.key_linear(key)  # (batch_size, seq_len_k, output_dim * heads)
        V = self.value_linear(value)  # (batch_size, seq_len_v, output_dim * heads)

        # 分割多个头
        Q = Q.view(Q.size(0), Q.size(1), self.heads, self.output_dim)  # (batch_size, seq_len_q, heads, output_dim)
        K = K.view(K.size(0), K.size(1), self.heads, self.output_dim)  # (batch_size, seq_len_k, heads, output_dim)
        V = V.view(V.size(0), V.size(1), self.heads, self.output_dim)  # (batch_size, seq_len_v, heads, output_dim)

        # 转置，使得头数是最外层维度
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, heads, seq_len_q, output_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, heads, seq_len_k, output_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, heads, seq_len_v, output_dim)

        # 计算注意力分数 Q * K^T
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, heads, seq_len_q, seq_len_k)

        # 通过缩放因子调整 sqrt(d_k)
        attention_scores = attention_scores / (self.output_dim ** 0.5)

        # 可选的 mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, heads, seq_len_q, seq_len_k)

        # 加权求和得到输出
        output = torch.matmul(attention_weights, V)  # (batch_size, heads, seq_len_q, output_dim)

        # 将多头的输出合并
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len_q, heads, output_dim)
        output = output.view(output.size(0), output.size(1),
                             self.heads * self.output_dim)  # (batch_size, seq_len_q, output_dim * heads)

        # 最终通过线性层映射到输出维度
        output = self.out_linear(output)  # (batch_size, seq_len_q, output_dim)

        return output
