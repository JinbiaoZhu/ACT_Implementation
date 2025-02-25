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


class SlotAttention(nn.Module):
    """
    槽注意力模型, 参考论文: https://arxiv.org/abs/2006.15055
    这份代码参考这个 Github: https://github.com/evelinehong/slot-attention-pytorch/blob/master/model.py
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim).apply(weight_init)  # 对线性层都做初始化
        self.to_k = nn.Linear(dim, dim).apply(weight_init)
        self.to_v = nn.Linear(dim, dim).apply(weight_init)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim).apply(weight_init)
        self.fc2 = nn.Linear(hidden_dim, dim).apply(weight_init)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        """
        输入 (batch_size, length, d_model) 维度的张量, 输出 (batch_size, num_slots, d_model) 维度的张量
        当 length >> num_slots 时候, 槽注意力模型可以看做是一种降维方法
        相当于 query 是槽, key 和 value 是输入图像, 且经过注意力机制后额外增加一个循环记忆网络
        注意力机制部分做了修改, FFNN 和残差部分没有修改
        :param inputs: 输入 (batch_size, length, d_model) 维度的张量
        :param num_slots: 可调用时选择, 也可直接类属性赋值
        :return: 输出 (batch_size, num_slots, d_model) 维度的张量
        """
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots
