import torch.nn as nn

from network.components.weight_init_tool import weight_init


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        """

        :param input_dim: 经过注意力机制计算后 query 的最后一维的维度数
        :param hidden_dim: 前向网络的中间层维度, 一般是 input_dim 的倍数
        :param output_dim: 前向网络的输出维度, 也就是 query 经过 FFNN 网络计算后最后一维维度
        :param dropout: 正则化率
        """
        super(FeedForwardNetwork, self).__init__()

        # 第一层线性变换 (input_dim -> hidden_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim).apply(weight_init)
        # 第二层线性变换 (hidden_dim -> output_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim).apply(weight_init)
        # 激活函数
        self.relu = nn.ReLU()
        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: 输入张量，形状 (batch_size, seq_len, input_dim)
        :return: 输出张量，形状 (batch_size, seq_len, output_dim)
        """
        # 第一层线性变换 + 激活函数
        x = self.relu(self.fc1(x))
        # Dropout
        x = self.dropout(x)
        # 第二层线性变换
        x = self.fc2(x)

        return x
