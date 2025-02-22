import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding1D(nn.Module):
    def __init__(self, embedding_dim):
        """
        :param embedding_dim: 嵌入维度
        """
        super(SinusoidalPositionEmbedding1D, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, length):
        """
        :param length: 序列长度
        :return: 1D 位置编码，形状为 (length, embedding_dim)
        """
        position = torch.arange(length).float().unsqueeze(1)  # (length, 1)

        # (embedding_dim // 2)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(math.log(10000.0) / self.embedding_dim))

        pos_embedding = torch.zeros(length, self.embedding_dim)

        pos_embedding[:, 0::2] = torch.sin(position * div_term)  # (length, embedding_dim//2)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)  # (length, embedding_dim//2)

        return pos_embedding


class SinusoidalPositionEmbedding2D(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionEmbedding2D, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, height, width):
        # 计算位置索引
        y_pos = torch.arange(height).float().unsqueeze(1)  # (height, 1)
        x_pos = torch.arange(width).float().unsqueeze(0)  # (1, width)

        # 计算频率
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * -(
                math.log(10000.0) / self.embedding_dim))  # (embedding_dim // 2)

        # 计算正弦和余弦
        pos_embedding = torch.zeros(height, width, self.embedding_dim)

        # 扩展位置索引的维度，使其能够广播
        y_pos = y_pos.repeat(1, width)  # (height, width)
        x_pos = x_pos.repeat(height, 1)  # (height, width)

        # 正弦嵌入 (偶数维度)
        pos_embedding[..., 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)  # (height, width, embedding_dim // 2)
        # 余弦嵌入 (奇数维度)
        pos_embedding[..., 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)  # (height, width, embedding_dim // 2)

        return pos_embedding
