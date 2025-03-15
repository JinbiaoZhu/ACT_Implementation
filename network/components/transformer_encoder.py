import torch.nn as nn

from network.components.tools import get_activation_function


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        """

        :param d_model: query / key / value 映射的维度
        :param nhead: 多头注意力的头数
        :param dim_feedforward: 前向神经网络的中间层维度数量
        :param dropout: 正则化率
        :param activation: 激活函数字符串, 具体激活函数用 get_activation_function 检索
        :param normalize_before: 进行前归一化 (True) 还是 后归一化 (False)
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_function(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        """
        如果位置编码是 None, 则直接输出原张量; 反之, 输出原张量和位置编码的相加
        :param tensor:
        :param pos:
        :return:
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)  # 强调 query 和 key 需要位置编码, 老版代码中对 query / key / value 都编码了
        # 自注意力机制 ==> 正则化 ==> 残差相加 ==> 层归一化处理 ==> 两层带正则化率的 FFNN ==> 正则化 ==> 残差相加 ==> 层归一化处理
        src2 = self.self_attn(query=q, key=k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask, src_key_padding_mask, pos):
        # 层归一化处理 ==> 自注意力机制 ==> 正则化 ==> 残差相加 ==> 层归一化处理 ==> 两层带正则化的 FFNN ==> 正则化 ==> 残差相加
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)  # 强调 query 和 key 需要位置编码, 老版代码中对 query / key / value 都编码了
        src2 = self.self_attn(query=q, key=k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask, src_key_padding_mask, pos):
        if self.normalize_before:  # 前归一化
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)  # 后归一化
