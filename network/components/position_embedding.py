import math

import torch
import torch.nn as nn


class PositionEmbedding_Sine_1D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        用于生成一维正弦位置编码, 常用于序列建模任务 (如 Transformer)
        :param num_pos_feats: 位置编码的特征维度, 默认为 64, 一般取偶数
        :param temperature: 缩放因子, 默认为 10000, 参考 Transformer 原论文
        :param normalize: 是否归一化坐标, 确保位置编码值范围合理
        :param scale: 归一化比例因子, 默认 2*pi, 如果传入 scale, 则 normalize 必须为 True
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor  # (batch_size, seq_length, d_model)
        not_mask = torch.ones_like(x[:, :, 0])  # (batch_size, seq_length)
        pos_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿着序列维度累加

        # 如果进行归一化
        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (pos_embed[:, -1:] + eps) * self.scale

        # 生成频率指数
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = pos_embed[:, :, None] / dim_t  # (batch_size, seq_length, num_pos_feats)

        # 交替计算 sin 和 cos
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        return pos  # (batch_size, seq_length, num_pos_feats)


class PositionEmbedding_Sine_2D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        用于生成二维正弦位置编码, 用于 ACT 网络中图像特征图处理
        :param num_pos_feats: 位置编码的特征维度, 一般取偶数, 默认为 64, 这里就默认 64 了
        :param temperature: 缩放因子, 默认为 10000, 用于指数衰减 (参考 Transformer 原论文)
        :param normalize: 是否归一化坐标, 确保位置编码值范围合理
        :param scale: 归一化比例因子, 默认 2*pi, 如果传入 scale, 则 normalize 必须为 True
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor  # (batch_size, channels, height, width)
        # 使用高级索引, 它会返回 x[0] (维度砍掉) 中的第 0 个元素, 保持其维度特性 (维度没砍掉)
        not_mask = torch.ones_like(x[0, [0]])  # (1, height, width)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # 沿着高度方向进行累加计算 (1, height, width)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # 沿着宽度方向进行累加计算 (1, height, width)

        # 如果进行正则化
        # 对高度和宽度方向, 分别除以各自方向的最大值, 此时数值范围是 [0, 1]. 注意加入一个极小量避免除数为 0.
        # 然后乘以比例因子, 将范围扩大至 [0, 2*pi]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 获取一个长度是 self.num_pos_feats 的序列
        # 使用 dim_t // 2 把整个序列从前到后两两分组 (0, 0, 1, 1, 2, 2, 3, 3, ...)
        # 乘以 2 是实现 $2i/d$ 也就是索引序列是 (0, 0, 2, 2, 4, 4, 6, 6, ...)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (1, height, width, self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t  # (1, height, width, self.num_pos_feats)

        # 正弦位置编码:
        # 对偶数索引位置正弦计算, 对奇数索引位置余弦计算
        # 使用 .stack() 操作将正弦和余弦交叉拼接起来
        # 使用 .flatten() 操作得到规整的张量 (1, height, width, self.num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # (1, 2 * self.num_pos_feats, height, width), 相当于特征通道数从 3 变成了 2 * self.num_pos_feats
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


if __name__ == "__main__":
    batch_size, channel, height, width = 32, 3, 128, 128
    x = torch.rand((batch_size, channel, height, width))
    pe = PositionEmbedding_Sine_2D(num_pos_feats=256)
    out = pe(x)
    print(out.shape)

    batch_size, seq_len, d_model = 32, 10, 1024
    x = torch.rand((seq_len, batch_size, d_model))
    pe1d = PositionEmbedding_Sine_1D(num_pos_feats=d_model)
    out = pe1d(x)
    print(out.shape)
