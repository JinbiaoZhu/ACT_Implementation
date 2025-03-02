import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots=None):
        """
        输入 (batch_size, length, d_model) 维度的张量, 输出 (batch_size, num_slots, d_model) 维度的张量
        当 length >> num_slots 时候, 槽注意力模型可以看做是一种降维方法
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

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))

        return slots


# if __name__ == "__main__":
#     net = SlotAttention(num_slots=3, dim=512)
#
#     batch_size, seq_len_image, height_32, width_32, d_model = 32, 4, 8, 8, 512
#     input_tensor = torch.rand((batch_size, seq_len_image * height_32 * width_32, d_model))
#     return_slots = net(input_tensor)
#     print(return_slots.shape)
