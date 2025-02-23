import math

import torch.nn as nn


def weight_init(m):
    """
    Custom weight init for Conv2D and Linear layers.
    :param m: model parameters
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, math.sqrt(2))
        m.bias.data.fill_(0.0)
