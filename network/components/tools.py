import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def get_activation_function(activation):
    """
    根据激活函数的名字字符串获得对应的激活函数
    :param activation:
    :return:
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def weight_init(m):
    for p in m.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
