import torch
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


def freeze_except_decoder(model):
    """
    冻结 model 中除 ACT 解码器部分之外的所有网络层参数,
    保留 ACT 解码器相关模块 (包括 ACT_decoder、query_embed 和 action_head) 可训练.
    """
    # 先冻结所有参数
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 定义需要解冻的模块名称标识（可以根据需要添加更多）
    modules_to_unfreeze = ["ACT_decoder", "query_embed", "action_head"]

    # 遍历所有参数名称，如果名称中包含需要解冻的模块标识，则开启梯度
    for name, param in model.named_parameters():
        for module_name in modules_to_unfreeze:
            if module_name in name:
                param.requires_grad = True
                break  # 一旦匹配上某个模块名称，就不需要再检查其它模块


def count_model_params(model: torch.nn.Module):
    """
    计算模型参数数量，包括：
      - 总参数数量
      - 可训练参数数量
      - 可训练参数占比
    :param model: torch.nn.Module, 模型实例
    :return: tuple (total_params, trainable_params, trainable_ratio)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    return total_params, trainable_params, trainable_ratio
