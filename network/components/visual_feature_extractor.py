import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from network.components.tools import is_main_process
from network.components.position_embedding import PositionEmbedding_Sine_2D


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:  # 确保只有当 mask 存在时才执行 .to(device)
            assert mask is not None  # 额外的安全性检查
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        eps = 1e-5
        scale = self.weight.reshape(1, -1, 1, 1) * (self.running_var.reshape(1, -1, 1, 1) + eps).rsqrt()
        bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean.reshape(1, -1, 1, 1) * scale
        return x * scale + bias


class VisionBackbone(nn.Module):
    def __init__(self, d_model,
                 name: str,
                 return_interm_layers: bool,
                 include_depth: bool
                 ):
        super().__init__()

        # 载入 torchvision 的 ResNet 模型骨干
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, False],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d
        )

        if include_depth:
            w = backbone.conv1.weight
            w = torch.cat([w, torch.full((64, 1, 7, 7), 0)], dim=1)
            backbone.conv1.weight = nn.Parameter(w)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        self.position_embedding = PositionEmbedding_Sine_2D(d_model // 2)

    def forward(self, nested_tensor: NestedTensor):
        """
        处理 NestedTensor，并正确传播 mask 信息
        :param nested_tensor:
        :return:
        """
        tensor, mask = nested_tensor.decompose()  # 解包 NestedTensor
        xs = self.body(tensor)  # 仅将 tensor 送入 ResNet 计算特征
        out, pos = [], []
        for name, x in xs.items():
            if mask is not None:
                # 调整 mask 尺寸，使其匹配输出特征图
                mask_resized = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            else:
                mask_resized = None

            out.append(NestedTensor(x, mask_resized))  # 保留 mask 信息
            pos.append(self.position_embedding(x).to(x.dtype))

        return out, pos


if __name__ == "__main__":
    def test_vision_backbone():
        class Args:
            backbone = "resnet18"
            lr_backbone = 1e-3
            include_depth = False

        args = Args()
        model = VisionBackbone(
            d_model=512,
            name=args.backbone,
            return_interm_layers=False,
            include_depth=args.include_depth,
        )
        model.eval()

        # 创建一个带 mask 的 NestedTensor
        batch_size, channels, height, width = 35, 3, 224, 224
        input_tensor = torch.rand(batch_size, channels, height, width)
        mask = torch.ones(batch_size, height, width, dtype=torch.bool)  # 全 1 表示可见区域

        nested_tensor = NestedTensor(input_tensor, mask)
        output, pos = model(nested_tensor)


        assert isinstance(output, list), "Output should be a list of tensors"
        assert isinstance(pos, list), "Position embeddings should be a list of tensors"
        print("Test passed!")


    # 运行测试
    test_vision_backbone()