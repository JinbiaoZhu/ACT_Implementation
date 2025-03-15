import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

from network.components.position_embedding import PositionEmbedding_Sine_1D
from network.components.transformer_encoder import TransformerEncoderLayer
from network.components.transformer_decoder import TransformerDecoderLayer
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor
from network.components.tools import weight_init

"""
Action Chunking Transformer (ACT) 算法的实现.

代码包含三部分:

# 1. 表征编码器部分

"""


class ActionChunkingTransformer(nn.Module):

    # def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries):
    def __init__(self,
                 d_model, d_proprioception, d_action, d_z_distribution, d_feedforward,
                 n_head, n_representation_encoder_layers, n_encoder_layers, n_decoder_layers,
                 chunk_size,  # 这一行是动作长度参数, chunk_size 相当于 num_queries
                 resnet_name: str, return_interm_layers: bool, include_depth: bool,  # 这一行是视觉骨干网络参数
                 dropout, activation, normalize_before: bool
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_z_distribution = d_z_distribution
        self.normalize_before = normalize_before

        # 表征编码器部分
        self.cls_embed = nn.Embedding(1, d_model)  # [cls] token 的映射, 原本用 nn.Linear 但是不合理, 故改之
        self.proprio_proj = nn.Linear(d_proprioception, d_model)  # 将本体维度线性映射到模型维度
        self.action_proj = nn.Linear(d_action, d_model)  # 将动作维度线性映射到模型维度
        self.latent_proj = nn.Linear(d_model, d_z_distribution * 2)  # 输出序列中 [cls] token 对应向量线性映射到 z 分布的均值和方差向量
        self.position_embedding_1d = PositionEmbedding_Sine_1D(d_model)  # 1 维位置嵌入, 用于处理序列数据
        self.representation_encoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerEncoderLayer(d_model, n_head, d_feedforward, dropout, activation, normalize_before)
                )
                for i in range(n_representation_encoder_layers)
            ]
        )
        if self.normalize_before:
            self.representation_encoder_output_layer_norm = nn.LayerNorm(d_model)

        # ACT 编码器-解码器部分
        self.latent_out_proj = nn.Linear(d_z_distribution, d_model)  # 将采样的 z 分布向量线性映射到模型维度, 作为一个 token
        self.input_proprio_proj = nn.Linear(d_proprioception, d_model)  # 将输入的本体数据线性映射到模型维度, 作为一个 token
        self.query_embed = nn.Embedding(chunk_size, d_model)  # 解码器端输入的预测动作长度, 也就是 chunk_size = num_queries
        self.additional_pos_embed = nn.Embedding(2, d_model)  # 为 z 表征和本体嵌入张量设计一个可学习的位置编码
        self.ACT_encoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerEncoderLayer(d_model, n_head, d_feedforward, dropout, activation, normalize_before)
                )
                for i in range(n_encoder_layers)
            ]
        )
        self.ACT_decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerDecoderLayer(d_model, n_head, d_feedforward, dropout, activation, normalize_before)
                )
                for i in range(n_decoder_layers)
            ]
        )
        if self.normalize_before:
            self.ACT_encoder_output_layer_norm = nn.LayerNorm(d_model)
        self.backbones = VisionBackbone(d_model, resnet_name, return_interm_layers, include_depth)
        self.input_proj = nn.Conv2d(self.backbones.num_channels, d_model, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(d_proprioception, d_model)
        self.action_head = nn.Linear(d_model, d_action)

        # 在此处统一进行权重初始化
        self.apply(weight_init)

    def forward(self, obs, actions=None):
        is_training = actions is not None  # 训练模式: 有动作输入, 包括训练和验证集测试; 仿真器/真机测试模式: 没有动作输入
        state = obs['state'] if self.backbones is not None else obs

        bs = state.shape[0]

        if is_training:  # 训练模式: 有动作输入, 包括训练和验证集测试;

            # ==========> 表征编码器部分
            # [cls] token 的处理: 将其复制到整个 batch 的数据样本中
            cls_embed = self.cls_embed.weight  # (1, d_model)
            cls_embed = cls_embed.unsqueeze(0).repeat(bs, 1, 1)  # (batch_size, 1, d_model)

            # 本体数据的处理: 将 batch 中的每个本体数据映射到 d_model 中, 并扩充一个维度
            state_embed = self.proprio_proj(state)  # (batch_size, d_model)
            state_embed = state_embed.unsqueeze(1)  # (batch_size, 1, d_model)

            # 动作序列的处理: 将 batch 中的每个动作序列映射到 d_model 中
            action_embed = self.action_proj(actions)  # (batch_size, seq_len, d_model)

            # 拼接 [cls] token 、本体数据和动作序列作为表征编码器的输入
            # 同时获得位置嵌入, encoder_input 和 pos_embed 的维度都是 (batch_size, seq_len+2, d_model)
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], dim=1)
            pos_embed = self.position_embedding_1d(encoder_input)

            # 修改维度  (seq_len+2, batch_size, d_model)
            encoder_input = encoder_input.permute(1, 0, 2)
            pos_embed = pos_embed.permute(1, 0, 2)

            # 表征编码器的输入没有掩码, False 表示没有掩码, True 表示是 pad 掩码
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device)

            # 表征编码器, encoder_output (seq_len+2, batch_size, d_model)
            # 在循环中, 将上一层输出, 也就是 encoder_input, 当做这一层输入参数 src
            for layer in self.representation_encoder:
                encoder_input = layer(
                    src=encoder_input,
                    src_mask=None,
                    src_key_padding_mask=is_pad,
                    pos=pos_embed
                )
            if self.normalize_before:
                encoder_input = self.representation_encoder_output_layer_norm(encoder_input)

            # 取第 0 个位置也就是 [cls] token 对应位置 token 的张量, (batch_size, d_model)
            encoder_output = encoder_input[0]

            # 将张量线性映射到两个张量, 分别代表 z 分布的均值向量和方差 (log) 向量, 并按维度提取之
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.d_z_distribution]
            logvar = latent_info[:, self.d_z_distribution:]
            # 计算实际方差值, 构建 z 分布并重参数化采样
            std = logvar.div(2).exp()
            eps = Variable(std.data.new(std.size()).normal_())
            z = mu + std * eps

            # 将重参数化采样的 z 张量线性映射到 d_model 作为一个 token
            latent_input = self.latent_out_proj(z)

        else:  # 仿真器/真机测试模式: 没有动作输入
            mu = logvar = None
            # z 分布只使用全 0 张量
            latent_sample = torch.zeros([bs, self.d_z_distribution], dtype=torch.float32).to(state.device)
            # 将重参数化采样的 z 张量线性映射到 d_model 作为一个 token
            latent_input = self.latent_out_proj(latent_sample)

        # ==========> ACT 编码器-解码器部分
        # 对多视角处理, 首先提取 RGB 数据, (batch_size, num_view, channel, height, width)
        vis_data = obs['rgb']
        # 如果观测 obs 数据中有深度数据的话, 在第 2 个维度做拼接, 相当于额外增加一个通道数
        if "depth" in obs:
            vis_data = torch.cat([vis_data, obs['depth']], dim=2)
        # 记录可以使用的视角数量
        num_cams = vis_data.shape[1]

        # 图像观测特征提取与位置编码
        all_cam_features, all_cam_pos = [], []
        for cam_id in range(num_cams):  # 在 for 循环中对每个视角都进行同样的特征提取
            features, pos = self.backbones(
                NestedTensor(vis_data[:, cam_id], None))  # features 和 pos 返回的是一个列表, 取列表第一个位置的特征张量
            features = features[0].tensors  # (batch_size, d_model, height, width)
            pos = pos[0]  # (1, d_model, height, width)
            all_cam_features.append(
                self.input_proj(features)  # 额外做一次转换, 形状不变 (batch_size, d_model, height, width)
            )
            all_cam_pos.append(pos)

        # 将本体数据线性映射到模型维度中, 作为一个 token
        proprio_input = self.input_proprio_proj(state)

        # ==========> ACT 编码器的输入部分
        # 整合所有图像成为一个 token 序列
        src = torch.cat(all_cam_features, dim=3)  # (batch_size, d_model, height, width*num_views)
        src = src.flatten(2).permute(2, 0, 1)  # (height*width*num_views, batch_size, d_model)
        # 整合 z 采样张量 和 本体嵌入 成为一个 token 序列
        addition_input = torch.stack([latent_input, proprio_input], dim=0)
        src = torch.cat([addition_input, src], dim=0)

        # 整合所有图像位置编码成为一个 token 序列
        pos_embed = torch.cat(all_cam_pos, dim=3)  # (1, d_model, height, width*num_views)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs,
                                                                 1)  # (height*width*num_views, batch_size, d_model)
        # 为 z 张量 和 本体数据嵌入 构建一个 2 token 的位置编码序列
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs,
                                                                                    1)  # (2, batch_size, d_model)
        # 整合 z 张量、本体嵌入和相机的 token 序列的位置编码
        # 最终的形状是 (2+height*width*num_views, batch_size, d_model)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], dim=0)

        # ==========> ACT 解码器的输入部分
        # 整合所有 action chunk 预测的 slots 成为一个 token 序列
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # ==========> ACT 编码器的前向计算 --> 得到 key 和 value 张量
        for layer in self.ACT_encoder:
            src = layer(
                src=src,
                src_mask=None,
                src_key_padding_mask=None,
                pos=pos_embed
            )
        if self.normalize_before:
            src = self.ACT_encoder_output_layer_norm(src)
        key_and_value = src
        for layer in self.ACT_decoder:
            tgt = layer(
                tgt=tgt, memory=key_and_value,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=pos_embed, query_pos=query_embed
            )
        hidden_state = tgt.transpose(1, 0)

        a_hat = self.action_head(hidden_state)
        return a_hat, [mu, logvar]


# ==========> 测试函数
def test_action_chunking_transformer():
    # 定义模型参数
    d_model = 64
    d_proprioception = 10
    d_action = 5
    d_z_distribution = 16
    d_feedforward = 128
    n_head = 8
    n_representation_encoder_layers = 2
    n_encoder_layers = 2
    n_decoder_layers = 2
    chunk_size = 50  # 相当于 num_queries
    resnet_name = "resnet18"
    return_interm_layers = False
    include_depth = False
    dropout = 0.1
    activation = "relu"
    normalize_before = False

    # 实例化模型
    model = ActionChunkingTransformer(d_model, d_proprioception, d_action, d_z_distribution,
                                      d_feedforward, n_head, n_representation_encoder_layers,
                                      n_encoder_layers, n_decoder_layers, chunk_size,
                                      resnet_name, return_interm_layers, include_depth,
                                      dropout, activation, normalize_before)

    model.eval()

    for n, p in model.named_parameters():
        print(n)

    # 构造 dummy 输入数据
    bs = 27  # batch size
    seq_len = chunk_size  # 动作序列长度
    num_views = 4  # 摄像头个数
    channel = 3
    height, width = 128, 128

    # 本体状态: (bs, d_proprioception)
    state = torch.randn(bs, d_proprioception)
    # 动作序列: (bs, seq_len, d_action)
    actions = torch.randn(bs, seq_len, d_action)
    # RGB 图像: (bs, num_views, channel, height, width)
    rgb = torch.randn(bs, num_views, channel, height, width)

    # 构造输入字典 (此处不提供 depth 数据)
    obs = {
        'state': state,
        'rgb': rgb
    }

    # 前向传播（训练模式，传入动作）
    a_hat, (mu, logvar) = model(obs, actions)

    # 输出结果形状
    # 理论上 a_hat 的形状应该为 (bs, chunk_size, d_action)
    print("a_hat shape:", a_hat.shape)
    # mu 与 logvar 的形状均应为 (bs, d_z_distribution)
    print("mu shape:", mu.shape)
    print("logvar shape:", logvar.shape)

    # 简单断言
    assert a_hat.shape[0] == bs, "输出 batch size 错误"
    assert a_hat.shape[1] == chunk_size, "输出的动作 chunk 数量错误"
    assert a_hat.shape[2] == d_action, "输出动作维度错误"
    assert mu.shape == (bs, d_z_distribution), "mu 形状错误"
    assert logvar.shape == (bs, d_z_distribution), "logvar 形状错误"
    print("测试通过！")


if __name__ == '__main__':
    # 注意：确保 network.components.* 中各组件已经正确导入
    test_action_chunking_transformer()
