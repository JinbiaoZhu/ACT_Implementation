import copy

import torch
import torch.nn as nn
from torch.autograd import Variable

from network.components.position_embedding import PositionEmbedding_Sine_1D
from network.components.transformer_encoder import TransformerEncoderLayer
from network.components.transformer_decoder import TransformerDecoderLayer
from network.components.visual_feature_extractor import VisionBackbone, NestedTensor
from network.components.tools import weight_init


class ActionChunkingTransformer(nn.Module):

    def __init__(self,
                 d_model, d_proprioception, d_action, d_z_distribution, d_feedforward,
                 n_head, n_representation_encoder_layers, n_encoder_layers, n_decoder_layers, n_frame_stack,
                 chunk_size,  # 这一行是动作长度参数, chunk_size 相当于 num_queries
                 resnet_name: str, return_interm_layers: bool, include_depth: bool,  # 这一行是视觉骨干网络参数
                 dropout, activation, normalize_before: bool
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_z_distribution = d_z_distribution
        self.normalize_before = normalize_before
        self.n_frame_stack = n_frame_stack

        # 表征编码器部分
        # [cls] token 的映射, 原本用 nn.Linear 但是不合理, 故改之
        self.cls_embed = nn.Embedding(1, d_model).apply(weight_init)

        # 将本体维度线性映射到模型维度 (frame_stack 存在)
        self.proprio_proj = nn.Linear(d_proprioception * n_frame_stack, d_model).apply(weight_init)

        # 将动作维度线性映射到模型维度
        self.action_proj = nn.Linear(d_action, d_model).apply(weight_init)

        # 输出序列中 [cls] token 对应向量线性映射到 z 分布的均值和方差向量
        self.latent_proj = nn.Linear(d_model, d_z_distribution * 2)

        # 1 维位置嵌入, 用于处理序列数据
        self.position_embedding_1d = PositionEmbedding_Sine_1D(d_model)

        # 表征编码器的核心
        self.representation_encoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerEncoderLayer(
                        d_model, n_head, d_feedforward, dropout, activation, normalize_before
                    ).apply(weight_init)
                )
                for i in range(n_representation_encoder_layers)
            ]
        )
        if self.normalize_before:
            self.representation_encoder_output_layer_norm = nn.LayerNorm(d_model)

        # ACT 编码器-解码器部分
        # 将采样的 z 分布向量线性映射到模型维度, 作为一个 token
        self.latent_out_proj = nn.Linear(d_z_distribution, d_model).apply(weight_init)

        # 将输入的本体数据线性映射到模型维度, 作为一个 token (frame_stack 存在)
        self.input_proprio_proj = nn.Linear(d_proprioception * n_frame_stack, d_model).apply(weight_init)

        # 解码器端输入的预测动作长度, 也就是 chunk_size = num_queries
        self.query_embed = nn.Embedding(chunk_size, d_model).apply(weight_init)

        # 为 z 表征和本体嵌入张量设计一个可学习的位置编码
        self.additional_pos_embed = nn.Embedding(2, d_model).apply(weight_init)

        # ACT 编码器的核心
        self.ACT_encoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerEncoderLayer(
                        d_model, n_head, d_feedforward, dropout, activation, normalize_before
                    ).apply(weight_init)
                )
                for i in range(n_encoder_layers)
            ]
        )
        # ACT 解码器的核心
        self.ACT_decoder = nn.ModuleList(
            [
                copy.deepcopy(
                    TransformerDecoderLayer(
                        d_model, n_head, d_feedforward, dropout, activation, normalize_before
                    ).apply(weight_init)
                )
                for i in range(n_decoder_layers)
            ]
        )
        if self.normalize_before:
            self.ACT_encoder_output_layer_norm = nn.LayerNorm(d_model)

        # 预训练视觉模型骨干, 和表征后处理
        self.backbones = VisionBackbone(d_model, resnet_name, return_interm_layers, include_depth)
        self.input_proj = nn.Conv2d(self.backbones.num_channels, d_model, kernel_size=1).apply(weight_init)

        # 将预测动作序列从 d_model 维度线性映射到 d_action 维度
        self.action_head = nn.Linear(d_model, d_action).apply(weight_init)

    def forward(self, obs, actions=None):
        """
        obs = {
            "state": (batch_size, d_proprioception * frame_stack),
            "rgb": (batch_size, num_views, channels * frame_stack, height, width),
        }
        actions = (batch_size, action_seq_len, d_action)
        """
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
        # 对多视角处理, 首先提取 RGB 数据, (batch_size, num_view, channel * frame_stack, height, width)
        vis_data = obs['rgb']
        batch_size, num_view, channel_framestack, height, width = vis_data.shape
        channel = channel_framestack // self.n_frame_stack  # 计算原始 channel 数
        # 重构为 (batch_size, num_view, frame_stack, channel, height, width)
        vis_data = vis_data.view(batch_size, num_view, self.n_frame_stack, channel, height, width)
        # 交换 num_view 与 frame_stack 维度, 得到 (batch_size, frame_stack, num_view, channel, height, width)
        vis_data = vis_data.transpose(1, 2)
        # 合并 frame_stack 和 num_view 维度，得到 (batch_size, frame_stack * num_view, channel, height, width)
        vis_data = vis_data.reshape(batch_size, num_view * self.n_frame_stack, channel, height, width)
        # 如果观测 obs 数据中有深度数据的话, 在第 2 个维度做拼接, 相当于额外增加一个通道数
        if "depth" in obs:
            vis_data = torch.cat([vis_data, obs['depth']], dim=2)
        # 记录可以使用的视角数量
        num_cams = vis_data.shape[1]  # frame_stack * num_view

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
        # (height*width*num_views, batch_size, d_model)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        # 为 z 张量 和 本体数据嵌入 构建一个 2 token 的位置编码序列
        # (2, batch_size, d_model)
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
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
    # --- 设置超参数 ---
    d_model = 128
    d_proprioception = 10
    d_action = 5
    d_z_distribution = 16
    d_feedforward = 256
    n_head = 8
    n_representation_encoder_layers = 2
    n_encoder_layers = 2
    n_decoder_layers = 2
    n_frame_stack = 4
    chunk_size = 16  # 相当于 num_queries
    resnet_name = "resnet18"
    return_interm_layers = False
    include_depth = False
    dropout = 0.1
    activation = "relu"
    normalize_before = True

    # --- 构建网络实例 ---
    model = ActionChunkingTransformer(
        d_model, d_proprioception, d_action, d_z_distribution, d_feedforward,
        n_head, n_representation_encoder_layers, n_encoder_layers, n_decoder_layers, n_frame_stack,
        chunk_size, resnet_name, return_interm_layers, include_depth,
        dropout, activation, normalize_before
    )

    # --- 设置测试输入 ---
    batch_size = 7
    # 本体状态: shape = (batch_size, d_proprioception * frame_stack)
    state = torch.randn(batch_size, d_proprioception * n_frame_stack)

    # RGB 数据: shape = (batch_size, num_views, channels * frame_stack, height, width)
    num_views = 1
    channels = 3  # RGB
    height = 84
    width = 84
    rgb = torch.randn(batch_size, num_views, channels * n_frame_stack, height, width)

    # 构造 obs 字典
    obs = {
        "state": state,
        "rgb": rgb,
        # 如果 include_depth 为 True，可添加 "depth": depth_tensor
    }

    # 训练模式下需要动作输入: shape = (batch_size, action_seq_len, d_action)
    action_seq_len = 8  # 这里选一个任意的动作序列长度
    actions = torch.randn(batch_size, action_seq_len, d_action)

    # --- 训练模式前向传播 ---
    model.train()  # 设置训练模式
    a_hat_train, (mu_train, logvar_train) = model(obs, actions=actions)
    print("训练模式输出：")
    print("a_hat_train shape:", a_hat_train.shape)
    print("mu_train shape:", mu_train.shape)
    print("logvar_train shape:", logvar_train.shape)

    # --- 推理模式前向传播 ---
    model.eval()  # 设置推理模式
    with torch.no_grad():
        a_hat_infer, _ = model(obs, actions=None)
    print("\n推理模式输出：")
    print("a_hat_infer shape:", a_hat_infer.shape)


if __name__ == "__main__":
    test_action_chunking_transformer()
