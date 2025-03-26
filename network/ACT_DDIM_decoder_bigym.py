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
        self.d_action = d_action
        self.d_z_distribution = d_z_distribution
        self.normalize_before = normalize_before
        self.n_frame_stack = n_frame_stack

        # --------------------------
        # 扩散过程相关参数设置
        # --------------------------
        self.num_diffusion_steps = 500  # 扩散步数, 可根据需要调整
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_diffusion_steps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)  # (num_diffusion_steps,)
        # 将这些张量注册为 buffer, 确保在 GPU 上移动
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        # 时间步嵌入，用于将时间信息注入到解码器输入中
        self.time_embed = nn.Embedding(self.num_diffusion_steps, d_model).apply(weight_init)

        # --------------------------
        # 表征编码器部分
        # --------------------------
        # [cls] token 的映射，原本用 nn.Linear 但是不合理, 故改之
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
                for _ in range(n_representation_encoder_layers)
            ]
        )
        if self.normalize_before:
            self.representation_encoder_output_layer_norm = nn.LayerNorm(d_model)

        # --------------------------
        # ACT 编码器-解码器部分
        # --------------------------
        # 将采样的 z 分布向量线性映射到模型维度, 作为一个 token
        self.latent_out_proj = nn.Linear(d_z_distribution, d_model).apply(weight_init)

        # 将输入的本体数据线性映射到模型维度, 作为一个 token (frame_stack 存在)
        self.input_proprio_proj = nn.Linear(d_proprioception * n_frame_stack, d_model).apply(weight_init)

        # 解码器端输入的预测动作长度, 也就是 chunk_size = num_queries
        self.query_embed = nn.Embedding(chunk_size, d_model).apply(weight_init)

        # 在去噪过程中将加了噪声的动作, 线性映射到模型维度
        self.noised_action_proj = nn.Linear(d_action, d_model).apply(weight_init)

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
                for _ in range(n_encoder_layers)
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
                for _ in range(n_decoder_layers)
            ]
        )
        if self.normalize_before:
            self.ACT_encoder_output_layer_norm = nn.LayerNorm(d_model)

        # --------------------------
        # 视觉骨干网络及后处理部分
        # --------------------------
        self.backbones = VisionBackbone(d_model, resnet_name, return_interm_layers, include_depth)
        self.input_proj = nn.Conv2d(self.backbones.num_channels, d_model, kernel_size=1).apply(weight_init)

        # 将预测动作序列从 d_model 维度线性映射到 d_action 维度
        self.action_head = nn.Linear(d_model, d_action).apply(weight_init)

    def get_diffusion_coeff(self, t):
        """
        时间步 t 的维度是 (batch_size,) 每个值在 [0, num_diffusion_steps-1] 区间
        这个函数将返回扩散系数:
            sqrt_alpha_bar: 用于缩放原始动作数据
            sqrt_one_minus_alpha_bar: 用于缩放噪声
        输出形状为 (batch_size, 1, 1), 便于后续广播到 (batch_size, seq_len, d_action)
        """
        # t 为长整型张量，取出对应的 alpha_bar 值
        alpha_bar_t = self.alpha_bars[t]  # shape: (batch_size,)
        sqrt_alpha_bar = alpha_bar_t.sqrt().unsqueeze(1).unsqueeze(2)  # (bs, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - alpha_bar_t).sqrt().unsqueeze(1).unsqueeze(2)  # (bs, 1, 1)
        return sqrt_alpha_bar, sqrt_one_minus_alpha_bar

    def forward(self, obs, actions):
        """
        obs = {
            "state": (batch_size, d_proprioception * frame_stack),
            "rgb": (batch_size, num_views, channels * frame_stack, height, width),
        }
        actions = (batch_size, action_seq_len, d_action)

        修改后的 forward 仅仅支持训练 (当输入动作时进行) 过程
        并在 ACT 解码器部分完成条件去噪扩散过程, 输出预测时间步相关的噪声
        """

        state = obs['state'] if self.backbones is not None else obs

        bs = state.shape[0]

        # ==========> 表征编码器部分
        # [cls] token 的处理：将其复制到整个 batch 的数据样本中
        cls_embed = self.cls_embed.weight  # (1, d_model)
        cls_embed = cls_embed.unsqueeze(0).repeat(bs, 1, 1)  # (bs, 1, d_model)

        # 本体数据处理：将每个本体数据映射到 d_model，并扩充一个维度
        state_embed = self.proprio_proj(state)  # (bs, d_model)
        state_embed = state_embed.unsqueeze(1)  # (bs, 1, d_model)

        # 动作序列处理：将每个动作序列映射到 d_model
        action_embed = self.action_proj(actions)  # (bs, seq_len, d_model)

        # 拼接 [cls] token、本体数据和（训练时的）动作序列作为表征编码器的输入
        encoder_input = torch.cat([cls_embed, state_embed, action_embed], dim=1)
        # 同时获得位置嵌入，encoder_input 和 pos_embed 的形状均为 (bs, seq_len+2, d_model)
        pos_embed = self.position_embedding_1d(encoder_input)

        # 修改维度：transpose 成 (seq_len+2, bs, d_model)
        encoder_input = encoder_input.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        # 构建一个全 False 的 pad 掩码，形状 (bs, seq_len+2)
        is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device)

        # 表征编码器前向传播（层与层之间将上一层输出作为输入）
        for layer in self.representation_encoder:
            encoder_input = layer(
                src=encoder_input,
                src_mask=None,
                src_key_padding_mask=is_pad,
                pos=pos_embed
            )
        if self.normalize_before:
            encoder_input = self.representation_encoder_output_layer_norm(encoder_input)

        # 取第 0 个位置，即 [cls] token 对应的输出 (bs, d_model)
        encoder_output = encoder_input[0]

        # 将 [cls] token 输出映射到 z 分布的均值和对数方差向量
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, :self.d_z_distribution]
        logvar = latent_info[:, self.d_z_distribution:]
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        z = mu + std * eps
        # 将采样的 z 映射到 d_model，作为一个 token
        latent_input = self.latent_out_proj(z)

        # ==========> ACT 编码器-解码器部分
        # 处理多视角 RGB 数据
        vis_data = obs['rgb']  # (bs, num_views, channels * frame_stack, height, width)
        batch_size, num_view, channel_framestack, height, width = vis_data.shape
        channel = channel_framestack // self.n_frame_stack  # 原始 channel 数
        # 重构为 (bs, num_view, frame_stack, channel, height, width)
        vis_data = vis_data.view(batch_size, num_view, self.n_frame_stack, channel, height, width)
        # 交换 num_view 与 frame_stack 维度 -> (bs, frame_stack, num_view, channel, height, width)
        vis_data = vis_data.transpose(1, 2)
        # 合并 frame_stack 和 num_view 维度 -> (bs, frame_stack * num_view, channel, height, width)
        vis_data = vis_data.reshape(batch_size, num_view * self.n_frame_stack, channel, height, width)
        # 如果存在深度数据，则在第 2 个维度拼接
        if "depth" in obs:
            vis_data = torch.cat([vis_data, obs['depth']], dim=2)
        num_cams = vis_data.shape[1]  # frame_stack * num_view

        # 图像观测特征提取与位置编码
        all_cam_features, all_cam_pos = [], []
        for cam_id in range(num_cams):
            features, pos = self.backbones(
                NestedTensor(vis_data[:, cam_id], None)
            )
            features = features[0].tensors  # (bs, d_model, height, width)
            pos = pos[0]  # (1, d_model, height, width)
            all_cam_features.append(self.input_proj(features))  # (bs, d_model, height, width)
            all_cam_pos.append(pos)

        # 将本体数据映射到模型维度，作为一个 token
        proprio_input = self.input_proprio_proj(state)

        # ==========> ACT 编码器部分
        # 整合所有图像特征成为一个 token 序列
        src = torch.cat(all_cam_features, dim=3)  # (bs, d_model, height, width*num_views)
        src = src.flatten(2).permute(2, 0, 1)  # (height*width*num_views, bs, d_model)
        # 整合 z token 和本体 token -> (2, bs, d_model)
        addition_input = torch.stack([latent_input, proprio_input], dim=0)
        src = torch.cat([addition_input, src], dim=0)

        # 整合所有图像位置编码成为一个 token 序列
        pos_embed = torch.cat(all_cam_pos, dim=3)  # (1, d_model, height, width*num_views)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)  # (height*width*num_views, bs, d_model)
        # 为 z token 和本体 token 构建 2 token 的位置编码 -> (2, bs, d_model)
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # 整合后得到最终位置编码 -> (2+height*width*num_views, bs, d_model)
        pos_embed = torch.cat([additional_pos_embed, pos_embed], dim=0)

        # ==========> ACT 解码器部分: 条件去噪扩散过程
        # 训练模式: 对 actions 按随机采样的时间步 t 加入噪声
        # actions: (bs, chunk_size, d_action)
        t = torch.randint(0, self.num_diffusion_steps, (bs,), device=state.device)  # (bs,)
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_diffusion_coeff(t)  # (bs, 1, 1)
        noise = torch.randn_like(actions)  # (bs, chunk_size, d_action)
        noisy_actions = sqrt_alpha_bar * actions + sqrt_one_minus_alpha_bar * noise
        # 将噪声加入后的动作经过线性映射 -> (bs, chunk_size, d_model)
        tgt = self.noised_action_proj(noisy_actions)
        # 加入时间步嵌入信息
        time_emb = self.time_embed(t)  # (bs, d_model)
        tgt = tgt + time_emb.unsqueeze(1)  # (bs, chunk_size, d_model)

        # 转置 tgt 为 (chunk_size, bs, d_model)，以符合 TransformerDecoder 要求
        tgt = tgt.transpose(0, 1)

        # ACT 编码器前向传播，得到 key_and_value 张量作为条件
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

        # ACT 解码器前向传播：利用条件信息预测与时间步相关的噪声
        for layer in self.ACT_decoder:
            tgt = layer(
                tgt=tgt,
                memory=key_and_value,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=pos_embed,
                query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            )
        hidden_state = tgt.transpose(1, 0)  # (bs, chunk_size, d_model)

        # 通过 action_head 将隐藏状态映射到 d_action，输出预测的噪声
        predicted_noise = self.action_head(hidden_state)
        return (predicted_noise, noise), [mu, logvar]

    def inference(self, obs, num_steps=None):
        """
        多步去噪推理 (使用 DDIM 采样):
        1. 首先利用 obs 计算条件信息 (即 representation encoder 与 ACT encoder 部分), 得到 key_and_value 与位置编码 pos_embed_total;
        2. 从纯噪声 x 开始，迭代反向扩散更新 (采用 DDIM 更新公式)，最终输出去噪后的动作预测.
        返回去噪后的动作 predicted_actions, 形状为 (batch_size, chunk_size, d_action)
        """
        device = obs["state"].device
        bs = obs["state"].shape[0]
        chunk_size = self.query_embed.num_embeddings
        d_action = self.action_head.out_features
        num_steps = num_steps if num_steps is not None else self.num_diffusion_steps

        # ==========> 计算条件信息 (与 forward 中相同的 representation encoder & ACT encoder 部分)
        # z 分布只使用全 0 张量
        latent_sample = torch.zeros([bs, self.d_z_distribution], dtype=torch.float32).to(device)
        # 将重参数化采样的 z 张量线性映射到 d_model 作为一个 token
        latent_input = self.latent_out_proj(latent_sample)

        # 视觉部分：处理 RGB 数据
        vis_data = obs['rgb']  # (bs, num_views, channels * frame_stack, height, width)
        batch_size, num_view, channel_framestack, height, width = vis_data.shape
        channel = channel_framestack // self.n_frame_stack
        vis_data = vis_data.view(batch_size, num_view, self.n_frame_stack, channel, height, width)
        vis_data = vis_data.transpose(1, 2)  # (bs, frame_stack, num_view, channel, height, width)
        vis_data = vis_data.reshape(batch_size, num_view * self.n_frame_stack, channel, height, width)
        if "depth" in obs:
            vis_data = torch.cat([vis_data, obs['depth']], dim=2)
        num_cams = vis_data.shape[1]
        all_cam_features, all_cam_pos = [], []
        for cam_id in range(num_cams):
            features, pos = self.backbones(
                NestedTensor(vis_data[:, cam_id], None)
            )
            features = features[0].tensors  # (bs, d_model, height, width)
            pos = pos[0]  # (1, d_model, height, width)
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)

        proprio_input = self.input_proprio_proj(obs["state"])  # (bs, d_model)

        # ACT 编码器部分：整合视觉特征与隐变量
        src = torch.cat(all_cam_features, dim=3)  # (bs, d_model, height, width*num_views)
        src = src.flatten(2).permute(2, 0, 1)  # (height*width*num_views, bs, d_model)
        addition_input = torch.stack([latent_input, proprio_input], dim=0)  # (2, bs, d_model)

        src = torch.cat([addition_input, src], dim=0)

        pos_embed_cam = torch.cat(all_cam_pos, dim=3)  # (1, d_model, height, width*num_views)
        pos_embed_cam = pos_embed_cam.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        pos_embed_total = torch.cat([additional_pos_embed, pos_embed_cam], dim=0)

        for layer in self.ACT_encoder:
            src = layer(
                src=src,
                src_mask=None,
                src_key_padding_mask=None,
                pos=pos_embed_total
            )
        if self.normalize_before:
            src = self.ACT_encoder_output_layer_norm(src)
        key_and_value = src

        # --------------------------
        # 推理：多步反向扩散迭代 (使用 DDIM 更新公式)
        # --------------------------
        # 初始化 x 为纯噪声: 作为最终 t = num_steps 时的动作样本
        x = torch.randn(bs, chunk_size, d_action, device=device)
        for t in reversed(range(num_steps)):
            t_tensor = torch.full((bs,), t, device=device, dtype=torch.long)  # (bs,)
            # 根据当前 t, 计算时间步扩散系数, 注意 DDIM 在这里没有 alpha_t 和 beta_t 两个, 它在下面使用了
            sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.get_diffusion_coeff(t_tensor)  # (bs,1,1)
            # 构造 ACT 解码器输入: 利用 action_proj 对当前 x 映射，并加入对应 t 的时间嵌入
            tgt = self.noised_action_proj(x)  # (bs, chunk_size, d_model)
            time_emb = self.time_embed(t_tensor)  # (bs, d_model)
            tgt = tgt + time_emb.unsqueeze(1)  # (bs, chunk_size, d_model)
            tgt = tgt.transpose(0, 1)  # (chunk_size, bs, d_model)
            # 通过 ACT 解码器预测当前噪声
            dec_t = tgt
            for layer in self.ACT_decoder:
                dec_t = layer(
                    tgt=dec_t,
                    memory=key_and_value,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos_embed_total,
                    query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
                )
            hidden_state = dec_t.transpose(1, 0)  # (bs, chunk_size, d_model)
            predicted_noise = self.action_head(hidden_state)  # (bs, chunk_size, d_action)

            # DDIM 更新公式：
            # 预测 x0
            x0_pred = (x - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
            # 获取前一时间步 t-1 的扩散系数 (当 t==0 时，直接输出 x0_pred)
            t_minus = (t_tensor - 1).clamp(min=0)
            sqrt_alpha_bar_prev = self.alpha_bars[t_minus].sqrt().unsqueeze(1).unsqueeze(2)
            sqrt_one_minus_alpha_bar_prev = (1 - self.alpha_bars[t_minus]).sqrt().unsqueeze(1).unsqueeze(2)
            # DDIM 更新: x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred + sqrt(1 - alpha_bar_{t-1}) * predicted_noise
            # 这里以 eta = 0，即确定性更新
            x = sqrt_alpha_bar_prev * x0_pred + sqrt_one_minus_alpha_bar_prev * predicted_noise

        predicted_actions = x  # 最终去噪后的动作预测
        return predicted_actions
