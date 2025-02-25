import copy

import torch
import torch.nn as nn
import torchvision.models as models
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from network.components.attention import Attention, SlotAttention
from network.components.ffnn import FeedForwardNetwork
from network.components.position_embedding import SinusoidalPositionEmbedding2D, SinusoidalPositionEmbedding1D
from network.components.weight_init_tool import weight_init


class ACTBackbone(nn.Module):
    """
    这一个类完成 ACT 模型的编码器和解码器两个实现, 目前的论文阅读中, 编码器/解码器的实现是近似的.
    """

    def __init__(self, d_model, num_heads, num_layers, dropout, device):
        super().__init__()

        self.num_layers = num_layers

        # 初始化 Attention 和 FFN 网络列表, 并在循环中进行初始化
        self.attention_list, self.ffn_list = [], []
        for _ in range(self.num_layers):
            self.attention_list.append(
                Attention(input_dim=d_model, output_dim=d_model, heads=num_heads).to(device)
            )
            self.ffn_list.append(
                FeedForwardNetwork(input_dim=d_model, hidden_dim=d_model * 6, output_dim=d_model, dropout=dropout).to(device)
            )

        # 对应的 LayerNorm 层
        self.layer_norm_1 = nn.LayerNorm(d_model).to(device)
        self.layer_norm_2 = nn.LayerNorm(d_model).to(device)

        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, query, key=None, value=None, mask=None):
        """
        ===> 1. 如果是自注意力, 只输入 query, 且 key 和 value 不需要输入;
        ===> 2. 在 ACT 实现中, 没有类似 BERT 或自回归形式的解码, 因此默认 mask 是 None
        :param query: tensor, (batch_size, seq_len_q, d_model)
        :param key: tensor, (batch_size, seq_len_k, d_model)
        :param value: tensor, (batch_size, seq_len_v, d_model)
        :param mask: None in this repo
        :return: tensor, (batch_size, seq_len_q, d_model)
        """
        for i in range(self.num_layers):
            # 对自注意力部分做处理
            if key is None:
                key = query
            if value is None:
                value = query
            # 注意力机制计算部分
            attn_output = self.attention_list[i](query, key, value, mask)  # tensor, (batch_size, seq_len, d_model)
            query = self.layer_norm_1(query + self.dropout(attn_output))  # 残差连接 + LayerNorm
            # Feed-Forward 部分
            ffn_output = self.ffn_list[i](query)
            query = self.layer_norm_2(query + self.dropout(ffn_output))  # 残差连接 + LayerNorm

        return query


class SlotBasedActionChunkTransformer(nn.Module):
    """
    # The Representation Encoder's input
    The input to the encoder are
        1) the [CLS] token, which consists of learned weights that are randomly initialized.
        2) embedded joint positions, which are joint positions projected to the embedding dimension
            using a linear layer.
        3) embedded action sequence, which is the action sequence projected to the embedding dimension
            using another linear layer.

    # The Representation Encoder's output
    We only take the first output, which corresponds to the [CLS] token, and use another linear network to predict
        the mean and variance of z’s distribution, parameterizing it as a diagonal Gaussian.

    # The ACT's encoder input
    For each of the image observations, it is first processed by a ResNet18 to obtain a feature map, and then
        flattened to get a sequence of features. These features are projected to the embedding dimension with a
        linear layer, and we add a 2D sinusoidal position embedding to perserve the spatial information. The feature
        sequence from each camera is then concatenated to be used as input to the transformer encoder.
    Two additional inputs are joint positions and z, which are also projected to the embedding dimension with two
        linear layers respectively.

    # The ACT's encoder output
    The output of the transformer encoder are then used as both "keys" and "values" in cross attention layers of the
        transformer decoder, which predicts action sequence given encoder output.

    # The ACT's decoder input
    The "queries" are fixed sinusoidal embeddings for the first layer.
    The output of ACT's decoder is the predicted action chunk.
    """

    def __init__(self, d_model, d_proprioception, d_action, d_z_distribution,
                 num_heads, num_encoder_layers, num_decoder_layers, num_slots, dropout,
                 dtype, device):
        super(SlotBasedActionChunkTransformer, self).__init__()

        # 这里做一些简单的输入变量检查和变换
        assert d_z_distribution % 2 == 0, "d_z_distribution 维度必须是偶数, 建议 64, 128, ..."
        if isinstance(device, str):
            device = device if torch.cuda.is_available() else "cpu"
            device = torch.device(device)

        # =========> 表征编码器网络部分
        # 1. 设置 [CLS] token, 变换成维度是 (1, 1, 1) 的张量, 并构建一个可学习的网络, 将维度 (1, 1, 1) 映射到 (1, 1, d_model)
        #    还有一种写法是直接将 [CLS] token 初始化成成维度是 (1, 1, d_model) 的张量, 但是这样写似乎不够直观...
        self.CLS_token_index = torch.tensor([1]).unsqueeze(0).unsqueeze(1).to(device=device, dtype=dtype)
        self.CLS_token_proj = nn.Linear(1, d_model)
        self.CLS_token_proj.apply(weight_init)

        # 2. 将本体数据通过线性层映射到 d_model 维度
        self.proprio_proj = nn.Linear(d_proprioception, d_model)
        self.proprio_proj.apply(weight_init)

        # 3. 将动作序列从动作空间维度映射到 d_model 维度
        self.action_seq_proj = nn.Linear(d_action, d_model)
        self.action_seq_proj.apply(weight_init)

        # 4. 将 Representation Encoder 的 [CLS] token 对应的张量通过线性层计算出 z 分布的均值和方差
        #    一个线性层直接映射出整个均值和方差, 然后再按照维度划分出两个张量
        self.z_mean_std = nn.Linear(d_model, d_z_distribution * 2)
        self.z_mean_std.apply(weight_init)

        # 5. 表征编码器的主体部分
        self.representation_encoder = ACTBackbone(d_model, num_heads, num_encoder_layers, dropout, device)
        self.representation_encoder.to(device)

        # =========> ACT 编码器网络部分
        # 1. 这里导入 ResNet18, 冻结预训练模型并在模型后面增加线性适应层调整维度映射, 用以获得图像观测的特征向量
        #    参考链接: https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-models/
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # 去掉最后的全连接层和池化层

        # 2. 设置特征图映射线性层，用于映射到整个模型的维度 d_model
        self.linear = nn.Linear(512, d_model)  # ResNet18 的输出通道是 512
        self.linear.apply(weight_init)

        # 3. 设置二维正弦位置嵌入层
        self.position_embedding_2d = SinusoidalPositionEmbedding2D(d_model)

        # 3. 设置基于槽的注意力机制, 对输入图片的 token 占用数降维, 如何降维可看 forward 方法实现
        self.slot_attention = SlotAttention(num_slots, d_model)
        self.slot_attention.to(device=device)

        # 4. 将本体数据通过线性层映射到 d_model 维度
        self.act_propr_proj = nn.Linear(d_proprioception, d_model)
        self.act_propr_proj.apply(weight_init)

        # 5. 将浅层 z 变量通过线性层映射到 d_model 维度
        self.act_z_proj = nn.Linear(d_z_distribution, d_model)
        self.act_z_proj.apply(weight_init)

        # 6. ACT 编码器网络部分
        self.ACT_encoder = ACTBackbone(d_model, num_heads, num_encoder_layers, dropout, device)
        self.ACT_encoder.to(device)

        # =========> ACT 解码器网络部分
        # 1. 设置一维正弦位置嵌入层
        self.position_embedding_1d = SinusoidalPositionEmbedding1D(d_model)
        # 2. ACT 解码器网络部分
        self.ACT_decoder = ACTBackbone(d_model, num_heads, num_decoder_layers, dropout, device)
        self.ACT_decoder.to(device)
        # 3. 设置动作解码网络
        self.action_seq_deproj = nn.Linear(d_model, d_action)
        self.action_seq_deproj.apply(weight_init)

        # =========> 做一些全局参数的保留
        self.d_z_distribution = d_z_distribution
        self.d_model = d_model
        self.d_action = d_action
        self.device = device
        self.dtype = dtype

    def forward(self, image_t, proprioception_t, args, action_chunk_t_tk=None, mask=None, inference_mode=False):
        """
        ACT 网络前向传播部分, 这是端到端的哦~ 参考网页: https://arxiv.org/pdf/2304.13705#page=13.33

        :param image_t: tensor, (batch_size, seq_len_image, channel, height, width) seq_len_image 实际就是相机视角的数量
        :param proprioception_t: tensor, (batch_size, 1, proprioception_dim) proprioception_dim 本体维度
        :param args: class, 传递整个训练过程的参数设置
        :param action_chunk_t_tk: tensor, (batch_size, k, action_dim) action_dim 动作空间维度
        :param mask: tensor, 这是个预留参数, 若以后改模型涉及到掩码部分, 可以提供一个参数入口
        :param inference_mode: 推理模式, 当模型在仿真环境中测试时使用, 训练和验证过程不用

        :return: tensor, (batch_size, seq_len_action, action_dim) action_dim 动作空间维度
        """
        # 解析维度变量
        batch_size, seq_len_image, channel, height, width = image_t.shape
        proprioception_t_for_act = copy.deepcopy(proprioception_t)  # 拷贝一份张量给 ACT 做输入

        if inference_mode:
            z = torch.zeros((batch_size, 1, self.d_z_distribution)).to(dtype=self.dtype, device=self.device)
        else:
            _, chunk_size, _ = action_chunk_t_tk.shape  # 训练过程和验证过程都要输入动作块
            # =========> 表征编码器前向传播
            # the [CLS] token
            cls_token = self.CLS_token_proj(self.CLS_token_index)  # (1, 1, d_model)
            cls_token = cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, d_model)

            # proprioception
            proprioception_t = proprioception_t.reshape(batch_size * 1, -1)
            proprioception_t = self.proprio_proj(proprioception_t)
            proprioception_t = proprioception_t.reshape(batch_size, 1, -1)  # (batch_size, 1, d_model)

            # action sequence
            action_chunk_t_tk = action_chunk_t_tk.reshape(batch_size * chunk_size, -1)
            action_chunk_t_tk = self.action_seq_proj(action_chunk_t_tk)
            action_chunk_t_tk = action_chunk_t_tk.reshape(batch_size, chunk_size, -1)  # (batch_size, k, d_model)

            # 合并 the [CLS] token / proprioception 和 action sequence, (batch_size, k+2, d_model)
            representation_encoder_input = torch.concat([cls_token, proprioception_t, action_chunk_t_tk], dim=1)
            representation_encoder_output = self.representation_encoder(representation_encoder_input)

            # 取第一个位置, 也就是 [CLS] token 对应的输出的张量作为 z 分布网络输入
            # TODO: 做一个实验测试取最后一个 token 的张量是不是会更好?
            # 回答: 根据论文 https://arxiv.org/pdf/2304.13705#page=8.86 图片里面取的是也就是 [CLS] token 对应的位置
            z_input = representation_encoder_output[:, 0, :]  # (batch_size, 1, d_model)

            # 输入 z_input 获得 z 分布的均值和方差, 并均等 2 分得到各自的均值和方差
            z_mean_and_std = self.z_mean_std(z_input)  # (batch_size, 1, d_z_distribution * 2)
            z_mean, z_std = z_mean_and_std.split(self.d_z_distribution, dim=-1)
            # 训练模式和验证模式用重参数化采样, 仿真器测试模式直接使用 0 向量
            z_distribution = Normal(loc=z_mean, scale=z_std.exp())
            z = z_distribution.rsample()

            # 这里再额外计算 z 分布与标准正态分布的 KL 散度损失项
            z_standard_distribution = Normal(
                loc=torch.zeros_like(z_mean).to(dtype=self.dtype, device=self.device),
                scale=torch.ones_like(z_std).to(dtype=self.dtype, device=self.device)
            )
            z_kl = kl_divergence(z_distribution, z_standard_distribution).mean()


        # =========> ACT 编码器前向传播
        # 使用 ResNet18 提取特征图
        image_t = image_t.reshape(batch_size * seq_len_image, channel, height, width)
        feature_map = self.resnet(image_t)  # (batch_size * seq_len_image, 512, height // 32, width // 32)
        # 展平特征图
        batch_size_m_seq_len_image, channels_512, height_32, width_32 = feature_map.shape
        # (batch_size_m_seq_len_image, height_32*width_32, channels_512)
        feature_map = feature_map.view(batch_size_m_seq_len_image, channels_512, -1).transpose(1, 2)
        # 通过线性层映射到嵌入维度: channels_512 ==> d_model
        feature_map = self.linear(feature_map)  # (batch_size_m_seq_len_image, height_32*width_32, d_model)
        # 获取 2D 位置嵌入
        # TODO: 修改 BUG (20250222 已解决)
        pos_embedding = self.position_embedding_2d(height_32, width_32)  # (height_32, width_32, d_model)
        # 扩展位置嵌入并加到特征上
        # (batch_size_m_seq_len_image, height_32*width_32, d_model)
        pos_embedding = pos_embedding.flatten(0, 1).unsqueeze(0).repeat(batch_size_m_seq_len_image, 1, 1)
        # 添加位置嵌入 (batch_size_m_seq_len_image, height_32*width_32, d_model)
        feature_map = feature_map + pos_embedding.to(self.device)
        # 还原成本质维度 (batch_size, seq_len_image * height_32 * width_32, d_model)
        feature_map = feature_map.reshape(batch_size, seq_len_image * height_32 * width_32, self.d_model)

        # ======> 使用槽模型把 seq_len_image * height_32 * width_32 降低成 num_slots
        # (batch_size, num_slots, d_model)
        feature_map = self.slot_attention(feature_map)

        # 将本体数据做映射
        proprioception_t_for_act = proprioception_t_for_act.reshape(batch_size * 1, -1)
        proprioception_t_for_act = self.act_propr_proj(proprioception_t_for_act)
        proprioception_t_for_act = proprioception_t_for_act.reshape(batch_size, 1, -1)

        # 将 z 变量做映射
        z = z.reshape(batch_size * 1, -1)
        z = self.act_z_proj(z)
        z = z.reshape(batch_size, 1, -1)

        # 合并 相机特征序列 / 本体数据 和 z 映射变量
        # (batch_size, seq_len_image*height_32*width_32 + 2, d_model)
        act_encoder_input = torch.cat([feature_map, proprioception_t_for_act, z], dim=1)
        # 获得 act 编码器的输出
        # (batch_size, seq_len_image*height_32*width_32 + 2, d_model)
        act_encoder_output = self.ACT_encoder(act_encoder_input)

        # =========> ACT 解码器前向传播
        chunk_size = args.chunk_size  # 如果是仿真器测试模式时, chunk_size 不可知, 因此需要导入参数设置来获得 chunk_size
        # 获得动作槽 (batch_size, chunk_size, d_model)
        action_slots = torch.zeros((batch_size, chunk_size, self.d_model))
        # 获取 1 维度正弦位置编码 (batch_size, chunk_size, d_model)
        action_position_embedding = self.position_embedding_1d(chunk_size).unsqueeze(0).repeat(batch_size, 1, 1)
        # 二者相加作为 ACT 解码器的输入 (batch_size, chunk_size, d_model)
        act_decoder_input = (action_slots + action_position_embedding).to(self.device)
        act_decoder_output = self.ACT_decoder(act_decoder_input, act_encoder_output, act_encoder_output)

        # 将 act_decoder_output 的输出结果映射到实际的动作空间维度中
        batch_size, chunk_size, d_model = act_decoder_output.shape
        act_decoder_output = act_decoder_output.reshape(batch_size * chunk_size, d_model)
        action_pred = self.action_seq_deproj(act_decoder_output)
        action_pred = action_pred.reshape(batch_size, chunk_size, self.d_action)

        if inference_mode:
            return action_pred
        else:
            return action_pred, z_kl

# 用于 debug
# if __name__ == "__main__":
#
#     class Args:
#
#         chunk_size = 66
#
#     args = Args()
#
#     batch_size, seq_len_image, channel, height, width, proprioception_dim, action_dim = 32, 4, 3, 84, 84, 66, 15
#     image_t = torch.rand((batch_size, seq_len_image, channel, height, width))
#     proprioception_t = torch.rand((batch_size, 1, proprioception_dim))
#     action_chunk_t_tk = torch.rand((batch_size, args.chunk_size, action_dim))
#     model = SlotBasedActionChunkTransformer(512, proprioception_dim, action_dim, 256, 8, 3, 7, 3,0.1, torch.float32, "cpu")
#     action_pred, z_kl = model(image_t, proprioception_t, args, action_chunk_t_tk, None, False)
#     print(action_chunk_t_tk.shape)
#     print(action_pred.shape)
