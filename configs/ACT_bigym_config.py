import torch


class Arguments:
    exp_name = "Bigym-ACT-Implementation"
    exp_day = "202502280950"
    exp_task = "DishwasherClose"  # 与模型保存有关, 训练时应事先看下这个变量
    exp_obs_type = "HeadRGB"  # 这里只使用头部 RGB, 如果使用其他信息则建议修改这个词条
    every_valid_step = 1000  # 每训练 1000 个 batch 就进行一次验证集评估
    every_test_step = 1000  # 每训练 1000 个 batch 就进行一次仿真器实际测试
    dtype = torch.float32  # 默认按照 32 位浮点数精度训练, 效果会好一点
    device = "cuda:0"  # 单卡情况下 "cuda:0" 或者 "cpu"
    record_video = False  # 默认不录制视频
    record_video_path = "./ACT_Implementation/videos/bigym/DishwasherClose"  # 如果录制视频, 应该放置的文件位置
    render = False  # False  # 默认不开渲染模式
    seed = 0  # 设置全局随机数种子
    torch_deterministic = True  # 置为 True 的话, 每次返回的卷积算法将是确定的, 即默认算法

    d_model = 512  # Transformer 模型的维度
    # Bigym 仿真器机器人本体维度和动作空间维度, 需要事先知道并手动填写: PickBox 是 70
    d_proprioception = 70
    d_action = 16
    d_z_distribution = 32  # 潜变量的维度
    num_heads = 8  # 自注意力的头数
    num_encoder_layers = 1  # 编码器的堆叠层数
    num_decoder_layers = 3  # 解码器的堆叠层数, 在简单的仿真器任务中进行模仿学习的话, 编码器和解码器的层数可以适当降低
    dropout = 0.1  # 正则化率

    chunk_size = 10  # 预测的动作块数量
    test_episode = 14  # 模型训练时放置于仿真环境测试的 episode 数
    # 模型未来动作的步数, ACT 论文是 100 ,这里用 10 可以增加数据集数量
    # 这里由于代码编写的原因, chunk_size 和 context_length 其实是同等数值!
    context_length = 10
    lr = 3e-4  # 全局模型学习率
    lr_min = 1e-5  # 使用余弦学习率调度器时模型学习率降低的最小值
    scale = 150  # 每条轨迹中采样的数据样本条数
    train_split = 0.8  # 训/验比是 8:2 且直接在仿真环境中部署做测试
    valid_split = 0.2  # 训/验比是 8:2 且直接在仿真环境中部署做测试
    batch_size = 128  # 因为图片很小, 所以 batch_size 可以调大一点
    num_step = 7500  # 每一个 step 表示一个 batch 做一次梯度下降
    kl_coefficient = 1  # 训练损失函数项中 KL 散度的权重, 原始 ACT 论文是 10, 仿真器任务中 1 会好一些.
