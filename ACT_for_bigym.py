import time
from dataclasses import dataclass
from datetime import datetime
import random

import tyro
import yaml
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import wandb

from network.ACT_bigym import ActionChunkingTransformer
from datasets.bigym_datasets import simple_env_and_dataloader
from env.make_bigym_envs import test_in_simulation
from utils import kl_divergence, save_ckpt


@dataclass
class Args:
    config_file_path: str = "configs/ACT_bigym_config.yaml"


# 因为各种原因, 可能配置文件不在 configs/ 文件夹内, 因此用终端传入轨迹
args = tyro.cli(Args)

if __name__ == "__main__":
    # ==========> 载入参数
    with open(args.config_file_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)  # 使用 safe_load 避免潜在的安全风险

    # ==========> 为每一次运行设置独一无二的名字
    run_name = f"{configs['exp_name']}-" + \
               f"{configs['envs']['env_id']}-" + \
               f"{configs['seed']}-" + \
               f"{datetime.today().strftime('%Y%m%d')}-" + \
               f"{int(time.time())}"
    print("当前的运行名是: ", run_name)

    # ==========> 设置随机数种子
    random.seed(configs["seed"])
    np.random.seed(configs["seed"])
    torch.manual_seed(configs["seed"])
    torch.backends.cudnn.deterministic = configs["torch_deterministic"]

    # ==========> 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() and configs["cuda"] else "cpu")
    dtype = torch.float32

    # ==========> 训练记录器: 本地用 Tensorboard, 云端用 Wandb
    if configs["wandb"]["need"]:
        config = configs
        wandb.init(
            project=configs["wandb"]["project_name"],
            config=config,
            name=run_name,
            sync_tensorboard=True  # 把 tensorboard 的记录数据传到 wandb 中
        )

    writer = SummaryWriter(f"runs/{run_name}")  # tensorboard 的初始化
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in configs.items()])),
    )

    # ==========> 初始化测试环境和数据加载器
    env, dataloader = simple_env_and_dataloader(
        env_id=configs["envs"]["env_id"],
        frame_stack=configs["frame_stack"],
        normalize_low_dim_obs=configs["normalize_low_dim_obs"],
        render_mode="human" if configs["render"] else "rgb_array",
        scale=configs["scale"],
        demo_storage_path=configs["demo_storage_path"],
        batch_size=configs["batch_size"],
        action_seq_len=configs["chunk_size"]
    )

    # ==========> 填充参数文件
    configs["d_proprioception"] = env.low_dim_observation_spec().shape[0] // configs["frame_stack"]
    configs["d_action"] = env.action_spec().shape[0]
    configs["video_path"] += configs["envs"]["env_id"]
    print(f"d_proprioception {configs['d_proprioception']}, d_action {configs['d_action']}")

    # ==========> 模型初始化
    model = ActionChunkingTransformer(
        d_model=configs["d_model"],
        d_proprioception=configs["d_proprioception"],
        d_action=configs["d_action"],
        d_z_distribution=configs["d_z_distribution"],
        d_feedforward=configs["d_feedforward"],
        n_head=configs["n_head"],
        n_representation_encoder_layers=configs["n_representation_encoder_layers"],
        n_encoder_layers=configs["n_encoder_layers"],
        n_decoder_layers=configs["n_decoder_layers"],
        n_frame_stack=configs["frame_stack"],
        chunk_size=configs["chunk_size"],
        resnet_name=configs["resnet_name"],
        return_interm_layers=configs["return_interm_layers"],
        include_depth=configs["include_depth"],
        dropout=configs["dropout"],
        activation=configs["activation"],
        normalize_before=configs["normalize_before"]
    ).to(dtype=dtype, device=device)  # 可能用 torch.bfloat16 这样的数据类型

    # ==========> 优化器设置
    param_dicts = [
        {
            # 如果是 ACT 模型的参数, 则用 "lr" 的学习率
            "params": [p for n, p in model.named_parameters() if "backbones" not in n and p.requires_grad]
        },
        {
            # 如果是视觉模型的参数, 则用 "lr" 的学习率
            "params": [p for n, p in model.named_parameters() if "backbones" in n and p.requires_grad],
            "lr": configs["lr_backbone"],
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=configs["lr"], weight_decay=configs["weight_decay"])
    # 当迭代次数达到总迭代数的三分之二时, 学习率开始下降, 这比余弦学习率好多了!
    lr_drop = int((2 / 3) * configs["total_iters"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # 开始训练进程
    for global_step, batch in tqdm(enumerate(dataloader)):

        # 训练模式
        model.train()

        # 从 batch 数据中提取: 输入图片/输入本体数据/标签动作序列
        # 把提取到的数据全部放到指定显卡中
        input_image = batch[0].to(device)  # (batch_size, num_views, channels * frame_stack, height, width)
        input_proprio = batch[1].to(device)  # (batch_size, d_proprioception * frame_stack)
        action_seq = batch[2].to(device)  # (batch_size, action_seq_len, d_action)

        current_batch_size = input_image.shape[0]

        # 将输入图片/输入本体数据送进 ACT 网络中,
        # 得到预测结果: 预测动作序列
        # 前向传播
        input_obs = {
            "state": input_proprio,
            "rgb": input_image / 255.0,
        }
        input_action = action_seq  # (batch_size, chunk_size, act_dim)
        a_hat, (mu, logvar) = model(input_obs, input_action)

        # 计算损失
        # 计算不同级别的 kl 散度
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        # 因为 F.l1_loss 是元素级绝对值计算, 因此交换数据和标签的位置不会影响结果, 损失数值形状和 a_hat 一样
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = l1 + total_kld[0] * configs["kl_weight"]
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # print(f"{l1}, {total_kld[0]}")

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        writer.add_scalar(f"train/weighted_sum_loss", total_loss, global_step)
        writer.add_scalar(f"train/kl_divergence", total_kld, global_step)

        if global_step % configs["eval_frequency"] == 0:
            model.eval()  # 进入验证模式

            # 从 batch 数据中提取: 输入图片/输入本体数据/标签动作序列
            # 把提取到的数据全部放到指定显卡中
            input_image = batch[0].to(device)  # (batch_size, num_views, channels * frame_stack, height, width)
            input_proprio = batch[1].to(device)  # (batch_size, d_proprioception * frame_stack)
            action_seq = batch[2].to(device)  # (batch_size, action_seq_len, d_action)

            current_batch_size = input_image.shape[0]

            # 将输入图片/输入本体数据送进 ACT 网络中,
            # 得到预测结果: 预测动作序列
            # 前向传播
            input_obs = {
                "state": input_proprio,
                "rgb": input_image / 255.0,
            }
            input_action = action_seq  # (batch_size, chunk_size, act_dim)
            a_hat, (mu, logvar) = model(input_obs, input_action)

            # 计算损失
            # 计算不同级别的 kl 散度
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            # 因为 F.l1_loss 是元素级绝对值计算, 因此交换数据和标签的位置不会影响结果, 损失数值形状和 a_hat 一样
            all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
            l1 = all_l1.mean()
            loss_dict = dict()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = l1 + total_kld[0] * configs["kl_weight"]
            total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

            writer.add_scalar(f"validation/weighted_sum_loss", total_loss, global_step)
            writer.add_scalar(f"validation/kl_divergence", total_kld, global_step)

        # 根据 args 规定的步数进行模型测试集评估, 第 0 步, 也就是最开始的时候, 不进行仿真器测试
        if (global_step > 0) and (global_step % configs["test_frequency"]) == 0:
            # 进入测试模式
            test_result = test_in_simulation(
                model=model,
                env=env,
                config=configs,
                device=device,
            )
            for key, value in test_result.items():
                writer.add_scalar(f"test/{key}", value, global_step)

        # 更新学习率变化
        lr_scheduler.step()

        # 保存模型和记录数据
        if configs["save_frequency"] is not None and global_step % configs["save_frequency"] == 0:
            save_ckpt(run_name, model, str(global_step))

        if global_step > configs["total_iters"]:
            print("训练结束!")
            break

    # 保存最后的模型和记录数据
    save_ckpt(run_name, model, "the-last-one")
    writer.close()
    env.close()
