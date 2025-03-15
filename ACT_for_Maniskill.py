from collections import defaultdict
from datetime import datetime
import time
from dataclasses import dataclass
import random
from tqdm import tqdm

import torchvision.transforms
import tyro
import yaml
import numpy as np
import torch
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from env.make_maniskill_envs import make_eval_envs, evaluate
from datasets.maniskill_datasets import ManiskillDataset
from datasets.maniskill_datasets_tools import IterationBasedBatchSampler, worker_init_fn
from network.ACT import ActionChunkingTransformer
from utils import kl_divergence, save_ckpt


@dataclass
class Args:
    config_file_path: str = "configs/ACT_maniskill_config.yaml"


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

    # ==========> 初始化仿真器测试环境
    env_kwargs = dict(
        control_mode=configs["dataset"]["control_mode"],  # 注意: Mani-Skill 的控制模式要和数据集一致!!!
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="rgb_array"  # 这里不设可调参数, 不是高频调参项, 必要时直接在这里改动即可
    )
    # 自定义每条 episode 的最大步数, 这是因为模仿学习到的技能, 完成任务所需的步数会超过环境最大步数
    if configs["envs"]["max_episode_steps"] is not None:
        env_kwargs["max_episode_steps"] = configs["envs"]["max_episode_steps"]
    other_kwargs = dict()
    envs = make_eval_envs(
        env_id=configs["envs"]["env_id"],
        num_envs=configs["num_eval_envs"],
        sim_backend=configs["sim_backend"],
        env_kwargs=env_kwargs,
        other_kwargs=other_kwargs,
        video_dir=f'runs/{run_name}/videos' if configs["capture_video"] else None
    )

    # ==========> 在配置文件中填充本体数据维度和动作空间维度
    configs["d_proprioception"] = envs.single_observation_space.spaces["agent"]["qpos"].shape[0] + \
                                  envs.single_observation_space.spaces["agent"]["qvel"].shape[0] + 1 + \
                                  envs.single_observation_space.spaces["extra"]["tcp_pose"].shape[0] + \
                                  envs.single_observation_space.spaces["extra"]["goal_pos"].shape[0]
    configs["d_action"] = envs.single_action_space.shape[0]
    print(f"d_proprioception {configs['d_proprioception']}, d_action {configs['d_action']}")

    # ==========> 载入数据集: 数据集预处理函数、数据集类、采样器和生成器
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    dataset = ManiskillDataset(
        chunk_size=configs["chunk_size"],  # ACT 模型中预测的动作时序长度
        num_traj=configs["dataset"]["num_traj"],  # 从数据集中载入多少演示轨迹
        data_path=configs["dataset"]["path"],  # 数据集的文件路径
        device=device,
        dtype=dtype,
        transform=transform,
        args=configs["dataset"]  # 其他参数
    )
    sampler = RandomSampler(dataset, replacement=False)
    # drop_last=True 表示如果数据集划分少于 batch_size, 则丢掉剩余的样本
    batch_sampler = BatchSampler(sampler, batch_size=configs["batch_size"], drop_last=True)
    # 包装这个类, 可以确保数据集的采样过程依赖于 iteration, 而不依赖于数据集的大小...? 我查阅资料是这样看的...
    batch_sampler = IterationBasedBatchSampler(batch_sampler, configs["total_iters"])
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=configs["dataset"]["num_dataload_workers"],
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=configs["seed"]),
    )
    # configs["num_demos"] = dataset.num_traj

    # ==========> 训练记录器: 本地用 Tensorboard, 云端用 Wandb
    if configs["wandb"]["need"]:
        import wandb

        config = configs
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=configs["num_eval_envs"],
            env_id=configs["envs"]["env_id"],
            env_horizon=configs["envs"]["max_episode_steps"]
        )
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

    # ==========> 模型初始化
    ACT = ActionChunkingTransformer(
        d_model=configs["d_model"],
        d_proprioception=configs["d_proprioception"],
        d_action=configs["d_action"],
        d_z_distribution=configs["d_z_distribution"],
        d_feedforward=configs["d_feedforward"],
        n_head=configs["n_head"],
        n_representation_encoder_layers=configs["n_representation_encoder_layers"],
        n_encoder_layers=configs["n_encoder_layers"],
        n_decoder_layers=configs["n_decoder_layers"],
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
            "params": [p for n, p in ACT.named_parameters() if "backbones" not in n and p.requires_grad]
        },
        {
            # 如果是视觉模型的参数, 则用 "lr" 的学习率
            "params": [p for n, p in ACT.named_parameters() if "backbones" in n and p.requires_grad],
            "lr": configs["lr_backbone"],
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=configs["lr"], weight_decay=configs["weight_decay"])
    # 当迭代次数达到总迭代数的三分之二时, 学习率开始下降, 这比余弦学习率好多了!
    lr_drop = int((2 / 3) * configs["total_iters"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, lr_drop)

    # ---------------------------------------------------------------------------- #
    # 训练
    # ---------------------------------------------------------------------------- #
    print("训练开始!")
    ACT.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for cur_iter, data_batch in enumerate(tqdm(train_dataloader, desc="Training Progress")):
        # data_batch 是一个字典, 包括:
        # 本体数据 "state" (batch_size, d_proprioception)
        # 动作数据 "actions" (batch_size, chunk_size, d_action)
        # 图片数据 "rgb" (batch_size, channel, height, width) ==> 因为是单视角, 所以需要额外增加一个维度

        last_tick = time.time()

        # 前向传播
        input_obs = {
            "state": data_batch["state"],
            "rgb": data_batch["rgb"].unsqueeze(1),  # 因为这里是单视角, 所以增加一个视角维度, 待修改
        }
        input_action = data_batch['actions']  # (batch_size, chunk_size, act_dim)
        a_hat, (mu, logvar) = ACT(input_obs, input_action)

        # 计算损失
        # 计算不同级别的 kl 散度
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        # 因为 F.l1_loss 是元素级绝对值计算, 因此交换数据和标签的位置不会影响结果, 损失数值形状和 a_hat 一样
        all_l1 = F.l1_loss(data_batch['actions'], a_hat, reduction='none')
        l1 = all_l1.mean()
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * configs["kl_weight"]
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        timings["update"] += time.time() - last_tick

        # ---------------------------------------------------------------------------- #
        # 模型评估
        # ---------------------------------------------------------------------------- #
        # 每隔 configs["eval_frequency"] 次迭代进行一次评估
        if cur_iter % configs["eval_frequency"] == 0:
            last_tick = time.time()  # 记录评估开始时间

            eval_metrics = evaluate(
                n=configs["num_eval_episodes"],  # 评估时运行的测试 episode 数量
                sample_fn=ACT,  # 训练中的模型
                eval_envs=envs,  # 评估环境
                device=device,  # 设备 (CPU 或 GPU)
                dtype=dtype,  # 数据类型 (如 float32 或 float16)
                transform=transform  # 用于预处理观测数据的变换函数
            )
            # 计算评估所消耗的时间并记录
            timings["eval"] += time.time() - last_tick
            # 打印评估时执行的 episode 数量
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")

            # 遍历评估指标, 将每个指标取均值, 并写入日志
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], cur_iter)
                print(f"{k}: {eval_metrics[k]:.4f}")

            # 需要根据评估结果保存最优模型的指标 (通常是成功率)
            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                # 如果当前评估指标优于之前的最优值，则更新最优值并保存模型
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, ACT, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')

        if cur_iter % configs["log_frequency"] == 0:
            # 对每次 iteration 做记录
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)

        # 保存模型权重
        if configs["save_frequency"] is not None and cur_iter % configs["save_frequency"] == 0:
            save_ckpt(run_name, ACT, str(cur_iter))

    envs.close()
    writer.close()
