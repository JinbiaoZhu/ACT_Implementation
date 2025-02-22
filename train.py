from tqdm import tqdm

from configs.ACT_config import Arguments
from network.ACT import ActionChunkTransformer
from dataset import get_dataset, CustomDataset, transform
from tools import repeater, get_config_dict
from logger import WandbLogger
from simulation_test import test_in_simulation

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

if __name__ == "__main__":

    # 参数
    act_config = Arguments()

    # 模型
    model = ActionChunkTransformer(
        d_model=act_config.d_model,
        d_proprioception=act_config.d_proprioception,
        d_action=act_config.d_action,
        d_z_distribution=act_config.d_z_distribution,
        num_heads=act_config.num_heads,
        num_encoder_layers=act_config.num_encoder_layers,
        num_decoder_layers=act_config.num_decoder_layers,
        dropout=act_config.dropout,
        dtype=act_config.dtype,
        device=act_config.device
    ).to(act_config.device)

    # 数据集
    training_set, valid_set = get_dataset(act_config)
    training_dataset = CustomDataset(training_set, transform)
    training_dataloader = DataLoader(training_dataset, batch_size=act_config.batch_size, shuffle=True)
    validation_dataset = CustomDataset(valid_set, transform)
    validation_dataloader = DataLoader(validation_dataset, batch_size=act_config.batch_size, shuffle=True)
    # 优化训练的 DataLoader 便于更好的训练
    training_dataloader = repeater(training_dataloader)

    # 创建优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=act_config.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=act_config.num_step, eta_min=act_config.lr_min
    )

    # 创建记录器
    wandb_logger = WandbLogger(project_name=act_config.exp_name, config=get_config_dict(act_config))

    # 开始训练进程
    for global_step in tqdm(range(1, act_config.num_step + 1)):

        # 训练模式
        model.train()

        # 从 batch 数据中提取: 输入图片/输入本体数据/标签动作序列
        # 输入图片/输入本体数据扩充一维度, 与动作序列对齐, 表示是一个 "token"
        # 把提取到的数据全部放到指定显卡中
        batch = next(training_dataloader)
        input_image, input_proprio, action_seq = batch
        input_image = input_image.unsqueeze(1).to(act_config.device)
        input_proprio = input_proprio.unsqueeze(1).to(act_config.device)
        action_seq = action_seq.to(act_config.device)

        # 将输入图片/输入本体数据送进 ACT 网络中,
        # 得到预测结果: 预测动作序列
        pred_act_seq = model(input_image, input_proprio, act_config, action_seq, None, inference_mode=False)

        # 计算动作序列损失, 然后加权计算总损失并求梯度
        action_seq_loss = F.l1_loss(pred_act_seq, action_seq)
        optimizer.zero_grad()
        action_seq_loss.backward()
        optimizer.step()

        wandb_logger.log(
            {
                "train/action_seq_loss": action_seq_loss.item(),
            }, global_step
        )

        # 根据 args 规定的步数进行模型验证集评估
        if global_step % act_config.every_valid_step == 0:
            model.eval()  # 进入验证模式
            eval_total_loss = 0
            for batch in validation_dataloader:
                # 从 batch 数据中提取: 输入图片/输入本体数据/标签动作序列
                # 输入图片/输入本体数据扩充一维度, 与动作序列对齐, 表示是一个 "token"
                # 把提取到的数据全部放到指定显卡中
                input_image, input_proprio, action_seq = batch
                input_image = input_image.unsqueeze(1).to(act_config.device)
                input_proprio = input_proprio.unsqueeze(1).to(act_config.device)
                action_seq = action_seq.to(act_config.device)

                pred_act_seq = model(input_image, input_proprio, act_config, action_seq, None, inference_mode=False)

                action_seq_loss = F.l1_loss(pred_act_seq, action_seq)

                wandb_logger.log(
                    {
                        "validation/action_seq_loss": action_seq_loss.item(),
                    }, global_step
                )

        # 根据 args 规定的步数进行模型测试集评估
        if global_step % act_config.every_test_step == 0:
            model.eval()  # 进入测试模式
            test_result = test_in_simulation(model, act_config)
            wandb_logger.log(test_result, global_step)

        # 更新学习率变化
        lr_scheduler.step()

    # 保存模型和记录数据
    wandb_logger.save_model(model,
                            model_name="./ckpts/ACT_" + act_config.exp_day + "_" + act_config.exp_task + "_" + \
                                       act_config.exp_obs_type + ".pt")
    wandb_logger.finish()
