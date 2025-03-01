import random
import PIL

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from buffer import ReplayBuffer
from tools import load_demo_dataset


def get_demo_dataset(args):
    # 从下载的数据集中载入指定数量的轨迹
    trajectories = load_demo_dataset(args.demo_path, num_traj=args.num_queries, concat=False)

    # 设置回放存储列表
    rb_list = []

    # 对轨迹集合中的每一条数据提取观测信息和动作信息
    for single_obs_traj, single_act_traj in zip(trajectories["observations"], trajectories["actions"]):

        # 定义经验回放池, 也就是存储状态转移的容器 (s, a, r, s', d) , 实际数据集从这里索引即可！
        # 如果任务是 goal-condition 的, 那么在状态空间做增广: s <- [s, g]
        rb = ReplayBuffer(
            proprioception_shape=(  # 增加了 tcp_pose 和 is_grasped 的标志, 相当于额外增加了 8 维度
                single_obs_traj["agent"]["qpos"].shape[1] + single_obs_traj["agent"]["qvel"].shape[1] +\
                single_obs_traj["extra"]["tcp_pose"].shape[1] + 1 + single_obs_traj["extra"]["goal_pos"].shape[1],
            ) if args.with_goal else (
                single_obs_traj["agent"]["qpos"].shape[1] + single_obs_traj["agent"]["qvel"].shape[1] +\
                single_obs_traj["extra"]["tcp_pose"].shape[1] + 1,
            ),
            obs_shape=(128, 128, 3),
            action_shape=(single_act_traj.shape[1],),
            capacity=300,
            device="cpu"
        )

        # 对每一条轨迹提取单步状态转移
        for t in range(1, single_obs_traj["agent"]["qpos"].shape[0]):
            is_grasped = np.array([1]) if single_obs_traj["extra"]["is_grasped"][t - 1] else np.array([0])
            if args.with_goal:
                pro = np.concatenate(
                    [
                        single_obs_traj["agent"]["qpos"][t - 1, :],
                        single_obs_traj["agent"]["qvel"][t - 1, :],
                        single_obs_traj["extra"]["tcp_pose"][t - 1, :],
                        is_grasped,
                        single_obs_traj["extra"]["goal_pos"][t - 1, :]
                    ], axis=0
                )
            else:
                pro = np.concatenate(
                    [
                        single_obs_traj["agent"]["qpos"][t - 1, :],
                        single_obs_traj["agent"]["qvel"][t - 1, :],
                        single_obs_traj["extra"]["tcp_pose"][t - 1, :],
                        is_grasped,
                    ], axis=0
                )
            obs = single_obs_traj["sensor_data"]["base_camera"]["rgb"][t - 1]
            action = single_act_traj[t - 1]
            is_grasped = np.array([1]) if single_obs_traj["extra"]["is_grasped"][t] else np.array([0])
            if args.with_goal:
                next_pro = np.concatenate(
                    [
                        single_obs_traj["agent"]["qpos"][t, :],
                        single_obs_traj["agent"]["qvel"][t, :],
                        single_obs_traj["extra"]["tcp_pose"][t - 1, :],
                        is_grasped,
                        single_obs_traj["extra"]["goal_pos"][t, :]
                    ], axis=0
                )
            else:
                next_pro = np.concatenate(
                    [
                        single_obs_traj["agent"]["qpos"][t, :],
                        single_obs_traj["agent"]["qvel"][t, :],
                        single_obs_traj["extra"]["tcp_pose"][t - 1, :],
                        is_grasped,
                    ], axis=0
                )
            next_obs = single_obs_traj["sensor_data"]["base_camera"]["rgb"][t]
            # done = demo.timesteps[t].termination or demo.timesteps[t].truncation
            rb.add(pro, obs, action, 0.0, next_pro, next_obs, 0.0)

        # 将经验回放池加入至回放存储列表中
        rb_list.append(rb)

    print("Put the demo dataset to the replay buffer")
    return rb_list


def get_dataset_index(rb_list, args):
    """
    获取用于训练、验证和测试的数据集索引
    """
    for rb in rb_list:
        # 如果存储的经验回放池动作样本数量小于 context_length 也就是 chunk_size 那么就舍弃这段演示轨迹, 太短了
        if rb.idx <= args.context_length:
            continue
        total = rb.idx  # 表示整个数据集可被索引的范围
        # 随机选择 scale 个不同的索引
        # 如果 args.scale < total, 索引数量实际是 args.scale, 反之索引数量是 total 即 rb.idx 也就是全部索引
        scale = min(args.scale, total)
        indices = random.sample(range(total), scale)  # 索引值
        # 计算每个部分的大小
        part1_size = int(scale * args.train_split)
        # 划分索引列表
        rb.train_index = indices[:part1_size]
        rb.valid_index = indices[part1_size:]

    print("The demo dataset in the replay buffer has been split.")


def get_dataset(args):
    rb_list = get_demo_dataset(args)
    get_dataset_index(rb_list, args)
    # 额外增加一个 "pad_length" 的键, 用于存储多少长度被整数 0 填充了
    train_set = {"image_datas": [], "proprioception_datas": [], "action_sequences": [], "pad_length": []}
    valid_set = {"image_datas": [], "proprioception_datas": [], "action_sequences": [], "pad_length": []}

    for rb in rb_list:
        for train_index in rb.train_index:
            train_set["image_datas"].append(rb.obses[train_index])
            train_set["proprioception_datas"].append(rb.proes[train_index])
            # 如果当前预设索引值加上 context_length 小于数据集的整体长度, 就直接放进去, 不做其他处理
            if train_index + args.context_length <= rb.idx:
                train_set["action_sequences"].append(rb.actions[train_index: train_index + args.context_length])
                train_set["pad_length"].append(0)  # "pad_length" 的键对应值的列表的对应位置是 0
            else:
                # 如果当前预设索引值加上 context_length 大于数据集的整体长度, 就要进行截断
                pad_length = train_index + args.context_length - rb.idx  # 截断的长度
                pad_action = np.zeros((pad_length, rb.actions.shape[1]))  # 设置一个 0 矩阵表示填充矩阵
                # 截断后的矩阵和填充矩阵拼接, 拼接后的矩阵具有 context_length 维度
                a = np.concatenate([rb.actions[train_index: rb.idx], pad_action], axis=0)
                train_set["action_sequences"].append(a)
                train_set["pad_length"].append(pad_length)  # 做记录
        for valid_index in rb.valid_index:
            valid_set["image_datas"].append(rb.obses[valid_index])
            valid_set["proprioception_datas"].append(rb.proes[valid_index])
            # 如果当前预设索引值加上 context_length 小于数据集的整体长度, 就直接放进去, 不做其他处理
            if valid_index + args.context_length <= rb.idx:
                valid_set["action_sequences"].append(
                    rb.actions[valid_index: valid_index + args.context_length]
                )
                valid_set["pad_length"].append(0)  # "pad_length" 的键对应值的列表的对应位置是 0
            else:
                # 如果当前预设索引值加上 context_length 大于数据集的整体长度, 就要进行截断
                pad_length = valid_index + args.context_length - rb.idx  # 截断的长度
                pad_action = np.zeros((pad_length, rb.actions.shape[1]))  # 设置一个 0 矩阵表示填充矩阵
                a = np.concatenate(
                    [rb.actions[valid_index: rb.idx], pad_action], axis=0
                )  # 截断后的矩阵和填充矩阵拼接, 拼接后的矩阵具有 context_length 维度
                valid_set["action_sequences"].append(a)
                valid_set["pad_length"].append(pad_length)  # 做记录

    return train_set, valid_set


class CustomDataset(Dataset):
    def __init__(self, data, transform):
        self.image_data = data['image_datas']
        self.proprioception_data = data['proprioception_datas']
        self.action_seq = data['action_sequences']
        self.pad_length = data["pad_length"]

        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # 从数据集中获取一组数据: 图像、本体、动作序列和填充长度
        input_image = self.image_data[idx]
        input_propr = self.proprioception_data[idx]
        pred_act_seq = self.action_seq[idx]
        pad_length = self.pad_length[idx]

        # 可以在此处进行额外的数据增强或预处理
        input_image = self.transform(input_image)

        return input_image, input_propr, pred_act_seq, pad_length


training_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    class Args:
        demo_path = "/home/zjb/.maniskill/demos/StackCube-v1/motionplanning/trajectory.h5"
        num_queries = 2
        scale = 150  # 每条轨迹中采样的数据样本条数
        train_split = 0.8  # 训/验比是 9:1 且直接在仿真环境中部署做测试
        valid_split = 0.2
        context_length = 10
        with_goal = False


    args = Args()
    training_demos, test_demos = get_dataset(args)
    dataset = CustomDataset(data=training_demos, transform=training_transform)
    input_image, input_propr, pred_act_seq, pad_length = dataset.__getitem__(2)

    print(input_image.dtype, input_image.shape)
