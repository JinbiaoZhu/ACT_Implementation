import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from datasets.maniskill_datasets_tools import load_demo_dataset


class ManiskillDataset(Dataset):  # Load everything into GPU memory
    """
        我对数据集做了处理, 原本 ManiSkill 不支持图像观测, 现在可以支持了
    """

    def __init__(self, chunk_size, num_traj, data_path, device, dtype, transform, args):
        """

        :param chunk_size: ACT 算法中 预测的动作块 的块长度
        :param num_traj: 从数据集中载入多少轨迹
        :param data_path: 数据集地址, 建议用全文件路径
        :param device: 数据集存储的设备
        :param args: 其他参数, 从主程序导入
        """
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)

        trajectories["state"] = []
        trajectories["rgb"] = []
        for i in range(len(trajectories["observations"])):  # 设置的读取数据集的条数
            trajectories["state"].append(
                torch.from_numpy(
                    np.concatenate(
                        (trajectories["observations"][i]["agent"]["qpos"],
                         trajectories["observations"][i]["agent"]["qvel"],
                         trajectories["observations"][i]["extra"]["is_grasped"].reshape((-1, 1)),
                         trajectories["observations"][i]["extra"]["tcp_pose"],
                         trajectories["observations"][i]["extra"]["goal_pos"]), axis=1
                    )
                ).to(device)
            )
            trajectories["rgb"].append(
                transform(
                    torch.from_numpy(
                        trajectories["observations"][i]["sensor_data"]["base_camera"]["rgb"]
                    ).to(dtype=dtype, device=device).permute(0, 3, 1, 2)
                )
            )
        for i in range(len(trajectories["actions"])):  # 设置的读取数据集的条数
            trajectories["actions"][i] = torch.from_numpy(trajectories["actions"][i]).to(device)

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if "delta_pos" in args["control_mode"] or args["control_mode"] == "base_pd_joint_vel_arm_pd_joint_vel":
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1] - 1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.slices = []
        self.num_traj = len(trajectories['actions'])
        for traj_idx in range(self.num_traj):
            episode_len = trajectories['actions'][traj_idx].shape[0]  # 每条轨迹的长度
            self.slices += [
                (traj_idx, ts) for ts in range(episode_len)
            ]

        print(f"数据集的长度是: {len(self.slices)}")

        self.chunk_size = chunk_size
        self.trajectories = trajectories
        self.delta_control = 'delta' in args["control_mode"]

        self.args = args

    def __getitem__(self, index):
        traj_idx, ts = self.slices[index]

        # get observation at ts only
        state = self.trajectories['state'][traj_idx][ts]
        rgb = self.trajectories['rgb'][traj_idx][ts]
        # get num_queries actions
        act_seq = self.trajectories['actions'][traj_idx][ts:ts + self.chunk_size]
        action_len = act_seq.shape[0]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.chunk_size:
            if 'delta_pos' in self.args["control_mode"] or \
                    self.args["control_mode"] == 'base_pd_joint_vel_arm_pd_joint_vel':
                gripper_action = act_seq[-1, -1]
                pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
                act_seq = torch.cat([act_seq, pad_action.repeat(self.chunk_size - action_len, 1)], dim=0)
                # making the robot (arm and gripper) stay still
            elif not self.delta_control:
                target = act_seq[-1]
                act_seq = torch.cat([act_seq, target.repeat(self.chunk_size - action_len, 1)], dim=0)

        return {
            'state': state,
            'rgb': rgb,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)
