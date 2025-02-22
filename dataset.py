import random
import PIL

import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from bigym.action_modes import JointPositionActionMode
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.reach_target import ReachTargetSingle, ReachTargetDual
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

from buffer import ReplayBuffer


def get_slice_demo_dataset():
    control_frequency = 50
    demos = []
    # 这里暂定这 4 个任务, 因为只有这 4 个任务是有 pixel-observation 的, 作者有提到未来会完善数据集
    # for cls in [MovePlate, MoveTwoPlates, ReachTargetSingle, ReachTargetDual]:
    for cls in [MoveTwoPlates]:
        env = cls(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=control_frequency,
            observation_config=ObservationConfig(cameras=[CameraConfig("head", resolution=(84, 84))]),
            render_mode="human",
        )
        metadata = Metadata.from_env(env)

        # Get demonstrations from DemoStore
        demo_store = DemoStore()
        demos += demo_store.get_demos(metadata, amount=-1, frequency=control_frequency)

    # 设置回放存储列表
    rb_list = []

    for demo in demos:
        # 定义经验回放池, 也就是存储状态转移的容器 (s, a, r, s', r, d) , 实际数据集从这里索引即可！
        rb = ReplayBuffer(
            proprioception_shape=(
                sum([key[0] for key in [demo.timesteps[0].observation['proprioception'].shape,
                                        demo.timesteps[0].observation['proprioception_floating_base'].shape,
                                        demo.timesteps[0].observation['proprioception_floating_base_actions'].shape,
                                        demo.timesteps[0].observation['proprioception_grippers'].shape]]),
            ),
            obs_shape=(3, 84, 84),
            action_shape=env.action_space.shape,
            capacity=300,
            device="cpu"
        )
        for t in range(1, len(demo.timesteps)):
            pro = np.concatenate(
                [demo.timesteps[t - 1].observation['proprioception'],
                 demo.timesteps[t - 1].observation['proprioception_floating_base'],
                 demo.timesteps[t - 1].observation['proprioception_floating_base_actions'],
                 demo.timesteps[t - 1].observation['proprioception_grippers']]
            )
            obs = demo.timesteps[t - 1].visual_observations["rgb_head"]
            action = demo.timesteps[t - 1].info["demo_action"]
            reward = demo.timesteps[t - 1].reward
            next_pro = np.concatenate(
                [demo.timesteps[t].observation['proprioception'],
                 demo.timesteps[t].observation['proprioception_floating_base'],
                 demo.timesteps[t].observation['proprioception_floating_base_actions'],
                 demo.timesteps[t].observation['proprioception_grippers']]
            )
            next_obs = demo.timesteps[t].visual_observations["rgb_head"]
            done = demo.timesteps[t].termination or demo.timesteps[t].truncation
            rb.add(pro, obs, action, reward, next_pro, next_obs, done)

        # 将经验回放池加入至回放存储列表中
        rb_list.append(rb)

    print("Put the demo dataset to the replay buffer")
    env.close()
    return rb_list


def get_dataset_index(rb_list, args):
    """
    获取用于训练、验证和测试的数据集索引
    """
    for rb in rb_list:
        if rb.idx <= args.context_length:
            continue
        total = rb.idx - args.context_length  # 表示整个数据集可被索引的范围
        scale = args.scale if total > args.scale else total
        indices = random.sample(range(total), scale)  # 随机选择 args.scale 个不同的索引
        # 计算每个部分的大小
        part1_size = int(scale * args.train_split)
        # 划分索引列表
        rb.train_index = indices[:part1_size]
        rb.valid_index = indices[part1_size:]

    print("The demo dataset in the replay buffer has been split.")


def get_dataset(args):
    rb_list = get_slice_demo_dataset()
    get_dataset_index(rb_list, args)
    train_set = {"image_datas": [], "proprioception_datas": [], "action_sequences": []}
    valid_set = {"image_datas": [], "proprioception_datas": [], "action_sequences": []}
    for rb in rb_list:
        for train_index in rb.train_index:
            train_set["image_datas"].append(rb.obses[train_index])
            train_set["proprioception_datas"].append(rb.proes[train_index])
            train_set["action_sequences"].append(rb.actions[train_index: train_index + args.context_length])
        for valid_index in rb.valid_index:
            valid_set["image_datas"].append(rb.obses[valid_index])
            valid_set["proprioception_datas"].append(rb.proes[valid_index])
            valid_set["action_sequences"].append(rb.actions[valid_index: valid_index + args.context_length])

    return train_set, valid_set


class CustomDataset(Dataset):
    def __init__(self, data, transform):
        self.image_data = data['image_datas']
        self.proprioception_data = data['proprioception_datas']
        self.action_seq = data['action_sequences']

        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # 从数据集中获取一张图片
        input_image = PIL.Image.fromarray(
            self.image_data[idx].transpose(1, 2, 0).astype(np.uint8)
        )  # 转换为 HWC 格式，并转为 uint8 类型
        input_propr = self.proprioception_data[idx]
        pred_act_seq = self.action_seq[idx]

        # 可以在此处进行额外的数据增强或预处理（如归一化、标准化等）
        input_image = self.transform(input_image)

        return input_image, input_propr, pred_act_seq


transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
