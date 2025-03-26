from dataclasses import dataclass

import torch
import tqdm
from pathlib import Path

import tyro
import yaml
from dm_env import specs, StepType
import numpy as np

from datasets.bigym_src.bigym_env import make
from datasets.bigym_src.replay_buffer import ReplayBufferStorage
from datasets.bigym_src.replay_buffer_action_sequence import make_replay_loader


def simple_env_and_dataloader(
        env_id,
        frame_stack,
        normalize_low_dim_obs,  # 是否在训练之前把本体状态 (低维度状态空间) 进行归一化, 这样有助于训练稳定
        render_mode,  # 只有 render_mode="human" 才可使用 env.render() 可以渲染, 其余都不能
        scale,  # scale 设置 -1 表示把所有演示轨迹数据集载入进来
        demo_storage_path,
        batch_size,
        action_seq_len,
):
    """
    原始脚本可以设置很多参数, 但是一部分参数不经常设置, 这些不常设置的参数在每个任务中都是一样的.
    因此编写一个简单的返回测试环境和数据载入器的函数, 这样可以简单一些.
    :return: 环境的示例 (用于测试) 以及数据集载入器
    """
    # ################################################################
    # 初始化环境, 这个环境可以用于训练模型的实际测试
    # ################################################################
    env = make(
        task_name=env_id,
        enable_all_floating_dof=True,  # 如果是 True, 那么可以进行空间 xyz 三个方向以及自旋运动
        action_mode="absolute",  # 默认是绝对关节位置
        demo_down_sample_rate=25,  # 默认 25
        episode_length=3000,  # 以上几个参数不经常改动
        frame_stack=frame_stack,
        camera_shape=(84, 84),  # 在仿真器 (3, 84, 84) 像素点足够了
        camera_keys=(
            "head",
            "right_wrist",
            "left_wrist"
        ),  # Bigym 的人形机器人具有三个视角: 头部, 左腕部和右腕部视角
        state_keys=(
            "proprioception",  # 本体数据中的关节角位置
            "proprioception_grippers",  # 本体数据中的两个夹爪开合状态
            "proprioception_floating_base",  # 本体数据中移动本体 (下肢) 的运动状态
        ),
        render_mode=render_mode,
        normalize_low_dim_obs=normalize_low_dim_obs
    )
    # ################################################################
    # 载入预先处理好的演示轨迹数据集
    # ################################################################
    demos = env.get_demos(scale)
    # ################################################################
    # 初始化数据集的存放空间 (状态空间, 动作空间, 其他信息等)
    # ################################################################
    data_specs = (
        env.rgb_raw_observation_spec(),
        env.low_dim_raw_observation_spec(),
        env.action_spec(),
        specs.Array((1,), np.float32, "reward"),
        specs.Array((1,), np.float32, "discount"),
        specs.Array((1,), np.float32, "demo"),
    )
    # ################################################################
    # 初始化数据集存储: 在一个循环中依次设置经验回放池, 在每个池子中对每条演示轨迹
    #       依次读取时间步 (timestep) 存储于经验回放池中.
    # ################################################################
    for demo in tqdm.tqdm(demos):
        demo_replay_storage = ReplayBufferStorage(
            data_specs,
            Path(demo_storage_path) / env_id,  # 存储文件夹的名字就是任务名字
            use_relabeling=True,
            is_demo_buffer=True,
        )
        for timestep in demo:
            demo_replay_storage.add(timestep)
    # ################################################################
    # 初始化数据加载器:
    # ################################################################
    loader = make_replay_loader(
        replay_dir=Path(demo_storage_path) / env_id,
        max_size=100000,
        batch_size=batch_size,
        num_workers=4,
        save_snapshot=False,
        nstep=1,
        discount=0.99,
        action_sequence=action_seq_len,
        frame_stack=frame_stack,
        fill_action="last_action"  # 因为上面 make 函数代码默认 env 是绝对的, 因此这里用 "last_action"
    )
    return env, loader


if __name__ == "__main__":

    @dataclass
    class Args:
        config_file_path: str = "/media/zjb/extend/zjb/ACT_Implementation/configs/ACT_bigym_config.yaml"


    # 因为各种原因, 可能配置文件不在 configs/ 文件夹内, 因此用终端传入轨迹
    args = tyro.cli(Args)

    # ==========> 载入参数
    with open(args.config_file_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)  # 使用 safe_load 避免潜在的安全风险

    configs["video_path"] += configs["envs"]["env_id"]  # deprecated
    configs["scale"] = -1

    env, loader = simple_env_and_dataloader(
        env_id=configs["envs"]["env_id"],
        frame_stack=configs["frame_stack"],
        normalize_low_dim_obs=configs["normalize_low_dim_obs"],
        render_mode="human" if configs["render"] else "rgb_array",
        scale=configs["scale"],
        demo_storage_path=configs["demo_storage_path"],
        batch_size=configs["batch_size"],
        action_seq_len=configs["chunk_size"]
    )

    timestep = env.reset()
    print(timestep.low_dim_obs.shape, timestep.rgb_obs.shape)
    obs_dict = {
        "state": torch.from_numpy(np.expand_dims(timestep.low_dim_obs, axis=0)),
        "rgb": torch.from_numpy(np.expand_dims(timestep.rgb_obs, axis=0) / 255.0)
    }
    action = np.random.random(env.action_spec().shape)
    timestep = env.step(action)
    print(f"reward: {timestep.reward}")
    if timestep.step_type == StepType.LAST:
        pass

    print("low dim observation space: ", env.low_dim_observation_spec().shape)
    print("rgb observation space: ", env.rgb_observation_spec().shape)
    print("action space: ", env.action_spec().shape)

    for e_num, batch in tqdm.tqdm(enumerate(loader)):

        print(f"number: {e_num}")
        for data, name in zip(batch[0:3], ["rgb", "state", "action_seq"]):
            print(f"name:{name}\t\tshape:{data.shape}")

        pass

        # 结果是这样的:
        # number: 0
        # name:rgb		shape:torch.Size([batch_size, num_views, channels * frame_stack, height, width])
        # name:state		shape:torch.Size([batch_size, d_proprioception * frame_stack])
        # name:action_seq		shape:torch.Size([batch_size, action_seq_len, d_action])
