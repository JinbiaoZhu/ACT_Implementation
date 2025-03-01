import torch
import torchvision.transforms
from tqdm import tqdm

import gymnasium as gym
import mani_skill.envs  # 这个必须加, 不然不知道这个任务环境来自哪个库的
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode

from _maniskill_dataset import validation_transform


def test_in_simulation(model, args):
    result = {}
    model.eval()
    env = gym.make(
        args.exp_task,
        num_envs=1,
        obs_mode="rgb",  # 用 rgb 观测模式以便获得图像
        control_mode="pd_ee_delta_pos",  # 控制模式应该和数据集一致 (查看 json 文件)
        render_mode="rgb_array"
    )
    # 因为数据集里面的演示数据都是 50-100 步之间, 但是默认初始环境是 50 步, 所以这边调大一些
    # env.max_episode_steps = 100
    if args.record_video:
        env = RecordEpisode(env, # output_dir="videos", save_trajectory=True, trajectory_name="trajectory", save_video=True, video_fps=30
                            output_dir=args.record_video_path,
                            save_trajectory=True,
                            trajectory_name=f"{args.exp_task}-simulation-test",
                            save_video=True,
                            video_fps=30)

    reward_all_episode = []

    for _ in tqdm(range(args.test_episode)):
        observation, info = env.reset()

        actually_action = env.action_space.sample()
        observation, _, _, _, info = env.step(actually_action)


        reward_this_episode, step_this_episode = 0, 0
        while True:
            if args.with_goal:
                input_proprio = np.concatenate(
                    [
                        observation["agent"]["qpos"],
                        observation["agent"]["qvel"],
                        observation["extra"]["tcp_pose"],
                        np.array([[1]]) if observation["extra"]["is_grasped"] else np.array([[0]]),
                        observation["extra"]["goal_pos"]
                    ], 1)
            else:
                input_proprio = np.concatenate(
                    [
                        observation["agent"]["qpos"],
                        observation["agent"]["qvel"],
                        observation["extra"]["tcp_pose"],
                        np.array([1]) if observation["extra"]["is_grasped"] else np.array([0]),
                ], 1)

            input_image = observation["sensor_data"]["base_camera"]["rgb"].numpy().reshape((128, 128, 3))

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(input_image)
            # plt.show()

            input_image = validation_transform(input_image)
            input_image = input_image.unsqueeze(0).unsqueeze(0).to(device=args.device)

            input_proprio = torch.from_numpy(input_proprio).unsqueeze(0).to(dtype=args.dtype, device=args.device)

            with torch.no_grad():
                # (1, chunk_size, action_dim)
                pred_act_seq = model(input_image, input_proprio, args, None, None, inference_mode=True)

            pred_act_seq = pred_act_seq.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
            actually_action = pred_act_seq[0]
            # print(actually_action)
            # actually_action = np.clip(actually_action, a_min=env.action_space.low, a_max=env.action_space.high)

            observation, reward, terminate, truncation, info = env.step(actually_action)

            reward_this_episode += reward
            step_this_episode += 1

            if terminate or truncation:
                break

            if args.render:
                env.render()

        reward_all_episode.append(reward_this_episode)

    env.close()

    result[args.exp_task] = sum(reward_all_episode) / len(reward_all_episode)

    return result


if __name__ == "__main__":
    from configs.ACT_maniskill_config import Arguments
    from network.ACT import ActionChunkTransformer

    act_config = Arguments()
    model = ActionChunkTransformer(
        d_model=act_config.d_model,
        d_proprioception=act_config.d_proprioception if not act_config.with_goal else act_config.d_proprioception + act_config.d_goal_pos,
        d_action=act_config.d_action,
        d_z_distribution=act_config.d_z_distribution,
        num_heads=act_config.num_heads,
        num_encoder_layers=act_config.num_encoder_layers,
        num_decoder_layers=act_config.num_decoder_layers,
        dropout=act_config.dropout,
        dtype=act_config.dtype,
        device=act_config.device
    ).to(act_config.device)
    model.load_state_dict(
        torch.load("./ckpts/ACT_202502282330_PickCube-v1_RGB.pt")
    )
    test_in_simulation(model, act_config)
