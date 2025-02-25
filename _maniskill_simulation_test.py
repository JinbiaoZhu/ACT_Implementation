import torch
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import mani_skill.envs  # 这个必须加, 不然不知道这个任务环境来自哪个库的
import numpy as np


def test_in_simulation(model, args):
    result = {}
    model.eval()
    env = gym.make(
        args.exp_task,  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
        num_envs=1,
        obs_mode="rgb",  # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pos",  # there is also "pd_joint_delta_pos", ...
        render_mode="human"
    )
    if args.record_video:
        env = RecordVideo(env, args.record_video_path)

    reward_all_episode = []

    for _ in tqdm(range(args.test_episode)):
        observation, info = env.reset()
        # action = env.action_space.sample()
        # observation, reward, terminate, truncation, info = env.step(action)
        reward_this_episode, step_this_episode = 0, 0
        while True:
            if args.with_goal:
                input_proprio = np.concatenate([observation["agent"]["qpos"], observation["agent"]["qvel"], observation["extra"]["goal_pos"]], 1)
            else:
                input_proprio = np.concatenate([observation["agent"]["qpos"], observation["agent"]["qvel"]], 1)
            input_image = observation["sensor_data"]["base_camera"]["rgb"].reshape((3, 128, 128))
            input_image = input_image.unsqueeze(0).unsqueeze(0).to(dtype=args.dtype, device=args.device)
            input_proprio = torch.from_numpy(input_proprio).unsqueeze(0).to(dtype=args.dtype, device=args.device)

            with torch.no_grad():
                # (1, chunk_size, action_dim)
                pred_act_seq = model(input_image, input_proprio, args, None, None, inference_mode=True)

            pred_act_seq = pred_act_seq.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
            actually_action = pred_act_seq[0]
            actually_action = np.clip(actually_action, a_min=env.action_space.low, a_max=env.action_space.high)

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
