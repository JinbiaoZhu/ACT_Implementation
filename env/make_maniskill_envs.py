from collections import defaultdict
from typing import Optional, Callable

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper


def make_eval_envs(env_id,
                   num_envs: int,
                   sim_backend: str,
                   env_kwargs: dict,
                   other_kwargs: dict,
                   video_dir: Optional[str] = None,
                   wrappers=None):
    """
    Create vectorized environment for evaluation and/or recording videos.
        For CPU vectorized environments only the first parallel environment is used to record videos.
        For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu"
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
        other_kwargs:
    """
    if wrappers is None:
        wrappers = []
    if sim_backend == "physx_cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs=None, other_kwargs=None):
            if env_kwargs is None:
                env_kwargs = dict()
            if other_kwargs is None:
                other_kwargs = dict()

            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True,
                                        source_type="act", source_desc="act evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x: gym.vector.AsyncVectorEnv(x,
                                                                                                        context="forkserver")
        env = vector_cls(
            [cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in
             range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True, source_type="act",
                                source_desc="act evaluation rollout", max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


def evaluate(n: int, sample_fn: Callable, eval_envs, device, dtype, transform):
    """
    Evaluate the agent on the evaluation environments for at least n episodes.

    Args:
        n: The minimum number of episodes to evaluate.
        sample_fn: The function to call to sample actions from the agent by passing in the observations
        eval_envs: The evaluation environments.

    Returns:
        A dictionary containing the evaluation results.
    """

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            # 对图像观测做了处理
            input_obs = {
                "state": torch.from_numpy(
                    np.concatenate(
                        (
                            obs["agent"]["qpos"],
                            obs["agent"]["qvel"],
                            obs["extra"]["is_grasped"].reshape((-1, 1)),
                            obs["extra"]["tcp_pose"],
                            obs["extra"]["goal_pos"]
                        ), axis=1
                    )
                ).to(device),
                "rgb": transform(
                    torch.from_numpy(
                        obs["sensor_data"]["base_camera"]["rgb"]
                    ).to(dtype=dtype, device=device).permute(0, 3, 1, 2)
                ).unsqueeze(1)
            }

            action, _ = sample_fn(input_obs)
            action = action[:, 0, :].data.cpu().numpy()
            obs, _, _, truncated, info = eval_envs.step(action)

            # note as there are no partial resets, truncated is True for all environments at the same time
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs

    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics
