from dataclasses import dataclass

import dm_env
import tyro
import yaml
from tqdm import tqdm

import numpy as np
import torch
from gymnasium.wrappers.record_video import RecordVideo
from datasets.bigym_datasets import simple_env_and_dataloader
from network.ACT_bigym import ActionChunkingTransformer
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.envs.dishwasher import DishwasherClose
from bigym.envs.move_plates import MovePlate


def test_in_simulation(model, env, config, device):
    model.eval()

    # TODO: 暂时不支持录制视频......
    # if config["capture_video"]:
    #     env = RecordVideo(env, config["video_path"])

    result = dict()
    reward_all_episode = []

    for _ in tqdm(range(config["num_eval_episodes"])):
        timestep = env.reset()
        reward_this_episode, step_this_episode = 0, 0
        while True:
            with torch.no_grad():
                obs_dict = {
                    "state": torch.from_numpy(
                        np.expand_dims(timestep.low_dim_obs, axis=0)
                    ).to(device),
                    "rgb": torch.from_numpy(
                        np.expand_dims(timestep.rgb_obs, axis=0)
                    ).to(device) / 255.0
                }
                a_hat, _ = model(obs_dict)

            pred_act_seq = a_hat.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)
            actually_action = pred_act_seq[0]
            actually_action = np.clip(actually_action, env._env.action_space.low, env._env.action_space.high)
            timestep = env.step(actually_action)

            reward_this_episode += timestep.reward
            step_this_episode += 1

            if timestep.step_type == dm_env.StepType.LAST:
                break

            if config["render"]:
                env.render()

            if step_this_episode >= 1999:
                break

        reward_all_episode.append(reward_this_episode)

    # env.close()

    result[config["envs"]["env_id"]] = sum(reward_all_episode) / len(reward_all_episode)

    return result


if __name__ == "__main__":
    @dataclass
    class Args:
        config_file_path: str = "/media/zjb/extend/zjb/ACT_Implementation/configs/ACT_bigym_config.yaml"


    # 因为各种原因, 可能配置文件不在 configs/ 文件夹内, 因此用终端传入轨迹
    args = tyro.cli(Args)

    # ==========> 载入参数
    with open(args.config_file_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)  # 使用 safe_load 避免潜在的安全风险

    configs["capture_video"] = False
    configs["render"] = True

    # ==========> 设置运行设备
    device = torch.device("cuda" if torch.cuda.is_available() and configs["cuda"] else "cpu")
    dtype = torch.float32

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

    # ==========> 填充参数文件
    configs["video_path"] += configs["envs"]["env_id"]
    configs["scale"] = 1
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

    test_in_simulation(model, env, configs, device)
