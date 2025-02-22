from tqdm import tqdm

import numpy as np
import torch
from gymnasium.wrappers.record_video import RecordVideo
from bigym.action_modes import JointPositionActionMode
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.reach_target import ReachTargetSingle, ReachTargetDual


def test_in_simulation(model, args):
    control_frequency = 50  # 这个不是超参数一般不改变
    result = {}
    model.eval()
    for cls, name in zip(
            # [MovePlate, MoveTwoPlates, ReachTargetSingle, ReachTargetDual],
            # ["move_plate", "move_two_plates", "reach_target_single", "reach_target_dual"]
            [ReachTargetDual],
            ["ReachTargetDual"]
    ):
        env = cls(
            action_mode=JointPositionActionMode(floating_base=True, absolute=True),
            control_frequency=control_frequency,
            observation_config=ObservationConfig(cameras=[CameraConfig("head", resolution=(84, 84))]),
            render_mode="human",
        )
        if args.record_video:
            env = RecordVideo(env, args.record_video_path)

        reward_all_episode = []

        weight = np.exp(-1 * args.inference_coefficient * np.arange(args.chunk_size).reshape(-1, 1)).tolist()

        for _ in tqdm(range(args.test_episode)):
            observation, info = env.reset()
            action = env.action_space.sample()
            observation, reward, terminate, truncation, info = env.step(action)
            reward_this_episode, step_this_episode = 0, 0
            while True:
                input_image = observation["rgb_head"]  # np.ndarray (3, 84, 84)
                input_proprio = np.concatenate(
                    [
                        observation['proprioception'],
                        observation['proprioception_floating_base'],
                        observation['proprioception_floating_base_actions'],
                        observation['proprioception_grippers']
                    ]  # np.ndarray (66,)
                )
                input_image = torch.from_numpy(input_image).unsqueeze(0).unsqueeze(0).to(
                    dtype=torch.float32, device=args.device
                )
                input_proprio = torch.from_numpy(input_proprio).unsqueeze(0).unsqueeze(0).to(
                    dtype=torch.float32, device=args.device
                )

                with torch.no_grad():
                    pred_act_seq = model(input_image, input_proprio, args, None, None,
                                         inference_mode=True)  # (1, chunk_size, action_dim)
                    pred_act_seq = pred_act_seq.squeeze(0).cpu().numpy()  # (chunk_size, action_dim)

                    actually_action = pred_act_seq[0]
                    actually_action = np.clip(actually_action, a_min=env.action_space.low, a_max=env.action_space.high)

                    # for index in range(1, args.chunk_size + 1):
                    #     actually_action = np.sum(pred_act_seq[0: index] * np.array(weight[0: index]), axis=0)
                    #     actually_action = np.clip(actually_action, a_min=env.action_space.low, a_max=env.action_space.high)

                    observation, reward, terminate, truncation, info = env.step(actually_action)

                    reward_this_episode += reward
                    step_this_episode += 1

                    if terminate or truncation:
                        break

                    if args.render:
                        env.render()

                if step_this_episode >= 999:
                    break

            reward_all_episode.append(reward_this_episode)

        env.close()

        result[name] = sum(reward_all_episode) / len(reward_all_episode)

    return result


if __name__ == "__main__":
    from configs.ACT_config import Arguments
    from network.ACT import ActionChunkTransformer

    act_config = Arguments()
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
    )
    model.load_state_dict(torch.load("./ckpts/ACT_202502221900_ReachTargetDual_HeadRGB.pt"))
    model = model.to(act_config.device)
    test_in_simulation(model, act_config)
