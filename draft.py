from gymnasium.wrappers.frame_stack import FrameStack
from bigym.envs.move_plates import MovePlate
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig

env = MovePlate(
    action_mode=JointPositionActionMode(
        floating_base=True,  # 如果 True, 那么人形的下半身会运动
        absolute=True,  # 如果 True, 动作是绝对关节位置, 而不是增量式
        floating_dofs=[
            PelvisDof.X,
            PelvisDof.Y,
            # PelvisDof.Z,
            PelvisDof.RZ
        ]  # 这个需要根据数据集名字来确定!
    ),
    control_frequency=50,
    observation_config=ObservationConfig(
        cameras=[
            CameraConfig("head", resolution=(84, 84))
        ]
    ),
    render_mode="rgb_array",
)

# env = FrameStack(env, 4)

observation, info = env.reset()

pass
