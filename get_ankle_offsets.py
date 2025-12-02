import numpy as np
from core.mujoco_env import MujocoEnv, MujocoEnvConfig

xml_path="assets/biped/biped.xml"

def make_env() -> MujocoEnv:
    cfg = MujocoEnvConfig(
        xml_path=xml_path,
        episode_length=5_000,
        frame_skip=5,
        reset_noise_scale=0.01,
        render=True,
        width=640,
        height=480,
    )
    return MujocoEnv(cfg)

env = make_env()
env.reset()

# biped
hip_L_idx = 7
knee_L_idx = 8
ankle_L_idx = 9

print("hip_L initial:", env._init_qpos[hip_L_idx]/np.pi, "pi")
print("knee_L initial:", env._init_qpos[knee_L_idx]/np.pi, "pi")
print("ankle_L initial:", env._init_qpos[ankle_L_idx]/np.pi, "pi")