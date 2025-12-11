import os
os.environ["MUJOCO_GL"] = "glfw"  # optional if already exported in shell

from dm_control import suite

env = suite.load('cartpole', 'swingup')
pixels = env.physics.render()
print(pixels.shape)