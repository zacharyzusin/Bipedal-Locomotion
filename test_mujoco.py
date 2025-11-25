import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("assets/walker2d/walker2d.xml")
data = mujoco.MjData(model)

# Try a few simulation steps
for _ in range(1000):
    mujoco.mj_step(model, data)

# Try offscreen rendering
renderer = mujoco.Renderer(model, width=320, height=240)
renderer.update_scene(data)
img = renderer.render()   # float32 RGB

print("OK: MuJoCo simulation + offscreen rendering works!")
print("Image shape:", img.shape)