# inspect_joints.py
import mujoco
import numpy as np

from core.mujoco_env import MujocoEnv, MujocoEnvConfig


def safe_name(model, obj_type, obj_id):
    """Return the name of a MuJoCo object or a placeholder if None."""
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"<unnamed_{obj_id}>"


def inspect_joints(xml_path: str):
    cfg = MujocoEnvConfig(
        xml_path=xml_path,
        episode_length=100,
        frame_skip=1,
        render=False,
    )
    env = MujocoEnv(cfg)
    model = env.model

    print("\n=============================================")
    print(f"XML: {xml_path}")
    print("=============================================")
    print(f"nq = {model.nq}, nv = {model.nv}, njnt = {model.njnt}\n")

    print(f"{'j':>2} | {'joint_name':>20} | {'type':>10} | {'q_idx':>5} | {'qd_idx':>6}")
    print("-" * 60)

    for j in range(model.njnt):
        name = safe_name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = model.jnt_type[j]

        # Convert joint type enum to readable string
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            jtype_str = "free"
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            jtype_str = "ball"
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            jtype_str = "slide"
        elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
            jtype_str = "hinge"
        else:
            jtype_str = f"type_{jtype}"

        q_idx = model.jnt_qposadr[j]
        v_idx = model.jnt_dofadr[j]

        print(f"{j:2d} | {name:20s} | {jtype_str:10s} | {q_idx:5d} | {v_idx:6d}")

    env.close()


if __name__ == "__main__":
    inspect_joints("assets/biped/biped.xml")
    print()
    inspect_joints("assets/walker2d/walker2d.xml")