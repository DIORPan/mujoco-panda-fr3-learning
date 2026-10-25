import mujoco
import mujoco.viewer
from fr3_model import build_fr3_model

model = build_fr3_model()
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

for name in ("panda_finger_joint1", "panda_finger_joint2"):
    data.joint(name).qpos[0] = 0.04

gripper_id = model.actuator("panda_actuator8").id
data.ctrl[gripper_id] = 255

mujoco.mj_forward(model, data)

print("抓取点世界坐标：", data.site("panda_grasp_site").xpos)
print("nq =", model.nq, "nv =", model.nv, "nu =", model.nu)

mujoco.viewer.launch(model, data)