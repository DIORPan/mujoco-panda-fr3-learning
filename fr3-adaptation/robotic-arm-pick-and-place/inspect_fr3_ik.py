import mujoco
import numpy as np
from fr3_model import build_fr3_model
from controllers.scripted_controller import ScriptedController

model = build_fr3_model()
data = mujoco.MjData(model)

controller = ScriptedController(
    model, data, site_name="panda_grasp_site"
)
controller.reset()

target = np.array([0.5, 0.0, 0.4])
before = data.qpos.copy()

# IK：从目标位置求七个机械臂关节角。
q = controller.ik(target, controller.HOME)

# FK：在独立状态中检查这些关节角对应的位置。
check = mujoco.MjData(model)
check.qpos[:] = data.qpos
check.qpos[:7] = q
mujoco.mj_forward(model, check)

actual = check.site("panda_grasp_site").xpos.copy()

print("目标位置：", target)
print("FK 抓取点位置：", actual)
print("位置误差（mm）：", np.linalg.norm(actual - target) * 1000)
print("现场关节最大变化：", np.max(np.abs(data.qpos - before)))