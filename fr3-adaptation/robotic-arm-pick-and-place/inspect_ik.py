import mujoco
import numpy as np
from controllers.scripted_controller import ScriptedController

model = mujoco.MjModel.from_xml_path(
    "assets/franka_emika_panda/scene.xml"
)
data = mujoco.MjData(model)
controller = ScriptedController(model, data)
controller.reset()

# 1. 取绿色方块的 x、y，高度改为控制器规定的悬停高度。
target = data.body("cube_green").xpos.copy()
target[2] = controller.HOVER_Z

# 2. 从 HOME 姿态开始求解，保存求解前的现场状态。
before = data.qpos[:7].copy()
q = controller.ik(target, controller.HOME)

# 3. 在另一份状态中，用 FK 检查求出的角度。
check_data = mujoco.MjData(model)
check_data.qpos[:] = data.qpos
check_data.qpos[:7] = q
mujoco.mj_forward(model, check_data)
actual = check_data.site("grasp_site").xpos.copy()

print("目标位置：", target)
print("IK 返回的七个关节角度：", q)
print("FK 算出的末端位置：", actual)
print("位置误差（米）：", np.linalg.norm(actual - target))
print("现场关节最大变化：", np.max(np.abs(data.qpos[:7] - before)))