import mujoco.viewer
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

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller._move_to(
        q, controller.GRIPPER_OPEN, seconds=2.0, viewer=viewer
    )

    # 保持当前控制目标，继续仿真，直到手动关闭窗口。
    while viewer.is_running():
        controller._settle(100, viewer)

# 根据运动后的实际关节状态，刷新末端坐标。
mujoco.mj_forward(model, data)
moved_tip = data.site("grasp_site").xpos.copy()

print("实际运动后的末端位置：", moved_tip)
print("实际运动后的位置误差（米）：",
      np.linalg.norm(moved_tip - target))

joint_error = data.qpos[:7] - data.ctrl[:7]

print("七个关节的跟踪误差（rad）：", np.round(joint_error, 6))
print("最大关节跟踪误差（rad）：", np.max(np.abs(joint_error)))
print("最大关节速度（rad/s）：", np.max(np.abs(data.qvel[:7])))