import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np
from fr3_model import build_fr3_model
from controllers.scripted_controller import ScriptedController

parser = argparse.ArgumentParser()
parser.add_argument("--zero-gravity", action="store_true")
args = parser.parse_args()

model = build_fr3_model()

if args.zero_gravity:
    model.opt.gravity[:] = [0, 0, 0]

print("本次重力加速度：", model.opt.gravity)
data = mujoco.MjData(model)

controller = ScriptedController(
    model, data, site_name="panda_grasp_site"
)
controller.reset()

target = np.array([0.5, 0.0, 0.4])
q = controller.ik(target, controller.HOME)

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller._move_to(
        q, controller.GRIPPER_OPEN, seconds=2.0, viewer=viewer
    )
    controller._settle(
        round(4.0 / model.opt.timestep), viewer=viewer
    )

    if data.time < 6.0 - 1e-8:
        raise RuntimeError("窗口提前关闭，本次实验未完成。")

    mujoco.mj_forward(model, data)
    actual = data.site("panda_grasp_site").xpos.copy()

    print("仿真时间（s）：", data.time)
    print("实际位置误差（mm）：", np.linalg.norm(actual - target) * 1000)
    print("最大关节跟踪误差（rad）：",
          np.max(np.abs(data.qpos[:7] - data.ctrl[:7])))
    print("最大关节速度（rad/s）：",
          np.max(np.abs(data.qvel[:7])))

    # 测量完成后保留画面，不再推进仿真。
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)