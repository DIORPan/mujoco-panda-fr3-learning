import time
import mujoco
import mujoco.viewer
import numpy as np
from fr3_model import build_fr3_model
from controllers.scripted_controller import ScriptedController

model = build_fr3_model(with_scene=True)
data = mujoco.MjData(model)

controller = ScriptedController(
    model, data, site_name="panda_grasp_site"
)
controller.reset()
controller._settle(round(1.0 / model.opt.timestep))

with mujoco.viewer.launch_passive(model, data) as viewer:
    success = controller.pick_and_place(
        "cube_green", "target_green", viewer=viewer
    )

    if not viewer.is_running():
        raise RuntimeError("窗口提前关闭，本次抓放未完整执行。")

    mujoco.mj_forward(model, data)
    cube = data.body("cube_green").xpos.copy()
    target = data.site("target_green").xpos.copy()

    print("绿色方块结果：", "SUCCESS" if success else "FAIL")
    print("方块最终坐标（m）：", cube)
    print("XY 误差（mm）：",
          np.linalg.norm(cube[:2] - target[:2]) * 1000)
    print("方块中心高度（m）：", cube[2])

    # 完成后保留画面，不再推进仿真。
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)