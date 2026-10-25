import mujoco
import mujoco.viewer
from fr3_model import build_fr3_model
from controllers.scripted_controller import ScriptedController

model = build_fr3_model(with_scene=True)
data = mujoco.MjData(model)

controller = ScriptedController(
    model, data, site_name="panda_grasp_site"
)
controller.reset()

# 先仿真 1 秒，让方块落稳。
controller._settle(round(1.0 / model.opt.timestep))
mujoco.mj_forward(model, data)

print("nq =", model.nq, "nv =", model.nv, "nu =", model.nu)
print("重力加速度：", model.opt.gravity)

mujoco.viewer.launch(model, data)