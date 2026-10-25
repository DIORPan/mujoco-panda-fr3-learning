import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/hand.xml")
data = mujoco.MjData(model)

last_open = None

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 每两秒切换一次张开、闭合。
        is_open = int(data.time / 2) % 2 == 0
        data.ctrl[0] = 255 if is_open else 0

        if is_open != last_open:
            print("张开" if is_open else "闭合")
            last_open = is_open

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)