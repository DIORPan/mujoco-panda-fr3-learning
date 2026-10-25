import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/franka_fr3/fr3.xml")
data = mujoco.MjData(model)

home_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

# 放大并显示夹爪安装点，仅改变显示效果。
mount_id = model.site("attachment_site").id
model.site_size[mount_id, 0] = 0.015
model.site_rgba[mount_id] = [1, 0, 1, 1]
model.site_group[mount_id] = 0

print("安装点在末端连杆坐标系中的位置：", model.site_pos[mount_id])
print("安装点在世界坐标系中的位置：", data.site_xpos[mount_id])

mujoco.viewer.launch(model, data)