import json
import time
from pathlib import Path
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

report = {
    "scene": "fixed",
    "order": ["green", "red", "blue"],
    "gravity_m_s2": model.opt.gravity.tolist(),
    "criterion": {"xy_error_m_lt": 0.05, "center_z_m_gt": 0.18},
    "sampling": "after_each_pick_and_place",
    "completed": False,
    "results": [],
}

with mujoco.viewer.launch_passive(model, data) as viewer:
    for color in report["order"]:
        success = controller.pick_and_place(
            f"cube_{color}", f"target_{color}", viewer=viewer
        )
        if not viewer.is_running():
            print("窗口提前关闭，本轮未完整执行。")
            break

        mujoco.mj_forward(model, data)
        cube = data.body(f"cube_{color}").xpos.copy()
        target = data.site(f"target_{color}").xpos.copy()
        xy_mm = float(np.linalg.norm(cube[:2] - target[:2]) * 1000)

        report["results"].append({
            "color": color,
            "success": bool(success),
            "cube_position_m": cube.tolist(),
            "xy_error_mm": xy_mm,
            "cube_center_z_m": float(cube[2]),
        })
        print(f"{color}: {'SUCCESS' if success else 'FAIL'}, "
              f"XY={xy_mm:.3f} mm, z={cube[2]:.4f} m")

    report["completed"] = len(report["results"]) == 3
    output_dir = Path("runs/fr3_migration")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fixed_{time.time_ns()}.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = sum(row["success"] for row in report["results"])
    print(f"本轮完成：{report['completed']}，原判据通过：{passed}/3")
    print("结果文件：", output.resolve())

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.05)