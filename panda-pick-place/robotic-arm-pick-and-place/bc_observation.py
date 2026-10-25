"""Shared 25-value state observation for the first green-cube BC experiment."""

import mujoco
import numpy as np


class ObservationReader:
    def __init__(self, model):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.arm_qpos = [int(model.jnt_qposadr[model.joint(f"joint{i}").id]) for i in range(1, 8)]
        self.arm_qvel = [int(model.jnt_dofadr[model.joint(f"joint{i}").id]) for i in range(1, 8)]
        self.fingers = [
            int(model.jnt_qposadr[model.joint(name).id])
            for name in ("finger_joint1", "finger_joint2")
        ]
        self.grasp_id = model.site("grasp_site").id
        self.cube_id = model.body("cube_green").id
        self.target_id = model.site("target_green").id

    def fields(self, data):
        # mj_step integrates qpos; derived world positions may still describe
        # its pre-integration state. Refresh poses on scratch data, so reading
        # an observation does not change the live physics or solver state.
        self.scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(self.model, self.scratch)
        return {
            "arm_qpos (rad)": data.qpos[self.arm_qpos].copy(),
            "arm_qvel (rad/s)": data.qvel[self.arm_qvel].copy(),
            "finger_qpos (m)": data.qpos[self.fingers].copy(),
            "grasp_position (m)": self.scratch.site_xpos[self.grasp_id].copy(),
            "cube_position (m)": self.scratch.xpos[self.cube_id].copy(),
            "target_position (m)": self.scratch.site_xpos[self.target_id].copy(),
        }

    def read(self, data):
        return np.concatenate(list(self.fields(data).values()))
