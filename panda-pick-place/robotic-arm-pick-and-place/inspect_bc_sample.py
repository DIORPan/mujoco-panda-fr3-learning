"""Inspect one state/action pair for behavior cloning; no training or file output."""

import os
from pathlib import Path

import mujoco
import numpy as np

from bc_observation import ObservationReader
from controllers.scripted_controller import ScriptedController


def main():
    # On this Windows setup MuJoCo rejects the Chinese absolute workspace path.
    os.chdir(Path(__file__).resolve().parent)
    model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    expert = ScriptedController(model, data)
    expert.reset(randomize=False)
    expert._settle(300)
    mujoco.mj_forward(model, data)

    # Observation s_t is copied BEFORE applying the action or advancing physics.
    fields = ObservationReader(model).fields(data)
    cube_position = fields["cube_position (m)"]
    obs = np.concatenate(list(fields.values()))

    # Match the first 2 ms of the expert's 1.5 s hover motion (_move_to).
    # IK only computes the destination. The action is the FIRST ramped ctrl,
    # rather than the final IK solution or the resulting observed joint angles.
    hover = [cube_position[0], cube_position[1], expert.HOVER_Z]
    destination = expert.ik(hover, expert.HOME)
    motion_steps = int(1.5 / model.opt.timestep)
    next_arm_ctrl = data.ctrl[:7] + (destination - data.ctrl[:7]) / motion_steps
    action = np.concatenate([next_arm_ctrl, [expert.GRIPPER_OPEN / 255.0]])

    np.set_printoptions(precision=5, suppress=True)
    print("\n--- Observation BEFORE action ---")
    for name, values in fields.items():
        print(f"{name:24s} {values}")
    print(f"obs.shape = {obs.shape}")
    print("\n--- Expert action ---")
    print(f"arm position targets (rad): {action[:7]}")
    print(f"gripper command [0, 1]:     {action[7]:.1f} (1=open, 0=close)")
    print(f"action.shape = {action.shape}")

    # Convert the normalized gripper command back to the model's [0, 255].
    data.ctrl[:7] = action[:7]
    data.ctrl[7] = action[7] * 255.0
    time_before = data.time
    mujoco.mj_step(model, data)
    # Refresh derived positions after integration, for inspecting the result.
    mujoco.mj_forward(model, data)
    print(f"\nApplied one action; simulated dt = {data.time - time_before:.4f} s")
    displacement = data.site_xpos[expert.site_id] - fields["grasp_position (m)"]
    print("grasp displacement (m):", np.array2string(displacement, precision=3, suppress_small=False))
    print("No dataset saved. No policy trained. This is one hover sample only.")


if __name__ == "__main__":
    main()
