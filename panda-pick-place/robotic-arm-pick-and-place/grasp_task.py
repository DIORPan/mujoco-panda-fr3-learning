"""Prepared-hover vertical grasp task with six observations and two actions."""

import contextlib
import io
from collections import deque

import mujoco
import numpy as np

from bc_observation import ObservationReader
from collect_bc_episode import GREEN_X_RANGE, GREEN_Y_RANGE
from controllers.scripted_controller import ScriptedController


OBSERVATION_NAMES = [
    "grasp_minus_cube_z_m", "cube_lift_m", "mean_finger_position_m",
    "previous_gripper_open", "z_reference_minus_grasp_z_m", "remaining_reference_lift_m",
]
# Fixed physical scales avoid amplifying tiny training-set velocity variations.
OBSERVATION_SCALES = np.asarray([0.2, 0.2, 0.04, 1.0, 0.02, 0.2], dtype=np.float32)


class VerticalGraspTask:
    CONTROL_DT = 0.02
    MAX_DELTA_Z = 0.002
    HORIZON = 6.0

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.servo = ScriptedController(model, self.data)
        self.reader = ObservationReader(model)
        self.substeps = int(round(self.CONTROL_DT / model.opt.timestep))
        if not np.isclose(self.substeps * model.opt.timestep, self.CONTROL_DT, rtol=0, atol=1e-12):
            raise ValueError("Control interval must contain whole physics steps.")
        self.hold_steps = int(round(0.5 / model.opt.timestep))
        self.pose_history = deque(maxlen=self.hold_steps + 1)

    def _ik(self, z):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.servo.ik([self.xy[0], self.xy[1], z], self.q_reference)

    def _record_pose(self):
        fields = self.reader.fields(self.data)
        self.pose_history.append((self.data.time, fields["cube_position (m)"], fields["grasp_position (m)"]))

    def reset(self, seed, viewer=None):
        self.servo.reset(randomize=False)
        rng = np.random.default_rng(seed)
        self.xy = np.asarray([rng.uniform(*GREEN_X_RANGE), rng.uniform(*GREEN_Y_RANGE)])
        joint = self.model.body("cube_green").jntadr[0]
        address = int(self.model.jnt_qposadr[joint])
        self.data.qpos[address:address + 7] = [*self.xy, 0.25, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.servo._settle(300, viewer)
        self.q_reference = self.servo.HOME.copy()
        self.z_reference = self.servo.HOVER_Z
        self.q_reference = self._ik(self.z_reference)
        self.servo._move_to(self.q_reference, self.servo.GRIPPER_OPEN, 1.5, viewer)
        self.servo._settle(100, viewer)
        if viewer is not None and not viewer.is_running():
            raise InterruptedError("Viewer closed during scripted setup.")
        self.gripper_open = 1.0
        self.initial_cube_z = float(self.reader.fields(self.data)["cube_position (m)"][2])
        self.start_time = self.data.time
        self.decisions = 0
        self.pose_history.clear()
        self._record_pose()
        return self.observe()

    def observe(self):
        fields = self.reader.fields(self.data)
        grasp_z = fields["grasp_position (m)"][2]
        cube_z = fields["cube_position (m)"][2]
        return np.asarray([
            grasp_z - cube_z, cube_z - self.initial_cube_z,
            fields["finger_qpos (m)"].mean(), self.gripper_open,
            self.z_reference - grasp_z, self.servo.HOVER_Z - self.z_reference,
        ], dtype=np.float64)

    def step(self, action, viewer=None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite action values.")
        if viewer is not None and not viewer.is_running():
            raise InterruptedError("Viewer closed during the task.")
        old_z = self.z_reference
        next_z = float(np.clip(old_z + np.clip(action[0], -1, 1) * self.MAX_DELTA_Z,
                               self.servo.GRASP_Z, self.servo.HOVER_Z))
        next_q = self._ik(next_z)
        start_q = self.data.ctrl[:7].copy()
        gripper_open = float(action[1] >= 0.5)
        old_time = self.data.time
        for index in range(self.substeps):
            self.data.ctrl[:7] = start_q + ((index + 1) / self.substeps) * (next_q - start_q)
            self.data.ctrl[7] = 255.0 * gripper_open
            mujoco.mj_step(self.model, self.data)
            self._record_pose()
        if not np.isclose(self.data.time - old_time, self.CONTROL_DT, rtol=0, atol=1e-8):
            raise RuntimeError("Unexpected physics time reset.")
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            raise RuntimeError("Non-finite physics state.")
        self.z_reference, self.q_reference, self.gripper_open = next_z, next_q, gripper_open
        self.decisions += 1
        if viewer is not None:
            viewer.sync()
        # Save the actual bounded reference increment, rather than an unclipped request.
        applied = np.asarray([(next_z - old_z) / self.MAX_DELTA_Z, gripper_open])
        return self.observe(), applied

    def result(self):
        samples = list(self.pose_history)
        cubes = np.asarray([item[1] for item in samples])
        grasps = np.asarray([item[2] for item in samples])
        hold_duration = float(samples[-1][0] - samples[0][0])
        lift = float(cubes[:, 2].min() - self.initial_cube_z)
        drift = float(np.linalg.norm(cubes - cubes[0], axis=1).max())
        distance = float(np.linalg.norm(cubes - grasps, axis=1).max())
        completed = self.decisions == int(round(self.HORIZON / self.CONTROL_DT))
        success = bool(completed and hold_duration >= 0.5 - 1e-8 and lift >= 0.10 and drift <= 0.01 and distance <= 0.04)
        return {"completed": completed, "grasp_success": success, "decisions": self.decisions,
                "min_lift_last_hold_m": lift, "hold_duration_s": hold_duration,
                "max_hold_displacement_m": drift, "max_grasp_distance_last_hold_m": distance}


def expert_action(observation):
    """Feedback demonstrator using only features also available to the policy."""
    relative_z, _, finger_position, previous_open, _, remaining_lift = observation
    downward_travel = ScriptedController.HOVER_Z - ScriptedController.GRASP_Z
    if previous_open >= 0.5:
        if remaining_lift < downward_travel - 1e-8:
            return np.asarray([-1.0, 1.0])
        # Wait for actual approach before requesting closure.
        return np.asarray([0.0, 0.0 if abs(relative_z) <= 0.02 else 1.0])
    # Fingers must approach contact before the vertical lifting command starts.
    if finger_position > 0.0205 or abs(relative_z) > 0.03:
        return np.asarray([0.0, 0.0])
    return np.asarray([1.0 if remaining_lift > 1e-8 else 0.0, 0.0])
