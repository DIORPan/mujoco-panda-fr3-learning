"""Fixed-start reach-and-grasp task: ten state features and four actions.

Setup moves to the same hover position for every cube. Task actions alone
change XYZ references; IK keeps the existing downward gripper orientation.
"""

import contextlib
import io

import mujoco
import numpy as np

from collect_bc_episode import GREEN_X_RANGE, GREEN_Y_RANGE
from grasp_task import VerticalGraspTask


OBSERVATION_NAMES = [
    "grasp_minus_cube_x_m", "grasp_minus_cube_y_m", "grasp_minus_cube_z_m",
    "cube_lift_m", "mean_finger_position_m", "previous_gripper_open",
    "reference_minus_grasp_x_m", "reference_minus_grasp_y_m", "reference_minus_grasp_z_m",
    "remaining_reference_lift_m",
]
OBSERVATION_SCALES = np.asarray([.2, .2, .2, .2, .04, 1., .02, .02, .02, .2], dtype=np.float32)


class ReachGraspTask(VerticalGraspTask):
    HORIZON = 8.0
    MAX_DELTA_XYZ = .002
    START_POSITION = np.asarray([.40, .0, .40])
    REFERENCE_LOWER = np.asarray([.34, -.18, .225])
    REFERENCE_UPPER = np.asarray([.64, .18, .40])

    def _ik(self, position):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.servo.ik(position, self.q_reference)

    def reset(self, seed, viewer=None):
        self.servo.reset(randomize=False)
        rng = np.random.default_rng(seed)
        self.requested_cube_xy = np.asarray([rng.uniform(*GREEN_X_RANGE), rng.uniform(*GREEN_Y_RANGE)])
        joint = self.model.body("cube_green").jntadr[0]
        address = int(self.model.jnt_qposadr[joint])
        self.data.qpos[address:address + 7] = [*self.requested_cube_xy, .25, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.servo._settle(300, viewer)
        # This reference is constant and independent of cube position.
        self.reference_position = self.START_POSITION.copy()
        self.q_reference = self.servo.HOME.copy()
        self.q_reference = self._ik(self.reference_position)
        self.servo._move_to(self.q_reference, self.servo.GRIPPER_OPEN, 1.5, viewer)
        self.servo._settle(100, viewer)
        if viewer is not None and not viewer.is_running():
            raise InterruptedError("Viewer closed during fixed-start setup.")
        fields = self.reader.fields(self.data)
        self.initial_cube_z = float(fields["cube_position (m)"][2])
        self.initial_grasp_position = fields["grasp_position (m)"].copy()
        self.initial_xy_error = float(np.linalg.norm(self.initial_grasp_position[:2] - fields["cube_position (m)"][:2]))
        self.gripper_open = 1.0
        self.start_time = self.data.time
        self.decisions = 0
        self.pose_history.clear()
        self._record_pose()
        return self.observe()

    def observe(self):
        fields = self.reader.fields(self.data)
        grasp = fields["grasp_position (m)"]
        cube = fields["cube_position (m)"]
        return np.concatenate([
            grasp - cube,
            [cube[2] - self.initial_cube_z, fields["finger_qpos (m)"].mean(), self.gripper_open],
            self.reference_position - grasp,
            [self.servo.HOVER_Z - self.reference_position[2]],
        ])

    def step(self, action, viewer=None):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Expected four finite values: normalized delta-XYZ and gripper open probability.")
        if viewer is not None and not viewer.is_running():
            raise InterruptedError("Viewer closed during the task.")
        old_reference = self.reference_position.copy()
        next_reference = np.clip(old_reference + np.clip(action[:3], -1, 1) * self.MAX_DELTA_XYZ,
                                 self.REFERENCE_LOWER, self.REFERENCE_UPPER)
        next_q = self._ik(next_reference)
        if not np.isfinite(next_q).all():
            raise RuntimeError("Non-finite IK target.")
        start_q = self.data.ctrl[:7].copy()
        gripper_open = float(action[3] >= .5)
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
        self.reference_position, self.q_reference = next_reference, next_q
        self.gripper_open = gripper_open
        self.decisions += 1
        if viewer is not None:
            viewer.sync()
        applied = np.concatenate([(next_reference - old_reference) / self.MAX_DELTA_XYZ, [gripper_open]])
        return self.observe(), applied


def expert_action(observation):
    """Feedback labels use only the ten policy inputs, with no hidden phase/timer."""
    relative = observation[:3]
    finger_position, previous_open = observation[4:6]
    # reference - cube = (reference - grasp) + (grasp - cube).
    reference_cube_error = observation[6:9] + relative
    remaining_lift = observation[9]
    action = np.asarray([0., 0., 0., 1. if previous_open >= .5 else 0.])
    if previous_open >= .5:
        if np.max(np.abs(reference_cube_error[:2])) > 1e-6:
            action[:2] = np.clip(-reference_cube_error[:2] / ReachGraspTask.MAX_DELTA_XYZ, -1, 1)
            return action
        if np.linalg.norm(relative[:2]) > .01:
            return action  # Wait for actual horizontal tracking before descending.
        downward_travel = ReachGraspTask.START_POSITION[2] - ReachGraspTask.REFERENCE_LOWER[2]
        if remaining_lift < downward_travel - 1e-8:
            action[2] = -1.
        elif abs(relative[2]) <= .02:
            action[3] = 0.
        return action
    if finger_position <= .0205 and abs(relative[2]) <= .03 and remaining_lift > 1e-8:
        action[2] = 1.
    return action
