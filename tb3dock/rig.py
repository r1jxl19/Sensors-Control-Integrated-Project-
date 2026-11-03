"""Test-rig utilities for OFFLINE calibration experiments only.

These mimic what a lab would do with a surveyed jig or a tape measure:
place the robot at a known pose, or move a calibration target. They read
and write simulator state directly and must never be used by the mission
pipeline (tb3dock/mission.py does not import this module).
"""
import numpy as np
import mujoco


def teleport(sim, pose):
    """Place the robot at (x, y, theta) at rest."""
    m, d = sim.model, sim.data
    x, y, th = pose
    d.qpos[0:3] = [x, y, d.qpos[2]]
    d.qpos[3:7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def place_board(sim, pos, R):
    """Move the mocap calibration board; R columns = board x, y, normal in world."""
    d, m = sim.data, sim.model
    bid = m.body("board").mocapid[0]
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    d.mocap_pos[bid] = pos
    d.mocap_quat[bid] = q
    mujoco.mj_forward(m, d)
