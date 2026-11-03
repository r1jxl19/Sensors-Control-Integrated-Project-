"""Trajectory follower: reference clock, align-then-track, saturation.

This is the glue between a reference (reference.Trajectory) and a feedback
law (controllers.*). It owns the reference clock t_ref:

    each control step:
        ref, u_ref = traj.sample(t_ref)
        u, e = controller.control(pose_est, ref, u_ref)
        if e_x > -WAIT:  t_ref += dt      # otherwise the reference waits

Why the reference waits: a time-indexed reference keeps moving whether or not
the robot keeps up. After a push or a stall the reference can be metres
ahead; the linearised controller is then far outside its design region and
saturates chasing it. Holding the clock while the robot is more than 10 cm
behind turns a large along-track error into a short pause instead.

Modes for standalone use (update()):
    ALIGN  rotate in place onto the path's initial heading (the LQR's lateral
           gain is designed for v_r >= 5 cm/s, and the path starts at v_r = 0)
    TRACK  feedback tracking as above
    DONE   reference finished; zero command
The full mission (tb3dock/mission.py) calls track_step() directly and keeps
its own phases (replanning, docking), so both share exactly this logic.
"""
from dataclasses import dataclass, field

import numpy as np

from .controllers import heading_control
from .limits import saturate, is_saturated
from .lqr import DT

WAIT = 0.10        # m behind the reference before its clock stops
ALIGN_TOL = 0.08   # rad, heading error to leave ALIGN


@dataclass
class Step:
    u: np.ndarray                      # saturated body command [v, w]
    wheels: tuple                      # wheel-rate commands (rad/s)
    e: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref: np.ndarray = None
    mode: str = "TRACK"
    saturated: bool = False


class TrajectoryTracker:
    def __init__(self, controller, geom, dt=DT, wait=WAIT, align_tol=ALIGN_TOL):
        self.ctrl, self.geom, self.dt = controller, geom, dt
        self.wait, self.align_tol = wait, align_tol
        self.traj = None
        self.t_ref = 0.0
        self.mode = "IDLE"
        self._last_sat = False

    def start(self, traj, align_first=True):
        self.traj = traj
        self.t_ref = 0.0
        self.ctrl.reset()
        self._last_sat = False
        self.mode = "ALIGN" if align_first else "TRACK"

    @property
    def finished(self):
        return self.traj is not None and self.t_ref >= self.traj.duration

    def track_step(self, pose_est):
        """One TRACK step without saturation: returns (u_cmd, e, ref)."""
        ref, u_ref = self.traj.sample(self.t_ref)
        u, e = self.ctrl.control(pose_est, ref, u_ref, saturated=self._last_sat)
        if e[0] > -self.wait:
            self.t_ref += self.dt
        return u, e, ref

    def note_saturation(self, u_cmd, u_sat):
        self._last_sat = is_saturated(u_cmd, u_sat)

    def update(self, pose_est):
        """Standalone use: align, track, saturate. Returns a Step."""
        e, ref = np.zeros(3), None
        if self.mode == "ALIGN":
            ref, _ = self.traj.sample(0.05)
            u, eh = heading_control(pose_est[2], ref[2])
            if abs(eh) < self.align_tol:
                self.mode = "TRACK"
        elif self.mode == "TRACK":
            u, e, ref = self.track_step(pose_est)
            if self.finished:
                self.mode = "DONE"
        else:
            u = np.zeros(2)
        u_sat, wheels = saturate(u, self.geom)
        if self.mode == "TRACK":
            self.note_saturation(u, u_sat)
        return Step(u=u_sat, wheels=wheels, e=e, ref=ref, mode=self.mode,
                    saturated=is_saturated(u, u_sat))
