"""Standalone test plant for designing and evaluating the controller.

Runs in milliseconds per trial, so controllers can be compared over hundreds
of paths without the full MuJoCo pipeline. It reproduces the effects that
matter to the controller, with numbers taken from the team's calibration and
evaluation results (results/*.json in the main repo):

  * TRUE wheel geometry differs from the CALIBRATED geometry the controller
    uses (r_l 33.6 vs 33.28 mm, r_r 32.7 vs 32.39 mm, b 163.5 vs 163.4 mm);
  * first-order wheel lag, tau = 24.6 ms identified (can be slowed);
  * optional left/right motor gain mismatch;
  * optional pure delay of whole control periods;
  * the controller sees an ESTIMATE = truth + slowly varying error
    (first-order Gauss-Markov per axis), sized to the full pipeline's
    nominal EKF error (about 16 mm RMS position, 0.85 deg heading);
  * optional pushes: a sudden lateral shove and yaw kick spread over 0.25 s.

The physics is integrated at 1 ms with commands held for each 50 ms period
(zero-order hold), like the real controller.
"""
from dataclasses import dataclass, field

import numpy as np

from .models import wheels_to_body, tracking_error, wrap
from .lqr import DT

TRUE_GEOM = dict(wheel_radius_l=0.0336, wheel_radius_r=0.0327, track=0.1635)
CAL_GEOM = dict(wheel_radius_l=0.033277, wheel_radius_r=0.032390, track=0.163433)


@dataclass
class Condition:
    name: str = "nominal"
    est_sigma_xy: float = 0.011        # m, per axis
    est_sigma_th: float = 0.015        # rad
    est_tau: float = 2.0               # s, correlation time of estimate error
    tau: float = 0.0246                # s, wheel lag
    motor_gain: tuple = (1.0, 1.0)     # actual / commanded wheel rate, (left, right)
    delay_steps: int = 0               # whole control periods of command delay
    pushes: list = field(default_factory=list)   # [(t, lateral m, yaw rad)]


def conditions():
    """The test conditions used in control_scripts/evaluate_control.py."""
    return {
        "perfect estimate": Condition("perfect estimate", est_sigma_xy=0.0, est_sigma_th=0.0),
        "nominal": Condition("nominal"),
        "3x estimate noise": Condition("3x estimate noise", est_sigma_xy=0.033, est_sigma_th=0.045),
        "motor mismatch": Condition("motor mismatch", motor_gain=(0.90, 1.08)),
        "slow actuator (60 ms)": Condition("slow actuator (60 ms)", tau=0.060),
        "50 ms delay": Condition("50 ms delay", delay_steps=1),
        "pushes": Condition("pushes", pushes=[(6.0, 0.06, 0.25), (14.0, -0.06, -0.30)]),
    }


class DiffDrivePlant:
    def __init__(self, pose0, cond: Condition, rng, h=0.001):
        self.x = np.array(pose0, float)
        self.wheel = np.zeros(2)
        self.c, self.rng, self.h = cond, rng, h
        self.t = 0.0
        self.err = np.zeros(3)
        self.queue = [(0.0, 0.0)] * cond.delay_steps

    def estimate(self):
        """Pose estimate the controller sees: truth + Gauss-Markov error."""
        return np.array([self.x[0] + self.err[0], self.x[1] + self.err[1], wrap(self.x[2] + self.err[2])])

    def _advance_error(self, dt):
        c = self.c
        a = np.exp(-dt / c.est_tau)
        sig = np.array([c.est_sigma_xy, c.est_sigma_xy, c.est_sigma_th])
        self.err = a * self.err + np.sqrt(1 - a * a) * sig * self.rng.standard_normal(3)

    def step(self, wl_cmd, wr_cmd, dt=DT):
        self.queue.append((wl_cmd, wr_cmd))
        wl_cmd, wr_cmd = self.queue.pop(0)
        g = np.asarray(self.c.motor_gain)
        cmd = np.array([wl_cmd, wr_cmd]) * g
        n = int(round(dt / self.h))
        for _ in range(n):
            self.wheel += (cmd - self.wheel) * (self.h / self.c.tau)
            v, w = wheels_to_body(self.wheel[0], self.wheel[1], TRUE_GEOM["wheel_radius_l"],
                                  TRUE_GEOM["wheel_radius_r"], TRUE_GEOM["track"])
            thm = self.x[2] + 0.5 * w * self.h
            self.x += np.array([v * np.cos(thm), v * np.sin(thm), w]) * self.h
            for (tp, lat, yaw) in self.c.pushes:
                if tp <= self.t < tp + 0.25:
                    k = self.h / 0.25
                    th = self.x[2]
                    self.x += np.array([-np.sin(th) * lat * k, np.cos(th) * lat * k, yaw * k])
            self.t += self.h
        self.x[2] = wrap(self.x[2])
        self._advance_error(dt)


def simulate(traj, tracker, cond: Condition, seed=0, start_offset=(0.0, 0.0, 0.0), extra_time=6.0):
    """Run one trajectory with a TrajectoryTracker. Returns a log dict and metrics."""
    rng = np.random.default_rng(seed)
    ref0, _ = traj.sample(0.0)
    c, s = np.cos(ref0[2]), np.sin(ref0[2])
    ox, oy, oth = start_offset
    pose0 = np.array([ref0[0] + c * ox - s * oy, ref0[1] + s * ox + c * oy, ref0[2] + oth])
    plant = DiffDrivePlant(pose0, cond, rng)
    tracker.start(traj, align_first=False)
    T = traj.duration + extra_time
    log = dict(t=[], truth=[], est=[], ref=[], e_est=[], e_true=[], u=[], sat=[])
    while plant.t < T:
        est = plant.estimate()
        st = tracker.update(est)
        if st.ref is not None:
            log["t"].append(plant.t)
            log["truth"].append(plant.x.copy())
            log["est"].append(est)
            log["ref"].append(st.ref)
            log["e_est"].append(st.e)
            log["e_true"].append(tracking_error(plant.x, st.ref))
            log["u"].append(st.u)
            log["sat"].append(st.saturated)
        plant.step(*st.wheels)
        if st.mode == "DONE" and plant.t > traj.duration + 1.0:
            break
    L = {k: np.array(v) for k, v in log.items()}
    et, ee, u = L["e_true"], L["e_est"], L["u"]
    end = np.array(traj.sample(traj.duration)[0])
    fin = tracking_error(plant.x, end)
    dw = np.diff(u[:, 1]) / DT
    m = dict(
        rms_lat_true=float(np.sqrt(np.mean(et[:, 1] ** 2))),
        rms_lat_est=float(np.sqrt(np.mean(ee[:, 1] ** 2))),
        max_lat_true=float(np.max(np.abs(et[:, 1]))),
        rms_along_true=float(np.sqrt(np.mean(et[:, 0] ** 2))),
        rms_head_true=float(np.sqrt(np.mean(et[:, 2] ** 2))),
        effort_dw=float(np.sqrt(np.mean(dw ** 2))),
        sat_frac=float(np.mean(L["sat"])),
        final_pos_err=float(np.hypot(fin[0], fin[1])),
        final_lat=float(fin[1]),
        finish_time=float(L["t"][-1]),
        nominal_time=float(traj.duration),
    )
    return L, m


def random_path(rng, room=(5.0, 4.0), n=None, margin=0.5, min_len=3.0):
    """Random waypoint path in the room, corners rounded, like the planner's output."""
    from .reference import round_corners
    while True:
        k = int(rng.integers(3, 6)) if n is None else n
        pts = np.column_stack([rng.uniform(margin, room[0] - margin, k), rng.uniform(margin, room[1] - margin, k)])
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if segs.min() < 0.6:
            continue
        # avoid near-reversals (the planner never produces them)
        d = np.diff(pts, axis=0)
        ang = np.abs(wrap(np.diff(np.arctan2(d[:, 1], d[:, 0]))))
        if len(ang) and ang.max() > np.radians(135):
            continue
        p = round_corners(pts)
        if np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)) >= min_len:
            return p
