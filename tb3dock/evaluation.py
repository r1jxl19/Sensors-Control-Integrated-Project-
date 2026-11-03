"""Trial runner and metrics. This is where ground truth is used - for scoring only.

The logger below is called *after* each mission step. It reads the mission's
internal estimate and the simulator ground truth side by side; nothing flows
back from here into the mission.
"""
import numpy as np

from .sim import SimRobot, GroundTruthProbe, Disturbances
from .mission import Mission, MissionParams
from .scene import random_scenario
from .frames import wrap
from .config import load_calibration, dock_goal


class Logger:
    def __init__(self, sim):
        self.gt = GroundTruthProbe(sim)
        self.rows = []

    def __call__(self, m: Mission):
        g = self.gt.pose()
        est = m.ekf.x if m.ekf.initialised else np.full(3, np.nan)
        P = m.ekf.P if m.ekf.initialised else np.full((3, 3), np.nan)
        ref = m.ref if m.ref is not None and m.phase in ("TRACK", "DOCK") else np.full(3, np.nan)
        self.rows.append(dict(
            t=m.sim.time, phase=m.phase, gt=g.copy(), est=np.array(est, float).copy(),
            P=np.array(P, float).copy(), ref=np.array(ref, float).copy(), u=m.u.copy(),
            n_obs=len(m.obs), n_acc=len(m.accepted_ids), err=m.err.copy()))

    def arrays(self):
        A = {k: np.array([r[k] for r in self.rows]) for k in ("t", "gt", "est", "P", "ref", "u", "n_obs", "n_acc", "err")}
        A["phase"] = np.array([r["phase"] for r in self.rows])
        return A


def contact_with_obstacles(sim):
    """True if any robot geom touches an obstacle/wall/dock (collision check)."""
    m, d = sim.model, sim.data
    robot = set(range(m.body("base").id, m.nbody))
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        n1, n2 = m.geom(c.geom1).name, m.geom(c.geom2).name
        if (b1 in robot) != (b2 in robot):
            other = n2 if b1 in robot else n1
            if other != "floor":
                return True
    return False


def run_trial(seed, params: MissionParams = None, dist: Disturbances = None, scenario=None,
              cal=None, keep_log=False, n_obstacles=None, frame_cb=None):
    rng = np.random.default_rng(seed)
    scenario = scenario or random_scenario(rng, n_obstacles=n_obstacles)
    sim = SimRobot(scenario, seed=seed, dist=dist)
    cal = cal or load_calibration()
    mission = Mission(sim, cal, params)
    log = Logger(sim)
    collided = [False]

    def cb(m):
        log(m)
        if contact_with_obstacles(sim):
            collided[0] = True
        if frame_cb is not None:
            frame_cb(m, log)

    outcome = mission.run(logger=cb)
    A = log.arrays()
    g = GroundTruthProbe(sim).pose()
    goal = np.array(dock_goal())
    e_dock = g - goal
    e_dock[2] = wrap(e_dock[2])
    valid = ~np.isnan(A["est"][:, 0])
    e = A["est"][valid] - A["gt"][valid]
    e[:, 2] = wrap(e[:, 2])
    P = A["P"][valid]
    nees = np.einsum("ti,tij,tj->t", e, np.linalg.inv(P), e)
    tr = np.isin(A["phase"], ["TRACK", "DOCK"]) & ~np.isnan(A["ref"][:, 0])
    # true tracking error: ground truth vs reference (what the controller actually achieved)
    if tr.any():
        from .model import tracking_error
        te = np.array([tracking_error(gp, rp) for gp, rp in zip(A["gt"][tr], A["ref"][tr])])
        ce = A["err"][tr]
    else:
        te = ce = np.zeros((1, 3))
    u = A["u"]
    res = dict(
        seed=seed, outcome=outcome, collided=collided[0],
        t_final=float(A["t"][-1]),
        dock_dx=float(e_dock[0]), dock_dy=float(e_dock[1]), dock_dth=float(e_dock[2]),
        dock_pos_err=float(np.hypot(e_dock[0], e_dock[1])),
        est_rmse_xy=float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2))) if len(e) else np.nan,
        est_rmse_th=float(np.sqrt(np.mean(e[:, 2] ** 2))) if len(e) else np.nan,
        est_final_err=float(np.hypot(*e[-1, :2])) if len(e) else np.nan,
        nees_mean=float(np.mean(nees)) if len(nees) else np.nan,
        nees_in95=float(np.mean((nees > 0.216) & (nees < 9.35))) if len(nees) else np.nan,
        track_rmse_lat=float(np.sqrt(np.mean(te[:, 1] ** 2))),
        track_rmse_lon=float(np.sqrt(np.mean(te[:, 0] ** 2))),
        track_rmse_head=float(np.sqrt(np.mean(te[:, 2] ** 2))),
        ctrl_rmse_lat=float(np.sqrt(np.mean(ce[:, 1] ** 2))),
        effort_w=float(np.sqrt(np.mean(u[:, 1] ** 2))),
        effort_dw=float(np.sqrt(np.mean(np.diff(u[:, 1]) ** 2)) / 0.05),
        n_replans=mission.n_replans, n_estops=mission.n_estops, n_relocalise=mission.n_relocalise,
        n_accepted=mission.ekf.stats["accepted"], n_gated=mission.ekf.stats["gated"],
        n_envelope=mission.ekf.stats["out_of_envelope"],
        events=mission.events, n_obstacles=len(scenario["obstacles"]),
    )
    tol = (0.03, 0.02, np.radians(3))
    res["success"] = bool(outcome == "DONE" and not collided[0] and abs(e_dock[0]) < tol[0]
                          and abs(e_dock[1]) < tol[1] and abs(e_dock[2]) < tol[2])
    if keep_log:
        res["log"] = A
        res["scenario"] = scenario
        res["grid"] = mission.grid
    return res
