"""Odometry calibration (wheel radii, track) and odometry noise model.

Part 1 - systematic errors.
  The robot drives 1.5 m squares clockwise and counter-clockwise (UMBmark,
  Borenstein & Feng 1996) plus straight runs, logging encoder ticks. The
  start and end poses are measured externally (in the lab: tape measure /
  motion capture; here: the simulator rig). Two estimators are compared:
    * the classical UMBmark closed-form correction (E_d, E_b);
    * a nonlinear least-squares fit of (r_l, r_r, b) that re-integrates the
      logged ticks and minimises end-pose error over all runs (our method:
      uses every run, includes straight runs to fix absolute scale, which
      UMBmark cannot observe).
  Both are validated on held-out random curvy trajectories.

Part 2 - random errors (EKF process noise).
  With calibrated parameters, 1-s odometry increments are compared with
  externally measured increments over random driving. Assuming independent
  wheel-arc noise with variance k |ds_wheel|, k is fitted from the residual
  variance of both the translation and rotation increments.

Outputs: config/calibration.json (wheel_radius_l/r, track, odom_k),
         results/odometry_calibration.{json,png}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.sim import SimRobot, GroundTruthProbe
from tb3dock.rig import teleport
from tb3dock.odometry import dead_reckon
from tb3dock.model import body_to_wheels
from tb3dock.frames import wrap
from tb3dock.config import NOMINAL, TRUE, RATES, save_calibration, RESULTS_DIR

R0, B0 = NOMINAL.wheel_radius_l, NOMINAL.track
DT = RATES.control_dt


def run_profile(profile, start):
    """profile: list of (v, w, duration). Returns ticks (T,2), gt poses (T,3)."""
    sim = SimRobot(dict(start=np.array(start), obstacles=[], no_walls=True), render_camera=False)
    teleport(sim, start)
    for _ in range(10):
        sim.step(0, 0)
    gt = GroundTruthProbe(sim)
    ticks, poses = [sim.read_encoders()], [gt.pose()]
    for v, w, dur in profile:
        wl, wr = body_to_wheels(v, w, R0, R0, B0)
        for _ in range(int(round(dur / DT))):
            sim.step(wl, wr)
            ticks.append(sim.read_encoders())
            poses.append(gt.pose())
    for _ in range(10):                          # come to rest before "measuring"
        sim.step(0, 0)
        ticks.append(sim.read_encoders())
        poses.append(gt.pose())
    return np.array(ticks), np.array(poses)


def square(side, cw, v=0.15, w=1.0):
    turn = -w if cw else w
    seg = [(v, 0.0, side / v), (0.0, 0.0, 0.3), (0.0, turn, (np.pi / 2) / w), (0.0, 0.0, 0.3)]
    return seg * 4


def random_profile(rng, T=20.0):
    prof, t = [], 0.0
    while t < T:
        d = rng.uniform(1.0, 3.0)
        prof.append((rng.uniform(0.05, 0.2), rng.uniform(-1.0, 1.0), d))
        t += d
    return prof


def end_residual(params, runs, w_th=0.3):
    rl, rr, b = params
    res = []
    for ticks, poses in runs:
        est = dead_reckon(ticks, poses[0], rl, rr, b)[-1]
        res += [est[0] - poses[-1, 0], est[1] - poses[-1, 1], w_th * wrap(est[2] - poses[-1, 2])]
    return np.array(res)


def umbmark(runs_cw, runs_ccw, L):
    """Borenstein & Feng closed-form correction, robot-frame errors.

    Returns (r_l, r_r, b). Errors are the centre of gravity of end-position
    errors expressed in the start frame (x along the first leg).
    """
    def cg(runs):
        errs = []
        for ticks, poses in runs:
            est = dead_reckon(ticks, poses[0], R0, R0, B0)[-1]
            th0 = poses[0, 2]
            e = np.array([poses[-1, 0] - est[0], poses[-1, 1] - est[1]])  # true - odometry
            c, s = np.cos(th0), np.sin(th0)
            errs.append([c * e[0] + s * e[1], -s * e[0] + c * e[1]])
        return np.mean(errs, 0)
    x_cw, x_ccw = cg(runs_cw)[0], cg(runs_ccw)[0]
    alpha = (x_cw + x_ccw) / (-4 * L)            # rad, track error
    beta = (x_cw - x_ccw) / (-4 * L)             # rad, diameter-ratio error
    Rcurv = (L / 2) / np.sin(beta / 2) if abs(beta) > 1e-9 else np.inf
    Ed = (Rcurv + B0 / 2) / (Rcurv - B0 / 2)
    b = (np.pi / 2) / (np.pi / 2 - alpha) * B0
    rl = 2 / (Ed + 1) * R0
    rr = 2 / (1 / Ed + 1) * R0
    return rl, rr, b, dict(alpha=float(alpha), beta=float(beta), Ed=float(Ed))


def main():
    rng = np.random.default_rng(7)
    L = 1.5
    start = (1.3, 1.25, 0.0)
    runs_cw = [run_profile(square(L, True, v=v), start) for v in (0.12, 0.15, 0.18)]
    start_ccw = (1.3, 2.75, 0.0)
    runs_ccw = [run_profile(square(L, False, v=v), start_ccw) for v in (0.12, 0.15, 0.18)]
    runs_line = [run_profile([(v, 0, 3.0 / v)], (0.8, y, 0.0)) for v, y in ((0.1, 1.0), (0.15, 2.0), (0.2, 3.0))]
    train = runs_cw + runs_ccw + runs_line
    val = [run_profile(random_profile(rng), (2.5, 2.0, rng.uniform(-np.pi, np.pi))) for _ in range(6)]

    ls = least_squares(end_residual, x0=[R0, R0, B0], args=(train,), x_scale=[1e-3, 1e-3, 1e-2])
    J = ls.jac
    s2 = np.sum(ls.fun ** 2) / max(1, len(ls.fun) - 3)
    cov = np.linalg.inv(J.T @ J) * s2
    rl, rr, b = ls.x
    url, urr, ub, uinfo = umbmark(runs_cw, runs_ccw, L)

    def val_err(p):
        e = np.array([end_residual(p, [r]) for r in val])
        return float(np.mean(np.hypot(e[:, 0], e[:, 1]))), float(np.mean(np.abs(e[:, 2]) / 0.3))

    sets = {"nominal": (R0, R0, B0), "UMBmark": (url, urr, ub), "least-squares": (rl, rr, b)}
    table = {}
    for k, p in sets.items():
        pe, he = val_err(p)
        table[k] = dict(r_l=float(p[0]), r_r=float(p[1]), b=float(p[2]), val_pos_err_m=pe, val_head_err_rad=he)
    table["truth(sim, hidden)"] = dict(r_l=TRUE.wheel_radius_l, r_r=TRUE.wheel_radius_r, b=TRUE.track)

    # ---------- Part 2: odometry noise from 1-s increments over random driving
    noise_runs = [run_profile(random_profile(rng, 30.0), (2.5, 2.0, rng.uniform(-np.pi, np.pi))) for _ in range(6)]
    win = int(round(1.0 / DT))
    e_ds, e_dth, sum_abs = [], [], []
    k_tick = 2 * np.pi / NOMINAL.encoder_ticks
    for ticks, poses in noise_runs:
        odo = dead_reckon(ticks, poses[0], rl, rr, b)
        dn = np.diff(ticks.astype(float), axis=0)
        for i in range(0, len(poses) - win, win):
            j = i + win
            # increments expressed in the frame at the window start
            def incr(P):
                c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
                dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
                return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(P[j, 2] - P[i, 2])])
            g, o = incr(poses), incr(odo)
            e_ds.append(o[0] - g[0])
            e_dth.append(wrap(o[2] - g[2]))
            sl = np.sum(np.abs(dn[i:j, 0])) * k_tick * rl
            sr = np.sum(np.abs(dn[i:j, 1])) * k_tick * rr
            sum_abs.append(sl + sr)
    e_ds, e_dth, sum_abs = map(np.array, (e_ds, e_dth, sum_abs))
    # var(ds) = k (|sl|+|sr|)/4 ; var(dth) = k (|sl|+|sr|)/b^2.
    # Robust (median-based) estimate: for e ~ N(0, s^2), median(e^2/s^2) = 0.455.
    # A plain mean was dominated by a handful of windows with transient chassis
    # pitch during hard acceleration (encoders turn relative to a rocking body).
    keep = sum_abs > 0.02
    k_ds = float(np.median(e_ds[keep] ** 2 / (sum_abs[keep] / 4)) / 0.455)
    k_dth = float(np.median(e_dth[keep] ** 2 / (sum_abs[keep] / b ** 2)) / 0.455)
    odom_k = float(max(k_ds, k_dth))   # conservative: the larger of the two
    out = dict(params=table, ls_std=np.sqrt(np.diag(cov)).tolist(), umbmark=uinfo,
               odom_noise=dict(k_from_translation=float(k_ds), k_from_rotation=float(k_dth), k_used=odom_k,
                               mean_err_ds=float(e_ds.mean()), mean_err_dth=float(e_dth.mean()),
                               n_windows=int(len(e_ds))))
    with open(os.path.join(RESULTS_DIR, "odometry_calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_calibration(dict(wheel_radius_l=float(rl), wheel_radius_r=float(rr), track=float(b),
                          odom_k=[odom_k, odom_k]))

    # ---------- figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for runs, lab in ((runs_cw, "CW"), (runs_ccw, "CCW")):
        ticks, poses = runs[1]
        ax[0].plot(poses[:, 0], poses[:, 1], "k-", lw=1.5, label=f"true {lab}")
        for name, p, st in (("nominal", sets["nominal"], ":"), ("least-squares", sets["least-squares"], "--")):
            o = dead_reckon(ticks, poses[0], *p)
            ax[0].plot(o[:, 0], o[:, 1], st, lw=1.2, label=f"odom {name} {lab}")
    ax[0].set_aspect("equal")
    ax[0].set_title("UMBmark squares: truth vs odometry")
    ax[0].legend(fontsize=6)
    ticks, poses = val[0]
    ax[1].plot(poses[:, 0], poses[:, 1], "k-", lw=1.5, label="true")
    for name, p in sets.items():
        o = dead_reckon(ticks, poses[0], *p)
        ax[1].plot(o[:, 0], o[:, 1], lw=1.1, label=name)
    ax[1].set_aspect("equal")
    ax[1].set_title("Held-out random trajectory")
    ax[1].legend(fontsize=7)
    ax[2].scatter(sum_abs, e_dth ** 2, s=8, alpha=0.5, label=r"$e_{\Delta\theta}^2$")
    xs = np.linspace(0, sum_abs.max(), 20)
    ax[2].plot(xs, odom_k * xs / b ** 2, "r-", label=f"model k={odom_k:.2e} m")
    ax[2].set_xlabel(r"$|\Delta s_l|+|\Delta s_r|$ over 1 s window (m)")
    ax[2].set_ylabel("squared heading-increment error (rad²)")
    ax[2].set_title("Odometry noise model fit")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "odometry_calibration.png"), dpi=130)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
