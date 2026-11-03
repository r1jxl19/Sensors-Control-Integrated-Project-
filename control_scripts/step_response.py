"""Recovery from an initial offset, and an example path, for the three controllers.

Straight 4 m line at 0.18 m/s, perfect estimate, starting
  small : 10 cm left of the path, heading 0.2 rad off
  large : 40 cm left of the path, heading 0.6 rad off (outside the linear region)
plus one random path under nominal conditions showing truth, estimate and reference.

  python control_scripts/step_response.py
Outputs: results/control/step_response.png, results/control/example.png, step_response.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3control import Trajectory, TrajectoryTracker, LQRController, LQIController, KanayamaController, resample
from tb3control.plant import CAL_GEOM, Condition, simulate, conditions, random_path

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "control")
CTRL = {"LQR": LQRController, "LQI": LQIController, "Kanayama": KanayamaController}
COL = {"LQR": "#3b6ea5", "LQI": "#4f9a62", "Kanayama": "#c0504d"}


def settle(t, e, tol):
    """First time after which |e| stays below tol."""
    bad = np.where(np.abs(e) >= tol)[0]
    if len(bad) == 0:
        return 0.0
    return float(t[bad[-1] + 1]) if bad[-1] + 1 < len(t) else float("nan")


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = Trajectory(resample(np.array([[0.5, 2.0], [5.5, 2.0]]), 0.02))
    perfect = Condition("perfect", est_sigma_xy=0, est_sigma_th=0)
    res = {}
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.6), sharex=True)
    for row, (name, off) in enumerate((("small", (0, 0.10, 0.2)), ("large", (0, 0.40, 0.6)))):
        for c in CTRL:
            L, m = simulate(tr, TrajectoryTracker(CTRL[c](), CAL_GEOM), perfect, start_offset=off)
            t, e, u = L["t"], L["e_true"], L["u"]
            s = settle(t, e[:, 1], 0.005)
            res[f"{name}/{c}"] = dict(settle_5mm_s=s, settle_distance_m=float(np.interp(s, t, np.cumsum(np.r_[0, np.abs(u[:-1, 0]) * 0.05]))) if np.isfinite(s) else None,
                                     peak_w=float(np.max(np.abs(u[:, 1]))), overshoot_mm=float(-1e3 * min(0, e[:, 1].min())),
                                     sat_frac=m["sat_frac"])
            ax[row, 0].plot(t, e[:, 1] * 1e3, color=COL[c], label=f"{c} (settles to 5 mm at {s:.1f} s)")
            ax[row, 1].plot(t, np.degrees(e[:, 2]), color=COL[c])
            ax[row, 2].plot(t, u[:, 1], color=COL[c])
        ax[row, 0].set_ylabel(f"{name} offset\ncross-track error (mm)")
        ax[row, 0].legend(fontsize=8)
        ax[row, 1].set_ylabel("heading error (deg)")
        ax[row, 2].set_ylabel("yaw-rate command (rad/s)")
        for a in ax[row]:
            a.axhline(0, color="k", lw=0.6)
            a.grid(alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("time (s)")
    ax[0, 0].set_title("Cross-track error")
    ax[0, 1].set_title("Heading error")
    ax[0, 2].set_title("Command")
    fig.suptitle("Recovery from an initial offset on a straight line at 0.18 m/s (perfect estimate)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "step_response.png"), dpi=130)

    # ---- example path, nominal conditions
    rng = np.random.default_rng(1003)
    p = random_path(rng)
    trj = Trajectory(p)
    L, m = simulate(trj, TrajectoryTracker(LQRController(), CAL_GEOM), conditions()["nominal"], seed=3)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax[0].plot(p[:, 0], p[:, 1], color="#c08a00", lw=1.5, ls=":", label="reference path")
    ax[0].plot(L["truth"][:, 0], L["truth"][:, 1], color="#2e7d32", lw=1.8, label="true path")
    ax[0].plot(L["est"][:, 0], L["est"][:, 1], color="#1f4e9c", lw=0.8, alpha=0.7, label="estimate the controller used")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"LQR on a random path, nominal: cross-track RMS {m['rms_lat_true'] * 1e3:.1f} mm vs truth")
    ax[1].plot(L["t"], L["e_true"][:, 1] * 1e3, color="#2e7d32", label="vs ground truth")
    ax[1].plot(L["t"], L["e_est"][:, 1] * 1e3, color="#1f4e9c", lw=0.8, label="vs estimate (controller's view)")
    ax[1].plot(L["t"], (L["e_est"][:, 1] - L["e_true"][:, 1]) * 1e3, color="#999999", lw=0.8,
               label="estimate error, cross-track component")
    ax2 = ax[1].twinx()
    ax2.fill_between(L["t"], 0, L["sat"].astype(float), color="#c0504d", alpha=0.12, step="post")
    ax2.set_ylim(0, 4)
    ax2.set_yticks([])
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("cross-track error (mm)")
    ax[1].set_title("Error seen vs error achieved (red: command saturated; peak = 135° corner)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "example.png"), dpi=130)
    res["example"] = m
    with open(os.path.join(OUT, "step_response.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
