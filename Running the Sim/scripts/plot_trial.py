"""Detailed plots of one trial: map, estimate vs truth, errors with 3-sigma bands.

  python scripts/plot_trial.py --seed 104 --dist push --out results/trial_push.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from tb3dock.frames import wrap
from tb3dock.scene import marker_map
from tb3dock.config import WORLD, RESULTS_DIR, dock_goal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import DIST  # noqa: E402

PH_COL = {"INIT": "#dddddd", "PLAN": "#f2e6c9", "ALIGN": "#f2e6c9", "TRACK": "#ffffff",
          "FACE_DOCK": "#e2efda", "DOCK": "#cfe5c0", "DONE": "#ffffff", "FAIL": "#f4cccc", "GYRO_BIAS": "#eeeeee"}


def draw_world(ax, scenario, grid=None):
    if grid is not None:
        occ = grid.occupied().T
        ext = [grid.origin[0], grid.origin[0] + grid.nx * grid.res, grid.origin[1], grid.origin[1] + grid.ny * grid.res]
        ax.imshow(np.where(occ, 0.35, 1.0), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.6)
    ax.add_patch(Rectangle((0, 0), WORLD.size_x, WORLD.size_y, fill=False, lw=2))
    for o in scenario["obstacles"]:
        if o["kind"] == "box":
            c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
            hx, hy = o["half"]
            pts = [o["c"] + np.array([c * a - s * b, s * a + c * b]) for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]
            ax.add_patch(Polygon(pts, closed=True, fc="#d98c5f", ec="k", alpha=0.5))
        else:
            ax.add_patch(Circle(o["c"], o["half"][0], fc="#7a8fd1", ec="k", alpha=0.5))
    for mid, m in marker_map().items():
        ax.plot(m["p"][0], m["p"][1], "s", color="k" if mid else "#2e7d32", ms=6)
        ax.annotate(str(mid), m["p"][:2], fontsize=7, xytext=(3, 3), textcoords="offset points")
    g = dock_goal()
    ax.plot(g[0], g[1], "*", color="#2e7d32", ms=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--dist", default="none")
    ap.add_argument("--gyro", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "trial_nominal.png"))
    a = ap.parse_args()
    r = run_trial(a.seed, MissionParams(use_gyro=a.gyro), DIST[a.dist](a.seed), keep_log=True)
    plot(r, a.out, f"seed {a.seed}, disturbance '{a.dist}'{' + gyro' if a.gyro else ''}")
    print(a.out, {k: r[k] for k in ("success", "dock_dx", "dock_dy", "dock_dth", "est_rmse_xy", "nees_mean")})


def plot(r, out, label):
    """Draw one trial (run with keep_log=True) to `out`."""
    A = r["log"]
    t, gt, est, P, ph = A["t"], A["gt"], A["est"], A["P"], A["phase"]
    v = ~np.isnan(est[:, 0])
    e = est - gt
    e[:, 2] = wrap(e[:, 2])
    sd = np.sqrt(np.einsum("tii->ti", P))

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_world(ax, r["scenario"], r["grid"])
    ax.plot(gt[:, 0], gt[:, 1], "-", color="#2e7d32", lw=2, label="ground truth")
    ax.plot(est[v, 0], est[v, 1], "--", color="#1f4e9c", lw=1.5, label="EKF estimate")
    ref = A["ref"]
    ax.plot(ref[:, 0], ref[:, 1], ":", color="#c08a00", lw=1.5, label="reference")
    for i in np.where(v)[0][::60]:
        vals, vecs = np.linalg.eigh(P[i, :2, :2])
        ang = np.linspace(0, 2 * np.pi, 40)
        ell = (vecs @ (3 * np.sqrt(np.maximum(vals, 0))[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))) * 10
        ax.plot(est[i, 0] + ell[0], est[i, 1] + ell[1], color="#1f4e9c", lw=0.7, alpha=0.7)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, WORLD.size_x + 0.2)
    ax.set_ylim(-0.2, WORLD.size_y + 0.2)
    ax.set_title(f"{label}: "
                 f"{'SUCCESS' if r['success'] else 'FAIL'}, dock err dx={r['dock_dx'] * 1e3:.1f} mm "
                 f"dy={r['dock_dy'] * 1e3:.1f} mm dθ={np.degrees(r['dock_dth']):.2f}°\n"
                 "grey = LiDAR occupancy map built online; ellipses = EKF 3σ (×10)", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)

    labels = ["x error (mm)", "y error (mm)", "heading error (deg)"]
    sc = [1e3, 1e3, np.degrees(1)]
    for k in range(3):
        a2 = fig.add_subplot(gs[k, 1])
        for p0, p1, name in _spans(t, ph):
            a2.axvspan(p0, p1, color=PH_COL.get(name, "w"), lw=0)
        a2.fill_between(t[v], -3 * sd[v, k] * sc[k], 3 * sd[v, k] * sc[k], color="#9ecae1", alpha=0.6, label="±3σ")
        a2.plot(t[v], e[v, k] * sc[k], color="#1f4e9c", lw=1, label="error (est − truth)")
        lim = np.nanpercentile(np.abs(e[v, k] * sc[k]), 99) * 2 + 1e-3
        a2.set_ylim(-lim, lim)
        a2.set_ylabel(labels[k], fontsize=8)
        if k == 0:
            a2.legend(fontsize=7, loc="upper right", ncol=2)
    a3 = fig.add_subplot(gs[3, 1])
    a3.step(t, A["n_obs"], where="post", color="#999999", lw=0.8, label="markers detected")
    a3.step(t, A["n_acc"], where="post", color="#2e7d32", lw=1, label="markers fused")
    a3.set_ylabel("markers", fontsize=8)
    a3.set_xlabel("time (s)")
    a3.legend(fontsize=7, loc="upper right")
    for p0, p1, name in _spans(t, ph):
        if p1 - p0 > 1.5:
            a3.text((p0 + p1) / 2, a3.get_ylim()[1] * 0.85, name, ha="center", fontsize=6, rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _spans(t, ph):
    out, s = [], 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[s]:
            out.append((t[s], t[min(i, len(t) - 1)], ph[s]))
            s = i
    return out


if __name__ == "__main__":
    main()
