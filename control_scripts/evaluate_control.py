"""Part D evaluation on the standalone plant (fast, so many paths per condition).

Experiments (all paired: the same 40 random paths and noise seeds everywhere)
  A  controllers x conditions : LQR (design), LQI (extension), Kanayama (baseline)
     under 7 conditions (tb3control.plant.conditions)
  B  LQR input-weight sweep    : R x {0.1, 0.3, 1, 3, 10}, nominal and 3x estimate noise
  C  docking approach          : 0.5 m straight line at 6 cm/s from a 2 cm / 2 deg offset,
                                 nominal and motor mismatch; final cross-track error

  python control_scripts/evaluate_control.py            (~4 min on 2 cores)
Outputs: results/control/*.csv, summary.md, tracking.png, tradeoff.png, docking.png, example.png
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3control import (Trajectory, TrajectoryTracker, LQRController, LQIController, KanayamaController, bryson,
                        straight_approach)
from tb3control.plant import CAL_GEOM, conditions, simulate, random_path, Condition

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "control")
N_PATHS = 40
CTRL = {"LQR": LQRController, "LQI": LQIController, "Kanayama": KanayamaController}
COL = {"LQR": "#3b6ea5", "LQI": "#4f9a62", "Kanayama": "#c0504d"}


def path_and_offset(i):
    rng = np.random.default_rng(1000 + i)
    p = random_path(rng)
    # ALIGN leaves up to 0.08 rad of heading error; start somewhere in that band
    return p, (0.0, 0.0, float(rng.uniform(-0.08, 0.08)))


def job_a(args):
    i, ctrl, cname = args
    p, off = path_and_offset(i)
    _, m = simulate(Trajectory(p), TrajectoryTracker(CTRL[ctrl](), CAL_GEOM), conditions()[cname], seed=i,
                    start_offset=off)
    return dict(exp="A", path=i, controller=ctrl, condition=cname, **m)


def job_b(args):
    i, scale, cname = args
    p, off = path_and_offset(i)
    Q, R = bryson()
    _, m = simulate(Trajectory(p), TrajectoryTracker(LQRController(Q, R * scale), CAL_GEOM), conditions()[cname],
                    seed=i, start_offset=off)
    return dict(exp="B", path=i, R_scale=scale, condition=cname, **m)


def job_c(args):
    i, ctrl, cname = args
    rng = np.random.default_rng(2000 + i)
    tr = straight_approach([4.13, 2.0], [4.68, 2.0], v=0.06, theta_end=0.0)
    off = (0.0, float(rng.uniform(-0.02, 0.02)), float(np.radians(rng.uniform(-2, 2))))
    cond = conditions()[cname]
    L, m = simulate(tr, TrajectoryTracker(CTRL[ctrl](), CAL_GEOM), cond, seed=i, start_offset=off, extra_time=3.0)
    return dict(exp="C", path=i, controller=ctrl, condition=cname, start_lat=off[1], **m)


def main():
    os.makedirs(OUT, exist_ok=True)
    conds = list(conditions())
    with Pool(os.cpu_count()) as pool:
        A = pd.DataFrame(pool.map(job_a, [(i, c, k) for k in conds for c in CTRL for i in range(N_PATHS)], chunksize=4))
        B = pd.DataFrame(pool.map(job_b, [(i, s, k) for k in ("nominal", "3x estimate noise")
                                           for s in (0.1, 0.3, 1, 3, 10) for i in range(N_PATHS)], chunksize=4))
        C = pd.DataFrame(pool.map(job_c, [(i, c, k) for k in ("nominal", "motor mismatch")
                                           for c in CTRL for i in range(N_PATHS)], chunksize=4))
    A.to_csv(os.path.join(OUT, "exp_a_controllers.csv"), index=False)
    B.to_csv(os.path.join(OUT, "exp_b_rsweep.csv"), index=False)
    C.to_csv(os.path.join(OUT, "exp_c_docking.csv"), index=False)
    report(A, B, C)


def report(A, B, C):
    conds = list(conditions())
    mm = lambda s: f"{1e3 * s:.1f}"
    lines = ["# Part D control evaluation (standalone plant, 40 paired random paths)\n"]
    lines.append("## A. Cross-track error vs ground truth, RMS per path (median over paths, mm), and p95 of max\n")
    lines.append("| Condition | LQR median | LQI median | Kanayama median | LQR p95 max | LQI p95 max | Kanayama p95 max |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in conds:
        g = A[A.condition == k]
        med = [mm(g[g.controller == c].rms_lat_true.median()) for c in CTRL]
        p95 = [mm(g[g.controller == c].max_lat_true.quantile(0.95)) for c in CTRL]
        lines.append(f"| {k} | " + " | ".join(med + p95) + " |")
    lines.append("\n## A. Other metrics (median over paths)\n")
    lines.append("| Condition | Controller | heading RMS (deg) | effort RMS dω/dt (rad/s²) | saturated (% of steps) | final position error (mm) | finish − nominal (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for k in conds:
        for c in CTRL:
            g = A[(A.condition == k) & (A.controller == c)]
            lines.append(f"| {k} | {c} | {np.degrees(g.rms_head_true.median()):.2f} | {g.effort_dw.median():.2f} | "
                         f"{100 * g.sat_frac.median():.1f} | {mm(g.final_pos_err.median())} | "
                         f"{(g.finish_time - g.nominal_time).median():.2f} |")
    # paired comparisons
    lines.append("\n## A. Paired differences vs LQR (same path, same noise): cross-track RMS\n")
    lines.append("| Condition | LQI better on | LQI median change (mm) | Kanayama better on | Kanayama median change (mm) |")
    lines.append("|---|---|---|---|---|")
    for k in conds:
        g = A[A.condition == k].pivot(index="path", columns="controller", values="rms_lat_true")
        dl, dk = g["LQI"] - g["LQR"], g["Kanayama"] - g["LQR"]
        lines.append(f"| {k} | {(dl < 0).sum()}/{len(g)} | {mm(dl.median())} | {(dk < 0).sum()}/{len(g)} | {mm(dk.median())} |")
    lines.append("\n## B. LQR input-weight sweep (median over paths)\n")
    lines.append("| Condition | R scale | cross-track vs truth (mm) | cross-track vs estimate (mm) | effort (rad/s²) | saturated (%) |")
    lines.append("|---|---|---|---|---|---|")
    for k in ("nominal", "3x estimate noise"):
        for s in (0.1, 0.3, 1, 3, 10):
            g = B[(B.condition == k) & (B.R_scale == s)]
            lines.append(f"| {k} | {s:g} | {mm(g.rms_lat_true.median())} | {mm(g.rms_lat_est.median())} | "
                         f"{g.effort_dw.median():.2f} | {100 * g.sat_frac.median():.1f} |")
    lines.append("\n## C. Docking approach: final cross-track error at the dock (mm)\n")
    lines.append("| Condition | Controller | median abs | p95 abs | max abs |")
    lines.append("|---|---|---|---|---|")
    for k in ("nominal", "motor mismatch"):
        for c in CTRL:
            g = C[(C.condition == k) & (C.controller == c)].final_lat.abs()
            lines.append(f"| {k} | {c} | {mm(g.median())} | {mm(g.quantile(0.95))} | {mm(g.max())} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(A, B, C)


def figures(A, B, C):
    conds = list(conditions())
    # ---- tracking by condition
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.26
    x = np.arange(len(conds))
    for j, c in enumerate(CTRL):
        g = A[A.controller == c].groupby("condition").rms_lat_true
        med = np.array([g.median()[k] for k in conds]) * 1e3
        lo = np.array([g.quantile(0.25)[k] for k in conds]) * 1e3
        hi = np.array([g.quantile(0.75)[k] for k in conds]) * 1e3
        ax.bar(x + (j - 1) * w, med, w, color=COL[c], label=c, yerr=[med - lo, hi - med], capsize=2,
               error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=12)
    ax.set_ylabel("cross-track RMS vs ground truth (mm)\nmedian over 40 paths, bars = IQR")
    ax.set_title("Estimation noise dominates tracking error; LQI helps only against a persistent bias (motor mismatch)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tracking.png"), dpi=130)

    # ---- trade-off (R sweep)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, k in zip(ax, ("nominal", "3x estimate noise")):
        g = B[B.condition == k].groupby("R_scale")
        e_t, e_e, eff = g.rms_lat_true.median() * 1e3, g.rms_lat_est.median() * 1e3, g.effort_dw.median()
        a.plot(eff, e_t, "o-", color="#3b6ea5", label="vs ground truth (what the robot did)")
        a.plot(eff, e_e, "s--", color="#c0504d", label="vs estimate (what the controller saw)")
        for s in e_t.index:
            a.annotate(f"R×{s:g}", (eff[s], e_t[s]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        a.set_xlabel("control effort: RMS dω/dt (rad/s²)")
        a.set_ylabel("cross-track RMS (mm)")
        a.set_title(f"R sweep, {k}")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=130)

    # ---- docking
    fig, ax = plt.subplots(figsize=(8, 4.2))
    data, labels, cols = [], [], []
    for k in ("nominal", "motor mismatch"):
        for c in CTRL:
            data.append(C[(C.condition == k) & (C.controller == c)].final_lat.abs().values * 1e3)
            labels.append(f"{c}\n{k}")
            cols.append(COL[c])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for b, c in zip(bp["boxes"], cols):
        b.set_facecolor(c)
        b.set_alpha(0.5)
    ax.axhline(20, color="k", ls="--", lw=1)
    ax.text(0.6, 21, "success tolerance 20 mm", fontsize=8)
    ax.set_ylabel("final cross-track error at dock (mm)")
    ax.set_title("Docking approach at 6 cm/s from a 2 cm / 2° offset")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "docking.png"), dpi=130)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(pd.read_csv(os.path.join(OUT, "exp_a_controllers.csv")), pd.read_csv(os.path.join(OUT, "exp_b_rsweep.csv")),
               pd.read_csv(os.path.join(OUT, "exp_c_docking.csv")))
    else:
        main()
