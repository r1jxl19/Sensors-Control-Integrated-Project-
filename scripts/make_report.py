"""Aggregate results/eval/*.csv into tables (results/summary.md) and figures."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.config import RESULTS_DIR

EVAL = os.path.join(RESULTS_DIR, "eval")
BLUE, RED, GREY, GREEN = "#3b6ea5", "#c0504d", "#7f7f7f", "#4f9a62"


def load(name):
    p = os.path.join(EVAL, f"{name}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d["condition"] = d["condition"].replace({"ekf": "nominal"})
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarise(df, by="condition"):
    rows = []
    for cond, g in df.groupby(by, sort=False):
        n, k = len(g), int(g.success.sum())
        lo, hi = wilson(k, n)
        ok = g[g.outcome == "DONE"]
        rows.append({
            "condition": cond, "n": n, "success": f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f})",
            "succ_rate": k / n,
            "dock |dy| med (mm)": 1e3 * ok.dock_dy.abs().median(),
            "dock |dy| p95 (mm)": 1e3 * ok.dock_dy.abs().quantile(0.95),
            "dock |dθ| p95 (deg)": np.degrees(ok.dock_dth.abs().quantile(0.95)),
            "est RMSE xy (mm)": 1e3 * g.est_rmse_xy.median(),
            "est RMSE θ (deg)": np.degrees(g.est_rmse_th.median()),
            "NEES mean": g.nees_mean.median(),
            "NEES in 95% band": g.nees_in95.median(),
            "track RMSE lat (mm)": 1e3 * g.track_rmse_lat.median(),
            "effort dω (rad/s²)": g.effort_dw.median(),
            "time (s)": g.t_final.median(),
            "replans": g.n_replans.mean(),
            "relocalise": g.n_relocalise.mean() if "n_relocalise" in g else np.nan,
            "collisions": int(g.collided.sum()),
        })
    return pd.DataFrame(rows)


def md(df, cols):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out


def classify_failure(r):
    if str(r.outcome).startswith("ERROR"):
        return "software error"
    if r.collided:
        return "collision"
    if r.outcome == "FAIL":
        ev = str(r.events)
        if "TRACK" not in ev:
            return "localisation/planning never completed"
        return "timeout"
    if r.est_final_err > 0.015:
        return "estimation (estimate wrong at dock)"
    return "control (estimate right, pose off)"


def main():
    parts = []
    figs = {}
    cols = ["condition", "n", "success", "dock |dy| med (mm)", "dock |dy| p95 (mm)", "dock |dθ| p95 (deg)",
            "est RMSE xy (mm)", "est RMSE θ (deg)", "NEES mean", "NEES in 95% band", "track RMSE lat (mm)",
            "time (s)", "replans", "relocalise", "collisions"]
    nom, rob = load("nominal"), load("robustness")
    odo, gyr, unc = load("baseline_odometry"), load("gyro"), load("uncalibrated")
    allf = [d for d in (nom, rob, odo, gyr, unc) if d is not None]
    if nom is not None:
        S = summarise(pd.concat([d for d in (nom, rob) if d is not None]))
        parts.append("## Nominal and disturbed conditions (EKF: odometry + ArUco)\n\n" + md(S, cols))
    if odo is not None or gyr is not None or unc is not None:
        S2 = summarise(pd.concat([d for d in (nom, odo, gyr, unc) if d is not None]))
        parts.append("## Estimator / sensor variants\n\n" + md(S2, cols))

    v2 = load("v2_livelock_fix")
    if v2 is not None and rob is not None:
        S3 = summarise(pd.concat([rob[rob.condition.isin(["push", "combined"])], v2]))
        parts.append("## v2: livelock re-localisation fix (same seeds)\n\n" + md(S3, cols))

    # ---- robustness figure
    if nom is not None and rob is not None:
        base = summarise(pd.concat([nom, rob]))
        g = summarise(gyr) if gyr is not None else None
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        x = np.arange(len(base))
        ax[0].bar(x - 0.2, base.succ_rate * 100, 0.4, color=BLUE, label="EKF (wheel odometry + ArUco)")
        if g is not None:
            gmap = {"gyro_none": "nominal", "gyro_push": "push", "gyro_slip": "slip", "gyro_combined": "combined"}
            gv = [g[g.condition == k].succ_rate.values[0] * 100 if (g.condition == k).any() else np.nan
                  for k in [next((kk for kk, vv in gmap.items() if vv == c), None) for c in base.condition]]
            ax[0].bar(x + 0.2, gv, 0.4, color=GREEN, label="+ gyro-aided heading")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(base.condition, rotation=20)
        ax[0].set_ylabel("dock success (%)")
        ax[0].set_ylim(0, 105)
        ax[0].legend(fontsize=8, loc="lower left")
        ax[0].set_title("Success: |dx|<30 mm, |dy|<20 mm, |dθ|<3°, no collision")
        data = [pd.concat([nom, rob]).query("condition == @c").est_rmse_xy.values * 1e3 for c in base.condition]
        ax[1].boxplot(data, tick_labels=list(base.condition), showfliers=True)
        ax[1].set_yscale("log")
        ax[1].set_ylabel("estimate RMSE xy over mission (mm)")
        ax[1].set_title("Estimation error by condition")
        ax[1].tick_params(axis="x", rotation=20)
        data = [pd.concat([nom, rob]).query("condition == @c & outcome == 'DONE'").dock_dy.abs().values * 1e3 for c in base.condition]
        ax[2].boxplot(data, tick_labels=list(base.condition))
        ax[2].axhline(20, color=RED, ls="--", lw=1, label="tolerance")
        ax[2].set_yscale("log")
        ax[2].set_ylabel("final lateral dock error |dy| (mm)")
        ax[2].set_title("Final docking accuracy (ground truth)")
        ax[2].tick_params(axis="x", rotation=20)
        ax[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "robustness.png"), dpi=130)

    # ---- EKF vs odometry-only
    if odo is not None and nom is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        pairs = [("nominal", "odom_none", "none"), ("push", "odom_push", "push"), ("slip", "odom_slip", "slip")]
        allrows = pd.concat([nom, rob, odo])
        for i, (a, b, lab) in enumerate(pairs):
            A_ = allrows[allrows.condition == a]
            B_ = allrows[allrows.condition == b]
            ax[0].bar(i - 0.2, A_.success.mean() * 100, 0.4, color=BLUE, label="EKF" if i == 0 else None)
            ax[0].bar(i + 0.2, B_.success.mean() * 100, 0.4, color=GREY, label="odometry only" if i == 0 else None)
            ax[1].bar(i - 0.2, A_.est_final_err.median() * 1e3, 0.4, color=BLUE)
            ax[1].bar(i + 0.2, B_.est_final_err.median() * 1e3, 0.4, color=GREY)
        for a_ in ax:
            a_.set_xticks(range(3))
            a_.set_xticklabels([p[2] for p in pairs])
        ax[0].set_ylabel("dock success (%)")
        ax[0].legend()
        ax[1].set_ylabel("median estimate error at end (mm)")
        ax[1].set_yscale("log")
        ax[0].set_title("Value of the camera updates")
        ax[1].set_title("Estimate error when the robot stops")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "ekf_vs_odometry.png"), dpi=130)

    # ---- estimator trade-off sweeps
    sa, sb = load("sweep_alpha"), load("sweep_beta")
    if sa is not None:
        sa["val"] = sa.condition.str.split("_").str[1].astype(float)
        G = sa.groupby("val").agg(nees=("nees_mean", "median"), rmse=("est_rmse_xy", "median"),
                                  rmse_th=("est_rmse_th", "median"), succ=("success", "mean"),
                                  inband=("nees_in95", "median"), gated=("n_gated", "median"),
                                  acc=("n_accepted", "median")).reset_index()
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        ax[0].loglog(G.val, G.nees, "o-", color=BLUE, label="EKF: process noise × α")
        ax[0].axhline(3, color="k", ls="--", lw=1, label="consistent (NEES = dim = 3)")
        if sb is not None:
            sb["val"] = sb.condition.str.split("_").str[1].astype(float)
            Gb = sb.groupby("val").agg(nees=("nees_mean", "median"), rmse=("est_rmse_xy", "median"),
                                       succ=("success", "mean"), gated=("n_gated", "median")).reset_index()
            Gb = pd.concat([Gb, pd.DataFrame([dict(val=1.0, nees=G[G.val == 1].nees.values[0],
                                                   rmse=G[G.val == 1].rmse.values[0],
                                                   succ=G[G.val == 1].succ.values[0],
                                                   gated=G[G.val == 1].gated.values[0])])]).sort_values("val")
            ax[0].loglog(Gb.val, Gb.nees, "s--", color=RED, label="EKF: measurement noise × β")
        ax[0].set_xlabel("noise scale factor")
        ax[0].set_ylabel("median mission NEES")
        ax[0].set_title("Consistency (push condition)")
        ax[0].legend(fontsize=8)
        ax[1].semilogx(G.val, G.rmse * 1e3, "o-", color=BLUE, label="α (process)")
        if sb is not None:
            ax[1].semilogx(Gb.val, Gb.rmse * 1e3, "s--", color=RED, label="β (measurement)")
        ax[1].set_xlabel("noise scale factor")
        ax[1].set_ylabel("median estimate RMSE xy (mm)")
        ax[1].set_title("Accuracy")
        ax[1].legend(fontsize=8)
        ax[2].scatter(G.nees, G.rmse * 1e3, c=np.log10(G.val), cmap="Blues", s=60, edgecolor="k")
        for _, r in G.iterrows():
            ax[2].annotate(f"α={r.val:g}", (r.nees, r.rmse * 1e3), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax[2].set_xscale("log")
        ax[2].axvline(3, color="k", ls="--", lw=1)
        ax[2].set_xlabel("median NEES (→ overconfident)")
        ax[2].set_ylabel("median RMSE xy (mm)")
        ax[2].set_title("Consistency vs accuracy trade-off")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "estimator_tradeoff.png"), dpi=130)
        parts.append("## Estimator noise sweep (push condition)\n\n" + md(G.rename(columns={"val": "alpha"}),
                     ["alpha", "nees", "inband", "rmse", "rmse_th", "succ", "gated", "acc"]))
        if sb is not None:
            parts.append("## Measurement noise sweep (push condition)\n\n" + md(Gb.rename(columns={"val": "beta"}),
                         ["beta", "nees", "rmse", "succ", "gated"]))

    sl = load("sweep_lqr")
    if sl is not None:
        sl["val"] = sl.condition.str.split("_").str[1].astype(float)
        G = sl.groupby("val").agg(lat=("track_rmse_lat", "median"), head=("track_rmse_head", "median"),
                                  lat_est=("ctrl_rmse_lat", "median"),
                                  eff=("effort_dw", "median"), effw=("effort_w", "median"),
                                  succ=("success", "mean"), t=("t_final", "median")).reset_index()
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(G.eff, G.lat * 1e3, "o-", color=BLUE, label="vs ground truth (what the robot achieved)")
        ax[0].plot(G.eff, G.lat_est * 1e3, "s--", color=RED, label="vs EKF estimate (what the controller sees)")
        ax[0].legend(fontsize=8)
        for _, r in G.iterrows():
            ax[0].annotate(f"R×{r.val:g}", (r.eff, r.lat * 1e3), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax[0].set_xlabel("control effort: RMS yaw-rate change (rad/s²)")
        ax[0].set_ylabel("RMS lateral tracking error (mm)")
        ax[0].set_title("Tracking error vs control effort (LQR R scaling)")
        ax[1].semilogx(G.val, G.succ * 100, "o-", color=BLUE)
        ax[1].set_xlabel("LQR R scale")
        ax[1].set_ylabel("dock success (%)")
        ax[1].set_ylim(0, 105)
        ax[1].set_title("Success vs LQR input weighting")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, "lqr_tradeoff.png"), dpi=130)
        parts.append("## LQR input-weight sweep (nominal)\n\n" + md(G.rename(columns={"val": "R_scale"}),
                     ["R_scale", "lat_est", "lat", "head", "eff", "effw", "succ", "t"]))

    # ---- failures
    if allf:
        A = pd.concat(allf + [d for d in (sa, sb, sl) if d is not None])
        F = A[~A.success.astype(bool)].copy()
        if len(F):
            F["cause"] = F.apply(classify_failure, axis=1)
            parts.append("## Failures (all experiments)\n\n" + md(
                F[["exp", "condition", "seed", "outcome", "cause", "dock_dx", "dock_dy", "dock_dth", "est_final_err",
                   "n_relocalise"]].round(4), ["exp", "condition", "seed", "outcome", "cause", "dock_dx", "dock_dy",
                                               "dock_dth", "est_final_err", "n_relocalise"]))
            parts.append("### Failure causes by condition\n\n" + md(
                F.groupby(["condition", "cause"]).size().reset_index(name="count"), ["condition", "cause", "count"]))
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("# Evaluation summary (generated by scripts/make_report.py)\n\n" + "\n\n".join(parts))
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
