"""Full-pipeline comparison of LQR (design) and LQI (extension) on paired seeds.

Reads results/eval/{nominal,robustness,control_lqi}.csv (scripts/evaluate.py)
and writes results/control/pipeline_lqi.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(ROOT, "results", "eval")


def main():
    nom = pd.read_csv(os.path.join(EV, "nominal.csv"))
    rob = pd.read_csv(os.path.join(EV, "robustness.csv"))
    lqi = pd.read_csv(os.path.join(EV, "control_lqi.csv"))
    seeds = sorted(lqi.seed.unique())
    pairs = [("none", nom[nom.condition.isin(["ekf", "nominal"])]), ("motor", rob[rob.condition == "motor"]),
             ("push", rob[rob.condition == "push"])]
    rows = ["| Condition | Controller | Success | Median dock \\|dy\\| (mm) | p95 dock \\|dy\\| (mm) | "
            "Median track RMS lateral vs truth (mm) | Median effort dω (rad/s²) | Median time (s) |",
            "|---|---|---|---|---|---|---|---|"]
    for name, base in pairs:
        base = base[base.seed.isin(seeds)]
        other = lqi[lqi.condition == f"lqi_{name}"]
        for lab, d in (("LQR", base), ("LQI", other)):
            rows.append(f"| {name} | {lab} | {int(d.success.sum())}/{len(d)} | {1e3 * d.dock_dy.abs().median():.1f} | "
                        f"{1e3 * d.dock_dy.abs().quantile(0.95):.1f} | {1e3 * d.track_rmse_lat.median():.1f} | "
                        f"{d.effort_dw.median():.2f} | {d.t_final.median():.1f} |")
    txt = "# Full pipeline: LQR vs LQI (paired seeds 100-111)\n\n" + "\n".join(rows) + "\n"
    with open(os.path.join(ROOT, "results", "control", "pipeline_lqi.md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
