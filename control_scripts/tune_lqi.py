"""Choose the LQI integral weight z_max: transient aggression vs offset rejection.

For each z_max: recovery from a 10 cm / 0.2 rad and a 40 cm / 0.6 rad offset
(perfect estimate), and the motor-mismatch straight line (steady offset and
time to remove it). Output: results/control/tune_lqi.json, tune_lqi.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3control import Trajectory, TrajectoryTracker, LQRController, LQIController, resample
from tb3control.plant import CAL_GEOM, Condition, simulate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "control")


def run(ctrl):
    tr = Trajectory(resample(np.array([[0.5, 2.0], [5.5, 2.0]]), 0.02))
    perfect = Condition("perfect", est_sigma_xy=0, est_sigma_th=0)
    out = {}
    for name, off in (("small", (0, 0.10, 0.2)), ("large", (0, 0.40, 0.6))):
        L, m = simulate(tr, TrajectoryTracker(ctrl(), CAL_GEOM), perfect, start_offset=off)
        out[f"{name}_max_heading_deg"] = float(np.degrees(np.abs(L["e_true"][:, 2]).max()))
        out[f"{name}_overshoot_mm"] = float(-1e3 * min(0.0, L["e_true"][:, 1].min()))
    L, m = simulate(tr, TrajectoryTracker(ctrl(), CAL_GEOM), Condition("mm", est_sigma_xy=0, est_sigma_th=0,
                                                                         motor_gain=(0.90, 1.08)))
    cruise = (L["t"] > 12) & (L["t"] < 18)
    out["mismatch_cruise_offset_mm"] = float(1e3 * np.abs(L["e_true"][cruise, 1]).mean())
    out["mismatch_rms_mm"] = 1e3 * m["rms_lat_true"]
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"LQR": run(LQRController)}
    zs = (0.03, 0.05, 0.1, 0.2, 0.4, 0.8)
    for z in zs:
        res[f"LQI z_max={z}"] = run(lambda z=z: LQIController(z_max=z))
        res[f"LQI z_max={z}"]["K"] = np.round(LQIController(z_max=z).gain(0.18, 0.0), 2).tolist()
    with open(os.path.join(OUT, "tune_lqi.json"), "w") as f:
        json.dump(res, f, indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(zs, [res[f"LQI z_max={z}"]["small_max_heading_deg"] for z in zs], "o-", label="LQI, 10 cm offset")
    ax[0].semilogx(zs, [res[f"LQI z_max={z}"]["large_max_heading_deg"] for z in zs], "s-", label="LQI, 40 cm offset")
    ax[0].axhline(res["LQR"]["small_max_heading_deg"], color="C0", ls=":", label="LQR, 10 cm")
    ax[0].axhline(res["LQR"]["large_max_heading_deg"], color="C1", ls=":", label="LQR, 40 cm")
    ax[0].set_xlabel("z_max (m·s)  →  weaker integral")
    ax[0].set_ylabel("peak heading error in recovery (deg)")
    ax[0].set_title("Transient aggression")
    ax[0].legend(fontsize=8)
    ax[1].semilogx(zs, [res[f"LQI z_max={z}"]["mismatch_rms_mm"] for z in zs], "o-", color="C2", label="LQI")
    ax[1].axhline(res["LQR"]["mismatch_rms_mm"], color="C2", ls=":", label="LQR")
    ax[1].set_xlabel("z_max (m·s)  →  weaker integral")
    ax[1].set_ylabel("cross-track RMS, motor mismatch (mm)")
    ax[1].set_title("Offset rejection")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.axvline(0.2, color="grey", lw=1)
        a.grid(alpha=0.3)
    fig.suptitle("Choosing the LQI integral weight (grey line = chosen z_max = 0.2)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tune_lqi.png"), dpi=130)
    for k, v in res.items():
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
