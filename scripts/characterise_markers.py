"""Characterise ArUco range/bearing measurements (bias, noise, detectability).

The robot is placed (test rig) at a grid of ranges, viewing angles and image
bearings relative to known markers. For each placement the camera image is
processed exactly as in the mission (calibrated K, MarkerDetector) and the
measured range/bearing is compared with the value implied by the surveyed
robot pose and the marker map.

Fitted and saved:
  * range bias correction   r_corr = r_meas - (c1 rn + c2 rn^2) (removes systematic bias),
  * range noise model       sigma_r = a + b (r * 0.16/size)^2  (PnP depth error
                            grows ~ r^2 and inversely with marker size),
  * bearing noise           sigma_b (rad, constant),
  * detection envelope       max reliable range / viewing angle (used to decide
                             when the dock marker can be expected in view).

Outputs: config/calibration.json, results/marker_characterisation.{json,png}
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

from tb3dock.sim import SimRobot
from tb3dock.rig import teleport
from tb3dock.scene import marker_map
from tb3dock.vision import MarkerDetector
from tb3dock.frames import map_to_base, wrap, t_B_C
from tb3dock.config import load_calibration, save_calibration, RESULTS_DIR, WORLD


def main(seed=3):
    rng = np.random.default_rng(seed)
    cal = load_calibration()
    mm = marker_map()
    sim = SimRobot(dict(start=np.array([2.5, 2.0, 0.0]), obstacles=[]), seed=seed)
    det = MarkerDetector(cal["K"], cal["dist"], mm)
    rows = []
    targets = [1, 3, 6, 8, WORLD.dock_marker_id]
    ranges = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5]
    views = np.radians([0, 20, 40, 55, 70])
    for mid in targets:
        p = np.array(mm[mid]["p"][:2])
        n = np.array(mm[mid]["n"][:2])
        for r in ranges:
            for va in views:
                for sgn in ((1,) if va == 0 else (1, -1)):
                    a = np.arctan2(n[1], n[0]) + sgn * va
                    for _ in range(2):
                        bear = rng.uniform(-0.45, 0.45)          # where the marker sits in the image
                        cam = p + r * np.array([np.cos(a), np.sin(a)])
                        th = np.arctan2(p[1] - cam[1], p[0] - cam[0]) - bear
                        base = cam - np.array([np.cos(th), np.sin(th)]) * t_B_C[0]
                        if not (0.15 < base[0] < WORLD.size_x - 0.15 and 0.15 < base[1] < WORLD.size_y - 0.15):
                            continue
                        pose = np.array([base[0], base[1], th])
                        teleport(sim, pose)
                        obs = [o for o in det.detect(sim.read_camera()) if o.id == mid]
                        tb = map_to_base(pose, p)
                        r_true, b_true = np.hypot(*tb), np.arctan2(tb[1], tb[0])
                        row = dict(id=mid, r_true=r_true, b_true=b_true, view=va, img_bearing=bear, detected=bool(obs))
                        if obs:
                            o = obs[0]
                            row.update(r_meas=o.range, b_meas=o.bearing, px=o.pixel_size,
                                       reproj=o.reproj_err, yaw_B=o.yaw_B)
                        rows.append(row)
    import csv
    with open(os.path.join(RESULTS_DIR, "marker_rows.csv"), "w", newline="") as f:
        keys = ["id", "r_true", "b_true", "view", "img_bearing", "detected", "r_meas", "b_meas", "px", "reproj", "yaw_B"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    det_rows = [r for r in rows if r["detected"]]
    rt = np.array([r["r_true"] for r in det_rows])
    rm = np.array([r["r_meas"] for r in det_rows])
    eb = wrap(np.array([r["b_meas"] - r["b_true"] for r in det_rows]))
    # range normalised to a 0.16 m marker: PnP depth error scales with r / marker size
    S0 = WORLD.landmark_size
    rn = np.array([r["r_true"] * S0 / mm[r["id"]]["size"] for r in det_rows])

    # 1) range bias: PnP over-estimates range by an amount that grows ~ r^2
    #    (a near-constant ~0.3 px inward shift of the detected corners, i.e.
    #    a relative error proportional to range). Robust (Huber) fit of
    #       r_meas - r_true = c1 * rn + c2 * rn^2   (rn = normalised range)
    #    on views inside the operating envelope.
    env = np.array([r["view"] <= np.radians(60) for r in det_rows])
    rn_m = np.array([r["r_meas"] * S0 / mm[r["id"]]["size"] for r in det_rows])
    # errors are expressed for a 0.16 m marker (x S0/size) so both marker sizes pool
    sizes = np.array([mm[r["id"]]["size"] for r in det_rows])
    sfit = least_squares(lambda c: (c[0] * rn_m + c[1] * rn_m ** 2 - (rm - rt) * S0 / sizes)[env], [0.0, 0.0],
                         loss="huber", f_scale=0.01)
    bias_c = [float(sfit.x[0]), float(sfit.x[1])]
    corr = rm - (bias_c[0] * rn_m + bias_c[1] * rn_m ** 2) * sizes / S0
    er = corr - rt
    ern = er * S0 / sizes          # normalised error for noise-model fitting
    # 2) noise model sigma_r = a + b r^2 via binned std
    bins = np.array([0, 0.5, 0.9, 1.4, 2.0, 2.7, 3.6, 4.6, 6.0])
    centers, stds, bstd = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rn >= lo) & (rn < hi)
        if m.sum() >= 6:
            centers.append(rn[m].mean())
            # robust std (MAD) - a few near-grazing views produce outliers
            stds.append(np.std(ern[m & env]))     # plain std inside envelope: keeps the tails honest
            bstd.append(1.4826 * np.median(np.abs(eb[m] - np.median(eb[m]))))
    centers, stds, bstd = map(np.array, (centers, stds, bstd))
    # Beyond ~3 m (normalised) the error distribution degrades sharply (grazing
    # views, few pixels per cell); measurements past R_MAX are rejected by the
    # EKF instead of being modelled, so the fit uses bins inside R_MAX.
    R_MAX = 2.5
    inside = centers <= R_MAX
    nfit = least_squares(lambda q: (q[0] + q[1] * centers[inside] ** 2) - stds[inside], [0.003, 0.005],
                         bounds=([0.002, 0], [1, 1]))   # 2 mm floor: unmodelled extrinsic/timing error
    a, b = map(float, nfit.x)
    sigma_b = float(np.max(bstd[inside]))
    # inflate by 1.5 to cover outliers / unmodelled correlation (checked later with NIS)
    INFL = 1.5
    # 3) detection envelope
    envelope = {}
    for r in ranges:
        m = [x for x in rows if abs(x["r_true"] - r) < 0.15]
        if m:
            envelope[str(r)] = float(np.mean([x["detected"] for x in m]))
    view_env = {}
    for va in views:
        m = [x for x in rows if abs(x["view"] - va) < 1e-6]
        view_env[f"{np.degrees(va):.0f}"] = float(np.mean([x["detected"] for x in m]))

    res = dict(n_placements=len(rows), n_detected=len(det_rows), range_bias_c=bias_c,
               range_bias_before_mean=float(np.mean(rm - rt)), range_bias_after_mean=float(np.mean(er)),
               sigma_r_model=[a * INFL, b * INFL], sigma_r_fit_raw=[a, b], sigma_b=sigma_b * INFL,
               sigma_b_raw=sigma_b, inflation=INFL,
               binned=dict(r=centers.tolist(), std_r=stds.tolist(), std_b=bstd.tolist()),
               max_range_norm=R_MAX, detect_vs_range=envelope, detect_vs_view_deg=view_env)
    with open(os.path.join(RESULTS_DIR, "marker_characterisation.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_calibration(dict(range_scale=1.0, range_bias_c=bias_c, meas_sigma_r=[a * INFL, b * INFL], meas_sigma_b=sigma_b * INFL,
                          meas_max_range_norm=R_MAX))

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ax[0].scatter(rt, rm - rt, s=8, alpha=0.5, label="raw")
    ax[0].scatter(rt, er, s=8, alpha=0.5, label=f"after bias model ({bias_c[0]*1e3:.1f} rn + {bias_c[1]*1e3:.1f} rn²) mm")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("true range (m)")
    ax[0].set_ylabel("range error (m)")
    ax[0].set_title("Range bias")
    ax[0].legend(fontsize=8)
    rr_ = np.linspace(0, 6.0, 50)
    ax[1].plot(centers, stds * 1e3, "o", label="binned robust std")
    ax[1].plot(rr_, (a + b * rr_ ** 2) * 1e3, "-", label=f"fit {a * 1e3:.1f} + {b * 1e3:.1f} r² mm")
    ax[1].plot(rr_, INFL * (a + b * rr_ ** 2) * 1e3, "--", label=f"EKF R (x{INFL})")
    ax[1].set_xlabel("range normalised to 0.16 m marker (m)")
    ax[1].set_ylabel("σ_r (mm)")
    ax[1].axvline(R_MAX, color="grey", ls=":", label="EKF rejects beyond")
    ax[1].set_title("Range noise model")
    ax[1].legend(fontsize=8)
    ax[2].scatter(rt, np.degrees(eb), s=8, alpha=0.5)
    ax[2].axhline(np.degrees(sigma_b * INFL), color="r", ls="--", label="±σ_b used")
    ax[2].axhline(-np.degrees(sigma_b * INFL), color="r", ls="--")
    ax[2].set_xlabel("true range (m)")
    ax[2].set_ylabel("bearing error (deg)")
    ax[2].set_title("Bearing error")
    ax[2].legend(fontsize=8)
    ax[3].plot([float(k) for k in envelope], list(envelope.values()), "o-", label="vs range (all views)")
    ax3b = ax[3].twiny()
    ax3b.plot([float(k) for k in view_env], list(view_env.values()), "s--", color="#c0504d", label="vs view angle")
    ax3b.set_xlabel("viewing angle (deg)", color="#c0504d")
    ax[3].set_xlabel("range (m)")
    ax[3].set_ylabel("detection rate")
    ax[3].set_title("Detectability", pad=28)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "marker_characterisation.png"), dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "binned"}, indent=1))
    print(json.dumps(res["binned"], indent=1))


if __name__ == "__main__":
    main()
