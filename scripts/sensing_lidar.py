"""LiDAR characterisation (Part B): noise, bias, dropouts, frame offset, map accuracy.

1. Static characterisation. The robot is placed (test rig) at 60 surveyed
   poses in the empty room and takes 3 scans at each. True ranges come from the
   room geometry, computed analytically (walls at x = 0, 5 and y = 0, 4) from
   the LiDAR origin; beams that hit the dock block are excluded. This gives
   bias and noise against range and incidence angle, and the dropout rate.
2. Frame offset. The same residuals recomputed as if the LiDAR sat at the axle
   centre (t_B_L ignored), showing the systematic error that frame management
   removes.
3. Map-point error budget. Measured sigma_r plus the EKF's pose error (from the
   team's nominal evaluation) gives the 1-sigma and 3-sigma error of a mapped
   point against range, compared with the 5 cm grid and 20 cm inflation.
4. Map accuracy in the loop. Occupancy grids built during 6 full missions
   (on the EKF estimate) are compared with the true obstacle/wall geometry.

Outputs: results/sensing/lidar.json, results/sensing/lidar.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.sim import SimRobot
from tb3dock.rig import teleport
from tb3dock.config import WORLD, LIDAR, RESULTS_DIR
from tb3sensing.frames import t_B_L, wrap
from tb3sensing.lidar import point_sigma

OUT = os.path.join(RESULTS_DIR, "sensing")
BLUE, RED, GREEN, GREY = "#3b6ea5", "#c0504d", "#4f9a62", "#7f7f7f"


def wall_range(o, a):
    """Analytic range from origin o along world angle a to the room walls, + incidence angle."""
    X, Y = WORLD.size_x, WORLD.size_y
    c, s = np.cos(a), np.sin(a)
    best, inc, hit = np.inf, np.nan, None
    for t, n, w in (((X - o[0]) / c if c > 1e-9 else np.inf, np.array([-1.0, 0]), "x5"),
                    ((0 - o[0]) / c if c < -1e-9 else np.inf, np.array([1.0, 0]), "x0"),
                    ((Y - o[1]) / s if s > 1e-9 else np.inf, np.array([0, -1.0]), "y4"),
                    ((0 - o[1]) / s if s < -1e-9 else np.inf, np.array([0, 1.0]), "y0")):
        if 0 < t < best:
            best, hit = t, w
            inc = float(np.arccos(np.clip(-(c * n[0] + s * n[1]), -1, 1)))
    p = o + best * np.array([c, s])
    on_dock = hit == "x5" and abs(p[1] - WORLD.dock_y) < 0.14
    return best, inc, on_dock


def static_characterisation(rng, n_poses=60, n_scans=3):
    sim = SimRobot(dict(start=np.array([2.5, 2.0, 0.0]), obstacles=[]), seed=5, render_camera=False)
    rows = []
    for _ in range(n_poses):
        pose = np.array([rng.uniform(0.4, 4.4), rng.uniform(0.4, 3.6), rng.uniform(-np.pi, np.pi)])
        teleport(sim, pose)
        # surveyed LiDAR origin and heading (test rig)
        site = sim.model.site("lidar_site").id
        o = sim.data.site_xpos[site][:2].copy()
        Rm = sim.data.site_xmat[site].reshape(3, 3)
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
        base_o = o - np.array([np.cos(yaw), np.sin(yaw)]) * t_B_L[0]   # axle centre (for the frame-offset study)
        for _ in range(n_scans):
            r = sim.read_lidar()
            for i, a_l in enumerate(sim.lidar_angles):
                a = yaw + a_l
                rt, inc, dock = wall_range(o, a)
                if dock:
                    continue
                rb, _, _ = wall_range(base_o, a)          # range if measured from the axle centre
                rows.append((rt, inc, r[i], rb, a_l))
    return np.array(rows, float)


def map_accuracy(seeds=(100, 101, 102, 103, 104, 105)):
    """Distance from each occupied cell to the nearest true surface, for full missions."""
    from tb3dock.evaluation import run_trial
    d_all, per = [], []
    for s in seeds:
        r = run_trial(s, keep_log=True)
        g, sc = r["grid"], r["scenario"]
        cells = np.argwhere(g.occupied())
        P = np.array([g.cell_to_world(c) for c in cells])
        d = np.minimum.reduce([np.abs(P[:, 0]), np.abs(P[:, 0] - WORLD.size_x), np.abs(P[:, 1]),
                               np.abs(P[:, 1] - WORLD.size_y)])
        # dock block
        dx = np.maximum(np.abs(P[:, 0] - (WORLD.size_x - WORLD.dock_block_depth / 2)) - WORLD.dock_block_depth / 2, 0)
        dy = np.maximum(np.abs(P[:, 1] - WORLD.dock_y) - 0.12, 0)
        d = np.minimum(d, np.hypot(dx, dy))
        for o in sc["obstacles"]:
            q = P - o["c"]
            if o["kind"] == "box":
                c, s_ = np.cos(o["yaw"]), np.sin(o["yaw"])
                lx, ly = c * q[:, 0] + s_ * q[:, 1], -s_ * q[:, 0] + c * q[:, 1]
                ex = np.maximum(np.abs(lx) - o["half"][0], 0)
                ey = np.maximum(np.abs(ly) - o["half"][1], 0)
                inside = (ex == 0) & (ey == 0)
                dd = np.where(inside, 0.0, np.hypot(ex, ey))
            else:
                dd = np.maximum(np.hypot(q[:, 0], q[:, 1]) - o["half"][0], 0)
            d = np.minimum(d, dd)
        d_all.append(d)
        per.append(dict(seed=s, n_cells=int(len(d)), median_mm=float(1e3 * np.median(d)),
                        p95_mm=float(1e3 * np.quantile(d, 0.95)), frac_within_10cm=float(np.mean(d < 0.10)),
                        est_rmse_xy_mm=1e3 * r["est_rmse_xy"]))
    return np.concatenate(d_all), per


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(11)
    D = static_characterisation(rng)
    rt, inc, rm, rb, al = D.T
    in_range = rt <= LIDAR.r_max - 0.05
    ret = ~np.isnan(rm)
    ok = in_range & ret
    e = rm[ok] - rt[ok]
    e_axle = rm[ok] - rb[ok]
    bins = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    rc, sd, bias, drop = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (rt >= lo) & (rt < hi)
        mm = in_range & (rt >= lo) & (rt < hi)
        if m.sum() > 50:
            ee = rm[m] - rt[m]
            rc.append((lo + hi) / 2)
            sd.append(float(np.std(ee)))
            bias.append(float(np.mean(ee)))
            drop.append(float(1 - np.mean(ret[mm])))
    ib = np.radians([0, 20, 40, 60, 75, 90])
    ic, isd = [], []
    for lo, hi in zip(ib[:-1], ib[1:]):
        m = ok & (inc >= lo) & (inc < hi)
        if m.sum() > 50:
            ic.append(float(np.degrees((lo + hi) / 2)))
            isd.append(float(np.std(rm[m] - rt[m])))
    sigma_r = float(np.std(e))
    # map-point error budget, EKF pose error from the nominal evaluation
    sig_xy, sig_th = 0.0161 / np.sqrt(2), np.radians(0.848)
    rr = np.linspace(0.1, 3.5, 60)
    ps = point_sigma(rr, sigma_r, sig_xy, sig_th)
    dmap, per = map_accuracy()
    res = dict(
        n_beams=int(len(rt)), n_valid=int(ok.sum()),
        bias_mm=float(1e3 * np.mean(e)), sigma_r_mm=1e3 * sigma_r,
        sigma_by_range=dict(r=rc, sigma_mm=[1e3 * x for x in sd], bias_mm=[1e3 * x for x in bias], dropout=drop),
        sigma_by_incidence=dict(deg=ic, sigma_mm=[1e3 * x for x in isd]),
        dropout_in_range=float(1 - np.mean(ret[in_range])),
        no_return_beyond_range=float(1 - np.mean(ret[rt > LIDAR.r_max + 0.05])) if np.any(rt > LIDAR.r_max + 0.05) else None,
        axle_frame_error_mm=dict(mean=float(1e3 * np.mean(e_axle)), max_abs=float(1e3 * np.max(np.abs(e_axle))),
                                 rms=float(1e3 * np.sqrt(np.mean(e_axle ** 2)))),
        map_point_sigma_mm={"0.5m": float(1e3 * point_sigma(0.5, sigma_r, sig_xy, sig_th)),
                            "1.5m": float(1e3 * point_sigma(1.5, sigma_r, sig_xy, sig_th)),
                            "3.5m": float(1e3 * point_sigma(3.5, sigma_r, sig_xy, sig_th))},
        map_accuracy=dict(median_mm=float(1e3 * np.median(dmap)), p95_mm=float(1e3 * np.quantile(dmap, 0.95)),
                          frac_within_5cm=float(np.mean(dmap < 0.05)), frac_within_10cm=float(np.mean(dmap < 0.10)),
                          per_trial=per),
    )
    with open(os.path.join(OUT, "lidar.json"), "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    ax[0].hist(e * 1e3, bins=60, range=(-50, 50), color=BLUE, alpha=0.8, density=True)
    xs = np.linspace(-40, 40, 200)
    ax[0].plot(xs, np.exp(-0.5 * (xs / (sigma_r * 1e3)) ** 2) / (sigma_r * 1e3 * np.sqrt(2 * np.pi)), color=RED,
               label=f"Gaussian σ = {sigma_r * 1e3:.1f} mm")
    ax[0].set_xlabel("range error, measured − surveyed (mm)")
    ax[0].set_title(f"Range noise ({ok.sum():,} returns): bias {np.mean(e) * 1e3:+.2f} mm")
    ax[0].legend(fontsize=8)
    ax[1].plot(rc, np.array(sd) * 1e3, "o-", color=BLUE, label="σ vs range")
    ax[1].plot(rc, np.array(bias) * 1e3, "s--", color=GREEN, label="bias vs range")
    ax1b = ax[1].twiny()
    ax1b.plot(ic, np.array(isd) * 1e3, "^:", color=RED, label="σ vs incidence")
    ax1b.set_xlabel("incidence angle (deg)", color=RED)
    ax[1].set_xlabel("range (m)")
    ax[1].set_ylabel("mm")
    ax[1].set_ylim(-3, 15)
    ax[1].legend(fontsize=8, loc="center left")
    ax[1].set_title("σ flat vs range and incidence; no bias", pad=28)
    order = np.argsort(al[ok])
    ax[2].plot(np.degrees(al[ok][order]), e_axle[order] * 1e3, ",", color=RED, alpha=0.3)
    ax[2].plot(np.degrees(al[ok][order]), e[order] * 1e3, ",", color=BLUE, alpha=0.3)
    ax[2].plot(np.degrees(np.sort(al)), -1e3 * t_B_L[0] * np.cos(np.sort(al)), color=RED, lw=1.5,
               label="ignoring t_B_L: ≈ 32 cos(a) mm")
    ax[2].plot([], [], color=BLUE, label="with t_B_L (correct frame)")
    ax[2].set_xlabel("beam angle in {L} (deg)")
    ax[2].set_ylabel("range residual (mm)")
    ax[2].set_ylim(-80, 80)
    ax[2].set_title("Frame management: the 3.2 cm mounting offset")
    ax[2].legend(fontsize=8)
    ax[3].plot(rr, ps * 1e3, color=BLUE, label="1σ mapped-point error")
    ax[3].plot(rr, 3 * ps * 1e3, color=BLUE, ls="--", label="3σ")
    ax[3].axhline(200, color=RED, lw=1, label="planning inflation 200 mm")
    ax[3].axhline(50, color=GREY, lw=1, ls=":", label="grid cell 50 mm")
    ax[3].set_xlabel("range (m)")
    ax[3].set_ylabel("mm")
    ax[3].set_title("Mapped-point error vs planning margin")
    ax[3].text(0.2, 120, f"in-loop maps vs truth:\nmedian {1e3 * np.median(dmap):.0f} mm, p95 {1e3 * np.quantile(dmap, 0.95):.0f} mm\n(25 mm = half a grid cell)", fontsize=8, color=GREY)
    ax[3].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lidar.png"), dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "map_accuracy"}, indent=1))
    print(json.dumps(res["map_accuracy"], indent=1))


if __name__ == "__main__":
    main()
