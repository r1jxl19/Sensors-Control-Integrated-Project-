"""Frame management checks (Part B).

1. End-to-end frame-chain check. At surveyed robot poses (test rig), each
   visible marker's four corners are projected from the map {M} through
   {B} and {C} to pixels with the calibrated intrinsics:
       p_B = R(theta)^T (p_M - [x, y, 0]),  p_C = R_B_C^T (p_B - t_B_C),  u = K p_C / z
   and compared with the corners the ArUco detector actually finds. A wrong
   frame convention, sign, extrinsic or marker-map entry shows up as a large,
   systematic reprojection error; a correct chain leaves only detector noise
   (sub-pixel). Repeated with the TRUE intrinsics to separate calibration
   error from frame error.
2. Extrinsic sensitivity. How much range/bearing bias a camera-mounting error
   would cause (position +/-5 mm, yaw/pitch +/-1 deg), in units of the
   measurement noise R the EKF assumes. Shows which extrinsic must be known
   best (yaw: a 1 deg error is several sigma of bearing).

Outputs: results/sensing/frames.json, results/sensing/frames.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.sim import SimRobot, GroundTruthProbe
from tb3dock.rig import teleport
from tb3dock.scene import marker_map
from tb3dock.config import load_calibration, RESULTS_DIR, WORLD
from tb3sensing.frames import R_B_C, t_B_C, true_intrinsics
from tb3sensing.vision import MarkerDetector
from tb3sensing.measurement import MarkerMeasurementModel

OUT = os.path.join(RESULTS_DIR, "sensing")
BLUE, RED, GREEN, GREY = "#3b6ea5", "#c0504d", "#4f9a62", "#7f7f7f"


def corners_M(m):
    """Marker's 4 black-square corners in {M}, OpenCV order (TL, TR, BR, BL as seen from the front)."""
    p, n, s = np.array(m["p"]), np.array(m["n"]), m["size"] / 2
    right = np.cross(-n, [0, 0, 1.0])
    up = np.array([0, 0, 1.0])
    return np.array([p - right * s + up * s, p + right * s + up * s, p + right * s - up * s, p - right * s - up * s])


def project(pose, P_M, K):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R_M_B = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    P_B = (P_M - np.array([x, y, 0.0])) @ R_M_B          # = R_M_B^T (p - t) for each row
    P_C = (P_B - t_B_C) @ R_B_C                          # = R_B_C^T (p_B - t_B_C)
    uv = (K @ (P_C / P_C[:, 2:3]).T).T
    return uv[:, :2], P_C[:, 2]


def chain_check(rng, n=150):
    cal = load_calibration()
    K_cal, K_true = np.array(cal["K"]), true_intrinsics()
    mm = marker_map()
    sim = SimRobot(dict(start=np.array([2.5, 2.0, 0.0]), obstacles=[]), seed=4)
    gt = GroundTruthProbe(sim)
    det = MarkerDetector(K_cal, cal["dist"], mm)
    errs_cal, errs_true, ranges = [], [], []
    for _ in range(n):
        pose = np.array([rng.uniform(0.5, 4.4), rng.uniform(0.5, 3.5), rng.uniform(-np.pi, np.pi)])
        teleport(sim, pose)
        truth = gt.pose()
        for o in det.detect(sim.read_camera()):
            P = corners_M(mm[o.id])
            uv_c, z = project(truth, P, K_cal)
            uv_t, _ = project(truth, P, K_true)
            errs_cal.append(np.linalg.norm(uv_c - o.corners, axis=1).mean())
            errs_true.append(np.linalg.norm(uv_t - o.corners, axis=1).mean())
            ranges.append(o.range)
    return np.array(errs_cal), np.array(errs_true), np.array(ranges)


def sensitivity():
    cal = load_calibration()
    meas = MarkerMeasurementModel(cal)
    def rb(p):
        return np.hypot(p[0], p[1]), np.arctan2(p[1], p[0])
    def rot(axis, a):
        return cv2.Rodrigues(np.asarray(axis, float) * a)[0]
    perturb = {
        "camera x +5 mm": (np.eye(3), np.array([0.005, 0, 0])),
        "camera y +5 mm": (np.eye(3), np.array([0, 0.005, 0])),
        "camera yaw +1°": (rot([0, 0, 1], np.radians(1)), np.zeros(3)),
        "camera pitch +1°": (rot([0, 1, 0], np.radians(1)), np.zeros(3)),
    }
    rows = []
    for r in (0.3, 1.0, 2.5):
        for b in (0.0, 0.35):
            p_B = np.array([r * np.cos(b), r * np.sin(b), 0.12])      # marker centre roughly at camera height
            p_C = R_B_C.T @ (p_B - t_B_C)                             # what the camera actually sees
            r0, b0 = rb(p_B)
            for name, (dR, dt) in perturb.items():
                # the software believes the camera is mounted at (dR R_B_C, t_B_C + dt)
                p_B_wrong = dR @ R_B_C @ p_C + t_B_C + dt
                r1, b1 = rb(p_B_wrong)
                sr = meas.sigma_range(r0, WORLD.landmark_size)
                rows.append(dict(range=r, bearing_deg=float(np.degrees(b)), error=name,
                                 range_bias_mm=float(1e3 * (r1 - r0)), bearing_bias_deg=float(np.degrees(b1 - b0)),
                                 range_bias_sigma=float((r1 - r0) / sr), bearing_bias_sigma=float((b1 - b0) / meas.sb)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(21)
    ec, et, rr = chain_check(rng)
    sens = sensitivity()
    res = dict(n_markers=int(len(ec)),
               reproj_calibrated_K=dict(median_px=float(np.median(ec)), p95_px=float(np.quantile(ec, 0.95)),
                                        max_px=float(ec.max())),
               reproj_true_K=dict(median_px=float(np.median(et)), p95_px=float(np.quantile(et, 0.95))),
               sensitivity=sens)
    with open(os.path.join(OUT, "frames.json"), "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    ax[0].scatter(rr, ec, s=10, color=BLUE, alpha=0.6, label=f"calibrated K (median {np.median(ec):.2f} px)")
    ax[0].scatter(rr, et, s=10, color=GREY, alpha=0.5, label=f"true K (median {np.median(et):.2f} px)")
    ax[0].set_xlabel("marker range (m)")
    ax[0].set_ylabel("mean corner reprojection error (px)")
    ax[0].set_title("Frame chain {M}→{B}→{C}→pixels closes to ~1 px (no systematic error)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    names = list(dict.fromkeys(r["error"] for r in sens))
    x = np.arange(len(names))
    for j, (rg, col) in enumerate(((0.3, GREEN), (1.0, BLUE), (2.5, RED))):
        rbv = [abs(next(s["range_bias_sigma"] for s in sens if s["error"] == n and s["range"] == rg and s["bearing_deg"] == 0)) for n in names]
        bbv = [abs(next(s["bearing_bias_sigma"] for s in sens if s["error"] == n and s["range"] == rg and s["bearing_deg"] == 0)) for n in names]
        ax[1].bar(x + (j - 1) * 0.27, np.maximum(rbv, bbv), 0.27, color=col, label=f"marker at {rg} m")
    ax[1].axhline(1, color="k", ls="--", lw=1)
    ax[1].text(len(names) - 0.6, 1.15, "1σ of R", fontsize=8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("worst of |range bias| / σ_r, |bearing bias| / σ_b")
    ax[1].set_title("Mounting errors that matter: camera yaw, and lateral offset up close")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "frames.png"), dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, indent=1))
    for s in sens:
        if s["bearing_deg"] == 0:
            print(s)


if __name__ == "__main__":
    main()
