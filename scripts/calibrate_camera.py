"""Camera intrinsic calibration from simulated checkerboard views.

A 9x6-inner-corner board (30 mm squares) is shown to the robot camera in
40 random poses (0.35-0.9 m, +/-35 deg tilt), mimicking a person waving a
board in front of the real camera. OpenCV's calibrateCamera estimates K and
radial distortion (k1, k2; tangential and k3 fixed at zero - with only ~40
views the extra terms trade off against cx/cy and were found to bias them); the result is compared with the intrinsics implied by the
simulator's field of view (validation only - the pipeline uses the
calibrated values).

Outputs: config/calibration.json (K, dist), results/camera_calibration.{json,png}
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
from tb3dock.rig import teleport, place_board
from tb3dock.scene import CHECKER_SQUARE
from tb3dock.frames import true_intrinsics, R_B_C, t_B_C
from tb3dock.config import save_calibration, RESULTS_DIR, CAMERA


def rot(axis, a):
    return cv2.Rodrigues(np.asarray(axis, float) * a)[0]


def main(n_views=40, seed=1):
    rng = np.random.default_rng(seed)
    sim = SimRobot(dict(start=np.array([1.0, 2.0, 0.0]), obstacles=[], calib_board=True), seed=seed)
    teleport(sim, (1.0, 2.0, 0.0))
    pose = GroundTruthProbe(sim).pose()       # rig: camera location for placing the board
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R_M_B = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cam_M = np.array([pose[0], pose[1], 0]) + R_M_B @ t_B_C
    R_M_C = R_M_B @ R_B_C

    pattern = (9, 6)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * CHECKER_SQUARE
    obj_pts, img_pts, used = [], [], []
    tries = 0
    while len(img_pts) < n_views and tries < 200:
        tries += 1
        z = rng.uniform(0.35, 0.9)
        xy = rng.uniform(-0.35, 0.35, 2) * z * np.array([1.0, 0.75])
        p_C = np.array([xy[0], xy[1], z])
        # board faces the camera (normal = -z_C), then random tilt
        R_C_board = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # x right, y up, n toward camera
        R_C_board = rot([1, 0, 0], rng.uniform(-0.6, 0.6)) @ rot([0, 1, 0], rng.uniform(-0.6, 0.6)) \
            @ rot([0, 0, 1], rng.uniform(-0.3, 0.3)) @ R_C_board
        place_board(sim, cam_M + R_M_C @ p_C, R_M_C @ R_C_board)
        img = sim.read_camera()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ok, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(img)

    size = (CAMERA.width, CAMERA.height)
    rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj_pts, img_pts, size, None, None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    std_int = std_int.ravel()
    Kt = true_intrinsics()
    res = dict(
        n_views=len(img_pts), attempts=tries, rms_px=float(rms),
        K=K.tolist(), dist=dist.ravel().tolist(),
        K_std=dict(fx=float(std_int[0]), fy=float(std_int[1]), cx=float(std_int[2]), cy=float(std_int[3])),
        K_true=Kt.tolist(),
        rel_err_f=float((K[0, 0] - Kt[0, 0]) / Kt[0, 0]),
        err_cx_px=float(K[0, 2] - Kt[0, 2]), err_cy_px=float(K[1, 2] - Kt[1, 2]),
        per_view_rms=per_view.ravel().tolist(),
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "camera_calibration.json"), "w") as f:
        json.dump(res, f, indent=2)
    save_calibration(dict(K=K.tolist(), dist=dist.ravel().tolist()))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vis = used[0].copy()
    cv2.drawChessboardCorners(vis, pattern, img_pts[0], True)
    ax[0].imshow(vis)
    ax[0].set_title("Example calibration view (detected corners)")
    ax[0].axis("off")
    ax[1].bar(range(len(per_view)), per_view.ravel(), color="#3b6ea5")
    ax[1].axhline(rms, color="k", ls="--", lw=1, label=f"overall RMS {rms:.3f} px")
    ax[1].set_xlabel("view")
    ax[1].set_ylabel("reprojection RMS (px)")
    ax[1].legend()
    ax[1].set_title("Per-view reprojection error")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "camera_calibration.png"), dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "per_view_rms"}, indent=1))


if __name__ == "__main__":
    main()
