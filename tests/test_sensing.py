"""Unit tests for the sensing package (Part B).   Run:  python -m pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import pytest

from tb3sensing.frames import (R_B_C, t_B_C, t_B_L, camera_to_base, map_to_base, base_to_map, wrap,
                               true_intrinsics)
from tb3sensing.vision import MarkerDetector
from tb3sensing.measurement import MarkerMeasurementModel, S0
from tb3sensing.odometry import WheelOdometry, dead_reckon, integrate
from tb3sensing.lidar import scan_to_points, points_to_map, point_sigma, valid_mask


# ------------------------------------------------------------------- frames
def test_camera_axes_in_base_frame():
    # optical z (out of lens) is base +x, optical x (image right) is base -y, optical y (down) is base -z
    assert np.allclose(R_B_C @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(R_B_C @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(R_B_C @ [0, 1, 0], [0, 0, -1])
    assert np.isclose(np.linalg.det(R_B_C), 1.0)                 # proper rotation
    assert np.allclose(camera_to_base([0, 0, 1.0]), t_B_C + [1, 0, 0])


def test_map_base_round_trip():
    pose = np.array([1.2, 3.4, 2.5])
    p = np.array([0.3, -0.7])
    assert np.allclose(map_to_base(pose, base_to_map(pose, p)), p)


def test_wrap():
    assert np.isclose(wrap(3 * np.pi / 2), -np.pi / 2)
    assert -np.pi <= wrap(np.pi) < np.pi


# --------------------------------------------------- vision: synthetic camera
def _render_marker(rvec, tvec, K, marker_id=1, size=0.16, shape=(480, 640)):
    """Draw ArUco id onto a white image at pose (rvec, tvec) in the camera frame."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    px, pad = 240, 60
    img = cv2.aruco.generateImageMarker(d, marker_id, px)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    s = size / 2
    full = s * (px + 2 * pad) / px
    obj = np.array([[-full, full, 0], [full, full, 0], [full, -full, 0], [-full, -full, 0]], float)
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    n = img.shape[0]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), np.float32(uv.reshape(4, 2)))
    out = cv2.warpPerspective(img, H, (shape[1], shape[0]), borderValue=255)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


@pytest.mark.parametrize("dist_m,bearing_deg", [(0.5, 0), (1.0, 10), (2.0, -12)])
def test_detector_recovers_range_and_bearing(dist_m, bearing_deg):
    K = true_intrinsics()
    b = np.radians(bearing_deg)
    p_B = np.array([dist_m * np.cos(b), dist_m * np.sin(b), t_B_C[2]])     # marker centre at camera height
    p_C = R_B_C.T @ (p_B - t_B_C)
    # marker faces the camera: its z axis (out of the marker) points back along -optical z
    R_C_K = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    rvec, _ = cv2.Rodrigues(R_C_K)
    img = _render_marker(rvec, p_C, K)
    mm = {1: dict(p=[0, 0, 0], n=[1, 0, 0], size=0.16)}
    obs = MarkerDetector(K, None, mm).detect(img)
    assert len(obs) == 1
    o = obs[0]
    r_true = np.hypot(p_B[0], p_B[1])
    assert abs(o.range - r_true) < 0.01 * r_true + 0.003
    assert abs(o.bearing - np.arctan2(p_B[1], p_B[0])) < np.radians(0.3)
    assert o.reproj_err < 1.0


def test_detector_ignores_unknown_ids():
    K = true_intrinsics()
    img = _render_marker(cv2.Rodrigues(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]]))[0], np.array([0, 0, 1.0]), K,
                         marker_id=7)
    assert MarkerDetector(K, None, {1: dict(p=[0, 0, 0], n=[1, 0, 0], size=0.16)}).detect(img) == []


# ------------------------------------------------------- measurement model
CAL = dict(range_bias_c=[-0.0018, 0.0070], meas_sigma_r=[0.003, 0.00295], meas_sigma_b=0.00215,
           meas_max_range_norm=2.5)


def test_bias_correction_and_noise_scale_with_marker_size():
    m = MarkerMeasurementModel(CAL)
    # a 100 mm marker at 0.625 m looks like a 160 mm marker at 1.0 m: same normalised range
    r16, r10 = 1.0, 1.0 * 0.10 / S0
    b16 = r16 - m.correct_range(r16, 0.16)
    b10 = r10 - m.correct_range(r10, 0.10)
    assert np.isclose(b10, b16 * 0.10 / S0)
    assert np.isclose(m.sigma_range(r10, 0.10), m.sigma_range(r16, 0.16) * 0.10 / S0)
    assert b16 > 0                                     # PnP reads long: correction shortens the range


def test_envelope():
    m = MarkerMeasurementModel(CAL)
    assert m.in_envelope(2.4, 0.16, np.radians(30))
    assert not m.in_envelope(2.6, 0.16, np.radians(30))     # too far
    assert not m.in_envelope(1.0, 0.16, np.radians(65))     # too oblique
    assert not m.in_envelope(1.6, 0.10, 0.0)                 # 1.6 m on a 100 mm marker = 2.56 m normalised


def test_covariance_grows_with_range_and_beta():
    m, m3 = MarkerMeasurementModel(CAL), MarkerMeasurementModel(CAL, beta=3)
    assert m.covariance(2.0, 0.16)[0, 0] > m.covariance(0.5, 0.16)[0, 0]
    assert np.allclose(m3.covariance(1.0, 0.16), 3 * m.covariance(1.0, 0.16))


# ---------------------------------------------------------------- odometry
def test_odometry_straight_and_turn():
    r, b, N = 0.033, 0.16, 4096
    odo = WheelOdometry(r, r, b, N)
    odo.increment([0, 0])
    ds, dth, _, _ = odo.increment([N, N])                    # one full wheel turn each
    assert np.isclose(ds, 2 * np.pi * r) and np.isclose(dth, 0)
    ds, dth, _, _ = odo.increment([N, N + N // 4])           # right wheel a quarter turn further
    assert dth > 0                                           # turns left (CCW)


def test_dead_reckon_circle_closes():
    r, b, N = 0.033, 0.16, 4096
    # constant wheel ratio -> circle; one full revolution of heading returns to the start
    k = 2 * np.pi / N
    R = 0.5
    dl, dr = (R - b / 2), (R + b / 2)
    steps = 2000
    arc_l, arc_r = 2 * np.pi * dl / steps, 2 * np.pi * dr / steps
    ticks = np.cumsum(np.vstack([[0, 0], np.tile([arc_l / (r * k), arc_r / (r * k)], (steps, 1))]), axis=0)
    path = dead_reckon(ticks, [0, 0, 0], r, r, b, N)
    assert np.allclose(path[-1, :2], [0, 0], atol=2e-3)
    assert np.isclose(wrap(path[-1, 2]), 0, atol=1e-3)


def test_midpoint_integration():
    p = integrate(np.zeros(3), 1.0, np.pi / 2)
    assert np.allclose(p, [np.cos(np.pi / 4), np.sin(np.pi / 4), np.pi / 2])


# ------------------------------------------------------------------- LiDAR
def test_lidar_beam_to_base_frame_uses_mounting_offset():
    ranges = np.array([1.0, np.nan, 2.0])
    angles = np.array([0.0, 1.0, np.pi / 2])
    pts = scan_to_points(ranges, angles)
    assert pts.shape == (2, 2)                               # NaN beam dropped
    assert np.allclose(pts[0], [1.0 + t_B_L[0], t_B_L[1]])
    assert np.allclose(pts[1], [t_B_L[0], 2.0 + t_B_L[1]])
    m = points_to_map(np.array([1.0, 2.0, np.pi / 2]), pts)
    assert np.allclose(m[0], [1.0 - t_B_L[1], 2.0 + 1.0 + t_B_L[0]])


def test_lidar_validity_and_error_budget():
    assert valid_mask(np.array([0.05, 1.0, np.nan, 4.0])).tolist() == [False, True, False, False]
    s = point_sigma(np.array([0.5, 3.5]), 0.01, 0.011, np.radians(0.85))
    assert s[1] > s[0] and 3 * s[1] < 0.20                    # 3-sigma at max range inside the 20 cm inflation
