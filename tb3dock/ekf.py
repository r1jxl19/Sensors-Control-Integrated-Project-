"""Extended Kalman filter: wheel odometry + ArUco range/bearing.

State      x = [x, y, theta]  (pose of {B} in {M})
Prediction from encoder arcs u = [ds_l, ds_r] (calibrated radii/track):
    ds = (ds_l + ds_r)/2,  dth = (ds_r - ds_l)/b
    x+ = x + ds cos(th + dth/2),  y+ = y + ds sin(th + dth/2),  th+ = th + dth
    P+ = F P F^T + G Qu G^T,   Qu = alpha * diag(k|ds_l| + q0, k|ds_r| + q0)
    (k from scripts/calibrate_odometry.py; alpha is the tuning factor studied
    in scripts/sweep_estimator.py; q0 keeps P growing slightly when parked.)
Update with each accepted marker j at known map position m_j:
    z = [r_meas - bias(r), bearing],   bias = (c1 rn + c2 rn^2) size/0.16, rn = r 0.16/size  h(x) = [|m_j - p|, atan2(m_j - p) - th]
    R = beta * diag(sigma_r(r)^2, sigma_b^2), sigma_r = (a + b rn^2) size/0.16 from scripts/characterise_markers.py
    Joseph-form covariance update for numerical robustness.
Gating: normalised range <= R_MAX, viewing angle <= 60 deg, and the
Mahalanobis distance of the innovation below chi2(2, 0.99) = 9.21.
"""
import numpy as np

from .frames import wrap
from .model import h_range_bearing, H_range_bearing
from tb3sensing.measurement import MarkerMeasurementModel

CHI2_2_99 = 9.21


class PoseEKF:
    def __init__(self, cal, alpha=1.0, beta=1.0, gate=CHI2_2_99, max_view_deg=60.0):
        self.b = cal["track"]
        self.k = float(np.max(cal["odom_k"]))
        self.q0 = 1e-9
        self.alpha, self.beta = alpha, beta
        # measurement model (bias correction, R, envelope) lives in the sensing package
        self.meas = MarkerMeasurementModel(cal, beta=beta, max_view_deg=max_view_deg)
        self.r_max = self.meas.r_max
        self.gate = gate
        self.max_view = self.meas.max_view
        self.x = None
        self.P = None
        self.stats = dict(accepted=0, gated=0, out_of_envelope=0)
        self.last_nis = []

    @property
    def initialised(self):
        return self.x is not None

    def init(self, x0, P0):
        self.x = np.array(x0, float)
        self.P = np.array(P0, float)

    # -------------------------------------------------------------- predict
    def predict(self, ds_l, ds_r):
        if self.x is None:
            return
        th = self.x[2]
        ds = 0.5 * (ds_l + ds_r)
        dth = (ds_r - ds_l) / self.b
        thm = th + 0.5 * dth
        c, s = np.cos(thm), np.sin(thm)
        self.x = np.array([self.x[0] + ds * c, self.x[1] + ds * s, wrap(th + dth)])
        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1.0]])
        # d(x,y,th)/d(ds_l, ds_r)
        G = np.array([[0.5 * c + ds * s / (2 * self.b), 0.5 * c - ds * s / (2 * self.b)],
                      [0.5 * s - ds * c / (2 * self.b), 0.5 * s + ds * c / (2 * self.b)],
                      [-1 / self.b, 1 / self.b]])
        Qu = self.alpha * np.diag([self.k * abs(ds_l) + self.q0, self.k * abs(ds_r) + self.q0])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T

    def predict_gyro(self, ds_l, ds_r, dth, var_dth):
        """Gyro-aided prediction: translation from wheels, rotation from the IMU.

        Rotation no longer depends on wheel slip / track error; its noise is
        the gyro white noise integrated over the step plus residual bias.
        """
        if self.x is None:
            return
        th = self.x[2]
        ds = 0.5 * (ds_l + ds_r)
        thm = th + 0.5 * dth
        c, s = np.cos(thm), np.sin(thm)
        self.x = np.array([self.x[0] + ds * c, self.x[1] + ds * s, wrap(th + dth)])
        F = np.array([[1, 0, -ds * s], [0, 1, ds * c], [0, 0, 1.0]])
        G = np.array([[0.5 * c, 0.5 * c, -0.5 * ds * s],
                      [0.5 * s, 0.5 * s, 0.5 * ds * c],
                      [0, 0, 1.0]])
        Qu = self.alpha * np.diag([self.k * abs(ds_l) + self.q0, self.k * abs(ds_r) + self.q0, var_dth])
        self.P = F @ self.P @ F.T + G @ Qu @ G.T

    # --------------------------------------------------------------- update
    def correct_range(self, r, size):
        return self.meas.correct_range(r, size)

    def meas_cov(self, r, size):
        return self.meas.covariance(r, size)

    def update_marker(self, obs, marker_map):
        """Fuse one MarkerObs. Returns (accepted, nis)."""
        if self.x is None:
            return False, None
        m = marker_map[obs.id]
        size = m["size"]
        if not self.meas.in_envelope(obs.range, size, obs.view_angle):
            self.stats["out_of_envelope"] += 1
            return False, None
        lm = m["p"][:2]
        z = np.array([self.correct_range(obs.range, size), obs.bearing])
        zhat = h_range_bearing(self.x, lm)
        y = z - zhat
        y[1] = wrap(y[1])
        H = H_range_bearing(self.x, lm)
        R = self.meas_cov(z[0], size)
        S = H @ self.P @ H.T + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > self.gate:
            self.stats["gated"] += 1
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.stats["accepted"] += 1
        self.last_nis.append(nis)
        return True, nis


def pose_from_marker(obs, marker_map):
    """Global pose from a single marker PnP (used once, for initialisation).

    The marker normal yaw is known in {M} and measured in {B}:
      theta = yaw_M(n) - yaw_B(n),  p = m - R(theta) p_B.
    The heading from a single small marker is the weakest part of this
    (normal estimation from 4 corners), hence the large initial heading
    variance and the subsequent localisation spin.
    """
    m = marker_map[obs.id]
    n = m["n"]
    th = wrap(np.arctan2(n[1], n[0]) - obs.yaw_B)
    c, s = np.cos(th), np.sin(th)
    pb = obs.p_B
    x = m["p"][0] - (c * pb[0] - s * pb[1])
    y = m["p"][1] - (s * pb[0] + c * pb[1])
    return np.array([x, y, th])
