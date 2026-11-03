"""Reference generation: turn a path into a time-parameterised trajectory.

Input: a dense polyline (e.g. the planner's A* path after shortcutting and
corner rounding, resampled every 2 cm). Output: a reference that can be
sampled at any time t for pose (x_r, y_r, theta_r) and feedforward
(v_r, w_r = v_r * kappa).

Speed profile (the classic "minimum-time under limits" construction):
    1. curvature kappa(s) from the heading along the path, lightly smoothed;
    2. cap v by the speed limit, by lateral acceleration (v <= sqrt(a_lat/|kappa|))
       and by yaw rate (v <= w_max/|kappa|);
    3. forward pass: v_i <= sqrt(v_{i-1}^2 + 2 a_lon ds)  (acceleration limit);
    4. backward pass: v_i <= sqrt(v_{i+1}^2 + 2 a_lon ds) (braking limit);
    5. time from t = integral ds / v.

Why these limits: they keep the feedforward inside what the actuators can
deliver with headroom for feedback (the LQR needs margin to correct errors),
and keep wheel slip small (lateral and longitudinal acceleration).
"""
import numpy as np

from .models import wrap


class Trajectory:
    """Time-parameterised reference along a smooth path."""

    def __init__(self, pts, v_max=0.18, a_lat=0.25, w_max=1.2, a_lon=0.25, v_end=0.0, theta_end=None):
        pts = np.asarray(pts, float)
        self.pts = pts
        d = np.diff(pts, axis=0)
        seg = np.linalg.norm(d, axis=1)
        s = np.r_[0, np.cumsum(seg)]
        th = np.arctan2(d[:, 1], d[:, 0])
        th = np.r_[th, th[-1] if theta_end is None else theta_end]
        th = np.unwrap(th)
        kappa = np.gradient(th, s, edge_order=1) if len(s) > 2 else np.zeros_like(s)
        if len(kappa) >= 5:
            kappa = np.convolve(kappa, np.ones(5) / 5, mode="same")
        v = np.full_like(s, v_max)
        with np.errstate(divide="ignore"):
            v = np.minimum(v, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)))
            v = np.minimum(v, w_max / np.maximum(np.abs(kappa), 1e-6))
        v[0] = 0.0
        v[-1] = v_end
        for i in range(1, len(s)):                         # forward: acceleration limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_lon * seg[i - 1]))
        for i in range(len(s) - 2, -1, -1):                # backward: braking limit
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_lon * seg[i]))
        v[0] = max(v[0], 0.0)
        vm = np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
        t = np.r_[0, np.cumsum(seg / vm)]
        self.s, self.t, self.th, self.v, self.kappa = s, t, th, v, kappa
        self.duration = t[-1]
        self.length = s[-1]

    def sample(self, tq):
        """Reference pose and feedforward (v_r, w_r) at time tq (clamped to the ends)."""
        tq = np.clip(tq, 0, self.duration)
        x = np.interp(tq, self.t, self.pts[:, 0])
        y = np.interp(tq, self.t, self.pts[:, 1])
        th = np.interp(tq, self.t, self.th)
        v = np.interp(tq, self.t, self.v)
        k = np.interp(tq, self.t, self.kappa)
        return np.array([x, y, wrap(th)]), np.array([v, v * k])


def resample(pts, ds=0.02):
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.r_[True, seg > 1e-9]]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if s[-1] < 1e-6:
        return pts[:1]
    n = max(2, int(np.ceil(s[-1] / ds)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], 1)


def round_corners(pts, ds=0.02, radius=0.25):
    """Polyline -> smooth path: quadratic Bezier blend at each corner."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return resample(pts, ds)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        la, lc = np.linalg.norm(b - a), np.linalg.norm(c - b)
        r = min(radius, 0.45 * la, 0.45 * lc)
        p0 = b + (a - b) / la * r
        p2 = b + (c - b) / lc * r
        t = np.linspace(0, 1, 12)[:, None]
        out.extend((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(pts[-1])
    return resample(np.array(out), ds)


def straight_approach(start_xy, goal_xy, v=0.06, a_lon=0.1, theta_end=None, ds=0.01):
    """Slow straight-line reference for the final docking approach."""
    pts = resample(np.vstack([start_xy, goal_xy]), ds)
    return Trajectory(pts, v_max=v, a_lon=a_lon, theta_end=theta_end)
