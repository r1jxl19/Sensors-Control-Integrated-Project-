"""Wheel-encoder odometry for the differential drive.

Encoder ticks -> wheel arc lengths -> body displacement (ds, dtheta) using
(calibrated) wheel radii r_l, r_r and track b:

    ds_l = r_l * 2*pi*dn_l / N,   ds_r = r_r * 2*pi*dn_r / N
    ds   = (ds_r + ds_l) / 2,     dth  = (ds_r - ds_l) / b
"""
import numpy as np

from tb3dock.config import NOMINAL


class WheelOdometry:
    def __init__(self, r_l, r_r, track, ticks=NOMINAL.encoder_ticks):
        self.r_l, self.r_r, self.b, self.N = r_l, r_r, track, ticks
        self.prev = None

    def increment(self, ticks):
        """Return (ds, dth, ds_l, ds_r) since the previous call."""
        ticks = np.asarray(ticks, np.int64)
        if self.prev is None:
            self.prev = ticks
            return 0.0, 0.0, 0.0, 0.0
        dn = ticks - self.prev
        self.prev = ticks
        k = 2 * np.pi / self.N
        ds_l = self.r_l * k * dn[0]
        ds_r = self.r_r * k * dn[1]
        return 0.5 * (ds_l + ds_r), (ds_r - ds_l) / self.b, ds_l, ds_r


def integrate(pose, ds, dth):
    """Midpoint (second-order Runge-Kutta) pose update."""
    x, y, th = pose
    thm = th + 0.5 * dth
    return np.array([x + ds * np.cos(thm), y + ds * np.sin(thm), th + dth])


def dead_reckon(ticks_seq, pose0, r_l, r_r, b, N=NOMINAL.encoder_ticks):
    """Integrate a logged tick sequence (T, 2) from pose0. Used for calibration."""
    k = 2 * np.pi / N
    dn = np.diff(np.asarray(ticks_seq, float), axis=0)
    ds_l, ds_r = r_l * k * dn[:, 0], r_r * k * dn[:, 1]
    p = np.array(pose0, float)
    out = [p.copy()]
    for a, c in zip(ds_l, ds_r):
        p = integrate(p, 0.5 * (a + c), (c - a) / b)
        out.append(p)
    return np.array(out)
