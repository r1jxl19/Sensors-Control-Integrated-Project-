"""Global (re)localisation from a stationary spin: robust batch least squares.

During one in-place revolution every accepted marker observation is stored
together with the odometry pose at that instant (odometry frame {O}, which
starts at the identity). The unknown is the transform T = (x0, y0, th0) of
{O} in the map; each observation gives a range/bearing residual

    r_k = [ |m_j - p_k| - z_r ,  wrap(atan2(m_j - p_k) - th_k - z_b) ] / sigma,
    (p_k, th_k) = T o odo_k

solved with a robust (soft-L1) loss from several starting guesses (one
single-marker PnP pose per observed marker). Two or more distinct markers make
the pose observable (scripts/analyse_model.py); with a single marker this
falls back to the PnP pose with a large covariance.

Why: the first version initialised the EKF from the first single-marker PnP
pose. In 1 of 16 nominal trials that pose was 24 cm off, every later
measurement failed the chi-square gate and the filter never recovered
(seed 102). The batch solve uses all markers in view and is robust to an
individual bad PnP.
"""
import numpy as np
from scipy.optimize import least_squares

from .frames import wrap
from .ekf import pose_from_marker
from .odometry import integrate


def compose(T, p):
    c, s = np.cos(T[2]), np.sin(T[2])
    return np.array([T[0] + c * p[0] - s * p[1], T[1] + s * p[0] + c * p[1], wrap(T[2] + p[2])])


def invert_compose(pose, odo):
    """T such that compose(T, odo) = pose."""
    th = wrap(pose[2] - odo[2])
    c, s = np.cos(th), np.sin(th)
    return np.array([pose[0] - (c * odo[0] - s * odo[1]), pose[1] - (s * odo[0] + c * odo[1]), th])


class SpinLocaliser:
    def __init__(self, ekf, marker_map):
        self.ekf = ekf
        self.map = marker_map
        self.reset()

    def reset(self):
        self.odo = np.zeros(3)
        self.spin = 0.0
        self.records = []        # (odo pose, id, z_r, z_b, sigma_r, sigma_b)
        self.guesses = []

    def add_motion(self, ds, dth):
        self.odo = integrate(self.odo, ds, dth)
        self.spin += abs(dth)

    def add_obs(self, o):
        m = self.map[o.id]
        size = m["size"]
        if o.view_angle > np.radians(60) or o.range * 0.16 / size > self.ekf.r_max:
            return
        r = self.ekf.correct_range(o.range, size)
        R = self.ekf.meas_cov(r, size)
        self.records.append((self.odo.copy(), o.id, r, o.bearing, np.sqrt(R[0, 0]), np.sqrt(R[1, 1])))
        if o.pixel_size > 20:
            ob_pB = o.p_B * np.array([1, 1, 0]) * r / o.range
            pose = pose_from_marker(_with(o, ob_pB), self.map)
            self.guesses.append(invert_compose(pose, self.odo))

    def n_markers(self):
        return len({r[1] for r in self.records})

    def solve(self):
        """Return (pose_now, P) or None."""
        if not self.records:
            return None
        if self.n_markers() < 2:
            if not self.guesses:
                return None
            T = self.guesses[-1]
            return compose(T, self.odo), np.diag([0.08 ** 2, 0.08 ** 2, 0.15 ** 2])

        def res(T):
            out = []
            for odo, mid, zr, zb, sr, sb in self.records:
                p = compose(T, odo)
                d = np.array(self.map[mid]["p"][:2]) - p[:2]
                out += [(np.hypot(*d) - zr) / sr, wrap(np.arctan2(d[1], d[0]) - p[2] - zb) / sb]
            return np.array(out)

        best = None
        for g in self.guesses[:: max(1, len(self.guesses) // 8)] or [np.zeros(3)]:
            sol = least_squares(res, g, loss="soft_l1", f_scale=3.0)
            if best is None or sol.cost < best.cost:
                best = sol
        T = best.x
        J = best.jac
        try:
            cov_T = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            return None
        # inflate (robust loss, correlated errors) and floor the covariance
        P = 4.0 * cov_T + np.diag([0.01 ** 2, 0.01 ** 2, 0.01 ** 2])
        inl = np.mean(np.abs(best.fun) < 3.0)
        self.last_inlier_ratio = float(inl)
        return compose(T, self.odo), P


class _with:
    """Lightweight copy of a MarkerObs with a replaced p_B."""

    def __init__(self, o, p_B):
        self.__dict__.update(o.__dict__)
        self.p_B = p_B
