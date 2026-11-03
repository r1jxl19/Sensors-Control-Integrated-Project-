"""LQR design: Bryson weights, discrete Riccati solution, gain scheduling.

Why discrete: the controller runs at 20 Hz and holds each command for 50 ms,
which is exactly a zero-order hold. Designing on the ZOH-discretised model
(models.c2d) means the closed loop the design predicts is the one that runs;
there is no Euler or Tustin approximation error to argue about.

Why scheduled: the error model depends on the reference speed v_r and yaw
rate w_r, and in particular the lateral error is coupled to the inputs only
through v_r (A[1, 2] = v_r). A gain designed at one speed would be too weak
at high speed or too aggressive at low speed, so K is recomputed from the
DARE at the current operating point and cached (rounded to 1 mm/s, 0.01 rad/s).

Why the speed floor: at v_r = w_r = 0 the linearisation has rank 2
(controllability matrix), i.e. lateral error cannot be corrected by any linear
feedback when the reference is stationary (Brockett's condition). The gain is
therefore computed at |v_r| >= V_FLOOR. Pure rotations use a separate heading
controller (controllers.heading_control).
"""
import numpy as np
from scipy.linalg import solve_discrete_are

from .models import error_model, integral_model, c2d

V_FLOOR = 0.05      # m/s, lowest speed the lateral gain is designed for
DT = 0.05           # s, control period (20 Hz)

# Bryson's rule: largest acceptable value of each state and input.
E_MAX = (0.05, 0.05, 0.15)     # along-track m, cross-track m, heading rad
U_MAX = (0.10, 1.0)            # speed correction m/s, yaw-rate correction rad/s


def bryson(e_max=E_MAX, u_max=U_MAX):
    """Q = diag(1/e_max^2), R = diag(1/u_max^2).

    Each weight is "how bad is one unit of this, relative to its tolerance":
    5 cm of cross-track error costs the same as 1 rad/s of extra yaw rate.
    """
    return np.diag(1.0 / np.square(e_max)), np.diag(1.0 / np.square(u_max))


def dlqr(Ad, Bd, Q, R):
    """Discrete infinite-horizon LQR: K minimises sum(e'Qe + u'Ru), u = -K e."""
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return K, P


def effective_speed(v_r):
    """Reference speed used for the gain: |v_r| floored at V_FLOOR, sign kept."""
    if v_r == 0:
        return V_FLOOR
    return float(np.sign(v_r) * max(abs(v_r), V_FLOOR))


def lqr_gain(v_r, w_r, Q, R, dt=DT):
    A, B = error_model(effective_speed(v_r), w_r)
    Ad, Bd = c2d(A, B, dt)
    return dlqr(Ad, Bd, Q, R)[0]


def lqi_gain(v_r, w_r, Q4, R, dt=DT):
    """Gain for the integral-augmented model, state [e_x, e_y, e_th, z]."""
    A, B = integral_model(effective_speed(v_r), w_r)
    Ad, Bd = c2d(A, B, dt)
    return dlqr(Ad, Bd, Q4, R)[0]


def closed_loop_poles(v_r, w_r, Q, R, dt=DT):
    """Continuous-equivalent closed-loop poles s = ln(z)/dt of the design model."""
    A, B = error_model(effective_speed(v_r), w_r)
    Ad, Bd = c2d(A, B, dt)
    K = dlqr(Ad, Bd, Q, R)[0]
    z = np.linalg.eigvals(Ad - Bd @ K)
    return np.log(z.astype(complex)) / dt, K


class GainSchedule:
    """Caches K(v_r, w_r) so the DARE is solved once per operating point."""

    def __init__(self, design_fn, Q, R, dt=DT):
        self.design_fn, self.Q, self.R, self.dt = design_fn, Q, R, dt
        self._cache = {}

    def __call__(self, v_r, w_r):
        v_eff = effective_speed(v_r)
        key = (round(v_eff, 3), round(float(w_r), 2))
        if key not in self._cache:
            # Designed at the first operating point seen in each bin (bins are
            # 1 mm/s x 0.01 rad/s, far finer than K changes). Kept this way so
            # the team's published results reproduce exactly.
            self._cache[key] = self.design_fn(v_eff, w_r, self.Q, self.R, self.dt)
        return self._cache[key]

    def __len__(self):
        return len(self._cache)
