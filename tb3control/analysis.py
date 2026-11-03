"""Design checks for the LQR: poles, actuator lag, delay, robustness margins.

Everything here is linear analysis of the design model at an operating point
(v_r, w_r). The nonlinear, noisy behaviour is measured separately in the
standalone plant (control_scripts/evaluate_control.py) and the full pipeline.
"""
import numpy as np

from .models import error_model, actuator_model, c2d
from .lqr import dlqr, effective_speed, DT


def design(v_r, w_r, Q, R, dt=DT):
    A, B = error_model(effective_speed(v_r), w_r)
    Ad, Bd = c2d(A, B, dt)
    K, P = dlqr(Ad, Bd, Q, R)
    return Ad, Bd, K


def poles_s(v_r, w_r, Q, R, dt=DT):
    Ad, Bd, K = design(v_r, w_r, Q, R, dt)
    z = np.linalg.eigvals(Ad - Bd @ K)
    return np.log(z.astype(complex)) / dt


def rho_with_actuator(v_r, w_r, Q, R, tau, dt=DT):
    """Spectral radius of the closed loop when the plant has first-order actuator lag tau.

    The gain is the 3-state design; the actuator states are not fed back (the
    controller cannot measure them). < 1 means stable.
    """
    _, _, K = design(v_r, w_r, Q, R, dt)
    A, B = actuator_model(tau, effective_speed(v_r), w_r)
    Ad, Bd = c2d(A, B, dt)
    Kf = np.hstack([K, np.zeros((2, 2))])
    return float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ Kf))))


def rho_with_delay(v_r, w_r, Q, R, n_delay, dt=DT):
    """Spectral radius when commands arrive n_delay control periods late.

    Augmented state [e, u_{k-1}, ..., u_{k-n}]: e+ = Ad e + Bd u_{k-n}, u_k = -K e.
    """
    Ad, Bd, K = design(v_r, w_r, Q, R, dt)
    if n_delay == 0:
        return float(np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))))
    n, m = Bd.shape
    N = n + m * n_delay
    M = np.zeros((N, N))
    M[:n, :n] = Ad
    M[:n, n + m * (n_delay - 1):] = Bd               # oldest command drives the plant
    M[n:n + m, :n] = -K                               # newest command
    for i in range(1, n_delay):                       # shift register
        M[n + m * i:n + m * (i + 1), n + m * (i - 1):n + m * i] = np.eye(m)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def max_stable_delay(v_r, w_r, Q, R, dt=DT, n_max=40):
    """Largest whole number of control periods of delay that stays stable."""
    for n in range(n_max + 1):
        if rho_with_delay(v_r, w_r, Q, R, n, dt) >= 1.0:
            return n - 1
    return n_max


def gain_margin(v_r, w_r, Q, R, dt=DT, lo=0.01, hi=50.0):
    """Range of a common input gain g (plant sees g*u) that keeps the loop stable.

    Classic LQR result (continuous time): gain margin [1/2, inf). Discrete-time
    LQR keeps a finite but large upper margin; this computes it numerically.
    """
    Ad, Bd, K = design(v_r, w_r, Q, R, dt)
    gs = np.geomspace(lo, hi, 400)
    stable = [np.max(np.abs(np.linalg.eigvals(Ad - g * Bd @ K))) < 1 for g in gs]
    stable = np.array(stable)
    if not stable.any():
        return (np.nan, np.nan)
    idx = np.where(stable)[0]
    return float(gs[idx[0]]), float(gs[idx[-1]])
