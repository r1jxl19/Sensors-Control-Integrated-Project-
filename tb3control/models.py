"""Models used to design and analyse the controller.

All angles in radians, lengths in metres, time in seconds.

Pose and inputs
    pose x = [x, y, theta] of the base frame {B} in the map {M}
    input u = [v, omega]: forward speed and yaw rate of {B}

Wheel kinematics (differential drive, rolling without slip)
    v     = (r_r w_r + r_l w_l) / 2
    omega = (r_r w_r - r_l w_l) / b

Tracking error (the controller's state)
    The error of the ACTUAL pose relative to the REFERENCE pose, expressed in
    the reference frame:
        e = [ R(theta_r)^T (p - p_r) ;  wrap(theta - theta_r) ]
          = [ along-track, cross-track (lateral), heading ]
    Exact error dynamics with v, omega the actual inputs:
        e_x' =  v cos(e_th) - v_r + omega_r e_y
        e_y' =  v sin(e_th)       - omega_r e_x
        e_th' = omega - omega_r
    Linearised about e = 0, du = u - u_r = 0:
        e' = A(v_r, omega_r) e + B du
        A = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],  B = [[1, 0], [0, 0], [0, 1]]

Integral-augmented model (LQI extension)
    adds z' = e_y (integral of cross-track error) so a constant disturbance
    such as a left/right motor-gain mismatch gives zero steady-state error.

Actuator-augmented model (validation only)
    adds first-order lags on v and omega, time constant tau, so the design
    can be checked against the identified actuator.
"""
import numpy as np
from scipy.linalg import expm


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------- kinematics
def wheels_to_body(wl, wr, r_l, r_r, b):
    return 0.5 * (r_r * wr + r_l * wl), (r_r * wr - r_l * wl) / b


def body_to_wheels(v, w, r_l, r_r, b):
    return (v - 0.5 * b * w) / r_l, (v + 0.5 * b * w) / r_r


def unicycle(x, u):
    """Continuous-time unicycle x' = f(x, u)."""
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


# ------------------------------------------------------------ error model
def tracking_error(pose, ref_pose):
    """e = [along, lateral, heading] of pose relative to ref_pose, in the ref frame."""
    dx, dy = pose[0] - ref_pose[0], pose[1] - ref_pose[1]
    c, s = np.cos(ref_pose[2]), np.sin(ref_pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap(pose[2] - ref_pose[2]))])


def error_dynamics(e, u, u_ref):
    """Exact (nonlinear) error dynamics, used by tests and analysis."""
    v, w = u
    v_r, w_r = u_ref
    return np.array([v * np.cos(e[2]) - v_r + w_r * e[1],
                     v * np.sin(e[2]) - w_r * e[0],
                     w - w_r])


def error_model(v_r, w_r):
    """Linearised tracking-error model (A, B) about (v_r, w_r)."""
    A = np.array([[0.0, w_r, 0.0], [-w_r, 0.0, v_r], [0.0, 0.0, 0.0]])
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return A, B


def integral_model(v_r, w_r):
    """Error model augmented with z' = e_y. State [e_x, e_y, e_th, z]."""
    A3, B3 = error_model(v_r, w_r)
    A = np.zeros((4, 4))
    A[:3, :3] = A3
    A[3, 1] = 1.0
    B = np.vstack([B3, np.zeros((1, 2))])
    return A, B


def actuator_model(tau, v_r, w_r):
    """Error model + first-order lag on (v, omega). State [e(3), dv, dw]."""
    A3, B3 = error_model(v_r, w_r)
    A = np.zeros((5, 5))
    A[:3, :3] = A3
    A[:3, 3:] = B3
    A[3:, 3:] = -np.eye(2) / tau
    B = np.zeros((5, 2))
    B[3:, :] = np.eye(2) / tau
    return A, B


def c2d(A, B, dt):
    """Exact zero-order-hold discretisation: the controller holds u for dt."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def ctrb_rank(A, B, tol=1e-9):
    return int(np.linalg.matrix_rank(ctrb(A, B), tol=tol))
