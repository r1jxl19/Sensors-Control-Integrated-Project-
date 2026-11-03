"""State-space models of the TurtleBot3 platform and their analysis.

Model hierarchy (assumptions stated in docs/model.md):

1. Wheel kinematics (rolling without slip, rigid planar body):
       v = (r_r w_r + r_l w_l) / 2,   omega = (r_r w_r - r_l w_l) / b

2. Unicycle pose kinematics, state x = [x, y, theta], input u = [v, omega]:
       x_dot = v cos(theta),  y_dot = v sin(theta),  theta_dot = omega

3. Actuator dynamics (identified, scripts/identify_actuator.py): each wheel
   behaves as a first-order lag  tau * w_dot = K * w_cmd - w.

4. Tracking-error model, linearised about a reference (x_r, v_r, omega_r),
   with e = [R(theta_r)^T (p - p_r), theta - theta_r] and du = u - u_r:
       e_dot = A_e e + B_e du
       A_e = [[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0]],  B_e = [[1, 0], [0, 0], [0, 1]]
   This is the model the LQR tracker is designed on.
"""
import numpy as np
from scipy.linalg import solve_discrete_are, expm

from .frames import wrap


# ---------------------------------------------------------------- kinematics
def wheels_to_body(wl, wr, r_l, r_r, b):
    return 0.5 * (r_r * wr + r_l * wl), (r_r * wr - r_l * wl) / b


def body_to_wheels(v, w, r_l, r_r, b):
    return (v - 0.5 * b * w) / r_l, (v + 0.5 * b * w) / r_r


def f_cont(x, u):
    return np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), u[1]])


def jac_pose(x, u):
    """Continuous-time Jacobians of the unicycle about (x, u)."""
    th, v = x[2], u[0]
    A = np.array([[0, 0, -v * np.sin(th)], [0, 0, v * np.cos(th)], [0, 0, 0.0]])
    B = np.array([[np.cos(th), 0], [np.sin(th), 0], [0, 1.0]])
    return A, B


# ------------------------------------------------------------ error model
def error_model(v_r, w_r):
    A = np.array([[0, w_r, 0], [-w_r, 0, v_r], [0, 0, 0.0]])
    B = np.array([[1, 0], [0, 0], [0, 1.0]])
    return A, B


def c2d(A, B, dt):
    """Exact zero-order-hold discretisation."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def tracking_error(pose, ref_pose):
    """Error of the actual pose relative to the reference, in the reference frame."""
    dx, dy = pose[0] - ref_pose[0], pose[1] - ref_pose[1]
    c, s = np.cos(ref_pose[2]), np.sin(ref_pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(pose[2] - ref_pose[2])])


def dlqr(A, B, Q, R):
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# ------------------------------------------------- actuator-augmented model
def augmented_model(tau, v_r, w_r):
    """Error model with first-order actuator lag on (v, omega).

    State [e_x, e_y, e_th, dv, dw], input [dv_cmd, dw_cmd]. Used to check that
    the 3-state design remains valid given the identified actuator bandwidth.
    """
    A3, B3 = error_model(v_r, w_r)
    A = np.zeros((5, 5))
    A[:3, :3] = A3
    A[:3, 3:] = B3
    A[3:, 3:] = -np.eye(2) / tau
    B = np.zeros((5, 2))
    B[3:, :] = np.eye(2) / tau
    return A, B


# ----------------------------------------------- controllability/observability
def ctrb(A, B):
    n = A.shape[0]
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(n)])


def obsv(A, C):
    n = A.shape[0]
    return np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])


def h_range_bearing(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    return np.array([np.hypot(dx, dy), wrap(np.arctan2(dy, dx) - x[2])])


def H_range_bearing(x, lm):
    dx, dy = lm[0] - x[0], lm[1] - x[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


def empirical_obs_gramian(x0, controls, dt, landmarks, meas="rb", eps=1e-5):
    """Local observability Gramian W = sum_k S_k^T S_k, S_k = d y_k / d x0.

    Simulates the nonlinear unicycle from x0 under `controls` (list of (v, w)),
    measuring every landmark at every step. `meas` in {"rb", "range", "bearing",
    "pose", "none"}. Returns singular values of W (rank = number of observable
    directions) and the least-observable direction.
    """
    def outputs(x_init):
        x = np.array(x_init, float)
        ys = []
        for u in controls:
            for lm in landmarks:
                h = h_range_bearing(x, lm)
                if meas == "rb":
                    ys.append(h)
                elif meas == "range":
                    ys.append(h[:1])
                elif meas == "bearing":
                    ys.append(h[1:])
            if meas == "pose":
                ys.append(x.copy())
            x = x + dt * f_cont(x, u)
        return np.concatenate(ys) if ys else np.zeros(0)

    y0 = outputs(x0)
    if y0.size == 0:
        return np.zeros(3), np.eye(3)[:, 0]
    S = np.zeros((y0.size, 3))
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        dy = outputs(np.array(x0) + d) - outputs(np.array(x0) - d)
        if meas == "rb":
            dy[1::2] = wrap(dy[1::2])
        elif meas == "bearing":
            dy = wrap(dy)
        S[:, i] = dy / (2 * eps)
    W = S.T @ S
    U, sv, _ = np.linalg.svd(W)
    return sv, U[:, -1]
