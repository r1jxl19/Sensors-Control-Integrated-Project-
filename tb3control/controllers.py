"""Feedback laws. All act on the ESTIMATED pose; none ever sees ground truth.

Common interface
    u, e = controller.control(pose_est, ref_pose, ref_u)
        pose_est  [x, y, theta]   estimate from the EKF
        ref_pose  [x_r, y_r, th_r] reference at this instant
        ref_u     [v_r, w_r]       feedforward from the reference
        u         [v, w]           body command before saturation
        e         [e_x, e_y, e_th] tracking error the controller saw

LQRController      the design used in the project: u = u_ref - K(v_r, w_r) e
LQIController      extension: adds integral action on cross-track error
KanayamaController baseline from the literature (Kanayama et al., 1990),
                   gains matched to the LQR at the design speed so the two
                   have the same small-error behaviour there
heading_control    in-place rotation (used where v_r = 0, see lqr.V_FLOOR)
"""
import numpy as np

from .models import tracking_error, wrap
from .lqr import bryson, lqr_gain, lqi_gain, GainSchedule, DT, V_FLOOR


class LQRController:
    """Gain-scheduled discrete LQR on the linearised tracking-error model."""

    name = "LQR"

    def __init__(self, Q=None, R=None, dt=DT):
        Qb, Rb = bryson()
        self.Q = Qb if Q is None else np.asarray(Q, float)
        self.R = Rb if R is None else np.asarray(R, float)
        self.dt = dt
        self.schedule = GainSchedule(lqr_gain, self.Q, self.R, dt)

    def gain(self, v_r, w_r):
        return self.schedule(v_r, w_r)

    def reset(self):
        pass

    def control(self, pose_est, ref_pose, ref_u, saturated=False):
        e = tracking_error(pose_est, ref_pose)
        K = self.gain(ref_u[0], ref_u[1])
        return np.asarray(ref_u, float) - K @ e, e


class LQIController:
    """LQR with integral action on cross-track error (LQI).

    Why: a constant disturbance on yaw rate - e.g. the left and right motors
    delivering 0.90x and 1.08x of the commanded wheel rate - is balanced in
    steady state only by a constant cross-track error with plain LQR:
    e_y = d / K_y (about 12 mm at 0.18 m/s for that mismatch). Integrating
    e_y lets the controller supply the balancing yaw rate with e_y = 0.

    Design: augmented state [e_x, e_y, e_th, z], z' = e_y, same Bryson weights
    plus 1/z_max^2 on z. Anti-windup: z is clamped to +/-z_lim, frozen while
    the command is saturated, and frozen while |v_r| < V_FLOOR (the reference
    is starting or stopping, cross-track error is not controllable there, and
    integrating it only winds up z), and only integrates once |e_y| < e_int
    (3 cm): the integrator is for small persistent offsets; large transients
    are left to the proportional part. reset() at the start of each trajectory.

    Tuning (control_scripts/tune_lqi.py): z_max = 0.20 m s. Augmenting the
    model also raises the proportional cross-track gain (K_y 16.7 -> 23.8 at
    0.18 m/s; with z_max = 0.05 it rises to 38.5 and transients get violent),
    so z_max trades steady-state rejection speed against transient aggression.
    """

    name = "LQI"

    def __init__(self, Q=None, R=None, z_max=0.20, z_lim=0.10, e_int=0.03, dt=DT):
        Qb, Rb = bryson()
        Q3 = Qb if Q is None else np.asarray(Q, float)
        self.R = Rb if R is None else np.asarray(R, float)
        self.Q = np.zeros((4, 4))
        self.Q[:3, :3] = Q3
        self.Q[3, 3] = 1.0 / z_max ** 2
        self.z_lim, self.e_int, self.dt = z_lim, e_int, dt
        self.schedule = GainSchedule(lqi_gain, self.Q, self.R, dt)
        self.z = 0.0

    def gain(self, v_r, w_r):
        return self.schedule(v_r, w_r)

    def reset(self):
        self.z = 0.0

    def control(self, pose_est, ref_pose, ref_u, saturated=False):
        e = tracking_error(pose_est, ref_pose)
        if not saturated and abs(ref_u[0]) >= V_FLOOR and abs(e[1]) < self.e_int:
            self.z = float(np.clip(self.z + e[1] * self.dt, -self.z_lim, self.z_lim))
        K = self.gain(ref_u[0], ref_u[1])
        return np.asarray(ref_u, float) - K @ np.r_[e, self.z], e


class KanayamaController:
    """Kanayama et al. (1990) nonlinear tracking law (baseline for comparison).

    Errors in the ROBOT frame, reference minus actual:
        [x_e, y_e] = R(theta)^T (p_r - p),  th_e = th_r - theta
        v = v_r cos(th_e) + k_x x_e
        w = w_r + v_r (k_y y_e + k_th sin(th_e))
    Gains are matched to the LQR at v_design so both have identical
    linearisations there (k_x = K_11, k_y = K_22 / v, k_th = K_23 / v).
    Unlike the LQR, its lateral and heading gains scale with v_r in fixed
    proportion; the LQR re-solves the Riccati equation at each speed.
    """

    name = "Kanayama"

    def __init__(self, v_design=0.18, Q=None, R=None, dt=DT):
        K = LQRController(Q, R, dt).gain(v_design, 0.0)
        self.kx = float(K[0, 0])
        self.ky = float(K[1, 1] / v_design)
        self.kth = float(K[1, 2] / v_design)

    def reset(self):
        pass

    def control(self, pose_est, ref_pose, ref_u, saturated=False):
        th = pose_est[2]
        c, s = np.cos(th), np.sin(th)
        dx, dy = ref_pose[0] - pose_est[0], ref_pose[1] - pose_est[1]
        xe, ye = c * dx + s * dy, -s * dx + c * dy
        the = float(wrap(ref_pose[2] - th))
        v_r, w_r = ref_u
        v = v_r * np.cos(the) + self.kx * xe
        w = w_r + v_r * (self.ky * ye + self.kth * np.sin(the))
        return np.array([v, w]), tracking_error(pose_est, ref_pose)


def heading_control(theta_est, theta_goal, kp=2.0, w_max=1.2):
    """Rotate in place towards theta_goal: w = sat(kp * error).

    kp = 2 s^-1 gives a 0.5 s time constant, well inside the actuator
    bandwidth; w_max = 1.2 rad/s keeps the camera image sharp enough to keep
    detecting markers while turning.
    """
    e = float(wrap(theta_goal - theta_est))
    return np.array([0.0, float(np.clip(kp * e, -w_max, w_max))]), e


CONTROLLERS = {"LQR": LQRController, "LQI": LQIController, "Kanayama": KanayamaController}
