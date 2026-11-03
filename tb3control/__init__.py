"""tb3control - trajectory-tracking control for the TurtleBot3 (Track A, Part D).

Self-contained: depends only on numpy and scipy, and acts only on an
estimated pose [x, y, theta]. Plugs into the full pipeline (tb3dock) or into
any estimator that outputs a pose.

    from tb3control import LQRController, TrajectoryTracker, Trajectory
    tracker = TrajectoryTracker(LQRController(), geom)        # geom = calibrated wheel geometry
    tracker.start(Trajectory(path_points))
    step = tracker.update(pose_estimate)                      # every 50 ms
    send_wheel_rates(*step.wheels)

Modules
    models       kinematics, tracking-error model, ZOH discretisation, controllability
    lqr          Bryson weights, discrete Riccati (DARE) LQR, gain scheduling
    controllers  LQR (design used), LQI (integral extension), Kanayama (baseline), heading
    reference    path -> time-parameterised trajectory under speed/accel/yaw-rate limits
    limits       curvature-preserving saturation
    tracker      reference clock (waits if the robot falls behind), align -> track
    analysis     closed-loop poles, actuator-lag and delay robustness checks
    plant        standalone test plant (actuator lag, saturation, noisy estimate, pushes)
"""
from .models import tracking_error, error_model, c2d, ctrb_rank, wrap
from .lqr import bryson, dlqr, lqr_gain, closed_loop_poles, V_FLOOR, DT
from .controllers import LQRController, LQIController, KanayamaController, heading_control, CONTROLLERS
from .reference import Trajectory, round_corners, resample, straight_approach
from .limits import saturate
from .tracker import TrajectoryTracker

__all__ = ["tracking_error", "error_model", "c2d", "ctrb_rank", "wrap", "bryson", "dlqr", "lqr_gain",
           "closed_loop_poles", "V_FLOOR", "DT", "LQRController", "LQIController", "KanayamaController",
           "heading_control", "CONTROLLERS", "Trajectory", "round_corners", "resample", "straight_approach",
           "saturate", "TrajectoryTracker"]
