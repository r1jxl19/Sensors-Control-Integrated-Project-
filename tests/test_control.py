"""Unit tests for the control package.   Run:  python -m pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from tb3control import (error_model, c2d, ctrb_rank, bryson, LQRController, LQIController, KanayamaController,
                        heading_control, saturate, Trajectory, TrajectoryTracker, tracking_error, resample)
from tb3control.models import error_dynamics, integral_model, body_to_wheels, wheels_to_body
from tb3control.lqr import lqr_gain, V_FLOOR
from tb3control.plant import CAL_GEOM, Condition, simulate, random_path
from tb3control import analysis


# ------------------------------------------------------------------ model
def test_linearisation_matches_nonlinear_error_dynamics():
    v_r, w_r = 0.15, 0.4
    A, B = error_model(v_r, w_r)
    e0, du0 = np.zeros(3), np.zeros(2)
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-6
        num = (error_dynamics(e0 + d, np.array([v_r, w_r]) + du0, (v_r, w_r)) -
               error_dynamics(e0 - d, np.array([v_r, w_r]) + du0, (v_r, w_r))) / 2e-6
        assert np.allclose(num, A[:, i], atol=1e-6)
    for j in range(2):
        d = np.zeros(2)
        d[j] = 1e-6
        num = (error_dynamics(e0, np.array([v_r, w_r]) + d, (v_r, w_r)) -
               error_dynamics(e0, np.array([v_r, w_r]) - d, (v_r, w_r))) / 2e-6
        assert np.allclose(num, B[:, j], atol=1e-6)


def test_controllability_rank():
    assert ctrb_rank(*error_model(0.0, 0.0)) == 2          # at rest: lateral error uncontrollable
    assert ctrb_rank(*error_model(0.15, 0.0)) == 3
    assert ctrb_rank(*error_model(0.0, 0.8)) == 3
    assert ctrb_rank(*integral_model(0.15, 0.0)) == 4


def test_wheel_kinematics_round_trip():
    g = CAL_GEOM
    wl, wr = body_to_wheels(0.12, 0.7, g["wheel_radius_l"], g["wheel_radius_r"], g["track"])
    v, w = wheels_to_body(wl, wr, g["wheel_radius_l"], g["wheel_radius_r"], g["track"])
    assert np.isclose(v, 0.12) and np.isclose(w, 0.7)


def test_tracking_error_frame():
    ref = np.array([1.0, 2.0, np.pi / 2])            # reference heading +y
    pose = np.array([0.9, 2.3, np.pi / 2 + 0.1])     # 30 cm ahead, 10 cm to the LEFT of the reference
    e = tracking_error(pose, ref)
    assert np.allclose(e, [0.3, 0.1, 0.1])


# -------------------------------------------------------------------- LQR
@pytest.mark.parametrize("v_r", [0.0, 0.03, 0.06, 0.12, 0.18, 0.22, -0.1])
@pytest.mark.parametrize("w_r", [-1.2, 0.0, 0.8])
def test_lqr_closed_loop_stable_over_envelope(v_r, w_r):
    Q, R = bryson()
    rho = analysis.rho_with_delay(v_r, w_r, Q, R, 0)
    assert rho < 1.0


def test_bryson_weights():
    Q, R = bryson()
    assert np.allclose(np.diag(Q), [400, 400, 1 / 0.15 ** 2])
    assert np.allclose(np.diag(R), [100, 1])


def test_gain_structure_and_speed_floor():
    Q, R = bryson()
    K = lqr_gain(0.18, 0.0, Q, R)
    assert np.allclose([K[0, 1], K[0, 2], K[1, 0]], 0, atol=1e-9)    # decoupled on a straight line
    assert K[0, 0] > 0 and K[1, 1] > 0 and K[1, 2] > 0
    assert np.allclose(lqr_gain(0.0, 0.0, Q, R), lqr_gain(V_FLOOR, 0.0, Q, R))


def test_lqr_correction_signs():
    c = LQRController()
    ref, uref = np.array([0.0, 0.0, 0.0]), np.array([0.15, 0.0])
    u, _ = c.control(np.array([0.0, 0.05, 0.0]), ref, uref)      # 5 cm left of the path
    assert u[1] < 0                                               # turn right (back towards it)
    u, _ = c.control(np.array([0.05, 0.0, 0.0]), ref, uref)      # 5 cm ahead
    assert u[0] < 0.15                                            # slow down


def test_actuator_lag_and_delay_robustness():
    Q, R = bryson()
    assert analysis.rho_with_actuator(0.18, 0.0, Q, R, tau=0.04) < 1
    assert analysis.max_stable_delay(0.18, 0.0, Q, R) >= 2         # at least 100 ms of pure delay


# --------------------------------------------------------------- saturation
def test_saturation_preserves_curvature():
    u = np.array([0.30, 3.0])
    us, (wl, wr) = saturate(u, CAL_GEOM)
    assert max(abs(wl), abs(wr)) <= 6.5 + 1e-9
    assert abs(us[0]) <= 0.22 + 1e-12
    # after the body clip (0.22, 2.84) curvature is preserved by the wheel scaling
    assert np.isclose(us[1] / us[0], 2.84 / 0.22)


def test_heading_control():
    u, e = heading_control(0.0, 0.5)
    assert u[0] == 0 and u[1] > 0 and np.isclose(e, 0.5)
    u, _ = heading_control(0.0, 3.0)
    assert np.isclose(u[1], 1.2)                                  # saturates


# ---------------------------------------------------------------- reference
def test_reference_respects_limits():
    rng = np.random.default_rng(0)
    p = random_path(rng)
    tr = Trajectory(p)
    assert tr.v.max() <= 0.18 + 1e-9
    assert np.all(np.abs(tr.v * tr.kappa) <= 1.2 + 1e-6)
    assert np.all(tr.v ** 2 * np.abs(tr.kappa) <= 0.25 + 1e-6)
    assert tr.v[0] == 0 and tr.v[-1] == 0
    ref, u = tr.sample(tr.duration)
    assert np.allclose(ref[:2], p[-1])


# ----------------------------------------------------------- closed loop
def _straight():
    return Trajectory(resample(np.array([[0.5, 2.0], [4.5, 2.0]]), 0.02))


@pytest.mark.parametrize("cls", [LQRController, LQIController, KanayamaController])
def test_recovers_from_offset(cls):
    tr = TrajectoryTracker(cls(), CAL_GEOM)
    L, m = simulate(_straight(), tr, Condition(est_sigma_xy=0, est_sigma_th=0), start_offset=(0, 0.10, 0.2))
    assert abs(L["e_true"][-1, 1]) < 0.01                         # ends within 1 cm laterally
    assert m["final_pos_err"] < 0.03


def test_lqi_removes_steady_state_error_from_motor_mismatch():
    cond = Condition(est_sigma_xy=0, est_sigma_th=0, motor_gain=(0.90, 1.08))
    tr = _straight()
    mid = lambda L: np.abs(L["e_true"][(L["t"] > 12) & (L["t"] < 18), 1]).mean()   # cruising section
    L1, _ = simulate(tr, TrajectoryTracker(LQRController(), CAL_GEOM), cond)
    L2, _ = simulate(tr, TrajectoryTracker(LQIController(), CAL_GEOM), cond)
    assert mid(L1) > 0.005            # LQR: steady cross-track offset of several mm
    assert mid(L2) < 0.25 * mid(L1)   # LQI removes most of it


def test_controller_never_needs_ground_truth():
    """The tracker's only inputs are the estimate and the reference."""
    import inspect
    from tb3control import tracker, controllers
    for mod in (tracker, controllers):
        src = inspect.getsource(mod)
        assert "GroundTruth" not in src and ".plant" not in src
