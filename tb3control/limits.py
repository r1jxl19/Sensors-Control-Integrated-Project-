"""Actuator limits and saturation.

TurtleBot3 Burger limits: v <= 0.22 m/s, |w| <= 2.84 rad/s (datasheet);
wheel rate <= 6.5 rad/s here (motor limit with margin).

Saturation preserves CURVATURE (w / v): when a wheel would exceed its limit
both wheel rates are scaled by the same factor, so the robot slows down along
the arc the controller asked for instead of being pushed onto a different arc.
Clipping each wheel separately would change w/v and turn a lateral correction
into a heading error.
"""
from dataclasses import dataclass

import numpy as np

from .models import body_to_wheels


@dataclass
class Limits:
    v_max: float = 0.22
    w_max: float = 2.84
    wheel_max: float = 6.5


def saturate(u, geom, v_max=0.22, w_max=2.84, wheel_max=6.5):
    """Apply body and wheel limits.

    geom: mapping with wheel_radius_l, wheel_radius_r, track (the calibrated
    geometry, e.g. config/calibration.json).
    Returns (u_saturated [v, w], (wl, wr) wheel-rate commands in rad/s).
    """
    v, w = float(np.clip(u[0], -v_max, v_max)), float(np.clip(u[1], -w_max, w_max))
    wl, wr = body_to_wheels(v, w, geom["wheel_radius_l"], geom["wheel_radius_r"], geom["track"])
    m = max(abs(wl), abs(wr))
    if m > wheel_max:
        v, w, wl, wr = v * wheel_max / m, w * wheel_max / m, wl * wheel_max / m, wr * wheel_max / m
    return np.array([v, w]), (wl, wr)


def is_saturated(u_cmd, u_sat, tol=1e-9):
    return bool(np.any(np.abs(np.asarray(u_cmd) - np.asarray(u_sat)) > tol))
