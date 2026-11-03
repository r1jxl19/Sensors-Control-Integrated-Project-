"""2D LiDAR processing: validity, frames, and the map-point error budget.

The LDS-01 returns 360 ranges per revolution (1 deg steps, 5 Hz). Each beam i
points at angle a_i in the LiDAR frame {L}, whose axes are parallel to the
base frame {B} and whose origin is t_B_L = (-0.032, 0, 0.140) m in {B}.

    beam -> point in {L}:  r_i [cos a_i, sin a_i]
    {L}  -> {B}:           + t_B_L[:2]           (no rotation: axes parallel)
    {B}  -> {M}:           R(theta) p_B + [x, y]  (estimated pose)

Forgetting t_B_L (3.2 cm behind the axle) biases every point by up to 3.2 cm
depending on the beam direction (scripts/sensing_frames.py quantifies it).

Error budget of a mapped LiDAR point (1-sigma, independent terms):
    sigma_pt^2 ~= sigma_r^2 + sigma_xy^2 + (r * sigma_theta)^2
range noise + pose-estimate position error + heading error swung through the
range. The heading term dominates beyond about 1 m, which is why the map is
built on the EKF estimate at 5 cm resolution and planning inflates obstacles
by 20 cm (footprint 11.5 cm + margin).
"""
import numpy as np

from .frames import t_B_L

R_MIN, R_MAX = 0.12, 3.5      # m, LDS-01 valid range


def valid_mask(ranges, r_min=R_MIN, r_max=R_MAX):
    """True where a beam returned a usable range (NaN = no return)."""
    r = np.asarray(ranges, float)
    return ~np.isnan(r) & (r >= r_min) & (r <= r_max)


def scan_to_points(ranges, angles, t_BL=t_B_L):
    """Valid beams -> (N, 2) points in the base frame {B}."""
    ok = ~np.isnan(ranges)
    a = np.asarray(angles)
    return (np.stack([np.cos(a[ok]), np.sin(a[ok])], 1) * ranges[ok, None]) + t_BL[:2]


def points_to_map(pose, pts_B):
    """(N, 2) points in {B} -> map frame {M} using pose (x, y, theta)."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.column_stack([x + c * pts_B[:, 0] - s * pts_B[:, 1], y + s * pts_B[:, 0] + c * pts_B[:, 1]])


def point_sigma(r, sigma_r, sigma_xy, sigma_th):
    """1-sigma position error of a mapped point at range r (independent terms)."""
    r = np.asarray(r, float)
    return np.sqrt(sigma_r ** 2 + sigma_xy ** 2 + (r * sigma_th) ** 2)
