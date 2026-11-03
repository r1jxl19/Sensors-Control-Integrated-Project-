"""Frame conventions and transforms.

Frames (all right-handed):
  {M}  map: origin at room corner, x along 5 m wall, z up.
  {B}  base: origin at wheel-axle midpoint on the floor projection,
       x forward, y left, z up. Robot pose is (x, y, theta) of {B} in {M}.
  {C}  camera optical frame (OpenCV): z out of the lens, x right, y down.
       Mounted at CAMERA.pos_b in {B}, looking along +x_B.
  {L}  LiDAR: origin at LIDAR.pos_b in {B}, axes parallel to {B};
       beam i points at angle i * 1 deg CCW from +x_L.
  {K}  marker k: origin at black-square centre, z out of the wall
       (into the room), x to the viewer's right... as returned by OpenCV
       solvePnP for ArUco (x right, y up in the marker image, z out).

Chain used for a landmark measurement:
  pixels --(K, dist; solvePnP)--> T_C_K  --(T_B_C)--> p_B  --> (range, bearing)
and the EKF predicts the same quantity from the map: p_M --(T_M_B(x)^-1)--> p_B.
"""
import numpy as np

from tb3dock.config import CAMERA, LIDAR

# Rotation taking camera-optical vectors into base frame (columns = optical axes in {B}).
R_B_C = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
t_B_C = np.array(CAMERA.pos_b, float)
t_B_L = np.array(LIDAR.pos_b, float)


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def se2(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def camera_to_base(p_C):
    """Point in camera optical frame -> base frame."""
    return R_B_C @ np.asarray(p_C, float) + t_B_C


def base_to_map(pose, p_B):
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * p_B[0] - s * p_B[1], y + s * p_B[0] + c * p_B[1]])


def map_to_base(pose, p_M):
    x, y, th = pose
    dx, dy = p_M[0] - x, p_M[1] - y
    c, s = np.cos(th), np.sin(th)
    return np.array([c * dx + s * dy, -s * dx + c * dy])


def true_intrinsics():
    """Pinhole intrinsics implied by the simulator's camera (for validation only)."""
    f = 0.5 * CAMERA.height / np.tan(np.radians(CAMERA.fovy_deg) / 2)
    return np.array([[f, 0, (CAMERA.width - 1) / 2],
                     [0, f, (CAMERA.height - 1) / 2],
                     [0, 0, 1.0]])
