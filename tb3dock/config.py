"""Central configuration: platform parameters, sensor specs, world layout.

Two sets of platform parameters are kept deliberately separate:

* ``NOMINAL`` - the datasheet values the *software* knows about (TurtleBot3
  Burger spec sheet). The estimator and controller only ever see these, or
  the calibrated values produced by ``scripts/calibrate_odometry.py``.
* ``TRUE`` - the values the *simulator* is built with. They differ slightly
  from nominal (unequal wheel radii, wider track) to mimic manufacturing
  tolerance and tyre wear, so that odometry calibration has something real
  to find. Nothing in the estimation/control pipeline imports ``TRUE``.
"""
from dataclasses import dataclass, field
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(ROOT, "assets")
CONFIG_DIR = os.path.join(ROOT, "config")
RESULTS_DIR = os.path.join(ROOT, "results")


@dataclass
class Platform:
    wheel_radius_l: float = 0.033   # m
    wheel_radius_r: float = 0.033   # m
    track: float = 0.160            # m, wheel separation
    v_max: float = 0.22             # m/s   (TB3 Burger spec)
    w_max: float = 2.84             # rad/s (TB3 Burger spec)
    encoder_ticks: int = 4096       # ticks / wheel rev (Dynamixel XL430)


NOMINAL = Platform()
# Simulator ground-truth geometry (hidden from the pipeline).
TRUE = Platform(wheel_radius_l=0.0336, wheel_radius_r=0.0327, track=0.1635)


@dataclass
class Rates:
    physics_dt: float = 0.002   # s, MuJoCo timestep
    control_dt: float = 0.05    # s, 20 Hz odometry + EKF predict + control
    camera_dt: float = 0.10     # s, 10 Hz camera
    lidar_dt: float = 0.20      # s, 5 Hz LiDAR (LDS-01 spins at 5 Hz)


RATES = Rates()


@dataclass
class CameraSpec:
    width: int = 640
    height: int = 480
    fovy_deg: float = 48.8      # Raspberry Pi Camera v2 vertical FOV
    # camera optical centre in base frame {B} (x fwd, y left, z up), metres
    pos_b: tuple = (0.070, 0.0, 0.090)
    pixel_noise: float = 1.5    # std of additive image noise (grey levels)


CAMERA = CameraSpec()


@dataclass
class LidarSpec:
    n_beams: int = 360
    r_min: float = 0.12         # m (LDS-01)
    r_max: float = 3.5          # m (LDS-01)
    sigma: float = 0.010        # m, range noise std
    pos_b: tuple = (-0.032, 0.0, 0.140)   # sensor origin in {B}


LIDAR = LidarSpec()


@dataclass
class World:
    size_x: float = 5.0
    size_y: float = 4.0
    wall_h: float = 0.40
    # Dock: ArUco id 0 on a small block on the +x wall.
    dock_marker_id: int = 0
    dock_marker_size: float = 0.10       # black square edge, m
    dock_y: float = 2.0
    dock_block_depth: float = 0.05
    dock_marker_h: float = 0.100         # centre height, m (near camera height so it stays in view when docked)
    dock_standoff: float = 0.20          # camera-to-marker distance when docked
    landmark_size: float = 0.16
    landmark_h: float = 0.16
    aruco_dict: str = "DICT_4X4_50"
    # Landmark markers on the walls: (id, wall, position along wall)
    landmarks: tuple = (
        (1, "y0", 1.25), (2, "y0", 3.75),
        (3, "y4", 1.25), (4, "y4", 3.75),
        (5, "x0", 1.00), (6, "x0", 3.00),
        (7, "x5", 0.80), (8, "x5", 3.20),
    )


WORLD = World()


def dock_goal():
    """Docked base pose (x, y, theta) in map frame {M}."""
    marker_x = WORLD.size_x - WORLD.dock_block_depth
    x = marker_x - WORLD.dock_standoff - CAMERA.pos_b[0]
    return (x, WORLD.dock_y, 0.0)


def load_calibration():
    """Calibrated parameters written by the calibration scripts.

    Falls back to nominal values if calibration has not been run, so the
    pipeline still works (with worse odometry) from a fresh clone.
    """
    path = os.path.join(CONFIG_DIR, "calibration.json")
    cal = {
        "wheel_radius_l": NOMINAL.wheel_radius_l,
        "wheel_radius_r": NOMINAL.wheel_radius_r,
        "track": NOMINAL.track,
        "K": None, "dist": None,
        "odom_k": [0.01, 0.01],          # odometry noise: var per metre travelled
        "meas_sigma_r": [0.005, 0.01],   # range std = a + b*r
        "meas_sigma_b": 0.01,            # bearing std, rad
    }
    if os.path.exists(path):
        with open(path) as f:
            cal.update(json.load(f))
    return cal


def save_calibration(update: dict):
    path = os.path.join(CONFIG_DIR, "calibration.json")
    cal = {}
    if os.path.exists(path):
        with open(path) as f:
            cal = json.load(f)
    cal.update(update)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)
    return cal
