"""Simulator wrapper.

``SimRobot`` is the *only* interface the autonomy pipeline uses. It exposes
what a real TurtleBot3 exposes - wheel velocity commands in, raw encoder
ticks, LiDAR ranges, camera images and gyro rate out - and nothing else.

Ground truth lives behind ``GroundTruthProbe``, which is used exclusively by
evaluation/logging code (``tb3dock/evaluation.py`` and ``scripts/``). The
mission code never constructs one; ``grep -n GroundTruthProbe tb3dock`` shows
this.
"""
import os
import sys
# Headless Linux (no display, e.g. a server or this repo's CI) -> software
# rendering via OSMesa. On macOS/Windows, or Linux with a display, MuJoCo's
# default (GLFW) is used so the live viewer and camera rendering both work.
if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ.get("MUJOCO_GL") in ("osmesa", "egl"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

from dataclasses import dataclass, field
import numpy as np
import mujoco

from .config import RATES, CAMERA, LIDAR, NOMINAL
from .scene import build_xml, G_ENV


@dataclass
class Disturbances:
    """Perturbations injected by the simulator (unknown to the pipeline)."""
    pushes: list = field(default_factory=list)       # [(t_start, duration, fx, fy)] world N
    camera_dropout: list = field(default_factory=list)  # [(t0, t1)] camera returns blank frames
    noise_scale: float = 1.0          # multiplies LiDAR, camera, gyro noise
    motor_gain: tuple = (1.0, 1.0)    # actual/commanded wheel rate (battery sag, gearing)
    wheel_friction: float = 1.2       # tyre friction coefficient
    gyro_bias: float = 0.0            # rad/s


class SimRobot:
    def __init__(self, scenario=None, seed=0, dist: Disturbances = None,
                 render_camera=True):
        self.rng = np.random.default_rng(seed)
        self.dist = dist or Disturbances()
        self.model = mujoco.MjModel.from_xml_string(build_xml(scenario))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(RATES.control_dt / m.opt.timestep))
        self._base = m.body("base").id
        self._lidar_site = m.site("lidar_site").id
        self._enc = [m.sensor("enc_l").adr[0], m.sensor("enc_r").adr[0]]
        self._gyro = m.sensor("gyro").adr[0]
        for name in ("wheel_l", "wheel_r"):
            gid = m.body_geomadr[m.body(name).id]
            m.geom_friction[gid, 0] = self.dist.wheel_friction
        self._renderer = mujoco.Renderer(m, CAMERA.height, CAMERA.width) if render_camera else None
        ang = np.arange(LIDAR.n_beams) * 2 * np.pi / LIDAR.n_beams
        self.lidar_angles = ang
        self._ray_local = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], 1)
        self._geomgroup = np.zeros(6, np.uint8)
        self._geomgroup[G_ENV] = 1
        mujoco.mj_forward(m, self.data)
        # let the robot settle onto its wheels
        for _ in range(100):
            mujoco.mj_step(m, self.data)
        self.data.time = 0.0
        self._enc_offset = np.array([self.data.sensordata[i] for i in self._enc])

    # ------------------------------------------------------------------ act
    @property
    def time(self):
        return self.data.time

    def step(self, wl_cmd, wr_cmd, dt=None):
        """Hold wheel-rate commands (rad/s) for one control period (or dt)."""
        d, m = self.data, self.model
        n_sub = self.n_sub if dt is None else max(1, int(round(dt / m.opt.timestep)))
        lim = m.actuator_ctrlrange[0, 1]
        g = self.dist.motor_gain
        d.ctrl[0] = np.clip(wl_cmd, -lim, lim) * g[0]
        d.ctrl[1] = np.clip(wr_cmd, -lim, lim) * g[1]
        for _ in range(n_sub):
            d.xfrc_applied[self._base, :] = 0
            for (t0, dur, fx, fy) in self.dist.pushes:
                if t0 <= d.time < t0 + dur:
                    d.xfrc_applied[self._base, 0] += fx
                    d.xfrc_applied[self._base, 1] += fy
            mujoco.mj_step(m, d)

    # -------------------------------------------------------------- sensors
    def read_encoders(self):
        """Raw cumulative encoder ticks (int) for left and right wheel."""
        q = np.array([self.data.sensordata[i] for i in self._enc]) - self._enc_offset
        return np.floor(q / (2 * np.pi) * NOMINAL.encoder_ticks).astype(np.int64)

    def read_gyro(self):
        """Yaw rate (rad/s) from the IMU, with bias and white noise."""
        wz = self.data.sensordata[self._gyro + 2]
        return wz + self.dist.gyro_bias + self.rng.normal(0, 0.005 * self.dist.noise_scale)

    def read_lidar(self):
        """360 ranges (m), NaN where no return (beyond range / too close)."""
        m, d = self.model, self.data
        R = d.site_xmat[self._lidar_site].reshape(3, 3)
        p = d.site_xpos[self._lidar_site].copy()
        vec = (self._ray_local @ R.T).reshape(-1)
        geomid = np.zeros(LIDAR.n_beams, np.int32)
        dist = np.zeros(LIDAR.n_beams)
        mujoco.mj_multiRay(m, d, p, vec, self._geomgroup, 1, -1, geomid, dist, None,
                           LIDAR.n_beams, LIDAR.r_max + 1.0)
        r = dist + self.rng.normal(0, LIDAR.sigma * self.dist.noise_scale, dist.shape)
        bad = (geomid < 0) | (dist < 0) | (r > LIDAR.r_max) | (r < LIDAR.r_min)
        # random dropouts (dark / specular surfaces): 1 %
        bad |= self.rng.random(r.shape) < 0.01
        r[bad] = np.nan
        return r

    def read_camera(self):
        """RGB uint8 image (H, W, 3) from the forward camera."""
        t = self.data.time
        for (t0, t1) in self.dist.camera_dropout:
            if t0 <= t < t1:
                return np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8)
        self._renderer.update_scene(self.data, "cam")
        img = self._renderer.render().astype(np.float32)
        img += self.rng.normal(0, CAMERA.pixel_noise * self.dist.noise_scale, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def render(self, camera="top", width=640, height=512):
        """Visualisation-only render (video), not a robot sensor."""
        key = (width, height)
        if not hasattr(self, "_vis") or self._vis[0] != key:
            self._vis = (key, mujoco.Renderer(self.model, height, width))
        r = self._vis[1]
        r.update_scene(self.data, camera)
        return r.render()


class GroundTruthProbe:
    """Reads simulator ground truth. EVALUATION AND LOGGING ONLY.

    Never pass this object (or anything it returns) into the estimator,
    planner or controller.
    """

    def __init__(self, sim: SimRobot):
        self._sim = sim

    def pose(self):
        d = self._sim.data
        x, y = d.qpos[0], d.qpos[1]
        qw, qx, qy, qz = d.qpos[3:7]
        th = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([x, y, th])

    def body_velocity(self):
        """(v forward, omega yaw) in base frame."""
        d = self._sim.data
        th = self.pose()[2]
        vx, vy = d.qvel[0], d.qvel[1]
        return np.array([np.cos(th) * vx + np.sin(th) * vy, d.qvel[5]])

    def wheel_rates(self):
        d, m = self._sim.data, self._sim.model
        return np.array([d.qvel[m.jnt_dofadr[m.joint("jl").id]],
                         d.qvel[m.jnt_dofadr[m.joint("jr").id]]])
