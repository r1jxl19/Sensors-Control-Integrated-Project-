"""Full autonomy pipeline: localise -> map/plan -> track -> visually dock.

The mission talks to the robot only through SimRobot's sensor/actuator
methods. It never reads simulator ground truth: there is no GroundTruthProbe
or ``sim.data`` access in this file. Every control command is computed from
the EKF estimate.

Phases
  GYRO_BIAS  stand still 1 s and average the gyro (bias estimate).
  INIT       spin one revolution in place, collecting marker observations,
             then solve for the global pose by robust batch least squares
             (tb3dock/localisation.py) and initialise the EKF from it.
             Re-entered (re-localisation) if the EKF rejects every marker for
             15 consecutive camera frames - the signature of divergence.
  PLAN       A* on the LiDAR occupancy grid to the pre-dock pose.
  ALIGN      rotate in place onto the path heading.
  TRACK      LQR tracking of the time-parameterised path; replan if a LiDAR
             scan blocks the path ahead; emergency stop on close returns.
  FACE_DOCK  rotate to face the dock marker.
  DOCK       slow straight-line LQR approach to the docked pose; the dock
             marker dominates the EKF at close range.
  DONE/FAIL
"""
from dataclasses import dataclass, field
import numpy as np

from .config import RATES, WORLD, dock_goal, LIDAR
from .ekf import PoseEKF
from .localisation import SpinLocaliser
from .odometry import WheelOdometry
from .vision import MarkerDetector
from .mapping import OccupancyGrid
from .planner import plan_path, round_corners, Trajectory, resample
from .controller import LQRTracker, LQIController, heading_control, saturate
from tb3control.tracker import TrajectoryTracker
from .scene import marker_map
from .frames import wrap, t_B_L
from tb3sensing.lidar import scan_to_points

ROBOT_RADIUS = 0.115           # circumscribed footprint radius about {B}
PLAN_INFLATION = 0.20          # planning clearance (footprint + 8.5 cm margin)
CHECK_INFLATION = 0.16         # clearance that triggers a replan
PREDOCK_OFFSET = 0.55          # m in front of the docked pose


@dataclass
class MissionParams:
    ekf_alpha: float = 1.0       # process-noise scale
    ekf_beta: float = 1.0        # measurement-noise scale
    lqr_Q: np.ndarray = None
    lqr_R: np.ndarray = None
    v_nav: float = 0.18
    v_dock: float = 0.06
    use_camera: bool = True      # False = odometry-only baseline (after init)
    use_gyro: bool = False       # True = gyro-aided heading prediction (extension)
    gyro_sigma: float = 0.005    # rad/s white noise (IMU datasheet-level; checked at start-up)
    timeout: float = 180.0
    dock_tol_pos: float = 0.01   # estimated-position tolerance to declare docked
    controller: str = "LQR"      # "LQR" (design) or "LQI" (integral-action extension)
    dock_guard: bool = False     # v3: approach only while the dock marker is in view, else re-localise
    livelock_relocalise: bool = False  # v2 fix: >= 8 replans in 10 s -> re-localise (see docs/development_log.md #6)


class Mission:
    def __init__(self, sim, cal, params: MissionParams = None):
        self.sim = sim
        self.cal = cal
        self.p = params or MissionParams()
        self.map = marker_map()
        self.odo = WheelOdometry(cal["wheel_radius_l"], cal["wheel_radius_r"], cal["track"])
        self.ekf = PoseEKF(cal, alpha=self.p.ekf_alpha, beta=self.p.ekf_beta)
        self.det = MarkerDetector(cal["K"], cal["dist"], self.map)
        self.grid = OccupancyGrid()
        if self.p.controller == "LQI":
            self.tracker = LQIController(self.p.lqr_Q, self.p.lqr_R)
        else:
            self.tracker = LQRTracker(self.p.lqr_Q, self.p.lqr_R)
        # reference clock + feedback law from the control package (tb3control.tracker)
        self.follow = TrajectoryTracker(self.tracker, cal)
        self.goal = np.array(dock_goal())
        self.predock = self.goal - np.array([PREDOCK_OFFSET, 0, 0])
        self.phase = "GYRO_BIAS"     # 1 s stationary: estimate gyro bias
        self._gyro_samples = []
        self.gyro_bias = 0.0
        self.k = 0
        self.traj = None
        self.t_ref = 0.0
        self.loc = SpinLocaliser(self.ekf, self.map)
        self.gate_streak = 0
        self.n_relocalise = 0
        self._dock_seen_t = -1e9
        self._replan_times = []
        self.u = np.zeros(2)
        self.ref = None
        self.err = np.zeros(3)
        self.obs = []
        self.accepted_ids = []
        self.events = []
        self.n_replans = 0
        self.n_estops = 0
        self.plan_fail = 0
        self._phase_t0 = 0.0
        self._observer = None
        self.last_image = None

    # ------------------------------------------------------------ helpers
    def _set_phase(self, ph):
        self.events.append((round(self.sim.time, 2), ph))
        self.phase = ph
        self._phase_t0 = self.sim.time

    def _scan_points_B(self, ranges):
        return scan_to_points(ranges, self.sim.lidar_angles)       # tb3sensing.lidar

    def _plan(self, target, v_max, theta_end=None):
        pts = plan_path(self.grid, self.ekf.x[:2], target[:2], PLAN_INFLATION)
        if pts is None:
            return None
        pts = round_corners(pts)
        if len(pts) < 3:
            pts = resample(np.vstack([self.ekf.x[:2], target[:2]]), 0.02)
        return Trajectory(pts, v_max=v_max, theta_end=theta_end)

    def _path_blocked(self):
        if self.traj is None:
            return False
        blocked = self.grid.inflated(CHECK_INFLATION)
        i0 = np.searchsorted(self.traj.t, self.t_ref)
        s0 = self.traj.s[min(i0, len(self.traj.s) - 1)]
        ahead = (self.traj.s >= s0) & (self.traj.s <= s0 + 1.2)
        cells = self.grid.world_to_cell(self.traj.pts[ahead])
        cells = cells[self.grid.in_bounds(cells)]
        return bool(blocked[cells[:, 0], cells[:, 1]].any()) if len(cells) else False

    def _relocalise(self):
        self.n_relocalise += 1
        self.loc.reset()
        self.traj = None
        self._set_phase("INIT")

    # --------------------------------------------------------------- step
    def step(self):
        sim, dt = self.sim, RATES.control_dt
        cam_every = int(round(RATES.camera_dt / dt))
        lidar_every = int(round(RATES.lidar_dt / dt))

        # 1. odometry -> EKF prediction
        ds, dth, dsl, dsr = self.odo.increment(sim.read_encoders())
        g = sim.read_gyro()
        if self.phase == "GYRO_BIAS":
            self._gyro_samples.append(g)
        if self.p.use_gyro:
            sig = self.p.gyro_sigma
            self.ekf.predict_gyro(dsl, dsr, (g - self.gyro_bias) * dt,
                                  (sig * dt) ** 2 + (0.2 * sig * dt) ** 2)
        else:
            self.ekf.predict(dsl, dsr)
        if self.phase == "INIT":
            self.loc.add_motion(ds, dth)

        # 2. camera -> marker updates
        self.obs, self.accepted_ids = [], []
        if self.k % cam_every == 0:
            self.last_image = sim.read_camera()
            self.obs = self.det.detect(self.last_image)
            if any(o.id == WORLD.dock_marker_id for o in self.obs):
                self._dock_seen_t = sim.time
            if self.phase == "INIT":
                for o in self.obs:
                    self.loc.add_obs(o)
            elif self.p.use_camera:
                n_env, n_ok = 0, 0
                for o in self.obs:
                    before = self.ekf.stats["out_of_envelope"]
                    ok, _ = self.ekf.update_marker(o, self.map)
                    n_env += int(self.ekf.stats["out_of_envelope"] == before)
                    if ok:
                        n_ok += 1
                        self.accepted_ids.append(o.id)
                if n_env > 0:
                    self.gate_streak = 0 if n_ok else self.gate_streak + 1
                if self.gate_streak >= 15 and self.phase in ("ALIGN", "TRACK", "PLAN"):
                    self.gate_streak = 0
                    self.n_relocalise += 1
                    self.loc.reset()
                    self.traj = None
                    self._set_phase("INIT")

        # 3. LiDAR -> occupancy grid, safety
        estop = False
        if self.ekf.initialised and self.k % lidar_every == 0:
            r = sim.read_lidar()
            self.grid.update(self.ekf.x, r, sim.lidar_angles)
            pts = self._scan_points_B(r)
            front = (pts[:, 0] > 0.0) & (pts[:, 0] < 0.24) & (np.abs(pts[:, 1]) < 0.12)
            estop = bool(front.sum() >= 2) and self.phase == "TRACK"
            if self.phase == "TRACK" and (estop or self._path_blocked()):
                self.n_replans += 1
                self._replan_times.append(sim.time)
                recent = [t for t in self._replan_times if sim.time - t < 10.0]
                if self.p.livelock_relocalise and len(recent) >= 8:
                    # replanning every scan = the LiDAR map disagrees with the pose:
                    # the estimate has diverged even though no marker is in view
                    self._replan_times = []
                    self.n_relocalise += 1
                    self.loc.reset()
                    self.grid = OccupancyGrid()      # map was built on the wrong pose
                    self.traj = None
                    self._set_phase("INIT")
                self.n_estops += int(estop)
                if self.phase == "TRACK":
                    self.traj = None
                    self._set_phase("PLAN")

        # 4. phase logic -> command
        u = np.zeros(2)
        x = self.ekf.x
        if self.phase == "GYRO_BIAS":
            if sim.time >= 1.0:
                self.gyro_bias = float(np.mean(self._gyro_samples))
                self._set_phase("INIT")
        elif self.phase == "INIT":
            u = np.array([0.0, 0.6])
            if self.loc.spin >= 2 * np.pi + 0.3:
                sol = self.loc.solve()
                if sol is not None:
                    self.ekf.init(*sol)
                    self.gate_streak = 0
                    self._set_phase("PLAN")
                elif self.loc.spin > 4 * np.pi:
                    self._set_phase("FAIL")
        elif self.phase == "PLAN":
            self.traj = self._plan(self.predock, self.p.v_nav, theta_end=0.0)
            if self.traj is None:
                self.plan_fail += 1
                u = np.array([0.0, 0.4])       # turn to gather more LiDAR / views
                if self.plan_fail > 200:
                    self._set_phase("FAIL")
            else:
                self.t_ref = 0.0
                self.follow.start(self.traj, align_first=False)
                self._set_phase("ALIGN")
        elif self.phase == "ALIGN":
            ref, _ = self.traj.sample(0.05)
            u, e = heading_control(x[2], ref[2])
            if abs(e) < 0.08:
                self._set_phase("TRACK")
        elif self.phase in ("TRACK", "DOCK"):
            # one tracking step: feedback on the estimate x; the reference clock
            # waits if the robot is > 10 cm behind (tb3control.tracker)
            u_track, self.err, ref = self.follow.track_step(x)
            u = u_track
            self.t_ref = self.follow.t_ref
            self.ref = ref
            done_ref = self.follow.finished
            if self.phase == "TRACK" and done_ref and np.hypot(*(x[:2] - self.predock[:2])) < 0.08:
                self._set_phase("FACE_DOCK")
            elif self.phase == "TRACK" and done_ref and sim.time - self._phase_t0 > self.traj.duration + 8:
                self._set_phase("PLAN")
            if self.phase == "DOCK" and self.p.dock_guard and sim.time - self._dock_seen_t > 1.5:
                self._relocalise()          # lost the dock marker during the approach
                u = np.zeros(2)
            if self.phase == "DOCK":
                along = (self.goal[0] - x[0])
                if done_ref and (abs(along) < self.p.dock_tol_pos or sim.time - self._phase_t0 > self.traj.duration + 5):
                    self._set_phase("DONE")
                    u = np.zeros(2)
        elif self.phase == "FACE_DOCK":
            u, e = heading_control(x[2], self.goal[2], w_max=0.8)
            if abs(e) < 0.03 and self.p.dock_guard and sim.time - self._dock_seen_t > 1.5:
                # facing the dock by the estimate, but the dock marker is not in view:
                # the heading estimate is wrong (unseen disturbance) -> re-localise
                self._relocalise()
                u = np.zeros(2)
            elif abs(e) < 0.03:
                pts = resample(np.vstack([x[:2], self.goal[:2]]), 0.01)
                self.traj = Trajectory(pts, v_max=self.p.v_dock, a_lon=0.1, theta_end=self.goal[2])
                self.t_ref = 0.0
                self.follow.start(self.traj, align_first=False)
                self._set_phase("DOCK")
        if self.phase in ("DONE", "FAIL"):
            u = np.zeros(2)
        if sim.time > self.p.timeout and self.phase not in ("DONE",):
            self._set_phase("FAIL")
            u = np.zeros(2)
        if estop:
            u = np.zeros(2)

        u_cmd = u
        u, (wl, wr) = saturate(u, self.cal)
        if self.phase in ("TRACK", "DOCK"):
            self.follow.note_saturation(u_cmd, u)     # LQI freezes its integrator while saturated
        self.u = u
        if self._observer is not None:
            # the estimate above refers to the sensor time, i.e. before actuation
            self._observer(self)
        sim.step(wl, wr)
        self.k += 1

    def run(self, logger=None, max_steps=None):
        n = int(self.p.timeout / RATES.control_dt) + 10 if max_steps is None else max_steps
        self._observer = logger
        for _ in range(n):
            self.step()
            if self.phase in ("DONE", "FAIL"):
                # a few more steps to let the robot come to rest
                for _ in range(10):
                    self.sim.step(0, 0)
                break
        return self.phase
