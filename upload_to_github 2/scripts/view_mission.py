"""Watch a mission live (needs a screen). Two display modes:

  --mode dashboard (default, works on Windows/macOS/Linux)
      A window with the same layout as the demo video: top view with the
      LiDAR map, planned path, true track (green) and EKF estimate + 3-sigma
      ellipse (blue); the robot's camera with marker detections; and live
      estimate-error plots with +/-3 sigma bands.

  --mode 3d
      The interactive MuJoCo viewer (rotate/zoom the scene yourself).
      macOS: start with `mjpython` instead of `python`. If it fails on macOS
      (offscreen camera rendering off the main thread), use the dashboard.

  python scripts/view_mission.py --seed 101                     # nominal
  python scripts/view_mission.py --seed 104 --dist combined     # disturbed
  python scripts/view_mission.py --seed 101 --speed 3           # 3x faster
  python scripts/view_mission.py --seed 101 --mode 3d --follow  # 3-D, camera follows robot

Overlays drawn in the 3-D scene:
  blue arrow       EKF pose estimate (the robot drawn by MuJoCo is the truth)
  blue disc        EKF 3-sigma position uncertainty, magnified x10
  yellow dots      planned path
  green/red bars   marker detections this frame (green = fused, red = rejected)
Viewer controls: drag to rotate, right-drag to pan, scroll to zoom,
space to pause, close the window to stop.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from tb3dock.config import RATES, WORLD
from evaluate import DIST

BLUE = np.array([0.15, 0.35, 0.9, 0.9], np.float32)
YELLOW = np.array([0.95, 0.8, 0.1, 0.9], np.float32)
GREEN = np.array([0.1, 0.8, 0.2, 0.9], np.float32)
RED = np.array([0.9, 0.15, 0.15, 0.9], np.float32)


class Stop(Exception):
    pass


def add_geom(scn, gtype, size, pos, mat, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                        np.asarray(mat, np.float64).reshape(-1), rgba)
    scn.ngeom += 1


def rotz(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def draw_overlays(scn, m):
    scn.ngeom = 0
    if m.traj is not None:
        for p in m.traj.pts[::6][:120]:
            add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [p[0], p[1], 0.01], np.eye(3), YELLOW)
    if m.ekf.initialised:
        x, P = m.ekf.x, m.ekf.P
        # arrow along the estimated heading (arrow geoms point along local z)
        R = rotz(x[2]) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ARROW, [0.012, 0.012, 0.22], [x[0], x[1], 0.22], R, BLUE)
        vals, vecs = np.linalg.eigh(P[:2, :2])
        r = 3 * np.sqrt(np.maximum(vals, 1e-10)) * 10
        Re = np.eye(3)
        Re[:2, :2] = vecs
        if np.linalg.det(Re) < 0:
            Re[:, 0] *= -1
        add_geom(scn, mujoco.mjtGeom.mjGEOM_ELLIPSOID, [max(r[0], 0.01), max(r[1], 0.01), 0.004],
                 [x[0], x[1], 0.006], Re, np.array([0.15, 0.35, 0.9, 0.35], np.float32))
    for o in m.obs:
        p = m.map[o.id]["p"]
        col = GREEN if o.id in m.accepted_ids else RED
        add_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.03, 0.03, 0.03], [p[0], p[1], p[2] + 0.17], np.eye(3), col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--dist", default="none", choices=list(DIST))
    ap.add_argument("--gyro", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed vs real time")
    ap.add_argument("--follow", action="store_true", help="3d mode: camera follows the robot")
    ap.add_argument("--mode", default="dashboard", choices=["dashboard", "3d"])
    ap.add_argument("--lqi", action="store_true", help="LQI controller instead of LQR")
    ap.add_argument("--dock-guard", action="store_true", help="v3 dock-marker guard")
    ap.add_argument("--livelock-fix", action="store_true", help="v2 livelock re-localisation")
    a = ap.parse_args()
    if a.mode == "dashboard":
        return dashboard(a)

    import mujoco.viewer
    state = dict(viewer=None, wall0=None, sim0=None, last_phase=None)

    def on_step(m, log):
        if state["viewer"] is None:
            v = mujoco.viewer.launch_passive(m.sim.model, m.sim.data, show_left_ui=False, show_right_ui=False)
            if a.follow:
                v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                v.cam.trackbodyid = m.sim.model.body("base").id
                v.cam.distance, v.cam.elevation = 1.6, -35
            else:
                v.cam.lookat[:] = [WORLD.size_x / 2, WORLD.size_y / 2, 0]
                v.cam.distance, v.cam.elevation, v.cam.azimuth = 6.5, -62, 90
            state.update(viewer=v, wall0=time.perf_counter(), sim0=m.sim.time)
        v = state["viewer"]
        if not v.is_running():
            raise Stop
        with v.lock():
            draw_overlays(v.user_scn, m)
        v.sync()
        if m.phase != state["last_phase"]:
            print(f"t={m.sim.time:6.2f}s  {m.phase}", flush=True)
            state["last_phase"] = m.phase
        # pace to real time x speed
        target = state["wall0"] + (m.sim.time - state["sim0"]) / a.speed
        dt = target - time.perf_counter()
        if dt > 0:
            time.sleep(dt)

    try:
        r = run_trial(a.seed, MissionParams(use_gyro=a.gyro, controller='LQI' if a.lqi else 'LQR', dock_guard=a.dock_guard, livelock_relocalise=a.livelock_fix), DIST[a.dist](a.seed), frame_cb=on_step)
        print(f"\n{'SUCCESS' if r['success'] else 'FAIL'}: dock error dx={r['dock_dx'] * 1e3:.1f} mm, "
              f"dy={r['dock_dy'] * 1e3:.1f} mm, dtheta={np.degrees(r['dock_dth']):.2f} deg")
        print("Close the viewer window to exit.")
        v = state["viewer"]
        while v is not None and v.is_running():
            v.sync()
            time.sleep(0.05)
    except Stop:
        print("viewer closed")


def dashboard(a):
    """Live version of the demo-video layout in a matplotlib window."""
    import matplotlib.pyplot as plt
    from make_video import Renderer

    plt.ion()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.canvas.manager.set_window_title("tb3dock live mission")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(np.full((720, 1280, 3), 245, np.uint8))
    clock = dict(wall0=None, sim0=None, sim=0.0)

    class Window:
        def append_data(self, frame):
            if not plt.fignum_exists(fig.number):
                raise Stop
            im.set_data(frame)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # pace to real time x speed (frames come every 2 control steps)
            if clock["wall0"] is None:
                clock["wall0"] = time.perf_counter()
            clock["sim"] += 2 * RATES.control_dt
            dt = clock["wall0"] + clock["sim"] / a.speed - time.perf_counter()
            if dt > 0:
                plt.pause(dt)
            else:
                plt.pause(0.001)

    title = f"seed {a.seed}, disturbance '{a.dist}'" + (" + gyro" if a.gyro else "")
    rend = Renderer(title, "", Window())
    try:
        r = run_trial(a.seed, MissionParams(use_gyro=a.gyro, controller='LQI' if a.lqi else 'LQR', dock_guard=a.dock_guard, livelock_relocalise=a.livelock_fix), DIST[a.dist](a.seed), frame_cb=rend)
        msg = (f"{'SUCCESS' if r['success'] else 'FAIL'}: dock error dx={r['dock_dx'] * 1e3:.1f} mm, "
               f"dy={r['dock_dy'] * 1e3:.1f} mm, dtheta={np.degrees(r['dock_dth']):.2f} deg")
        print(msg)
        ax.set_title(msg, y=0.42, fontsize=14, backgroundcolor="white")
        plt.ioff()
        plt.show()
    except Stop:
        print("window closed")


if __name__ == "__main__":
    main()
