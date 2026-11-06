"""Run the mission in a room YOU lay out: robot start, obstacles, dock position.

Edit the values in the EDIT HERE block, then:

  python scripts/custom_scene.py               # live window (same dashboard as view_mission.py)
  python scripts/custom_scene.py --no-window   # just print the result
  python scripts/custom_scene.py --speed 3     # live window, 3x faster

Room: 5 m (x) by 4 m (y), origin in the bottom-left corner, x to the right.
The robot does NOT get told where the obstacles are: it finds them with the
LiDAR, exactly as in the random seeds. It only knows the marker map and the dock.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

# ============================== EDIT HERE ==================================
# Robot start: x (m), y (m), heading (degrees; 0 = facing the dock wall, 90 = facing +y)
START = (0.7, 3.0, 0.0)

# Dock position along the right-hand wall (x = 5 m). Default 2.0.
# Keep it between about 1.3 and 2.7 so it doesn't sit on top of the wall
# markers at y = 0.8 and y = 3.2.
DOCK_Y = 2.0

# Obstacles. Each is a box or a cylinder:
#   ("box",      x, y, half_length, half_width, yaw_degrees)
#   ("cylinder", x, y, radius)
# Keep them inside the room, away from the start, and leave about 0.5 m
# clear in front of the dock so the robot can line up.
OBSTACLES = [
    ("box", 2.2, 2.2, 0.20, 0.15, 20),
    ("cylinder", 3.0, 1.0, 0.12),
    ("box", 1.8, 0.9, 0.15, 0.15, 0),
]
# ===========================================================================


def build_scenario():
    obs = []
    for o in OBSTACLES:
        if o[0] == "box":
            _, x, y, hl, hw, yaw = o
            obs.append(dict(kind="box", c=np.array([x, y]), half=np.array([hl, hw]),
                            yaw=np.radians(yaw), rad=float(np.hypot(hl, hw))))
        elif o[0] == "cylinder":
            _, x, y, r = o
            obs.append(dict(kind="cylinder", c=np.array([x, y]), half=np.array([r, r]), yaw=0.0, rad=float(r)))
        else:
            raise ValueError(f"unknown obstacle type {o[0]!r} (use 'box' or 'cylinder')")
    x, y, th = START
    return dict(start=np.array([x, y, np.radians(th)]), obstacles=obs)


def check(sc):
    from tb3dock.config import WORLD, dock_goal
    x, y, _ = sc["start"]
    if not (0.3 <= x <= WORLD.size_x - 0.3 and 0.3 <= y <= WORLD.size_y - 0.3):
        sys.exit(f"START ({x}, {y}) is too close to a wall or outside the 5 x 4 m room")
    if not (1.0 <= DOCK_Y <= 3.0):
        sys.exit("DOCK_Y must be between 1.0 and 3.0 m")
    g = np.array(dock_goal()[:2])
    for o in sc["obstacles"]:
        c, r = o["c"], o["rad"]
        if not (r < c[0] < WORLD.size_x - r and r < c[1] < WORLD.size_y - r):
            sys.exit(f"obstacle at {tuple(c)} is outside the room")
        if np.linalg.norm(c - [x, y]) < r + 0.3:
            sys.exit(f"obstacle at {tuple(c)} overlaps the robot's start")
        if np.linalg.norm(c - g) < r + 0.4:
            print(f"warning: obstacle at {tuple(c)} is very close to the dock; docking may fail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--lqi", action="store_true")
    ap.add_argument("--dock-guard", action="store_true")
    ap.add_argument("--livelock-fix", action="store_true")
    ap.add_argument("--seed", type=int, default=1, help="sensor-noise seed (layout comes from this file)")
    a = ap.parse_args()

    from tb3dock.config import WORLD
    WORLD.dock_y = DOCK_Y               # dock, its marker and the docking goal all follow this
    sc = build_scenario()
    check(sc)

    from tb3dock import evaluation
    from tb3dock.mission import MissionParams
    params = MissionParams(controller="LQI" if a.lqi else "LQR", dock_guard=a.dock_guard,
                           livelock_relocalise=a.livelock_fix)

    if a.no_window:
        r = evaluation.run_trial(a.seed, params, scenario=sc)
        print(f"{'SUCCESS' if r['success'] else 'FAIL'}: dock error dx={r['dock_dx'] * 1e3:.1f} mm, "
              f"dy={r['dock_dy'] * 1e3:.1f} mm, dtheta={np.degrees(r['dock_dth']):.2f} deg")
        return

    import view_mission
    real = view_mission.run_trial
    view_mission.run_trial = lambda seed, params, dist, frame_cb=None: real(seed, params, dist, scenario=sc,
                                                                           frame_cb=frame_cb)
    view_mission.dashboard(argparse.Namespace(seed=a.seed, dist="none", gyro=False, speed=a.speed, lqi=a.lqi,
                                              dock_guard=a.dock_guard, livelock_fix=a.livelock_fix))


if __name__ == "__main__":
    main()
