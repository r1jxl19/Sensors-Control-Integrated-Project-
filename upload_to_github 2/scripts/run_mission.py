"""Run one mission and print the result.

  python scripts/run_mission.py --seed 101                 # nominal
  python scripts/run_mission.py --seed 101 --dist combined # disturbed
  python scripts/run_mission.py --seed 101 --gyro          # gyro-aided EKF
  python scripts/run_mission.py --seed 106 --dist combined --livelock-fix   # v2 fix for the replan livelock
  python scripts/run_mission.py --seed 109 --dist push --lqi                # LQI: blind-docking failure
  python scripts/run_mission.py --seed 109 --dist push --lqi --dock-guard   # ... fixed by the dock guard
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from evaluate import DIST

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=101)
ap.add_argument("--dist", default="none", choices=list(DIST))
ap.add_argument("--gyro", action="store_true")
ap.add_argument("--no-camera", action="store_true", help="odometry-only after initial localisation")
ap.add_argument("--lqi", action="store_true", help="LQI controller (integral action) instead of LQR")
ap.add_argument("--dock-guard", action="store_true", help="v3: approach only while the dock marker is in view")
ap.add_argument("--livelock-fix", action="store_true", help="v2: re-localise after 8 replans in 10 s")
a = ap.parse_args()
params = MissionParams(use_gyro=a.gyro, use_camera=not a.no_camera, controller="LQI" if a.lqi else "LQR",
                       dock_guard=a.dock_guard, livelock_relocalise=a.livelock_fix)
r = run_trial(a.seed, params, DIST[a.dist](a.seed))
print(f"outcome {r['outcome']}  success {r['success']}  collided {r['collided']}")
print(f"dock error: dx {r['dock_dx']*1e3:.1f} mm, dy {r['dock_dy']*1e3:.1f} mm, dtheta {np.degrees(r['dock_dth']):.2f} deg")
print(f"estimate RMSE {r['est_rmse_xy']*1e3:.1f} mm / {np.degrees(r['est_rmse_th']):.2f} deg, mean NEES {r['nees_mean']:.1f}")
ev = r["events"]
if len(ev) > 14:          # long replan sequences: show the start and the end
    ev = ev[:6] + [("...", "...")] + ev[-6:]
print("phases:", " -> ".join(f"{p}@{t}s" for t, p in ev))
print(f"re-localisations {r['n_relocalise']}, replans {r['n_replans']}, markers fused {r['n_accepted']}, gated {r['n_gated']}")
