"""Run one mission and print the result.

  python scripts/run_mission.py --seed 101                 # nominal
  python scripts/run_mission.py --seed 101 --dist combined # disturbed
  python scripts/run_mission.py --seed 101 --gyro          # gyro-aided EKF
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
a = ap.parse_args()
r = run_trial(a.seed, MissionParams(use_gyro=a.gyro, use_camera=not a.no_camera), DIST[a.dist](a.seed))
print(f"outcome {r['outcome']}  success {r['success']}  collided {r['collided']}")
print(f"dock error: dx {r['dock_dx']*1e3:.1f} mm, dy {r['dock_dy']*1e3:.1f} mm, dtheta {np.degrees(r['dock_dth']):.2f} deg")
print(f"estimate RMSE {r['est_rmse_xy']*1e3:.1f} mm / {np.degrees(r['est_rmse_th']):.2f} deg, mean NEES {r['nees_mean']:.1f}")
print("phases:", " -> ".join(f"{p}@{t}s" for t, p in r["events"]))
