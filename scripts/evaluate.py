"""Monte Carlo evaluation: nominal performance, robustness and trade-off studies.

Every condition re-uses the same seeds (same obstacle layouts and start poses)
so comparisons between conditions are paired.

  python scripts/evaluate.py --exp all            # everything (~1.5 h on 2 cores)
  python scripts/evaluate.py --exp nominal --n 10 # quick check

Results: results/eval/<experiment>.csv  (one row per trial)
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from tb3dock.sim import Disturbances
from tb3dock.controller import bryson
from tb3dock.config import load_calibration, NOMINAL, RESULTS_DIR

EVAL_DIR = os.path.join(RESULTS_DIR, "eval")
SEED0 = 100


# ----------------------------------------------------------- disturbances
def d_none(seed):
    return Disturbances()


def d_push(seed):
    rng = np.random.default_rng(seed + 7)
    pushes = []
    for t in (rng.uniform(16, 22), rng.uniform(26, 32)):
        a = rng.uniform(-np.pi, np.pi)
        pushes.append((t, 0.25, 4.0 * np.cos(a), 4.0 * np.sin(a)))   # 4 N: largest that never tips the robot
    return Disturbances(pushes=pushes)


def d_dropout(seed):
    rng = np.random.default_rng(seed + 11)
    t0 = rng.uniform(16, 22)
    return Disturbances(camera_dropout=[(t0, t0 + 10.0)])


def d_noise(seed):
    return Disturbances(noise_scale=3.0)


def d_motor(seed):
    return Disturbances(motor_gain=(0.90, 1.08))


def d_slip(seed):
    return Disturbances(wheel_friction=0.4)


def d_combined(seed):
    p = d_push(seed).pushes
    rng = np.random.default_rng(seed + 11)
    t0 = rng.uniform(16, 22)
    return Disturbances(pushes=p, camera_dropout=[(t0, t0 + 6.0)], noise_scale=2.0,
                        motor_gain=(0.93, 1.05), wheel_friction=0.6, gyro_bias=0.01)


DIST = dict(none=d_none, push=d_push, dropout=d_dropout, noise=d_noise, motor=d_motor,
            slip=d_slip, combined=d_combined)


def uncalibrated():
    cal = load_calibration()
    cal.update(wheel_radius_l=NOMINAL.wheel_radius_l, wheel_radius_r=NOMINAL.wheel_radius_r,
               track=NOMINAL.track, range_bias_c=[0.0, 0.0])
    return cal


# ----------------------------------------------------------- experiments
def experiments(n):
    E = {}
    E["nominal"] = [("ekf", {}, "none", None, 2 * n)]
    E["robustness"] = [(dn, {}, dn, None, n) for dn in ("push", "dropout", "noise", "motor", "slip", "combined")]
    E["baseline_odometry"] = [(f"odom_{dn}", {"use_camera": False}, dn, None, n) for dn in ("none", "push", "slip")]
    E["gyro"] = [(f"gyro_{dn}", {"use_gyro": True}, dn, None, n) for dn in ("none", "push", "slip", "combined")]
    E["uncalibrated"] = [("uncal", {}, "none", "uncal", n)]
    # v2: livelock re-localisation fix, verified on the disturbance conditions that exposed it
    E["v2_livelock_fix"] = [(f"v2_{dn}", {"livelock_relocalise": True}, dn, None, n) for dn in ("push", "combined")]
    # Part D extension: LQI (integral action) in the full pipeline, same seeds as nominal/robustness
    E["control_lqi"] = [(f"lqi_{dn}", {"controller": "LQI"}, dn, None, n) for dn in ("none", "motor", "push")]
    # v3: dock-marker guard (control-side docking safety), same seeds
    E["dock_guard"] = ([(f"guard_lqr_{dn}", {"dock_guard": True}, dn, None, n) for dn in ("none", "push", "combined")] +
                       [("guard_lqi_push", {"dock_guard": True, "controller": "LQI"}, "push", None, n)])
    m = max(6, (2 * n) // 3)
    E["sweep_alpha"] = [(f"alpha_{a}", {"ekf_alpha": a}, "push", None, m) for a in (0.1, 0.3, 1, 3, 10, 30, 100)]
    E["sweep_beta"] = [(f"beta_{b}", {"ekf_beta": b}, "push", None, m) for b in (0.1, 0.3, 3, 10)]
    Qb, Rb = bryson()
    E["sweep_lqr"] = [(f"lqrR_{s}", {"lqr_Q": Qb, "lqr_R": Rb * s}, "none", None, m) for s in (0.1, 0.3, 1, 3, 10)]
    return E


def _job(args):
    exp, cond, seed, pkw, dname, calname = args
    params = MissionParams(**pkw)
    cal = uncalibrated() if calname == "uncal" else None
    t0 = time.time()
    try:
        r = run_trial(seed, params, DIST[dname](seed), cal=cal)
    except Exception as e:                      # record crashes as failures, don't lose the batch
        r = dict(seed=seed, outcome=f"ERROR {type(e).__name__}: {e}", success=False)
    r.update(exp=exp, condition=cond, disturbance=dname, wall_s=round(time.time() - t0, 1))
    r["events"] = ";".join(f"{t}:{p}" for t, p in r.get("events", []))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all")
    ap.add_argument("--n", type=int, default=20, help="seeds per condition (nominal uses 2n)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(EVAL_DIR, exist_ok=True)
    E = experiments(a.n)
    names = list(E) if a.exp == "all" else a.exp.split(",")
    for name in names:
        jobs = []
        for cond, pkw, dname, calname, n in E[name]:
            jobs += [(name, cond, SEED0 + i, pkw, dname, calname) for i in range(n)]
        t0 = time.time()
        with Pool(a.workers) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
        keys = sorted({k for r in rows for k in r})
        with open(os.path.join(EVAL_DIR, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        succ = np.mean([r["success"] for r in rows])
        print(f"[{name}] {len(rows)} trials in {time.time() - t0:.0f} s, overall success {succ:.2f}", flush=True)


if __name__ == "__main__":
    main()
