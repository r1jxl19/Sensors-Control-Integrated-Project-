"""Run one mission headless and save everything a person needs to see it: the video, a results
figure and a text summary. This is what the "Run simulation" button on GitHub runs
(.github/workflows/simulation.yml); it works the same on any computer:

  python scripts/run_on_github.py --seed 101
  python scripts/run_on_github.py --seed 104 --dist combined
  python scripts/run_on_github.py --seed 106 --dist combined --livelock-fix
  python scripts/run_on_github.py --custom-room            # the layout in scripts/custom_scene.py

Outputs (in sim_output/):
  mission.mp4     the dashboard video (top view + robot camera + error plots), 2x real time
  mission.png     map, estimate vs truth, errors with 3-sigma bands, markers fused
  summary.md      result, docking error, phases, re-localisations (also shown on the GitHub run page)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from evaluate import DIST
from make_video import Renderer, writer, title_card, FONT
from plot_trial import plot

OUT = "sim_output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101, help="random room and start (101 = the nominal demo)")
    ap.add_argument("--dist", default="none", choices=list(DIST), help="disturbance condition")
    ap.add_argument("--gyro", action="store_true", help="gyro-aided EKF")
    ap.add_argument("--lqi", action="store_true", help="LQI controller instead of LQR")
    ap.add_argument("--dock-guard", action="store_true", help="v3 docking guard")
    ap.add_argument("--livelock-fix", action="store_true", help="v2 livelock re-localisation")
    ap.add_argument("--custom-room", action="store_true", help="use the room in scripts/custom_scene.py")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    scenario, room = None, f"random room, seed {a.seed}"
    if a.custom_room:
        import custom_scene as cs
        from tb3dock.config import WORLD
        WORLD.dock_y = cs.DOCK_Y
        scenario = cs.build_scenario()
        cs.check(scenario)
        room = "custom room from scripts/custom_scene.py"
    extras = [n for n, on in (("gyro", a.gyro), ("LQI", a.lqi), ("dock guard", a.dock_guard),
                              ("livelock fix", a.livelock_fix)) if on]
    label = f"{room}, disturbance '{a.dist}'" + (f" + {', '.join(extras)}" if extras else "")
    params = MissionParams(use_gyro=a.gyro, controller="LQI" if a.lqi else "LQR", dock_guard=a.dock_guard,
                           livelock_relocalise=a.livelock_fix)

    print(f"Running: {label}", flush=True)
    w = writer(os.path.join(OUT, "mission.mp4"))
    title_card(w, [("TurtleBot3 visual docking - 41014 Integrated Project (Track A)", 0.9),
                   (label[:90], 0.65), ("playback 2x real time", 0.6)], 30)
    rend = Renderer(label, "", w)
    r = run_trial(a.seed, params, DIST[a.dist](a.seed), scenario=scenario, keep_log=True, frame_cb=rend)
    ok = "SUCCESS" if r["success"] else "FAIL"
    line = (f"{ok}: dock error dx={r['dock_dx'] * 1e3:.1f} mm, dy={r['dock_dy'] * 1e3:.1f} mm, "
            f"dtheta={np.degrees(r['dock_dth']):.2f} deg")
    end = rend.last.copy()
    cv2.rectangle(end, (200, 300), (1080, 380), (255, 255, 255), -1)
    cv2.putText(end, line, (215, 350), FONT, 0.7, (20, 20, 20), 2)
    for _ in range(40):
        w.append_data(end)
    w.close()
    plot(r, os.path.join(OUT, "mission.png"), label)

    phases = " -> ".join(f"{p} @ {t} s" for t, p in r["events"])
    md = "\n".join([
        f"## {'✅' if r['success'] else '❌'} {ok}",
        "",
        f"**Run:** {label}",
        "",
        "| Result | Value |",
        "|---|---|",
        f"| Docked | {'yes' if r['success'] else 'no'} |",
        f"| Dock error along / sideways / heading | {r['dock_dx'] * 1e3:.1f} mm / {r['dock_dy'] * 1e3:.1f} mm / "
        f"{np.degrees(r['dock_dth']):.2f}° |",
        f"| EKF position error (RMS against truth) | {r['est_rmse_xy'] * 1e3:.1f} mm |",
        f"| Markers fused / rejected by the gate | {r['n_accepted']} / {r['n_gated']} |",
        f"| Path replans / re-localisations | {r['n_replans']} / {r['n_relocalise']} |",
        "",
        f"**Phases:** {phases}",
        "",
        "Download the **simulation-output** artifact at the bottom of this page for the video "
        "(`mission.mp4`) and the results figure (`mission.png`).",
    ])
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(md + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(line)
    print(f"saved {OUT}/mission.mp4, {OUT}/mission.png, {OUT}/summary.md")


if __name__ == "__main__":
    main()
