"""Render demo clips: top view + robot camera + estimate/uncertainty vs truth.

Each frame shows
  left   top-down render with overlays: LiDAR occupancy map (grey), planned
         path (yellow), ground-truth track (green), EKF estimate track (blue),
         EKF 3-sigma position ellipse (drawn x10 for visibility) and heading;
  right  the robot's camera image with accepted (green) / rejected (red)
         marker detections, and live plots of estimate error with +/-3 sigma
         bands for x, y and theta.

  python scripts/make_video.py                # all clips in results/videos/
  python scripts/make_video.py --clip nominal

Ground truth is drawn for the viewer only; it comes from the evaluation
logger, never from the mission.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2
import imageio.v2 as imageio

from tb3dock.evaluation import run_trial
from tb3dock.mission import MissionParams
from tb3dock.sim import Disturbances
from tb3dock.frames import wrap
from tb3dock.config import RESULTS_DIR, WORLD

OUT = os.path.join(RESULTS_DIR, "videos")
W, H = 1280, 720
TOP_W, TOP_H = 760, 608
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Renderer:
    def __init__(self, title, disturbance_note="", writer=None):
        self.title, self.note = title, disturbance_note
        self.writer = writer
        self.last = None
        self.hist = []
        self.k = 0
        f = 0.5 * TOP_H / np.tan(np.radians(45) / 2)
        self.f, self.h = f, 7.2

    def w2p(self, x, y):
        u = TOP_W / 2 + self.f * (x - WORLD.size_x / 2) / self.h
        v = TOP_H / 2 - self.f * (y - WORLD.size_y / 2) / self.h
        return int(round(u)), int(round(v))

    def __call__(self, m, log):
        self.k += 1
        r = log.rows[-1]
        self.hist.append((r["t"], r["gt"], r["est"], r["P"]))
        if self.k % 2:
            return
        sim = m.sim
        top = sim.render("top", TOP_W, TOP_H).copy()
        # occupancy overlay
        if m.ekf.initialised:
            occ = m.grid.occupied()
            idx = np.argwhere(occ)
            for c in idx[:: max(1, len(idx) // 4000)]:
                p = m.grid.cell_to_world(c)
                cv2.circle(top, self.w2p(*p), 2, (60, 60, 60), -1)
        if m.traj is not None:
            pts = np.array([self.w2p(*p) for p in m.traj.pts[::3]], np.int32)
            cv2.polylines(top, [pts], False, (240, 200, 40), 2)
        gt = np.array([h[1] for h in self.hist])
        est = np.array([h[2] for h in self.hist])
        cv2.polylines(top, [np.array([self.w2p(*p[:2]) for p in gt], np.int32)], False, (40, 170, 60), 2)
        ok = ~np.isnan(est[:, 0])
        if ok.sum() > 1:
            cv2.polylines(top, [np.array([self.w2p(*p[:2]) for p in est[ok]], np.int32)], False, (40, 90, 220), 2)
        if m.ekf.initialised:
            x, P = m.ekf.x, m.ekf.P
            vals, vecs = np.linalg.eigh(P[:2, :2])
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            ax = [max(2, int(3 * np.sqrt(max(v, 0)) * 10 * self.f / self.h)) for v in vals[::-1]]
            cv2.ellipse(top, self.w2p(*x[:2]), (ax[0], ax[1]), -ang, 0, 360, (40, 90, 220), 2)
            p2 = x[:2] + 0.25 * np.array([np.cos(x[2]), np.sin(x[2])])
            cv2.arrowedLine(top, self.w2p(*x[:2]), self.w2p(*p2), (40, 90, 220), 2, tipLength=0.3)
        frame = np.full((H, W, 3), 245, np.uint8)
        frame[56:56 + TOP_H, 10:10 + TOP_W] = top
        # camera view
        cam = m.last_image if m.last_image is not None else np.zeros((480, 640, 3), np.uint8)
        cam = cam.copy()
        for o in m.obs:
            col = (40, 200, 60) if o.id in m.accepted_ids else (220, 50, 50)
            cv2.polylines(cam, [o.corners.astype(np.int32)], True, col, 3)
            cv2.putText(cam, f"id{o.id} {o.range:.2f}m", tuple(o.corners[0].astype(int) + [0, -8]), FONT, 0.7, col, 2)
        cam = cv2.resize(cam, (480, 360))
        frame[56:416, 785:1265] = cam
        cv2.putText(frame, "robot camera (green = fused, red = rejected)", (785, 48), FONT, 0.5, (30, 30, 30), 1)
        # error plots
        self._plots(frame, 430, 785, 480, 280)
        # header / legend
        cv2.putText(frame, self.title, (12, 34), FONT, 0.85, (20, 20, 20), 2)
        cv2.rectangle(frame, (14, 60), (330, 88), (255, 255, 255), -1)
        cv2.putText(frame, f"t = {r['t']:5.1f} s   phase: {m.phase}", (20, 80), FONT, 0.6, (20, 20, 20), 1)
        if self.note:
            cv2.putText(frame, self.note, (12, 690), FONT, 0.55, (180, 40, 40), 2)
        lg = [((40, 170, 60), "ground truth"), ((40, 90, 220), "EKF estimate (ellipse: 3-sigma x10)"),
              ((240, 200, 40), "planned path"), ((60, 60, 60), "LiDAR map")]
        for i, (c, t) in enumerate(lg):
            y = 710
            x0 = 12 + i * 190 if i < 2 else 12 + 2 * 190 + 150 + (i - 2) * 130
            cv2.line(frame, (x0, y - 5), (x0 + 22, y - 5), c, 3)
            cv2.putText(frame, t, (x0 + 28, y), FONT, 0.42, (30, 30, 30), 1)
        self.last = frame
        self.writer.append_data(frame)

    def _plots(self, frame, y0, x0, w, h):
        hist = [hh for hh in self.hist if not np.isnan(hh[2][0])]
        names = ["x error (est - truth)", "y error (est - truth)", "heading error (est - truth)"]
        ph = h // 3
        T = 60.0
        for i in range(3):
            top = y0 + i * ph
            cv2.rectangle(frame, (x0, top), (x0 + w, top + ph - 6), (255, 255, 255), -1)
            cv2.rectangle(frame, (x0, top), (x0 + w, top + ph - 6), (200, 200, 200), 1)
            cv2.putText(frame, names[i], (x0 + 4, top + 14), FONT, 0.42, (40, 40, 40), 1)
            if len(hist) < 2:
                continue
            t = np.array([hh[0] for hh in hist])
            e = np.array([hh[2][i] - hh[1][i] for hh in hist])
            sd = np.array([np.sqrt(hh[3][i, i]) for hh in hist])
            if i == 2:
                e = np.degrees(wrap(e))
                sd = np.degrees(sd)
            scale = 0.05 if i < 2 else 3.0
            t0 = max(0, t[-1] - T)
            sel = t >= t0
            tx = x0 + (t[sel] - t0) / T * w
            mid = top + (ph - 6) / 2
            ys = lambda v: np.clip(mid - v / scale * (ph - 10) / 2, top + 1, top + ph - 7)
            band = np.vstack([np.c_[tx, ys(3 * sd[sel])], np.c_[tx[::-1], ys(-3 * sd[sel])[::-1]]]).astype(np.int32)
            ov = frame.copy()
            cv2.fillPoly(ov, [band], (200, 215, 245))
            frame[:] = cv2.addWeighted(ov, 0.6, frame, 0.4, 0)
            cv2.line(frame, (x0, int(mid)), (x0 + w, int(mid)), (170, 170, 170), 1)
            cv2.polylines(frame, [np.c_[tx, ys(e[sel])].astype(np.int32)], False, (40, 90, 220), 2)
            lab = f"axis +/-{scale * 100:.0f} cm" if i < 2 else f"axis +/-{scale:.0f} deg"
            cv2.putText(frame, lab + ", band +/-3 sigma", (x0 + w - 210, top + 14), FONT, 0.38, (90, 90, 90), 1)


def title_card(writer, lines, n=50):
    f = np.full((H, W, 3), 30, np.uint8)
    for i, (txt, sz) in enumerate(lines):
        cv2.putText(f, txt, (80, 250 + i * 60), FONT, sz, (240, 240, 240), 2)
    for _ in range(n):
        writer.append_data(f)


CLIPS = {
    "nominal": dict(seed=101, params={}, dist=Disturbances(), title="Nominal trial: localise, map, plan, track, dock",
                    note=""),
    "disturbed": dict(seed=104, params={}, title="Disturbed: 2 pushes + 8 s camera blackout + 2x noise",
                      dist=Disturbances(pushes=[(19.0, 0.25, 0.0, 4.0), (29.0, 0.25, -2.8, -2.8)],
                                        camera_dropout=[(22.0, 30.0)], noise_scale=2.0),
                      note="disturbances: pushes at t=19 s and 29 s, camera blacked out 22-30 s"),
    "slip_gyro": dict(seed=106, params={"use_gyro": True}, title="Low tyre friction (0.4) with gyro-aided EKF",
                      dist=Disturbances(wheel_friction=0.4, pushes=[(20.0, 0.25, 3.4, 2.0)]),
                      note="wheel slip + push; heading predicted from the IMU instead of the wheels"),
}


def writer(path):
    return imageio.get_writer(path, fps=20, codec="libx264", quality=7, macro_block_size=8)


def make(name):
    c = CLIPS[name]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.mp4")
    w = writer(path)
    title_card(w, [(c["title"], 0.9)], 30)
    rend = Renderer(c["title"], c["note"], w)
    res = run_trial(c["seed"], MissionParams(**c["params"]), c["dist"], frame_cb=rend)
    summary = (f"result: {'SUCCESS' if res['success'] else 'FAIL'}  dock error dx={res['dock_dx'] * 1e3:.1f} mm, "
               f"dy={res['dock_dy'] * 1e3:.1f} mm, dth={np.degrees(res['dock_dth']):.2f} deg")
    end = rend.last.copy()
    cv2.rectangle(end, (200, 300), (1080, 380), (255, 255, 255), -1)
    cv2.putText(end, summary, (215, 350), FONT, 0.7, (20, 20, 20), 2)
    for _ in range(40):
        w.append_data(end)
    w.close()
    print(name, summary, "->", path, flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="all")
    a = ap.parse_args()
    names = list(CLIPS) if a.clip == "all" else [a.clip]
    paths = []
    intro = os.path.join(OUT, "_intro.mp4")
    os.makedirs(OUT, exist_ok=True)
    w = writer(intro)
    title_card(w, [("TurtleBot3 visual docking - 41014 Integrated Project (Track A)", 1.0),
                   ("EKF (wheel odometry + ArUco range/bearing) + LiDAR A* + LQR tracking", 0.75),
                   ("playback 2x real time", 0.6)])
    w.close()
    for n in names:
        paths.append(make(n))
    if a.clip == "all":
        lst = os.path.join(OUT, "_list.txt")
        with open(lst, "w") as f:
            for p in [intro] + paths:
                f.write(f"file '{p}'\n")
        import imageio_ffmpeg, subprocess
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", lst, "-c", "copy", os.path.join(OUT, "demo_draft.mp4")], check=True)


if __name__ == "__main__":
    main()
