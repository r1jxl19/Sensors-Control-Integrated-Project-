"""Build the silent 5-minute presentation video (visuals only; narration is recorded over it).

Slides are drawn from the figures in results/ and the demo clips in results/videos/
(make those first with scripts/make_video.py). Output:
  results/videos/presentation_silent.mp4
  results/videos/presentation_timeline.txt   (start time of every section, for the narrators)

  python scripts/make_presentation_video.py
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont

from tb3dock.config import RESULTS_DIR

R = RESULTS_DIR
V = os.path.join(R, "videos")
W, H, FPS = 1280, 720, 20
BG, INK, MUTED, ACCENT, LINE = (244, 244, 244), (28, 30, 34), (96, 100, 108), (59, 110, 165), (210, 212, 216)
FONT_DIR = "/usr/share/fonts/opentype/inter"
REPO = "github.com/r1jxl19/Sensors-Control-Integrated-Project-"


def font(weight, size):
    for name in (f"Inter-{weight}.otf", "Inter-Regular.otf"):
        p = os.path.join(FONT_DIR, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype("DejaVuSans.ttf", size)


F_TAG, F_TITLE, F_BULLET = font("SemiBold", 15), font("SemiBold", 34), font("Regular", 21)
F_BIG, F_SUB = font("Bold", 52), font("Regular", 24)

# (kind, seconds or clip, section tag, title, figure, bullets)
SLIDES = [
    ("title", 8),
    ("pipeline", 16),
    ("fig", 20, "PART A · MODEL", "State-space model, controllability and observability", "model_analysis.png",
     ["Unicycle kinematics + identified actuator lag (τ = 24.6 ms)",
      "Tracking-error model loses rank at rest (Brockett): lateral error is only correctable while moving",
      "One marker leaves the pose unobservable, so the robot localises from two or more"]),
    ("fig", 10, "PART B · SENSING", "Coordinate frames", "frames.png",
     ["Map {M} · base {B} · camera {C} · LiDAR {L} · marker {K}",
      "Whole chain checked end-to-end: markers reproject to 0.92 px (median)"]),
    ("fig", 15, "PART B · SENSING", "Calibration: camera and wheel odometry", "odometry_calibration.png",
     ["Camera intrinsics from 40 checkerboard views: focal length within 0.5 %",
      "Least-squares odometry fit: held-out drift 473 mm → 72 mm"]),
    ("fig", 15, "PART B · SENSING", "ArUco measurement noise, measured not guessed", "marker_characterisation.png",
     ["758 surveyed placements: PnP range bias grows with r², modelled and removed (26 → 4 mm)",
      "R from the data: σr = 3.0 + 2.9 rn² mm, σβ = 0.12° · reject beyond 2.5 m or 60°"]),
    ("fig", 15, "PART B · SENSING", "2D LiDAR: noise, mounting offset and map accuracy", "sensing/lidar.png",
     ["σ = 10.1 mm, no bias, 1 % dropouts masked · 3.2 cm mounting offset corrected",
      "Mapped-point 3σ error at 3.5 m is 162 mm, inside the 200 mm planning inflation"]),
    ("fig", 18, "PART C · ESTIMATION", "EKF: wheel odometry fused with camera range and bearing", "ekf_vs_odometry.png",
     ["Odometry predicts (Q from calibration); each marker corrects, with a χ² gate on outliers",
      "Remove the camera updates and docking success falls to 0/12"]),
    ("fig", 20, "PART D · CONTROL", "Gain-scheduled LQR on the estimated state", "control/design.png",
     ["LQR on the linearised tracking error, Bryson weights, gains scheduled on speed",
      "Closed-loop poles at least 6× slower than the actuator · stable up to 200 ms delay"]),
    ("fig", 15, "PART D · CONTROL", "LQR vs LQI vs Kanayama", "control/tracking.png",
     ["Estimation noise, not the control law, dominates tracking error",
      "LQI's integral state removes the motor-mismatch bias (14 → 11 mm)"]),
    ("card", 3, "DEMO 1", "Nominal mission", "localise → map → plan → track → dock"),
    ("clip", "nominal.mp4"),
    ("card", 3, "DEMO 2", "Disturbed mission", "pushes, camera blackout and extra sensor noise"),
    ("clip", "disturbed.mp4"),
    ("card", 3, "DEMO 3", "Low tyre friction, gyro-aided EKF", "wheel slip the encoders cannot see"),
    ("clip", "slip_gyro.mp4"),
    ("fig", 20, "EVALUATION", "Monte Carlo evaluation across seven conditions", "robustness.png",
     ["Random obstacle layout and start pose every trial · success = docked within 30 mm / 20 mm / 3°",
      "Nominal 30/30 (median 1.0 mm) · disturbed 71/72 · odometry only 0/12 · uncalibrated 1/12"]),
    ("fig", 15, "DIAGNOSIS", "The one failure: traced and fixed", "trial_combined_fail_106.png",
     ["A push on low friction spun the robot 45° unseen → skewed map → endless replanning",
      "Fix: 8 replans in 10 s → discard the map and re-localise · combined 11/12 → 12/12"]),
    ("end", 6),
]


def base(tag=None, progress=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    if tag:
        tw = d.textlength(tag, font=F_TAG)
        d.rounded_rectangle([48, 34, 48 + tw + 24, 62], 14, fill=ACCENT)
        d.text((60, 38), tag, font=F_TAG, fill="white")
    if progress is not None:
        d.rectangle([0, H - 5, W, H], fill=LINE)
        d.rectangle([0, H - 5, int(W * progress), H], fill=ACCENT)
    return im, d


def paste_fit(im, path, box):
    fig = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    s = min((x1 - x0) / fig.width, (y1 - y0) / fig.height)
    fig = fig.resize((int(fig.width * s), int(fig.height * s)), Image.LANCZOS)
    x, y = x0 + (x1 - x0 - fig.width) // 2, y0 + (y1 - y0 - fig.height) // 2
    im.paste(fig, (x, y))
    ImageDraw.Draw(im).rectangle([x - 1, y - 1, x + fig.width, y + fig.height], outline=LINE)
    return y + fig.height


def fig_slide(tag, title, figure, bullets, progress):
    im, d = base(tag, progress)
    d.text((48, 74), title, font=F_TITLE, fill=INK)
    fig = Image.open(os.path.join(R, figure)).convert("RGB")
    top, bot = 130, H - 24
    bh = 36 * len(bullets) + 18
    sc = min((W - 96) / fig.width, (bot - top - bh) / fig.height)
    fig = fig.resize((int(fig.width * sc), int(fig.height * sc)), Image.LANCZOS)
    y = top + (bot - top - fig.height - bh) // 2
    x = (W - fig.width) // 2
    im.paste(fig, (x, y))
    d.rectangle([x - 1, y - 1, x + fig.width, y + fig.height], outline=LINE)
    y += fig.height + 18
    for b in bullets:
        d.ellipse([52, y + 9, 60, y + 17], fill=ACCENT)
        d.text((72, y), b, font=F_BULLET, fill=INK)
        y += 36
    return im


def title_slide():
    im, d = base()
    hero = Image.open(os.path.join(R, "trial_nominal.png")).convert("RGB")
    hero = hero.crop((0, 40, int(hero.width * 0.52), int(hero.height * 0.86)))
    s = 420 / hero.height
    hero = hero.resize((int(hero.width * s), 420), Image.LANCZOS)
    im.paste(hero, (W - hero.width - 40, 150))
    d.text((60, 210), "Visual navigation", font=F_BIG, fill=INK)
    d.text((60, 272), "and ArUco docking", font=F_BIG, fill=INK)
    d.rectangle([60, 350, 140, 355], fill=ACCENT)
    d.text((60, 378), "TurtleBot3 in MuJoCo · EKF + LQR", font=F_SUB, fill=MUTED)
    d.text((60, 412), "41014 Sensors and Control · Integrated Project · Track A", font=F_SUB, fill=MUTED)
    return im


def pipeline_slide(progress):
    im, d = base("OVERVIEW", progress)
    d.text((48, 74), "The task: find the charger and dock to within millimetres", font=F_TITLE, fill=INK)
    steps = [("Localise", "ArUco markers\non the walls"), ("Map", "unknown obstacles\nfrom 2D LiDAR"),
             ("Plan", "A* path through\nthe free space"), ("Track", "gain-scheduled LQR\non the estimate"),
             ("Dock", "visual approach to\nthe ArUco charger")]
    bw, gap, y0 = 200, 38, 190
    x = (W - (5 * bw + 4 * gap)) // 2
    for i, (h, sub) in enumerate(steps):
        d.rounded_rectangle([x, y0, x + bw, y0 + 150], 16, fill="white", outline=LINE, width=2)
        d.text((x + 20, y0 + 18), f"{i + 1}", font=F_TAG, fill=ACCENT)
        d.text((x + 20, y0 + 40), h, font=F_TITLE, fill=INK)
        d.multiline_text((x + 20, y0 + 88), sub, font=font("Regular", 17), fill=MUTED, spacing=4)
        if i < 4:
            ax = x + bw + 8
            d.polygon([(ax, y0 + 65), (ax + gap - 16, y0 + 75), (ax, y0 + 85)], fill=ACCENT)
        x += bw + gap
    d.rounded_rectangle([140, 395, W - 140, 455], 12, fill=(227, 236, 246))
    msg = "Every command comes from the EKF estimate. Ground truth is only used to score the runs."
    d.text(((W - d.textlength(msg, font=F_BULLET)) // 2, 412), msg, font=F_BULLET, fill=INK)
    stats = [("30/30", "nominal trials docked"), ("71/72", "disturbed trials docked"), ("1.0 mm", "median lateral docking error")]
    x = 140
    cw = (W - 280) // 3
    for big, small in stats:
        d.text((x + (cw - d.textlength(big, font=F_BIG)) // 2, 500), big, font=F_BIG, fill=ACCENT)
        d.text((x + (cw - d.textlength(small, font=F_BULLET)) // 2, 570), small, font=F_BULLET, fill=MUTED)
        x += cw
    return im


def card_slide(tag, title, sub, progress):
    im, d = base(None, progress)
    tw = d.textlength(tag, font=F_TAG)
    d.rounded_rectangle([(W - tw) // 2 - 12, 270, (W + tw) // 2 + 12, 298], 14, fill=ACCENT)
    d.text(((W - tw) // 2, 274), tag, font=F_TAG, fill="white")
    d.text(((W - d.textlength(title, font=F_BIG)) // 2, 320), title, font=F_BIG, fill=INK)
    d.text(((W - d.textlength(sub, font=F_SUB)) // 2, 395), sub, font=F_SUB, fill=MUTED)
    return im


def end_slide():
    im, d = base()
    t = "Thank you"
    d.text(((W - d.textlength(t, font=F_BIG)) // 2, 250), t, font=F_BIG, fill=INK)
    for i, (s, f, c) in enumerate([("Code, data and every figure:", F_SUB, MUTED), (REPO, font("SemiBold", 26), ACCENT),
                                   ("Rebuild everything with ./reproduce.sh", F_SUB, MUTED)]):
        d.text(((W - d.textlength(s, font=f)) // 2, 345 + 40 * i), s, font=f, fill=c)
    return im


def clip_len(name):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          os.path.join(V, name)], capture_output=True, text=True, check=True).stdout
    return float(out)


def main():
    durs = [clip_len(s[1]) if s[0] == "clip" else s[1] for s in SLIDES]
    total = sum(durs)
    tmp = tempfile.mkdtemp()
    parts, t, timeline = [], 0.0, []
    enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-preset", "medium", "-crf", "20"]
    for i, (s, dur) in enumerate(zip(SLIDES, durs)):
        prog = (t + dur) / total
        kind = s[0]
        seg = os.path.join(tmp, f"{i:02d}.mp4")
        if kind == "clip":
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(V, s[1]), "-an",
                            "-vf", f"scale={W}:{H},fps={FPS}", *enc, seg], check=True)
            label = f"clip {s[1]}"
        else:
            if kind == "title":
                im, label = title_slide(), "Title"
            elif kind == "pipeline":
                im, label = pipeline_slide(prog), "Overview: the task and headline results"
            elif kind == "fig":
                im, label = fig_slide(s[2], s[3], s[4], s[5], prog), f"{s[2].title()}: {s[3]}"
            elif kind == "card":
                im, label = card_slide(s[2], s[3], s[4], prog), f"{s[2].title()}: {s[3]}"
            else:
                im, label = end_slide(), "End card"
            png = os.path.join(tmp, f"{i:02d}.png")
            im.save(png)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", png, "-t", f"{dur:.3f}", *enc, seg],
                           check=True)
        timeline.append(f"{int(t // 60)}:{t % 60:05.2f}  ({dur:5.1f} s)  {label}")
        parts.append(seg)
        t += dur
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    out = os.path.join(V, "presentation_silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    "-movflags", "+faststart", out], check=True)
    timeline.append(f"{int(t // 60)}:{t % 60:05.2f}  end")
    with open(os.path.join(V, "presentation_timeline.txt"), "w") as f:
        f.write("\n".join(timeline) + "\n")
    print("\n".join(timeline))
    print(out)


if __name__ == "__main__":
    main()
