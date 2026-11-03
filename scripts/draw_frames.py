"""Draw the frame diagram used on the project page (results/frames.png)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

from tb3dock.config import RESULTS_DIR, CAMERA, LIDAR, WORLD


def axes(ax, o, th, L, lab, col, ylab="y", xlab="x"):
    c, s = np.cos(th), np.sin(th)
    ax.annotate("", o + L * np.array([c, s]), o, arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.annotate("", o + L * np.array([-s, c]), o, arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.text(*(o + 1.12 * L * np.array([c, s])), xlab, color=col, fontsize=9, ha="center", va="center")
    ax.text(*(o + 1.12 * L * np.array([-s, c])), ylab, color=col, fontsize=9, ha="center", va="center")
    ax.text(*(o - 0.25 * L * np.array([c + s, s - c]) * 0.7), lab, color=col, fontsize=11, weight="bold",
            ha="center", va="center")


def main():
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a = ax[0]
    a.add_patch(Rectangle((0, 0), WORLD.size_x, WORLD.size_y, fill=False, lw=2))
    axes(a, np.array([0.0, 0.0]), 0.0, 0.6, "{M}", "k")
    pose = np.array([2.0, 1.6, 0.5])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    body = np.array([[-0.09, -0.07], [0.05, -0.07], [0.05, 0.07], [-0.09, 0.07]]) * 3
    a.fill(*(pose[:2] + body @ R.T).T, color="#333333", alpha=0.25)
    axes(a, pose[:2], pose[2], 0.5, "{B}", "#1f4e9c")
    cam = pose[:2] + R @ (np.array(CAMERA.pos_b[:2]) * 3)
    axes(a, cam, pose[2] - np.pi / 2, 0.3, "{C}", "#2e7d32", ylab="z", xlab="x")
    a.text(*(cam + R @ np.array([0.05, -0.32])), "y_C into page\n(OpenCV: x right, y down, z out of lens)",
           fontsize=7, color="#2e7d32")
    ldr = pose[:2] + R @ (np.array(LIDAR.pos_b[:2]) * 3)
    a.add_patch(Circle(ldr, 0.05, color="#c0504d"))
    a.text(*(ldr + np.array([-0.45, -0.18])), "{L} LiDAR\n(axes ∥ {B})", fontsize=8, color="#c0504d")
    m = np.array([3.5, WORLD.size_y])
    axes(a, m - np.array([0, 0.02]), np.pi, 0.35, "{K} marker", "#8e44ad", ylab="z (out of wall)", xlab="x")
    a.plot([cam[0], m[0]], [cam[1], m[1]], ":", color="#8e44ad")
    d = m - pose[:2]
    a.text(*(pose[:2] + 0.55 * d + np.array([0.05, -0.15])), "range r, bearing β\n(measured in {B})",
           color="#8e44ad", fontsize=9)
    a.annotate("", pose[:2], np.array([0, 0]), arrowprops=dict(arrowstyle="->", color="#1f4e9c", ls="--"))
    a.text(0.9, 0.55, "EKF state\nx = (x, y, θ) of {B} in {M}", color="#1f4e9c", fontsize=9)
    g = (WORLD.size_x - WORLD.dock_block_depth - WORLD.dock_standoff - CAMERA.pos_b[0], WORLD.dock_y)
    a.add_patch(Rectangle((WORLD.size_x - 0.05, WORLD.dock_y - 0.12), 0.05, 0.24, color="#2e7d32"))
    a.plot(*g, "*", color="#2e7d32", ms=14)
    a.text(g[0] - 1.1, g[1] - 0.35, "docked pose\n(4.68, 2.00, 0)", color="#2e7d32", fontsize=9)
    a.set_xlim(-0.4, WORLD.size_x + 0.3)
    a.set_ylim(-0.4, WORLD.size_y + 0.4)
    a.set_aspect("equal")
    a.set_title("Frames in the room (top view; robot drawn 3× scale)")
    a.axis("off")

    b = ax[1]
    b.axis("off")
    chain = [
        ("pixels  (u, v) of 4 marker corners", "ArUco detector, sub-pixel corners"),
        ("T_C_K   marker pose in camera {C}", "solvePnP (IPPE_SQUARE) with calibrated K, dist"),
        ("p_B = R_B_C t_C_K + t_B_C", f"camera extrinsic: t_B_C = {CAMERA.pos_b} m,\nR_B_C maps z_C → x_B, x_C → −y_B, y_C → −z_B"),
        ("z = (r − bias(r), β) = (|p_B,xy|, atan2(p_B,y, p_B,x))", "range bias model from characterisation"),
        ("EKF update vs  h(x) = map point m_K expressed in {B}", "m_K known in {M}; h uses estimated x"),
        ("x̂ = (x, y, θ) in {M}  →  tracking error e in reference frame", "e = [R(θ_r)ᵀ(p − p_r), θ − θ_r]"),
        ("LQR:  (v, ω) = u_ref − K e", "body command"),
        ("wheel rates  ω_l,r = (v ∓ bω/2) / r_l,r", "calibrated r_l, r_r, b"),
    ]
    y = 0.95
    for i, (main_, side) in enumerate(chain):
        b.add_patch(Rectangle((0.0, y - 0.075), 0.55, 0.085, transform=b.transAxes, fc="#eef3fa", ec="#1f4e9c"))
        b.text(0.01, y - 0.032, main_, transform=b.transAxes, fontsize=8.5, va="center")
        b.text(0.58, y - 0.032, side, transform=b.transAxes, fontsize=7.5, va="center", color="#444444")
        if i < len(chain) - 1:
            b.annotate("", (0.27, y - 0.1), (0.27, y - 0.078), xycoords="axes fraction",
                       arrowprops=dict(arrowstyle="->", lw=1.5))
        y -= 0.118
    b.set_title("From sensor measurement to the quantity the controller acts on")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "frames.png"), dpi=130)


if __name__ == "__main__":
    main()
