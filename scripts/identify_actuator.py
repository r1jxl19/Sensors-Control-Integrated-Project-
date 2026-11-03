"""Identify wheel actuator dynamics from encoder step responses.

Steps in commanded wheel rate (straight-line and in-place-turn) are applied
from rest; wheel rates are obtained by differencing encoder ticks at 100 Hz
(the same signal available on the real robot). First-order
    w(t) = K u (1 - exp(-t / tau))
and second-order (wn, zeta) models are fitted by least squares.

The identified tau sets the bandwidth ceiling for the trajectory tracker
(closed-loop poles must stay well below 1/tau) and is used in the
actuator-augmented model in scripts/analyse_model.py.

Outputs: config/calibration.json (tau, K_motor), results/actuator_id.{json,png}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.sim import SimRobot
from tb3dock.rig import teleport
from tb3dock.config import save_calibration, RESULTS_DIR, NOMINAL

DT = 0.01


def first_order(t, K, tau):
    return K * (1 - np.exp(-t / tau))


def second_order(t, K, wn, zeta):
    zeta = np.clip(zeta, 0.05, 0.999)
    wd = wn * np.sqrt(1 - zeta ** 2)
    return K * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * t)))


def step_test(u_l, u_r, T=0.8):
    sim = SimRobot(dict(start=np.array([2.5, 2.0, 0.0]), obstacles=[]), render_camera=False)
    teleport(sim, (2.5, 2.0, 0.0))
    ticks = [sim.read_encoders()]
    for _ in range(int(T / DT)):
        sim.step(u_l, u_r, dt=DT)
        ticks.append(sim.read_encoders())
    ticks = np.array(ticks, float)
    w = np.diff(ticks, axis=0) * 2 * np.pi / NOMINAL.encoder_ticks / DT
    t = (np.arange(len(w)) + 0.5) * DT       # rate estimate sits mid-interval
    return t, w


def main():
    tests = [("straight 2 rad/s", 2, 2), ("straight 4 rad/s", 4, 4), ("straight 6 rad/s", 6, 6),
             ("turn +/-3 rad/s", -3, 3), ("turn +/-6 rad/s", -6, 6)]
    rows = []
    fig, axes = plt.subplots(1, len(tests), figsize=(16, 3.4), sharey=False)
    for ax, (name, ul, ur) in zip(axes, tests):
        t, w = step_test(ul, ur)
        for j, (u, col) in enumerate(((ul, "#3b6ea5"), (ur, "#c0504d"))):
            y = w[:, j] / u
            p1, _ = curve_fit(first_order, t, y, p0=[1, 0.05], bounds=([0.5, 1e-3], [1.5, 1]))
            p2, _ = curve_fit(second_order, t, y, p0=[1, 30, 0.7], bounds=([0.5, 1, 0.05], [1.5, 300, 0.999]))
            r1 = np.sqrt(np.mean((first_order(t, *p1) - y) ** 2))
            r2 = np.sqrt(np.mean((second_order(t, *p2) - y) ** 2))
            rows.append(dict(test=name, wheel="LR"[j], u=u, K=p1[0], tau=p1[1], rms1=r1,
                             K2=p2[0], wn=p2[1], zeta=p2[2], rms2=r2, peak=float(y.max())))
            ax.plot(t, w[:, j], ".", ms=3, color=col, alpha=0.6, label=f"{'LR'[j]} encoder")
            ax.plot(t, u * first_order(t, *p1), "-", color=col, lw=1.2)
            ax.plot(t, u * second_order(t, *p2), "--", color=col, lw=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("t (s)")
    axes[0].set_ylabel("wheel rate (rad/s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Wheel step responses from encoder differencing: solid = 1st order fit, dashed = 2nd order fit", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "actuator_id.png"), dpi=130)

    tau = float(np.median([r["tau"] for r in rows]))
    K = float(np.median([r["K"] for r in rows]))
    summary = dict(tau=tau, K_motor=K,
                   tau_range=[float(min(r["tau"] for r in rows)), float(max(r["tau"] for r in rows))],
                   wn=float(np.median([r["wn"] for r in rows])),
                   zeta=float(np.median([r["zeta"] for r in rows])),
                   rows=rows)
    with open(os.path.join(RESULTS_DIR, "actuator_id.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    save_calibration(dict(tau=tau, K_motor=K))
    for r in rows:
        print(f"{r['test']:18s} {r['wheel']}  K={r['K']:.3f} tau={r['tau'] * 1e3:5.1f} ms rms={r['rms1']:.3f} | "
              f"wn={r['wn']:5.1f} zeta={r['zeta']:.2f} rms={r['rms2']:.3f} peak={r['peak']:.3f}")
    print(f"tau = {tau * 1e3:.1f} ms, K = {K:.3f}")


if __name__ == "__main__":
    main()
