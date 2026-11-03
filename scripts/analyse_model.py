"""Part A analysis: controllability, observability and LQR/actuator interaction.

1. Controllability of the linearised tracking-error model over the operating
   envelope (v_r, w_r): rank and conditioning of the controllability matrix.
2. Controllability of the actuator-augmented 5-state model.
3. LQR closed loop: eigenvalues of the 3-state design, and of the true
   5-state plant (with the identified actuator lag) under the same gain,
   to confirm the design is valid at the identified bandwidth.
4. Observability of the pose from each sensor configuration (empirical local
   observability Gramian of the nonlinear model) - which states each sensor
   makes observable, and what is lost when the marker leaves the view.

Outputs: results/model_analysis.{json,png}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3dock.model import (error_model, augmented_model, ctrb, c2d, dlqr, empirical_obs_gramian,
                           H_range_bearing, obsv, jac_pose)
from tb3dock.controller import bryson
from tb3dock.config import load_calibration, RESULTS_DIR, RATES, NOMINAL


def main():
    cal = load_calibration()
    tau = cal.get("tau", 0.025)
    tau_hi = 0.040                     # upper end of the identified range (conservative)
    dt = RATES.control_dt
    out = {}

    # ---- 1. controllability over the envelope
    vs = np.linspace(-0.22, 0.22, 45)
    ws = np.linspace(-1.5, 1.5, 61)
    smin = np.zeros((len(ws), len(vs)))
    rank = np.zeros_like(smin)
    for i, w in enumerate(ws):
        for j, v in enumerate(vs):
            A, B = error_model(v, w)
            C = ctrb(A, B)
            sv = np.linalg.svd(C, compute_uv=False)
            smin[i, j] = sv[-1]
            rank[i, j] = np.linalg.matrix_rank(C, tol=1e-9)
    A0, B0 = error_model(0, 0)
    out["ctrb_rank_at_rest"] = int(np.linalg.matrix_rank(ctrb(A0, B0)))
    out["ctrb_rank_moving"] = int(np.linalg.matrix_rank(ctrb(*error_model(0.15, 0))))
    out["ctrb_rank_turning_in_place"] = int(np.linalg.matrix_rank(ctrb(*error_model(0.0, 0.8))))
    Aa, Ba = augmented_model(tau, 0.15, 0.0)
    out["ctrb_rank_augmented"] = int(np.linalg.matrix_rank(ctrb(Aa, Ba)))

    # ---- 3. LQR poles, 3-state design vs 5-state plant
    Q, R = bryson()
    poles = {}
    for v_r in (0.06, 0.12, 0.18):
        A, B = error_model(v_r, 0.0)
        Ad, Bd = c2d(A, B, dt)
        K, _ = dlqr(Ad, Bd, Q, R)
        z_design = np.linalg.eigvals(Ad - Bd @ K)
        s_design = np.log(z_design.astype(complex)) / dt
        res = dict(K=K.tolist(), design_poles_s=[[float(p.real), float(p.imag)] for p in s_design])
        for name, tt in (("tau_id", tau), ("tau_hi", tau_hi)):
            Aa, Ba = augmented_model(tt, v_r, 0.0)
            Aad, Bad = c2d(Aa, Ba, dt)
            Kaug = np.hstack([K, np.zeros((2, 2))])      # same gain, actuator states unmeasured
            z = np.linalg.eigvals(Aad - Bad @ Kaug)
            res[f"spectral_radius_{name}"] = float(np.max(np.abs(z)))
            res[f"poles_s_{name}"] = [[float(p.real), float(p.imag)] for p in np.log(z.astype(complex)) / dt]
        poles[str(v_r)] = res
    out["lqr"] = poles
    out["actuator_bandwidth_rad_s"] = 1 / tau

    # ---- 4. observability of the pose from different sensor configurations
    x0 = np.array([2.0, 1.5, 0.3])
    lm1 = [np.array([3.5, 2.5])]
    lm2 = [np.array([3.5, 2.5]), np.array([3.0, 0.0])]
    still = [(0.0, 0.0)] * 20
    move = [(0.15, 0.3)] * 20
    cases = [
        ("wheel odometry only", still, [], "none"),
        ("1 marker, bearing only, parked", still, lm1, "bearing"),
        ("1 marker, range only, parked", still, lm1, "range"),
        ("1 marker, range+bearing, parked", still, lm1, "rb"),
        ("1 marker, range+bearing, moving", move, lm1, "rb"),
        ("1 marker, range only, moving", move, lm1, "range"),
        ("2 markers, range+bearing, parked", still, lm2, "rb"),
        ("marker PnP pose (full pose)", still, [], "pose"),
    ]
    obs_rows = []
    for name, ctrl, lms, meas in cases:
        sv, vmin = empirical_obs_gramian(x0, ctrl, dt, lms, meas)
        r = int(np.sum(sv > 1e-6 * max(sv.max(), 1e-12))) if sv.max() > 0 else 0
        obs_rows.append(dict(case=name, rank=r, sv=sv.tolist(), weakest_dir=vmin.round(3).tolist()))
    out["observability"] = obs_rows
    # analytic: null space of H for one landmark = rotation about the landmark
    H = H_range_bearing(x0, lm1[0])
    _, _, Vt = np.linalg.svd(H)
    out["H_nullspace_1marker"] = Vt[-1].round(4).tolist()
    d = lm1[0] - x0[:2]
    rot = np.array([d[1], -d[0], 1.0])        # rigid rotation of robot about the landmark
    out["rotation_about_landmark_dir"] = (rot / np.linalg.norm(rot)).round(4).tolist()

    with open(os.path.join(RESULTS_DIR, "model_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].pcolormesh(vs, ws, np.log10(smin + 1e-12), shading="auto", cmap="viridis", vmin=-4, vmax=0)
    ax[0].plot(0, 0, "rx", ms=10, mew=2)
    ax[0].annotate("rank 2 at v_r = w_r = 0\n(Brockett: cannot correct\nlateral error by linear\nfeedback at rest)", (0, 0),
                   (0.02, 0.6), color="w", fontsize=8, arrowprops=dict(color="w", arrowstyle="->"))
    ax[0].set_xlabel("reference speed v_r (m/s)")
    ax[0].set_ylabel("reference yaw rate w_r (rad/s)")
    ax[0].set_title("Controllability: log10 smallest singular value")
    fig.colorbar(im, ax=ax[0])
    cols = {"0.06": "#9ecae1", "0.12": "#4292c6", "0.18": "#08519c"}
    for v_r, res in poles.items():
        p = np.array(res["design_poles_s"])
        ax[1].plot(p[:, 0], p[:, 1], "o", color=cols[v_r], label=f"3-state design, v_r={v_r}")
        p = np.array(res["poles_s_tau_hi"])
        ax[1].plot(p[:, 0], p[:, 1], "x", color=cols[v_r], ms=8, label=f"with actuator lag τ={tau_hi*1e3:.0f} ms")
    ax[1].axvline(-1 / tau_hi, color="grey", ls=":", lw=1)
    ax[1].text(-1 / tau_hi, 1, "  actuator pole", fontsize=8, color="grey")
    ax[1].set_xlim(-40, 1)
    ax[1].set_xlabel("Re(s) (1/s)")
    ax[1].set_ylabel("Im(s) (rad/s)")
    ax[1].set_title("LQR closed-loop poles (continuous equivalent)")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)
    names = [r["case"] for r in obs_rows]
    svm = np.array([np.sort(r["sv"]) for r in obs_rows])
    for k in range(3):
        ax[2].barh(np.arange(len(names)) + (k - 1) * 0.25, np.log10(svm[:, k] + 1e-12), height=0.25,
                   label=f"σ{k + 1}" if k else "σ_min")
    ax[2].set_yticks(range(len(names)))
    ax[2].set_yticklabels([f"{n} (rank {r['rank']})" for n, r in zip(names, obs_rows)], fontsize=7)
    ax[2].set_xlabel("log10 singular value of observability Gramian")
    ax[2].axvline(-6, color="k", lw=0.8, ls=":")
    ax[2].set_xlim(-13, 6)
    ax[2].set_title("Pose observability by sensor configuration")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_analysis.png"), dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k != "lqr"}, indent=1))
    for v_r, res in poles.items():
        print(v_r, "K=", np.round(res["K"], 2).tolist(), "rho(tau_id)=", round(res["spectral_radius_tau_id"], 4),
              "rho(tau_hi)=", round(res["spectral_radius_tau_hi"], 4))
        print("   design poles", np.round(res["design_poles_s"], 2).tolist())


if __name__ == "__main__":
    main()
