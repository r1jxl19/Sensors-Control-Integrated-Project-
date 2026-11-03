"""Part D design report: gains, poles and robustness margins of the LQR.

  python control_scripts/design_report.py
Outputs: results/control/design.json, results/control/design.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tb3control import bryson, lqr_gain, ctrb_rank, error_model
from tb3control.models import integral_model
from tb3control import analysis
from tb3control.controllers import KanayamaController

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "control")
BLUE, RED, GREEN, GREY = "#3b6ea5", "#c0504d", "#4f9a62", "#7f7f7f"


def main():
    os.makedirs(OUT, exist_ok=True)
    Q, R = bryson()
    rows = []
    for v in (0.06, 0.10, 0.14, 0.18, 0.22):
        K = lqr_gain(v, 0.0, Q, R)
        p = analysis.poles_s(v, 0.0, Q, R)
        rows.append(dict(v_r=v, K=np.round(K, 3).tolist(),
                         poles=sorted([round(float(x.real), 3) for x in p]),
                         max_delay_steps=analysis.max_stable_delay(v, 0.0, Q, R),
                         gain_margin_upper=round(analysis.gain_margin(v, 0.0, Q, R)[1], 2),
                         rho_tau_25ms=round(analysis.rho_with_actuator(v, 0.0, Q, R, 0.0246), 4),
                         rho_tau_60ms=round(analysis.rho_with_actuator(v, 0.0, Q, R, 0.060), 4),
                         rho_design=round(analysis.rho_with_delay(v, 0.0, Q, R, 0), 4)))
    out = dict(
        Q=np.diag(Q).tolist(), R=np.diag(R).tolist(),
        ctrb_rank=dict(rest=ctrb_rank(*error_model(0, 0)), moving=ctrb_rank(*error_model(0.15, 0)),
                       turning=ctrb_rank(*error_model(0, 0.8)), lqi_moving=ctrb_rank(*integral_model(0.15, 0))),
        by_speed=rows,
        kanayama_gains=vars(KanayamaController()),
    )
    # delay robustness vs R scaling (what limits the gains)
    delay = {}
    for s in (0.1, 0.3, 1, 3, 10):
        delay[str(s)] = [round(analysis.rho_with_delay(0.18, 0.0, Q, R * s, n), 4) for n in range(0, 9)]
    out["rho_vs_delay_by_Rscale"] = delay
    out["max_delay_by_Rscale"] = {str(s): analysis.max_stable_delay(0.18, 0.0, Q, R * s) for s in (0.1, 0.3, 1, 3, 10)}
    # ------------------------------------------------------------ figure
    vs = np.linspace(0.05, 0.22, 35)
    Ks = np.array([lqr_gain(v, 0.0, Q, R) for v in vs])
    kan = KanayamaController()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(vs, Ks[:, 1, 1], color=BLUE, lw=2, label="LQR  K_y (cross-track)")
    ax[0].plot(vs, Ks[:, 1, 2], color=RED, lw=2, label="LQR  K_θ (heading)")
    ax[0].plot(vs, Ks[:, 0, 0], color=GREEN, lw=2, label="LQR  K_x (along-track)")
    ax[0].plot(vs, kan.ky * vs, color=BLUE, ls="--", lw=1.2, label="Kanayama v_r·k_y")
    ax[0].plot(vs, kan.kth * vs, color=RED, ls="--", lw=1.2, label="Kanayama v_r·k_θ")
    ax[0].axvline(0.18, color=GREY, ls=":", lw=1)
    ax[0].set_xlabel("reference speed v_r (m/s)")
    ax[0].set_ylabel("gain")
    ax[0].set_title("Scheduled LQR gains vs speed (matched to Kanayama at 0.18 m/s)")
    ax[0].legend(fontsize=7)
    P = np.array([sorted(-analysis.poles_s(v, 0.0, Q, R).real) for v in vs])
    ax[1].semilogy(vs, P[:, 2], color=RED, lw=2, label="heading mode")
    ax[1].semilogy(vs, P[:, 1], color=GREEN, lw=2, label="along-track mode")
    ax[1].semilogy(vs, P[:, 0], color=BLUE, lw=2, label="cross-track mode (∝ v_r)")
    ax[1].axhline(1 / 0.0246, color=GREY, ls=":", lw=1.2)
    ax[1].text(0.052, 1 / 0.0246 * 1.12, "actuator bandwidth 1/τ = 41 s⁻¹", fontsize=8, color=GREY)
    lam = float(np.mean(vs / P[:, 0]))
    ax[1].text(0.055, 0.75, f"cross-track error e-folds every\n≈ {lam * 100:.0f} cm travelled, at any speed", fontsize=8, color=BLUE)
    ax[1].set_ylim(0.1, 80)
    ax[1].set_xlabel("reference speed v_r (m/s)")
    ax[1].set_ylabel("closed-loop decay rate −Re(s) (1/s)")
    ax[1].set_title("Closed-loop poles: all ≥ 6× slower than the actuator")
    ax[1].legend(fontsize=8, loc="center right")
    ax[1].grid(alpha=0.3, which="both")
    out["cross_track_distance_constant_m"] = lam
    ns = np.arange(0, 9)
    for s, c in zip(("0.1", "1", "10"), (RED, BLUE, GREEN)):
        ax[2].plot(ns * 50, delay[s], "o-", color=c, label=f"R × {s}  (max stable {out['max_delay_by_Rscale'][s] * 50} ms)")
    ax[2].axhline(1, color="k", lw=1, ls="--")
    ax[2].set_xlabel("pure command delay (ms)")
    ax[2].set_ylabel("closed-loop spectral radius (< 1 stable)")
    ax[2].set_title("Delay tolerance: smaller R (higher gain) tolerates less")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "design.png"), dpi=130)
    with open(os.path.join(OUT, "design.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rho_vs_delay_by_Rscale"}, indent=1))


if __name__ == "__main__":
    main()
