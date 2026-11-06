"""Make all figures in figures/ from results/ (run after run_mnist.py, run_toy.py)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

# fixed entity -> colour mapping (validated categorical palette, light mode)
COL = {"rescaled": "#2a78d6", "malenia": "#eb6834", "ringleader": "#1baf7a",
       "vanilla": "#eda100", "delay_adaptive": "#e87ba4", "ringleader_reuse": "#6250d6",
       "delay_adaptive_mishchenko": "#e34948"}
MARK = {"rescaled": "o", "malenia": "s", "ringleader": "^", "vanilla": "D",
        "delay_adaptive": "v", "ringleader_reuse": "P", "delay_adaptive_mishchenko": "X"}
LAB = {"rescaled": "Rescaled ASGD", "malenia": "Malenia SGD", "ringleader": "Ringleader ASGD",
       "vanilla": "Vanilla ASGD", "delay_adaptive": "Delay-Adaptive ASGD",
       "ringleader_reuse": "Ringleader (table-reuse variant)",
       "delay_adaptive_mishchenko": "Delay-Adaptive ASGD (min(1,n/δ) rule)"}
SET = {"fixed": "(a) Fixed times", "exp": "(b) Fluctuating times (Exp)"}

# values read off the rendered Fig. 1 of the paper (approx. +-5%)
PAPER_FIG1 = {
    "fixed": {"t": [2000, 10000, 20000, 30000],
              "rescaled": [0.276, 0.100, 0.051, 0.029], "malenia": [0.388, 0.228, 0.156, 0.120],
              "ringleader": [0.354, 0.196, 0.130, 0.096]},
    "exp": {"t": [2000, 10000, 20000, 30000],
            "rescaled": [0.275, 0.092, 0.051, 0.029], "malenia": [0.447, 0.263, 0.195, 0.152],
            "ringleader": [0.43, 0.254, 0.185, 0.140]},
}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def _load_runs(setting, method, gamma):
    L, grid = [], None
    for sd in range(5):
        p = os.path.join(RES, "mnist", f"{setting}_{method}_g{gamma:g}_s{sd}.npz")
        if os.path.exists(p):
            z = np.load(p)
            L.append(z["loss"])
            grid = z["grid"]
    return grid, (np.stack(L) if L else None)


def fig1(methods, fname, title_extra="", show_paper=True):
    with open(os.path.join(RES, "mnist_summary.json")) as fh:
        S = json.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, setting in zip(axes, ("fixed", "exp")):
        for m in methods:
            if m not in S.get(setting, {}):
                continue
            g = S[setting][m]["gamma"]
            grid, L = _load_runs(setting, m, g)
            if L is None:
                continue
            med = np.median(L, axis=0)
            ax.fill_between(grid, L.min(0), L.max(0), color=COL[m], alpha=0.22, lw=0)
            ax.plot(grid, med, color=COL[m], lw=2, marker=MARK[m], markevery=13, ms=6,
                    label=f"{LAB[m]} (γ={g:g}, {len(L)} seeds)")
            if show_paper and m in PAPER_FIG1[setting]:
                ax.plot(PAPER_FIG1[setting]["t"], PAPER_FIG1[setting][m], ls="none",
                        marker=MARK[m], ms=9, mfc="none", mec=COL[m], mew=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Simulated wall-clock time")
        ax.set_title(SET[setting] + title_extra)
    axes[0].set_ylabel("Training cross-entropy loss F(x)")
    h, lab = axes[0].get_legend_handles_labels()
    if show_paper:
        h.append(plt.Line2D([], [], ls="none", marker="o", mfc="none", mec="0.3", ms=9))
        lab.append("hollow markers: paper Fig. 1 (read off)")
    axes[0].legend(h, lab, fontsize=8, loc="upper right")
    hh, ll = axes[1].get_legend_handles_labels()
    axes[1].legend(hh, ll, fontsize=8, loc="upper right")
    fig.suptitle("Label-partitioned MNIST, 2-layer MLP, n=10; median over seeds, band = min/max",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)


def _seed0(setting, method, gamma, sub="mnist"):
    p = os.path.join(RES, sub, f"{setting}_{method}_g{gamma:g}_s0.npz")
    if not os.path.exists(p):
        return None, None
    z = np.load(p)
    return z["grid"], z["loss"]


def fig1_diag():
    """Baselines vs the paper's curves: diagnostic single-seed runs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, setting in zip(axes, ("fixed", "exp")):
        curves = [("malenia", 0.1, "mnist", "-", "Malenia γ=0.1 (tuned)"),
                  ("malenia", 0.05, "diag", "--", "Malenia γ=0.05 [diag.]"),
                  ("ringleader", 0.1, "mnist", "-", "Ringleader γ=0.1 (tuned)"),
                  ("ringleader", 0.01, "mnist", "--", "Ringleader γ=0.01"),
                  ("ringleader_reuse", 0.1, "mnist", "-.", "Ringleader table-reuse γ=0.1 [diag.]"),
                  ("ringleader_reuse", 0.01, "mnist", ":", "Ringleader table-reuse γ=0.01 [diag.]")]
        for m, g, sub, ls, lab in curves:
            grid, L = _seed0(setting, m, g, sub)
            if grid is None:
                continue
            ax.plot(grid, L, color=COL[m], ls=ls, lw=1.8, label=lab + " (seed 0)")
        for m in ("malenia", "ringleader"):
            ax.plot(PAPER_FIG1[setting]["t"], PAPER_FIG1[setting][m], ls="none", marker=MARK[m], ms=9,
                    mfc="none", mec=COL[m], mew=1.5, label=f"paper {LAB[m]} (read off)")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 3)
        ax.set_xlabel("Simulated wall-clock time")
        ax.set_title(SET[setting])
        ax.legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylabel("Training cross-entropy loss F(x)")
    fig.suptitle("Diagnostics: which baseline settings reproduce the paper's Malenia / Ringleader curves", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_diagnostics.png"), dpi=130)
    plt.close(fig)


def fig_tuning(methods, fname):
    with open(os.path.join(RES, "mnist_summary.json")) as fh:
        S = json.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, setting in zip(axes, ("fixed", "exp")):
        for m in methods:
            if m not in S.get(setting, {}):
                continue
            tab = S[setting][m]["tuning_final_loss_seed0"]
            gs = np.array([float(k) for k in tab])
            vs = np.array([tab[k] for k in tab])
            vs = np.where(np.isfinite(vs), vs, np.nan)
            ax.plot(gs, vs, color=COL[m], marker=MARK[m], lw=1.5, ms=7, label=LAB[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("stepsize parameter γ")
        ax.set_title(SET[setting])
    axes[0].set_ylabel("final loss at T=30,000 (seed 0)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)


def fig2():
    z = np.load(os.path.join(RES, "schedule.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, mode in zip(axes, ("fixed", "exp")):
        for m in ("rescaled", "malenia", "ringleader"):
            t, c = z[f"{mode}_{m}_t"], z[f"{mode}_{m}_c"]
            ax.step(np.r_[0, t, 14.3], np.r_[0, c, c[-1] if len(c) else 0], where="post",
                    color=COL[m], lw=2, label=LAB[m] + (" (seed 0)" if mode == "exp" else ""))
            if mode == "exp":
                tg, q = z[f"{mode}_{m}_band_t"], z[f"{mode}_{m}_band_q"]
                ax.fill_between(tg, q[0], q[2], color=COL[m], alpha=0.15, lw=0, step="post")
                ax.plot(tg, q[1], color=COL[m], lw=1, ls=":", drawstyle="steps-post")
        if f"{mode}_ringleader_reuse_t" in z.files:
            t, c = z[f"{mode}_ringleader_reuse_t"], z[f"{mode}_ringleader_reuse_c"]
            ax.step(np.r_[0, t, 14.3], np.r_[0, c, c[-1]], where="post", color=COL["ringleader_reuse"],
                    lw=1.5, ls="--", label=LAB["ringleader_reuse"] + " [diagnostic]")
        ax.set_xlabel("Wall-clock time (× τmax)")
        ax.set_title({"fixed": "(a) Fixed",
                      "exp": "(b) Fluctuating: seed 0 (solid), median (dotted), 10-90% band of 100 seeds"}[mode],
                     fontsize=9)
        ax.set_xlim(0, 14.6)
        ax.set_ylim(0, 16)
    axes[0].set_ylabel("Cumulative step taken (× α)")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_cumulative_step.png"), dpi=130)
    plt.close(fig)


def fig3():
    z = np.load(os.path.join(RES, "toy_traces.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for r, (vname, vlab) in enumerate([("as_written", "as written: same cycle stepsize α = 0.01"),
                                       ("inferred_from_figure",
                                        "inferred from figure: Rescaled eq.(2) γ=0.01, Vanilla γ_i=0.01")]):
        for m in ("rescaled", "vanilla"):
            k = f"fig3_{vname}_{m}"
            axes[r, 0].plot(z[k + "__t"], z[k + "__x"], color=COL[m], lw=1.2, label=LAB[m])
            axes[r, 1].plot(z[k + "__cycle_t"], z[k + "__cycle_x"], color=COL[m], lw=1.8,
                            marker=MARK[m], markevery=15, ms=6, label=LAB[m])
        for c in (0, 1):
            ax = axes[r, c]
            ax.axhline(-2 / 3, color="k", lw=1, label="x* = −2/3")
            ax.axhline(0.5, color="k", lw=1, ls="--", label="x̃* = +1/2")
        axes[r, 0].set_title(f"all iterates — {vlab}", fontsize=9)
        axes[r, 1].set_title("cycle iterates", fontsize=9)
        axes[r, 0].set_ylabel("x")
    paper = {"t": [0, 30, 60, 90], "rescaled": [5, 2.39, 0.98, 0.22], "vanilla": [5, 1.74, 0.81, 0.56]}
    for m in ("rescaled", "vanilla"):
        axes[1, 1].plot(paper["t"], paper[m], ls="none", marker=MARK[m], ms=10, mfc="none", mec="k")
    axes[1, 1].plot([], [], ls="none", marker="o", mfc="none", mec="k", label="paper Fig. 3(b) (read off)")
    axes[0, 1].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    for c in (0, 1):
        axes[1, c].set_xlabel("Simulated wall-clock time")
    fig.suptitle("App. F.1 (Fig. 3): τ=(1,2), F1=(x−4)², F2=2(x+3)², x0=5, σ=0", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_vanilla_vs_rescaled.png"), dpi=130)
    plt.close(fig)


def fig4():
    z = np.load(os.path.join(RES, "toy_traces.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for m, ls in (("rescaled", "-"), ("delay_adaptive", "-"), ("delay_adaptive_mishchenko", ":"),
                  ("vanilla", "--")):
        k = f"fig4_{m}"
        lab = LAB[m] + (" [diagnostic, γ=0.01]" if m == "vanilla" else "")
        if m == "rescaled":
            axes[0].plot(z[k + "__cycle_t"], z[k + "__cycle_x"], color=COL[m], lw=2, marker="o", ms=5,
                         label=lab + " (cycle iterates)")
        else:
            axes[0].plot(z[k + "__t"], z[k + "__x"], color=COL[m], lw=1.6, ls=ls, label=lab)
        if m != "rescaled":
            axes[1].plot(z[k + "__t"], z[k + "__x"], color=COL[m], lw=1.4, ls=ls, label=lab)
    paper_t = [0, 100, 200, 300, 400, 500, 1500]
    paper_x = [3, 1.85, 1.13, 0.68, 0.41, 0.24, -0.04]
    axes[0].plot(paper_t, paper_x, ls="none", marker="o", ms=10, mfc="none", mec="k",
                 label="paper Fig. 4 Rescaled (read off)")
    for ax in axes:
        ax.axhline(0, color="k", lw=1)
        ax.axhline(1, color="k", lw=1, ls="--")
        ax.set_xlabel("Simulated wall-clock time")
    axes[0].set_ylabel("x")
    axes[0].set_title("App. F.2 (Fig. 4): τ=(1,100), F1=(x−1)², F2=(x+1)², x0=3", fontsize=9)
    axes[1].set_xlim(0, 450)
    axes[1].set_ylim(0.9, 1.35)
    axes[1].set_title("zoom: the paper's curve shows dips of ≈0.045 at every slow update", fontsize=9)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_delay_adaptive.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    if os.path.exists(os.path.join(RES, "mnist_summary.json")):
        fig1(["rescaled", "malenia", "ringleader"], "fig1_mnist.png")
        fig1_diag()
        fig_tuning(["rescaled", "malenia", "ringleader"], "fig1_tuning.png")
    if os.path.exists(os.path.join(RES, "schedule.npz")):
        fig2()
    if os.path.exists(os.path.join(RES, "toy_traces.npz")):
        fig3()
        fig4()


if __name__ == "__main__":
    main()
