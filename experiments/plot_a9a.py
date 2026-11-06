"""Figures for the a9a experiments (paper Figs. 1-11). Reads results/a9a_*.npz, writes figures/."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from loscar.plotting import COLORS, LABELS, MARKERS, RAMP4, band, method_band, plt  # noqa: E402

LS, OW, DC = "local_sparse", "overlap_overwrite", "overlap_dc"
SPARSE3 = (LS, DC, OW)  # legend order as in the paper
FIG = os.path.join(ROOT, "figures")
YL = {"train_loss": "Train loss", "val_loss": "Val loss", "train_acc": "Train accuracy",
      "val_acc": "Val accuracy", "grad_norm": "Train gradient norm"}
XL = {"round": "Round", "time": "Cumulative logical time", "bits": "Cumulative logical total bits"}


def load(exp):
    return np.load(os.path.join(ROOT, "results", f"a9a_{exp}.npz"))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("wrote", name)


def panel(ax, z, methods, xkey, ykey, log_x=False, log_y=False, title=None, keyfmt="{m}"):
    for m in methods:
        method_band(ax, z, keyfmt.format(m=m), m, xkey=xkey, ykey=ykey, log_x=log_x)
    ax.set_xlabel(XL[xkey])
    ax.set_ylabel(YL[ykey])
    if log_y:
        ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.legend()


def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    panel(axs[0], load("e1_main"), SPARSE3, "round", "train_loss",
          title="(a) Overlap comparison (M=3, ζ=6, p=0.3)")
    z = load("e3_sparsity")
    for c, p in zip(RAMP4, (0.001, 0.01, 0.1, 1.0)):
        band(axs[1], z[f"{DC}|p={p}::bits"][0], z[f"{DC}|p={p}::train_loss"], c, f"p = {p}",
             log_x=True)
    axs[1].set(xlabel=XL["bits"], ylabel=YL["train_loss"],
               title="(b) Sparsity vs. communication (delay-corrected)")
    axs[1].legend()
    panel(axs[2], load("e2_long_delay"), (DC, OW), "round", "train_loss", log_x=True,
          title="(c) Long-delay merge rule (M=2, ζ=24)")
    save(fig, "fig1_summary.png")


def fig2_3_4():
    z = load("e1_main")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for i, yk in enumerate(("train_acc", "val_acc")):
        panel(axs[i, 0], z, SPARSE3, "round", yk, title=f"{YL[yk]} vs round")
        panel(axs[i, 1], z, SPARSE3, "round", yk, log_x=True, title=f"{YL[yk]} vs round, log scale")
    save(fig, "fig2_accuracy.png")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    panel(ax, z, SPARSE3, "round", "grad_norm", log_y=True, title="Training gradient norm ‖∇F(x̄)‖")
    save(fig, "fig3_gradnorm.png")
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    for j, yk in enumerate(("train_loss", "val_loss")):
        panel(axs[0, j], z, SPARSE3, "time", yk, log_x=True, title=f"{YL[yk]} vs logical time")
        panel(axs[1, j], z, SPARSE3, "bits", yk, log_x=True, title=f"{YL[yk]} vs communication cost")
    save(fig, "fig4_resources.png")
    # extra (not in the paper): all five methods of App. B on the same logical-time budget
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, yk in zip(axs, ("train_loss", "val_acc")):
        panel(ax, z, ("sync_sgd", "fedavg_full", LS, OW, DC), "time", yk,
              title=f"All five methods: {YL[yk]} (t ≤ 480)")
    save(fig, "extra_five_methods.png")


def fig5_7():
    regs = [("e2_long_compute", "Long compute, short delay (M=8, ζ=6, τ=(1,2,3,6))"),
            ("e2_long_delay", "Short compute, long delay (M=2, ζ=24, τ=(1,2,3,6))"),
            ("e2_hetero_speed", "Very heterogeneous speeds (M=2, ζ=20, τ=(1,2,10,20))")]
    fig, axs = plt.subplots(3, 2, figsize=(10, 9))
    for i, (e, t) in enumerate(regs):
        z = load(e)
        for j, yk in enumerate(("train_loss", "val_loss")):
            panel(axs[i, j], z, (DC, OW), "round", yk, log_x=True, title=f"{t}\n{YL[yk]}")
    save(fig, "fig5_7_merge_rule.png")


def fig8():
    z = load("e3_sparsity")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axs, (LS, OW, DC)):
        for c, p in zip(RAMP4, (0.001, 0.01, 0.1, 1.0)):
            band(ax, z[f"{m}|p={p}::bits"][0], z[f"{m}|p={p}::train_loss"], c, f"p = {p}", log_x=True)
        ax.set(xlabel=XL["bits"], title=LABELS[m] + " (M=4, ζ=6)")
        ax.legend()
    axs[0].set_ylabel(YL["train_loss"])
    save(fig, "fig8_sparsity.png")


def fig9():
    z = load("e4_budget")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, m in zip(axs, (LS, OW, DC)):
        for c, M in zip(RAMP4, (1, 4, 16, 64)):
            band(ax, z[f"{m}|M={M}::time"][0], z[f"{m}|M={M}::train_loss"], c, f"M={M}", log_x=True,
                 ms=3.5)
        ax.set(xlabel=XL["time"], title=LABELS[m] + " (ζ=6, p=0.3)")
        ax.legend(title="M")
    axs[0].set_ylabel(YL["train_loss"])
    save(fig, "fig9_budget.png")


def fig10():
    z = load("e5_delay")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, zeta in zip(axs, (12, 48)):
        panel(ax, z, SPARSE3, "time", "train_loss", log_x=True, keyfmt="{m}|zeta=%d" % zeta,
              title=f"{'Moderate' if zeta == 12 else 'Large'} delay, ζ={zeta} (M=8, p=0.3)")
    save(fig, "fig10_delay.png")


def fig11():
    z = load("e6_noniid")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, yk in zip(axs, ("train_acc", "val_acc")):
        panel(ax, z, SPARSE3, "time", yk,
              title=f"Dirichlet(0.03) split, M=1, ζ=96, p=0.01, η=0.3: {YL[yk]}")
    save(fig, "fig11_noniid.png")


if __name__ == "__main__":
    fig1()
    fig2_3_4()
    fig5_7()
    fig8()
    fig9()
    fig10()
    fig11()
