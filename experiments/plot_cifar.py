"""Figures for the scaled-down CIFAR-10 experiment (paper Figs. 12-13)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from loscar.plotting import COLORS, LABELS, MARKERS, plt  # noqa: E402

METHODS = ("local_sparse", "overlap_dc", "overlap_overwrite")
TITLES = {"normal": "normal regime (p=0.25, M=4, ζ=6)", "stress": "communication stress (p=0.10, M=2, ζ=24)"}


def main():
    z = np.load(os.path.join(ROOT, "results", "cifar10.npz"))
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6))
    for i, reg in enumerate(("normal", "stress")):
        for m in METHODS:
            k = f"{reg}|{m}"
            if f"{k}::time" not in z.files:
                continue
            kw = dict(color=COLORS[m], marker=MARKERS[m], ms=3, label=LABELS[m])
            t, ex, r = z[f"{k}::time"][0], z[f"{k}::examples"][0], z[f"{k}::round"][0]
            axs[i, 0].plot(t, z[f"{k}::train_acc"][0], **kw)
            axs[i, 1].plot(t, z[f"{k}::val_acc"][0], **kw)
            axs[i, 2].plot(r, z[f"{k}::train_acc"][0], **kw)
            axs[i, 3].plot(ex[1:], z[f"{k}::train_acc"][0][1:], **kw)
        axs[i, 3].set_xscale("log")
        for j, (xl, yl) in enumerate([("Cumulative logical time", "Train accuracy"),
                                      ("Cumulative logical time", "Val accuracy"),
                                      ("Round", "Train accuracy"),
                                      ("Cumulative processed examples", "Train accuracy")]):
            axs[i, j].set(xlabel=xl, ylabel=yl)
            axs[i, j].set_title(f"CIFAR-10, {TITLES[reg]}", fontsize=8.5)
            axs[i, j].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig12_13_cifar10.png"))
    print("wrote fig12_13_cifar10.png")


if __name__ == "__main__":
    main()
