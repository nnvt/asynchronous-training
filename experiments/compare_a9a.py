"""Print paper-vs-ours tables for the a9a experiments.

Paper values were read off the rendered figures of arXiv:2605.20866v1 (embedded 900x500 PNGs,
extracted with pdfimages); reading precision is roughly +-0.003 in loss and +-0.005 in accuracy.
Ours: mean +- std over 10 seeds.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LS, OW, DC = "local_sparse", "overlap_overwrite", "overlap_dc"
SHORT = {LS: "LS", OW: "OW", DC: "DC"}

# (experiment, key, metric, eval index, paper value, figure)
ROWS = [
    ("e1_main", LS, "train_loss", 1, 0.552, "1a/4a"), ("e1_main", OW, "train_loss", 1, 0.531, "1a/4a"),
    ("e1_main", DC, "train_loss", 1, 0.525, "1a/4a"), ("e1_main", LS, "train_loss", 20, 0.336, "1a/4a"),
    ("e1_main", OW, "train_loss", 20, 0.332, "1a/4a"), ("e1_main", DC, "train_loss", 20, 0.331, "1a/4a"),
    ("e1_main", LS, "val_loss", 20, 0.348, "4b"), ("e1_main", OW, "val_loss", 20, 0.345, "4b"),
    ("e1_main", DC, "val_loss", 20, 0.343, "4b"),
    ("e1_main", LS, "val_acc", 1, 0.740, "2d"), ("e1_main", OW, "val_acc", 1, 0.754, "2d"),
    ("e1_main", DC, "val_acc", 1, 0.760, "2d"), ("e1_main", DC, "val_acc", 20, 0.843, "2c"),
    ("e1_main", LS, "grad_norm", 20, 0.038, "3"), ("e1_main", OW, "grad_norm", 20, 0.031, "3"),
    ("e1_main", DC, "grad_norm", 20, 0.028, "3"),
    ("e2_long_compute", OW, "train_loss", 1, 0.453, "5a"), ("e2_long_compute", DC, "train_loss", 1, 0.450, "5a"),
    ("e2_long_compute", OW, "train_loss", 20, 0.324, "5a"), ("e2_long_compute", DC, "train_loss", 20, 0.324, "5a"),
    ("e2_long_delay", OW, "train_loss", 1, 0.510, "6a/1c"), ("e2_long_delay", DC, "train_loss", 1, 0.487, "6a/1c"),
    ("e2_long_delay", OW, "train_loss", 20, 0.330, "6a/1c"), ("e2_long_delay", DC, "train_loss", 20, 0.327, "6a/1c"),
    ("e2_long_delay", OW, "val_loss", 1, 0.517, "6b"), ("e2_long_delay", DC, "val_loss", 1, 0.495, "6b"),
    ("e2_hetero_speed", OW, "train_loss", 1, 0.485, "7a"), ("e2_hetero_speed", DC, "train_loss", 1, 0.476, "7a"),
    ("e2_hetero_speed", OW, "train_loss", 20, 0.326, "7a"), ("e2_hetero_speed", DC, "train_loss", 20, 0.326, "7a"),
    ("e3_sparsity", f"{LS}|p=0.001", "train_loss", 1, 0.525, "8a"),
    ("e3_sparsity", f"{LS}|p=0.001", "train_loss", 20, 0.333, "8a"),
    ("e3_sparsity", f"{LS}|p=1.0", "train_loss", 20, 0.330, "8a"),
    ("e3_sparsity", f"{OW}|p=0.1", "train_loss", 1, 0.507, "8b"),
    ("e3_sparsity", f"{OW}|p=1.0", "train_loss", 1, 0.525, "8b"),
    ("e3_sparsity", f"{DC}|p=0.001", "train_loss", 1, 0.504, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=0.001", "train_loss", 20, 0.330, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=1.0", "train_loss", 20, 0.327, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=0.001", "bits", 1, 287, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=0.01", "bits", 1, 574, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=0.1", "bits", 1, 3.72e3, "8c/1b"),
    ("e3_sparsity", f"{DC}|p=1.0", "bits", 1, 3.57e4, "8c/1b"),
    ("e4_budget", f"{LS}|M=1", "train_loss", 1, 0.641, "9a"), ("e4_budget", f"{LS}|M=1", "train_loss", 20, 0.382, "9a"),
    ("e4_budget", f"{LS}|M=16", "train_loss", 1, 0.405, "9a"), ("e4_budget", f"{LS}|M=64", "train_loss", 20, 0.322, "9a"),
    ("e4_budget", f"{OW}|M=1", "train_loss", 1, 0.601, "9b"), ("e4_budget", f"{OW}|M=1", "train_loss", 20, 0.357, "9b"),
    ("e4_budget", f"{DC}|M=1", "train_loss", 1, 0.588, "9c"), ("e4_budget", f"{DC}|M=1", "train_loss", 20, 0.348, "9c"),
    ("e4_budget", f"{DC}|M=4", "train_loss", 1, 0.503, "9c"),
    ("e5_delay", f"{LS}|zeta=12", "train_loss", 1, 0.461, "10a"), ("e5_delay", f"{OW}|zeta=12", "train_loss", 1, 0.446, "10a"),
    ("e5_delay", f"{DC}|zeta=12", "train_loss", 1, 0.441, "10a"),
    ("e5_delay", f"{LS}|zeta=48", "train_loss", 1, 0.461, "10b"), ("e5_delay", f"{OW}|zeta=48", "train_loss", 1, 0.418, "10b"),
    ("e5_delay", f"{DC}|zeta=48", "train_loss", 1, 0.405, "10b"),
    ("e5_delay", f"{LS}|zeta=48", "train_loss", 2, 0.403, "10b"), ("e5_delay", f"{OW}|zeta=48", "train_loss", 2, 0.374, "10b"),
    ("e5_delay", f"{DC}|zeta=48", "train_loss", 2, 0.362, "10b"),
    ("e6_noniid", LS, "train_acc", 1, 0.665, "11a"), ("e6_noniid", OW, "train_acc", 1, 0.776, "11a"),
    ("e6_noniid", DC, "train_acc", 1, 0.760, "11a"),
    ("e6_noniid", LS, "train_acc", 40, 0.791, "11a"), ("e6_noniid", OW, "train_acc", 40, 0.778, "11a"),
    ("e6_noniid", DC, "train_acc", 40, 0.742, "11a"),
    ("e6_noniid", LS, "val_acc", 40, 0.786, "11b"), ("e6_noniid", OW, "val_acc", 40, 0.774, "11b"),
    ("e6_noniid", DC, "val_acc", 40, 0.736, "11b"),
]


def main():
    cache = {}
    print("| Fig. | experiment | method | metric @ round | paper (read off) | ours (mean ± std, 10 seeds) |")
    print("|---|---|---|---|---|---|")
    for exp, key, metric, r, paper, fig in ROWS:
        if exp not in cache:
            cache[exp] = np.load(os.path.join(ROOT, "results", f"a9a_{exp}.npz"))
        z = cache[exp]
        rr = list(z[f"{key}::round"][0]).index(r)
        v = z[f"{key}::{metric}"][:, rr]
        m, extra = key.split("|")[0], ("|" + key.split("|")[1]) if "|" in key else ""
        if metric == "bits":
            ours, pv = f"{v.mean():.0f}", f"≈{paper:.3g}"
        else:
            ours, pv = f"{v.mean():.4f} ± {v.std():.4f}", f"{paper:.3f}"
        print(f"| {fig} | {exp} | {SHORT[m]}{extra} | {metric} @ {r} | {pv} | {ours} |")


if __name__ == "__main__":
    main()
