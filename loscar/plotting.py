"""Shared matplotlib styling: fixed colour/marker per method (validated palette), ordinal ramp for
swept parameters, mean +- std bands over seeds."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .core import LABELS  # noqa: E402

COLORS = {
    "local_sparse": "#2a78d6",
    "overlap_dc": "#eb6834",
    "overlap_overwrite": "#1baf7a",
    "fedavg_full": "#eda100",
    "sync_sgd": "#e87ba4",
}
MARKERS = {"local_sparse": "o", "overlap_dc": "s", "overlap_overwrite": "^", "fedavg_full": "D",
           "sync_sgd": "v"}
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp, light -> dark

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9, "axes.titlesize": 9.5,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
    "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
})


def band(ax, x, Y, color, label, marker="o", log_x=False, shade=True, ms=None, **kw):
    """Plot mean of Y (seeds x points) against x with a +-1 std band."""
    Y = np.asarray(Y, float)
    mu, sd = Y.mean(0), Y.std(0)
    x = np.asarray(x, float)
    if log_x:  # log(0) undefined: drop the t = 0 / bits = 0 point like the paper does
        keep = x > 0
        x, mu, sd = x[keep], mu[keep], sd[keep]
    if shade and Y.shape[0] > 1:
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.16, lw=0)
    ax.plot(x, mu, color=color, marker=marker, label=label, ms=ms, **kw)
    if log_x:
        ax.set_xscale("log")


def method_band(ax, z, key, method, xkey="round", ykey="train_loss", log_x=False, label=None, **kw):
    band(ax, z[f"{key}::{xkey}"][0], z[f"{key}::{ykey}"], COLORS[method], label or LABELS[method],
         marker=MARKERS[method], log_x=log_x, **kw)
