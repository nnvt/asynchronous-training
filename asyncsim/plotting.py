import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots 1-3 (all-pairs safe), light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"ringmaster": "#2a78d6", "ringmaster_nostop": "#2a78d6", "rennala": "#eb6834",
          "da_asgd": "#1baf7a", "asgd": "#6250d6"}
NAMES = {"ringmaster": "Ringmaster ASGD (Alg. 5)", "ringmaster_nostop": "Ringmaster ASGD (Alg. 4, no stops)",
         "rennala": "Rennala SGD", "da_asgd": "Delay-Adaptive ASGD", "asgd": "Asynchronous SGD"}
MARKERS = {"ringmaster": "*", "rennala": "h", "da_asgd": "^"}
ORDER = ["ringmaster", "ringmaster_nostop", "rennala", "da_asgd", "asgd"]


def fmt_params(p):
    out = []
    g = p.get("gamma")
    if g is not None:
        out.append(f"γ={g:g}")
    for k in ("R", "B"):
        if k in p:
            out.append(f"{k}={p[k]}")
    return ", ".join(out)


def plot_runs(runs, path, title, ylabel=r"$f(x^t) - f^{\inf}$", paper_points=None, ylim=None):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    labels = sorted(runs, key=lambda lb: ORDER.index(runs[lb]["method"]))
    for label in labels:
        o = runs[label]
        m = o["method"]
        g = np.where(np.isfinite(o["gaps"]), o["gaps"], np.nan)
        g = np.where(g > 0, g, np.nan)
        with np.errstate(all="ignore"):
            med = np.exp(np.nanmedian(np.log(g), axis=0))
            lo, hi = np.nanmin(g, axis=0), np.nanmax(g, axis=0)
        ls = "--" if m == "ringmaster_nostop" else "-"
        n_div = int(np.sum(o["diverged"]))
        extra = f" — diverged {n_div}/{len(o['diverged'])}" if n_div else ""
        ax.plot(o["t"], med, ls, color=COLORS[m], lw=2, label=f"{NAMES[m]}: {fmt_params(o['params'])}{extra}")
        if g.shape[0] > 1:
            ax.fill_between(o["t"], lo, hi, color=COLORS[m], alpha=0.15, lw=0)
    if paper_points:
        first = True
        for m, pts in paper_points.items():
            ts, vs = zip(*sorted(pts.items()))
            ax.plot(ts, vs, MARKERS.get(m, "o"), ms=9, mfc="none", mew=1.5, color=COLORS[m],
                    label="paper figure (read off by eye)" if first else None)
            first = False
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, max(runs[lb]["t"][-1] for lb in labels))
    ax.set_xlabel("Simulated runtime (seconds)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    leg = ax.legend(fontsize=8, frameon=True, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
