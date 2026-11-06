"""Figures and summary tables."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiment import BUDGET, load_results  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
# validated categorical slots 1-3 (all-pairs safe on the light surface)
COLORS = {"ringleader": "#2a78d6", "malenia": "#eb6834", "ia2sgd": "#1baf7a"}
NAMES = {"ringleader": "Ringleader ASGD", "malenia": "Malenia SGD", "ia2sgd": r"IA$^2$SGD"}
MARKERS = {"ringleader": "*", "malenia": "h", "ia2sgd": "^"}
ORDER = ["ringleader", "malenia", "ia2sgd"]
TITLES = {"mnist": "(a) MNIST", "fmnist": "(b) Fashion-MNIST"}


def smooth(t, y, width):
    """Centered moving average over a time window `width` (post-aggregation);
    the initial point is kept unchanged, as in the paper."""
    out = np.array(y, dtype=float)
    for i in range(1, len(t)):
        m = np.abs(t - t[i]) <= width / 2
        out[i] = np.mean(y[m])
    return out


def aggregate(runs, metric="g2"):
    g = np.array([r[metric] for r in runs], dtype=float)
    g = np.where(np.isfinite(g), g, np.nan)
    with np.errstate(all="ignore"):
        q = np.nanpercentile(g, [25, 50, 75], axis=0)
    return runs[0]["t"], q


def tail_mean(t, g2, T, frac=0.1):
    m = t >= (1 - frac) * T - 1e-9
    v = g2[m]
    if not np.all(np.isfinite(v)):
        return np.inf
    return float(np.mean(np.log10(v)))


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def load_paper(res_dir, ds):
    p = os.path.join(res_dir, "paper", f"fig1_{ds}_digitized.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


SMOOTH_FRAC = 0.08  # moving-average window as a fraction of the budget (paper: unspecified)


def plot_fig1(res_dir, path, suffix="", title=None, smooth_frac=SMOOTH_FRAC, show_paper=True, metric="g2"):
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    any_ok = False
    for ax, ds in zip(axes, ["mnist", "fmnist"]):
        f = os.path.join(res_dir, f"final_{ds}{suffix}.npz")
        _style(ax)
        ax.set_title(TITLES[ds], color=INK, fontsize=10)
        if not os.path.exists(f):
            ax.text(0.5, 0.5, "not run", ha="center", transform=ax.transAxes, color=INK2)
            continue
        any_ok = True
        runs = load_results(f)
        T = BUDGET[ds] * (runs[0]["T"] / BUDGET[ds])
        paper = load_paper(res_dir, ds) if show_paper else None
        for m in ORDER:
            rs = [r for r in runs if r["method"] == m]
            if not rs:
                continue
            t, (q25, q50, q75) = aggregate(rs, metric)
            w = smooth_frac * T
            s25, s50, s75 = (smooth(t, q, w) for q in (q25, q50, q75))
            c = COLORS[m]
            lab = f"{NAMES[m]}  (γ={rs[0]['gamma']:g}, {len(rs)} seeds)"
            ax.fill_between(t, s25, s75, color=c, alpha=0.16, lw=0)
            ax.plot(t, s50, "-", color=c, lw=2, label=lab, marker=MARKERS[m], markevery=max(1, len(t) // 12),
                    ms=8 if m == "ringleader" else 6, mec=SURFACE, mew=0.8)
            if paper and m in paper and metric == "g2":
                pt, pv = np.array(paper[m]["t"]), np.array(paper[m]["v"])
                ax.plot(pt, pv, "--", color=c, lw=1.1, alpha=0.85,
                        label=f"{NAMES[m]}  paper Fig. 1 (digitized median)")
        ax.set_yscale("log")
        ax.set_xlim(0, T)
        ax.set_xlabel("simulated wall-clock time", color=INK)
        if metric == "g2":
            ax.set_ylabel(r"$\|\nabla f(x^t)\|^2$ (full batch)", color=INK)
        else:
            ax.axhline(np.log(10), color=INK2, lw=0.8, ls=":")
            ax.annotate("ln 10 (uniform prediction)", xy=(0.02 * T, np.log(10) * 1.06), ha="left", fontsize=7,
                        color=INK2)
            ax.set_ylabel(r"training loss $f(x^t)$ (full batch, mean CE)", color=INK)
        leg = ax.legend(fontsize=7, frameon=True, loc="upper right", framealpha=0.92)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
    if not any_ok:
        plt.close(fig)
        return
    fig.suptitle(title or "Fig. 1 replication: median (line) and IQR (band) over seeds, "
                 f"centered moving average over {smooth_frac:.0%} of the budget", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_tuning(res_dir, path, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    any_ok = False
    for ax, ds in zip(axes, ["mnist", "fmnist"]):
        f = os.path.join(res_dir, f"tune_{ds}{suffix}.npz")
        _style(ax)
        ax.set_title(TITLES[ds], color=INK, fontsize=10)
        if not os.path.exists(f):
            continue
        any_ok = True
        runs = load_results(f)
        T = runs[0]["T"]
        for m in ORDER:
            rs = sorted([r for r in runs if r["method"] == m], key=lambda r: r["gamma"])
            gs = sorted({r["gamma"] for r in rs})
            sc = [np.mean([tail_mean(r["t"], r["g2"], T) for r in rs if r["gamma"] == g]) for g in gs]
            sc = np.array(sc)
            fin = np.isfinite(sc)
            ax.plot(np.array(gs)[fin], sc[fin], "-", marker=MARKERS[m], color=COLORS[m], lw=2, ms=7,
                    label=NAMES[m])
            div = [f"{g:g}" for g in np.array(gs)[~fin]]
            if div:
                ax.annotate(f"{NAMES[m]} diverged: γ ∈ {{{', '.join(div)}}}", xy=(0.02, 0.03 + 0.07 * ORDER.index(m)),
                            xycoords="axes fraction", fontsize=7, color=INK2)
            if fin.any():
                b = int(np.nanargmin(np.where(fin, sc, np.nan)))
                ax.plot([gs[b]], [sc[b]], "o", mfc="none", mec=INK, ms=12, mew=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(r"stepsize $\gamma$", color=INK)
        ax.set_ylabel(r"mean $\log_{10}\|\nabla f\|^2$ over last 10% of budget", color=INK)
        leg = ax.legend(fontsize=8)
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
    if not any_ok:
        plt.close(fig)
        return
    fig.suptitle("Stepsize sweep on the paper's grid (seed 0, full budget); circle = selected γ", color=INK,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def summary_table(res_dir, suffix=""):
    lines = ["| dataset | method | γ | seeds | median ‖∇f‖² at T/4 | at T/2 | at T | tail (mean log10, last 10%) "
             "| paper tail (digitized) | median final train loss | median #updates | diverged |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ["mnist", "fmnist"]:
        f = os.path.join(res_dir, f"final_{ds}{suffix}.npz")
        if not os.path.exists(f):
            continue
        runs = load_results(f)
        T = runs[0]["T"]
        paper = load_paper(res_dir, ds)
        for m in ORDER:
            rs = [r for r in runs if r["method"] == m]
            if not rs:
                continue
            t, (q25, q50, q75) = aggregate(rs)
            at = [q50[np.argmin(np.abs(t - fr * T))] for fr in (0.25, 0.5, 1.0)]
            tails = [tail_mean(r["t"], r["g2"], T) for r in rs]
            ptail = ""
            if paper and m in paper:
                pt, pv = np.array(paper[m]["t"]), np.array(paper[m]["v"])
                ptail = f"{np.mean(np.log10(pv[pt >= 0.9 * T])):.2f}"
            lines.append(f"| {ds} | {NAMES[m].replace('$^2$', '²')} | {rs[0]['gamma']:g} | {len(rs)} | "
                         + " | ".join(f"{v:.2e}" for v in at)
                         + f" | {np.median(tails):.2f} [{np.percentile(tails, 25):.2f}, {np.percentile(tails, 75):.2f}]"
                         + f" | {ptail} | {np.median([r['loss'][-1] for r in rs]):.3f}"
                         + f" | {int(np.median([r['iters'] for r in rs]))} | "
                         + f"{sum(bool(r['diverged']) for r in rs)} |")
    return "\n".join(lines)


def first_below(t, y, thr):
    i = np.nonzero(y <= thr)[0]
    return float(t[i[0]]) if len(i) else np.inf


THRESH = {"mnist": (2e-3, 5e-4), "fmnist": (1e-2, 2e-3)}


def threshold_table(res_dir, suffix="", smooth_frac=SMOOTH_FRAC):
    """First time the smoothed median curve drops below fixed levels (ours vs. digitized paper curve,
    the paper curve smoothed the same way on its own pixel grid)."""
    lines = ["| dataset | method | level | ours: first t with smoothed median ≤ level | paper Fig. 1 |",
             "|---|---|---|---|---|"]
    for ds in ["mnist", "fmnist"]:
        f = os.path.join(res_dir, f"final_{ds}{suffix}.npz")
        if not os.path.exists(f):
            continue
        runs = load_results(f)
        T = runs[0]["T"]
        paper = load_paper(res_dir, ds)
        for thr in THRESH[ds]:
            for m in ORDER:
                rs = [r for r in runs if r["method"] == m]
                t, (q25, q50, q75) = aggregate(rs)
                ours = first_below(t, smooth(t, q50, smooth_frac * T), thr)
                pt, pv = np.array(paper[m]["t"]), np.array(paper[m]["v"])
                theirs = first_below(pt, pv, thr)
                fmt = lambda v: "never" if not np.isfinite(v) else f"{v / 1000:.0f}k"  # noqa: E731
                lines.append(f"| {ds} | {NAMES[m].replace('$^2$', '²')} | {thr:g} | {fmt(ours)} | {fmt(theirs)} |")
    return "\n".join(lines)


def make_all(res_dir, fig_dir):
    suffixes = sorted({os.path.basename(p)[len("final_mnist"):-4] for p in glob.glob(os.path.join(res_dir, "final_mnist*.npz"))}
                      | {os.path.basename(p)[len("final_fmnist"):-4] for p in glob.glob(os.path.join(res_dir, "final_fmnist*.npz"))})
    for suf in suffixes:
        lab = "" if not suf else f"Diagnostic '{suf.strip('_')}': "
        plot_fig1(res_dir, os.path.join(fig_dir, f"fig1{suf}.png"), suffix=suf,
                  title=None if not suf else lab + f"median (line) and IQR (band) over seeds, moving average over {SMOOTH_FRAC:.0%} of the budget")
        plot_fig1(res_dir, os.path.join(fig_dir, f"fig1{suf}_loss.png"), suffix=suf, metric="loss",
                  title=lab + "training loss of the same runs (median and IQR over seeds, same smoothing)")
        print(f"--- final{suf}\n" + summary_table(res_dir, suf))
        print(threshold_table(res_dir, suf))
    tsuf = sorted({os.path.basename(p)[len("tune_mnist"):-4] for p in glob.glob(os.path.join(res_dir, "tune_mnist*.npz"))})
    for suf in tsuf:
        plot_tuning(res_dir, os.path.join(fig_dir, f"tuning{suf}.png"), suffix=suf)
