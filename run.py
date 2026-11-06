"""Reproduce the experiments of "Ringmaster ASGD" (arXiv:2501.16168, App. G).

Examples
--------
  python run.py --setup text --mode paper        # Fig. 2, legend hyper-params
  python run.py --setup text --mode tune         # Fig. 2, re-tuned as in the paper
  python run.py --setup fig2match --mode paper   # reverse-engineered Fig. 2 setup
  python run.py --setup mnist --mode paper       # Fig. 3
  python run.py --setup mnist --mode tune
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from asyncsim import experiment as E  # noqa: E402
from asyncsim.plotting import NAMES, fmt_params, plot_runs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["ringmaster", "ringmaster_nostop", "rennala", "da_asgd"]


def summarize(runs, T):
    checkpoints = [T / 10, T / 4, T / 2, T]
    lines = [f"{'method':40s} {'params':18s} " + " ".join(f"t={c:<7g}" for c in checkpoints) + "  iters(median)"]
    for label, o in runs.items():
        t = o["t"]
        g = np.where(np.isfinite(o["gaps"]) & (o["gaps"] > 0), o["gaps"], np.nan)
        with np.errstate(all="ignore"):
            med = np.exp(np.nanmedian(np.log(g), axis=0))
        vals = [med[np.searchsorted(t, c) if c < t[-1] else -1] for c in checkpoints]
        div = sum(o["diverged"])
        lines.append(f"{NAMES[o['method']]:40s} {fmt_params(o['params']):18s} "
                     + " ".join(f"{v:9.2e}" for v in vals)
                     + f"  {int(np.median(o['iters']))}" + (f"  diverged {div}/{len(o['diverged'])}" if div else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=sorted(E.SETUPS), default="text")
    ap.add_argument("--mode", choices=["paper", "tune"], default="paper")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS + ["asgd"])
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the final curves")
    ap.add_argument("--tune-seeds", type=int, default=2, help="seeds averaged per grid point when tuning")
    ap.add_argument("--criterion", choices=["final", "auc"], default="final",
                    help="tuning objective: final = mean log10 gap over last 5%% of the run; "
                         "auc = mean log10 gap over the whole run")
    ap.add_argument("--tune-points", type=int, default=400, help="time-grid points per tuning run")
    ap.add_argument("--points", type=int, default=1000, help="time-grid points per final run")
    ap.add_argument("--processes", type=int, default=None)
    args = ap.parse_args()

    setup = E.SETUPS[args.setup]
    family = setup["problem"]["kind"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    tag = f"{args.setup}_{args.mode}" + (f"-{args.criterion}" if args.mode == "tune" else "")
    t0 = time.time()

    if args.mode == "paper":
        configs = {m: (m, E.PAPER_HPARAMS[family][m]) for m in args.methods if m in E.PAPER_HPARAMS[family]}
    else:
        best, rows = E.tune(setup, args.methods, seeds=range(args.tune_seeds), processes=args.processes,
                            criterion=args.criterion, num_points=args.tune_points)
        with open(os.path.join(HERE, "results", f"{tag}_grid.json"), "w") as fh:
            json.dump(dict(best=best, grid=sorted(rows, key=lambda r: (r["method"], r["score"]))), fh, indent=1)
        print(f"best by criterion '{args.criterion}' (score = mean log10 gap):")
        for m, b in best.items():
            print(f"  {NAMES[m]:40s} {fmt_params(b['params']):20s} score={b['score']:.3f}")
        configs = {m: (m, best[m]["params"]) for m in args.methods if np.isfinite(best[m]["score"])}

    runs = E.multi_seed(setup, configs, seeds=range(args.seeds), processes=args.processes,
                        num_points=args.points)
    E.save_runs(os.path.join(HERE, "results", f"{tag}.npz"), runs)
    print(summarize(runs, setup["T"]))

    paper_pts = E.PAPER_POINTS[family] if args.mode == "paper" else None
    fig_no = "Fig. 3" if family == "mnist" else "Fig. 2"
    what = ("hyper-parameters from the paper's legend" if args.mode == "paper"
            else f"re-tuned on the paper's grid ({args.criterion} criterion)")
    title = f"{fig_no} replication — setup '{args.setup}', {what}\nmedian of {args.seeds} seeds, band = min–max"
    ylabel = r"training loss  $f(x^t)$  ($f^{\inf}=0$)" if family == "mnist" else r"$f(x^t) - f^{*}$"
    plot_runs(runs, os.path.join(HERE, "figures", f"{tag}.png"), title, ylabel=ylabel, paper_points=paper_pts)
    print(f"done in {time.time() - t0:.0f}s -> results/{tag}.npz, figures/{tag}.png")


if __name__ == "__main__":
    main()
