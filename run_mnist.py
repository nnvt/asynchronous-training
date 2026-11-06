"""Fig. 1 of arXiv:2605.13434: label-partitioned MNIST, n = 10 workers,
tau in {1,2,4,8,16} (two workers each), fixed or Exp(mean tau_i) times,
budget T = 30,000, stepsize grid {1e-4,...,1} tuned on final loss (seed 0),
then 5 seeds (0..4) at the selected stepsize.

usage: python run_mnist.py [--procs 5] [--methods rescaled,malenia,ringleader,...]
                           [--settings fixed,exp] [--T 30000]
Results: results/mnist/<setting>_<method>_g<gamma>_s<seed>.npz and
results/mnist_summary.json
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from rasgd.mnist import LabelPartitionedMLP  # noqa: E402
from rasgd.sim import METHODS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "mnist")
TAUS = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]  # worker i holds digit i
GAMMAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [0, 1, 2, 3, 4]

_P = None


def _problem():
    global _P
    if _P is None:
        _P = LabelPartitionedMLP()
    return _P


def fname(setting, method, gamma, seed):
    return os.path.join(OUT, f"{setting}_{method}_g{gamma:g}_s{seed}.npz")


def run_one(task):
    setting, method, gamma, seed, T = task
    path = fname(setting, method, gamma, seed)
    if os.path.exists(path):
        return task, "cached"
    P = _problem()
    grid = np.linspace(0.0, T, 201)
    t0 = time.time()
    r = METHODS[method](P, TAUS, setting, gamma, T, grid, seed, log_steps=False)
    wall = time.time() - t0
    np.savez_compressed(path + ".tmp.npz", grid=grid, loss=r["loss"].astype(np.float32),
                        diverged=r["diverged"], n_updates=r["n_updates"], n_grads=r["n_grads"],
                        wall=wall)
    os.replace(path + ".tmp.npz", path)
    return task, f"{wall:.0f}s final={r['loss'][-1]:.4g} upd={r['n_updates']} grads={r['n_grads']}"


def load(setting, method, gamma, seed):
    z = np.load(fname(setting, method, gamma, seed))
    return {k: z[k] for k in z.files}


def final_loss(setting, method, gamma, seed):
    v = load(setting, method, gamma, seed)["loss"][-1]
    return float(v) if np.isfinite(v) else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--methods", default="rescaled,malenia,ringleader,vanilla,delay_adaptive")
    ap.add_argument("--settings", default="fixed,exp")
    ap.add_argument("--T", type=float, default=30000.0)
    ap.add_argument("--gammas", default=",".join(f"{g:g}" for g in GAMMAS),
                    help="tuning grid (default: the paper's {1e-4,...,1})")
    args = ap.parse_args()
    grid_g = [float(g) for g in args.gammas.split(",")]
    os.makedirs(OUT, exist_ok=True)
    methods = args.methods.split(",")
    settings = args.settings.split(",")
    pool = mp.get_context("fork").Pool(args.procs)

    def run(tasks):
        # longest first (ASGD-type methods do one gradient per event)
        for task, msg in pool.imap_unordered(run_one, tasks):
            print(task[:4], msg, flush=True)

    # stage 1: tuning on seed 0
    run([(s, m, g, 0, args.T) for s in settings for m in methods for g in grid_g])
    summary = {}
    if os.path.exists(os.path.join(ROOT, "results", "mnist_summary.json")):
        with open(os.path.join(ROOT, "results", "mnist_summary.json")) as fh:
            summary = json.load(fh)
    best = {}
    for s in settings:
        for m in methods:
            finals = {g: final_loss(s, m, g, 0) for g in grid_g}
            g_best = min(finals, key=finals.get)
            best[(s, m)] = g_best
            summary.setdefault(s, {})[m] = {"tuning_final_loss_seed0": {f"{g:g}": finals[g] for g in grid_g},
                                            "gamma": g_best}
    # stage 2: remaining seeds at the selected stepsize
    run([(s, m, best[(s, m)], sd, args.T) for s in settings for m in methods for sd in SEEDS[1:]])
    for s in settings:
        for m in methods:
            g = best[(s, m)]
            runs = [load(s, m, g, sd) for sd in SEEDS]
            L = np.stack([r["loss"] for r in runs])
            grid = runs[0]["grid"]
            entry = summary[s][m]
            entry["final_loss_per_seed"] = [float(x) for x in L[:, -1]]
            entry["final_loss_median"] = float(np.median(L[:, -1]))
            for tq in (2000, 10000, 20000):
                j = int(np.argmin(np.abs(grid - tq)))
                entry[f"median_loss_t{tq}"] = float(np.median(L[:, j]))
            entry["n_updates_median"] = float(np.median([r["n_updates"] for r in runs]))
            entry["n_grads_median"] = float(np.median([r["n_grads"] for r in runs]))
    with open(os.path.join(ROOT, "results", "mnist_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
