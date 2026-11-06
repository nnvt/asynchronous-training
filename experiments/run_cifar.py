"""Scaled-down CIFAR-10 experiment (paper App. B.6, Figs. 12-13).

Two regimes x three sparse methods, single seed (11) as in the paper; logical-time budget scaled
down from ~35000 to --budget (default 3000). Usage:
    python experiments/run_cifar.py [--budget 3000] [--procs 5] [--threads 2]
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TAUS = (1, 2, 3, 6)
REGIMES = {"normal": dict(p=0.25, M=4, zeta=6), "stress": dict(p=0.10, M=2, zeta=24)}
METHODS = ("overlap_dc", "overlap_overwrite", "local_sparse")


def _task(args):
    regime, method, budget, threads, seed, eval_every = args
    import torch
    torch.set_num_threads(threads)
    from loscar.cifar import Cifar
    from loscar.core import run
    t0 = time.time()
    prob = Cifar(len(TAUS), seed=seed)
    c = REGIMES[regime]
    T = c["M"] * 6 + c["zeta"]
    rounds = budget // T
    h = run(prob, method, taus=list(TAUS), M=c["M"], zeta=c["zeta"], p=c["p"], rounds=rounds,
            seed=seed, eval_every=eval_every, verbose=True)
    return regime, method, h, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--eval_every", type=int, default=4)
    a = ap.parse_args()
    tasks = [(r, m, a.budget, a.threads, a.seed, a.eval_every) for m in METHODS for r in REGIMES]
    out = {}
    with get_context("spawn").Pool(a.procs) as pool:
        for regime, method, h, dt in pool.imap_unordered(_task, tasks):
            print(f"== {regime} {method} done in {dt / 60:.1f} min", flush=True)
            for k, v in h.items():
                out[f"{regime}|{method}::{k}"] = np.asarray(v)[None] if np.ndim(v) else np.asarray(v)
            np.savez_compressed(os.path.join(ROOT, "results", "cifar10.npz"), **out)


if __name__ == "__main__":
    main()
