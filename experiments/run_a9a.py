"""Run every a9a logistic-regression experiment of the paper (Sec. 5 / App. B.1-B.5).

Usage:  python experiments/run_a9a.py [--seeds 10] [--procs 5] [--only e1,e3]
Writes results/a9a_<exp>.npz with arrays of shape (seeds, eval points) for every metric.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from loscar.core import run  # noqa: E402
from loscar.data import a9a  # noqa: E402
from loscar.logreg import LogReg  # noqa: E402

LS, OW, DC = "local_sparse", "overlap_overwrite", "overlap_dc"
SPARSE3 = (LS, OW, DC)
BASE = dict(taus=(1, 2, 3, 6), p=0.3, rounds=20, lr=0.1, batch=256, partition="iid", alpha=None,
            min_size=10, lam=0.0, theta=1.0)


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


EXPERIMENTS = {
    # B.1 / Fig. 1a, 2, 3, 4: controlled sparse-overlap comparison (+ the two dense baselines of
    # App. B, which the paper lists but never plots).
    "e1_main": {m: cfg(method=m, M=3, zeta=6) for m in SPARSE3 + ("fedavg_full",)}
    | {"sync_sgd": cfg(method="sync_sgd", M=3, zeta=6, time_budget=480)},
    # B.2 / Figs. 1c, 5-7: merge rule across regimes (LS also run, for reference only).
    "e2_long_compute": {m: cfg(method=m, M=8, zeta=6) for m in SPARSE3},
    "e2_long_delay": {m: cfg(method=m, M=2, zeta=24) for m in SPARSE3},
    "e2_hetero_speed": {m: cfg(method=m, M=2, zeta=20, taus=(1, 2, 10, 20)) for m in SPARSE3},
    # B.3 / Figs. 1b, 8: sparsity level.
    "e3_sparsity": {f"{m}|p={p}": cfg(method=m, M=4, zeta=6, p=p)
                    for m in SPARSE3 for p in (0.001, 0.01, 0.1, 1.0)},
    # B.4 / Fig. 9: local computation budget.
    "e4_budget": {f"{m}|M={M}": cfg(method=m, M=M, zeta=6) for m in SPARSE3 for M in (1, 4, 16, 64)},
    # B.5 / Fig. 10: communication delay. M is not stated in the text; M = 8 is read off the
    # figure's time axis (first point at t = 60 for zeta = 12 and t = 96 for zeta = 48).
    "e5_delay": {f"{m}|zeta={z}": cfg(method=m, M=8, zeta=z) for m in SPARSE3 for z in (0, 12, 48)},
    # B.5 stress test / Fig. 11: label-Dirichlet(0.03) heterogeneous data, very slow comm.
    "e6_noniid": {m: cfg(method=m, M=1, zeta=96, p=0.01, rounds=40, lr=0.3,
                         partition="dirichlet", alpha=0.03) for m in SPARSE3},
    # ---- diagnostics (NOT in the paper), to probe the Fig. 11 mismatch ----------------------
    # d1: same stress test with denser communication (does OW separate from DC?)
    "d1_noniid_p": {f"{m}|p={p}": cfg(method=m, M=1, zeta=96, p=p, rounds=40, lr=0.3,
                                      partition="dirichlet", alpha=0.03)
                    for m in SPARSE3 for p in (0.1, 0.3, 1.0)},
    # x1: E1 setup with the nonconvex Geman-McClure regulariser (eq. 89). The paper ran this but
    # gives neither lambda/theta nor plots; lambda = 1e-3, theta = 1 is our choice.
    "x1_nonconvex": {m: cfg(method=m, M=3, zeta=6, lam=1e-3, theta=1.0) for m in SPARSE3},
    # d2: same stress test, but every worker owns >= 256 examples (one full batch)
    "d2_noniid_min256": {m: cfg(method=m, M=1, zeta=96, p=0.01, rounds=40, lr=0.3,
                                partition="dirichlet", alpha=0.03, min_size=256) for m in SPARSE3},
}


def _task(args):
    exp, seed = args
    t0 = time.time()
    Xtr, ytr, Xv, yv = a9a(seed)
    probs = {}
    out = {}
    for label, c in EXPERIMENTS[exp].items():
        key = (c["lr"], c["batch"], c["partition"], c["alpha"], c["min_size"], c["lam"], c["theta"])
        if key not in probs:
            probs[key] = LogReg(Xtr, ytr, Xv, yv, len(c["taus"]), batch=c["batch"], lr=c["lr"],
                                partition=c["partition"], alpha=c["alpha"],
                                min_size=c["min_size"], lam=c["lam"], theta=c["theta"],
                                seed=seed)
        h = run(probs[key], c["method"], taus=list(c["taus"]), M=c["M"], zeta=c["zeta"], p=c["p"],
                rounds=c["rounds"], seed=seed, time_budget=c.get("time_budget"))
        out[label] = h
    return exp, seed, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    exps = [e for e in EXPERIMENTS if not a.only or any(e.startswith(o) for o in a.only.split(","))]
    tasks = [(e, s) for e in exps for s in range(a.seeds)]
    res = {e: {} for e in exps}
    t0 = time.time()
    with Pool(a.procs) as pool:
        for exp, seed, out, dt in pool.imap_unordered(_task, tasks):
            res[exp][seed] = out
            print(f"{exp} seed {seed} done in {dt:.1f}s", flush=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for exp in exps:
        arrs = {}
        seeds = sorted(res[exp])
        for label in EXPERIMENTS[exp]:
            hs = [res[exp][s][label] for s in seeds]
            for k, v in hs[0].items():
                if np.ndim(v) == 0:
                    arrs[f"{label}::{k}"] = np.asarray(v)
                else:
                    arrs[f"{label}::{k}"] = np.stack([h[k] for h in hs])
        np.savez_compressed(os.path.join(ROOT, "results", f"a9a_{exp}.npz"), **arrs)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
