"""Hyper-parameter sweeps and multi-seed runs, parallelised over processes."""
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from .methods import run_method
from .problems import MnistMLP, Quadratic
from .timing import TimeModel

N_WORKERS = 6174

# ---------------------------------------------------------------- setups ----
# "text":      the setup exactly as written in App. G of the paper.
# "fig2match": reverse-engineered variant whose Ringmaster/Rennala curves match
#              the paper's Fig. 2 numerically (b = 0, noiseless gradients,
#              tau_i = sqrt(i) + |eta_i| fixed per worker). Diagnostic only.
SETUPS = {
    "text": dict(problem=dict(kind="quadratic", d=1729, noise="iid", noise_std=0.01, b_scale=1.0),
                 time=dict(kind="per_job", base="linear"), T=2000.0),
    "text-fixedtime": dict(problem=dict(kind="quadratic", d=1729, noise="iid", noise_std=0.01, b_scale=1.0),
                           time=dict(kind="fixed", base="linear"), T=2000.0),
    "fig2match": dict(problem=dict(kind="quadratic", d=1729, noise="iid", noise_std=0.0, b_scale=0.0),
                      time=dict(kind="fixed", base="sqrt"), T=2000.0),
    "mnist": dict(problem=dict(kind="mnist", hidden=32, batch=4, subset=3000),
                  time=dict(kind="per_job", base="linear"), T=1000.0),
}

# Hyper-parameters printed in the legends of Fig. 2 and Fig. 3.
PAPER_HPARAMS = {
    "quadratic": {"ringmaster": dict(gamma=0.2, R=6), "ringmaster_nostop": dict(gamma=0.2, R=6),
                  "rennala": dict(gamma=1.0, B=6), "da_asgd": dict(gamma=0.00032)},
    "mnist": {"ringmaster": dict(gamma=1.0, R=6), "ringmaster_nostop": dict(gamma=1.0, R=6),
              "rennala": dict(gamma=5.0, B=6), "da_asgd": dict(gamma=0.0016)},
}

# Values read off the paper's figures by eye (log-scale, so ~10-20% accuracy).
PAPER_POINTS = {
    "quadratic": {
        "ringmaster": {200: 1e-2, 400: 1.3e-3, 600: 4e-4, 800: 2e-4, 1000: 1.1e-4, 1400: 6e-5, 2000: 2.5e-5},
        "rennala": {200: 4e-2, 400: 6e-3, 600: 1.5e-3, 800: 7e-4, 1000: 4e-4, 1400: 1.5e-4, 2000: 5e-5},
        "da_asgd": {230: 90, 420: 30, 600: 3, 1000: 2, 1400: 0.4, 1800: 0.04, 2000: 0.02},
    },
    "mnist": {
        "ringmaster": {200: 1e-8, 300: 1e-11, 400: 2e-14},
        "rennala": {400: 1e-5, 500: 1e-7, 600: 1e-9, 700: 1e-12},
        "da_asgd": {200: 1e-2, 500: 3e-4, 1000: 1e-5},
    },
}


def gamma_grid():
    """{5^p : p in [-5, 5]} (paper)."""
    return [5.0 ** p for p in range(-5, 6)]


def rb_grid(n=N_WORKERS):
    """{ceil(n/4^p) : p in N0} from the paper, plus the floor() variant: the
    paper's reported R = B = 6 is floor(6174/4^5) and not in the ceil set."""
    vals = set()
    p = 0
    while True:
        q = n / 4 ** p
        vals.add(math.ceil(q))
        vals.add(max(1, math.floor(q)))
        if q < 1:
            break
        p += 1
    return sorted(vals)


def method_grid(method, n=N_WORKERS):
    if method in ("asgd", "da_asgd"):
        return [dict(gamma=g) for g in gamma_grid()]
    key = "B" if method == "rennala" else "R"
    return [{"gamma": g, key: r} for g in gamma_grid() for r in rb_grid(n)]


# ------------------------------------------------------------- execution ----
_PROBLEM_CACHE = {}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def make_problem(spec):
    key = json.dumps(spec, sort_keys=True)
    if key not in _PROBLEM_CACHE:
        s = dict(spec)
        kind = s.pop("kind")
        if kind == "quadratic":
            _PROBLEM_CACHE[key] = Quadratic(**s)
        elif kind == "mnist":
            _PROBLEM_CACHE[key] = MnistMLP(DATA_DIR, **s)
        else:
            raise ValueError(kind)
    return _PROBLEM_CACHE[key]


def _task(args):
    setup, method, params, seed, num_points = args
    problem = make_problem(setup["problem"])
    tm = TimeModel(N_WORKERS, setup["time"]["kind"], seed=10_000 + seed, base=setup["time"]["base"])
    r = run_method(method, problem, tm, setup["T"], params, seed=seed, num_points=num_points)
    return dict(method=method, params=params, seed=seed, t=r["t"], gap=r["gap"],
                iters=r["iters"], diverged=r["diverged"])


def run_tasks(tasks, processes=None):
    processes = processes or max(1, (os.cpu_count() or 2) - 2)
    # Longest tasks first for better load balancing (DA-ASGD has most events).
    with Pool(processes, initializer=_limit_threads) as pool:
        return pool.map(_task, tasks, chunksize=1)


def _limit_threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def score(gap, criterion="final"):
    """Tuning objective (lower is better).
    final: mean log10 gap over the last 5% of the horizon;
    auc:   mean log10 gap over the whole horizon (area under the log curve)."""
    tail = gap[-max(1, int(len(gap) * 0.05)):] if criterion == "final" else gap[1:]
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0):
        return np.inf
    return float(np.mean(np.log10(tail)))


def tune(setup, methods, seeds=(0, 1), num_points=400, processes=None, log=print, criterion="final"):
    tasks = [(setup, m, p, s, num_points) for m in methods for p in method_grid(m) for s in seeds]
    log(f"tuning: {len(tasks)} runs ...")
    results = run_tasks(tasks, processes)
    table = {}
    for r in results:
        k = (r["method"], json.dumps(r["params"], sort_keys=True))
        table.setdefault(k, []).append(score(r["gap"], criterion))
    best, rows = {}, []
    for (m, pj), scores in table.items():
        s = float(np.mean(scores))
        rows.append(dict(method=m, params=json.loads(pj), score=s))
        if m not in best or s < best[m]["score"]:
            best[m] = dict(params=json.loads(pj), score=s)
    return best, rows


def multi_seed(setup, configs, seeds=range(5), num_points=1000, processes=None):
    """configs: {label: (method, params)} -> {label: dict(t, gaps[seed, t], ...)}"""
    tasks, labels = [], []
    for label, (m, p) in configs.items():
        for s in seeds:
            tasks.append((setup, m, p, s, num_points))
            labels.append(label)
    results = run_tasks(tasks, processes)
    out = {}
    for label, r in zip(labels, results):
        o = out.setdefault(label, dict(method=r["method"], params=r["params"], t=r["t"], gaps=[], iters=[], diverged=[]))
        o["gaps"].append(r["gap"])
        o["iters"].append(r["iters"])
        o["diverged"].append(r["diverged"])
    for o in out.values():
        o["gaps"] = np.array(o["gaps"])
    return out


def save_runs(path, runs):
    arrays = {}
    meta = {}
    for label, o in runs.items():
        arrays[f"{label}__t"] = o["t"]
        arrays[f"{label}__gaps"] = o["gaps"]
        meta[label] = dict(method=o["method"], params=o["params"], iters=o["iters"], diverged=o["diverged"])
    np.savez_compressed(path, **arrays, meta=json.dumps(meta))


def load_runs(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {label: dict(t=z[f"{label}__t"], gaps=z[f"{label}__gaps"], **m) for label, m in meta.items()}
