"""Tasks, parallel execution, tuning and multi-seed runs."""
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from .methods import RUNNERS, Recorder
from .problem import MLPProblem
from .timing import TimeModel

N_CLIENTS = 100
# Wall-clock budgets read off the x-axes of Fig. 1 (a) / (b).
BUDGET = {"mnist": 200_000.0, "fmnist": 375_000.0}
GAMMAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]  # App. A grid
METHODS = ["ringleader", "malenia", "ia2sgd"]


def make_grid(T, num_points):
    return np.linspace(0.0, T, num_points)


def run_one(task):
    """task: dict(dataset, method, gamma, seed, T, num_points, timing, [grid])"""
    t0 = time.perf_counter()
    p = MLPProblem(task["dataset"], n=N_CLIENTS, seed=task["seed"])
    tm = TimeModel(N_CLIENTS, task.get("timing", "per_job"), seed=[task["seed"], 3])
    grid = task.get("grid")
    if grid is None:
        grid = make_grid(task["T"], task["num_points"])
    rec = Recorder(p, grid)
    info = RUNNERS[task["method"]](p, tm, task["T"], task["gamma"], rec)
    out = dict(task, t=rec.grid, g2=rec.g2, loss=rec.loss, diverged=rec.diverged,
               wall=time.perf_counter() - t0, eval_wall=rec.eval_time, **info)
    out.pop("grid", None)
    return out


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def task_key(task):
    return (f"{task['dataset']}_{task['method']}_g{task['gamma']:g}_s{task['seed']}_{task.get('timing', 'per_job')}"
            f"_T{task['T']:g}_n{len(task['grid']) if task.get('grid') is not None else task['num_points']}")


def run_tasks(tasks, processes=5, log=print, raw_dir=None):
    """Runs tasks in a pool. If raw_dir is given, every finished run is also
    written to raw_dir/<key>.npz and already-present runs are not recomputed."""
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
        cached, todo = {}, []
        for i, tk in enumerate(tasks):
            f = os.path.join(raw_dir, task_key(tk) + ".npz")
            if os.path.exists(f):
                cached[i] = load_results(f)[0]
            else:
                todo.append(i)
        if cached and log:
            log(f"  {len(cached)} runs loaded from {raw_dir}")
        results = [cached.get(i) for i in range(len(tasks))]
        sub = [tasks[i] for i in todo]
        for i, r in zip(todo, _run_pool(sub, processes, log, raw_dir)):
            results[i] = r
        return results
    return _run_pool(tasks, processes, log, None)


def _run_pool(tasks, processes, log, raw_dir):
    if not tasks:
        return []
    # heaviest first: IA2SGD has ~7x more server updates than Ringleader
    cost = {"ia2sgd": 3.0, "ringleader": 1.5, "malenia": 1.0}
    order = sorted(range(len(tasks)), key=lambda i: -cost[tasks[i]["method"]] * tasks[i]["T"])
    results = [None] * len(tasks)
    t0 = time.time()
    with Pool(processes, initializer=_init_worker) as pool:
        for done, (i, r) in enumerate(pool.imap_unordered(_indexed, [(i, tasks[i]) for i in order]), 1):
            results[i] = r
            if raw_dir:
                save_results(os.path.join(raw_dir, task_key(tasks[i]) + ".npz"), [r])
            if log and (done % max(1, len(tasks) // 20) == 0 or done == len(tasks)):
                log(f"  {done}/{len(tasks)} runs done ({time.time() - t0:.0f}s)")
    return results


def _indexed(arg):
    i, task = arg
    return i, run_one(task)


def tail_score(g2, frac=0.1):
    """Tuning objective (lower is better): mean log10 ||grad f||^2 over the
    last `frac` of the wall-clock budget; +inf if the run diverged."""
    tail = g2[-max(1, int(round(len(g2) * frac))):]
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0):
        return np.inf
    return float(np.mean(np.log10(tail)))


def save_results(path, results):
    """Compact npz: one row per run (curves) + JSON metadata."""
    keys = ["dataset", "method", "gamma", "seed", "T", "num_points", "timing", "diverged", "wall",
            "eval_wall", "iters", "rounds", "max_delay", "grads_used"]
    meta = [{k: r.get(k) for k in keys if k in r} for r in results]
    np.savez_compressed(path, t=np.stack([r["t"] for r in results]), g2=np.stack([r["g2"] for r in results]),
                        loss=np.stack([r["loss"] for r in results]), meta=json.dumps(meta))


def load_results(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    out = []
    for i, m in enumerate(meta):
        out.append(dict(m, t=z["t"][i], g2=z["g2"][i], loss=z["loss"][i]))
    return out
