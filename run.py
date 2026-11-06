"""Replicate Fig. 1 (App. A) of "Ringleader ASGD" (Maranjyan & Richtarik, ICLR 2026).

  python run.py tune  --dataset mnist            # gamma sweep on the paper's grid (seed 0)
  python run.py final --dataset mnist --seeds 10 # tuned gammas, seeds 1..10
  python run.py plot                             # figures/ + summary table

Options --timing fixed / --tag <name> run diagnostic variants into separate files.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from rlsim import experiment as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")


def tune_grid(T):
    return np.unique(np.concatenate([np.linspace(0, 0.9 * T, 28), np.linspace(0.9 * T, T, 11)]))


def final_grid(T, points=101):
    return np.unique(np.concatenate([np.linspace(0, T, points), np.linspace(0, 0.025 * T, 11)]))


def tail_score(t, g2, T):
    m = t >= 0.9 * T - 1e-9
    tail = g2[m]
    if not np.all(np.isfinite(tail)) or np.any(tail <= 0):
        return np.inf
    return float(np.mean(np.log10(tail)))


def name(kind, dataset, timing, tag):
    s = f"{kind}_{dataset}"
    if timing != "per_job":
        s += f"_{timing}"
    if tag:
        s += f"_{tag}"
    return s


def log_to(path):
    fh = open(path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def cmd_tune(a):
    tasks = []
    for ds in a.dataset:
        T = E.BUDGET[ds] * a.budget_scale
        tasks += [dict(dataset=ds, method=m, gamma=g, seed=s, T=T, grid=tune_grid(T), timing=a.timing)
                  for m in a.methods for g in (a.gammas or E.GAMMAS) for s in a.tune_seeds]
    log = log_to(os.path.join(RES, "logs", name("tune", "-".join(a.dataset), a.timing, a.tag) + ".log"))
    log(f"tuning {a.dataset} timing={a.timing} scale={a.budget_scale}: {len(tasks)} runs on {a.processes} processes")
    results = E.run_tasks(tasks, a.processes, log=log, raw_dir=os.path.join(RES, "raw"))
    for ds in a.dataset:
        T = E.BUDGET[ds] * a.budget_scale
        base = name("tune", ds, a.timing, a.tag)
        rs = [r for r in results if r["dataset"] == ds]
        path = os.path.join(RES, base + ".npz")
        if os.path.exists(path) and a.append:
            rs = E.load_results(path) + rs
        E.save_results(path, rs)
        log(f"{ds}:")
        best = summarize_tuning(rs, T, log)
        with open(os.path.join(RES, base + "_best.json"), "w") as fh:
            json.dump(best, fh, indent=1)


def summarize_tuning(results, T, log=print):
    table = {}
    for r in results:
        table.setdefault((r["method"], r["gamma"]), []).append(tail_score(r["t"], r["g2"], T))
    best = {}
    for m in sorted({k[0] for k in table}):
        row = sorted((g, float(np.mean(v)), len(v)) for (mm, g), v in table.items() if mm == m)
        log(f"  {m:11s} " + "  ".join(f"{g:g}:{s:.2f}" for g, s, _ in row))
        g, s, _ = min(row, key=lambda z: z[1])
        best[m] = dict(gamma=g, score=s)
        log(f"  {m:11s} best gamma = {g:g} (mean log10 tail ||grad f||^2 = {s:.3f})")
    return best


def best_by_loss(results):
    """gamma minimising the mean training loss over the last 10% of the budget (diverged runs = +inf)."""
    best = {}
    for r in results:
        t, L, T = r["t"], r["loss"], r["T"]
        tail = L[t >= 0.9 * T - 1e-9]
        sc = float(np.mean(tail)) if np.all(np.isfinite(tail)) else np.inf
        m = r["method"]
        if m not in best or sc < best[m]["score"]:
            best[m] = dict(gamma=r["gamma"], score=sc)
    return best


def cmd_final(a):
    tasks, gam = [], {}
    for ds in a.dataset:
        T = E.BUDGET[ds] * a.budget_scale
        tb = name("tune", ds, a.timing if a.tune_from_timing else "per_job", a.tune_tag)
        if a.criterion == "grad":
            with open(os.path.join(RES, tb + "_best.json")) as fh:
                best = json.load(fh)
        else:  # diagnostic: pick gamma by the final training loss of the same sweep
            best = best_by_loss(E.load_results(os.path.join(RES, tb + ".npz")))
            with open(os.path.join(RES, tb + "_best_by_loss.json"), "w") as fh:
                json.dump(best, fh, indent=1)
        for item in a.gamma_override or []:
            d, m, g = item.split(":")
            if d == ds:
                best[m] = dict(gamma=float(g), score=None)
        gam[ds] = {m: best[m]["gamma"] for m in a.methods}
        seeds = list(range(a.first_seed, a.first_seed + a.seeds))
        tasks += [dict(dataset=ds, method=m, gamma=best[m]["gamma"], seed=s, T=T, grid=final_grid(T, a.points),
                       timing=a.timing) for m in a.methods for s in seeds]
    log = log_to(os.path.join(RES, "logs", name("final", "-".join(a.dataset), a.timing, a.tag) + ".log"))
    log(f"final timing={a.timing} gammas={gam} seeds={a.first_seed}..{a.first_seed + a.seeds - 1}: {len(tasks)} runs")
    results = E.run_tasks(tasks, a.processes, log=log, raw_dir=os.path.join(RES, "raw"))
    for ds in a.dataset:
        rs_ds = [r for r in results if r["dataset"] == ds]
        E.save_results(os.path.join(RES, name("final", ds, a.timing, a.tag) + ".npz"), rs_ds)
        for m in a.methods:
            rs = [r for r in rs_ds if r["method"] == m]
            g = np.array([r["g2"] for r in rs])
            log(f"  {ds:6s} {m:11s} gamma={rs[0]['gamma']:g} median final ||grad f||^2 = {np.median(g[:, -1]):.3e}; "
                f"iters {int(np.median([r['iters'] for r in rs]))}; max delay {max(r.get('max_delay') or 0 for r in rs)}; "
                f"wall {np.mean([r['wall'] for r in rs]):.0f}s; diverged {sum(bool(r['diverged']) for r in rs)}")


def cmd_plot(a):
    from rlsim import plotting
    plotting.make_all(RES, FIG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("tune", "final"):
        s = sub.add_parser(c)
        s.add_argument("--dataset", choices=sorted(E.BUDGET), nargs="+", required=True)
        s.add_argument("--methods", nargs="+", default=E.METHODS, choices=E.METHODS)
        s.add_argument("--timing", choices=["per_job", "fixed"], default="per_job")
        s.add_argument("--processes", type=int, default=5)
        s.add_argument("--tag", default="")
        s.add_argument("--budget-scale", type=float, default=1.0)
    s = sub.choices["tune"]
    s.add_argument("--gammas", type=float, nargs="+", default=None)
    s.add_argument("--tune-seeds", type=int, nargs="+", default=[0])
    s.add_argument("--append", action="store_true", help="append to an existing tuning file")
    s = sub.choices["final"]
    s.add_argument("--seeds", type=int, default=10)
    s.add_argument("--first-seed", type=int, default=1)
    s.add_argument("--tune-tag", default="")
    s.add_argument("--tune-from-timing", action="store_true",
                   help="read tuned gammas from the tuning run with the same --timing")
    s.add_argument("--gamma-override", nargs="+", default=None, metavar="DATASET:METHOD:GAMMA")
    s.add_argument("--points", type=int, default=81, help="uniform evaluation points (plus 10 early ones)")
    s.add_argument("--criterion", choices=["grad", "loss"], default="grad",
                   help="gamma selection from the sweep: tail ||grad f||^2 (paper metric) or tail training loss")
    sub.add_parser("plot")
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    {"tune": cmd_tune, "final": cmd_final, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    main()
