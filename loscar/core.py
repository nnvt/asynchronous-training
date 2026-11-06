"""Logical-time simulator for LOSCAR-SGD (Algorithm 1) and the baselines of Appendix B.

All workers live in one process. Logical time follows Assumption 2.1: worker i needs tau_i time
units per stochastic gradient, tau = lcm(tau_i), one round lasts M*tau + zeta.

Methods (names as in the paper's legends):
  local_sparse       blocking "Local Sparse": N_i = M tau / tau_i local steps, then the workers
                     idle for zeta while the sparse average (mask S_r, |S_r| = K) is exchanged;
                     merge = overwrite (y_i = z_i, so overwrite == delay-correction here).
  overlap_overwrite  N_i steps -> y_i, send Proj_S(y_i), Q_i = zeta / tau_i further steps -> z_i,
                     x_i <- z_i off S,  x_i <- ybar on S.
  overlap_dc         LOSCAR-SGD (Alg. 1): as above but x_i <- ybar + (z_i - y_i) on S.
  fedavg_full        local_sparse with K = d (dense blocking model averaging).
  sync_sgd           minibatch SGD: every worker computes one stochastic gradient at the shared
                     model, gradients are averaged; one step lasts max_i tau_i + zeta.

Metrics are evaluated at the average model xbar^r = (1/n) sum_i x_i^r (the quantity in Thm 4.1).
"""
import math
from functools import reduce

import numpy as np

METHODS = ("local_sparse", "overlap_overwrite", "overlap_dc", "fedavg_full", "sync_sgd")
LABELS = {
    "local_sparse": "Local Sparse",
    "overlap_overwrite": "Overlap overwrite",
    "overlap_dc": "Overlap delay-correction",
    "fedavg_full": "FedAvg-Full",
    "sync_sgd": "Sync SGD",
}


def lcm(xs):
    return reduce(lambda a, b: a * b // math.gcd(a, b), xs)


def n_coords(p, d):
    """K = ceil(p d) (>= 1). ceil (not round) is what reproduces the paper's bit counts:
    K = 1, 2, 13, 124 for p = 0.001, 0.01, 0.1, 1 with d = 124."""
    return max(1, min(d, int(math.ceil(p * d - 1e-9))))


def schedule(taus, M, zeta):
    tau = lcm(taus)
    assert zeta % tau == 0, "zeta must be a multiple of lcm(tau_i)"
    N = [M * tau // t for t in taus]
    Q = [zeta // t for t in taus]
    return tau, N, Q


def run(problem, method, *, taus, M, zeta, p, rounds, seed, n=None, eval_every=1,
        bits_per_value=32, time_budget=None, verbose=False):
    """Simulate one run; returns dict of 1-D arrays indexed by evaluation point."""
    n = n or len(taus)
    assert len(taus) == n
    d = problem.d
    tau, N, Q = schedule(taus, M, zeta)
    T_round = M * tau + zeta
    K = d if method in ("fedavg_full", "sync_sgd") else n_coords(p, d)
    # logical bits per round: n uplink + n downlink messages of K 32-bit values, plus the K shared
    # mask indices (32 bit). (2n+1)*32*K reproduces the x-axis of the paper's Figs. 1b, 4c, 8.
    bits_round = (2 * n + 1) * bits_per_value * K

    x0 = problem.init(seed)
    xs = [problem.copy(x0) for _ in range(n)]
    workers = [problem.make_worker(i, seed) for i in range(n)]
    mask_rng = np.random.default_rng([seed, 777])

    hist = {"round": [], "time": [], "bits": [], "examples": [], "disagreement": []}
    state = {"time": 0.0, "bits": 0.0, "examples": 0}

    def record(r):
        xbar = problem.mean(xs)
        m = problem.evaluate(xbar)
        hist["round"].append(r)
        hist["time"].append(state["time"])
        hist["bits"].append(state["bits"])
        hist["examples"].append(state["examples"])
        hist["disagreement"].append(float(np.mean([problem.sqdist(x, xbar) for x in xs])))
        for k, v in m.items():
            hist.setdefault(k, []).append(v)
        if verbose:
            print(method, r, state["time"], {k: round(v, 4) for k, v in m.items()}, flush=True)

    record(0)
    if method == "sync_sgd":
        T_step = max(taus) + zeta
        n_steps = rounds if time_budget is None else int(time_budget // T_step)
        for r in range(1, n_steps + 1):
            outs = []
            for i in range(n):
                xi, ex = problem.local_steps(xs[i], workers[i], 1)
                outs.append(xi)
                state["examples"] += ex
            xbar = problem.mean(outs)
            xs = [problem.copy(xbar) for _ in range(n)]
            state["time"] += T_step
            state["bits"] += 2 * n * bits_per_value * d
            if r % eval_every == 0 or r == n_steps:
                record(r)
        return {k: np.asarray(v) for k, v in hist.items()} | {"K": K, "T_round": T_step}

    overlap = method in ("overlap_overwrite", "overlap_dc")
    for r in range(1, rounds + 1):
        S = mask_rng.choice(d, size=K, replace=False) if K < d else None
        if S is not None and hasattr(problem, "as_index"):
            S = problem.as_index(S)
        ys, zs = [], []
        for i in range(n):
            yi, ex = problem.local_steps(xs[i], workers[i], N[i])
            state["examples"] += ex
            if overlap and Q[i] > 0:
                zi, ex = problem.local_steps(yi, workers[i], Q[i])
                state["examples"] += ex
            else:
                zi = yi
            ys.append(yi)
            zs.append(zi)
        if S is None:
            ybar_S = problem.mean(ys)
        else:
            ybar_S = problem.mean([yi[S] for yi in ys])
        new = []
        for i in range(n):
            xi = problem.copy(zs[i])
            if S is None:
                xi = problem.copy(ybar_S) + ((zs[i] - ys[i]) if method == "overlap_dc" else 0.0)
            elif method == "overlap_dc":
                xi[S] = ybar_S + (zs[i][S] - ys[i][S])
            else:  # overwrite (also local_sparse / fedavg_full, where z_i = y_i)
                xi[S] = ybar_S
            new.append(xi)
        xs = new
        state["time"] += T_round
        state["bits"] += bits_round
        if r % eval_every == 0 or r == rounds:
            record(r)
    return {k: np.asarray(v) for k, v in hist.items()} | {"K": K, "T_round": T_round}
