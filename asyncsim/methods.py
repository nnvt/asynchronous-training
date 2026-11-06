"""Event-driven simulation of asynchronous / semi-asynchronous SGD methods.

All methods share the same model: n workers, each computes one stochastic
gradient at a time; worker i needs a (random) duration drawn from TimeModel.
Simulated wall-clock time advances from one gradient arrival to the next.

Implemented (paper numbering, arXiv:2501.16168):
  * asgd           -- Algorithm 1, constant stepsize
  * da_asgd        -- Algorithm 1 with the delay-adaptive stepsize of
                      Mishchenko et al. (2022), as in their released code:
                      gamma_k = gamma * min(1, n / delta_k)
  * rennala        -- Algorithm 2 (Rennala SGD), literal: the worker that
                      delivers the B-th gradient is re-assigned to the *old*
                      x^k (as in both papers' pseudocode); set
                      fresh_restart=True to give it x^{k+1} instead.
  * ringmaster     -- Algorithm 5 (with calculation stops)   [default]
  * ringmaster_nostop -- Algorithm 4 (ignore gradients with delay >= R)
"""
import numpy as np


class Recorder:
    """Records gap(x) on a uniform time grid. The iterate x is piecewise
    constant in time; the value at grid time s is the iterate after all
    arrivals with time <= s."""

    def __init__(self, T, num_points, gap_fn, blowup):
        self.grid = np.linspace(0.0, T, num_points)
        self.vals = np.full(num_points, np.nan)
        self.i = 0
        self.gap_fn = gap_fn
        self.blowup = blowup
        self.diverged = False

    def advance(self, t, x):
        """Record every grid point strictly before time t using iterate x.
        Returns True if the run diverged."""
        grid, n = self.grid, len(self.grid)
        if self.i < n and grid[self.i] < t:
            v = self.gap_fn(x)
            if not np.isfinite(v) or v > self.blowup:
                self.diverged = True
                self.vals[self.i:] = np.inf
                self.i = n
                return True
            while self.i < n and grid[self.i] < t:
                self.vals[self.i] = v
                self.i += 1
        return False

    def close(self, x):
        if self.i < len(self.grid):
            self.advance(np.inf, x)
        return self.grid, self.vals


def _result(rec, k, extra=None):
    grid, vals = rec.grid, rec.vals
    out = dict(t=grid, gap=vals, iters=k, diverged=rec.diverged)
    if extra:
        out.update(extra)
    return out


def run_asgd(problem, tm, T, gamma, adaptive=False, seed=0, num_points=1000):
    """Algorithm 1. Every gradient is applied; adaptive=True gives
    Delay-Adaptive ASGD (gamma_k = gamma * min(1, n/delta_k))."""
    rng = np.random.default_rng(seed)
    n = tm.n
    x = problem.x0.copy()
    k = 0
    finish = tm.sample_all()
    start_iter = np.zeros(n, dtype=np.int64)
    # Gradients are computed eagerly when a job starts (all of them are
    # used); jobs that cannot finish before T are never materialised.
    pending = {j: problem.grad(x, rng) for j in np.nonzero(finish <= T)[0]}
    rec = Recorder(T, num_points, problem.gap, problem.blowup)
    max_delay = 0
    with np.errstate(all="ignore"):
        while True:
            j = int(np.argmin(finish))
            t = finish[j]
            if t > T or rec.advance(t, x):
                break
            delta = k - start_iter[j]
            max_delay = max(max_delay, delta)
            g = pending.pop(j)
            step = gamma * min(1.0, n / delta) if (adaptive and delta > 0) else gamma
            x = x - step * g
            k += 1
            start_iter[j] = k
            finish[j] = t + tm.sample(j)
            if finish[j] <= T:
                pending[j] = problem.grad(x, rng)
    rec.close(x)
    return _result(rec, k, dict(max_delay=int(max_delay)))


def run_ringmaster(problem, tm, T, gamma, R, stops=True, seed=0, num_points=1000):
    """Algorithm 5 (stops=True) or Algorithm 4 (stops=False)."""
    rng = np.random.default_rng(seed)
    n = tm.n
    R = int(R)
    x = problem.x0.copy()
    # Iterates are never modified in place (x = x - ...), so we keep plain
    # references keyed by iteration and prune the ones no usable job needs.
    snaps = {0: x}
    k = 0
    finish = tm.sample_all()
    start_iter = np.zeros(n, dtype=np.int64)
    rec = Recorder(T, num_points, problem.gap, problem.blowup)
    ignored = 0
    with np.errstate(all="ignore"):
        while True:
            j = int(np.argmin(finish))
            t = finish[j]
            if t > T or rec.advance(t, x):
                break
            delta = k - start_iter[j]
            if delta < R:
                g = problem.grad(snaps[int(start_iter[j])], rng)
                x = x - gamma * g
                k += 1
                snaps[k] = x
                if k % 256 == 0:
                    live = start_iter[(finish <= T) & (start_iter > k - R)]
                    keep = set(live.tolist()) | {k}
                    snaps = {s: v for s, v in snaps.items() if s in keep}
            else:  # only reachable without stops (Algorithm 4)
                ignored += 1
            start_iter[j] = k
            finish[j] = t + tm.sample(j)
            if stops:
                # Algorithm 5: abort every computation whose delay w.r.t. the
                # new iterate is >= R and restart it at x^k right now.
                stale = np.nonzero(start_iter <= k - R)[0]
                if stale.size:
                    start_iter[stale] = k
                    finish[stale] = t + tm.sample(stale)
    rec.close(x)
    return _result(rec, k, dict(ignored=ignored))


def run_rennala(problem, tm, T, gamma, B, seed=0, num_points=1000, fresh_restart=False):
    """Algorithm 2 (Rennala SGD)."""
    rng = np.random.default_rng(seed)
    n = tm.n
    B = int(B)
    x = problem.x0.copy()
    k = 0
    finish = tm.sample_all()
    start_iter = np.zeros(n, dtype=np.int64)
    acc = np.zeros_like(x)
    b = 0
    rec = Recorder(T, num_points, problem.gap, problem.blowup)
    with np.errstate(all="ignore"):
        while True:
            j = int(np.argmin(finish))
            t = finish[j]
            if t > T or rec.advance(t, x):
                break
            if start_iter[j] == k:
                acc += problem.grad(x, rng)
                b += 1
            start_iter[j] = k
            finish[j] = t + tm.sample(j)
            if b == B:
                x = x - gamma * acc / B
                k += 1
                acc[:] = 0.0
                b = 0
                if fresh_restart:
                    start_iter[j] = k
    rec.close(x)
    return _result(rec, k)


def run_method(method, problem, tm, T, params, seed=0, num_points=1000):
    if method == "asgd":
        return run_asgd(problem, tm, T, params["gamma"], adaptive=False, seed=seed, num_points=num_points)
    if method == "da_asgd":
        return run_asgd(problem, tm, T, params["gamma"], adaptive=True, seed=seed, num_points=num_points)
    if method == "ringmaster":
        return run_ringmaster(problem, tm, T, params["gamma"], params["R"], stops=True, seed=seed, num_points=num_points)
    if method == "ringmaster_nostop":
        return run_ringmaster(problem, tm, T, params["gamma"], params["R"], stops=False, seed=seed, num_points=num_points)
    if method == "rennala":
        return run_rennala(problem, tm, T, params["gamma"], params["B"], seed=seed, num_points=num_points,
                           fresh_restart=params.get("fresh_restart", False))
    raise ValueError(method)
