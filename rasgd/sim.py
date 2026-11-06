"""Event-driven wall-clock simulation of the methods compared in
arXiv:2605.13434 (Rescaled ASGD; Malenia SGD; Ringleader ASGD; plus Vanilla
and Delay-Adaptive ASGD used in the toy experiments of App. F).

Conventions
-----------
* Worker i needs tau_i time units per stochastic gradient ("fixed") or an
  independent Exp(mean tau_i) draw per gradient ("exp", the paper's
  "fluctuating" setting).
* Simultaneous deliveries (only possible with fixed times) are processed in
  ascending worker index (App. D.2 of the paper); workers are ordered by
  ascending tau, so fast workers are processed first.
* Communication is instantaneous. A worker that receives a model starts its
  next gradient immediately.
* `problem` must provide: d, init(seed), loss(x), grad(i, x, rng) and
  grad_multi(i, x, rng, nb) (= average of nb independent minibatch gradients
  of worker i at the same point x).
* The loss of the *server* model is recorded on a uniform time grid: the value
  at grid time t is the loss after every event with time <= t was processed.
"""
import math

import numpy as np


class Timing:
    def __init__(self, taus, mode, rng):
        if mode not in ("fixed", "exp"):
            raise ValueError(mode)
        self.taus = np.asarray(taus, dtype=float)
        self.mode = mode
        self.rng = rng

    def draw(self, i):
        if self.mode == "fixed":
            return self.taus[i]
        return self.rng.exponential(self.taus[i])

    def draw_all(self):
        if self.mode == "fixed":
            return self.taus.copy()
        return self.rng.exponential(self.taus)


def harmonic_mean(taus):
    taus = np.asarray(taus, dtype=float)
    return len(taus) / np.sum(1.0 / taus)


# ---------------------------------------------------------------------------
# stepsize rules for Algorithm 1 (plain asynchronous SGD template)
# ---------------------------------------------------------------------------
def rescaled_rule(gamma, taus):
    """Eq. (2): gamma_i = gamma * tau_i * tau_H / (n * tau_max)."""
    taus = np.asarray(taus, dtype=float)
    n, tH, tmax = len(taus), harmonic_mean(taus), taus.max()
    gi = gamma * taus * tH / (n * tmax)
    return lambda i, delay: gi[i]


def vanilla_rule(gamma):
    return lambda i, delay: gamma


def delay_adaptive_rule(gamma, kind="paper", n=None):
    """kind='paper': gamma/(1+delay)  (App. D.2 of arXiv:2605.13434);
    kind='mishchenko': gamma*min(1, n/delay) (released code of
    Mishchenko et al. 2022)."""
    if kind == "paper":
        return lambda i, delay: gamma / (1.0 + delay)
    if kind == "mishchenko":
        return lambda i, delay: gamma * min(1.0, n / max(delay, 1))
    raise ValueError(kind)


class _Recorder:
    """Records loss(x) on a time grid and guards against divergence."""

    def __init__(self, problem, grid, blowup):
        self.problem, self.grid, self.blowup = problem, grid, blowup
        self.vals = np.full(len(grid), np.nan)
        self.j = 0
        self.diverged = False

    def advance(self, t_next, x):
        """Record all grid points strictly before the next event time."""
        while self.j < len(self.grid) and self.grid[self.j] < t_next:
            v = self.problem.loss(x)
            if not np.isfinite(v) or v > self.blowup:
                self.vals[self.j:] = np.inf
                self.j = len(self.grid)
                self.diverged = True
                return
            self.vals[self.j] = v
            self.j += 1

    @property
    def done(self):
        return self.j >= len(self.grid)


def _result(rec, n_updates, n_grads, steps_t, steps_w, extra=None):
    out = dict(loss=rec.vals, diverged=rec.diverged, n_updates=n_updates,
               n_grads=n_grads, steps_t=np.asarray(steps_t), steps_w=np.asarray(steps_w, dtype=np.int16))
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Algorithm 1: Asynchronous SGD with a stepsize rule (Rescaled / Vanilla / DA)
# ---------------------------------------------------------------------------
def run_asgd(problem, taus, mode, rule, T, grid, seed, blowup=50.0, log_steps=True, trace=False):
    rng_t = np.random.default_rng([seed, 1])
    rng_d = np.random.default_rng([seed, 2])
    timing = Timing(taus, mode, rng_t)
    n = len(taus)
    x = problem.init(seed)
    Y = np.repeat(x[None, :], n, axis=0)  # model held by each worker
    read = np.zeros(n, dtype=np.int64)  # server iteration at which worker read its model
    fin = timing.draw_all()
    rec = _Recorder(problem, grid, blowup)
    k = 0
    steps_t, steps_w, xs, lrs = [], [], [], []
    while True:
        i = int(np.argmin(fin))
        t = fin[i]
        rec.advance(min(t, math.nextafter(T, math.inf)), x)
        if t > T or rec.done:
            break
        g = problem.grad(i, Y[i], rng_d)
        lr = rule(i, k - read[i])
        x -= lr * g
        k += 1
        Y[i] = x
        read[i] = k
        fin[i] = t + timing.draw(i)
        if log_steps:
            steps_t.append(t)
            steps_w.append(i)
        if trace:  # full iterate trajectory (small d only)
            xs.append(x.copy())
            lrs.append(lr)
    rec.advance(math.inf, x)
    extra = dict(xs=np.asarray(xs), lrs=np.asarray(lrs)) if trace else None
    return _result(rec, k, k, steps_t, steps_w, extra)


# ---------------------------------------------------------------------------
# Malenia SGD (Tyurin & Richtarik 2023) with the parameter-free stopping rule
# "at least one gradient from every worker" (Ringleader paper, App. F; this is
# the description given in Sec. 1 of arXiv:2605.13434). Synchronous update;
# computations in progress at the broadcast are discarded.
# ---------------------------------------------------------------------------
def run_malenia(problem, taus, mode, gamma, T, grid, seed, blowup=50.0, log_steps=True):
    rng_t = np.random.default_rng([seed, 1])
    rng_d = np.random.default_rng([seed, 2])
    timing = Timing(taus, mode, rng_t)
    n = len(taus)
    x = problem.init(seed)
    b = np.zeros(n, dtype=np.int64)
    fin = timing.draw_all()
    rec = _Recorder(problem, grid, blowup)
    k = ngrads = 0
    steps_t, steps_w = [], []
    while True:
        i = int(np.argmin(fin))
        t = fin[i]
        rec.advance(min(t, math.nextafter(T, math.inf)), x)
        if t > T or rec.done:
            break
        b[i] += 1
        fin[i] = t + timing.draw(i)
        if b.min() >= 1:
            g = np.zeros_like(x)
            for j in range(n):
                g += problem.grad_multi(j, x, rng_d, int(b[j]))
            x -= (gamma / n) * g
            k += 1
            ngrads += int(b.sum())
            b[:] = 0
            fin = t + timing.draw_all()  # broadcast: everybody restarts at new x
            if log_steps:
                steps_t.append(t)
                steps_w.append(i)
    rec.advance(math.inf, x)
    return _result(rec, k, ngrads, steps_t, steps_w)


# ---------------------------------------------------------------------------
# Ringleader ASGD (Maranjyan & Richtarik 2026, Algorithm 1), verbatim.
# ---------------------------------------------------------------------------
def run_ringleader(problem, taus, mode, gamma, T, grid, seed, blowup=50.0, log_steps=True,
                   reuse_table=False):
    """reuse_table=False: Algorithm 1 of Maranjyan & Richtarik verbatim.
    reuse_table=True: DIAGNOSTIC variant (not in any paper): at the end of
    Phase 2, entries of workers that sent nothing new are kept (stale) instead
    of being emptied, so Phase 1 is complete at once and the next Phase 2 starts
    immediately. With harmonic fixed times this reproduces the update schedule
    drawn in Fig. 2(a) of arXiv:2605.13434 (10 updates per tau_max). A kept
    entry is *replaced* (not accumulated) by the worker's first gradient at its
    new point, so every entry always averages gradients from a single point."""
    rng_t = np.random.default_rng([seed, 1])
    rng_d = np.random.default_rng([seed, 2])
    timing = Timing(taus, mode, rng_t)
    n = len(taus)
    x = problem.init(seed)
    d = x.size
    Y = np.repeat(x[None, :], n, axis=0)
    G = np.zeros((n, d), dtype=x.dtype)
    bG = np.zeros(n, dtype=np.int64)
    S = np.zeros(n, dtype=bool)
    Gp = np.zeros((n, d), dtype=x.dtype)
    bGp = np.zeros(n, dtype=np.int64)
    Sp = np.zeros(n, dtype=bool)
    rem = np.zeros(n, dtype=bool)  # workers still to be updated in Phase 2
    A = np.zeros((n, d), dtype=x.dtype)  # A_j = G_j / b_j (valid in Phase 2)
    m = np.zeros(d, dtype=x.dtype)  # mean_j A_j
    phase = 1
    ver = np.zeros(n, dtype=np.int64)  # version of the point each worker computes at
    ent_ver = np.zeros(n, dtype=np.int64)  # version of the gradients in main-table entry j
    fin = timing.draw_all()
    rec = _Recorder(problem, grid, blowup)
    k = ngrads = 0
    steps_t, steps_w = [], []
    while True:
        i = int(np.argmin(fin))
        t = fin[i]
        rec.advance(min(t, math.nextafter(T, math.inf)), x)
        if t > T or rec.done:
            break
        g = problem.grad(i, Y[i], rng_d)
        ngrads += 1
        updated = False
        if reuse_table and phase == 2 and rem[i] and ent_ver[i] != ver[i]:
            G[i] = 0  # stale kept entry: start afresh at the worker's current point
            bG[i] = 0
            ent_ver[i] = ver[i]
        if phase == 1:
            G[i] += g
            bG[i] += 1
            S[i] = True
            if S.all():  # Phase 1 complete -> first update of Phase 2
                np.divide(G, bG[:, None].astype(x.dtype), out=A)
                m = A.mean(axis=0)
                rem[:] = True
                Gp[:] = 0
                bGp[:] = 0
                Sp[:] = False
                phase = 2
                updated = True
        else:
            if rem[i]:
                G[i] += g
                bG[i] += 1
                newA = G[i] / bG[i]
                m += (newA - A[i]) / n
                A[i] = newA
                updated = True
            else:
                Gp[i] += g
                bGp[i] += 1
                Sp[i] = True
        if updated:
            x -= gamma * m
            k += 1
            Y[i] = x
            ver[i] += 1
            rem[i] = False
            if log_steps:
                steps_t.append(t)
                steps_w.append(i)
            if not rem.any():  # Phase 2 complete -> transfer buffer, back to Phase 1
                if reuse_table:
                    G[Sp] = Gp[Sp]
                    bG[Sp] = bGp[Sp]
                    ent_ver[Sp] = ver[Sp]
                    np.divide(G, bG[:, None].astype(x.dtype), out=A)
                    m = A.mean(axis=0)
                    rem[:] = True
                    Gp[:] = 0
                    bGp[:] = 0
                    Sp[:] = False
                else:
                    G, Gp = Gp, G
                    bG, bGp = bGp, bG
                    S, Sp = Sp, S
                    phase = 1
        fin[i] = t + timing.draw(i)
    rec.advance(math.inf, x)
    return _result(rec, k, ngrads, steps_t, steps_w)


METHODS = {
    "rescaled": lambda P, taus, mode, gamma, T, grid, seed, **kw: run_asgd(
        P, taus, mode, rescaled_rule(gamma, taus), T, grid, seed, **kw),
    "vanilla": lambda P, taus, mode, gamma, T, grid, seed, **kw: run_asgd(
        P, taus, mode, vanilla_rule(gamma), T, grid, seed, **kw),
    "delay_adaptive": lambda P, taus, mode, gamma, T, grid, seed, **kw: run_asgd(
        P, taus, mode, delay_adaptive_rule(gamma, "paper"), T, grid, seed, **kw),
    "malenia": run_malenia,
    "ringleader": run_ringleader,
    "ringleader_reuse": lambda *a, **kw: run_ringleader(*a, reuse_table=True, **kw),
}

LABELS = {
    "rescaled": "Rescaled ASGD",
    "malenia": "Malenia SGD",
    "ringleader": "Ringleader ASGD",
    "vanilla": "Vanilla ASGD",
    "delay_adaptive": "Delay-Adaptive ASGD",
    "ringleader_reuse": "Ringleader ASGD (table-reuse variant)",
}
