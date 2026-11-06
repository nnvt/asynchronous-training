"""Event-driven simulation of Ringleader ASGD, Malenia SGD and IA^2SGD.

Model (Sec. 2 of the paper): n workers; worker j needs a (random) duration
from `TimeModel` per stochastic gradient; communication is instantaneous.
Simulated wall-clock time jumps from one gradient arrival to the next.

Stochastic gradients are *computed lazily but exactly*: a worker's gradient
is evaluated at the model it held when the computation started. Whenever a
table entry only averages gradients that were all computed at the same
point, G_j / b_j equals the mean-loss gradient over the union of the b_j
minibatches (all minibatches have the same size), so we evaluate it with one
batched matmul instead of b_j small ones; this is numerically the same
estimator (up to float32 summation order).

Implemented
  * ringleader -- Algorithm 1 (Phase 1 collection, Phase 2 one update per
                  worker, temporary buffer for the next round).
  * malenia    -- Malenia SGD (Tyurin & Richtarik, 2024) with the
                  parameter-free stopping rule of App. I (= condition (4)
                  with max(1, sigma^2/(n eps)) = 1): collect until every
                  worker has >= 1 gradient at x^k, update with the average of
                  per-worker averages, broadcast x^{k+1} to all workers and
                  restart every worker (ongoing computations are discarded).
  * ia2sgd     -- IA^2SGD (Wang et al., 2025) as described in Sec. 3: table
                  of the latest gradient of every worker, initialised with
                  one gradient per worker at x0; afterwards every arrival
                  overwrites its entry and triggers x <- x - gamma * mean(table);
                  the new model is sent to the arriving worker.
"""
import time

import numpy as np

from .blas import gemm_tn_acc


class Recorder:
    """Evaluates (||grad f||^2, f) on a time grid. The server iterate is
    piecewise constant in time; the value at grid time s is the iterate after
    all arrivals with time <= s."""

    def __init__(self, problem, grid, blowup=1e12):
        self.p = problem
        self.grid = np.asarray(grid, dtype=np.float64)
        self.g2 = np.full(len(self.grid), np.nan)
        self.loss = np.full(len(self.grid), np.nan)
        self.i = 0
        self.blowup = blowup
        self.diverged = False
        self.eval_time = 0.0
        self.next_t = self.grid[0] if len(self.grid) else np.inf

    def advance(self, t, x):
        """Record every grid point strictly before time t with iterate x.
        Returns True if the run diverged (caller should stop)."""
        if t <= self.next_t:
            return False
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):
            g2, loss = self.p.full_eval(x)
        self.eval_time += time.perf_counter() - t0
        if not np.isfinite(g2) or g2 > self.blowup or not np.all(np.isfinite(x[:16])):
            self.diverged = True
            self.g2[self.i:] = np.inf
            self.loss[self.i:] = np.inf
            self.i = len(self.grid)
            self.next_t = np.inf
            return True
        n = len(self.grid)
        while self.i < n and self.grid[self.i] < t:
            self.g2[self.i] = g2
            self.loss[self.i] = loss
            self.i += 1
        self.next_t = self.grid[self.i] if self.i < n else np.inf
        return False

    def close(self, x):
        if self.diverged:
            self.g2[self.i:] = np.inf
            self.loss[self.i:] = np.inf
            self.i = len(self.grid)
        if self.i < len(self.grid):
            self.advance(np.inf, x)


def _sum(arrs):
    s = arrs[0].copy()
    for a in arrs[1:]:
        s += a
    return s


def run_ringleader(p, tm, T, gamma, rec):
    n = p.n
    c = np.float32(gamma / n)
    x = p.x0.copy()
    pt = [x] * n                      # model each worker is computing at
    ptk = np.zeros(n, dtype=np.int64)  # iteration index of that model
    finish = tm.all()
    b = np.zeros(n, dtype=np.int64)    # main table counts (lazy: gradients are
    bplus = np.zeros(n, dtype=np.int64)  # evaluated when first needed)
    avg = [None] * n
    inS = np.zeros(n, dtype=bool)
    inSp = np.zeros(n, dtype=bool)
    nS = 0
    phase = 1
    Ssum = None
    k = rounds = 0
    max_delay = 0
    with np.errstate(all="ignore"):
        while True:
            j = int(np.argmin(finish))
            t = finish[j]
            if t > T or rec.advance(t, x):
                break
            if phase == 1:
                b[j] += 1
                if not inS[j]:
                    inS[j] = True
                    nS += 1
                if nS == n:  # table complete -> Phase 2, first update for worker j
                    avg = [p.grad(pt[i], p.sample(i, p.batch * int(b[i]))) for i in range(n)]
                    Ssum = _sum(avg)
                    # every main-table entry stays at its worker's point of
                    # this moment for the whole round -> delays k - ent_min
                    ent_min = int(ptk.min())
                    max_delay = max(max_delay, k - ent_min)
                    x = x - c * Ssum
                    k += 1
                    pt[j] = x
                    ptk[j] = k
                    inS[j] = False
                    nS -= 1
                    bplus[:] = 0
                    inSp[:] = False
                    phase = 2
            else:
                if inS[j]:
                    g = p.grad(pt[j], p.sample(j))
                    bj = int(b[j])
                    new = avg[j] + (g - avg[j]) / np.float32(bj + 1)  # (b*avg + g)/(b+1)
                    Ssum += new
                    Ssum -= avg[j]
                    avg[j] = new
                    b[j] = bj + 1
                    max_delay = max(max_delay, k - ent_min)
                    x = x - c * Ssum
                    k += 1
                    pt[j] = x
                    ptk[j] = k
                    inS[j] = False
                    nS -= 1
                    if nS == 0:  # end of round: transfer buffer to main table
                        b, bplus = bplus.copy(), np.zeros(n, dtype=np.int64)
                        inS, inSp = inSp.copy(), np.zeros(n, dtype=bool)
                        nS = int(inS.sum())
                        avg = [None] * n
                        phase = 1
                        rounds += 1
                        if not np.isfinite(x[:64]).all():
                            rec.diverged = True
                            break
                else:
                    bplus[j] += 1
                    inSp[j] = True
            finish[j] = t + tm.one(j)
    rec.close(x)
    return dict(iters=k, rounds=rounds, max_delay=max_delay)


def run_malenia(p, tm, T, gamma, rec):
    n = p.n
    x = p.x0.copy()
    t0 = 0.0
    k = 0
    nb = p.batch
    grads_used = 0
    with np.errstate(all="ignore"):
        while True:
            d1 = tm.all()
            R = float(d1.max())
            t_end = t0 + R
            if t_end > T or rec.advance(t_end, x):
                break
            # number of gradients each worker finishes at x^k before t_end
            M = int(np.ceil(R / tm.base[0])) + 1
            D = np.concatenate([d1[:, None], tm.many((n, M - 1))], axis=1)
            b = (np.cumsum(D, axis=1) <= R).sum(axis=1)
            idx = np.concatenate([p.sample(i, nb * int(b[i])) for i in range(n)])
            w = np.repeat((1.0 / (n * nb * b)).astype(np.float32), nb * b)
            g = p.grad(x, idx, w)
            x = x - np.float32(gamma) * g
            k += 1
            grads_used += int(b.sum())
            t0 = t_end
            if k % 20 == 0 and not np.isfinite(x[:64]).all():
                rec.diverged = True
                break
    rec.close(x)
    return dict(iters=k, rounds=k, grads_used=grads_used)


def run_ia2sgd(p, tm, T, gamma, rec, resum_every=2000):
    """IA^2SGD. Each worker's gradient is evaluated eagerly at job start (it
    only depends on the model the worker received and its minibatch) and kept
    in factored form (idx, dh, small) with gW1 = X[idx]^T dh, so the server
    model can be updated in place and no snapshots are stored."""
    n = p.n
    c = np.float32(gamma / n)
    o1 = int(p.offsets[1])
    W1shape = p.shapes[0]
    x = p.x0.copy()
    jobs = [p.job(x, i) for i in range(n)]   # all workers start at x0
    jobk = np.zeros(n, dtype=np.int64)
    finish = tm.all()
    table = [None] * n
    filled = 0
    S = None
    k = 0
    max_delay = 0
    X = p.X

    def dense_sum():  # c * sum of the table, dense
        idx = np.concatenate([e[0] for e in table])
        dh = np.concatenate([e[1] for e in table])
        out = np.empty(p.d, dtype=np.float32)
        np.matmul(X[idx].T, dh, out=out[:o1].reshape(W1shape))
        out[o1:] = _sum([e[2] for e in table])
        out *= c
        return out

    # S holds c * sum_i g_i, so that x <- x - S is the IA^2SGD step
    with np.errstate(all="ignore"):
        while True:
            j = int(np.argmin(finish))
            t = finish[j]
            if t > T or rec.advance(t, x):
                break
            new = jobs[j]
            if filled < n:  # initialisation: one gradient per worker at x0
                if table[j] is None:
                    filled += 1
                table[j] = new
                if filled < n:
                    jobs[j] = p.job(x, j)  # still at x0
                    finish[j] = t + tm.one(j)
                    continue
                S = dense_sum()
                SW = S[:o1].reshape(W1shape)
            else:
                d = k - int(jobk[j])
                if d > max_delay:
                    max_delay = d
                old = table[j]
                table[j] = new
                # S += c (g_new - g_old); W1 part as one rank-2b accumulate
                U = X[np.concatenate([new[0], old[0]])]
                V = np.concatenate([new[1], -old[1]])
                gemm_tn_acc(U, V, SW, c, 1.0)
                S[o1:] += c * (new[2] - old[2])
            x -= S          # x <- x - gamma/n * sum_i g_i  (in place), sent to worker j
            k += 1
            if k % resum_every == 0:  # limit float32 drift of the running sum
                S = dense_sum()
                SW = S[:o1].reshape(W1shape)
                if not np.isfinite(x[:64]).all():
                    rec.diverged = True
                    break
            jobs[j] = p.job(x, j)
            jobk[j] = k
            finish[j] = t + tm.one(j)
    rec.close(x)
    return dict(iters=k, max_delay=max_delay)


RUNNERS = {"ringleader": run_ringleader, "malenia": run_malenia, "ia2sgd": run_ia2sgd}
