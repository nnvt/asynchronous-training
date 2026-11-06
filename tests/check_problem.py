"""Sanity checks: partition, init grad norm, gradient vs finite differences."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rlsim.problem import MLPProblem

for ds in ["mnist", "fmnist"]:
    t0 = time.perf_counter()
    p = MLPProblem(ds, seed=0)
    print(ds, "setup %.2fs" % (time.perf_counter() - t0), "d =", p.d, "clients", p.clients.shape,
          "mean/std %.4f/%.4f" % (p.mean, p.std))
    H = p.label_hist()
    assert (H.sum(1) == p.m).all() and len(np.unique(p.clients)) == p.n * p.m
    top = np.sort(H, axis=1)[:, ::-1]
    print("  share of top-1 / top-2 class per client (median): %.2f / %.2f" %
          (np.median(top[:, 0] / p.m), np.median(top[:, :2].sum(1) / p.m)),
          " #classes with >=5%% of client data (median): %d" % np.median((H >= 0.05 * p.m).sum(1)))
    t0 = time.perf_counter()
    g2, loss = p.full_eval(p.x0)
    print("  ||grad f(x0)||^2 = %.4f, f(x0) = %.4f, eval %.3fs" % (g2, loss, time.perf_counter() - t0))
    # finite-difference check in float64 on a small batch
    idx = p.sample(3, 16)
    x = p.x0.astype(np.float64)
    def f(z):
        W1, b1, W2, b2 = p.unpack(z)
        hh = np.maximum(p.X[idx].astype(np.float64) @ W1 + b1, 0)
        zz = hh @ W2 + b2
        zz = zz - zz.max(1, keepdims=True)
        return float(np.mean(np.log(np.exp(zz).sum(1)) - zz[np.arange(len(idx)), p.y[idx]]))
    g = p.grad(p.x0, idx).astype(np.float64)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.standard_normal(p.d); v /= np.linalg.norm(v)
        e = 1e-4
        fd = (f(x + e * v) - f(x - e * v)) / (2 * e)
        print("  dir-deriv fd=%.6e analytic=%.6e" % (fd, g @ v))
    t0 = time.perf_counter()
    for _ in range(2000):
        p.grad(p.x0, p.sample(5))
    print("  batch-4 grad: %.1f us" % ((time.perf_counter() - t0) / 2000 * 1e6))
    t0 = time.perf_counter()
    for _ in range(50):
        p.grad(p.x0, p.sample(5, 400))
    print("  batch-400 grad: %.1f us" % ((time.perf_counter() - t0) / 50 * 1e6))
