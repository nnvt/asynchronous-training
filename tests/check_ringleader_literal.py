"""Check the lazy/batched Ringleader simulator against a literal line-by-line
transcription of Algorithm 1 (dense G_i accumulation, every gradient
evaluated on arrival at the worker's model). Sampling is made deterministic
(each client always returns the same 8 samples) so both must give the same
iterates up to float32 rounding; also checks the delay bound of Lemma 4.1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from rlsim.methods import Recorder, run_ringleader  # noqa: E402
from rlsim.problem import MLPProblem  # noqa: E402
from rlsim.timing import TimeModel  # noqa: E402


def literal(p, tm, T, gamma):
    n = p.n
    x = p.x0.copy()
    pt = [x] * n
    ptk = np.zeros(n, dtype=int)
    finish = tm.all()
    k = 0
    S = set()
    G = [np.zeros(p.d, np.float32) for _ in range(n)]
    b = np.zeros(n, int)
    max_delay = 0

    def receive():
        j = int(np.argmin(finish))
        t = finish[j]
        if t > T:
            raise StopIteration
        g = p.grad(pt[j], p.sample(j))
        finish[j] = t + tm.one(j)
        return j, g

    def update(j):
        nonlocal x, k, max_delay
        max_delay = max(max_delay, k - min(ent_k))
        x = x - np.float32(gamma / n) * sum(G[i] / b[i] for i in range(n))
        pt[j] = x
        k += 1
        ptk[j] = k

    try:
        while True:
            while S != set(range(n)):                       # Phase 1
                j, g = receive()
                G[j] = G[j] + g
                b[j] += 1
                S.add(j)
            ent_k = ptk.copy()
            update(j)                                       # Phase 2, line 11-13
            S.discard(j)
            Gp = [np.zeros(p.d, np.float32) for _ in range(n)]
            bp = np.zeros(n, int)
            Sp = set()
            while S:
                j, g = receive()
                if j in S:
                    G[j] = G[j] + g
                    b[j] += 1
                    update(j)
                    S.discard(j)
                else:
                    Gp[j] = Gp[j] + g
                    bp[j] += 1
                    Sp.add(j)
            G, b, S = Gp, bp, Sp                            # line 26
    except StopIteration:
        pass
    return x, k, max_delay


def main():
    p = MLPProblem("mnist", n=10, seed=0)
    fixed = {j: p.clients[j, :8] for j in range(p.n)}
    p.sample = lambda j, count=None: fixed[j]
    T = 3000.0
    x_lit, k_lit, d_lit = literal(p, TimeModel(p.n, seed=5), T, 0.05)
    rec = Recorder(p, [0.0])
    info = run_ringleader(p, TimeModel(p.n, seed=5), T, 0.05, rec)
    # recover final x of the fast implementation by re-running with a recorder at T
    rec2 = Recorder(p, [T])
    xs = {}
    orig_close = rec2.close
    rec2.close = lambda x: (xs.setdefault("x", x.copy()), orig_close(x))
    run_ringleader(p, TimeModel(p.n, seed=5), T, 0.05, rec2)
    x_fast = xs["x"]
    rel = np.linalg.norm(x_fast - x_lit) / np.linalg.norm(x_lit - p.x0)
    print(f"iterations: literal {k_lit}, fast {info['iters']}; max delay literal {d_lit}, fast {info['max_delay']}"
          f" (Lemma 4.1 bound 2n-2 = {2 * p.n - 2}); ||x_fast - x_lit|| / ||x_lit - x0|| = {rel:.2e}")
    assert k_lit == info["iters"] and d_lit == info["max_delay"] and d_lit <= 2 * p.n - 2 and rel < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
