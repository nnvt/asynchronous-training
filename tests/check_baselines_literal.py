"""Check the fast IA^2SGD and Malenia SGD simulators against literal,
event-by-event transcriptions (dense gradients evaluated on arrival at the
worker's model), with deterministic per-client samples."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from rlsim.methods import Recorder, run_ia2sgd, run_malenia  # noqa: E402
from rlsim.problem import MLPProblem  # noqa: E402
from rlsim.timing import TimeModel  # noqa: E402


def ia2_literal(p, tm, T, gamma):
    n = p.n
    x = p.x0.copy()
    pt = [x] * n
    finish = tm.all()
    table = [None] * n
    k = 0
    while True:
        j = int(np.argmin(finish))
        t = finish[j]
        if t > T:
            break
        g = p.grad(pt[j], p.sample(j))
        table[j] = g
        if all(e is not None for e in table):  # initialised -> update on every arrival
            x = x - np.float32(gamma / n) * sum(table)
            pt[j] = x
            k += 1
        finish[j] = t + tm.one(j)
    return x, k


def malenia_literal(p, tm, T, gamma):
    """Event-driven: all workers restart at every synchronisation."""
    n = p.n
    x = p.x0.copy()
    t = 0.0
    k = 0
    while True:
        first = tm.all()
        R = first.max()
        if t + R > T:
            break
        # replay each worker's job sequence within the round (same draws as the fast version)
        M = int(np.ceil(R / tm.base[0])) + 1
        D = np.concatenate([first[:, None], tm.many((n, M - 1))], axis=1)
        acc = []
        for i in range(n):
            ends = np.cumsum(D[i])
            bi = int((ends <= R).sum())
            acc.append(sum(p.grad(x, p.sample(i)) for _ in range(bi)) / bi)
        x = x - np.float32(gamma) * sum(acc) / n
        k += 1
        t += R
    return x, k


def final_x(runner, p, seed, T, gamma):
    rec = Recorder(p, [T])
    box = {}
    orig = rec.close
    rec.close = lambda x: (box.setdefault("x", x.copy()), orig(x))
    info = runner(p, TimeModel(p.n, seed=seed), T, gamma, rec)
    return box["x"], info


def main():
    p = MLPProblem("mnist", n=10, seed=0)
    fixed = {j: p.clients[j, :8] for j in range(p.n)}
    p.batch = 8
    p.sample = lambda j, count=None: fixed[j] if count is None or count == 8 else np.tile(fixed[j], count // 8)
    T, gamma = 2000.0, 0.05
    xl, kl = ia2_literal(p, TimeModel(p.n, seed=7), T, gamma)
    xf, info = final_x(run_ia2sgd, p, 7, T, gamma)
    rel = np.linalg.norm(xf - xl) / np.linalg.norm(xl - p.x0)
    print(f"IA2SGD : iters literal {kl}, fast {info['iters']}; rel. diff {rel:.2e}")
    assert kl == info["iters"] and rel < 1e-3
    xl, kl = malenia_literal(p, TimeModel(p.n, seed=8), T, gamma)
    xf, info = final_x(run_malenia, p, 8, T, gamma)
    rel = np.linalg.norm(xf - xl) / np.linalg.norm(xl - p.x0)
    print(f"Malenia: iters literal {kl}, fast {info['iters']}; rel. diff {rel:.2e}")
    assert kl == info["iters"] and rel < 1e-4
    print("OK")


if __name__ == "__main__":
    main()
