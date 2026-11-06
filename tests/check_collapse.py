"""Diagnose the collapsed state: run Ringleader with a collapsing (tuned) and a
healthy stepsize for a short horizon and report the fraction of hidden ReLUs
that are inactive on every training sample, the training loss and accuracy."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from rlsim.methods import RUNNERS, Recorder  # noqa: E402
from rlsim.problem import MLPProblem  # noqa: E402
from rlsim.timing import TimeModel  # noqa: E402


def final_state(ds, method, gamma, T, seed=1):
    p = MLPProblem(ds, seed=seed)
    rec = Recorder(p, [T])
    box = {}
    orig = rec.close
    rec.close = lambda x: (box.setdefault("x", x.copy()), orig(x))
    RUNNERS[method](p, TimeModel(p.n, seed=[seed, 3]), T, gamma, rec)
    x = box["x"]
    W1, b1, W2, b2 = p.unpack(x)
    h = np.maximum(p.X @ W1 + b1, 0)
    dead = float(np.mean((h > 0).sum(0) == 0))
    acc = float(np.mean((h @ W2 + b2).argmax(1) == p.y))
    return dead, float(rec.loss[-1]), float(rec.g2[-1]), acc


if __name__ == "__main__":
    for ds, method, g, T in [("mnist", "ringleader", 0.05, 20000.0), ("mnist", "ringleader", 0.005, 20000.0),
                             ("fmnist", "ringleader", 0.1, 150000.0), ("fmnist", "ringleader", 0.005, 20000.0)]:
        dead, loss, g2, acc = final_state(ds, method, g, T)
        print(f"{ds:6s} {method} gamma={g:<6g} t={T:>8g}: dead hidden units {dead:6.1%}, train loss {loss:.3f}, "
              f"train acc {acc:.3f}, ||grad f||^2 {g2:.2e}")
