"""Problems without data: 1-D quadratics of App. F and a schedule-only dummy."""
import numpy as np


class Quadratics1D:
    """F_i(x) = a_i (x - c_i)^2, deterministic gradients (sigma = 0)."""

    def __init__(self, a, c, x0):
        self.a = np.asarray(a, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.x0 = float(x0)
        self.d = 1

    def init(self, seed):
        return np.array([self.x0])

    def grad(self, i, x, rng):
        return 2.0 * self.a[i] * (x - self.c[i])

    def grad_multi(self, i, x, rng, nb):
        return self.grad(i, x, rng)

    def loss(self, x):  # equal-weighted F
        return float(np.mean(self.a * (x[0] - self.c) ** 2))

    def minimizer(self, w=None):
        w = np.ones_like(self.a) if w is None else np.asarray(w, dtype=float)
        return float(np.sum(w * self.a * self.c) / np.sum(w * self.a))


class ScheduleOnly:
    """Zero gradients; used to extract the update schedule (Fig. 2)."""

    d = 1

    def init(self, seed):
        return np.zeros(1)

    def grad(self, i, x, rng):
        return np.zeros(1)

    def grad_multi(self, i, x, rng, nb):
        return np.zeros(1)

    def loss(self, x):
        return 0.0
