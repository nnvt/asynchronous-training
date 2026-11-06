"""Simulated worker computation times.

Paper (App. G): tau_i = i + |eta_i|, eta_i ~ N(0, i), i in [n].
We read N(0, i) as variance i (std sqrt(i)).

Two readings of "simulated as" are supported:
  * per_job (default): a fresh duration is drawn every time a worker starts
    (or is restarted on) a stochastic-gradient computation;
  * fixed: each worker draws one duration at the start and keeps it forever
    (the classical "fixed computation model").
"""
import numpy as np


class TimeModel:
    def __init__(self, n, kind="per_job", seed=0, base="linear"):
        if kind not in ("per_job", "fixed"):
            raise ValueError(kind)
        self.n = n
        self.kind = kind
        i = np.arange(1, n + 1, dtype=np.float64)
        # base="linear": tau_i = i + |eta_i|,  eta_i ~ N(0, i)   (paper text)
        # base="sqrt":   tau_i = sqrt(i) + |eta_i|, eta_i ~ N(0, sqrt(i))  (diagnostic only)
        self.base = {"linear": i, "sqrt": np.sqrt(i)}[base]
        self.std = np.sqrt(self.base)
        self.rng = np.random.default_rng(seed)
        if kind == "fixed":
            self.fixed = self.base + np.abs(self.rng.normal(0.0, self.std))

    def sample(self, idx):
        """Durations for workers `idx` (int or int array, 0-based)."""
        if self.kind == "fixed":
            return self.fixed[idx]
        return self.base[idx] + np.abs(self.rng.normal(0.0, self.std[idx]))

    def sample_all(self):
        return self.sample(np.arange(self.n))
