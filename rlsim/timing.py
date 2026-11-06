"""Simulated worker computation times.

Derived from asyncsim/timing.py on branch `ringmaster-asgd` (sibling
replication of arXiv:2501.16168), adapted for block-wise random draws.

Paper (App. A): tau_i = i + |eta_i|, eta_i ~ N(0, i), i in [n].
We read N(0, i) as variance i (std sqrt(i)).

Two readings of "assigning each client i a base delay and a random jitter":
  * per_job (default): a fresh duration is drawn every time a worker starts
    a stochastic-gradient computation;
  * fixed: each worker draws one duration at the start of the run and keeps
    it forever (the "fixed computation model" (2) used by the theory).
"""
import numpy as np


class TimeModel:
    def __init__(self, n, kind="per_job", seed=0, block=1 << 16):
        if kind not in ("per_job", "fixed"):
            raise ValueError(kind)
        self.n = n
        self.kind = kind
        i = np.arange(1, n + 1, dtype=np.float64)
        self.base = i
        self.std = np.sqrt(i)
        self.rng = np.random.default_rng(seed)
        self.fixed = self.base + np.abs(self.rng.normal(0.0, self.std))
        self._block = block
        self._buf = np.abs(self.rng.standard_normal(block))
        self._pos = 0

    def _normals(self, m):
        if self._pos + m > self._block:
            self._buf = np.abs(self.rng.standard_normal(self._block))
            self._pos = 0
        out = self._buf[self._pos:self._pos + m]
        self._pos += m
        return out

    def one(self, j):
        """Duration of the next job of worker j (0-based)."""
        if self.kind == "fixed":
            return self.fixed[j]
        if self._pos >= self._block:
            self._buf = np.abs(self.rng.standard_normal(self._block))
            self._pos = 0
        z = self._buf[self._pos]
        self._pos += 1
        return self.base[j] + self.std[j] * z

    def many(self, shape):
        """Durations of an (n, m) array of jobs: row j = next m jobs of worker j."""
        n, m = shape
        if self.kind == "fixed":
            return np.repeat(self.fixed[:, None], m, axis=1)
        z = self.rng.standard_normal((n, m))
        return self.base[:, None] + self.std[:, None] * np.abs(z)

    def all(self):
        if self.kind == "fixed":
            return self.fixed.copy()
        return self.base + self.std * np.abs(self.rng.standard_normal(self.n))
