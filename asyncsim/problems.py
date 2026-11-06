"""Optimization problems from App. G of arXiv:2501.16168."""
import gzip
import os
import urllib.request

import numpy as np


class Quadratic:
    """f(x) = 1/2 x^T A x - b^T x with A = 1/4 tridiag(-1, 2, -1),
    b = 1/4 (-1, 0, ..., 0). Stochastic gradient: grad f(x) + xi,
    xi ~ N(0, noise_std^2 I_d). Starting point x0 = (sqrt(d), 0, ..., 0)
    as in the Rennala SGD experiments (Tyurin & Richtarik, 2023, App. J)."""

    def __init__(self, d=1729, noise_std=0.01, noise="iid", b_scale=1.0):
        if noise not in ("iid", "scalar"):
            raise ValueError(noise)
        self.d = d
        self.noise_std = noise_std
        # noise="iid":    xi ~ N(0, s^2 I_d)
        # noise="scalar": one xi ~ N(0, s^2) added to every coordinate
        #                 (what `grad + np.random.normal(0, s)` does in numpy)
        self.noise = noise
        self.b = np.zeros(d)
        self.b[0] = -0.25 * b_scale
        self.x0 = np.zeros(d)
        self.x0[0] = np.sqrt(d)
        # Closed form minimiser: tridiag(-1,2,-1) x = -e1  =>  x_j = -(d+1-j)/(d+1)
        j = np.arange(1, d + 1)
        self.xstar = -b_scale * (d + 1 - j) / (d + 1)
        assert np.allclose(self._Ax(self.xstar), self.b)
        self.fstar = 0.5 * self.xstar @ self._Ax(self.xstar) - self.b @ self.xstar
        self.L = 0.25 * (2 + 2 * np.cos(np.pi / (d + 1)))
        self.gap0 = self.gap(self.x0)
        self.blowup = 1e6 * self.gap0

    @staticmethod
    def _Ax(x):
        y = 2.0 * x
        y[1:] -= x[:-1]
        y[:-1] -= x[1:]
        return 0.25 * y

    def f(self, x):
        return 0.5 * x @ self._Ax(x) - self.b @ x

    def gap(self, x):
        """f(x) - f* computed as 1/2 (x-x*)^T A (x-x*) (no cancellation)."""
        e = x - self.xstar
        return 0.5 * e @ self._Ax(e)

    def grad(self, x, rng):
        g = self._Ax(x) - self.b
        if self.noise_std > 0:
            if self.noise == "iid":
                g += self.noise_std * rng.standard_normal(self.d)
            else:
                g += self.noise_std * rng.standard_normal()
        return g


# --------------------------------------------------------------------------
# MNIST, 2-layer ReLU MLP (App. G.1). Details not given in the paper are
# configurable; see README for the defaults chosen.
# --------------------------------------------------------------------------

MNIST_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
MNIST_FILES = {"x": "train-images-idx3-ubyte.gz", "y": "train-labels-idx1-ubyte.gz"}


def load_mnist(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    arrays = {}
    for key, fname in MNIST_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            last = None
            for base in MNIST_URLS:
                try:
                    urllib.request.urlretrieve(base + fname, path + ".part")
                    os.replace(path + ".part", path)
                    break
                except Exception as e:  # try next mirror
                    last = e
            else:
                raise RuntimeError(f"could not download {fname}: {last}")
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
        if key == "x":
            n = int.from_bytes(raw[4:8], "big")
            arrays[key] = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 784)
        else:
            arrays[key] = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return arrays["x"], arrays["y"]


class MnistMLP:
    """Mean cross-entropy of a 784-H-10 ReLU MLP on a (sub)set of MNIST
    train images. Parameters are a flat float64 vector. f_inf is taken as 0
    (cross-entropy is non-negative), so gap(x) = training loss."""

    def __init__(self, data_dir, hidden=32, batch=4, subset=3000, scale=1 / 255.0, seed=0):
        X, y = load_mnist(data_dir)
        rng = np.random.default_rng(seed)
        if subset and subset < len(y):
            idx = rng.choice(len(y), size=subset, replace=False)
            X, y = X[idx], y[idx]
        self.X = X.astype(np.float64) * scale
        self.y = y.astype(np.int64)
        self.N = len(self.y)
        self.H = hidden
        self.batch = batch
        self.shapes = [(784, hidden), (hidden,), (hidden, 10), (10,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.d = sum(self.sizes)
        # PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        parts = []
        for shape, fan_in in zip(self.shapes, [784, 784, hidden, hidden]):
            bound = 1 / np.sqrt(fan_in)
            parts.append(rng.uniform(-bound, bound, size=shape).ravel())
        self.x0 = np.concatenate(parts)
        self.gap0 = self.gap(self.x0)
        self.blowup = 1e8

    def _unpack(self, x):
        out, o = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(x[o:o + size].reshape(shape))
            o += size
        return out

    @staticmethod
    def _ce(z, y):
        """Per-sample cross-entropy, accurate down to ~1e-300 for confident,
        correct predictions (needed to see losses like 1e-14)."""
        rows = np.arange(len(y))
        m = z - z[rows, y][:, None]
        m[rows, y] = -np.inf
        a = np.maximum(m.max(axis=1), 0.0)
        s = np.exp(m - a[:, None]).sum(axis=1)
        return np.where(a == 0.0, np.log1p(s), a + np.log(np.exp(-a) + s))

    def gap(self, x):
        W1, b1, W2, b2 = self._unpack(x)
        h = np.maximum(self.X @ W1 + b1, 0.0)
        return float(self._ce(h @ W2 + b2, self.y).mean())

    def grad(self, x, rng):
        W1, b1, W2, b2 = self._unpack(x)
        idx = rng.integers(self.N, size=self.batch)
        Xb, yb = self.X[idx], self.y[idx]
        pre = Xb @ W1 + b1
        h = np.maximum(pre, 0.0)
        z = h @ W2 + b2
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(yb)), yb] -= 1.0
        dz = p / len(yb)
        gW2 = h.T @ dz
        gb2 = dz.sum(axis=0)
        dh = (dz @ W2.T) * (pre > 0)
        gW1 = Xb.T @ dh
        gb1 = dh.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
