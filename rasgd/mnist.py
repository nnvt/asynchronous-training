"""Label-partitioned MNIST + 2-layer ReLU MLP (Sec. 5 / App. E of arXiv:2605.13434).

The MLP forward/backward code is adapted from `asyncsim/problems.py` on the
sibling branch `ringmaster-asgd` (homogeneous-data version); here it is made
float32, worker-local (each worker samples only from its own shard) and the
reported loss is the equal-weighted global objective F = 1/n sum_i F_i.
"""
import gzip
import os

import numpy as np

DEFAULT_DATA_DIR = "/home/funny36/Documents/mrtvc/shared-data"
FILES = {"x": "train-images-idx3-ubyte.gz", "y": "train-labels-idx1-ubyte.gz"}
MEAN, STD = 0.1307, 0.3081  # "standard normalization" (torchvision MNIST constants)


def load_mnist(data_dir=DEFAULT_DATA_DIR):
    out = {}
    for key, fname in FILES.items():
        with gzip.open(os.path.join(data_dir, fname), "rb") as fh:
            raw = fh.read()
        if key == "x":
            n = int.from_bytes(raw[4:8], "big")
            out[key] = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 784)
        else:
            out[key] = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return out["x"], out["y"]


class LabelPartitionedMLP:
    """n = 10 workers; worker i holds only images of digit `classes[i]`.

    Every class is trimmed to the size of the smallest class (5421 = #digit 5),
    so all workers hold the same number of samples and F (the mean loss over
    the 54,210 kept images) equals 1/n sum_i F_i exactly.
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR, hidden=128, batch=64,
                 classes=tuple(range(10)), trim_seed=0, dtype=np.float32):
        X, y = load_mnist(data_dir)
        X = ((X.astype(np.float32) / 255.0) - MEAN) / STD
        per = min(np.bincount(y, minlength=10))  # 5421
        rng = np.random.default_rng(trim_seed)
        shards = []
        for c in classes:
            idx = np.flatnonzero(y == c)
            idx = np.sort(rng.choice(idx, size=per, replace=False)) if len(idx) > per else idx
            shards.append(idx)
        self.n = len(classes)
        self.per = per
        order = np.concatenate(shards)
        self.X = np.ascontiguousarray(X[order], dtype=dtype)
        self.y = y[order].astype(np.int64)
        # worker i owns rows [i*per, (i+1)*per)
        self.offsets = np.arange(self.n) * per
        self.H = hidden
        self.batch = batch
        self.dtype = dtype
        self.shapes = [(784, hidden), (hidden,), (hidden, 10), (10,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.d = sum(self.sizes)

    # -- parameters ---------------------------------------------------------
    def init(self, seed):
        """PyTorch nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
        rng = np.random.default_rng(10_000 + seed)
        parts = []
        for shape, fan_in in zip(self.shapes, [784, 784, self.H, self.H]):
            b = 1.0 / np.sqrt(fan_in)
            parts.append(rng.uniform(-b, b, size=shape).ravel())
        return np.concatenate(parts).astype(self.dtype)

    def _unpack(self, x):
        out, o = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(x[o:o + size].reshape(shape))
            o += size
        return out

    # -- objective ----------------------------------------------------------
    def loss(self, x, chunk=20000):
        """Global objective F(x): mean cross-entropy over all 54,210 images."""
        W1, b1, W2, b2 = self._unpack(x)
        tot = 0.0
        for s in range(0, len(self.y), chunk):
            Xc, yc = self.X[s:s + chunk], self.y[s:s + chunk]
            h = np.maximum(Xc @ W1 + b1, 0.0)
            z = (h @ W2 + b2).astype(np.float64)
            zmax = z.max(axis=1, keepdims=True)
            lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
            tot += float((lse - z[np.arange(len(yc)), yc]).sum())
        return tot / len(self.y)

    def sample(self, i, rng, nbatches=1):
        """Indices of `nbatches` independent local minibatches of worker i
        (each drawn uniformly without replacement from worker i's shard)."""
        if nbatches == 1:
            loc = rng.choice(self.per, size=self.batch, replace=False)
        else:
            loc = np.concatenate([rng.choice(self.per, size=self.batch, replace=False)
                                  for _ in range(nbatches)])
        return loc + self.offsets[i]

    def grad_idx(self, x, idx):
        """Mean cross-entropy gradient over rows `idx` (= average of the
        minibatch gradients when idx is a concatenation of equal-size batches)."""
        W1, b1, W2, b2 = self._unpack(x)
        Xb, yb = self.X[idx], self.y[idx]
        pre = Xb @ W1 + b1
        h = np.maximum(pre, 0.0)
        z = h @ W2 + b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(yb)), yb] -= 1.0
        dz = p / np.float32(len(yb))
        gW2 = h.T @ dz
        gb2 = dz.sum(axis=0)
        dh = (dz @ W2.T) * (pre > 0)
        gW1 = Xb.T @ dh
        gb1 = dh.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])

    def grad(self, i, x, rng):
        return self.grad_idx(x, self.sample(i, rng))

    def grad_multi(self, i, x, rng, nb):
        """Average of nb independent local minibatch gradients at the same x."""
        return self.grad_idx(x, self.sample(i, rng, nb))
