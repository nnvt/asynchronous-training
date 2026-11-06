"""Heterogeneous MNIST / Fashion-MNIST problem of App. A (Ringleader ASGD).

* Dataset: the 60k training images, "standard normalization" (dataset-wide
  mean/std of the [0,1] pixels: MNIST 0.1307/0.3081, FMNIST 0.2860/0.3530).
* n = 100 clients, equal-size Dirichlet(alpha = 0.1) label partition with the
  rounding / top-up procedure described in App. A.
* Model: Linear(784,128) -> ReLU -> Linear(128,10), mean cross-entropy,
  PyTorch default init (U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights and
  biases). Parameters are one flat float32 vector (d = 101,770).
* f(x) = 1/n sum_i f_i(x) = mean loss over all N samples (equal client sizes),
  so the reported ||grad f(x)||^2 is the full-batch gradient norm squared.
"""
import gzip
import os

import numpy as np

DATA_ROOT = os.environ.get("RL_DATA", "/home/funny36/Documents/mrtvc/shared-data")
DATASETS = {
    "mnist": os.path.join(DATA_ROOT),
    "fmnist": os.path.join(DATA_ROOT, "fashion-mnist"),
}
_CACHE = {}


def load_dataset(name):
    """Normalized float32 images (N, 784) and int64 labels (N,)."""
    if name in _CACHE:
        return _CACHE[name]
    d = DATASETS[name]
    with gzip.open(os.path.join(d, "train-images-idx3-ubyte.gz"), "rb") as fh:
        raw = fh.read()
    X = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    with gzip.open(os.path.join(d, "train-labels-idx1-ubyte.gz"), "rb") as fh:
        raw = fh.read()
    y = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    mean, std = float(X.mean()), float(X.std())
    X = (X - np.float32(mean)) / np.float32(std)
    _CACHE[name] = (np.ascontiguousarray(X, dtype=np.float32), y, mean, std)
    return _CACHE[name]


def dirichlet_equal_partition(y, n, alpha, rng, num_classes=10):
    """Equal-size Dirichlet label partition (App. A).

    For each client j: p_j ~ Dir(alpha,...,alpha); a rounded class-allocation
    vector summing exactly to N/n (largest-remainder rounding); take the
    requested number from each class pool; if a pool runs dry, the shortfall
    is topped up from the other classes that still have examples (visited in
    random order). Returns an (n, N/n) int array of sample indices.
    """
    N = len(y) - len(y) % n  # trim so that N is divisible by n
    y = y[:N]
    m = N // n
    pools = [list(rng.permutation(np.nonzero(y == c)[0])) for c in range(num_classes)]
    out = np.empty((n, m), dtype=np.int64)
    for j in range(n):
        p = rng.dirichlet(alpha * np.ones(num_classes))
        raw = p * m
        alloc = np.floor(raw).astype(int)
        rem = m - alloc.sum()
        if rem > 0:
            alloc[np.argsort(-(raw - alloc), kind="stable")[:rem]] += 1
        take = []
        for c in range(num_classes):
            k = min(alloc[c], len(pools[c]))
            take.extend(pools[c][:k])
            del pools[c][:k]
        short = m - len(take)
        if short > 0:
            for c in rng.permutation(num_classes):
                if short == 0:
                    break
                k = min(short, len(pools[c]))
                take.extend(pools[c][:k])
                del pools[c][:k]
                short -= k
        assert len(take) == m
        out[j] = rng.permutation(np.array(take, dtype=np.int64))
    return out


class MLPProblem:
    def __init__(self, dataset="mnist", n=100, alpha=0.1, batch=4, hidden=128, seed=0):
        X, y, self.mean, self.std = load_dataset(dataset)
        self.dataset = dataset
        rng = np.random.default_rng([seed, 1])
        self.clients = dirichlet_equal_partition(y, n, alpha, rng)
        used = self.clients.ravel()
        # The full objective is the mean over the (trimmed) union of the clients.
        self.X, self.y = X, y
        self.all_idx = np.sort(used)
        self.full_all = len(self.all_idx) == len(y)
        self.n, self.m = self.clients.shape
        self.batch = batch
        self.H = H = hidden
        self.shapes = [(784, H), (H,), (H, 10), (10,)]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.offsets = np.cumsum([0] + self.sizes)
        self.d = int(self.offsets[-1])
        parts = []
        for shape, fan_in in zip(self.shapes, [784, 784, H, H]):
            bound = 1.0 / np.sqrt(fan_in)
            parts.append(rng.uniform(-bound, bound, size=shape).ravel())
        self.x0 = np.concatenate(parts).astype(np.float32)
        # sampling stream (i.i.d. uniform draws, with replacement, from the client's data)
        self.rng = np.random.default_rng([seed, 2])
        self._blk = 1 << 18
        self._ints = self.rng.integers(0, self.m, size=self._blk)
        self._pos = 0

    # ------------------------------------------------------------ helpers --
    def unpack(self, x):
        o = self.offsets
        return (x[o[0]:o[1]].reshape(self.shapes[0]), x[o[1]:o[2]],
                x[o[2]:o[3]].reshape(self.shapes[2]), x[o[3]:o[4]])

    def sample(self, j, count=None):
        """`count` (default: batch) i.i.d. uniform sample indices of client j."""
        c = self.batch if count is None else count
        if self._pos + c > self._blk:
            self._ints = self.rng.integers(0, self.m, size=max(self._blk, c))
            self._pos = 0
        r = self._ints[self._pos:self._pos + c]
        self._pos += c
        return self.clients[j, r]

    def label_hist(self):
        return np.stack([np.bincount(self.y[c], minlength=10) for c in self.clients])

    # ----------------------------------------------------------- gradients --
    def _backprop(self, x, Xb, yb, w):
        """Gradient of sum_s w_s * CE_s at x; returns (flat grad, sum_s w_s*CE_s)."""
        W1, b1, W2, b2 = self.unpack(x)
        pre = Xb @ W1
        pre += b1
        h = np.maximum(pre, 0.0)
        z = h @ W2
        z += b2
        z -= z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        se = ez.sum(axis=1, keepdims=True)
        rows = np.arange(len(yb))
        loss = float(np.sum(w * (np.log(se[:, 0]) - z[rows, yb]), dtype=np.float64))
        p = ez / se
        p[rows, yb] -= 1.0
        dz = p * w[:, None] if w.ndim else p * w
        out = np.empty(self.d, dtype=np.float32)
        o = self.offsets
        np.matmul(h.T, dz, out=out[o[2]:o[3]].reshape(self.shapes[2]))
        out[o[3]:o[4]] = dz.sum(axis=0)
        dh = dz @ W2.T
        dh *= pre > 0
        np.matmul(Xb.T, dh, out=out[o[0]:o[1]].reshape(self.shapes[0]))
        out[o[1]:o[2]] = dh.sum(axis=0)
        return out, loss

    def job(self, x, j):
        """Eagerly evaluate the batch-`batch` stochastic gradient of client j
        at x in factored form: (idx, dh, gb1, gW2, gb2) with
        gW1 = X[idx]^T dh. Used by IA^2SGD so that no model snapshots are needed
        (the gradient only depends on the model at job start and the batch)."""
        idx = self.sample(j)
        W1, b1, W2, b2 = self.unpack(x)
        Xb, yb = self.X[idx], self.y[idx]
        pre = Xb @ W1
        pre += b1
        h = np.maximum(pre, 0.0)
        z = h @ W2
        z += b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        p[np.arange(len(yb)), yb] -= 1.0
        p *= np.float32(1.0 / len(yb))
        dh = p @ W2.T
        dh *= pre > 0
        small = np.empty(self.d - self.offsets[1], dtype=np.float32)  # [gb1, gW2, gb2]
        o = self.offsets - self.offsets[1]
        small[o[1]:o[2]] = dh.sum(axis=0)
        np.matmul(h.T, p, out=small[o[2]:o[3]].reshape(self.shapes[2]))
        small[o[3]:o[4]] = p.sum(axis=0)
        return idx, dh, small

    def grad(self, x, idx, w=None):
        """Mean-CE gradient over samples `idx` (or weighted sum with weights w)."""
        if w is None:
            w = np.float32(1.0 / len(idx))
        return self._backprop(x, self.X[idx], self.y[idx], w)[0]

    def full_eval(self, x, chunk=20000):
        """(||grad f(x)||^2, f(x)) over the full (trimmed) training set."""
        g = np.zeros(self.d, dtype=np.float64)
        loss = 0.0
        idx_all = self.all_idx
        N = len(idx_all)
        w = np.float32(1.0 / N)
        for s in range(0, N, chunk):
            if self.full_all:
                Xb, yb = self.X[s:s + chunk], self.y[s:s + chunk]
            else:
                ii = idx_all[s:s + chunk]
                Xb, yb = self.X[ii], self.y[ii]
            gi, li = self._backprop(x, Xb, yb, w)
            g += gi
            loss += li
        return float(g @ g), loss
