"""Dataset loading, normalisation, train/val split and worker partitions."""
import os

import numpy as np

SHARED = os.environ.get("LOSCAR_DATA", "/home/funny36/Documents/mrtvc/shared-data")


def load_libsvm(path, n_features=None):
    """Minimal LIBSVM text parser (pure Python, no external deps). Returns dense X, y in {-1,+1}."""
    labels, rows, cols, vals = [], [], [], []
    with open(path) as f:
        for r, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            labels.append(float(parts[0]))
            for tok in parts[1:]:
                j, v = tok.split(":")
                rows.append(r)
                cols.append(int(j) - 1)
                vals.append(float(v))
    d = n_features or (max(cols) + 1)
    X = np.zeros((len(labels), d))
    X[np.asarray(rows), np.asarray(cols)] = np.asarray(vals)
    y = np.asarray(labels)
    y = np.where(y > 0, 1.0, -1.0)
    return X, y


def a9a(seed, val_frac=0.1, normalize=True):
    """a9a train file -> (Xtr, ytr, Xval, yval). 10% of the training data held out for validation
    (seeded), features standardised with training-split statistics."""
    X, y = load_libsvm(os.path.join(SHARED, "a9a", "a9a"), n_features=123)
    rng = np.random.default_rng(10_000 + seed)
    perm = rng.permutation(len(y))
    n_val = int(round(val_frac * len(y)))
    iv, it = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xv, yv = X[it], y[it], X[iv], y[iv]
    if normalize:
        mu = Xtr.mean(0)
        sd = Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Xtr = (Xtr - mu) / sd
        Xv = (Xv - mu) / sd
    return Xtr, ytr, Xv, yv


def iid_partition(m, n, rng):
    """Uniformly random equal-size shards."""
    return np.array_split(rng.permutation(m), n)


def dirichlet_partition(labels, n, alpha, rng, min_size=10, max_tries=1000):
    """Label-Dirichlet split (as in NIID-Bench / FedML): for every class draw proportions
    ~ Dir(alpha * 1_n) and cut that class's (shuffled) indices accordingly. Resample until
    every worker owns at least `min_size` examples."""
    classes = np.unique(labels)
    for _ in range(max_tries):
        shards = [[] for _ in range(n)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            prop = rng.dirichlet(alpha * np.ones(n))
            cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx, cuts)):
                shards[i].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [rng.permutation(np.asarray(s)) for s in shards]
    raise RuntimeError("could not draw a Dirichlet partition with the requested min size")


class BatchStream:
    """DataLoader-like stream over one shard: reshuffle every epoch, batches of size b,
    last (partial) batch kept (drop_last=False)."""

    def __init__(self, idx, b, rng):
        self.idx = np.asarray(idx)
        self.b = b
        self.rng = rng
        self._new_epoch()

    def _new_epoch(self):
        self.perm = self.idx[self.rng.permutation(len(self.idx))]
        self.pos = 0

    def next(self):
        if self.pos >= len(self.perm):
            self._new_epoch()
        out = self.perm[self.pos:self.pos + self.b]
        self.pos += self.b
        return out
