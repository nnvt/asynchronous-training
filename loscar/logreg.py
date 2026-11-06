"""Binary logistic regression (eqs. 86-89) as a numpy 'problem' for loscar.core.run.

Model: nn.Linear(d_feat, 1)-style affine predictor, params w = (weights, bias) in R^{d_feat+1}.
Initialisation mimics PyTorch's default nn.Linear init (U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for
weights and bias); with standardised a9a features this gives an initial loss ~0.73, matching the
round-0 value in the paper's Fig. 1a.
"""
import numpy as np

from .data import BatchStream, iid_partition, dirichlet_partition


def _logloss_and_grad(w, X, y, lam=0.0, theta=1.0, need_grad=True):
    z = X @ w[:-1] + w[-1]
    m = y * z
    loss = np.mean(np.logaddexp(0.0, -m))
    if lam:
        wj2 = w[:-1] ** 2
        loss += lam * np.sum(wj2 / (wj2 + theta ** 2))
    if not need_grad:
        return loss, None, z
    s = -y * _sigmoid(-m)  # d loss / d z
    g = np.empty_like(w)
    g[:-1] = X.T @ s / len(y)
    g[-1] = s.mean()
    if lam:
        wj = w[:-1]
        g[:-1] += lam * 2 * wj * theta ** 2 / (wj ** 2 + theta ** 2) ** 2
    return loss, g, z


def _sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


class LogReg:
    def __init__(self, Xtr, ytr, Xval, yval, n_workers, batch=256, lr=0.1, partition="iid",
                 alpha=None, lam=0.0, theta=1.0, min_size=10, seed=0):
        self.Xtr, self.ytr, self.Xval, self.yval = Xtr, ytr, Xval, yval
        self.d = Xtr.shape[1] + 1
        self.batch, self.lr, self.lam, self.theta = batch, lr, lam, theta
        prng = np.random.default_rng([seed, 31])
        if partition == "iid":
            self.shards = iid_partition(len(ytr), n_workers, prng)
        elif partition == "dirichlet":
            self.shards = dirichlet_partition(ytr, n_workers, alpha, prng, min_size=min_size)
        else:
            raise ValueError(partition)

    # --- problem interface -------------------------------------------------------------
    def init(self, seed):
        rng = np.random.default_rng([seed, 1])
        bound = 1.0 / np.sqrt(self.d - 1)
        return rng.uniform(-bound, bound, size=self.d)

    def make_worker(self, i, seed):
        return BatchStream(self.shards[i], self.batch, np.random.default_rng([seed, 100 + i]))

    @staticmethod
    def copy(v):
        return v.copy()

    @staticmethod
    def mean(vs):
        return np.mean(vs, axis=0)

    @staticmethod
    def sqdist(a, b):
        return float(np.sum((a - b) ** 2))

    def local_steps(self, w, stream, k):
        w = w.copy()
        ex = 0
        for _ in range(k):
            idx = stream.next()
            _, g, _ = _logloss_and_grad(w, self.Xtr[idx], self.ytr[idx], self.lam, self.theta)
            w -= self.lr * g
            ex += len(idx)
        return w, ex

    def evaluate(self, w):
        ltr, gtr, ztr = _logloss_and_grad(w, self.Xtr, self.ytr, self.lam, self.theta)
        lv, _, zv = _logloss_and_grad(w, self.Xval, self.yval, self.lam, self.theta, need_grad=False)
        return {
            "train_loss": float(ltr),
            "val_loss": float(lv),
            "train_acc": float(np.mean(np.sign(ztr) == self.ytr)),
            "val_acc": float(np.mean(np.sign(zv) == self.yval)),
            "grad_norm": float(np.linalg.norm(gtr)),
        }
