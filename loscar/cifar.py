"""CIFAR-10 CNN 'problem' (PyTorch, CPU) for loscar.core.run (paper App. B.6, scaled down).

Each logical worker is a separate copy of the parameters (a flat float32 vector); local SGD steps
load the vector into one shared nn.Module, take plain-SGD steps (weight decay 5e-4 added to the
gradient, lr 0.05, batch 256, random crop (pad 4) + horizontal flip), and read the vector back.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import SHARED, BatchStream

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


class _SafeUnpickler(pickle.Unpickler):
    """Only allow the numpy reconstruction globals the CIFAR batches need (no arbitrary code)."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_cifar10():
    base = os.path.join(SHARED, "cifar10", "cifar-10-batches-py")
    xs, ys = [], []
    for k in range(1, 6):
        with open(os.path.join(base, f"data_batch_{k}"), "rb") as f:
            dct = _SafeUnpickler(f, encoding="bytes").load()
        xs.append(np.asarray(dct[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        ys.append(np.asarray(dct[b"labels"], dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys)


class SmallCNN(nn.Module):
    """Stride-2 3x3 conv stem + 3 conv blocks (conv3x3-ReLU-maxpool) -> fc(128) + dropout(0.1)
    -> fc(10). With widths (16, 32, 64): 57.8 k params. The stride-2 stem was chosen purely for CPU
    cost (~55 ms per batch-256 step on 2 busy cores)."""

    def __init__(self, w=(16, 32, 64), n_classes=10, dropout=0.1, stem_stride=2):
        super().__init__()
        self.c1 = nn.Conv2d(3, w[0], 3, stride=stem_stride, padding=1)
        self.c2 = nn.Conv2d(w[0], w[1], 3, padding=1)
        self.c3 = nn.Conv2d(w[1], w[2], 3, padding=1)
        self.drop = nn.Dropout(dropout)
        side = 32 // stem_stride // 8
        self.fc1 = nn.Linear(w[2] * side * side, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


def augment(xb, gen):
    """Random crop with zero padding 4 + random horizontal flip, batched."""
    B = xb.shape[0]
    xp = F.pad(xb, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (B,), generator=gen)
    dy = torch.randint(0, 9, (B,), generator=gen)
    ar = torch.arange(32)
    rows = (dy[:, None] + ar[None, :])[:, None, :, None].expand(B, 3, 32, 40)
    cols = (dx[:, None] + ar[None, :])[:, None, None, :].expand(B, 3, 32, 32)
    out = torch.gather(xp, 2, rows)    # (B,3,32,40): crop rows
    out = torch.gather(out, 3, cols)   # (B,3,32,32): crop cols
    flip = torch.rand(B, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


class Cifar:
    def __init__(self, n_workers, seed, batch=256, lr=0.05, wd=5e-4, val_frac=0.1, n_train=None,
                 eval_n=5000, width=(16, 32, 64), threads=None):
        if threads:
            torch.set_num_threads(threads)
        X, y = load_cifar10()
        rng = np.random.default_rng([seed, 5])
        perm = rng.permutation(len(y))
        n_val = int(round(val_frac * len(y)))
        iv, it = perm[:n_val], perm[n_val:]
        if n_train:
            it = it[:n_train]
        self.Xtr, self.ytr = torch.from_numpy(X[it]), torch.from_numpy(y[it])
        self.Xv, self.yv = torch.from_numpy(X[iv]), torch.from_numpy(y[iv])
        self.ev_tr = torch.from_numpy(rng.choice(len(it), size=min(eval_n, len(it)), replace=False))
        self.batch, self.lr, self.wd = batch, lr, wd
        self.width = width
        self.model = SmallCNN(width)
        self.d = sum(p.numel() for p in self.model.parameters())
        # data-homogeneous: every worker samples from the whole training split (paper B.6)
        self.n_workers = n_workers

    # --- problem interface -------------------------------------------------------------
    def init(self, seed):
        torch.manual_seed(seed)
        m = SmallCNN(self.width)
        return torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone()

    def make_worker(self, i, seed):
        rng = np.random.default_rng([seed, 200 + i])
        gen = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
        return {"stream": BatchStream(np.arange(len(self.ytr)), self.batch, rng), "gen": gen}

    @staticmethod
    def as_index(S):
        return torch.from_numpy(S)

    @staticmethod
    def copy(v):
        return v.clone()

    @staticmethod
    def mean(vs):
        return torch.stack(vs).mean(0)

    @staticmethod
    def sqdist(a, b):
        return float(((a - b) ** 2).sum())

    def _prep(self, xb_u8):
        return (xb_u8.float() / 255.0 - MEAN) / STD

    def local_steps(self, w, worker, k):
        m = self.model
        torch.nn.utils.vector_to_parameters(w.clone(), m.parameters())  # params become views: copy!
        m.train()
        ex = 0
        for _ in range(k):
            idx = torch.from_numpy(worker["stream"].next())
            xb = augment(self._prep(self.Xtr[idx]), worker["gen"])
            yb = self.ytr[idx]
            torch.manual_seed(int(torch.randint(2 ** 31, (1,), generator=worker["gen"])))  # dropout
            loss = F.cross_entropy(m(xb), yb)
            m.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(p.grad + self.wd * p, alpha=-self.lr)
            ex += len(idx)
        return torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone(), ex

    @torch.no_grad()
    def _eval(self, X, y):
        m = self.model
        m.eval()
        loss, correct = 0.0, 0
        for s in range(0, len(y), 1000):
            xb = self._prep(X[s:s + 1000])
            out = m(xb)
            loss += float(F.cross_entropy(out, y[s:s + 1000], reduction="sum"))
            correct += int((out.argmax(1) == y[s:s + 1000]).sum())
        return loss / len(y), correct / len(y)

    def evaluate(self, w):
        torch.nn.utils.vector_to_parameters(w, self.model.parameters())
        trl, tra = self._eval(self.Xtr[self.ev_tr], self.ytr[self.ev_tr])
        vl, va = self._eval(self.Xv, self.yv)
        return {"train_loss": trl, "train_acc": tra, "val_loss": vl, "val_acc": va}
