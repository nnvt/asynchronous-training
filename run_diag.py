"""Diagnostic single runs not in the paper's protocol (results/diag/):
Malenia SGD at gamma = 0.05 (off the paper grid) to quantify how far
the paper's Malenia curve is from ours (see README)."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402

from rasgd.mnist import LabelPartitionedMLP  # noqa: E402
from rasgd.sim import METHODS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
TAUS = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]
P = LabelPartitionedMLP()
grid = np.linspace(0, 30000.0, 201)
for setting, method, gamma in [("fixed", "malenia", 0.05), ("exp", "malenia", 0.05)]:
    path = os.path.join(ROOT, "results", "diag", f"{setting}_{method}_g{gamma:g}_s0.npz")
    if os.path.exists(path):
        continue
    r = METHODS[method](P, TAUS, setting, gamma, 30000.0, grid, 0, log_steps=False)
    np.savez_compressed(path, grid=grid, loss=r["loss"].astype(np.float32), n_updates=r["n_updates"])
    sel = [np.searchsorted(grid, t) for t in (2000, 10000, 20000, 30000)]
    print(setting, method, gamma, np.round(r["loss"][sel], 4), flush=True)
