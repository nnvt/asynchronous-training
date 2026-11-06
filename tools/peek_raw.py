"""Print per-run checkpoints of cached runs in results/raw (quick look while a pool runs)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlsim.experiment import load_results  # noqa: E402

pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/raw", pat + ".npz"))):
    r = load_results(f)[0]
    t, g, T = r["t"], r["g2"], r["T"]
    tail = np.mean(np.log10(g[t >= 0.9 * T])) if np.all(np.isfinite(g)) else np.inf
    idx = [np.argmin(abs(t - fr * T)) for fr in (0.01, 0.1, 0.25, 0.5, 1.0)]
    print(f"{os.path.basename(f)[:34]:34s} tail {tail:6.2f} | " + " ".join(f"{g[i]:.1e}" for i in idx)
          + f" | max {np.max(g):.1e} iters {r['iters']} wall {r['wall']:.0f}s")
