"""Short runs of every method to check behaviour and speed."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rlsim.experiment import run_one
T = float(sys.argv[1]) if len(sys.argv) > 1 else 20000
ds = sys.argv[2] if len(sys.argv) > 2 else "mnist"
gam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
for m in ["malenia", "ringleader", "ia2sgd"]:
    r = run_one(dict(dataset=ds, method=m, gamma=gam, seed=0, T=T, num_points=11))
    print(f"{m:11s} wall {r['wall']:6.1f}s (eval {r['eval_wall']:5.1f}s) iters {r['iters']:8d} "
          f"rounds {r.get('rounds', '-')} maxdelay {r.get('max_delay', '-')} div {r['diverged']}")
    print("   g2:", np.array2string(r["g2"], precision=3, max_line_width=200))
