# Replication: Ringmaster ASGD (arXiv:2501.16168)

Branch `ringmaster-asgd`. Replicates the experiments (App. G) of
*Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity*
(Maranjyan, Tyurin, Richtárik, 2025). No official code was released, so
everything here is implemented from the paper.

## Bottom line

| Paper claim / figure | Our result | Status |
|---|---|---|
| **Fig. 2** (quadratic, n = 6174, d = 1729): Ringmaster ASGD converges faster than Rennala SGD, which is far faster than Delay-Adaptive ASGD | Same ordering at every time point, in every setup variant and with both tuning criteria. With the paper's hyper-parameters Ringmaster reaches a given accuracy **1.6–1.8× sooner** than Rennala; Delay-Adaptive ASGD never gets below 0.1 within 2000 s | **Replicated (qualitative)** |
| Fig. 2 tuned hyper-parameters (Ringmaster γ = 0.2, R = 6; Rennala γ = 1, B = 6) | Tuning on the paper's grid (whole-run criterion) picks **γ = 0.2, R = 7** and **γ = 1, B = 6**. R = 7 = ⌈n/4⁵⌉; the paper's R = 6 is not in its own stated grid | **Replicated** |
| Fig. 2 absolute levels (Ringmaster ≈ 2.5e-5 at 2000 s) | The setup as written plateaus at ≈ 1.6e-2. We show this is unavoidable for that problem (see below). A variant (b = 0, noiseless gradients, compute times based on √i) matches the paper's Ringmaster and Rennala curves to within ~0.1 decade | **Not replicated as written**; figure reproduced by the variant |
| **Fig. 3** (2-layer ReLU MLP on MNIST): Ringmaster beats DA-ASGD and Rennala and is more robust | Re-tuned on the paper's grid, Ringmaster reaches the lowest loss (0.26 vs 0.43 and 0.46 at 1000 s). With the paper's legend γ, neither Ringmaster nor Rennala trains in our MLP. The paper's 1e-14 losses are not reproduced; the setup is underspecified | **Partially replicated** |

## What is simulated

Everything is an event-driven simulation of n workers. Simulated time
advances from one gradient arrival to the next. Worker i needs
τᵢ = i + |ηᵢ|, ηᵢ ~ N(0, i) seconds per stochastic gradient, as in the
paper. By default we draw a fresh duration for every job (`per_job`); the
`fixed` variant draws it once per worker. Methods, in `asyncsim/methods.py`:

| name | algorithm |
|---|---|
| `ringmaster` | Alg. 5: Ringmaster ASGD with calculation stops (jobs whose delay reaches R are restarted at the current iterate) |
| `ringmaster_nostop` | Alg. 4: Ringmaster ASGD that ignores arrivals with delay ≥ R |
| `rennala` | Alg. 2: Rennala SGD. Taken literally: the worker that delivers the B-th gradient is re-assigned to the *old* xᵏ, as in both papers' pseudocode |
| `da_asgd` | Alg. 1 with Mishchenko et al. (2022)'s delay-adaptive stepsize, as in [their code](https://github.com/konstmish/asynchronous_sgd): γₖ = γ·min(1, n/δₖ) |
| `asgd` | Alg. 1, constant stepsize |

A 2000-s run with n = 6174 takes 0.1–0.5 s, so we tune the full grids:
γ ∈ {5ᵖ : p ∈ [−5, 5]} and R, B ∈ {⌈n/4ᵖ⌉} ∪ {⌊n/4ᵖ⌋}. The floor variant
is there because the paper's reported R = B = 6 equals ⌊6174/4⁵⌋, while
⌈6174/4⁵⌉ = 7.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python run.py --setup text      --mode paper               # Fig. 2, legend hyper-params
.venv/bin/python run.py --setup text      --mode tune --criterion auc # Fig. 2, re-tuned (whole-run criterion)
.venv/bin/python run.py --setup text      --mode tune --criterion final
.venv/bin/python run.py --setup fig2match --mode paper               # variant that matches Fig. 2
.venv/bin/python run.py --setup mnist     --mode paper  --methods ringmaster rennala da_asgd
.venv/bin/python run.py --setup mnist     --mode tune --criterion auc --methods ringmaster rennala da_asgd --tune-seeds 1 --tune-points 100 --points 400
```

Each run writes `results/<setup>_<mode>.npz` (all curves), `results/*_grid.json`
(every grid point's score when tuning) and `figures/<setup>_<mode>.png`. Final
curves show the median of 5 seeds, with a min–max band. Hollow markers are
values we read off the paper's figure by eye.

## Fig. 2 — quadratic

Setup as written (`--setup text`): f(x) = ½xᵀAx − bᵀx with
A = ¼·tridiag(−1, 2, −1) and b = ¼(−1, 0, …, 0). The stochastic gradient is
∇f(x) + ξ with ξ ~ N(0, 0.01² I_d). The paper doesn't state x₀; we use
(√d, 0, …, 0) from the Rennala SGD paper's version of this experiment
(Tyurin & Richtárik 2023, App. J). It gives f(x₀) − f* = 443, which matches
the starting level in Fig. 2. Gaps are computed against the exact f*
(closed-form x*).

Median f(xᵗ) − f* over 5 seeds:

| setup / hyper-params | method | t = 200 | 500 | 1000 | 2000 s |
|---|---|---|---|---|---|
| **paper's Fig. 2 (read off)** | Ringmaster γ=0.2, R=6 | 1e-2 | ~6e-4 | 1.1e-4 | 2.5e-5 |
| | Rennala γ=1, B=6 | 4e-2 | ~3e-3 | 4e-4 | 5e-5 |
| | DA-ASGD γ=0.00032 | ~1e2 | ~10 | 2 | 2e-2 |
| `text`, paper's hyper-params | Ringmaster (Alg. 5) | 8.0e-2 | 3.0e-2 | 2.0e-2 | 1.65e-2 |
| | Ringmaster (Alg. 4) | 8.8e-2 | 3.1e-2 | 2.1e-2 | 1.67e-2 |
| | Rennala | 2.2e-1 | 5.1e-2 | 2.6e-2 | 1.83e-2 |
| | DA-ASGD | 3.2e2 | 1.6e2 | 4.7e1 | 1.1e1 |
| `text`, tuned (auc): γ=0.2,R=7 / γ=1,B=6 / γ=0.04 | Ringmaster (Alg. 5) | 7.4e-2 | 2.9e-2 | 2.0e-2 | 1.78e-2 |
| | Rennala | 2.2e-1 | 5.1e-2 | 2.6e-2 | 1.83e-2 |
| | DA-ASGD | 1.7e1 | 1.9e1 | 1.0e1 | 2.4 |
| `text`, tuned (final): γ=0.04,R=1543 / γ=1,B=7 / γ=0.008 | Ringmaster (Alg. 5) | 1.7e1 | 1.3e-1 | 1.6e-2 | 1.10e-2 |
| | Rennala | 2.6e-1 | 5.6e-2 | 2.7e-2 | 1.78e-2 |
| | DA-ASGD | 2.8e1 | 1.8e1 | 1.3e1 | 2.1 |
| `fig2match`, paper's hyper-params | Ringmaster (Alg. 5) | 8.8e-3 | 9.1e-4 | 1.6e-4 | 2.85e-5 |
| | Rennala | 3.4e-2 | 3.6e-3 | 6.4e-4 | 1.13e-4 |
| | DA-ASGD | 3.8e1 | 8.9e-1 | 5.3e-2 | 9.9e-3 |

Time to reach a target (median curve), `text` setup with the paper's hyper-parameters:

| target | 0.1 | 0.05 | 0.03 | 0.02 |
|---|---|---|---|---|
| Ringmaster (Alg. 5) | 176 s | 286 s | 474 s | 991 s |
| Rennala | 310 s | 505 s | 847 s | 1558 s |
| DA-ASGD | never | never | never | never |

Figures: `figures/text_paper.png`, `figures/text_tune-auc.png`,
`figures/text_tune-final.png`, `figures/fig2match_paper.png`,
`figures/text-fixedtime_paper.png`.

Notes:

- **Tuning criterion matters, and the paper doesn't state one.** `auc`
  (mean log₁₀ gap over the whole run) reproduces the paper's choices for
  Ringmaster and Rennala. `final` (mean log₁₀ gap over the last 5% of the
  run) prefers small steps that lower the noise floor. Under that criterion
  Ringmaster ends lowest (1.1e-2 vs 1.8e-2 for Rennala) but is slower than
  Rennala before ~700 s.
- **Alg. 4 vs Alg. 5** (with or without stops) are nearly identical here;
  stops help slightly.
- **Fixed vs per-job compute times** makes no visible difference (`text-fixedtime`).

### Why the paper's absolute levels can't be reached as written

1. The term b creates a component of x₀ − x* (the linear ramp
   x*ⱼ = −(d+1−j)/(d+1)) whose gap under any gradient method decays only like
   1/√S, where S = Σγₖ. Even **noiseless** GD sits at 5.4e-3 at S = 473 (where
   Ringmaster is at t = 2000 s) and at 3.4e-3 at S = 1000. Fig. 1 of the same
   paper, copied from Tyurin & Richtárik with d = 1000, does bottom out near
   7e-3, consistent with this. Fig. 2 goes to 2.5e-5.
2. With ξ ~ N(0, 0.01² I_d), σ² = 0.17. The stationary noise level of
   Ringmaster at γ = 0.2 is ≈ 1e-2, which is the plateau we see.
3. Delay-Adaptive ASGD at γ = 0.00032 performs ~12.7k updates in 2000 s, so
   S ≤ 4 and the gap stays ≥ 10. The paper's 2e-2 needs S ≳ 50.

### The `fig2match` variant

We scored eight combinations of readings against the paper's figure:
noise i.i.d. vs one scalar broadcast to all coordinates; b as written vs
b = 0; τ based on i vs √i; plus noiseless and fixed-time options
(`asyncsim/experiment.py: SETUPS`). The closest is **b = 0, noiseless
gradients, τᵢ = √i + |ηᵢ| with ηᵢ ~ N(0, √i), fixed per worker**. With the
paper's legend hyper-parameters it matches the Ringmaster curve to
~0.08 decade and the Rennala curve to ~0.13 decade at every point we read
off. It also produces DA-ASGD's oscillations, but not its timing (~1 decade
off).

The evidence is mixed. Tuning under `fig2match` picks γ = 0.00032 for
DA-ASGD, exactly the paper's value. But it picks R = 2 and B = 1 rather than
6, which only the noisy as-written setup reproduces. So we don't claim
`fig2match` is what the authors ran. It shows that Fig. 2's curves are
consistent with a noiseless, b = 0 version of the problem, and that the
qualitative conclusion is the same either way.

## Fig. 3 — MNIST, 2-layer ReLU MLP

The paper gives only: "small 2-layer [the text says 20-layer, likely a typo]
NN with ReLU on MNIST, n = 6174, same time distributions", plus the legend
hyper-parameters (Ringmaster γ = 1, R = 6; Rennala γ = 5, B = 6; DA-ASGD
γ = 0.0016). Our choices for everything unstated, all configurable in
`SETUPS["mnist"]`:

- a 784–32–10 ReLU MLP, PyTorch-default init, mean cross-entropy, float64
- a fixed random subset of 3000 training images, pixels scaled to [0, 1]
- stochastic gradient = a minibatch of 4 samples (as in the Rennala paper's MNIST experiment)
- f_inf = 0, so we plot the training loss on the subset

Median training loss over 5 seeds:

| hyper-params | method | t = 100 | 250 | 500 | 1000 s |
|---|---|---|---|---|---|
| **paper's Fig. 3 (read off)** | Ringmaster | ~1e-1 | ~1e-9 | 2e-14 (float noise) | 2e-14 |
| | Rennala | ~1e-2 (after a 1e6 spike) | ~1e-2 | ~1e-7 | ~1e-14 |
| | DA-ASGD | ~1e3 spike | ~1e-2 | 3e-4 | 1e-5 |
| paper's legend | Ringmaster γ=1, R=6 | 2.37 | 2.39 | 2.35 | 2.37 (never trains) |
| | Rennala γ=5, B=6 | 2.42 | 2.40 | 2.44 | 2.37 (never trains) |
| | DA-ASGD γ=0.0016 | 2.20 | 1.94 | 1.35 | 0.67 |
| re-tuned on the paper's grid (auc) | **Ringmaster γ=0.04, R=386** | 1.12 | **0.67** | **0.47** | **0.26** |
| | Rennala γ=0.2, B=6 | 1.73 | 1.03 | 0.60 | 0.46 |
| | DA-ASGD γ=0.04 | **0.98** | 0.69 | 0.53 | 0.43 |

Figures: `figures/mnist_paper.png`, `figures/mnist_tune-auc.png`.
Per-γ scores for every grid point: `results/mnist_tune-auc_grid.json`.

- **Claim "Ringmaster outperforms DA-ASGD and Rennala": partially
  replicated.** With each method tuned on the paper's grid, Ringmaster
  reaches the lowest loss from ~150 s onward (0.26 vs 0.43 and 0.46 at
  1000 s). DA-ASGD is equal or slightly better in the first ~100 s.
- **Claim "more robust": not clearly supported.** At every γ on the grid,
  Ringmaster's best score over R is at least as good as DA-ASGD's. But
  neither Ringmaster nor Rennala learns at all for γ ≥ 1, which includes the
  paper's own legend values.
- **The paper's curves are not reproduced.** Training losses of 1e-14 and
  straight-line (linear-rate) decay are not what SGD on an MLP does. The
  1e-14 floor looks like float64 rounding of a loss that has converged
  exactly. We tried raw 0–255 pixels: γ = 1 and γ = 5 then blow up to
  10⁶–10⁷, like the paper's spikes. Afterwards the ReLUs die and the loss
  freezes at the label entropy (2.2985), i.e. the network predicts a
  constant. With full-batch gradients the dead networks stall ~2e-4 above
  that, not at 1e-14. Without the authors' code we can't pin down Fig. 3's
  setup. Our best guess is that its curves show convergence of a degenerate
  (diverged-then-dead) network to a non-zero f_inf, not genuine training.

## Files

- `asyncsim/timing.py`: compute-time models
- `asyncsim/methods.py`: event-driven simulators of all methods
- `asyncsim/problems.py`: quadratic and numpy MLP/MNIST problems
- `asyncsim/experiment.py`: setups, the paper's hyper-parameters and figure values, grid tuning, multi-seed runs
- `asyncsim/plotting.py`, `run.py`: plotting and CLI
