# Replication: Rescaled Asynchronous SGD (arXiv:2605.13434)

Replication of the experiments of *"Rescaled Asynchronous SGD: Optimal Distributed
Optimization under Data and System Heterogeneity"* (A. Mahran, A. Maranjyan,
P. Richtárik, arXiv:2605.13434v1, May 2026), branch `rescaled-asgd`.

No official code is linked from the paper (no footnote / repository; a web search found
none), so everything here is re-implemented from the paper text, the Ringleader ASGD
paper (arXiv:2509.22860, Alg. 1 and App. F) for the two baselines, and the rendered
figures. All experiments are wall-clock *simulations* (event-driven, instantaneous
communication), exactly like the paper's "simulated wall-clock time".

**Bottom line.** The qualitative claims all hold in our runs, and our Rescaled ASGD
results match the paper closely. Several things in the paper's figures do not follow
from the setup as written: the Malenia and Ringleader curves in Fig. 1, the Ringleader
schedule in Fig. 2(a), the stepsizes in Fig. 3 and the Delay-Adaptive curve in Fig. 4.
Each one is pinned down with a clearly labelled diagnostic run below.

## Summary: paper claim vs our result

| Experiment | Paper claim / value | Our result | Status |
|---|---|---|---|
| E1 Fig. 1(a) MNIST, fixed τ | Rescaled ASGD beats Malenia and Ringleader; final loss ≈0.029 / 0.120 / 0.096 | Rescaled 0.029 (median of 5 seeds; curve matches the paper at every read-off point). Tuned baselines: Malenia 0.068, Ringleader 0.099 | **Replicated** (Rescaled wins by 2.3–3.4×, versus about 3–4× in the paper) |
| E1 tuning | selected γ lie in the interior of {1e-4,…,1} | γ = 0.1 for all three methods (interior) | Replicated |
| E2 Fig. 1(b) MNIST, Exp times | Malenia and Ringleader slowed; Rescaled "virtually identical" (≈0.029 / 0.152 / 0.140) | Rescaled 0.030 (same as fixed); Malenia 0.101 (vs 0.068 fixed); Ringleader 0.122 (vs 0.099 fixed) | **Replicated** |
| Baseline curves themselves | – | the paper's Malenia curve equals our Malenia at **γ = 0.05** (off-grid); its Ringleader curve equals our verbatim Ringleader at **γ = 0.01**, in both settings to within about 3% | Discrepancy in the baselines' stepsizes (see E1) |
| E3 Fig. 2 cumulative step | fixed: all advance α per τ_max; fluctuating: Malenia and Ringleader stall | Rescaled and Malenia match exactly. Ringleader (Alg. 1 verbatim) makes only 0.665 α/τ_max with fixed harmonic times (9.0 vs 13.7 at 14.3 τ_max). The fluctuating panel matches | **Partially** (Ringleader schedule in 2(a) not reproducible with Alg. 1 as published) |
| E4 Fig. 3 Vanilla vs Rescaled | Rescaled → x* = −2/3, Vanilla → x̃* = 1/2; Vanilla faster "for the same cycle stepsize" | Targets replicated. The paper's curves are reproduced to 2 decimals only with *different* cycle stepsizes (0.0133 vs 0.03); with equal α, Rescaled is faster | Targets replicated; speed statement **not replicated as written** |
| E5 Fig. 4 Delay-Adaptive | DA ends near x1* = 1; Rescaled near x* = 0 | DA → 0.9995, Rescaled → −0.047 (paper −0.04; cycle iterates match to ±0.01) | **Replicated**; the plotted DA curve looks like Vanilla ASGD (no delay shrinkage) |


## What the paper does (experiment list)

| # | Where | Setup (as stated) | Claim supported |
|---|---|---|---|
| E1 | Fig. 1(a), Sec. 5, App. E | MNIST, n=10 workers, worker i holds only one class (trimmed to 5,421 images each), MLP 784-128-10 (ReLU), batch 64, cross-entropy; τ ∈ {1,2,4,8,16}, two workers each, fixed times; budget 30,000; γ tuned over {1e-4,…,1} to minimise loss; 5 seeds, median + min/max band | Rescaled ASGD beats Malenia SGD and Ringleader ASGD; "selected stepsizes all fall in the interior of the grid" |
| E2 | Fig. 1(b) | same, but each gradient takes Exp(mean τ_i) time | Malenia and Ringleader are slowed down (longer gathering phases); Rescaled ASGD "virtually identical" in both settings |
| E3 | Fig. 2(a,b) | cumulative stepsize vs time, with Rescaled step ατ_i/(nτ_max), Ringleader α/n, Malenia α | fixed: all three advance α per τ_max; fluctuating: Malenia/Ringleader stall, Rescaled keeps α/τ_max |
| E4 | Fig. 3, App. F.1 | τ=(1,2), F1=(x−4)², F2=2(x+3)², σ=0, "same cycle stepsize α=0.01" | Rescaled → x* = −2/3 (equal-weighted F), Vanilla → x̃* = +1/2 (frequency-weighted F̃); Vanilla faster for the same α; small staleness bias for Vanilla |
| E5 | Fig. 4, App. F.2 | τ=(1,100), F1=(x−1)², F2=(x+1)², σ=0 (stepsizes not given) | Delay-Adaptive ASGD (γ_k = γ/(1+δ_k)) ends near x1* = 1; Rescaled converges to a small neighbourhood of x* = 0 |

The theory (Thms 4.1/4.3, Cor. 4.2/4.4) is not re-derived; the experiments are its empirical support.

## Implemented methods (`rasgd/sim.py`)

* **Rescaled ASGD**: Algorithm 1 (plain ASGD) with γ_i = γ·τ_i·τ_H/(n·τ_max) (eq. (2)), τ_i = the (mean) compute time.
* **Vanilla ASGD**: Algorithm 1 with a constant stepsize γ.
* **Delay-Adaptive ASGD**: Algorithm 1 with γ_k = γ/(1+δ_k) (App. D.2), δ_k = number of server updates between the worker reading the model and its gradient being applied. The rule from Mishchenko et al.'s released code, γ·min(1, n/δ_k), is also available.
* **Malenia SGD** (Tyurin & Richtárik 2023) with the parameter-free stopping rule "at least one gradient from every worker" (Ringleader paper, App. F; also how Sec. 1 of this paper describes it). Synchronous update x ← x − γ·(1/n)Σ_i ḡ_i, where ḡ_i averages the b_i gradients of worker i. Computations in progress at the broadcast are discarded.
* **Ringleader ASGD**: Algorithm 1 of arXiv:2509.22860, verbatim (gradient table, Phase 1 gathering, Phase 2 of n updates x ← x − γ·(1/n)Σ_i G_i/b_i, buffer for the next round).
* **Ringleader, table-reuse variant (diagnostic, not from any paper)**: at the end of Phase 2, entries of workers that sent nothing new are kept (stale) rather than emptied, so the next Phase 2 starts at once. It is only used to explain the Fig. 2(a) discrepancy (see E3).

Event simulator: worker i finishes a gradient every τ_i time units (fixed) or after an
independent Exp(mean τ_i) draw (fluctuating). Simultaneous deliveries are processed in
ascending worker index, as App. D.2 does. Workers are sorted by τ, so the fast workers go
first. The server-model loss is evaluated exactly at 201 equally spaced times in [0, T].

## E1/E2 — Fig. 1: label-partitioned MNIST

![fig1](figures/fig1_mnist.png)

Median over seeds and [min, max] (`results/mnist_summary.json`; hollow markers in the plot are values read off the paper's figure, about ±5%):

| setting | method | γ (tuned, seed 0) | t=2,000 | t=10,000 | t=20,000 | t=30,000 [min, max] | #updates | paper t=2k/10k/20k/30k |
|---|---|---|---|---|---|---|---|---|
| fixed | Rescaled ASGD | 0.1 | 0.258 | 0.093 | 0.048 | **0.0292** [0.0280, 0.0300] | 116,250 | 0.276/0.100/0.051/0.029 |
| fixed | Malenia SGD | 0.1 | 0.307 | 0.154 | 0.095 | **0.0676** [0.0664, 0.0698] | 1,875 | 0.388/0.228/0.156/0.120 |
| fixed | Ringleader ASGD | 0.1 | 0.642 | 0.155 | 0.116 | **0.099** [0.074, 0.221] | 12,492 | 0.354/0.196/0.130/0.096 |
| exp | Rescaled ASGD | 0.1 | 0.301 | 0.098 | 0.048 | **0.0302** [0.0271, 0.0329] | ≈116,000 | 0.275/0.092/0.051/0.029 |
| exp | Malenia SGD | 0.1 | 0.359 | 0.207 | 0.136 | **0.101** [0.1005, 0.1089] | ≈1,140 | 0.447/0.263/0.195/0.152 |
| exp | Ringleader ASGD | 0.1 | 0.970 | 0.245 | 0.130 | **0.122** [0.089, 0.153] | 7,240 | 0.43/0.254/0.185/0.140 |

Tuning (final loss at T = 30,000, seed 0; see `figures/fig1_tuning.png`):

| setting / method | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1 |
|---|---|---|---|---|---|
| fixed Rescaled | 1.64 | 0.414 | 0.185 | **0.029** | 1.78 |
| fixed Malenia | 2.07 | 0.667 | 0.270 | **0.068** | 1.18 |
| fixed Ringleader | 0.890 | 0.306 | 0.095 | **0.074** | diverged |
| exp Rescaled | 1.64 | 0.414 | 0.186 | **0.027** | 1.80 |
| exp Malenia | 2.19 | 0.973 | 0.317 | **0.105** | 1.71 |
| exp Ringleader | 1.32 | 0.360 | 0.143 | **0.122** | diverged |

* **Rescaled ASGD replicates quantitatively.** Its curve is within ±10% of the paper's
  read-off values at every point, in both settings. Fixed and fluctuating times give the
  same result (0.029 vs 0.030), as claimed.
* **Ordering replicated.** Rescaled < Malenia ≈ Ringleader throughout, in both settings.
  Malenia is slowed by fluctuating times (0.068 → 0.101; 1,875 → about 1,140 updates).
  Ringleader also slows (12,492 → 7,240 updates).
* **Discrepancy: the baselines are better in our runs than in the paper.**
  `figures/fig1_diagnostics.png` shows single-seed diagnostic runs:
  * The paper's Malenia curve coincides with our Malenia at **γ = 0.05**.
    Fixed: 0.383/0.223/0.157/0.119 vs paper 0.388/0.228/0.156/0.120.
    Exp: 0.464/0.271/0.207/0.168 vs paper 0.447/0.263/0.195/0.152.
    γ = 0.05 is not on the paper's grid, so the paper's Malenia effectively used half
    of the grid value 0.1 (an extra factor 1/2 somewhere?). We cannot tell why.
  * The paper's Ringleader curve coincides with our *verbatim* Ringleader at **γ = 0.01**.
    Fixed: 0.357/0.195/0.129/0.095 vs paper 0.354/0.196/0.130/0.096.
    Exp: 0.432/0.246/0.182/0.143 vs paper 0.43/0.254/0.185/0.140.
    Our seed-0 tuning picks γ = 0.1 instead (0.074 vs 0.095). At γ = 0.1 Ringleader is at
    the edge of stability: there is a loss spike early on, the median at t=2,000 is 0.64,
    and the 5-seed final losses span 0.074–0.221 (median 0.099). So the choice between
    0.1 and 0.01 depends on the tuning seed.
  * **[diagnostic]** Ringleader, table-reuse variant (the one that matches the Fig. 2(a)
    schedule), fixed times, seed 0: final loss **0.067 at γ = 0.01** (18,741 updates) and
    0.165 at γ = 0.1. At γ = 0.01 it is clearly *better* than both the paper's Ringleader
    (0.096) and our verbatim Ringleader at the same γ (0.095). So the paper's Fig. 1
    Ringleader curve is the verbatim algorithm, not this variant. Fig. 1 and Fig. 2(a)
    therefore seem to come from different Ringleader implementations.
  * Net effect: with the paper's (apparent) baseline stepsizes, the paper's 3–4× final-loss
    gap is reproduced exactly. With baselines tuned by the stated protocol, the gap
    shrinks to 2.3× (Malenia, fixed) to 3.4× (Ringleader, fixed), and Rescaled ASGD still wins.

> **Note on the headline comparison.** All headline numbers (table above,
> `figures/fig1_mnist.png`) use the stepsize **γ = 0.1** that the paper's protocol selects
> for every method. The paper's baseline curves match Malenia at γ = 0.05 and Ringleader
> at γ = 0.01. Those values come from single-seed diagnostic runs
> (`figures/fig1_diagnostics.png`) and are given here only as a note. Which stepsizes
> the headline should use is still an open decision.


## E3 — Fig. 2: cumulative stepsize (schedule only)

![fig2](figures/fig2_cumulative_step.png)

Cumulative step (× α) at t = 14.3 τ_max (`results/schedule_summary.json`):

| | Rescaled | Malenia | Ringleader (Alg. 1 verbatim) | Ringleader table-reuse [diag.] |
|---|---|---|---|---|
| paper Fig. 2(a), fixed | 14.1 | 14.0 | **13.7** | – |
| ours, fixed | 14.15 | 14.0 | **9.0** (long-run rate 0.665 α/τ_max) | **13.7** (rate 0.998) |
| paper Fig. 2(b), fluctuating (1 seed) | 15.2 | 9 | 5.2 | – |
| ours, fluctuating: median [10%, 90%] of 100 seeds | 14.1 [13.1, 15.0] | 8 [6, 10.1] | 5.0 [4.0, 6.9] | 7.9 [5.9, 10.8] |
| ours, fluctuating long-run rate (α/τ_max) | 1.00 | 0.60 (mean round 1.68 τ_max) | 0.39 (mean round 2.59 τ_max) | 0.60 |

* Rescaled ASGD and Malenia SGD match the paper in both panels. In the fluctuating setting
  Malenia and Ringleader stall, as claimed: replicated.
* **Discrepancy (Ringleader, fixed times).** With exactly harmonic times, Algorithm 1 of
  Ringleader cannot make 10 updates per τ_max. The two τ=16 workers deliver at the same
  instant. Only one of them can complete Phase 1 and be updated immediately; the other
  one's gradient went into the table, so it is updated only at its next delivery, 16
  units later, and it then holds up the next Phase 1. The result is 20 updates per
  48 time units, whatever tie-breaking order is used (we checked ascending, descending,
  FIFO, LIFO and random; see `rasgd/sim.py` and `run_toy.py`). The paper's curve rises by
  1.0 α per τ_max, with steps at +0, +1, +2, +4 and +8 time units after every multiple of
  τ_max. Our table-reuse variant gives exactly that pattern and exactly 13.7 at
  14.3 τ_max. So the paper's simulation evidently does not wait for an empty table in that
  case. Whether this is an implementation detail or a bug cannot be decided without the
  code. In the *fluctuating* panel the paper's single-seed Ringleader value (5.2) is
  typical of the verbatim algorithm (median 5.0) and at the low end of the reuse variant
  (10% quantile 5.9).

## E4 — Fig. 3 / App. F.1: Vanilla ASGD targets the frequency-weighted average

![fig3](figures/fig3_vanilla_vs_rescaled.png)

Cycle iterates x^m (server model after all events at times ≤ mτ_max):

| t | 30 | 60 | 90 | 300 (end) |
|---|---|---|---|---|
| paper Rescaled (read off) | 2.39 | 0.98 | 0.22 | −0.66 |
| ours Rescaled, **as written** (cycle stepsize α = 0.01: γ_i = ατ_i/(nτ_max) = 0.0025, 0.005) | 2.91 | 1.59 | 0.76 | −0.620 |
| ours Rescaled, **inferred** (eq. (2) with γ = 0.01: γ_i = 0.0033, 0.0067; cycle stepsize γτ_H = 0.0133) | **2.39** | **0.98** | **0.22** | **−0.670** |
| paper Vanilla (read off) | 1.74 | 0.81 | 0.56 | ≈0.45 |
| ours Vanilla, **as written** (γ_i = α/K = 0.0033, cycle stepsize 0.01) | 3.49 | 2.49 | 1.82 | 0.560 |
| ours Vanilla, **inferred** (constant γ_i = 0.01, cycle stepsize 0.03) | **1.74** | **0.82** | **0.55** | **0.448** |

* **Replicated (targets).** In every variant Rescaled ASGD goes to x* = −2/3 and Vanilla
  ASGD to x̃* = +1/2. Vanilla's cycle iterates settle slightly *below* 1/2 (0.448 vs 0.5;
  this is the staleness bias the paper points out, and it shrinks with the stepsize).
  The within-cycle oscillation disappears in the cycle iterates.
* **Discrepancy (stepsizes / speed claim).** The text says both methods use "the same cycle
  stepsize α = 0.01". With that setting Rescaled is *faster* than Vanilla (per-cycle
  contraction 1−3α vs 1−8α/3 for this pair of quadratics), so the sentence "Vanilla ASGD
  converges faster for the same cycle stepsize α" does not hold here. The paper's curves
  are reproduced to two decimals by Rescaled with eq. (2), γ = 0.01 (cycle stepsize
  0.0133) and Vanilla with a constant per-step stepsize 0.01 (cycle stepsize 0.03). So
  Vanilla had a 2.25× larger cycle stepsize in the figure.

## E5 — Fig. 4 / App. F.2: Delay-Adaptive ASGD is biased toward the fast worker

![fig4](figures/fig4_delay_adaptive.png)

The stepsizes are not stated in the paper. We inferred them from the figure.
Rescaled: eq. (2) with γ = 0.1, i.e. γ_i = 0.00099, 0.099; this reproduces the per-cycle
contraction of about 0.61 and the limit of about −0.04. Delay-Adaptive: γ = 0.01; the
decay from 3 to 1.27 over the first 99 fast steps fixes this value.

| quantity | paper (read off) | Rescaled | DA γ/(1+δ) (as written) | DA γ·min(1,n/δ) | Vanilla γ=0.01 [diag.] |
|---|---|---|---|---|---|
| Rescaled cycle iterates t=100,200,300,400,500,1500 | 1.85, 1.13, 0.68, 0.41, 0.24, −0.04 | 1.85, 1.13, 0.68, 0.41, 0.23, −0.047 | | | |
| DA: x(t=99) | 1.27 | | 1.271 | 1.271 | 1.271 |
| DA: drop at the first slow update (t=100) | ≈0.085 | | 0.0008 | 0.0016 | 0.080 |
| DA: dip at each later slow update | ≈0.045 | | 0.0004 | 0.0008 | 0.039 |
| DA: late range | ≈[0.95, 1.0] | | [0.9995, 0.9999] | [0.9991, 0.9999] | [0.955, 0.994] |

* **Replicated (claim).** Delay-Adaptive ASGD ends at x ≈ 1 = x1*. Its implicit
  weights (App. D.2) are ŵ = (0.9999, 0.0001), giving the minimiser 0.9998. Rescaled ASGD
  ends in a small neighbourhood of x* = 0 (−0.047, versus −0.04 in the paper).
* **Discrepancy.** With the rule as written, the slow worker's step is shrunk by
  1/(1+δ) = 1/101. Its updates are then invisible (dips of 4e-4). The paper's curve
  instead shows dips of about 0.045–0.085 at every slow delivery. That corresponds to a
  slow-worker step of about 0.01 = γ, i.e. *no* delay shrinkage, and plain Vanilla ASGD
  with γ = 0.01 reproduces it. Vanilla also ends near 1, since its target is 0.980. So
  the conclusion is unaffected, but the plotted curve looks like Vanilla, not
  γ/(1+δ_k).

## Assumptions for details the paper leaves open

| Detail | Our choice | Why / note |
|---|---|---|
| Which class sits on which worker | worker i holds digit i; τ = (1,1,2,2,4,4,8,8,16,16), so digits 0,1 are fastest and 8,9 slowest | not stated; this is the natural ordering |
| Trimming to 5,421 per class | random subset per class (fixed seed 0) | 5,421 = size of the smallest class (digit 5), as the paper says |
| "Standard normalization" | (x/255 − 0.1307)/0.3081 | the usual torchvision MNIST constants |
| Initialisation | PyTorch `nn.Linear` default, U(±1/√fan_in), different for each seed | initial loss 2.36, matching the paper's ≈2.3 starting point |
| Minibatches | 64 indices drawn without replacement from the worker's shard, independently for every gradient | |
| Reported loss | mean cross-entropy over all 54,210 kept images = F (equal-weighted global objective) | the paper says "training loss" |
| Stepsize parameter γ | Rescaled: eq. (2); Malenia: x ← x − γ·mean_i ḡ_i; Ringleader: x ← x − γ·mean_i G_i/b_i | these reproduce the Rescaled curve and the Fig. 2 normalisation |
| Tuning criterion | final training loss at T = 30,000, seed 0, grid {1e-4, 1e-3, 1e-2, 1e-1, 1}; then seeds 0–4 at the selected γ | the paper says "tune … within a fixed wall-clock budget … to minimize loss" |
| Seeds | a seed sets the initialisation, the minibatch stream and (fluctuating case) the compute-time draws | |
| Simultaneous events (fixed times) | processed in ascending worker index | App. D.2 |
| Malenia stopping rule | ≥1 gradient from every worker; in-progress work is discarded at the broadcast and every worker restarts with a fresh time draw | Sec. 1 of the paper and Ringleader App. F |
| Fluctuating times | each gradient takes an independent Exp(mean τ_i) time | App. E |
| Loss trajectory | evaluated exactly at 201 equally spaced times (no interpolation needed) | the paper interpolates on 200 points |
| Arithmetic | float32 numpy MLP; CPU only | |
| Fig. 3 / Fig. 4 stepsizes, start points | x0 = 5 (Fig. 3) and 3 (Fig. 4), read off the plots; stepsizes as discussed in E4/E5 | not, or not consistently, given in the text |

No experiment was scaled down: the MNIST experiment runs at the paper's full size
(n = 10, 54,210 images, 128 hidden units, batch 64, T = 30,000, i.e. ≈116k stochastic
gradients per run, five seeds).


## How to run

Python: the shared venv `/home/funny36/Documents/mrtvc/asynchronous-training/.venv`
(numpy 2.5, matplotlib); MNIST `train-*-ubyte.gz` read from
`/home/funny36/Documents/mrtvc/shared-data` (path in `rasgd/mnist.py`).

```bash
PY=/home/funny36/Documents/mrtvc/asynchronous-training/.venv/bin/python
$PY run_toy.py                     # Figs. 2-4 (schedules + toy problems), ~1 min
$PY run_mnist.py --procs 4 --methods rescaled,malenia,ringleader   # Fig. 1, tuning + 5 seeds
$PY -c 'import run_mnist as R; [R.run_one(("fixed", "ringleader_reuse", g, 0, 30000.0)) for g in (0.1, 0.01)]'  # diagnostic, seed 0
$PY run_diag.py                    # Malenia at gamma=0.05 (diagnostic)
$PY plot.py                        # figures/*.png
```

`run_mnist.py` caches every run in `results/mnist/<setting>_<method>_g<gamma>_s<seed>.npz`
and writes `results/mnist_summary.json` (tuning tables, per-seed final losses, medians).
A run at T = 30,000 takes 2–5 min on one core of this (shared, loaded) laptop. The whole
replication took about 2 h 20 min (the machine was shared with other jobs, load ≈ 13–23; a run took 2–10 min) of wall time with 4–5 processes.

### Files

* `rasgd/sim.py` – event-driven simulator and all methods (Rescaled, Vanilla, Delay-Adaptive ASGD; Malenia SGD; Ringleader ASGD and its diagnostic variant).
* `rasgd/mnist.py` – label-partitioned MNIST + numpy MLP. The forward/backward code is adapted from `asyncsim/problems.py` on the sibling branch `ringmaster-asgd`; the rest of the simulator was written anew, because the methods and the data heterogeneity differ.
* `rasgd/toy.py` – 1-D quadratics (App. F) and a schedule-only dummy problem.
* `run_mnist.py`, `run_toy.py`, `run_diag.py`, `plot.py` – drivers; `results/` – raw results (small); `figures/` – plots; `logs/` – run logs.

