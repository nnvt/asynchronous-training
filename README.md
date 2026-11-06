# Replication: LOSCAR-SGD

**Paper:** Y. Maziane, A. Mahran, A. Maranjyan, P. Richtárik, *LOSCAR-SGD: Local SGD with
Communication-Computation Overlap and Delay-Corrected Sparse Model Averaging*, arXiv:2605.20866v1
(20 May 2026). Branch: `loscar-sgd`. The paper links no official code, so everything here is
re-implemented from the text (Algorithm 1, Section 5, Appendix B). Paper numbers come from reading
the rendered figures: the a9a figures are embedded 900×500 PNGs, which we extracted with `pdfimages`.

## TL;DR

| # | Experiment (paper) | Status |
|---|---|---|
| E1 | a9a controlled comparison: Local Sparse vs Overlap overwrite vs Overlap delay-corrected (Figs. 1a, 2, 3, 4) | **replicated**. Loss and gradient-norm curves agree with the figures to within ≈0.003 |
| E2 | Merge rule (overwrite vs delay-corrected) in three regimes (Figs. 1c, 5–7) | **replicated** |
| E3 | Sparsity ablation p ∈ {0.001, 0.01, 0.1, 1}: loss vs bits (Figs. 1b, 8) | **replicated**, including the bit counts (288·K bits/round) |
| E4 | Local-budget ablation M ∈ {1, 4, 16, 64} (Fig. 9) | **replicated** |
| E5 | Communication-delay ablation ζ ∈ {12, 48} (Fig. 10) | **replicated** (M = 8 is read off the figure; the text does not give it) |
| E6 | Non-IID stress test, label-Dirichlet α = 0.03 (Fig. 11) | **partially replicated**. Overlap gains fast early and then ends below Local Sparse, as in the paper. The paper's "delay-corrected is worst" is **not** reproduced: here DC ≈ OW |
| E7 | CIFAR-10 CNN, normal and stress regimes (Figs. 12–13) | **replicated qualitatively**: small CNN, ≈1/12 of the paper's logical-time budget, 1 seed |
| E8 | Tiny ImageNet ResNet-18 (Figs. 14–15) | **not run** (GPU-scale) |

The paper also lists *Sync SGD* and *FedAvg-Full* and says it ran a nonconvex Geman–McClure
variant, but it shows no plots for any of them. We ran all three as extras (see "Extras").

## What is simulated

The paper's experiments use a *PyTorch logical-time simulator*: all workers live in one process,
and time is the integer logical clock of Assumption 2.1. We built the same thing (`loscar/core.py`):

* n = 4 workers with step times τ_i (default (1, 2, 3, 6)), τ = lcm(τ_i), and one round lasting
  Mτ + ζ. Worker i takes N_i = Mτ/τ_i local SGD steps to reach y_i. It sends Proj_S(y_i), where S
  is a common uniform random K-subset. The overlap methods then take Q_i = ζ/τ_i more steps to reach z_i.
* **Local Sparse** (blocking): N_i steps, then idle for ζ. Then x_i ← ȳ on S and y_i off S.
* **Overlap overwrite**: x_i ← ȳ on S and z_i off S.
* **Overlap delay-corrected (LOSCAR-SGD, Alg. 1)**: x_i ← ȳ + (z_i − y_i) on S and z_i off S.
* FedAvg-Full = Local Sparse with K = d. Sync SGD = one stochastic gradient per worker per step,
  gradients averaged, each step lasting max τ_i + ζ (our own definition; the paper gives none).
* Metrics are evaluated at the **average model x̄ = (1/n) Σ x_i**, the quantity in Theorem 4.1.
  The figures confirm this: in Fig. 8 every p gives an identical first point, which holds only
  for x̄. Under delay correction x̄^{r+1} = z̄^r for any mask, and under Local Sparse x̄^{r+1} = ȳ^r.
* Logical time after r rounds = r(Mτ + ζ). "Logical total bits" per round = (2n+1)·32·K
  (n uplink and n downlink messages of K float32 values, plus the K shared mask indices).
  This formula reproduces the paper's bit axis exactly: 288, 576, 3744 and 35712 bits after round 1
  for p = 0.001, 0.01, 0.1 and 1. The paper's Fig. 8 reads ≈287, ≈574, ≈3.7e3 and ≈3.6e4.

### a9a logistic-regression setup (App. B)

* a9a (LIBSVM train file, 32 561 × 123). 10 % of it is held out for validation (random, seeded).
  Features are standardised with training-split statistics. The model is affine: 123 weights + bias,
  so d = 124.
* Init: PyTorch `nn.Linear(123, 1)` default, U(±1/√123) for the weights and the bias. Every worker
  starts from the same x⁰. This gives an initial loss ≈ 0.72–0.73, matching the round-0 value of
  0.733 in Fig. 1a (zero init would give 0.693). Our 10-seed mean is 0.719; the expected value
  under this init is ≈ 0.735.
* Plain SGD, η = 0.1, batch 256. Each worker reads its own IID shard (a random equal split) with
  per-epoch reshuffling, like a DataLoader with `drop_last=False`. K = ⌈p·d⌉.
* 20 rounds (40 in E6). **10 seeds** (the paper does not give its seed count). A seed fixes the
  val split, the partition, the init, the batch streams and the mask sequence, and all methods
  share them. Plots show mean ± 1 std across seeds.

## How to run

```bash
PY=/path/to/venv/bin/python          # numpy, matplotlib; torch (CPU) for CIFAR only
# data: a9a in $LOSCAR_DATA/a9a/a9a, CIFAR-10 python batches in $LOSCAR_DATA/cifar10/cifar-10-batches-py
export LOSCAR_DATA=/home/funny36/Documents/mrtvc/shared-data
OPENBLAS_NUM_THREADS=2 $PY experiments/run_a9a.py --seeds 10 --procs 5   # ≈1.5 min, all a9a runs
$PY experiments/plot_a9a.py                                              # figures/fig1..fig11
$PY experiments/compare_a9a.py                                           # paper-vs-ours table
$PY experiments/run_cifar.py --budget 3000 --procs 5 --threads 2         # ≈15 min wall (6 runs)
$PY experiments/plot_cifar.py
```

Layout: `loscar/core.py` (simulator and all five methods), `loscar/logreg.py`, `loscar/cifar.py`
(problems), `loscar/data.py` (a9a parser, splits, Dirichlet partition), `loscar/plotting.py`,
`experiments/` (drivers), `results/*.npz` (raw curves), `figures/*.png`.

## Results

### Claims and status

| Exp. | Paper claim | Our result | Status |
|---|---|---|---|
| E1 (Figs. 1a, 2–4) | DC beats OW, and OW beats LS, on every metric and axis | Loss, val loss and gradient norm keep the order DC < OW < LS for **10/10 seeds at every round**. Final train loss: LS 0.3375, OW 0.3337, DC 0.3324 (paper 0.336 / 0.332 / 0.331). Final ‖∇F‖: 0.0375 / 0.0308 / 0.0282 (paper 0.038 / 0.031 / 0.028). Accuracy keeps the order only early (round 1: 0.755 / 0.765 / 0.772); by round 20 all three methods sit at 0.849 | replicated (the accuracy ordering is an early-round effect, as also visible in the paper's Fig. 2) |
| E1 (Fig. 4) | The ordering is unchanged against logical time and bits | Holds trivially, because round duration and bits per round are the same for all three methods | replicated |
| E2 (Figs. 5–7) | DC has lower train and val loss than OW in all three regimes. The gap is small for long compute, largest for long delay and moderate for heterogeneous speeds | Paired OW − DC train loss at rounds 1 / 20: long compute +0.0041 / +0.0002, long delay +0.0349 / +0.0024, heterogeneous speeds +0.0134 / +0.0007. Positive for 10/10 seeds at every round in all three regimes | replicated |
| E3 (Figs. 1b, 8) | Smaller p moves the curves left by orders of magnitude in bits and costs only a mildly worse loss; DC ≥ OW | The curves sit 1–2 decades apart in bits. Final loss for LS goes from 0.3349 (p = 0.001) to 0.3316 (p = 1); for DC from 0.3314 to 0.3290. DC ≤ OW at every p, with equality at p ≤ 0.01. All p share the first point, as in the paper | replicated |
| E4 (Fig. 9) | LS is sensitive to M early on; OW less so; the DC curves "collapse"; all approach the same floor | Same picture. Spread of the mean train loss across M at a fixed logical time: LS 0.042 / 0.038 / 0.026 at t = 30 / 120 / 240; OW 0.011 / 0.004 / 0.002; DC 0.0008 / 0.0085 / 0.0049. DC collapses best early on. At t ≈ 120–600 OW and DC are comparable, because DC with M = 16 lags slightly; the paper's Fig. 9c shows the same slight lag. Common floor 0.3232 (paper ≈ 0.322) | replicated (DC's collapse is "strongest" only for t ≲ 60) |
| E5 (Fig. 10) | For ζ > 0 overlap beats LS and DC is best; the DC-over-OW gap grows with ζ. At ζ = 0 all three coincide | Round-1 loss for ζ = 12: LS 0.460, OW 0.449, DC 0.441. For ζ = 48: 0.460 / 0.428 / 0.405 (paper 0.461 / 0.418 / 0.405). At ζ = 0 all three curves are bit-identical | replicated |
| E6 (Fig. 11) | Overlap improves very fast and then plateaus **below** LS; DC is worst. Final train acc LS 0.79, OW 0.78, DC 0.74 | Overlap is ahead at round 1 (0.755 vs LS 0.711), and LS overtakes at ≈ 800 time units (round 8). Final train acc LS 0.759, OW 0.743, DC 0.743 (± 0.05–0.07 across seeds). **DC ≈ OW** | partial: the "overlap can hurt" part replicates; "DC worst" and the absolute levels do not |
| E7 (Figs. 12–13) | Overlap improves train and val accuracy vs logical time in both regimes, more clearly in the stress regime. Against processed examples the methods are close, and the overlap variants do not lag or become unstable | Val acc at t = 3000, normal regime: LS 0.493, OW 0.513, DC 0.519. Stress regime: LS 0.364, OW 0.524, DC 0.530. At an equal number of processed examples the curves overlap: train acc at 9·10⁵ examples (normal) is 0.442 / 0.437 / 0.444, and at 5·10⁵ (stress) 0.368 / 0.371 / 0.376 | replicated qualitatively (scaled down, single seed) |
| E8 (Figs. 14–15) | Tiny ImageNet: overlap dominates, OW ≈ DC, and in the stress regime LS is worse even per processed example | not run | not replicated (compute) |

### Paper vs ours, point by point (a9a)

The paper columns are read off the figures. Reading precision is about ±0.003 in loss and ±0.005
in accuracy, plus the unknown seed spread of the paper's own runs. Ours: mean ± std over 10 seeds.
The table is generated by `experiments/compare_a9a.py`.

| Fig. | experiment | method | metric @ round | paper (read off) | ours (mean ± std, 10 seeds) |
|---|---|---|---|---|---|
| 1a/4a | e1_main | LS | train_loss @ 1 | 0.552 | 0.5481 ± 0.0127 |
| 1a/4a | e1_main | OW | train_loss @ 1 | 0.531 | 0.5318 ± 0.0118 |
| 1a/4a | e1_main | DC | train_loss @ 1 | 0.525 | 0.5221 ± 0.0108 |
| 1a/4a | e1_main | LS | train_loss @ 20 | 0.336 | 0.3375 ± 0.0009 |
| 1a/4a | e1_main | OW | train_loss @ 20 | 0.332 | 0.3337 ± 0.0009 |
| 1a/4a | e1_main | DC | train_loss @ 20 | 0.331 | 0.3324 ± 0.0008 |
| 4b | e1_main | LS | val_loss @ 20 | 0.348 | 0.3367 ± 0.0067 |
| 4b | e1_main | OW | val_loss @ 20 | 0.345 | 0.3332 ± 0.0068 |
| 4b | e1_main | DC | val_loss @ 20 | 0.343 | 0.3318 ± 0.0071 |
| 2d | e1_main | LS | val_acc @ 1 | 0.740 | 0.7548 ± 0.0134 |
| 2d | e1_main | OW | val_acc @ 1 | 0.754 | 0.7651 ± 0.0138 |
| 2d | e1_main | DC | val_acc @ 1 | 0.760 | 0.7722 ± 0.0112 |
| 2c | e1_main | DC | val_acc @ 20 | 0.843 | 0.8493 ± 0.0056 |
| 3 | e1_main | LS | grad_norm @ 20 | 0.038 | 0.0375 ± 0.0010 |
| 3 | e1_main | OW | grad_norm @ 20 | 0.031 | 0.0308 ± 0.0005 |
| 3 | e1_main | DC | grad_norm @ 20 | 0.028 | 0.0282 ± 0.0005 |
| 5a | e2_long_compute | OW | train_loss @ 1 | 0.453 | 0.4540 ± 0.0061 |
| 5a | e2_long_compute | DC | train_loss @ 1 | 0.450 | 0.4498 ± 0.0061 |
| 5a | e2_long_compute | OW | train_loss @ 20 | 0.324 | 0.3253 ± 0.0009 |
| 5a | e2_long_compute | DC | train_loss @ 20 | 0.324 | 0.3252 ± 0.0009 |
| 6a/1c | e2_long_delay | OW | train_loss @ 1 | 0.510 | 0.5204 ± 0.0237 |
| 6a/1c | e2_long_delay | DC | train_loss @ 1 | 0.487 | 0.4856 ± 0.0087 |
| 6a/1c | e2_long_delay | OW | train_loss @ 20 | 0.330 | 0.3302 ± 0.0012 |
| 6a/1c | e2_long_delay | DC | train_loss @ 20 | 0.327 | 0.3278 ± 0.0009 |
| 6b | e2_long_delay | OW | val_loss @ 1 | 0.517 | 0.5196 ± 0.0235 |
| 6b | e2_long_delay | DC | val_loss @ 1 | 0.495 | 0.4842 ± 0.0076 |
| 7a | e2_hetero_speed | OW | train_loss @ 1 | 0.485 | 0.4880 ± 0.0111 |
| 7a | e2_hetero_speed | DC | train_loss @ 1 | 0.476 | 0.4746 ± 0.0080 |
| 7a | e2_hetero_speed | OW | train_loss @ 20 | 0.326 | 0.3276 ± 0.0009 |
| 7a | e2_hetero_speed | DC | train_loss @ 20 | 0.326 | 0.3269 ± 0.0009 |
| 8a | e3_sparsity | LS|p=0.001 | train_loss @ 1 | 0.525 | 0.5221 ± 0.0108 |
| 8a | e3_sparsity | LS|p=0.001 | train_loss @ 20 | 0.333 | 0.3349 ± 0.0010 |
| 8a | e3_sparsity | LS|p=1.0 | train_loss @ 20 | 0.330 | 0.3316 ± 0.0008 |
| 8b | e3_sparsity | OW|p=0.1 | train_loss @ 1 | 0.507 | 0.5028 ± 0.0093 |
| 8b | e3_sparsity | OW|p=1.0 | train_loss @ 1 | 0.525 | 0.5221 ± 0.0108 |
| 8c/1b | e3_sparsity | DC|p=0.001 | train_loss @ 1 | 0.504 | 0.5019 ± 0.0094 |
| 8c/1b | e3_sparsity | DC|p=0.001 | train_loss @ 20 | 0.330 | 0.3314 ± 0.0009 |
| 8c/1b | e3_sparsity | DC|p=1.0 | train_loss @ 20 | 0.327 | 0.3290 ± 0.0009 |
| 8c/1b | e3_sparsity | DC|p=0.001 | bits @ 1 | ≈287 | 288 |
| 8c/1b | e3_sparsity | DC|p=0.01 | bits @ 1 | ≈574 | 576 |
| 8c/1b | e3_sparsity | DC|p=0.1 | bits @ 1 | ≈3.72e+03 | 3744 |
| 8c/1b | e3_sparsity | DC|p=1.0 | bits @ 1 | ≈3.57e+04 | 35712 |
| 9a | e4_budget | LS|M=1 | train_loss @ 1 | 0.641 | 0.6341 ± 0.0213 |
| 9a | e4_budget | LS|M=1 | train_loss @ 20 | 0.382 | 0.3821 ± 0.0022 |
| 9a | e4_budget | LS|M=16 | train_loss @ 1 | 0.405 | 0.4047 ± 0.0036 |
| 9a | e4_budget | LS|M=64 | train_loss @ 20 | 0.322 | 0.3232 ± 0.0009 |
| 9b | e4_budget | OW|M=1 | train_loss @ 1 | 0.601 | 0.6004 ± 0.0180 |
| 9b | e4_budget | OW|M=1 | train_loss @ 20 | 0.357 | 0.3555 ± 0.0018 |
| 9c | e4_budget | DC|M=1 | train_loss @ 1 | 0.588 | 0.5829 ± 0.0155 |
| 9c | e4_budget | DC|M=1 | train_loss @ 20 | 0.348 | 0.3492 ± 0.0010 |
| 9c | e4_budget | DC|M=4 | train_loss @ 1 | 0.503 | 0.5019 ± 0.0094 |
| 10a | e5_delay | LS|zeta=12 | train_loss @ 1 | 0.461 | 0.4600 ± 0.0068 |
| 10a | e5_delay | OW|zeta=12 | train_loss @ 1 | 0.446 | 0.4488 ± 0.0069 |
| 10a | e5_delay | DC|zeta=12 | train_loss @ 1 | 0.441 | 0.4411 ± 0.0056 |
| 10b | e5_delay | LS|zeta=48 | train_loss @ 1 | 0.461 | 0.4600 ± 0.0068 |
| 10b | e5_delay | OW|zeta=48 | train_loss @ 1 | 0.418 | 0.4278 ± 0.0181 |
| 10b | e5_delay | DC|zeta=48 | train_loss @ 1 | 0.405 | 0.4047 ± 0.0036 |
| 10b | e5_delay | LS|zeta=48 | train_loss @ 2 | 0.403 | 0.4028 ± 0.0043 |
| 10b | e5_delay | OW|zeta=48 | train_loss @ 2 | 0.374 | 0.3756 ± 0.0120 |
| 10b | e5_delay | DC|zeta=48 | train_loss @ 2 | 0.362 | 0.3625 ± 0.0022 |
| 11a | e6_noniid | LS | train_acc @ 1 | 0.665 | 0.7113 ± 0.0340 |
| 11a | e6_noniid | OW | train_acc @ 1 | 0.776 | 0.7546 ± 0.0487 |
| 11a | e6_noniid | DC | train_acc @ 1 | 0.760 | 0.7549 ± 0.0486 |
| 11a | e6_noniid | LS | train_acc @ 40 | 0.791 | 0.7588 ± 0.0498 |
| 11a | e6_noniid | OW | train_acc @ 40 | 0.778 | 0.7433 ± 0.0673 |
| 11a | e6_noniid | DC | train_acc @ 40 | 0.742 | 0.7433 ± 0.0671 |
| 11b | e6_noniid | LS | val_acc @ 40 | 0.786 | 0.7581 ± 0.0508 |
| 11b | e6_noniid | OW | val_acc @ 40 | 0.774 | 0.7415 ± 0.0658 |
| 11b | e6_noniid | DC | val_acc @ 40 | 0.736 | 0.7416 ± 0.0657 |

### Figures

| Paper | Ours |
|---|---|
| Fig. 1 (summary) | `figures/fig1_summary.png` |
| Fig. 2 (accuracy) | `figures/fig2_accuracy.png` |
| Fig. 3 (gradient norm) | `figures/fig3_gradnorm.png` |
| Fig. 4 (loss vs time and bits) | `figures/fig4_resources.png` |
| Figs. 5–7 (merge rule) | `figures/fig5_7_merge_rule.png` |
| Fig. 8 (sparsity) | `figures/fig8_sparsity.png` |
| Fig. 9 (local budget M) | `figures/fig9_budget.png` |
| Fig. 10 (delay ζ) | `figures/fig10_delay.png` |
| Fig. 11 (non-IID stress) | `figures/fig11_noniid.png` |
| Figs. 12–13 (CIFAR-10) | `figures/fig12_13_cifar10.png` |
| extra: all five methods | `figures/extra_five_methods.png` |

![fig1](figures/fig1_summary.png)
![fig9](figures/fig9_budget.png)
![fig10](figures/fig10_delay.png)
![fig11](figures/fig11_noniid.png)
![cifar](figures/fig12_13_cifar10.png)

### CIFAR-10 (E7, scaled down)

All six runs ran in parallel in ≈15 min of wall time. Setup as in App. B.6: 4 logical workers with
τ = (1, 2, 3, 6); every worker samples from the full training split (45k images; 5k are held out
for validation); random crop (pad 4) + horizontal flip; dropout 0.1; weight decay 5·10⁻⁴; batch 256;
lr 0.05; seed 11; float32 values. Normal regime: p = 0.25, M = 4, ζ = 6. Stress regime: p = 0.10,
M = 2, ζ = 24. Accuracy is measured at x̄, on a fixed 5k subset of the training split (no
augmentation) and on the 5k validation images.

| regime | method | train acc @ t=3000 | val acc @ t=3000 | processed examples | paper (t ≈ 35000), train / val |
|---|---|---|---|---|---|
| normal | Local Sparse | 0.486 | 0.493 | 1.23 M | ≈0.94 / ≈0.84 |
| normal | Overlap overwrite | 0.507 | 0.513 | 1.53 M | ≈0.95 / ≈0.84 |
| normal | Overlap delay-corr. | 0.514 | 0.519 | 1.53 M | ≈0.95 / ≈0.84 |
| stress | Local Sparse | 0.362 | 0.364 | 0.51 M | ≈0.82 / ≈0.80 |
| stress | Overlap overwrite | 0.515 | 0.524 | 1.53 M | ≈0.95 / ≈0.84 |
| stress | Overlap delay-corr. | 0.527 | 0.530 | 1.53 M | ≈0.95 / ≈0.84 |

The ordering and the regime dependence match Figs. 12–13. In the normal regime overlap gives a
modest gain over LS. In the stress regime the gain is large, because LS idles for 24 of every 36
time units. Plotted against processed examples, all three methods lie on one curve. Our OW–DC gap
(≈0.006–0.012) comes from a single seed and is within noise; the paper's OW and DC curves also
nearly coincide. Absolute accuracies are far lower than the paper's, because we train a 58k-parameter
CNN for ≈34 epoch-equivalents, while the paper trains its CNN for 400 epochs.


### Extras (run but not plotted in the paper)

* **All five methods** (E1 setup, logical-time budget 480; `figures/extra_five_methods.png`).
  Final train loss: Sync SGD 0.407, FedAvg-Full 0.3365, Local Sparse 0.3375, OW 0.3337, DC 0.3324.
  Sync SGD is far behind on this time budget: each step costs 12 time units, so it gets only 40
  steps. FedAvg-Full (dense blocking averaging) is only marginally better than sparse Local Sparse.
  Sparsity barely matters here, while overlap does.
* **Nonconvex Geman–McClure regulariser** (eq. 89). The paper gives neither λ nor θ; we used
  λ = 1e-3 and θ = 1 in the E1 setup. Final train loss: LS 0.3386, OW 0.3349, DC 0.3336.
  Gradient norm: 0.0372 / 0.0304 / 0.0278. Same ordering, which supports the paper's remark that
  these runs were "qualitatively similar" (`results/a9a_x1_nonconvex.npz`).

## Assumptions (details the paper leaves unspecified)

| Detail | Our choice | Evidence |
|---|---|---|
| Evaluation point | average model x̄ | the identical first points across p in Fig. 8 hold only for x̄; Theorem 4.1 also uses x̄ |
| Bias term / d | affine model, d = 124 | the bit counts give d ≈ 124.2; the loss floor of ≈ 0.322 needs an intercept |
| K from p | K = ⌈p·d⌉ (at least 1) | bits after round 1 are in ratio 1 : 2 : 13 : 124 |
| Bits accounting | (2n+1)·32·K per round | matches Figs. 1b, 4c and 8 to the pixel |
| Init | PyTorch `nn.Linear` default | round-0 loss 0.733 (expected value 0.735) |
| Seeds / shading | 10 seeds, ±1 std | the paper says only "multi-seed" |
| Worker data (IID) | random equal shards, per-epoch reshuffle | the paper mentions "worker shards"; the choice is immaterial for IID data |
| Validation split | random 10 % of the a9a train file, per seed | — |
| M in B.5 (Fig. 10) | M = 8 | first point at t = 60 for ζ = 12 and t = 96 for ζ = 48, so Mτ = 48 |
| p in B.5 (Fig. 10) | p = 0.3 | the default of every other a9a run |
| Dirichlet partition (E6) | per class, proportions ~ Dir(0.03·1₄), cut the class's indices; resample until every worker has ≥ 10 examples (the NIID-Bench convention) | not specified; with α = 0.03, workers are often left with only 10–100 samples |
| Sync SGD timing | one gradient per worker per step, duration max τ_i + ζ | not specified, and not plotted in the paper |

## Deviations and scale-downs

* **a9a (E1–E6): no scale-down.** Every a9a run is at full paper scale: same rounds, η, batch, p,
  M, ζ and τ. The whole a9a suite takes ≈ 1.5 min on the CPU.
* **CIFAR-10 (E7): heavily scaled down.** (i) Model: the paper's "convolutional classifier" is
  not specified. We use a 57.8k-parameter CNN: a stride-2 3×3 conv stem with 16 channels, then
  conv-pool blocks with 16/32/64 channels, fc 128, dropout 0.1 and fc 10 (`loscar/cifar.py`). We
  chose it for CPU cost: ≈55–90 ms per batch-256 step on 2 busy cores. (ii) Budget: logical time
  3000 instead of ≈35000. That is 100 rounds in the normal regime and 83 in the stress regime, or
  ≈1.5 M processed examples (≈34 epochs) for the overlap methods, against the paper's 400-epoch
  target. (iii) Plain SGD: the paper names no momentum for CIFAR. Weight decay is added to the
  gradient. (iv) Single seed (11), as in the paper. (v) Train accuracy is evaluated on a fixed 5k
  training subset rather than the full 45k.
* **Tiny ImageNet (E8): not run.** ResNet-18 × 100 epochs × 90k images at 64×64, for 3 methods ×
  2 regimes, is GPU-scale (CPU estimate: days). The download alone is 237 MB. We did not substitute a
  proxy, because the claims of interest (OW ≈ DC on a deep net, LS worse per processed example in
  the stress regime) are architecture-specific.

## Discrepancies and notes

1. **Fig. 11 / E6 (stress test).** In the paper, DC ends ≈ 3.6 accuracy points below OW, and it is
   already ≈ 1.6 points below after round 1. With p = 0.01 as stated, K = ⌈0.01·124⌉ = 2
   coordinates are merged per round. OW and DC differ *only* on those K coordinates of x̄, so the
   two methods are almost identical: in our runs their accuracies differ by at most 0.003 for any
   seed and round. Two diagnostics (not in the paper) did not recover the paper's gap. (a) Denser masks, p ∈ {0.1, 0.3, 1}
   (`results/a9a_d1_noniid_p.npz`), still give DC ≈ OW (e.g. 0.771 vs 0.773 at p = 0.3).
   (b) A minimum of 256 examples per worker (`results/a9a_d2_noniid_min256.npz`) gives LS 0.771
   and OW = DC = 0.757. LS ends above both overlap methods in every variant, so the paper's main
   message for this test ("aggressive overlap can hurt under strong data heterogeneity") holds.
   The paper's DC-specific degradation and its higher accuracy levels (≈ 0.79 for LS) presumably
   come from an unstated partition or implementation detail. The partition protocol is the
   ambiguity that would need the authors' code to resolve.
2. **Validation loss offset (Figs. 4b, 6b).** Our final validation losses are ≈ 0.011 lower than
   the paper's (e.g. DC 0.332 vs 0.343), while train losses match to ≈ 0.001. A random 10 % split
   gives val ≈ train loss. In the paper, val exceeds train by ≈ 0.012, which is ≈ 1.6× the
   single-seed std of our val loss. Using the first or last 10 % of the file (unshuffled) as the
   split does not explain it (0.336 either way). This looks like seed or split luck; it does not
   change any ordering.
3. **Paper figure labelling.** Figs. 2a and 2b are captioned "Training accuracy versus round", but
   the embedded images plot *training loss*: 2a is the same image object as Fig. 1a, and 2b is
   train loss vs round on a log x-axis. We plot real training accuracy in `fig2_accuracy.png`. Fig. 6a is the same image as Fig. 1c.
   Fig. 11's x-axis is logical time (40 rounds × 102).
4. Our val accuracy is slightly higher (round 20: 0.849 vs 0.843; round 1: 0.755–0.772 vs
   0.740–0.760), which is consistent with note 2: our val split is slightly "easier".

## Compute

All on CPU (Intel Meteor Lake laptop, shared with other jobs). The a9a suite (E1–E6, extras and
diagnostics; 65 configurations × 10 seeds = 650 runs) takes ≈ 1.5 min with 5 processes.
CIFAR-10 takes ≈ 15 min wall time with 5 processes × 2 threads; the longest single run took
15.1 min. Raw curves are in `results/*.npz` (< 1 MB); figures take 1.5 MB.
