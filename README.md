# Replication: LOSCAR-SGD

Paper: [LOSCAR-SGD: Local SGD with Communication-Computation Overlap and Delay-Corrected Sparse Model Averaging](https://arxiv.org/abs/2605.20866) (Maziane, Mahran, Maranjyan, Richtárik, 2026). No official code exists. We re-implemented the paper's logical-time simulator. The a9a experiments were run at full scale with 10 seeds.

**Verdict:** Mostly replicated. The a9a results match closely; the non-IID test is partial; CIFAR-10 was scaled down; Tiny ImageNet was not run.

## Results

Methods: LS = Local Sparse, OW = overlap + overwrite, DC = overlap + delay-corrected (LOSCAR).

| Experiment | Paper | Ours | Status | What doesn't match, and why |
|---|---|---|---|---|
| E1, a9a: DC < OW < LS (Figs. 1a, 2–4), final train loss DC / OW / LS | 0.331 / 0.332 / 0.336 | 0.3324 / 0.3337 / 0.3375; ordering holds in 10/10 seeds at every round | ✅ | — |
| E2, merge rule, long delay (Figs. 5–7), round-1 train loss DC / OW | 0.487 / 0.510 | 0.486 / 0.520 | ✅ | Our OW is 2% higher |
| E3, sparsity (Fig. 8), bits after round 1 for p = 0.001 / 0.01 / 0.1 / 1 | ≈287 / 574 / 3.7e3 / 3.6e4 | 288 / 576 / 3744 / 35712 | ✅ | — |
| E4, local budget M (Fig. 9), common loss floor | 0.322 | 0.323 | ✅ | — |
| E5, comm. delay ζ = 48 (Fig. 10), round-1 loss LS / OW / DC | 0.461 / 0.418 / 0.405 | 0.460 / 0.428 / 0.405 | ✅ | Our OW is 2% higher |
| E6, non-IID, Dirichlet 0.03 (Fig. 11), final train accuracy LS / OW / DC | 0.79 / 0.78 / 0.74 (DC worst) | 0.759 / 0.743 / 0.743 (±0.05–0.07 across seeds) | ⚠️ Partial | "Overlap can end below LS" replicates: LS overtakes at round 8. "DC is worst" doesn't: our OW falls to DC's level (0.743 vs the paper's 0.78), and our LS is 0.03 lower. At p = 0.01 only 2 coordinates are merged per round, so OW and DC can barely differ. Denser masks and larger shards didn't restore the gap. Most likely cause: the paper's data split; settling it needs the authors' code |
| E7, CIFAR-10 (Figs. 12–13), val accuracy LS / OW / DC | Stress regime ≈0.80 / 0.84 / 0.84 at t ≈ 35000; normal ≈0.84 for all | Stress 0.364 / 0.524 / 0.530 at t = 3000; normal 0.493 / 0.513 / 0.519 | ⚠️ Ordering only | Accuracy is 0.3–0.45 lower because the run is scaled down: 1/12 of the time budget (≈34 vs 400 epochs), a 58k-parameter CNN (the paper doesn't give its model), 1 seed. The ordering, the larger gain in the stress regime, and the methods coinciding per processed example all match |
| E8, Tiny ImageNet, ResNet-18 (Figs. 14–15) | Overlap dominates; OW ≈ DC | Not run | ❌ | GPU-scale; this machine has no GPU |

## Plots

a9a plots use 10 seeds; CIFAR-10 uses 1 seed.

**Fig. 1, summary**
![](figures/fig1_summary.png)

**Fig. 2, training accuracy**
![](figures/fig2_accuracy.png)

**Fig. 3, gradient norm**
![](figures/fig3_gradnorm.png)

**Fig. 4, loss vs time and bits**
![](figures/fig4_resources.png)

**Figs. 5–7, merge rule in three regimes**
![](figures/fig5_7_merge_rule.png)

**Fig. 8, sparsity**
![](figures/fig8_sparsity.png)

**Fig. 9, local budget M**
![](figures/fig9_budget.png)

**Fig. 10, communication delay**
![](figures/fig10_delay.png)

**Fig. 11, non-IID stress test**
![](figures/fig11_noniid.png)

**Figs. 12–13, CIFAR-10 (scaled down)**
![](figures/fig12_13_cifar10.png)

**Extra: all five methods, including Sync SGD and FedAvg-Full**
![](figures/extra_five_methods.png)

## Notes

- Read off the figures: metrics at the averaged model, bits = 9·32·K, and M=8 for E5.

## Run

```bash
export LOSCAR_DATA=/path/to/data   # a9a and CIFAR-10
python experiments/run_a9a.py --seeds 10 --procs 5 && python experiments/plot_a9a.py
python experiments/run_cifar.py --budget 3000 --procs 5 && python experiments/plot_cifar.py
```

Full details: [DETAILS.md](DETAILS.md).
