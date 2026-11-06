# Replication: Rescaled ASGD

Paper: [Rescaled Asynchronous SGD: Optimal Distributed Optimization under Data and System Heterogeneity](https://arxiv.org/abs/2605.13434) (Mahran, Maranjyan, Richtárik, 2026). No official code exists. All experiments were run at full scale in simulated wall-clock time, 5 seeds.

**Verdict:** The main result replicates; Rescaled ASGD matches the paper's loss exactly. Several smaller figures don't follow from the setup as written.

## Results

| Experiment | Paper (Rescaled / Malenia / Ringleader) | Ours | Status |
|---|---|---|---|
| Fig. 1a, MNIST, fixed times, final loss | 0.029 / 0.120 / 0.096 | 0.029 / 0.068 / 0.099 | ✅ Replicated (Rescaled wins 2.3–3.4×) |
| Fig. 1b, MNIST, random times | 0.029 / 0.152 / 0.140 | 0.030 / 0.101 / 0.122 | ✅ Replicated |
| Fig. 2, cumulative step | All methods advance at the same rate | Rescaled and Malenia match. Ringleader as published reaches 9.0, not 13.7 | ⚠️ Partial |
| Fig. 3, Vanilla vs Rescaled | Converge to +1/2 and −2/3; Vanilla faster | Targets match. With equal stepsizes, Rescaled is faster | ⚠️ Targets only |
| Fig. 4, Delay-Adaptive vs Rescaled | Converge to ≈1 and ≈−0.04 | 0.9995 and −0.047 | ✅ Replicated |

## Plots

**Fig. 1, MNIST, fixed and random compute times (5 seeds)**
![](figures/fig1_mnist.png)

**Fig. 1 stepsize tuning (seed 0)**
![](figures/fig1_tuning.png)

**Fig. 1 diagnostics: baselines at the stepsizes that match the paper's curves (single seed)**
![](figures/fig1_diagnostics.png)

**Fig. 2, cumulative stepsize over time**
![](figures/fig2_cumulative_step.png)

**Fig. 3, Vanilla vs Rescaled ASGD on the toy problem**
![](figures/fig3_vanilla_vs_rescaled.png)

**Fig. 4, Delay-Adaptive vs Rescaled ASGD on the toy problem**
![](figures/fig4_delay_adaptive.png)

## Notes

- Our tuned baselines (γ=0.1) beat the paper's. Its baseline curves match Malenia at γ=0.05 (off the grid) and Ringleader at γ=0.01.
- Ringleader's Fig. 2 value of 13.7 needs a variant that reuses stale gradient-table entries, and that variant doesn't match its Fig. 1 curve. The two figures seem to come from different code.
- The paper's Delay-Adaptive curve looks like plain ASGD, not its stated γ/(1+δ) rule.
- Assumptions: worker i holds digit i, PyTorch default init, γ tuned on seed 0 by final loss.

## Run

```bash
python run_toy.py                                            # Figs. 2-4
python run_mnist.py --procs 4 --methods rescaled,malenia,ringleader   # Fig. 1
python plot.py
```

Full details: [DETAILS.md](DETAILS.md).
