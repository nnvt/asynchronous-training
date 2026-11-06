# Replication: Ringleader ASGD

Paper: [Ringleader ASGD: The First Asynchronous SGD with Optimal Time Complexity under Data Heterogeneity](https://proceedings.iclr.cc/paper_files/paper/2026/hash/08d70284b013c03ba89cd2b642bc864b-Abstract-Conference.html) (Maranjyan, Richtárik, ICLR 2026). No official code exists. The paper's only experiment (Fig. 1) was run at full scale: 100 clients, Dirichlet(0.1) labels, 784-128-10 MLP, 5 seeds.

**Verdict:** Ringleader wins on the paper's metric, the squared gradient norm ‖∇f‖². On training loss it does not: the winning stepsizes collapse the network.

## Results

| Experiment | Paper (Ringleader / Malenia / IA²SGD) | Ours | Status | What doesn't match, and why |
|---|---|---|---|---|
| Fig. 1a, MNIST, final ‖∇f‖² | 2.1e-4 / 1.6e-3 / 1.2e-3 | 5.1e-4 / 3.2e-3 / 2.5e-3 | ✅ Ordering | Our values are 2–2.4× higher. Per seed, Ringleader beats Malenia in 4/5 and IA²SGD in only 3/5 (we ran 5 seeds, the paper 30) |
| Fig. 1b, Fashion-MNIST, final ‖∇f‖² | 5.9e-4 / 1.3e-2 / 1.4e-3; Ringleader passes IA²SGD at t ≈ 200k | 4.4e-4 / 1.5e-2 / 2.9e-3; passes at t ≈ 130k | ⚠️ Depends on tuning rule | If γ is chosen by mean log ‖∇f‖² over the whole run instead of the tail, IA²SGD ends lowest. The paper doesn't say which rule it used |
| Claim: Ringleader converges faster, i.e. trains better | Only ‖∇f‖² reported | The winning γ (0.1) collapses the network in 5/5 seeds: final loss 2.10–2.30 (chance = ln 10 = 2.30), train accuracy 25% on MNIST and 11.6% on FMNIST, 80% dead ReLUs. IA²SGD reaches loss 0.013 on MNIST. With γ tuned on loss, Ringleader is slowest: MNIST 0.032 / 0.020 / 0.005, FMNIST 0.23 / 0.20 / 0.15 | ❌ | The small ‖∇f‖² comes from a dead network, not a trained one. Ringleader applies ~100 near-identical updates per round (effective step ≈ 100γ), which blows up early. We can't check the paper's own runs: it reports no loss or accuracy, though its early spikes look similar |

## Plots

Dashed lines are the paper's medians, digitized from its Fig. 1. Bands are the IQR over seeds.

**Fig. 1 replication, ‖∇f‖² (γ tuned on tail ‖∇f‖², 5 seeds)**
![](figures/fig1.png)

**Same runs, training loss: Ringleader collapses to ln 10**
![](figures/fig1_loss.png)

**Stepsize sweep on the paper's grid**
![](figures/tuning.png)

**γ chosen by mean log ‖∇f‖² over the whole run: ‖∇f‖², then loss (3 seeds)**
![](figures/fig1_auc.png)
![](figures/fig1_auc_loss.png)

**γ tuned on training loss: ‖∇f‖², then loss (3 seeds)**
![](figures/fig1_losstuned.png)
![](figures/fig1_losstuned_loss.png)

## Notes

- Assumptions: per-gradient time jitter, PyTorch default init, γ tuned on 1 seed, budgets read off the figure axes.

## Run

```bash
export RL_DATA=/path/to/data   # MNIST gz files; Fashion-MNIST in $RL_DATA/fashion-mnist
python run.py tune  --dataset mnist fmnist
python run.py final --dataset mnist fmnist --seeds 5
python run.py final --dataset mnist fmnist --seeds 3 --tag losstuned --criterion loss
python run.py plot
```

Full details: [DETAILS.md](DETAILS.md).
