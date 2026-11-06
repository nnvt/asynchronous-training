# Replication: Ringmaster ASGD

Paper: [Ringmaster ASGD: The First Asynchronous SGD with Optimal Time Complexity](https://arxiv.org/abs/2501.16168) (Maranjyan, Tyurin, Richtárik, 2025). No official code exists; this is a re-implementation as an event-driven simulation of 6174 workers.

**Verdict:** The method ranking replicates. The paper's absolute curves do not come out of the setup it describes.

## Results

| Experiment | Paper | Ours | Status |
|---|---|---|---|
| Fig. 2, quadratic: Ringmaster > Rennala ≫ Delay-Adaptive ASGD | Ringmaster fastest | Same ordering at every time. Ringmaster reaches each target 1.6–1.8× sooner than Rennala (gap 0.03 at 474 s vs 847 s); Delay-Adaptive never gets below 0.1 within 2000 s | ✅ Replicated |
| Fig. 2 tuned hyper-parameters | γ=0.2, R=6 / γ=1, B=6 | γ=0.2, R=7 / γ=1, B=6 | ✅ Replicated |
| Fig. 2 values | Ringmaster 2.5e-5 at 2000 s | 1.6e-2 with the setup as written. A variant (b=0, noiseless, compute times based on √i) gives 2.9e-5 | ❌ Not as written |
| Fig. 3, MNIST MLP: Ringmaster best | — | Re-tuned: Ringmaster 0.26 vs 0.43 (Delay-Adaptive) and 0.46 (Rennala) training loss at 1000 s. At the paper's stepsizes Ringmaster and Rennala don't train (loss stays 2.37) | ⚠️ Partial |

## Plots

Median of 5 seeds with a min–max band. Hollow markers are values read off the paper's figures.

**Fig. 2, setup as written, paper's hyper-parameters**
![](figures/text_paper.png)

**Fig. 2, setup as written, re-tuned (criterion: mean log gap over the whole run)**
![](figures/text_tune-auc.png)

**Fig. 2, setup as written, re-tuned (criterion: final gap)**
![](figures/text_tune-final.png)

**Fig. 2, fixed per-worker compute times (sensitivity check)**
![](figures/text-fixedtime_paper.png)

**Fig. 2, variant matching the paper (b=0, noiseless, √i times), paper's hyper-parameters**
![](figures/fig2match_paper.png)

**Fig. 2, matching variant, re-tuned (whole run / final)**
![](figures/fig2match_tune-auc.png)
![](figures/fig2match_tune-final.png)

**Fig. 3, MNIST, paper's stepsizes**
![](figures/mnist_paper.png)

**Fig. 3, MNIST, re-tuned**
![](figures/mnist_tune-auc.png)

## Notes

- As written, the problem can't go below ~3e-3 even with noiseless gradient descent. The paper's 2.5e-5 must come from a different setup.
- MNIST details are unstated. We used a 784-32-10 MLP, 3000 images, and batch 4. The paper's 1e-14 losses are not reproduced; they look like a network that blew up and stopped learning.
- Stepsize tuning uses the paper's grid; the criterion is mean log gap over the run.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install numpy matplotlib
.venv/bin/python run.py --setup text  --mode paper   # Fig. 2
.venv/bin/python run.py --setup text  --mode tune --criterion auc
.venv/bin/python run.py --setup mnist --mode tune --criterion auc --methods ringmaster rennala da_asgd
```

Full details: [DETAILS.md](DETAILS.md).
