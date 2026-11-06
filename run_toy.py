"""App. F toy experiments of arXiv:2605.13434 (Figs. 3 and 4) and the update
schedule of Fig. 2. Cheap (seconds); single process.

Fig. 3: tau = (1, 2), F1 = (x-4)^2, F2 = 2(x+3)^2, x0 = 5, sigma = 0, T = 300.
Fig. 4: tau = (1, 100), F1 = (x-1)^2, F2 = (x+1)^2, x0 = 3, sigma = 0, T = 1500.
Fig. 2: cumulative stepsize (in units of the cycle stepsize alpha) of
        Rescaled / Malenia / Ringleader for tau in {1,2,4,8,16}x2.

Writes results/toy_summary.json and results/schedule_summary.json,
results/toy_traces.npz, results/schedule.npz.
"""
import json
import os

import numpy as np

from rasgd.sim import (delay_adaptive_rule, harmonic_mean, run_asgd, run_malenia,
                       run_ringleader, vanilla_rule)
from rasgd.toy import Quadratics1D, ScheduleOnly

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")


def const_rule(gis):
    gis = np.asarray(gis, dtype=float)
    return lambda i, delay: gis[i]


def toy_run(prob, taus, rule, T):
    grid = np.array([0.0, T])
    r = run_asgd(prob, taus, "fixed", rule, T, grid, 0, blowup=1e9, trace=True)
    t = np.concatenate([[0.0], r["steps_t"]])
    x = np.concatenate([[prob.x0], r["xs"][:, 0]])
    w = np.concatenate([[-1], r["steps_w"]])
    lr = np.concatenate([[np.nan], r["lrs"]])
    tmax = max(taus)
    # cycle iterate x^m = server model after all events with time <= m * tau_max
    mt = np.arange(0, T + 1e-9, tmax)
    idx = np.searchsorted(t, mt, side="right") - 1
    return dict(t=t, x=x, w=w, lr=lr, cycle_t=mt, cycle_x=x[idx])


def at(res, times):
    idx = np.searchsorted(res["cycle_t"], times)
    return [float(res["cycle_x"][j]) for j in idx]


def fig3():
    taus = [1.0, 2.0]
    prob = Quadratics1D(a=[1.0, 2.0], c=[4.0, -3.0], x0=5.0)
    n, tmax, K = 2, 2.0, 3
    tH = harmonic_mean(taus)
    alpha = 0.01
    xstar = prob.minimizer()  # -2/3
    w_freq = 1.0 / np.asarray(taus)
    xtilde = prob.minimizer(w_freq)  # +1/2
    variants = {
        # (a) as written in App. F.1: both methods with cycle stepsize alpha = 0.01
        "as_written": {
            "rescaled": [alpha * t / (n * tmax) for t in taus],  # sum over a cycle = alpha
            "vanilla": [alpha / K] * n,  # gamma_i = gamma/K, sum over a cycle = alpha
        },
        # (b) what reproduces the paper's Fig. 3 curves (see README): Rescaled with
        #     eq. (2) and gamma = 0.01 (cycle stepsize gamma*tau_H = 0.0133) and
        #     Vanilla ASGD with constant per-step stepsize 0.01 (cycle stepsize 0.03)
        "inferred_from_figure": {
            "rescaled": [0.01 * t * tH / (n * tmax) for t in taus],
            "vanilla": [0.01] * n,
        },
    }
    paper = {  # cycle iterates read off the rendered Fig. 3(b)
        "times": [30, 60, 90, 300],
        "rescaled": [2.39, 0.98, 0.22, -0.66],
        "vanilla": [1.74, 0.81, 0.56, 0.45],
    }
    out = {"xstar": xstar, "xtilde": xtilde, "paper_readoff": paper, "variants": {}}
    traces = {}
    for vname, gis in variants.items():
        out["variants"][vname] = {}
        for m, g in gis.items():
            res = toy_run(prob, taus, const_rule(g), 300.0)
            traces[f"fig3_{vname}_{m}"] = res
            tail = res["cycle_x"][-10:]
            out["variants"][vname][m] = {
                "stepsizes": g, "cycle_stepsize": float(sum(gi * tmax / t for gi, t in zip(g, taus))),
                "cycle_x_at_paper_times": at(res, paper["times"]),
                "final_cycle_x": float(res["cycle_x"][-1]),
                "dist_to_xstar": float(abs(res["cycle_x"][-1] - xstar)),
                "dist_to_xtilde": float(abs(res["cycle_x"][-1] - xtilde)),
                "tail_all_iterates_range": [float(res["x"][-30:].min()), float(res["x"][-30:].max())],
                "tail_cycle_mean": float(tail.mean()),
            }
    return out, traces


def fig4():
    taus = [1.0, 100.0]
    prob = Quadratics1D(a=[1.0, 1.0], c=[1.0, -1.0], x0=3.0)
    n, tmax = 2, 100.0
    tH = harmonic_mean(taus)
    T = 1500.0
    # Stepsizes are not given in App. F.2. Inferred from the rendered Fig. 4:
    #   Rescaled: cycle contraction ~0.61 and limit ~-0.04 => eq. (2) with gamma = 0.1
    #   Delay-Adaptive: initial decay 3 -> 1.27 over 100 fast steps => gamma = 0.01
    g_res = 0.1
    g_da = 0.01
    rules = {
        "rescaled": const_rule([g_res * t * tH / (n * tmax) for t in taus]),
        "delay_adaptive": delay_adaptive_rule(g_da, "paper"),  # gamma/(1+delta)
        "delay_adaptive_mishchenko": delay_adaptive_rule(g_da, "mishchenko", n=n),
        "vanilla": vanilla_rule(g_da),  # diagnostic: no delay adaptation
    }
    paper = {
        "rescaled_cycle_times": [100, 200, 300, 400, 500, 1500],
        "rescaled_cycle_x": [1.85, 1.13, 0.68, 0.41, 0.24, -0.04],
        "da_x_before_first_slow_update_t99": 1.27,
        "da_jump_at_t100": 0.085,
        "da_dip_depth_steady": 0.045,
        "da_plateau_range": [0.95, 1.0],
    }
    out = {"xstar": 0.0, "x1star": 1.0, "paper_readoff": paper, "gamma_rescaled": g_res,
           "gamma_da": g_da, "methods": {}}
    traces = {}
    for m, rule in rules.items():
        res = toy_run(prob, taus, rule, T)
        traces[f"fig4_{m}"] = res
        t, x, w, lr = res["t"], res["x"], res["w"], res["lr"]
        slow = w == 1
        x99 = float(x[np.searchsorted(t, 99.0, side="right") - 1])
        # jump of the first slow update (t = 100): x right before vs. right after
        j = np.flatnonzero(slow)[0]
        jump1 = float(x[j - 1] - x[j])
        # steady-state dip depth (last slow update)
        jl = np.flatnonzero(slow)[-1]
        dip = float(x[jl - 1] - x[jl])
        late = t > 1000
        out["methods"][m] = {
            "cycle_x_at": {str(int(tt)): float(v) for tt, v in
                           zip(paper["rescaled_cycle_times"], at(res, paper["rescaled_cycle_times"]))},
            "x_t99": x99, "jump_first_slow_update": jump1, "dip_last_slow_update": dip,
            "late_range": [float(x[late].min()), float(x[late].max())],
            "final_x": float(x[-1]),
            "fast_step": float(lr[w == 0][5]), "slow_step": float(lr[slow][-1]),
        }
    # weights of the objective targeted by DA (App. D.2): w_i ~ Gamma_i = sum of its stepsizes per cycle
    lr = traces["fig4_delay_adaptive"]["lr"]
    w = traces["fig4_delay_adaptive"]["w"]
    t = traces["fig4_delay_adaptive"]["t"]
    sel = (t > 1000) & (t <= 1100)
    G = np.array([lr[sel & (w == i)].sum() for i in range(2)])
    wh = G / G.sum()
    out["delay_adaptive_target_weights"] = wh.tolist()
    out["delay_adaptive_target_minimizer"] = float(prob.minimizer(wh))
    out["vanilla_target_minimizer"] = float(prob.minimizer(1.0 / np.asarray(taus)))
    return out, traces


def schedule():
    """Fig. 2: cumulative stepsize in units of alpha.
    Rescaled: worker i steps alpha*tau_i/(n*tau_max); Ringleader: alpha/n; Malenia: alpha."""
    taus = np.array([1, 1, 2, 2, 4, 4, 8, 8, 16, 16], dtype=float)
    n, tmax = len(taus), taus.max()
    P = ScheduleOnly()
    T_plot = 14.3 * tmax
    out, arrays = {}, {}

    def incr(method, w):
        if method == "rescaled":
            return taus[w] / (n * tmax)
        if method in ("ringleader", "ringleader_reuse"):
            return np.full(len(w), 1.0 / n)
        return np.ones(len(w))

    runners = {"rescaled": lambda mode, T, s: run_asgd(P, taus, mode, lambda i, d: 0.0, T, np.array([T]), s),
               "malenia": lambda mode, T, s: run_malenia(P, taus, mode, 0.0, T, np.array([T]), s),
               "ringleader": lambda mode, T, s: run_ringleader(P, taus, mode, 0.0, T, np.array([T]), s),
               # diagnostic variant (see rasgd/sim.py)
               "ringleader_reuse": lambda mode, T, s: run_ringleader(P, taus, mode, 0.0, T, np.array([T]), s,
                                                                     reuse_table=True)}
    for mode in ("fixed", "exp"):
        out[mode] = {}
        for m, fn in runners.items():
            r = fn(mode, T_plot, 0)
            arrays[f"{mode}_{m}_t"] = r["steps_t"] / tmax
            arrays[f"{mode}_{m}_c"] = np.cumsum(incr(m, r["steps_w"]))
            # long-run rate (alpha per tau_max) and value at 14.3 tau_max over many seeds
            seeds = range(1) if mode == "fixed" else range(100)
            at14, rates, curves = [], [], []
            tg = np.linspace(0, 14.3, 287)
            for s in seeds:
                rr = fn(mode, 400 * tmax, 1000 + s)
                c = np.r_[0.0, np.cumsum(incr(m, rr["steps_w"]))]
                tt = rr["steps_t"]
                at14.append(c[np.searchsorted(tt, T_plot, side="right")])
                rates.append(c[-1] / (400.0))
                curves.append(c[np.searchsorted(tt, tg * tmax, side="right")])
            arrays[f"{mode}_{m}_band_t"] = tg
            arrays[f"{mode}_{m}_band_q"] = np.quantile(np.array(curves), [0.1, 0.5, 0.9], axis=0)
            out[mode][m] = {"cum_step_at_14.3tmax_seed0": float(arrays[f"{mode}_{m}_c"][-1]) if len(arrays[f"{mode}_{m}_c"]) else 0.0,
                            "cum_step_at_14.3tmax_median": float(np.median(at14)),
                            "cum_step_at_14.3tmax_q10_q90": [float(np.quantile(at14, 0.1)), float(np.quantile(at14, 0.9))],
                            "long_run_rate_alpha_per_tmax": float(np.mean(rates))}
            if m != "rescaled":
                # one round (Malenia: 1 update of alpha; Ringleader: n updates of alpha/n)
                # advances the cumulative step by exactly alpha
                out[mode][m]["mean_round_duration_in_tmax"] = float(1.0 / np.mean(rates))
    out["paper_readoff"] = {"fixed": {"rescaled": 14.1, "malenia": 14.0, "ringleader": 13.7},
                            "exp": {"rescaled": 15.2, "malenia": 9.0, "ringleader": 5.2}}
    return out, arrays


def main():
    os.makedirs(RES, exist_ok=True)
    s3, t3 = fig3()
    s4, t4 = fig4()
    with open(os.path.join(RES, "toy_summary.json"), "w") as fh:
        json.dump({"fig3": s3, "fig4": s4}, fh, indent=1)
    flat = {}
    for k, v in {**t3, **t4}.items():
        for kk, vv in v.items():
            flat[f"{k}__{kk}"] = vv
    np.savez_compressed(os.path.join(RES, "toy_traces.npz"), **flat)
    ss, sa = schedule()
    with open(os.path.join(RES, "schedule_summary.json"), "w") as fh:
        json.dump(ss, fh, indent=1)
    np.savez_compressed(os.path.join(RES, "schedule.npz"), **sa)
    print(json.dumps({"fig3": s3, "fig4": s4}, indent=1))
    print(json.dumps(ss, indent=1))


if __name__ == "__main__":
    main()
