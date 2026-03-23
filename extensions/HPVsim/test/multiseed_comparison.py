#!/usr/bin/env python
"""
Multi-seed trajectory comparison for Python hpvsim.

Runs N_SEEDS simulations for each scenario, saves per-seed and mean
trajectories to JSON for the Julia multiseed_comparison.jl to read.
"""

import os, json, time
import numpy as np
import hpvsim as hpv

OUT_DIR   = os.path.join(os.path.dirname(__file__), "cross_validation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_AGENTS  = 5_000
N_YEARS   = 50
DT        = 0.25
START     = 2000
N_SEEDS   = 50


def run_multiseed(scenario_name, make_sim_fn, extract_fn):
    """Run N_SEEDS simulations and collect trajectories."""
    print(f"\n{'='*60}")
    print(f"  {scenario_name}: running {N_SEEDS} seeds")
    print(f"{'='*60}")

    all_results = []
    t0 = time.time()
    for seed in range(1, N_SEEDS + 1):
        sim = make_sim_fn(seed)
        sim.run()
        res = extract_fn(sim)
        all_results.append(res)
        if seed % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Seed {seed}/{N_SEEDS} done ({elapsed:.1f}s)")

    # Compute means across seeds for each trajectory key
    keys = [k for k in all_results[0] if isinstance(all_results[0][k], list)]
    mean_results = {}
    for k in keys:
        stacked = np.array([r[k] for r in all_results])
        mean_results[k] = stacked.mean(axis=0).tolist()
        mean_results[f'{k}_std'] = stacked.std(axis=0).tolist()

    # Save
    out = {
        'scenario': scenario_name,
        'n_seeds': N_SEEDS,
        'n_agents': N_AGENTS,
        'n_years': N_YEARS,
        'dt': DT,
        'mean': mean_results,
    }
    path = os.path.join(OUT_DIR, f"multiseed_{scenario_name}_python.json")
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f"  Saved → {path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    return out


# ── Scenario 1: Single HPV16 ──

def scenario1():
    def make_sim(seed):
        return hpv.Sim(
            n_agents       = N_AGENTS,
            genotypes      = [16],
            n_years        = N_YEARS,
            dt             = DT,
            start          = START,
            rand_seed      = seed,
            verbose        = 0,
            ms_agent_ratio = 1,
        )

    def extract(sim):
        return dict(
            year     = sim.results['year'].tolist(),
            hpv_prev = sim.results['hpv_prevalence'].values.tolist(),
            cin_prev = sim.results['cin_prevalence'].values.tolist(),
        )

    return run_multiseed('single_hpv16', make_sim, extract)


# ── Scenario 2: Two-genotype HPV16+18 ──

def scenario2():
    def make_sim(seed):
        return hpv.Sim(
            n_agents       = N_AGENTS,
            genotypes      = [16, 18],
            n_years        = N_YEARS,
            dt             = DT,
            start          = START,
            rand_seed      = seed,
            verbose        = 0,
            ms_agent_ratio = 1,
        )

    def extract(sim):
        bg_prev = sim.results['hpv_prevalence_by_genotype'].values
        bg_cin  = sim.results['cin_prevalence_by_genotype'].values
        return dict(
            year          = sim.results['year'].tolist(),
            hpv_prev      = sim.results['hpv_prevalence'].values.tolist(),
            cin_prev      = sim.results['cin_prevalence'].values.tolist(),
            hpv_prev_hpv16 = bg_prev[0].tolist(),
            hpv_prev_hpv18 = bg_prev[1].tolist(),
            cin_prev_hpv16 = bg_cin[0].tolist(),
            cin_prev_hpv18 = bg_cin[1].tolist(),
        )

    return run_multiseed('twogen_hpv16_18', make_sim, extract)


if __name__ == '__main__':
    scenario1()
    scenario2()
    print("\nDone!")
