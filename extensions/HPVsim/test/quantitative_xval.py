#!/usr/bin/env python
"""
Quantitative cross-validation: Python hpvsim reference data.

Runs N_SEEDS simulations for two scenarios and writes summary statistics
(mean ± 95% CI) to JSON for comparison with Julia.

Scenarios:
  1. HPV16 only — natural history
  2. HPV16 + vaccination starting year 20

Python hpvsim uses:
  - Duration-based CIN model (event-scheduled at infection time)
  - Demographics (births, deaths, migration)
  - dt = 0.25 years (quarterly)
"""

import os
import json
import numpy as np
import hpvsim as hpv

OUT_DIR = os.path.join(os.path.dirname(__file__), "quantitative_xval_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_AGENTS = 5_000
N_YEARS  = 50
DT       = 0.25
START    = 2000
N_SEEDS  = 20

# Timepoints to sample (in years since start)
SAMPLE_YEARS = [10, 25, 50]


def get_timepoint_index(sim, year_offset):
    """Get the result index closest to start + year_offset."""
    target = START + year_offset
    years = sim.results['year']
    idx = int(np.argmin(np.abs(years - target)))
    return idx


def run_scenario1(seed):
    """HPV16 only — natural history."""
    sim = hpv.Sim(
        n_agents   = N_AGENTS,
        genotypes  = [16],
        n_years    = N_YEARS,
        dt         = DT,
        start      = START,
        rand_seed  = seed,
        verbose    = 0,
    )
    sim.run()

    metrics = {}
    for yr in SAMPLE_YEARS:
        idx = get_timepoint_index(sim, yr)
        metrics[f'hpv_prev_yr{yr}'] = float(sim.results['hpv_prevalence'].values[idx])
        metrics[f'cin_prev_yr{yr}'] = float(sim.results['cin_prevalence'].values[idx])
        metrics[f'cancer_inc_yr{yr}'] = float(sim.results['cancer_incidence'].values[idx])

    # Cumulative cancers from incidence timeseries
    cancer_inc = sim.results['cancer_incidence'].values
    n_female = sim.results['n_females_alive'].values if 'n_females_alive' in sim.results else None
    metrics['cum_cancers'] = float(np.nansum(cancer_inc))

    return metrics


def run_scenario2(seed):
    """HPV16 + bivalent vaccination starting year 20."""
    # Baseline (no vaccination)
    sim_base = hpv.Sim(
        n_agents   = N_AGENTS,
        genotypes  = [16],
        n_years    = N_YEARS,
        dt         = DT,
        start      = START,
        rand_seed  = seed,
        verbose    = 0,
    )
    sim_base.run()

    # With vaccination
    vx = hpv.routine_vx(
        prob       = 0.9,
        product    = 'bivalent',
        age_range  = [9, 14],
        start_year = START + 20,
    )
    sim_vx = hpv.Sim(
        n_agents      = N_AGENTS,
        genotypes     = [16],
        n_years       = N_YEARS,
        dt            = DT,
        start         = START,
        rand_seed     = seed,
        verbose       = 0,
        interventions = [vx],
    )
    sim_vx.run()

    idx_50 = get_timepoint_index(sim_vx, 50)
    base_prev = float(sim_base.results['hpv_prevalence'].values[idx_50])
    vx_prev   = float(sim_vx.results['hpv_prevalence'].values[idx_50])
    base_cin  = float(sim_base.results['cin_prevalence'].values[idx_50])
    vx_cin    = float(sim_vx.results['cin_prevalence'].values[idx_50])

    prev_reduction = (1 - vx_prev / base_prev) * 100 if base_prev > 0 else 0
    cin_reduction  = (1 - vx_cin / base_cin) * 100 if base_cin > 0 else 0

    metrics = {
        'hpv_prev_yr50_base': base_prev,
        'hpv_prev_yr50_vx':   vx_prev,
        'cin_prev_yr50_base': base_cin,
        'cin_prev_yr50_vx':   vx_cin,
        'hpv_prev_reduction_pct': float(prev_reduction),
        'cin_prev_reduction_pct': float(cin_reduction),
    }

    # Cancer reduction
    base_cum = float(np.nansum(sim_base.results['cancer_incidence'].values))
    vx_cum   = float(np.nansum(sim_vx.results['cancer_incidence'].values))
    metrics['cum_cancers_base'] = base_cum
    metrics['cum_cancers_vx']   = vx_cum
    metrics['cancer_reduction_pct'] = (1 - vx_cum / base_cum) * 100 if base_cum > 0 else 0

    return metrics


def compute_stats(all_metrics):
    """Compute mean and 95% CI for each metric across seeds."""
    keys = all_metrics[0].keys()
    stats = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if not np.isnan(m[k])]
        if len(vals) == 0:
            stats[k] = {'mean': 0, 'ci_lo': 0, 'ci_hi': 0, 'std': 0, 'n': 0}
            continue
        arr = np.array(vals)
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        se   = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        ci_lo = mean - 1.96 * se
        ci_hi = mean + 1.96 * se
        stats[k] = {
            'mean':  mean,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'std':   std,
            'n':     len(arr),
            'values': [float(v) for v in arr],
        }
    return stats


def print_stats_table(name, stats):
    """Print a formatted comparison table."""
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    print(f"  {'Metric':<35} {'Mean':>10} {'95% CI':>22} {'Std':>8}")
    print(f"  {'-'*75}")
    for k, v in stats.items():
        if k == 'values':
            continue
        ci_str = f"[{v['ci_lo']:.6f}, {v['ci_hi']:.6f}]"
        print(f"  {k:<35} {v['mean']:>10.6f} {ci_str:>22} {v['std']:>8.6f}")


if __name__ == '__main__':
    print(f"Running Python hpvsim quantitative cross-validation")
    print(f"  N_AGENTS={N_AGENTS}, N_YEARS={N_YEARS}, DT={DT}, N_SEEDS={N_SEEDS}")
    print(f"  hpvsim version: {hpv.__version__}")

    # Scenario 1: HPV16 only
    print(f"\n--- Scenario 1: HPV16 natural history ---")
    s1_metrics = []
    for seed in range(1, N_SEEDS + 1):
        print(f"  Seed {seed}/{N_SEEDS}...", end='', flush=True)
        m = run_scenario1(seed)
        s1_metrics.append(m)
        print(f" HPV prev@50={m['hpv_prev_yr50']:.4f}")

    s1_stats = compute_stats(s1_metrics)
    print_stats_table("Scenario 1: HPV16 natural history", s1_stats)

    # Scenario 2: HPV16 + vaccination
    print(f"\n--- Scenario 2: HPV16 + vaccination ---")
    s2_metrics = []
    for seed in range(1, N_SEEDS + 1):
        print(f"  Seed {seed}/{N_SEEDS}...", end='', flush=True)
        m = run_scenario2(seed)
        s2_metrics.append(m)
        print(f" HPV prev reduction={m['hpv_prev_reduction_pct']:.1f}%")

    s2_stats = compute_stats(s2_metrics)
    print_stats_table("Scenario 2: HPV16 + vaccination", s2_stats)

    # Save results
    results = {
        'scenario1': s1_stats,
        'scenario2': s2_stats,
        'config': {
            'n_agents': N_AGENTS,
            'n_years': N_YEARS,
            'dt': DT,
            'n_seeds': N_SEEDS,
            'start': START,
            'sample_years': SAMPLE_YEARS,
        }
    }
    out_path = os.path.join(OUT_DIR, 'python_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
