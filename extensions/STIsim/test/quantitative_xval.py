#!/usr/bin/env python
"""
Quantitative cross-validation: Python stisim reference runs.

Runs HIV and Syphilis scenarios across 20 seeds and outputs JSON with
per-seed time series for comparison with the Julia companion script.

Usage:
    source /Users/sdwfrost/Projects/starsim/code/Starsim.jl/.venv/bin/activate
    python quantitative_xval.py
"""

import json
import sys
import io
import numpy as np

import starsim as ss
import stisim as sti


class _suppress_stdout:
    """Context manager that silences stdout (stisim prints init messages)."""
    def __enter__(self):
        self._orig = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._orig


# ── Shared constants (must match Julia script) ───────────────────────────────
N_AGENTS = 5000
N_SEEDS = 20
SEEDS = list(range(1, N_SEEDS + 1))

# ── Explicit parameters for both languages ───────────────────────────────────
# HIV parameters — Python defaults
HIV_PARS = dict(
    init_prev=ss.bernoulli(0.05),
    beta_m2f=0.05,
)

# Syphilis parameters — Python defaults, with init_prev set explicitly
SYPH_PARS = dict(
    init_prev=ss.bernoulli(0.05),
    beta_m2f=0.1,
)


def timestep_to_year(ti, start, dt):
    """Convert timestep index to year offset from start."""
    return ti * dt


def get_prevalence_at_years(prevalence_array, start, dur, target_years):
    """Extract prevalence values at specific year offsets."""
    n = len(prevalence_array)
    dt = dur / (n - 1) if n > 1 else 1.0
    result = {}
    for yr in target_years:
        idx = min(int(round(yr / dt)), n - 1)
        result[f"prev_yr{yr}"] = float(prevalence_array[idx])
    return result


# ── Scenario 1: HIV basic ────────────────────────────────────────────────────

def run_hiv_basic(seed):
    with _suppress_stdout():
        sim = sti.Sim(
            diseases=sti.HIV(**HIV_PARS),
            n_agents=N_AGENTS,
            start=2000,
            dur=40,
            rand_seed=seed,
        )
        sim.run(verbose=0)

    r = sim.results.hiv
    prev = np.asarray(r.prevalence, dtype=float)
    n_inf = np.asarray(r.n_infected, dtype=float)
    new_inf = np.asarray(r.new_infections, dtype=float)

    # Get prevalence at specific years
    yr_metrics = get_prevalence_at_years(prev, 2000, 40, [10, 20, 30, 40])

    result = dict(
        prevalence=prev.tolist(),
        final_prevalence=float(prev[-1]),
        cum_infections=float(np.sum(new_inf)),
        n_infected_final=float(n_inf[-1]),
        **yr_metrics,
    )

    # Deaths if available
    if hasattr(r, 'new_deaths'):
        deaths = np.asarray(r.new_deaths, dtype=float)
        result['cum_deaths'] = float(np.sum(deaths))
    else:
        result['cum_deaths'] = 0.0

    return result


# ── Scenario 2: Syphilis basic ──────────────────────────────────────────────

def run_syphilis_basic(seed):
    with _suppress_stdout():
        sim = sti.Sim(
            diseases=sti.Syphilis(**SYPH_PARS),
            n_agents=N_AGENTS,
            start=2000,
            dur=20,
            rand_seed=seed,
        )
        sim.run(verbose=0)

    r = sim.results.syph
    prev = np.asarray(r.prevalence, dtype=float)
    n_inf = np.asarray(r.n_infected, dtype=float)

    yr_metrics = get_prevalence_at_years(prev, 2000, 20, [5, 10, 15, 20])

    result = dict(
        prevalence=prev.tolist(),
        final_prevalence=float(prev[-1]),
        n_infected_final=float(n_inf[-1]),
        n_primary=float(np.asarray(r.n_primary, dtype=float)[-1]),
        n_secondary=float(np.asarray(r.n_secondary, dtype=float)[-1]),
        n_latent=float(np.asarray(r.n_latent, dtype=float)[-1]),
        n_tertiary=float(np.asarray(r.n_tertiary, dtype=float)[-1]),
        **yr_metrics,
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print("=" * 72, file=sys.stderr)
    print(f"Quantitative cross-validation: Python stisim ({N_SEEDS} seeds)", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    # Scenario 1: HIV
    print("\n[1/2] HIV basic (40 years) ...", file=sys.stderr)
    hiv_results = []
    for seed in SEEDS:
        print(f"  seed={seed}", end="", file=sys.stderr, flush=True)
        hiv_results.append(run_hiv_basic(seed))
    print(file=sys.stderr)
    results["hiv_basic"] = hiv_results

    # Scenario 2: Syphilis
    print("\n[2/2] Syphilis basic (20 years) ...", file=sys.stderr)
    syph_results = []
    for seed in SEEDS:
        print(f"  seed={seed}", end="", file=sys.stderr, flush=True)
        syph_results.append(run_syphilis_basic(seed))
    print(file=sys.stderr)
    results["syphilis_basic"] = syph_results

    # Summary
    print("\n" + "=" * 72, file=sys.stderr)
    print("Python summary:", file=sys.stderr)
    for metric in ["prev_yr10", "prev_yr20", "prev_yr30", "prev_yr40", "cum_infections", "cum_deaths"]:
        vals = [r.get(metric, 0) for r in hiv_results]
        a = np.array(vals)
        print(f"  HIV {metric}: {a.mean():.4f} ± {a.std():.4f}", file=sys.stderr)

    for metric in ["prev_yr5", "prev_yr10", "prev_yr15", "prev_yr20"]:
        vals = [r.get(metric, 0) for r in syph_results]
        a = np.array(vals)
        print(f"  Syph {metric}: {a.mean():.4f} ± {a.std():.4f}", file=sys.stderr)

    # JSON to stdout
    json.dump(results, sys.stdout, indent=2)
    print()
    print("\nDone.", file=sys.stderr)


if __name__ == "__main__":
    main()
