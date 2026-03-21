#!/usr/bin/env python
"""
Quantitative cross-validation: Python rotasim vs Julia RotaABM.
Runs matching scenarios across 20 seeds, loads Julia results, prints comparison.

Usage:
    source .venv/bin/activate
    python extensions/RotaABM/test/quantitative_xval.py
"""

import json
import os
import sys
import time

import numpy as np
import starsim as ss
import rotasim as rs

os.environ["SCIRIS_BACKEND"] = "agg"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JULIA_RESULTS_FILE = os.path.join(SCRIPT_DIR, "quantitative_xval_julia.json")

N_SEEDS = 20
N_AGENTS = 10_000
DUR_YEARS = 5


# ── Helpers ────────────────────────────────────────────────────────────────

def attack_rate(disease, n_agents):
    return float(np.sum(disease.results.new_infections.values)) / n_agents


def peak_prev(disease):
    return float(np.max(disease.results.prevalence.values))


def mean_prev(disease):
    return float(np.mean(disease.results.prevalence.values))


def equilibrium_prev(disease, last_years=2, dt_days=1):
    prev = disease.results.prevalence.values
    n_steps = int(round(last_years * 365.25 / dt_days))
    n_steps = min(n_steps, len(prev))
    tail = prev[-n_steps:]
    return float(np.mean(tail))


def total_new_infections(sim):
    total = 0.0
    for d in sim.diseases.values():
        if hasattr(d, "G"):
            total += float(np.sum(d.results.new_infections.values))
    return total


def count_active_strains(sim):
    n = 0
    for d in sim.diseases.values():
        if hasattr(d, "G"):
            if float(np.sum(d.results.new_infections.values)) > 0:
                n += 1
    return n


def reassortment_count(sim):
    for c in sim.connectors.values():
        if hasattr(c, "results") and hasattr(c.results, "n_reassortments"):
            return float(np.sum(c.results.n_reassortments.values))
    return 0.0


def summarize(values):
    values = np.array(values, dtype=float)
    n = len(values)
    m = np.mean(values)
    s = np.std(values, ddof=1) if n > 1 else 0.0
    se = s / np.sqrt(n)
    ci95 = 1.96 * se
    return {"mean": m, "std": s, "ci95": ci95, "lo": m - ci95, "hi": m + ci95}


def ci_overlap(py_summary, jl_summary):
    """Check if 95% CIs overlap"""
    return py_summary["lo"] <= jl_summary["hi"] and jl_summary["lo"] <= py_summary["hi"]


# ── Scenario 1: Single strain ─────────────────────────────────────────────

def run_scenario1(seed):
    sim = rs.Sim(
        scenario={"strains": {(1, 8): {"fitness": 1.0, "prevalence": 0.01}}, "default_fitness": 1.0},
        n_agents=N_AGENTS,
        start="2000-01-01",
        stop=f"{2000 + DUR_YEARS}-01-01",
        dt=ss.days(1),
        rand_seed=seed,
        verbose=0,
    )
    sim.run()

    d = sim.diseases["G1P8"]
    return {
        "attack_rate": attack_rate(d, N_AGENTS),
        "peak_prev": peak_prev(d),
        "mean_prev": mean_prev(d),
        "equil_prev": equilibrium_prev(d, last_years=2),
    }


# ── Scenario 2: Multi-strain ──────────────────────────────────────────────

def run_scenario2(seed):
    sim = rs.Sim(
        scenario="simple",
        n_agents=N_AGENTS,
        start="2000-01-01",
        stop=f"{2000 + DUR_YEARS}-01-01",
        dt=ss.days(1),
        rand_seed=seed,
        verbose=0,
    )
    sim.run()

    # Total prevalence
    all_prev = []
    for d in sim.diseases.values():
        if hasattr(d, "G"):
            all_prev.append(d.results.prevalence.values)
    total_prev = np.sum(all_prev, axis=0)

    total_peak = float(np.max(total_prev))
    total_mean = float(np.mean(total_prev))
    n_last = int(round(2.0 * 365.25))
    n_last = min(n_last, len(total_prev))
    total_equil = float(np.mean(total_prev[-n_last:]))

    result = {
        "total_attack_rate": total_new_infections(sim) / N_AGENTS,
        "total_peak_prev": total_peak,
        "total_mean_prev": total_mean,
        "total_equil_prev": total_equil,
        "active_strains": float(count_active_strains(sim)),
        "reassortments": float(reassortment_count(sim)),
    }

    # Per-strain
    for d in sim.diseases.values():
        if hasattr(d, "G"):
            nm = f"G{d.G}P{d.P}"
            result[f"{nm}_attack_rate"] = attack_rate(d, N_AGENTS)
            result[f"{nm}_peak_prev"] = peak_prev(d)

    return result


# ── Print comparison table ─────────────────────────────────────────────────

def print_comparison(scenario_name, py_summary, jl_summary):
    print(f"\n{'=' * 90}")
    print(f"  {scenario_name}")
    print(f"{'=' * 90}")
    header = f"{'Metric':<25} | {'Python mean ± 95%CI':>25} | {'Julia mean ± 95%CI':>25} | Match?"
    print(header)
    print("-" * 90)

    all_keys = sorted(set(list(py_summary.keys()) + list(jl_summary.keys())))
    n_match = 0
    n_total = 0

    for k in all_keys:
        py = py_summary.get(k)
        jl = jl_summary.get(k)
        if py is None or jl is None:
            continue

        py_str = f"{py['mean']:8.4f} ± {py['ci95']:.4f}"
        jl_str = f"{jl['mean']:8.4f} ± {jl['ci95']:.4f}"
        match = ci_overlap(py, jl)
        n_total += 1
        if match:
            n_match += 1
        status = "  ✓" if match else "  ✗"
        print(f"{k:<25} | {py_str:>25} | {jl_str:>25} | {status}")

    print("-" * 90)
    print(f"  Matched: {n_match}/{n_total} metrics (95% CIs overlap)")
    return n_match, n_total


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Python rotasim — Quantitative Cross-Validation")
    print(f"N_SEEDS={N_SEEDS}  N_AGENTS={N_AGENTS}  DUR={DUR_YEARS} years")
    print("=" * 60)

    seeds = list(range(1, N_SEEDS + 1))

    # Scenario 1
    print("\nRunning Scenario 1: Single strain G1P8...")
    t0 = time.time()
    s1_results = [run_scenario1(s) for s in seeds]
    print(f"  Done in {time.time() - t0:.1f}s")

    s1_keys = sorted(s1_results[0].keys())
    s1_summary = {}
    for k in s1_keys:
        vals = [r[k] for r in s1_results]
        s1_summary[k] = summarize(vals)

    # Scenario 2
    print("\nRunning Scenario 2: Multi-strain (simple)...")
    t0 = time.time()
    s2_results = [run_scenario2(s) for s in seeds]
    print(f"  Done in {time.time() - t0:.1f}s")

    s2_all_keys = set()
    for r in s2_results:
        s2_all_keys |= set(r.keys())
    s2_keys = sorted(s2_all_keys)
    s2_summary = {}
    for k in s2_keys:
        vals = [r.get(k, np.nan) for r in s2_results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            s2_summary[k] = summarize(vals)

    # Save Python results
    py_results = {
        "n_seeds": N_SEEDS,
        "n_agents": N_AGENTS,
        "dur_years": DUR_YEARS,
        "scenario1": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in s1_summary.items()},
        "scenario2": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in s2_summary.items()},
    }
    py_results_file = os.path.join(SCRIPT_DIR, "quantitative_xval_python.json")
    with open(py_results_file, "w") as f:
        json.dump(py_results, f, indent=2)
    print(f"\nPython results saved to: {py_results_file}")

    # Load Julia results
    if os.path.exists(JULIA_RESULTS_FILE):
        with open(JULIA_RESULTS_FILE) as f:
            jl_data = json.load(f)
        jl_s1 = jl_data.get("scenario1", {})
        jl_s2 = jl_data.get("scenario2", {})

        total_match = 0
        total_metrics = 0

        m, t = print_comparison("Scenario 1: Single Strain SIRS", s1_summary, jl_s1)
        total_match += m
        total_metrics += t

        m, t = print_comparison("Scenario 2: Multi-Strain", s2_summary, jl_s2)
        total_match += m
        total_metrics += t

        print(f"\n{'=' * 90}")
        print(f"  OVERALL: {total_match}/{total_metrics} metrics matched (95% CIs overlap)")
        if total_match == total_metrics:
            print("  ✓ PASS — Julia and Python are in quantitative agreement")
        else:
            print("  ✗ FAIL — Some metrics do not overlap")
        print(f"{'=' * 90}")
    else:
        print(f"\nJulia results not found at: {JULIA_RESULTS_FILE}")
        print("Run quantitative_xval.jl first, then re-run this script.")

        # Print Python-only summary
        print(f"\nScenario 1 (Python only):")
        for k in s1_keys:
            s = s1_summary[k]
            print(f"  {k:<25}: {s['mean']:.4f} ± {s['ci95']:.4f}")
        print(f"\nScenario 2 (Python only):")
        for k in s2_keys:
            if k in s2_summary:
                s = s2_summary[k]
                print(f"  {k:<25}: {s['mean']:.4f} ± {s['ci95']:.4f}")


if __name__ == "__main__":
    main()
