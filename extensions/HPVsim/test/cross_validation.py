#!/usr/bin/env python
"""
Cross-validation of Python hpvsim against Julia HPVsim.

Runs four scenarios with Python hpvsim and writes summary CSV files
that the Julia cross_validation.jl script reads for comparison.

Scenarios:
  1. Single genotype HPV16 (basic)
  2. Multi-genotype HPV16 + HPV18
  3. HPV16+18 with prophylactic vaccination
  4. HPV16 with screening programme

Usage:
  python cross_validation.py
"""

import os
import json
import numpy as np
import hpvsim as hpv

OUT_DIR = os.path.join(os.path.dirname(__file__), "cross_validation_results")
os.makedirs(OUT_DIR, exist_ok=True)

N_AGENTS  = 5_000
N_YEARS   = 50
DT        = 0.25
START     = 2000
SEED      = 42


def extract_results(sim, genotype_idx=None):
    """Pull key metrics from a completed sim."""
    years = sim.results['year']
    out = dict(
        year       = years.tolist(),
        hpv_prev   = sim.results['hpv_prevalence'].values.tolist(),
        cin_prev   = sim.results['cin_prevalence'].values.tolist(),
        cancer_inc = sim.results['cancer_incidence'].values.tolist(),
    )

    # Genotype-specific prevalence (shape: n_genotypes × npts)
    bg_prev = sim.results['hpv_prevalence_by_genotype'].values
    bg_cin  = sim.results['cin_prevalence_by_genotype'].values
    if genotype_idx is not None:
        for label, idx in genotype_idx.items():
            out[f'hpv_prev_{label}'] = bg_prev[idx].tolist()
            out[f'cin_prev_{label}'] = bg_cin[idx].tolist()

    # Intervention trackers (if present)
    for key in ['cum_doses', 'cum_screened', 'cum_cin_treated',
                'new_screened', 'n_cin_treated']:
        r = sim.results.get(key)
        if r is not None:
            vals = r.values if hasattr(r, 'values') else r
            if hasattr(vals, 'tolist'):
                out[key] = vals.tolist()

    return out


def save(name, data):
    """Persist results as JSON for the Julia side to read."""
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


# ── Scenario 1: Single genotype HPV16 ──────────────────────────────────────

def scenario1():
    print("\n╔══ Scenario 1: Single genotype HPV16 ══╗")
    sim = hpv.Sim(
        n_agents  = N_AGENTS,
        genotypes = [16],
        n_years   = N_YEARS,
        dt        = DT,
        start     = START,
        rand_seed = SEED,
        verbose   = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia (no multiscale implementation)
    )
    sim.run()

    res = extract_results(sim)
    res['scenario'] = 'single_hpv16'

    print(f"  HPV prevalence: {res['hpv_prev'][0]:.4f} → {res['hpv_prev'][-1]:.4f}")
    print(f"  CIN prevalence: {res['cin_prev'][0]:.6f} → {res['cin_prev'][-1]:.6f}")
    print(f"  Cancer incidence (final): {res['cancer_inc'][-1]:.4f} per 100k")

    save('scenario1_python', res)
    return res


# ── Scenario 2: Multi-genotype HPV16 + HPV18 ──────────────────────────────

def scenario2():
    print("\n╔══ Scenario 2: Multi-genotype HPV16+18 ══╗")
    sim = hpv.Sim(
        n_agents  = N_AGENTS,
        genotypes = [16, 18],
        n_years   = N_YEARS,
        dt        = DT,
        start     = START,
        rand_seed = SEED,
        verbose   = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia
    )
    sim.run()

    gidx = {k: v for v, k in enumerate(sim['genotype_map'].values())}
    res = extract_results(sim, genotype_idx=gidx)
    res['scenario'] = 'multi_hpv16_18'

    print(f"  Total HPV prevalence (final): {res['hpv_prev'][-1]:.4f}")
    for label in gidx:
        print(f"    {label} prevalence (final): {res[f'hpv_prev_{label}'][-1]:.4f}")
    print(f"  CIN prevalence (final): {res['cin_prev'][-1]:.6f}")

    save('scenario2_python', res)
    return res


# ── Scenario 3: Vaccination ───────────────────────────────────────────────

def scenario3():
    print("\n╔══ Scenario 3: HPV16+18 with vaccination ══╗")

    # Baseline (no vaccination)
    sim_base = hpv.Sim(
        n_agents  = N_AGENTS,
        genotypes = [16, 18],
        n_years   = N_YEARS,
        dt        = DT,
        start     = START,
        rand_seed = SEED,
        verbose   = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia
    )
    sim_base.run()
    res_base = extract_results(sim_base)

    # With vaccination starting year 20 (= 2020)
    vx = hpv.routine_vx(
        prob       = 0.9,
        product    = 'bivalent',
        age_range  = [9, 14],
        start_year = START + 20,
    )
    sim_vx = hpv.Sim(
        n_agents      = N_AGENTS,
        genotypes     = [16, 18],
        n_years       = N_YEARS,
        dt            = DT,
        start         = START,
        rand_seed     = SEED,
        verbose       = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia
        interventions = [vx],
    )
    sim_vx.run()

    gidx = {k: v for v, k in enumerate(sim_vx['genotype_map'].values())}
    res_vx = extract_results(sim_vx, genotype_idx=gidx)
    res_vx['scenario'] = 'vaccination'

    # Prevalence reduction
    base_final = res_base['hpv_prev'][-1]
    vx_final   = res_vx['hpv_prev'][-1]
    reduction  = (1 - vx_final / base_final) * 100 if base_final > 0 else 0
    res_vx['baseline_hpv_prev'] = res_base['hpv_prev']
    res_vx['baseline_cin_prev'] = res_base['cin_prev']
    res_vx['baseline_cancer_inc'] = res_base['cancer_inc']
    res_vx['hpv_prev_reduction_pct'] = reduction

    print(f"  Baseline HPV prev (final): {base_final:.4f}")
    print(f"  Vaccinated HPV prev (final): {vx_final:.4f}")
    print(f"  Prevalence reduction: {reduction:.1f}%")
    print(f"  cum_doses: {res_vx.get('cum_doses', [0])[-1]:.0f}")

    save('scenario3_python', res_vx)
    return res_vx


# ── Scenario 4: Screening ─────────────────────────────────────────────────

def scenario4():
    print("\n╔══ Scenario 4: HPV16 with screening ══╗")

    # Baseline (no screening)
    sim_base = hpv.Sim(
        n_agents  = N_AGENTS,
        genotypes = [16],
        n_years   = N_YEARS,
        dt        = DT,
        start     = START,
        rand_seed = SEED,
        verbose   = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia
    )
    sim_base.run()
    res_base = extract_results(sim_base)

    # With screening starting year 20 (= 2020)
    screen_eligible = lambda sim: (
        np.isnan(sim.people.date_screened) |
        (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    )
    screen = hpv.routine_screening(
        product     = 'hpv',
        prob        = 0.10,
        eligibility = screen_eligible,
        age_range   = [25, 65],
        start_year  = START + 20,
        label       = 'screening',
    )
    screen_positive = lambda sim: sim.get_intervention('screening').outcomes['positive']
    assign_tx = hpv.routine_triage(
        prob        = 1.0,
        product     = 'tx_assigner',
        eligibility = screen_positive,
        label       = 'assign_tx',
    )
    to_ablate = lambda sim: sim.get_intervention('assign_tx').outcomes['ablation']
    ablation = hpv.treat_num(
        prob        = 0.9,
        product     = 'ablation',
        eligibility = to_ablate,
        label       = 'ablation',
    )
    to_excise = lambda sim: sim.get_intervention('assign_tx').outcomes['excision']
    excision = hpv.treat_num(
        prob        = 0.9,
        product     = 'excision',
        eligibility = to_excise,
        label       = 'excision',
    )

    sim_scr = hpv.Sim(
        n_agents      = N_AGENTS,
        genotypes     = [16],
        n_years       = N_YEARS,
        dt            = DT,
        start         = START,
        rand_seed     = SEED,
        verbose       = 0,
        ms_agent_ratio = 1,  # Disable multiscale to match Julia
        interventions = [screen, assign_tx, ablation, excision],
    )
    sim_scr.run()

    res_scr = extract_results(sim_scr)
    res_scr['scenario'] = 'screening'
    res_scr['baseline_hpv_prev']  = res_base['hpv_prev']
    res_scr['baseline_cin_prev']  = res_base['cin_prev']
    res_scr['baseline_cancer_inc'] = res_base['cancer_inc']

    base_final = res_base['cin_prev'][-1]
    scr_final  = res_scr['cin_prev'][-1]
    cin_red    = (1 - scr_final / base_final) * 100 if base_final > 0 else 0
    res_scr['cin_prev_reduction_pct'] = cin_red

    print(f"  Baseline CIN prev (final): {base_final:.6f}")
    print(f"  Screened CIN prev (final): {scr_final:.6f}")
    print(f"  CIN prevalence reduction: {cin_red:.1f}%")

    save('scenario4_python', res_scr)
    return res_scr


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary(r1, r2, r3, r4):
    print("\n" + "=" * 72)
    print("  PYTHON HPVSIM — CROSS-VALIDATION SUMMARY")
    print("=" * 72)
    header = f"{'Metric':<40} {'Value':>14}"
    print(header)
    print("-" * 72)

    def row(label, val, fmt=".4f"):
        print(f"  {label:<38} {val:{fmt}}")

    print("[Scenario 1] Single HPV16")
    row("HPV prevalence (year 10)",  r1['hpv_prev'][10])
    row("HPV prevalence (year 25)",  r1['hpv_prev'][25])
    row("HPV prevalence (final)",    r1['hpv_prev'][-1])
    row("CIN prevalence (final)",    r1['cin_prev'][-1], ".6f")
    row("Cancer inc (final, /100k)", r1['cancer_inc'][-1], ".2f")

    print("\n[Scenario 2] Multi-genotype HPV16+18")
    row("Total HPV prevalence (final)", r2['hpv_prev'][-1])
    row("HPV16 prevalence (final)",     r2['hpv_prev_hpv16'][-1])
    row("HPV18 prevalence (final)",     r2['hpv_prev_hpv18'][-1])
    row("CIN prevalence (final)",       r2['cin_prev'][-1], ".6f")

    print("\n[Scenario 3] Vaccination")
    row("Baseline HPV prev (final)",    r3['baseline_hpv_prev'][-1])
    row("Vaccinated HPV prev (final)",  r3['hpv_prev'][-1])
    row("HPV prevalence reduction (%)", r3['hpv_prev_reduction_pct'], ".1f")

    print("\n[Scenario 4] Screening")
    row("Baseline CIN prev (final)",    r4['baseline_cin_prev'][-1], ".6f")
    row("Screened CIN prev (final)",    r4['cin_prev'][-1], ".6f")
    row("CIN prevalence reduction (%)", r4['cin_prev_reduction_pct'], ".1f")

    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    r1 = scenario1()
    r2 = scenario2()
    r3 = scenario3()
    r4 = scenario4()
    print_summary(r1, r2, r3, r4)
    print(f"\nResults saved in {OUT_DIR}/")
