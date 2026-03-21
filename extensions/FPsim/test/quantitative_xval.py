"""
Quantitative cross-validation: Python fpsim baseline.
Runs N_SEEDS simulations with Kenya parameters and saves summary statistics.
"""

import sys
import os
import json
import numpy as np

# Activate venv
venv_activate = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.venv', 'bin', 'activate_this.py')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'fpsim'))

import fpsim as fp
import starsim as ss

# Configuration — must match Julia exactly
N_SEEDS    = 20
N_AGENTS   = 5000
START_YEAR = 2000
STOP_YEAR  = 2020
LOCATION   = 'kenya'
DT         = 1/12

# Output file
OUTFILE = os.path.join(os.path.dirname(__file__), 'xval_python_results.json')


def compute_annual_births(births_per_step, mpy=12):
    """Sum births in 12-month windows."""
    n = len(births_per_step)
    annual = []
    for start in range(0, n - mpy + 1, mpy):
        annual.append(float(np.sum(births_per_step[start:start + mpy])))
    return annual


def compute_tfr_from_births(sim, year_idx):
    """
    Compute TFR for a specific annual window (year_idx-th year).
    TFR = sum over 5-year age groups of (births_in_group / women_in_group) × 5
    """
    ti_start = year_idx * 12
    ti_end = min(ti_start + 12, sim.connectors.fp.t.npts)
    
    ppl = sim.people
    fp_mod = sim.connectors.fp
    
    # Count births by mother age in this year
    age_bins = np.array([15, 20, 25, 30, 35, 40, 45, 50])
    births_by_bin = np.zeros(len(age_bins) - 1)
    women_by_bin = np.zeros(len(age_bins) - 1)
    
    # Average over timesteps in this year
    for ti in range(ti_start, ti_end):
        # Count women by age bin at this timestep
        ages = ppl.age[:].copy()
        female = ppl.female[:].copy()
        for b in range(len(age_bins) - 1):
            mask = female & (ages >= age_bins[b]) & (ages < age_bins[b+1])
            women_by_bin[b] += np.sum(mask)
    
    women_by_bin /= max(ti_end - ti_start, 1)  # average women count

    # Use ASFR from results if available
    asfr_data = sim.connectors.fp.asfr
    if asfr_data is not None and asfr_data.shape[1] >= ti_end:
        # ASFR bins in Python: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100]
        # We want bins 3-9 (ages 15-50)
        tfr = 0.0
        for b in range(3, 10):  # indices for 15-20, 20-25, ..., 45-50
            asfr_avg = np.mean(asfr_data[b, ti_start:ti_end])
            tfr += asfr_avg * 5 / 1000  # 5-year width, per 1000
        return tfr
    return np.nan


def run_one_seed(seed):
    """Run one simulation and return summary metrics."""
    sim = fp.Sim(
        n_agents=N_AGENTS,
        location=LOCATION,
        start=START_YEAR,
        stop=STOP_YEAR,
        rand_seed=seed,
        verbose=0,
    )
    sim.run()
    
    res = sim.connectors.fp.results
    ppl = sim.people
    npts = sim.connectors.fp.t.npts
    
    # Annual births
    births = np.array(res.births[:npts], dtype=float)
    annual_births = compute_annual_births(births)
    
    # Cumulative births
    cum_births = np.cumsum(births)
    
    # Total events
    total_births = float(np.sum(births))
    total_miscarriages = float(np.sum(res.miscarriages[:npts]))
    total_stillbirths = float(np.sum(res.stillbirths[:npts]))
    total_abortions = float(np.sum(res.abortions[:npts]))
    total_mat_deaths = float(np.sum(res.maternal_deaths[:npts]))
    total_inf_deaths = float(np.sum(res.infant_deaths[:npts]))
    
    # Population at various years
    final_pop = len(ppl)
    
    # TFR at various years using ASFR
    tfr_values = {}
    for yr in [5, 10, 15, 19]:
        tfr = compute_tfr_from_births(sim, yr)
        tfr_values[f'tfr_yr{yr}'] = float(tfr) if not np.isnan(tfr) else None
    
    # Mean TFR over stable period (year 2 to end, skip burn-in)
    tfr_stable = []
    for yr in range(2, STOP_YEAR - START_YEAR):
        t = compute_tfr_from_births(sim, yr)
        if not np.isnan(t):
            tfr_stable.append(t)
    mean_tfr = float(np.mean(tfr_stable)) if tfr_stable else None
    
    # CPR (using method != 0 among 15-49 women)
    fp_mod = sim.connectors.fp
    female_15_49 = ppl.female & (ppl.age >= 15) & (ppl.age < 50)
    n_women_15_49 = float(np.sum(female_15_49))
    n_on_contra = float(np.sum(fp_mod.on_contra[female_15_49.uids]))
    cpr = n_on_contra / n_women_15_49 if n_women_15_49 > 0 else 0.0
    
    # Birth rate per 1000 women-years (last 5 years)
    last_5yr_births = float(np.sum(births[-60:]))
    # Average women 15-49 in last year
    n_fecund_avg = float(np.mean(res.n_fecund[-12:]))
    birth_rate = (last_5yr_births / 5.0) / n_fecund_avg * 1000 if n_fecund_avg > 0 else 0
    
    # Pregnancy rate per step (average over stable period)
    pregnancies = np.array(res.pregnancies[:npts], dtype=float)
    preg_rate_annual = float(np.sum(pregnancies[24:])) / max(npts - 24, 1) * 12
    
    # Per-pregnancy outcome ratios (architecture-independent)
    total_conceptions = total_births + total_miscarriages + total_stillbirths + total_abortions
    live_birth_ratio = total_births / total_conceptions if total_conceptions > 0 else 0
    miscarriage_ratio = total_miscarriages / total_conceptions if total_conceptions > 0 else 0
    stillbirth_ratio = total_stillbirths / total_conceptions if total_conceptions > 0 else 0
    abortion_ratio = total_abortions / total_conceptions if total_conceptions > 0 else 0
    
    # Annual births per 1000 population (crude birth rate)
    crude_birth_rate = (total_births / 20.0) / (N_AGENTS) * 1000
    
    # Births per year (average over stable period years 3-20)
    births_per_year = float(np.sum(births[24:])) / max((STOP_YEAR - START_YEAR - 2), 1)
    
    return {
        'seed': seed,
        'total_births': total_births,
        'total_miscarriages': total_miscarriages,
        'total_stillbirths': total_stillbirths,
        'total_abortions': total_abortions,
        'total_mat_deaths': total_mat_deaths,
        'total_inf_deaths': total_inf_deaths,
        'annual_births': annual_births,
        'final_pop': final_pop,
        'cpr': cpr,
        'birth_rate_per1000': birth_rate,
        'mean_tfr': mean_tfr,
        'preg_rate_annual': preg_rate_annual,
        'live_birth_ratio': live_birth_ratio,
        'miscarriage_ratio': miscarriage_ratio,
        'stillbirth_ratio': stillbirth_ratio,
        'abortion_ratio': abortion_ratio,
        'crude_birth_rate': crude_birth_rate,
        'births_per_year': births_per_year,
        **tfr_values,
    }


def main():
    print(f'Running Python fpsim cross-validation: {N_SEEDS} seeds × {N_AGENTS} agents')
    print(f'Location: {LOCATION}, Period: {START_YEAR}-{STOP_YEAR}')
    
    all_results = []
    for i, seed in enumerate(range(1, N_SEEDS + 1)):
        print(f'  Seed {seed}/{N_SEEDS}...', end=' ', flush=True)
        result = run_one_seed(seed)
        all_results.append(result)
        print(f'births={result["total_births"]:.0f}, TFR={result["mean_tfr"]:.2f}' if result["mean_tfr"] else f'births={result["total_births"]:.0f}')
    
    # Compute summary statistics
    metrics = ['total_births', 'total_miscarriages', 'total_stillbirths', 'total_abortions',
               'total_mat_deaths', 'total_inf_deaths', 'final_pop', 'cpr',
               'birth_rate_per1000', 'mean_tfr', 'preg_rate_annual',
               'live_birth_ratio', 'miscarriage_ratio', 'stillbirth_ratio', 'abortion_ratio',
               'crude_birth_rate', 'births_per_year',
               'tfr_yr5', 'tfr_yr10', 'tfr_yr15', 'tfr_yr19']
    
    summary = {}
    for m in metrics:
        vals = [r[m] for r in all_results if r.get(m) is not None]
        if vals:
            vals = np.array(vals, dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            ci95 = 1.96 * std / np.sqrt(len(vals))
            summary[m] = {'mean': mean, 'std': std, 'ci95': ci95, 'n': len(vals),
                          'lo': mean - ci95, 'hi': mean + ci95}
    
    output = {
        'config': {
            'n_seeds': N_SEEDS, 'n_agents': N_AGENTS, 'start': START_YEAR,
            'stop': STOP_YEAR, 'location': LOCATION, 'dt': DT,
        },
        'summary': summary,
        'raw': all_results,
    }
    
    with open(OUTFILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print table
    print(f'\n{"="*80}')
    print(f'Python fpsim Cross-Validation Summary ({N_SEEDS} seeds)')
    print(f'{"="*80}')
    print(f'{"Metric":<25} {"Mean":>10} {"± 95%CI":>10} {"Std":>10}')
    print(f'{"-"*55}')
    for m in metrics:
        if m in summary:
            s = summary[m]
            print(f'{m:<25} {s["mean"]:>10.3f} {s["ci95"]:>10.3f} {s["std"]:>10.3f}')
    
    print(f'\nResults saved to {OUTFILE}')
    return output


if __name__ == '__main__':
    main()
