"""
Compare Python and Julia FPsim cross-validation results.
Reads the JSON output from both quantitative_xval.py and quantitative_xval.jl.
Prints a comparison table showing whether means fall within each other's 95% CIs.
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PY_FILE = os.path.join(SCRIPT_DIR, 'xval_python_results.json')
JL_FILE = os.path.join(SCRIPT_DIR, 'xval_julia_results.json')


def load_results(path):
    with open(path) as f:
        return json.load(f)


def check_overlap(py_summary, jl_summary):
    """Check if means fall within each other's 95% CIs."""
    py_mean = py_summary['mean']
    py_lo, py_hi = py_summary['lo'], py_summary['hi']
    jl_mean = jl_summary['mean']
    jl_lo, jl_hi = jl_summary['lo'], jl_summary['hi']

    # Julia mean in Python CI?
    jl_in_py = py_lo <= jl_mean <= py_hi
    # Python mean in Julia CI?
    py_in_jl = jl_lo <= py_mean <= jl_hi
    # CIs overlap at all?
    ci_overlap = py_lo <= jl_hi and jl_lo <= py_hi

    return jl_in_py or py_in_jl, ci_overlap


def main():
    if not os.path.exists(PY_FILE):
        print(f"ERROR: Python results not found at {PY_FILE}")
        print("Run: python quantitative_xval.py")
        sys.exit(1)
    if not os.path.exists(JL_FILE):
        print(f"ERROR: Julia results not found at {JL_FILE}")
        print("Run: julia quantitative_xval.jl")
        sys.exit(1)

    py = load_results(PY_FILE)
    jl = load_results(JL_FILE)

    py_summary = py['summary']
    jl_summary = jl['summary']

    metrics = ['total_births', 'total_miscarriages', 'total_stillbirths', 'total_abortions',
               'total_mat_deaths', 'total_inf_deaths', 'final_pop', 'cpr',
               'birth_rate_per1000', 'mean_tfr', 'preg_rate_annual',
               'live_birth_ratio', 'miscarriage_ratio', 'stillbirth_ratio', 'abortion_ratio',
               'crude_birth_rate', 'births_per_year',
               'tfr_yr5', 'tfr_yr10', 'tfr_yr15', 'tfr_yr19']

    print('=' * 100)
    print('QUANTITATIVE CROSS-VALIDATION: Python fpsim vs Julia FPsim')
    print(f'Config: {py["config"]["n_seeds"]} seeds × {py["config"]["n_agents"]} agents, '
          f'{py["config"]["location"]}, {py["config"]["start"]}-{py["config"]["stop"]}')
    print('=' * 100)
    print(f'{"Metric":<25} {"Python mean ± 95%CI":<25} {"Julia mean ± 95%CI":<25} {"Match?":<10} {"Ratio":>8}')
    print('-' * 93)

    n_match = 0
    n_overlap = 0
    n_compared = 0

    for m in metrics:
        if m not in py_summary or m not in jl_summary:
            py_str = f'{py_summary[m]["mean"]:.3f} ± {py_summary[m]["ci95"]:.3f}' if m in py_summary else 'N/A'
            jl_str = f'{jl_summary[m]["mean"]:.3f} ± {jl_summary[m]["ci95"]:.3f}' if m in jl_summary else 'N/A'
            print(f'{m:<25} {py_str:<25} {jl_str:<25} {"—":<10}')
            continue

        ps = py_summary[m]
        js = jl_summary[m]
        n_compared += 1

        py_str = f'{ps["mean"]:.3f} ± {ps["ci95"]:.3f}'
        jl_str = f'{js["mean"]:.3f} ± {js["ci95"]:.3f}'

        match, overlap = check_overlap(ps, js)
        if match:
            status = '✓'
            n_match += 1
        elif overlap:
            status = '~'
            n_overlap += 1
        else:
            status = '✗'

        ratio = js["mean"] / ps["mean"] if ps["mean"] != 0 else float('inf')
        print(f'{m:<25} {py_str:<25} {jl_str:<25} {status:<10} {ratio:>8.3f}')

    print('-' * 93)
    print(f'Results: {n_match}/{n_compared} exact matches (mean in other CI), '
          f'{n_overlap} CI overlaps, {n_compared - n_match - n_overlap} mismatches')

    # Detailed annual births comparison
    print(f'\n{"="*80}')
    print('Annual Births Trajectory (averages across seeds)')
    print(f'{"="*80}')
    
    import numpy as np
    py_annual = np.array([r['annual_births'] for r in py['raw']])
    jl_annual = np.array([r['annual_births'] for r in jl['raw']])
    
    n_years = min(py_annual.shape[1], jl_annual.shape[1])
    print(f'{"Year":<8} {"Python":>10} {"Julia":>10} {"Ratio":>8}')
    print('-' * 36)
    for yr in range(n_years):
        py_mean = np.mean(py_annual[:, yr])
        jl_mean = np.mean(jl_annual[:, yr])
        ratio = jl_mean / py_mean if py_mean > 0 else float('inf')
        print(f'{2000 + yr:<8} {py_mean:>10.1f} {jl_mean:>10.1f} {ratio:>8.3f}')

    if n_compared - n_match - n_overlap == 0:
        print('\n✅ ALL METRICS PASS: Julia FPsim matches Python fpsim within statistical tolerance.')
    else:
        print(f'\n⚠️  {n_compared - n_match - n_overlap} metrics have non-overlapping CIs — investigate.')

    return n_compared - n_match - n_overlap == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
