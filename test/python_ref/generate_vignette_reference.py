"""
Generate reference data from Python starsim v3.2.2 for vignette cross-validation.
Each scenario matches the corresponding Julia vignette with identical parameters.

Usage:
    .venv/bin/python generate_vignette_reference.py
"""
import json
import os
import numpy as np
import starsim as ss

os.environ['STARSIM_INSTALL_FONTS'] = '0'


def run_01_introduction():
    """Vignette 01: Basic SIR."""
    sim = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim.run()
    prev = sim.results.sir.prevalence.values
    n_rec = sim.results.sir.n_recovered.values
    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        final_prevalence=float(prev[-1]),
        attack_rate=float(n_rec[-1] / 5000),
        final_susceptible=float(sim.results.sir.n_susceptible.values[-1]),
        final_recovered=float(n_rec[-1]),
        prevalence=prev.tolist(),
    )


def run_02_building_a_model():
    """Vignette 02: SIR with Poisson contacts, single network."""
    sim = ss.Sim(
        n_agents=2000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.1, dur_inf=10, init_prev=0.02),
        dt=1.0, start=0, stop=180,
        rand_seed=42, verbose=0,
    )
    sim.run()
    prev = sim.results.sir.prevalence.values
    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        final_prevalence=float(prev[-1]),
        attack_rate=float(sim.results.sir.n_recovered.values[-1] / 2000),
        prevalence=prev.tolist(),
    )


def run_03_demographics():
    """Vignette 03: SIR with births/deaths."""
    # No demographics baseline
    sim_base = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_base.run()

    # With demographics
    sim_demo = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        demographics=[ss.Births(birth_rate=20), ss.Deaths(death_rate=15)],
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_demo.run()

    # Growth scenario (short duration to avoid population explosion)
    sim_grow = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        demographics=[ss.Births(birth_rate=30), ss.Deaths(death_rate=10)],
        dt=1.0, start=0, stop=50,
        rand_seed=42, verbose=0,
    )
    sim_grow.run()

    # Decline scenario (short duration)
    sim_decline = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        demographics=[ss.Births(birth_rate=10), ss.Deaths(death_rate=30)],
        dt=1.0, start=0, stop=50,
        rand_seed=42, verbose=0,
    )
    sim_decline.run()

    return dict(
        base_peak_prevalence=float(max(sim_base.results.sir.prevalence.values)),
        demo_final_alive=float(sim_demo.results.n_alive.values[-1]),
        demo_total_births=float(sim_demo.results.births.new.values.sum()),
        demo_total_deaths=float(sim_demo.results.deaths.new.values.sum()),
        demo_peak_prevalence=float(max(sim_demo.results.sir.prevalence.values)),
        grow_final_alive=float(sim_grow.results.n_alive.values[-1]),
        decline_final_alive=float(sim_decline.results.n_alive.values[-1]),
    )


def run_04_diseases():
    """Vignette 04: Multiple disease types."""
    # SIR
    sim_sir = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_sir.run()

    # SIS
    sim_sis = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIS(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_sis.run()

    # Multi-disease
    sim_multi = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=[
            ss.SIR(name='flu', beta=0.08, dur_inf=7, init_prev=0.01),
            ss.SIS(name='cold', beta=0.05, dur_inf=5, init_prev=0.02),
        ],
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_multi.run()

    # Beta variation
    beta_peaks = {}
    for b in [0.02, 0.05, 0.10]:
        sim = ss.Sim(
            n_agents=5000,
            networks=ss.RandomNet(n_contacts=10),
            diseases=ss.SIR(beta=b, dur_inf=10, init_prev=0.01),
            dt=1.0, start=0, stop=365,
            rand_seed=42, verbose=0,
        )
        sim.run()
        beta_peaks[str(b)] = float(max(sim.results.sir.prevalence.values))

    return dict(
        sir_peak_prevalence=float(max(sim_sir.results.sir.prevalence.values)),
        sir_attack_rate=float(sim_sir.results.sir.n_recovered.values[-1] / 5000),
        sis_peak_prevalence=float(max(sim_sis.results.sis.prevalence.values)),
        sis_final_prevalence=float(sim_sis.results.sis.prevalence.values[-1]),
        multi_flu_peak=float(max(sim_multi.results.flu.prevalence.values)),
        multi_cold_peak=float(max(sim_multi.results.cold.prevalence.values)),
        multi_cold_final=float(sim_multi.results.cold.prevalence.values[-1]),
        beta_peaks=beta_peaks,
    )


def run_05_networks():
    """Vignette 05: Different network types."""
    results = {}
    for net_name, net_obj in [
        ('random', ss.RandomNet(n_contacts=10)),
        ('static', ss.StaticNet(n_contacts=10)),
        ('mixingpool', ss.MixingPool(n_contacts=10)),
    ]:
        sim = ss.Sim(
            n_agents=2000,
            networks=net_obj,
            diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
            dt=1.0, start=0, stop=200,
            rand_seed=42, verbose=0,
        )
        sim.run()
        prev = sim.results.sir.prevalence.values
        results[net_name] = dict(
            peak_prevalence=float(max(prev)),
            peak_day=int(np.argmax(prev)),
            attack_rate=float(sim.results.sir.n_recovered.values[-1] / 2000),
        )

    return results


def run_12_multisim():
    """Vignette 12: MultiSim mean prevalence."""
    base = ss.Sim(
        n_agents=3000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=200,
        rand_seed=42, verbose=0,
    )
    msim = ss.MultiSim(base, n_runs=5)
    msim.run(parallel=False)

    prevs = [s.results.sir.prevalence.values for s in msim.sims]
    mean_prev = np.mean(prevs, axis=0)
    individual_peaks = [float(max(p)) for p in prevs]

    return dict(
        mean_peak_prevalence=float(max(mean_prev)),
        mean_final_prevalence=float(mean_prev[-1]),
        individual_peaks=individual_peaks,
        mean_attack_rate=float(np.mean([
            s.results.sir.n_recovered.values[-1] / 3000 for s in msim.sims
        ])),
        n_runs=5,
    )


def run_13_calibration():
    """Vignette 13: Calibration — grid search recovers known beta."""
    true_sim = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=200,
        rand_seed=42, verbose=0,
    )
    true_sim.run()
    target = true_sim.results.sir.prevalence.values.copy()

    betas = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    losses = []
    for b in betas:
        sim = ss.Sim(
            n_agents=5000,
            networks=ss.RandomNet(n_contacts=10),
            diseases=ss.SIR(beta=b, dur_inf=10, init_prev=0.01),
            dt=1.0, start=0, stop=200,
            rand_seed=42, verbose=0,
        )
        sim.run()
        p = sim.results.sir.prevalence.values
        n = min(len(p), len(target))
        mse = float(np.mean((p[:n] - target[:n])**2))
        losses.append(mse)

    best_idx = int(np.argmin(losses))
    return dict(
        true_beta=0.05,
        betas=betas,
        losses=losses,
        best_beta=betas[best_idx],
        best_mse=losses[best_idx],
        target_peak_prevalence=float(max(target)),
        target_peak_day=int(np.argmax(target)),
    )


def run_06_interventions():
    """Vignette 06: SIR with routine vaccination vs baseline."""
    # Baseline (no intervention)
    sim_base = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_base.run()

    # With routine vaccination (efficacy=0.9, prob=0.01 per day)
    vaccine = ss.simple_vx(efficacy=0.9)
    routine = ss.routine_vx(product=vaccine, prob=0.01)
    sim_vx = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
        interventions=routine,
        dt=1.0, start=0, stop=365,
        rand_seed=42, verbose=0,
    )
    sim_vx.run()

    prev_base = sim_base.results.sir.prevalence.values
    prev_vx = sim_vx.results.sir.prevalence.values

    return dict(
        base_peak_prevalence=float(max(prev_base)),
        vx_peak_prevalence=float(max(prev_vx)),
        base_attack_rate=float(sim_base.results.sir.n_recovered.values[-1] / 5000),
        vx_attack_rate=float(sim_vx.results.sir.n_recovered.values[-1] / 5000),
        base_prevalence=prev_base.tolist(),
        vx_prevalence=prev_vx.tolist(),
    )


def run_07_connectors():
    """Vignette 07: SIS with seasonality connector vs constant beta."""
    # SIS baseline (constant beta, 2 years)
    sim_base = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIS(beta=0.05, dur_inf=10, init_prev=0.01),
        dt=1.0, start=0, stop=730,
        rand_seed=42, verbose=0,
    )
    sim_base.run()

    # SIS with seasonality (scale=0.5, shift=0 → peak at year start)
    sim_s = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=10),
        diseases=ss.SIS(beta=0.05, dur_inf=10, init_prev=0.01),
        connectors=ss.seasonality(diseases='sis', scale=0.5, shift=0.0),
        dt=1.0, start=0, stop=730,
        rand_seed=42, verbose=0,
    )
    sim_s.run()

    prev_base = sim_base.results.sis.prevalence.values
    prev_s = sim_s.results.sis.prevalence.values

    return dict(
        base_peak_prevalence=float(max(prev_base)),
        base_final_prevalence=float(prev_base[-1]),
        seasonal_peak_prevalence=float(max(prev_s)),
        seasonal_final_prevalence=float(prev_s[-1]),
        base_prevalence=prev_base.tolist(),
        seasonal_prevalence=prev_s.tolist(),
    )


def run_08_measles():
    """Vignette 08: Measles-like SIR outbreak (high beta, high contacts)."""
    sim = ss.Sim(
        n_agents=10000,
        networks=ss.RandomNet(n_contacts=15),
        diseases=ss.SIR(beta=0.3, dur_inf=19, init_prev=0.001),
        dt=1.0, start=0, stop=180,
        rand_seed=42, verbose=0,
    )
    sim.run()

    prev = sim.results.sir.prevalence.values
    n_rec = sim.results.sir.n_recovered.values

    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        attack_rate=float(n_rec[-1] / 10000),
        final_susceptible=float(sim.results.sir.n_susceptible.values[-1]),
        prevalence=prev.tolist(),
    )


def run_09_cholera():
    """Vignette 09: Cholera-scale SIR (simplified proxy for dual-transmission model)."""
    sim = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=8),
        diseases=ss.SIR(beta=0.5, dur_inf=5, init_prev=0.005),
        dt=1.0, start=0, stop=200,
        rand_seed=42, verbose=0,
    )
    sim.run()

    prev = sim.results.sir.prevalence.values
    n_rec = sim.results.sir.n_recovered.values

    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        attack_rate=float(n_rec[-1] / 5000),
        prevalence=prev.tolist(),
    )


def run_10_ebola():
    """Vignette 10: Ebola-scale SIR (simplified proxy for severity-structured model)."""
    sim = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=5),
        diseases=ss.SIR(beta=0.5, dur_inf=20, init_prev=0.005),
        dt=1.0, start=0, stop=300,
        rand_seed=42, verbose=0,
    )
    sim.run()

    prev = sim.results.sir.prevalence.values
    n_rec = sim.results.sir.n_recovered.values

    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        attack_rate=float(n_rec[-1] / 5000),
        prevalence=prev.tolist(),
    )


def run_11_hiv():
    """Vignette 11: SIR on MFNet sexual network (simplified HIV-like dynamics)."""
    sim = ss.Sim(
        n_agents=5000,
        networks=ss.MFNet(),
        diseases=ss.SIR(beta=dict(mf=0.08), dur_inf=200, init_prev=0.05),
        dt=1.0, start=0, stop=365 * 5,
        rand_seed=42, verbose=0,
    )
    sim.run()

    prev = sim.results.sir.prevalence.values
    n_rec = sim.results.sir.n_recovered.values

    return dict(
        peak_prevalence=float(max(prev)),
        peak_day=int(np.argmax(prev)),
        attack_rate=float(n_rec[-1] / 5000),
        final_prevalence=float(prev[-1]),
        prevalence=prev.tolist(),
    )


def run_14_malaria():
    """Vignette 14: Multi-patch Ross-Macdonald malaria ODE (deterministic)."""
    a, b, c_param = 0.3, 0.1, 0.214
    r, mu, tau = 1.0 / 150.0, 1.0 / 10.0, 10.0
    hvector = np.array([5000.0, 10000.0, 8000.0])
    ivector = np.array([0.001, 0.003, 0.002])
    pij = np.array([
        [0.85, 0.10, 0.05],
        [0.08, 0.80, 0.12],
        [0.06, 0.09, 0.85],
    ])

    # Analytical mosquito density (equilibrium)
    xvector = ivector / r
    gvector = xvector * r / (1 - xvector)
    n = len(xvector)
    fvector = np.zeros(n)
    for i in range(n):
        k_i = np.sum(pij[:, i] * xvector * hvector) / np.sum(pij[:, i] * hvector)
        fvector[i] = b * c_param * k_i / (a * c_param * k_i / mu + 1)
    fmatrix = np.diag(fvector)
    cvector = np.linalg.solve(pij @ fmatrix, gvector)
    m_base = cvector * mu / a**2 / np.exp(-mu * tau)
    m_base = np.clip(m_base, 1e-10, None)

    def seasonal_sinusoidal(t, amplitude=0.8, peak_day=180):
        day_of_year = t % 365
        return max(0.0, 1 + amplitude * np.sin(
            2 * np.pi * (day_of_year - peak_day + 365 / 4) / 365))

    # Euler integration (matches Julia MalariaODE.step!)
    X = ivector / r
    C_cum = np.zeros(3)
    nsteps = 365 * 5
    dt = 1.0

    X_history = [X.copy()]

    for step in range(nsteps):
        t = step * dt
        ms = m_base * seasonal_sinusoidal(t)
        k = (pij.T @ (X * hvector)) / (pij.T @ hvector)
        Z_numer = a**2 * b * c_param * np.exp(-mu * tau) * k
        Z_denom = a * c_param * k + mu
        dC = pij @ (ms * Z_numer / Z_denom) * (1 - X)
        dX = dC - r * X
        X = X + dX * dt
        C_cum = C_cum + dC * dt
        X_history.append(X.copy())

    X_arr = np.array(X_history)

    return dict(
        final_prevalence=[float(X[i]) for i in range(3)],
        peak_prevalence=[float(np.max(X_arr[:, i])) for i in range(3)],
        mean_prevalence=[float(np.mean(X_arr[:, i])) for i in range(3)],
        final_cumulative=[float(C_cum[i]) for i in range(3)],
    )


if __name__ == '__main__':
    print(f"Generating vignette reference data (starsim v{ss.__version__})...")

    results = dict(starsim_version=ss.__version__)

    scenarios = [
        ('v01_introduction', run_01_introduction),
        ('v02_building_a_model', run_02_building_a_model),
        ('v03_demographics', run_03_demographics),
        ('v04_diseases', run_04_diseases),
        ('v05_networks', run_05_networks),
        ('v06_interventions', run_06_interventions),
        ('v07_connectors', run_07_connectors),
        ('v08_measles', run_08_measles),
        ('v09_cholera', run_09_cholera),
        ('v10_ebola', run_10_ebola),
        ('v11_hiv', run_11_hiv),
        ('v12_multisim', run_12_multisim),
        ('v13_calibration', run_13_calibration),
        ('v14_malaria', run_14_malaria),
    ]

    for name, fn in scenarios:
        print(f"  Running {name}...", end=" ", flush=True)
        results[name] = fn()
        print("done")

    outpath = os.path.join(os.path.dirname(__file__), 'vignette_reference.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    v01 = results['v01_introduction']
    print(f"\n  V01 SIR: peak_prev={v01['peak_prevalence']:.4f}, "
          f"attack_rate={v01['attack_rate']:.4f}")
    v03 = results['v03_demographics']
    print(f"  V03 demo: final_alive={v03['demo_final_alive']:.0f}, "
          f"grow={v03['grow_final_alive']:.0f}, "
          f"decline={v03['decline_final_alive']:.0f}")
    v04 = results['v04_diseases']
    print(f"  V04 multi: flu_peak={v04['multi_flu_peak']:.4f}, "
          f"cold_peak={v04['multi_cold_peak']:.4f}")
    v06 = results['v06_interventions']
    print(f"  V06 vx: base_peak={v06['base_peak_prevalence']:.4f}, "
          f"vx_peak={v06['vx_peak_prevalence']:.4f}")
    v07 = results['v07_connectors']
    print(f"  V07 seasonal: base_peak={v07['base_peak_prevalence']:.4f}, "
          f"seasonal_peak={v07['seasonal_peak_prevalence']:.4f}")
    v08 = results['v08_measles']
    print(f"  V08 measles: peak={v08['peak_prevalence']:.4f}, "
          f"attack={v08['attack_rate']:.4f}")
    v11 = results['v11_hiv']
    print(f"  V11 HIV/MFNet: peak={v11['peak_prevalence']:.4f}, "
          f"attack={v11['attack_rate']:.4f}")
    v12 = results['v12_multisim']
    print(f"  V12 multisim: mean_peak={v12['mean_peak_prevalence']:.4f}")
    v13 = results['v13_calibration']
    print(f"  V13 calib: best_beta={v13['best_beta']}, true_beta={v13['true_beta']}")
    v14 = results['v14_malaria']
    print(f"  V14 malaria: final_prev={[f'{x:.4f}' for x in v14['final_prevalence']]}")
    print(f"\n  Saved to {outpath}")
