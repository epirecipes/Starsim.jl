"""
Generate reference SIR results from Python starsim.
Outputs JSON with time-series data for cross-validation.
"""
import json
import numpy as np
import starsim as ss
import os
os.environ['STARSIM_INSTALL_FONTS'] = '0'

def run_sir_reference():
    """Run a deterministic SIR simulation and save results."""
    sim = ss.Sim(
        n_agents=5000,
        networks=ss.RandomNet(n_contacts=ss.poisson(10)),
        diseases=ss.SIR(
            beta=0.05,
            init_prev=0.01,
            dur_inf=10,  # days
        ),
        start=0,
        stop=365,
        dt=1,  # 1 day
        rand_seed=42,
        verbose=0,
    )
    sim.run()

    # Extract results
    results = {
        'n_agents': int(sim.pars.n_agents),
        'dt': float(sim.pars.dt),
        'npts': int(sim.t.npts),
        'rand_seed': 42,
        'beta': 0.05,
        'init_prev': 0.01,
        'dur_inf': 10,
        'n_contacts': 10,
        'n_susceptible': sim.results.sir.n_susceptible.values.tolist(),
        'n_infected': sim.results.sir.n_infected.values.tolist(),
        'n_recovered': sim.results.sir.n_recovered.values.tolist(),
        'prevalence': sim.results.sir.prevalence.values.tolist(),
        'new_infections': sim.results.sir.new_infections.values.tolist(),
        'tvec': [float(x) for x in sim.t.tvec],
    }

    # Summary statistics for easy comparison
    results['peak_prevalence'] = float(max(results['prevalence']))
    results['final_susceptible'] = float(results['n_susceptible'][-1])
    results['final_infected'] = float(results['n_infected'][-1])
    results['final_recovered'] = float(results['n_recovered'][-1])
    results['total_infections'] = float(sum(results['new_infections']))
    results['attack_rate'] = results['total_infections'] / results['n_agents']

    return results

def run_sir_demographics_reference():
    """Run SIR with births/deaths and save results."""
    sim = ss.Sim(
        n_agents=2000,
        networks=ss.RandomNet(n_contacts=ss.poisson(10)),
        diseases=ss.SIR(
            beta=0.05,
            init_prev=0.01,
            dur_inf=10,
        ),
        demographics=[
            ss.Births(pars={'birth_rate': 20}),
            ss.Deaths(pars={'death_rate': 10}),
        ],
        start=2000,
        stop=2010,
        dt=1,
        rand_seed=42,
        verbose=0,
    )
    sim.run()

    results = {
        'n_agents': int(sim.pars.n_agents),
        'final_alive': int(sim.people.alive.sum()),
        'total_births': float(sim.results.births.new.values.sum()),
        'total_deaths': float(sim.results.deaths.new.values.sum()),
        'final_susceptible': float(sim.results.sir.n_susceptible.values[-1]),
        'final_infected': float(sim.results.sir.n_infected.values[-1]),
        'final_recovered': float(sim.results.sir.n_recovered.values[-1]),
    }

    return results


if __name__ == '__main__':
    print("Generating Python starsim reference data...")

    sir_results = run_sir_reference()
    print(f"  SIR: peak_prev={sir_results['peak_prevalence']:.4f}, "
          f"attack_rate={sir_results['attack_rate']:.4f}, "
          f"final S/I/R={sir_results['final_susceptible']:.0f}/"
          f"{sir_results['final_infected']:.0f}/"
          f"{sir_results['final_recovered']:.0f}")

    demo_results = run_sir_demographics_reference()
    print(f"  SIR+demo: final_alive={demo_results['final_alive']}, "
          f"births={demo_results['total_births']:.0f}, "
          f"deaths={demo_results['total_deaths']:.0f}")

    all_results = {
        'starsim_version': '3.2.2',
        'sir': sir_results,
        'sir_demographics': demo_results,
    }

    outpath = os.path.join(os.path.dirname(__file__), 'reference_data.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"  Saved to {outpath}")
