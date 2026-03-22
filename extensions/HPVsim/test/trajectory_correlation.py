"""
HPVsim trajectory correlation: save full time series from Python for comparison.
Runs 10 seeds, saves mean trajectories as JSON.
"""
import hpvsim as hpv
import numpy as np
import json
import sys

n_seeds = 10
n_agents = 5000

results_by_seed = []

for seed in range(n_seeds):
    sim = hpv.Sim(
        n_agents=n_agents,
        genotypes=[16],
        dt=1.0,
        start=2000,
        end=2050,
        rand_seed=seed,
        verbose=0,
    )
    sim.run()
    
    r = sim.results
    data = {}
    # Collect all available numeric time series
    for key in r.keys():
        try:
            vals = np.array(r[key], dtype=float)
            if vals.ndim == 1 and len(vals) > 1:
                data[key] = vals.tolist()
        except (TypeError, ValueError):
            pass
    
    # Also get year vector
    if 'year' in r:
        data['year'] = np.array(r['year'], dtype=float).tolist()
    elif 'yearvec' in r:
        data['year'] = np.array(r['yearvec'], dtype=float).tolist()
    
    results_by_seed.append(data)
    print(f"  Seed {seed}: {len(data)} time series collected", file=sys.stderr)

# Compute means across seeds
keys = sorted(results_by_seed[0].keys())
mean_trajectories = {}
for k in keys:
    arrays = [np.array(r[k]) for r in results_by_seed if k in r]
    if len(arrays) == n_seeds:
        stacked = np.stack(arrays)
        mean_trajectories[k] = {
            'mean': stacked.mean(axis=0).tolist(),
            'std': stacked.std(axis=0).tolist(),
            'n_timepoints': len(arrays[0]),
        }

outpath = '/Users/sdwfrost/Projects/starsim/code/Starsim.jl/extensions/HPVsim/test/python_trajectories.json'
with open(outpath, 'w') as f:
    json.dump(mean_trajectories, f)

print(f"Saved {len(mean_trajectories)} mean trajectories to {outpath}", file=sys.stderr)
print(f"Keys: {sorted(mean_trajectories.keys())}", file=sys.stderr)
