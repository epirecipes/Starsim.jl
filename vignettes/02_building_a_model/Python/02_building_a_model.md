# Building a Model (Python)
Simon Frost

- [Component-based setup](#component-based-setup)
- [Multiple networks](#multiple-networks)
- [Exporting to DataFrame](#exporting-to-dataframe)

## Component-based setup

``` python
import starsim as ss

net = ss.RandomNet(n_contacts=10)
disease = ss.SIR(beta=0.1, dur_inf=10, init_prev=0.02)

sim = ss.Sim(
    n_agents=2_000,
    networks=net,
    diseases=disease,
    dt=1.0,
    start=0,
    stop=180,
    rand_seed=42,
    verbose=0,
)
sim.run()
```

    Sim(n=2000; 0—180; networks=randomnet; diseases=sir)

## Multiple networks

``` python
sim_multi = ss.Sim(
    n_agents=2_000,
    networks=[
        ss.RandomNet(name='household', n_contacts=4),
        ss.RandomNet(name='workplace', n_contacts=8),
    ],
    diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
    dt=1.0, start=0, stop=180, rand_seed=42, verbose=0,
)
sim_multi.run()

prev = sim_multi.results.sir.prevalence.values
print(f"Peak prevalence with 2 networks: {max(prev):.4f}")
```

    Peak prevalence with 2 networks: 0.8694

## Exporting to DataFrame

``` python
df = sim.to_df()
print(f"Columns: {list(df.columns)[:10]}...")
df.head()
```

    Columns: ['timevec', 'randomnet_n_edges', 'sir_n_susceptible', 'sir_n_infected', 'sir_n_recovered', 'sir_prevalence', 'sir_new_infections', 'sir_cum_infections', 'n_alive', 'n_female']...

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | timevec | randomnet_n_edges | sir_n_susceptible | sir_n_infected | sir_n_recovered | sir_prevalence | sir_new_infections | sir_cum_infections | n_alive | n_female | new_deaths | new_emigrants | cum_deaths |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | 0.0 | 10000.0 | 1934.0 | 66.0 | 0.0 | 0.0330 | 28.0 | 28.0 | 2000.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | 1.0 | 10000.0 | 1880.0 | 120.0 | 0.0 | 0.0600 | 54.0 | 82.0 | 2000.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2 | 2.0 | 10000.0 | 1785.0 | 215.0 | 0.0 | 0.1075 | 95.0 | 177.0 | 2000.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 3 | 3.0 | 10000.0 | 1602.0 | 398.0 | 0.0 | 0.1990 | 183.0 | 360.0 | 2000.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 4 | 4.0 | 10000.0 | 1307.0 | 693.0 | 0.0 | 0.3465 | 295.0 | 655.0 | 2000.0 | 0.0 | 0.0 | 0.0 | 0.0 |

</div>
