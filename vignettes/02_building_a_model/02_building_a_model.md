# Building a Model
Simon Frost

- [Overview](#overview)
- [Component-based setup](#component-based-setup)
- [Multiple networks](#multiple-networks)
- [Controlling time](#controlling-time)
- [Accessing and exporting results](#accessing-and-exporting-results)
- [Saving and loading](#saving-and-loading)

## Overview

This vignette shows how to build Starsim.jl simulations from components:
People, Networks, Diseases, and parameters. We cover dict-based vs
component-based setup, heterogeneous contacts, and result export.

## Component-based setup

Each simulation component is a Julia struct. Pass them directly to
`Sim`:

``` julia
using Starsim
using Plots

# Create components explicitly
n_contacts = 10
beta = 0.5 / (2 * n_contacts)
net = RandomNet(n_contacts=n_contacts)
disease = SIR(beta=beta, dur_inf=4.0, init_prev=0.02)

sim = Sim(
    n_agents = 1_000,
    networks = net,
    diseases = disease,
    dt = 1.0,
    stop = 40.0,
    rand_seed = 42,
    verbose = 0,
)
run!(sim)
```

    Sim(1000 agents, 0.0→40.0, dt=1.0, nets=1, dis=1, status=complete)

## Multiple networks

You can pass a vector of networks. Each gets a unique name:

``` julia
n_contacts_hh = 4
n_contacts_work = 8
beta = 0.5 / (2 * (n_contacts_hh + n_contacts_work))

sim_multi = Sim(
    n_agents = 1_000,
    networks = [
        RandomNet(name=:household, n_contacts=n_contacts_hh),
        RandomNet(name=:workplace, n_contacts=n_contacts_work),
    ],
    diseases = SIR(beta=beta, dur_inf=4.0, init_prev=0.01),
    dt = 1.0,
    stop = 40.0,
    rand_seed = 42,
    verbose = 0,
)
run!(sim_multi)

prev = get_result(sim_multi, :sir, :prevalence)
println("Peak prevalence with 2 networks: $(round(maximum(prev), digits=4))")
```

    Peak prevalence with 2 networks: 0.049

## Controlling time

The `dt` parameter sets the timestep. With `dt=1`, each step is 1 day.

``` julia
# Compare dt=1 (daily) vs dt=0.5 (twice daily)
n_contacts = 10
beta = 0.5 / (2 * n_contacts)

sim1 = Sim(n_agents=1000, networks=RandomNet(n_contacts=n_contacts),
    diseases=SIR(beta=beta, dur_inf=4.0, init_prev=0.02),
    dt=1.0, stop=40.0, rand_seed=42, verbose=0)
run!(sim1)

sim2 = Sim(n_agents=1000, networks=RandomNet(n_contacts=n_contacts),
    diseases=SIR(beta=beta, dur_inf=4.0, init_prev=0.02),
    dt=0.5, stop=40.0, rand_seed=42, verbose=0)
run!(sim2)

p1 = get_result(sim1, :sir, :prevalence)
p2 = get_result(sim2, :sir, :prevalence)
println("dt=1.0 peak: $(round(maximum(p1), digits=4))")
println("dt=0.5 peak: $(round(maximum(p2), digits=4))")
```

    dt=1.0 peak: 0.076
    dt=0.5 peak: 0.051

## Accessing and exporting results

``` julia
using DataFrames

df = to_dataframe(sim)
println("Available columns: ", names(df))
first(df, 5)
```

    Available columns: ["time", "n_alive", "sir_new_infections", "sir_n_susceptible", "sir_n_infected", "sir_n_recovered", "sir_prevalence"]

<div><div style = "float: left;"><span>5×7 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;">

| Row | time | n_alive | sir_new_infections | sir_n_susceptible | sir_n_infected | sir_n_recovered | sir_prevalence |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 | Float64 |
| 1 | 0.0 | 1000.0 | 3.0 | 977.0 | 23.0 | 0.0 | 0.023 |
| 2 | 1.0 | 1000.0 | 4.0 | 973.0 | 27.0 | 0.0 | 0.027 |
| 3 | 2.0 | 1000.0 | 8.0 | 965.0 | 35.0 | 0.0 | 0.035 |
| 4 | 3.0 | 1000.0 | 12.0 | 953.0 | 47.0 | 0.0 | 0.047 |
| 5 | 4.0 | 1000.0 | 11.0 | 942.0 | 58.0 | 0.0 | 0.058 |

</div>

## Saving and loading

Simulations can be serialized with Julia’s `Serialization` stdlib:

``` julia
using Serialization

# Save
io = IOBuffer()
serialize(io, sim.pars)
println("SimPars serialized: $(position(io)) bytes")
```

    SimPars serialized: 95 bytes
