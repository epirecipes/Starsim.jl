# Getting started

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/epirecipes/Starsim.jl")
```

## Your first simulation

Create and run a basic SIR model with 5,000 agents on a random contact network:

```julia
using Starsim

sim = Sim(
    n_agents = 5_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
    dt = 1.0,
    stop = 365.0,
    verbose = 0,
)
run!(sim)
```

Or use the built-in [`demo`](@ref) for a pre-configured simulation:

```julia
sim = demo()
run!(sim)
```

## Accessing results

Results are stored per-module. Use [`get_result`](@ref) to retrieve a specific time series:

```julia
prev = get_result(sim, :sir, :prevalence)
n_sus = get_result(sim, :sir, :n_susceptible)
```

## Export to DataFrame

Convert all results to a tidy [`DataFrame`](https://dataframes.juliadata.org/) with [`to_dataframe`](@ref):

```julia
using DataFrames
df = to_dataframe(sim)
```

## Adding demographics

Include births and deaths with crude rates (per 1,000 per year):

```julia
sim = Sim(
    n_agents = 5_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05, dur_inf=10.0),
    demographics = [Births(birth_rate=20.0), Deaths(death_rate=10.0)],
    dt = 1.0,
    stop = 365.0,
)
run!(sim)
```

## Adding interventions

Deliver a vaccine via routine or campaign delivery:

```julia
vx = Vx(efficacy=0.9)

sim = Sim(
    n_agents = 5_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.01),
    interventions = [RoutineDelivery(product=vx, prob=0.02, disease_name=:sir)],
    dt = 1.0,
    stop = 365.0,
)
run!(sim)
```

## Running multiple simulations

Use [`MultiSim`](@ref) to run replicate simulations and aggregate results:

```julia
msim = MultiSim(sim; n_runs=10)
run!(msim)
reduce!(msim)
```

## Next steps

- [Diseases guide](diseases.md) — built-in disease models and custom disease extensions
- [Networks guide](networks.md) — contact network types and graph interop
- [Demographics guide](demographics.md) — births, deaths, and pregnancy
- [Interventions guide](interventions.md) — vaccines, treatment, and screening delivery
- [CRN guide](crn.md) — common random numbers for scenario comparison
- [API Reference](@ref) — complete type and function documentation
