# Diseases

Starsim.jl provides three built-in disease models:

## SIR — Susceptible → Infected → Recovered

```julia
sir = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01, p_death=0.0)
```

- `beta`: transmission rate (per day when dt=1)
- `dur_inf`: mean duration of infection (days)
- `init_prev`: initial prevalence (fraction)
- `p_death`: probability of death given infection

## SIS — Susceptible → Infected → Susceptible

```julia
sis = SIS(beta=0.1, dur_inf=20.0, init_prev=0.05)
```

Agents return to the susceptible pool after recovery (no permanent immunity).

## SEIR — Susceptible → Exposed → Infected → Recovered

```julia
seir = SEIR(beta=0.3, dur_exp=8.0, dur_inf=11.0, init_prev=0.001)
```

- `dur_exp`: latent period (days) — agents are infected but not yet infectious
- `dur_inf`: infectious period (days)

## Custom diseases

Extend `AbstractInfection` and implement the lifecycle methods:

```julia
mutable struct MyDisease <: AbstractInfection
    infection::InfectionData
    # custom fields...
end

# Required methods:
init_pre!(d::MyDisease, sim) = ...
init_post!(d::MyDisease, sim) = ...
step_state!(d::MyDisease, sim) = ...
step!(d::MyDisease, sim) = ...
update_results!(d::MyDisease, sim) = ...
step_die!(d::MyDisease, death_uids::UIDs) = ...
finalize!(d::MyDisease) = ...
```
