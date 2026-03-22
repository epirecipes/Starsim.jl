# Starsim.jl

*Agent-based modeling framework for simulating disease transmission.*

[Starsim.jl](https://github.com/epirecipes/Starsim.jl) is a Julia port of the Python [Starsim](https://starsim.org) framework for agent-based simulation of infectious disease transmission over contact networks.

## Features

- **Modular architecture** — diseases, networks, demographics, interventions, analyzers, and connectors are independent, composable modules
- **Built-in disease models** — [`SIR`](@ref), [`SIS`](@ref), [`SEIR`](@ref), with a simple API for defining custom diseases
- **Flexible networks** — [`RandomNet`](@ref), [`MFNet`](@ref), [`MSMNet`](@ref), [`StaticNet`](@ref), [`MixingPool`](@ref), and maternal/household networks
- **Common Random Numbers (CRN)** — slot-based RNG streams for robust counterfactual scenario comparison
- **GPU-ready** — generic array backend with Metal.jl / CUDA.jl support
- **AD-compatible** — designed for automatic differentiation through simulations (ForwardDiff, Enzyme)
- **Calibration** — built-in [`Calibration`](@ref) framework with Optimization.jl integration
- **Categorical composition** — Catlab.jl extension for compositional epidemiological modeling

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/epirecipes/Starsim.jl")
```

## Quick start

```julia
using Starsim

sim = Sim(
    n_agents = 5_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
    dt = 1.0,
    stop = 365.0,
)
run!(sim)

# Access results
df = to_dataframe(sim)
prev = get_result(sim, :sir, :prevalence)
```

Or use the one-liner [`demo`](@ref) to get a pre-configured simulation:

```julia
sim = demo()
run!(sim)
```

## Module types

| Category | Types | Description |
|----------|-------|-------------|
| **Diseases** | [`SIR`](@ref), [`SIS`](@ref), [`SEIR`](@ref) | Infectious disease models |
| **Networks** | [`RandomNet`](@ref), [`MFNet`](@ref), [`MSMNet`](@ref), [`StaticNet`](@ref), [`MixingPool`](@ref) | Contact structures |
| **Demographics** | [`Births`](@ref), [`Deaths`](@ref), [`Pregnancy`](@ref) | Population dynamics |
| **Interventions** | [`RoutineDelivery`](@ref), [`CampaignDelivery`](@ref) | Vaccines, treatment, screening |
| **Connectors** | [`Seasonality`](@ref), [`CoinfectionConnector`](@ref) | Cross-module interactions |
| **Analyzers** | [`FunctionAnalyzer`](@ref), [`Snapshot`](@ref), [`InfectionLog`](@ref) | Custom result tracking |

## Documentation outline

```@contents
Pages = [
    "guide/getting_started.md",
    "guide/diseases.md",
    "guide/networks.md",
    "guide/demographics.md",
    "guide/interventions.md",
    "guide/crn.md",
    "api.md",
]
Depth = 2
```
