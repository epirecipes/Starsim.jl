# Starsim.jl

[![Build Status](https://github.com/epirecipes/Starsim.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/epirecipes/Starsim.jl/actions/workflows/CI.yml)

[Starsim.jl](https://github.com/epirecipes/Starsim.jl) is a Julia port of the [Starsim](https://starsim.org) agent-based modeling framework for simulating disease transmission. It supports co-transmission of multiple diseases, dynamic contact networks, demographic processes, intervention strategies, and optional GPU acceleration via Metal.jl, CUDA.jl, and AMDGPU.jl.

## Quick start

```julia
using Starsim

sim = Sim(
    n_agents = 5_000,
    networks = [RandomNet(n_contacts=10)],
    diseases = [SIR(beta=0.05, init_prev=0.01)],
)
run!(sim)
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/epirecipes/Starsim.jl")
```

GPU backends are optional and loaded separately:

```julia
Pkg.add("Metal")   # Apple Silicon
Pkg.add("CUDA")    # NVIDIA
Pkg.add("AMDGPU")  # AMD
```

## Documentation

See the [documentation](https://epirecipes.github.io/Starsim.jl/stable/) for tutorials, user guide, and API reference.
