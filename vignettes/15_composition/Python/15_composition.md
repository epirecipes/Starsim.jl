# Model Composition in Python starsim
Simon Frost

- [Overview](#overview)
- [Setup](#setup)
- [Basic SIR simulation](#basic-sir-simulation)
- [Multi-module assembly](#multi-module-assembly)
- [Comparison with Julia’s categorical
  composition](#comparison-with-julias-categorical-composition)
- [Notes](#notes)

## Overview

Python starsim assembles simulations by passing module objects to the
`Sim` constructor. While this is flexible, it relies on the user to
ensure that modules are compatible and correctly connected. There is no
formal composition framework analogous to the category-theory approach
in Starsim.jl.

This companion vignette shows the manual assembly equivalent of the
Julia composition examples.

## Setup

``` python
import starsim as ss
import numpy as np
```

## Basic SIR simulation

``` python
n_contacts = 10
beta = 0.5 / n_contacts

sim = ss.Sim(
    n_agents=1000,
    diseases=ss.SIR(beta=beta, dur_inf=4),
    networks=ss.RandomNet(n_contacts=n_contacts),
    start=0,
    stop=40,
    rand_seed=42,
)
sim.run(verbose=0)

prev = sim.results.sir.prevalence
print(f"Peak prevalence: {max(prev):.3f}")
```

    Initializing sim with 1000 agents
    Peak prevalence: 0.245

## Multi-module assembly

Assembling multiple modules requires passing them all to the
constructor:

``` python
sim2 = ss.Sim(
    n_agents=1000,
    diseases=ss.SIR(beta=beta, dur_inf=4, init_prev=0.01),
    networks=ss.RandomNet(n_contacts=n_contacts),
    demographics=[ss.Births(pars=dict(birth_rate=20)), ss.Deaths(pars=dict(death_rate=15))],
    start=0,
    stop=40,
    rand_seed=42,
)
sim2.run(verbose=0)

print(f"Final population: {sim2.results.n_alive[-1]}")
```

    Initializing sim with 1000 agents
    Final population: 1169.0

## Comparison with Julia’s categorical composition

In Julia’s Starsim.jl, the same model is assembled via:

``` python
# Julia equivalent (not runnable in Python):
# disease_s = EpiSharer(:disease, SIR(beta=0.05))
# network_s = EpiSharer(:network, RandomNet(n_contacts=10))
# demog_s   = EpiSharer(:demographics, [Births(rate=20), Deaths(rate=15)])
# sim = compose_epi([disease_s, network_s, demog_s]; n_agents=1000)
```

The categorical approach provides formal verification that modules share
compatible state, while Python relies on runtime checks within the `Sim`
initialization.

## Notes

- Python starsim does not have category-theory composition
- Module compatibility is checked at runtime during `sim.init()`
- The Julia approach using Catlab.jl provides compile-time guarantees
  about module interfaces via undirected wiring diagrams
