# Common Random Numbers (CRN)

## Overview

CRN ensures that adding/removing agents or interventions doesn't shift random
number streams for unaffected agents. This enables precise scenario comparison
at the individual agent level.

## Enabling CRN

```julia
Starsim.OPTIONS.slot_scale = 5.0  # Enable CRN with slot scale factor 5
```

## How it works

1. **Per-decision PRNG streams**: Each stochastic decision (transmission, recovery
   time, birth sex) gets its own independent RNG seeded from `hash(name) ⊻ base_seed`

2. **Timestep jumping**: On timestep k, each stream resets to a deterministic state
   depending only on (seed, timestep)

3. **Slot-based indexing**: Each agent has a `slot`. When drawing random numbers,
   `max(slot)+1` values are drawn and indexed by slot. Adding/removing agents
   doesn't shift other agents' draws.

4. **Newborn slots**: Drawn from `Uniform(N, slot_scale*N)` using parent's slot,
   avoiding sequential assignment.

5. **Pairwise XOR for transmission**: Edge-level random numbers use
   `xor(u_i*u_j, u_i-u_j) / typemax(UInt64)` for CRN-safe pairwise draws.

## Example: comparing scenarios

```julia
Starsim.OPTIONS.slot_scale = 5.0

# Baseline
sim1 = Sim(n_agents=5000, networks=RandomNet(n_contacts=10),
    diseases=SIR(beta=0.05, dur_inf=10.0), dt=1.0, stop=365.0,
    rand_seed=42, verbose=0)
run!(sim1)

# With intervention (same seed)
sim2 = Sim(n_agents=5000, networks=RandomNet(n_contacts=10),
    diseases=SIR(beta=0.05, dur_inf=10.0),
    interventions=[RoutineDelivery(product=Vx(efficacy=0.9), prob=0.02, disease_name=:sir)],
    dt=1.0, stop=365.0, rand_seed=42, verbose=0)
run!(sim2)

# Differences are due to the intervention, not random noise
Starsim.OPTIONS.slot_scale = 0.0  # Reset
```

## API

- [`MultiRandom`](@ref): Pairwise CRN-safe random numbers
- [`combine_rvs`](@ref): XOR combining of random value vectors
- [`crn_enabled`](@ref): Check if CRN is active

## GPU CRN support

When CRN is enabled and the Metal GPU extension is loaded, `gpu_step_fused!`
automatically uses CRN-aware kernels:

- **Deterministic seeding**: Per-agent seeds derived from `sim.pars.rand_seed` and
  agent slots via Knuth multiplicative hashing
- **Per-timestep reset**: Seeds reset to `base + ti * 1000` each timestep, ensuring
  order-independence within a timestep
- **Pairwise XOR combining**: Transmission draws combine source and target agent
  RNG streams, matching the CPU `MultiRandom.combine_rvs` logic

```julia
Starsim.OPTIONS.slot_scale = 5.0

sim = Sim(n_agents=100_000, diseases=SIR(beta=0.3, dur_inf=0.05, init_prev=0.01),
    networks=RandomNet(n_contacts=4), dt=1/365, stop=2000.1, rand_seed=42, verbose=0)
init!(sim)
for (_, net) in sim.networks; Starsim.step!(net, sim); end

gsim = to_gpu(sim)           # CRN mode auto-detected
cache_edges!(gsim)
gpu_step_fused!(gsim, :sir; current_ti=1)  # Uses CRN kernels

Starsim.OPTIONS.slot_scale = 0.0  # Reset
```
