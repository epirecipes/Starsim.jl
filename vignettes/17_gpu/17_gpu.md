# GPU acceleration
Simon Frost

- [Overview](#overview)
- [Quick start](#quick-start)
- [Cached static networks](#cached-static-networks)
- [Common Random Numbers (CRN)](#common-random-numbers-crn)
- [Supported features](#supported-features)
- [Restrictions](#restrictions)
- [When GPU helps](#when-gpu-helps)
- [Implementation notes](#implementation-notes)

## Overview

Starsim.jl supports GPU acceleration through three backend extensions:

| Backend   | Hardware      | Package   |
|-----------|---------------|-----------|
| `:metal`  | Apple Silicon | Metal.jl  |
| `:cuda`   | NVIDIA        | CUDA.jl   |
| `:amdgpu` | AMD (ROCm)    | AMDGPU.jl |

The relevant extension is loaded automatically when you load the
corresponding package. A `:gpu` / `:auto` shortcut picks the loaded
backend if exactly one is available.

## Quick start

``` julia
using Starsim
using Metal       # or CUDA / AMDGPU

sim = Sim(
    n_agents = 100_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.001, p_death=0.0),
    dt = 1.0, stop = 100.0, rand_seed = 1,
)

run!(sim; backend=:metal)        # or :cuda, :amdgpu, :gpu (auto)
plot(sim)
```

## Cached static networks

For repeat runs over a static network (e.g. parameter sweeps,
calibration), upload the edge list once and reuse it across timesteps:

``` julia
run_gpu!(sim; backend=:metal, cache_edges=true)
```

The cached path skips per-step edge uploads, which is a significant
speedup on large networks. Per-network `beta` values are honored —
multiple networks with different `beta_per_dt[net_name]` produce the
correct transmission rates on each network.

## Common Random Numbers (CRN)

GPU runs honor the `Starsim.OPTIONS.crn_enabled = true` flag and
`OPTIONS.slot_scale` for slot-based agent indexing, matching the CPU CRN
scheme. Reproducibility holds run-to-run with the same seed and slot
scale, including for cached-edge runs.

## Supported features

The GPU path supports:

- `SIR`, `SIS`, and `SEIR` diseases
- `RandomNet`, `StaticNet`, and other `AbstractNetwork`s exposing
  per-step edges
- Multiple diseases co-existing in one sim
- Multiple networks with per-network `beta` (passed as
  `Dict{Symbol,Float64}`)

## Restrictions

The GPU path will `error` (not silently degrade) for sims with:

- `interventions`, `analyzers`, `connectors`, or `extra_modules`
- `pars.use_aging = true`
- `disease.p_death > 0` (use CPU `run!()` for disease-induced death)
- Disease subtypes other than `SIR` / `SIS` / `SEIR` (e.g. STIsim’s
  `SEIS`, `HIV`, `Syphilis`)

For these, fall back to the CPU `run!(sim)` path.

## When GPU helps

GPU acceleration is most useful for large agent populations (50k+
agents) with many edges per step. For small sims, kernel launch and
host↔device transfer overhead exceeds the parallel work.

The cached-edge path widens the win when the network is static.

## Implementation notes

The GPU extension reuses several preallocated buffers on every step to
avoid GC pressure:

- GPU edge buffers (`edge_p1`, `edge_p2`, `edge_beta`, `rng_buf`)
- CPU staging buffers (`cpu_p1_buf`, `cpu_p2_buf`, `cpu_beta_buf`,
  `cpu_rng_buf`) used to assemble data before `copyto!` into GPU memory
- A shared `new_infected` flag buffer (per-disease counts are
  snapshotted immediately after each `gpu_transmit!` to support
  `new_infections` result tracking)

For the SIS disease only, immunity waning and boost still require a
GPU↔CPU sync each step (because the `immunity` array is CPU-side). SIR
and SEIR avoid that round-trip and run a “fast path” that only
synchronizes results once per step.
