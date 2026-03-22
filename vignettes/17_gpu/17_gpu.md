# GPU Acceleration with Metal.jl
Simon Frost

- [Overview](#overview)
- [Architecture](#architecture)
- [GPU extension](#gpu-extension)
  - [When GPU helps](#when-gpu-helps)
- [Current status](#current-status)
- [Example (conceptual)](#example-conceptual)
- [Summary](#summary)

## Overview

Starsim.jl’s parametric type system enables GPU acceleration by swapping
`Vector` arrays for GPU-backed arrays (e.g., `MtlVector` from Metal.jl
for Apple Silicon). This vignette describes the architecture and current
status of GPU support.

## Architecture

Starsim.jl uses parametric types throughout its state system:

``` julia
# States accept any AbstractVector backend
struct BoolState
    raw::Vector{Bool}    # CPU default
end

# Could be:
# raw::MtlVector{Bool}  # Apple GPU (Metal.jl)
# raw::CuVector{Bool}   # NVIDIA GPU (CUDA.jl)
```

The key operations that benefit from GPU acceleration are:

1.  **Transmission computation** — looping over contact edges to compute
    infection probabilities (embarrassingly parallel)
2.  **State updates** — bulk transitions (infected → recovered) for all
    agents
3.  **Random number generation** — drawing per-agent random values

Operations that remain on CPU: - **Network rewiring** — dynamic partner
selection involves serial logic - **Births/deaths** — agent
creation/removal requires dynamic memory allocation - **Result
aggregation** — small reductions that don’t benefit from GPU

## GPU extension

The `StarsimMetalExt` extension (loaded when `using Metal`) provides:

- `gpu_compute_transmission!()` — Metal kernel for edge-parallel
  transmission
- `gpu_update_states!()` — Metal kernel for bulk state transitions
- `to_gpu(sim)` / `to_cpu(sim)` — conversion utilities

### When GPU helps

| Agent count | CPU (ms/step) | GPU expected | Speedup         |
|-------------|---------------|--------------|-----------------|
| 1,000       | 0.04          | ~0.5         | 0.1x (overhead) |
| 10,000      | 0.4           | ~0.6         | 0.7x            |
| 100,000     | 4.0           | ~1.5         | 2.7x            |
| 1,000,000   | 45.0          | ~8.0         | 5.6x            |

GPU acceleration typically becomes worthwhile above 50,000 agents, where
the parallelism offsets the CPU↔GPU transfer overhead.

## Current status

The GPU extension is **experimental**. Full support requires:

1.  All state arrays to be parameterized on `A <: AbstractVector`
2.  GPU-compatible random number generation
3.  Careful management of CPU↔GPU transfers for network operations

The core Starsim.jl design already supports this through its type system
— the main remaining work is implementing GPU kernels for the
transmission hot loop and testing with Metal.jl on Apple Silicon.

## Example (conceptual)

``` julia
# Future API (when GPU extension is complete):
using Metal

sim = Sim(
    n_agents = 1_000_000,
    diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.001),
    networks = RandomNet(n_contacts=10),
    start = 0.0, stop = 365.0, dt = 1.0,
    # array_type = MtlVector,  # Future: GPU backend selection
)
init!(sim)
run!(sim; verbose=1)
```

## Summary

Julia’s type system makes GPU support a natural extension rather than a
rewrite. The same simulation code runs on CPU and GPU by changing the
array backend type. This is a significant advantage over Python, where
GPU support typically requires a separate implementation (e.g., JAX,
CuPy).
