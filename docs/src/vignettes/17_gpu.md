# GPU Acceleration
Simon Frost

- [Overview](#overview)
- [Architecture](#architecture)
- [Supported backends](#supported-backends)
- [Supported diseases](#supported-diseases)
- [Usage](#usage)
- [Performance](#performance)
- [API reference](#api-reference)
- [Summary](#summary)

## Overview

Starsim.jl provides GPU acceleration through three package extensions
supporting all major GPU platforms. Disease dynamics (transmission,
recovery, state transitions) run on GPU while structurally dynamic
operations (network rewiring, demographics) remain on CPU.

## Architecture

The GPU extension uses a **hybrid CPU/GPU** approach:

**GPU (MtlVector, Float32/UInt8):**
- Agent state arrays: `susceptible`, `infected`, `recovered`, `exposed`
- Disease timing arrays: `ti_infected`, `ti_recovered`, `ti_exposed`
- Transmission kernel (per-edge probability evaluation)
- Recovery kernel (`ti_recovered ≤ current_ti` check)
- SEIR exposure-to-infection transition
- GPU-side result counting (`sum()` on MtlVector)

**CPU (Vector, Float64/Bool):**
- Network edge lists (rebuilt each timestep for dynamic networks)
- Recovery duration sampling (lognormal distribution)
- People management (births, deaths, UID tracking)
- SIS immunity waning (requires per-agent float arithmetic)

The GPU loop follows the exact same 16-step integration order as the CPU,
ensuring identical disease dynamics.

## Supported backends

| Backend | Extension | GPU type | Package | Platform |
|---------|-----------|----------|---------|----------|
| Metal   | `StarsimMetalExt` | `MtlVector` | [Metal.jl](https://github.com/JuliaGPU/Metal.jl) | Apple Silicon (macOS) |
| CUDA    | `StarsimCUDAExt`  | `CuVector`  | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | NVIDIA GPUs |
| ROCm    | `StarsimAMDGPUExt`| `ROCVector` | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) | AMD GPUs |

Extensions load automatically when the corresponding GPU package is
imported alongside Starsim:

``` julia
using Starsim, Metal   # Apple Silicon
using Starsim, CUDA    # NVIDIA
using Starsim, AMDGPU  # AMD
```

All three backends share the same API (`run_gpu!`, `to_gpu`, `to_cpu`,
etc.) and implement identical algorithms. All use Float32/UInt8 on GPU
for maximum performance.

## Supported diseases

| Disease | GPU support | Notes |
|---------|:-----------:|-------|
| SIR     | ✅          | Full support including lognormal recovery |
| SIS     | ✅          | Immunity waning handled via CPU roundtrip |
| SEIR    | ✅          | E→I transition with recovery time on GPU |

## Usage

``` julia
using Starsim, Metal

# Simple — use run_gpu! directly
sim = Sim(
    n_agents = 100_000,
    diseases = [SIR(beta=0.05, dur_inf=10.0, init_prev=0.01)],
    networks = [RandomNet(n_contacts=10)],
    stop = 50.0, dt = 1.0,
)
Starsim.run_gpu!(sim; verbose=1, backend=:metal)

# Static network mode (edges generated once, cached on GPU)
sim2 = Sim(
    n_agents = 1_000_000,
    diseases = [SIR(beta=0.05, dur_inf=10.0, init_prev=0.01)],
    networks = [RandomNet(n_contacts=10)],
    stop = 50.0, dt = 1.0,
)
Starsim.run_gpu!(sim2; verbose=1, backend=:metal, cache_edges=true)
```

## Performance

Benchmarks on Apple M2 Ultra (Metal GPU, 76 GPU cores):

**Dynamic edges (default — edges regenerated each step):**

| Agents | CPU (M a-ts/s) | GPU (M a-ts/s) | Speedup |
|--------|:-:|:-:|:-:|
| 1K     | 12 | 0.2 | 0.02x |
| 10K    | 12 | 1.6 | 0.13x |
| 100K   | 12 | 4.2 | 0.36x |
| 500K   | 9  | 5.1 | 0.55x |
| 1M     | 9  | 5.2 | 0.60x |
| 5M     | 6  | 4.8 | 0.79x |

**Cached edges (static network — single upload):**

| Agents | CPU (M a-ts/s) | GPU (M a-ts/s) | Speedup |
|--------|:-:|:-:|:-:|
| 10K    | 12 | 2.0 | 0.16x |
| 100K   | 11 | 6.7 | 0.59x |
| 500K   | 9  | 8.0 | 0.91x |
| 1M     | 9  | 8.2 | 0.94x |
| 5M     | 6  | 7.8 | **1.28x** |

GPU overtakes CPU at ~5M agents with cached edges — the crossover where
GPU parallelism outweighs kernel launch overhead. Julia's CPU code is
highly optimized on Apple Silicon (native SIMD), making GPU acceleration
less impactful than on platforms with weaker CPUs or discrete GPUs. The
GPU path is most useful for:

- Very large simulations (1M+ agents) with static networks
- CUDA.jl / AMDGPU.jl on discrete GPUs (where dedicated VRAM and higher
  memory bandwidth should improve the crossover point)
- Demonstrating the GPU-ready architecture

**Correctness**: GPU vs CPU mean trajectory correlation r > 0.999 for
all three disease types (SIR, SIS, SEIR) over 30 seeds.

## API reference

| Function | Description |
|----------|-------------|
| `to_gpu(sim; backend=:auto)` | Convert initialized Sim to GPUSim |
| `to_cpu(gsim)` | Copy GPU state back to CPU Sim |
| `run_gpu!(sim; backend=:auto)` | Full GPU simulation lifecycle |
| `gpu_step_state!(gsim, :sir; current_ti=ti)` | Recovery transitions on GPU |
| `gpu_transmit!(gsim, :sir; current_ti=ti)` | Transmission with edge upload |
| `cache_edges!(gsim)` | Upload edges once for static networks |
| `gpu_transmit_cached!(gsim, :sir; current_ti=ti)` | Transmission with cached edges |
| `sync_to_gpu!(gsim)` | Re-upload CPU state after CPU-side modifications |

## Summary

Starsim.jl provides GPU acceleration for SIR, SIS, and SEIR disease
models across three platforms: Apple Silicon (Metal.jl), NVIDIA (CUDA.jl),
and AMD (AMDGPU.jl). All backends share the same API and implement
identical algorithms. Metal benchmarks on Apple Silicon show GPU overtaking
CPU at ~5M agents with cached edges; discrete NVIDIA/AMD GPUs with
dedicated VRAM are expected to show larger speedups at smaller agent
counts. The GPU path produces statistically identical results to the CPU
path (r > 0.999).
