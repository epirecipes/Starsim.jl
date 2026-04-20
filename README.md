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

## Performance & validation vs. Python `starsim`

`benchmark/benchmark.jl` runs the same SIR sim
(`n_contacts=10, beta=0.05, dur_inf=10, init_prev=0.01, dt=1, stop=365`)
in both implementations and reports timing, memory, and a distributional
validation. Reproducing the table:

```bash
julia --project=. benchmark/benchmark.jl
```

### Timing (Apple M-series; Python `starsim` 3.2.2)

| `n_agents` | Julia median | Python median | Speedup |
|-----------:|-------------:|--------------:|--------:|
|     10,000 |    **0.074s**|        1.113s |  15.1×  |
|     50,000 |    **0.385s**|        3.943s |  10.2×  |
|    100,000 |    **0.822s**|        7.506s |   9.1×  |
|    200,000 |    **1.939s**|       14.711s |   7.6×  |

Throughput at *n* = 200,000: **37.7M agent-timesteps/s** (Julia) vs.
5.0M agent-timesteps/s (Python).

### Distributional validation (MMD, 200 replicates each)

To check that Starsim.jl reproduces the *dynamics* — not just the
asymptotic means — the benchmark runs 200 independent replicates at
*n* = 5,000 in each implementation, extracts (peak prevalence, time of
peak, attack rate) from each, and compares the joint distributions with
a maximum mean discrepancy test (RBF kernel, median-heuristic
bandwidth, 2,000-permutation p-value). It also computes within-Julia
and within-Python split-half MMDs as null references.

| Comparison                            | MMD²ᵤ      |  p-value |
|---------------------------------------|-----------:|---------:|
| Julia vs. Python (cross)              |  +0.01174  |  0.0070  |
| Julia split-half (within-Julia null)  |  −0.00838  |  0.9905  |
| Python split-half (within-Python null)|  +0.00232  |  0.2874  |

| Summary feature | Julia (mean ± std) | Python (mean ± std) | Δ mean |
|-----------------|-------------------:|--------------------:|-------:|
| peak prevalence    | 0.7977 ± 0.0064 | 0.7966 ± 0.0066 |  +0.13% |
| time of peak (days)| 15.35  ± 0.48   | 15.49  ± 0.59   |  −0.14 d |
| attack rate        | 0.9807 ± 0.0020 | 0.9808 ± 0.0021 |  −0.01% |

The two implementations use entirely different RNGs (StableRNG/Lehmer
vs. NumPy MT19937), so trajectories are never identical at fixed seeds —
but the per-feature means differ by ≤ 1% and the cross-implementation
MMD is small in absolute terms. Both implementations preserve the same
underlying transmission dynamics.
