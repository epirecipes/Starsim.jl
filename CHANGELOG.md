# Changelog

All notable changes to Starsim.jl are documented here.

## Unreleased

### Fixed
- Multi-network cached `beta_per_dt` collapse on the GPU path. `cache_edges!`
  used to overwrite a single per-disease beta on every iteration, so
  multi-network sims silently used the last network's beta for transmission
  on all networks. The cached path now records per-network ranges and looks
  up `beta_per_dt[net_name]` per network. (`52ca76b`)
- `gpu_step_fused!` now `error`s when given a `SEIR` disease instead of
  silently corrupting state. (`52ca76b`)
- `_run_gpu_impl!` now `error`s on `extra_modules`, `interventions`,
  `analyzers`, `connectors`, `use_aging=true`, `disease.p_death > 0`, or
  unsupported disease subtypes (anything other than `SIR`/`SIS`/`SEIR`)
  instead of silently skipping them. (`52ca76b`, this release)
- GPU "fast path" (non-SIS) now logs `new_infections` per disease per
  step. Per-disease counts are snapshotted before the next disease's
  `gpu_transmit!` overwrites the shared `new_infected` flag buffer. (`f88696d`)
- Removed duplicate `plot_sim` / `plot_disease` / `plot_comparison` stubs
  and exports from `src/Starsim.jl`. (`52ca76b`)
- Removed vestigial trailing `end # module` from `ext/StarsimGPUCommon.jl`
  that broke Julia 1.12 precompile (the file is included into a backend
  module that supplies the `module` keyword). (`52ca76b`)
- `@gpu_launch` macros in Metal / CUDA / AMDGPU extensions no longer
  trip Julia 1.12's `Base.Docs.docm` macroexpand. The new pattern emits
  a call to a regular `_gpu_kernel_launch` function that internally
  invokes the backend launch macro, sidestepping `Meta.isexpr(call, :call)`
  failing on `Expr(:escape, …)`. (`ef8edb2`)

### Performance
- `gpu_transmit!` and `cache_edges!` now reuse CPU staging buffers
  (`cpu_p1_buf`, `cpu_p2_buf`, `cpu_beta_buf`, `cpu_rng_buf`) on `GPUSim`
  instead of allocating fresh arrays per step per network. (`cfbf0be`)

### Tests
- Renamed `run_cpu_gpu_parity` → `run_cpu_gpu_recovery_parity` to clarify
  scope. (`edb70e7`)
- Added GPU regression tests:
  - `run_gpu_new_infections_logged` — fast path populates new_infections
  - `run_gpu_multinetwork_beta` — cached vs uncached match on multi-net sims
  - `run_gpu_multidisease` — SIR + SEIR coexisting on GPU
  - `run_gpu_alloc_guard` — `gpu_transmit!` per-step allocations stay bounded

### Docs
- Refreshed `vignettes/17_gpu/17_gpu.qmd` from "experimental future API" to
  reflect the working multi-backend `run!(sim; backend=…)` interface,
  cached-edge mode, CRN behavior, and current restrictions.
- Updated `docs/src/api.md` GPU section to cover all three backends.
