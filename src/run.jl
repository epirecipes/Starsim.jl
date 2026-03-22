"""
Multi-simulation and parallel execution for Starsim.jl.

Mirrors Python starsim's `run.py`. Provides `MultiSim` for running
multiple simulations and `Scenarios` for parameter sweeps.

Unlike Python starsim (limited by the GIL), Julia's `MultiSim` uses true
multi-threading via `Threads.@threads` for parallel simulation runs. Each
sim gets its own independent RNG state, so parallel runs are deterministic
and thread-safe.
"""

# ============================================================================
# MultiSim
# ============================================================================

"""
    ReducedResult

A single result reduced across multiple simulation runs. Stores the central
value (median or mean) and lower/upper bounds.

# Fields
- `name::Symbol` — result name
- `values::Vector{Float64}` — central values (median or mean)
- `low::Vector{Float64}` — lower bound (quantile or mean - k*std)
- `high::Vector{Float64}` — upper bound (quantile or mean + k*std)
"""
struct ReducedResult
    name::Symbol
    values::Vector{Float64}
    low::Vector{Float64}
    high::Vector{Float64}
end

"""
    MultiSim

Run multiple simulations and aggregate results. Uses Julia's native
multi-threading for true parallelism (no GIL limitation).

# Fields
- `base_sim::Sim` — the template simulation
- `sims::Vector{Sim}` — vector of individual simulations
- `n_runs::Int` — number of runs
- `results::Dict{Symbol, Matrix{Float64}}` — raw results (name → npts × n_runs)
- `reduced::Dict{Symbol, ReducedResult}` — reduced results (after `reduce!`)
- `which::Union{Nothing, Symbol}` — `:reduced` after `reduce!`, nothing otherwise
- `complete::Bool` — whether all runs are complete

# Example
```julia
base = Sim(diseases=SIR(beta=0.05), n_agents=10_000)
msim = MultiSim(base, n_runs=10)
run!(msim)
reduce!(msim)                         # median + 10th/90th quantiles
reduce!(msim; use_mean=true, bounds=2) # mean ± 2σ
```
"""
mutable struct MultiSim
    base_sim::Sim
    sims::Vector{Sim}
    n_runs::Int
    results::Dict{Symbol, Matrix{Float64}}
    reduced::Dict{Symbol, ReducedResult}
    which::Union{Nothing, Symbol}
    complete::Bool
end

"""
    MultiSim(base_sim::Sim; n_runs=5)
    MultiSim(sims::Vector{Sim})

Create a MultiSim from a base simulation (replicated `n_runs` times with
different random seeds) or from a pre-built list of sims.
"""
function MultiSim(base_sim::Sim; n_runs::Int=5)
    sims = Vector{Sim}(undef, n_runs)
    Threads.@threads for i in 1:n_runs
        sim = deepcopy(base_sim)
        sim.pars.rand_seed = base_sim.pars.rand_seed + i
        sims[i] = sim
    end
    MultiSim(base_sim, sims, n_runs,
             Dict{Symbol, Matrix{Float64}}(),
             Dict{Symbol, ReducedResult}(),
             nothing, false)
end

function MultiSim(sims::Vector{Sim})
    n_runs = length(sims)
    n_runs > 0 || throw(ArgumentError("Must supply at least one sim"))
    base_sim = sims[1]
    MultiSim(base_sim, sims, n_runs,
             Dict{Symbol, Matrix{Float64}}(),
             Dict{Symbol, ReducedResult}(),
             nothing, false)
end

function Base.show(io::IO, ms::MultiSim)
    status = ms.complete ? (ms.which === :reduced ? "reduced" : "complete") : "created"
    nt = Threads.nthreads()
    print(io, "MultiSim(n_runs=$(ms.n_runs), status=$status, threads=$nt)")
end

Base.length(ms::MultiSim) = ms.n_runs

"""
    run!(msim::MultiSim; parallel=true, verbose=0)

Run all simulations. When `parallel=true` and Julia was started with multiple
threads (`julia -t N`), uses `Threads.@threads` for true parallel execution —
unlike Python starsim which is limited by the GIL.

# Thread safety
Each sim has independent state (people, RNG, networks), so parallel runs are
fully thread-safe and deterministic.
"""
function run!(msim::MultiSim; parallel::Bool=true, verbose::Int=0)
    nt = Threads.nthreads()
    use_threads = parallel && nt > 1

    if verbose >= 1
        par_str = use_threads ? "parallel ($nt threads)" : "serial"
        println("Running MultiSim ($(msim.n_runs) simulations, $par_str)")
    end

    if use_threads
        Threads.@threads for i in 1:msim.n_runs
            run!(msim.sims[i]; verbose=max(0, verbose - 1))
        end
    else
        for i in 1:msim.n_runs
            if verbose >= 1
                println("  Run $i / $(msim.n_runs)")
            end
            run!(msim.sims[i]; verbose=max(0, verbose - 1))
        end
    end

    _aggregate_results!(msim)
    msim.complete = true

    if verbose >= 1
        println("MultiSim complete")
    end

    return msim
end

"""Aggregate raw results from individual simulations into matrices."""
function _aggregate_results!(msim::MultiSim)
    empty!(msim.results)
    first_sim = msim.sims[1]
    npts = first_sim.t.npts

    # Collect from all module types that have results
    for (mod_name, mod) in all_modules(first_sim)
        mr = module_results(mod)
        for (res_name, _) in mr.data
            key = Symbol("$(mod_name)_$(res_name)")
            mat = zeros(Float64, npts, msim.n_runs)
            for j in 1:msim.n_runs
                sim_j = msim.sims[j]
                mod_j = _get_module(sim_j, mod_name)
                mod_j === nothing && continue
                vals = module_results(mod_j)[res_name].values
                n = min(npts, length(vals))
                @inbounds for k in 1:n
                    mat[k, j] = vals[k]
                end
            end
            msim.results[key] = mat
        end
    end

    return msim
end

"""Look up a module by name across all module containers."""
function _get_module(sim::Sim, name::Symbol)
    for container in (sim.diseases, sim.networks, sim.demographics,
                      sim.interventions, sim.analyzers, sim.connectors,
                      sim.extra_modules)
        haskey(container, name) && return container[name]
    end
    return nothing
end

"""
    reduce!(msim::MultiSim; quantiles=nothing, use_mean=false, bounds=2.0)

Combine multiple simulation results into summary statistics. Matches
Python starsim's `MultiSim.reduce()`.

# Keyword arguments
- `use_mean::Bool` — if `true`, use mean ± `bounds`×σ; if `false` (default), use median with quantile bounds
- `quantiles::Tuple{Float64,Float64}` — lower and upper quantiles (default `(0.1, 0.9)`)
- `bounds::Float64` — standard deviation multiplier when `use_mean=true` (default `2.0`)

# Example
```julia
msim = MultiSim(Sim(diseases=SIR(beta=0.05)), n_runs=20)
run!(msim)

# Default: median with 10th/90th percentile bounds
reduce!(msim)

# Mean with ±2σ bounds
reduce!(msim; use_mean=true, bounds=2.0)
```
"""
function reduce!(msim::MultiSim;
                 quantiles::Union{Nothing, Tuple{Float64,Float64}}=nothing,
                 use_mean::Bool=false,
                 bounds::Float64=2.0)
    msim.complete || error("MultiSim must be run before reducing")

    if quantiles === nothing
        quantiles = (0.1, 0.9)
    end

    empty!(msim.reduced)

    for (key, mat) in msim.results
        npts = size(mat, 1)
        central = Vector{Float64}(undef, npts)
        low     = Vector{Float64}(undef, npts)
        high    = Vector{Float64}(undef, npts)

        if use_mean
            @inbounds for t in 1:npts
                row = @view mat[t, :]
                μ = mean(row)
                σ = std(row)
                central[t] = μ
                low[t]     = μ - bounds * σ
                high[t]    = μ + bounds * σ
            end
        else
            @inbounds for t in 1:npts
                row = @view mat[t, :]
                central[t] = quantile(row, 0.5)
                low[t]     = quantile(row, quantiles[1])
                high[t]    = quantile(row, quantiles[2])
            end
        end

        msim.reduced[key] = ReducedResult(key, central, low, high)
    end

    msim.which = :reduced
    return msim
end

"""
    mean!(msim::MultiSim; bounds=2.0)

Alias for `reduce!(msim; use_mean=true, bounds=bounds)`.
"""
mean!(msim::MultiSim; bounds::Float64=2.0) = reduce!(msim; use_mean=true, bounds=bounds)

"""
    mean_result(msim::MultiSim, key::Symbol) → Vector{Float64}

Get the mean across runs for a raw result.
"""
function mean_result(msim::MultiSim, key::Symbol)
    haskey(msim.results, key) || error("Result $key not found. Available: $(collect(keys(msim.results)))")
    return vec(mean(msim.results[key], dims=2))
end

"""
    quantile_result(msim::MultiSim, key::Symbol, q::Float64) → Vector{Float64}

Get a quantile across runs for a raw result.
"""
function quantile_result(msim::MultiSim, key::Symbol, q::Float64)
    haskey(msim.results, key) || error("Result $key not found. Available: $(collect(keys(msim.results)))")
    mat = msim.results[key]
    npts = size(mat, 1)
    out = Vector{Float64}(undef, npts)
    @inbounds for i in 1:npts
        out[i] = quantile(@view(mat[i, :]), q)
    end
    return out
end

"""
    result_keys(msim::MultiSim) → Vector{Symbol}

List all available result keys.
"""
result_keys(msim::MultiSim) = collect(keys(msim.results))

export MultiSim, ReducedResult, reduce!, mean!, mean_result, quantile_result, result_keys

# ============================================================================
# Scenarios
# ============================================================================

"""
    Scenarios

Run simulations with different parameter sets. Each scenario is a `MultiSim`
with modified parameters.

# Example
```julia
scenarios = Scenarios(
    base_sim = Sim(diseases=SIR()),
    scenarios = Dict(
        :low_beta => Dict(:beta => 0.02),
        :high_beta => Dict(:beta => 0.1),
    ),
    n_runs = 5,
)
run!(scenarios)
```
"""
mutable struct Scenarios
    base_sim::Sim
    scenario_pars::Dict{Symbol, Dict}
    n_runs::Int
    multisims::Dict{Symbol, MultiSim}
    complete::Bool
end

function Scenarios(;
    base_sim::Sim,
    scenarios::Dict = Dict{Symbol, Dict}(),
    n_runs::Int = 5,
)
    Scenarios(base_sim, scenarios, n_runs, Dict{Symbol, MultiSim}(), false)
end

function Base.show(io::IO, s::Scenarios)
    status = s.complete ? "complete" : "created"
    print(io, "Scenarios($(length(s.scenario_pars)) scenarios, n_runs=$(s.n_runs), status=$status)")
end

"""
    run!(scenarios::Scenarios; verbose=1, parallel=true)

Run all scenario simulations. Each scenario's `MultiSim` is run sequentially,
but simulations within each `MultiSim` are parallelized across threads.
"""
function run!(scenarios::Scenarios; verbose::Int=1, parallel::Bool=true)
    for (name, pars) in scenarios.scenario_pars
        if verbose >= 1
            println("Running scenario: $name")
        end

        sim = deepcopy(scenarios.base_sim)
        for (k, v) in pars
            if hasfield(SimPars, Symbol(k))
                setfield!(sim.pars, Symbol(k), v)
            end
        end

        msim = MultiSim(sim; n_runs=scenarios.n_runs)
        run!(msim; parallel=parallel, verbose=max(0, verbose - 1))
        scenarios.multisims[name] = msim
    end

    scenarios.complete = true
    return scenarios
end

export Scenarios
