"""
Simulation orchestrator for Starsim.jl.

Mirrors Python starsim's `sim.py`. The `Sim` struct ties everything
together: people, networks, diseases, demographics, interventions,
analyzers, connectors, and the integration loop.
"""

# ============================================================================
# Sim — simulation container
# ============================================================================

"""
    Sim

The main simulation object. Holds all modules and orchestrates
initialization and execution.

# Keyword arguments
- `pars::SimPars` — simulation parameters (or keyword overrides)
- `n_agents::Int` — number of agents (default 10_000)
- `start::Real` — start year (default 0.0)
- `stop::Real` — stop year (default 10.0)
- `dt::Real` — timestep in years (default 1.0)
- `rand_seed::Int` — RNG seed (default 0)
- `networks` — network spec (AbstractNetwork, Dict, Vector, or nothing)
- `diseases` — disease spec (AbstractDisease, Dict, Vector, or nothing)
- `demographics` — demographics spec (AbstractDemographics, Dict, Vector, or nothing)
- `interventions` — intervention spec (or nothing)
- `analyzers` — analyzer spec (or nothing)
- `connectors` — connector spec (or nothing)

# Example
```julia
n_contacts = 10
beta = 0.5 / n_contacts

sim = Sim(
    n_agents = 1000,
    networks = RandomNet(n_contacts=n_contacts),
    diseases = SIR(beta=beta, dur_inf=4.0, init_prev=0.01),
    stop = 40.0,
)
run!(sim)
```
"""
mutable struct Sim
    pars::SimPars
    people::People
    t::Timeline
    loop::Loop
    results::Results

    # Module containers (ordered dicts for deterministic iteration)
    networks::OrderedDict{Symbol, AbstractNetwork}
    diseases::OrderedDict{Symbol, AbstractDisease}
    demographics::OrderedDict{Symbol, AbstractDemographics}
    interventions::OrderedDict{Symbol, AbstractIntervention}
    analyzers::OrderedDict{Symbol, AbstractAnalyzer}
    connectors::OrderedDict{Symbol, AbstractConnector}
    extra_modules::OrderedDict{Symbol, AbstractModule}  # Generic modules (ODE, etc.)

    initialized::Bool
    complete::Bool
end

function Sim(;
    pars::Union{SimPars, Nothing} = nothing,
    n_agents::Int = 10_000,
    start::Real = 0.0,
    stop::Real = 10.0,
    dt::Union{Real, Duration} = 1.0,
    rand_seed::Int = 0,
    pop_scale::Real = 1.0,
    use_aging::Bool = true,
    verbose::Int = 1,
    networks = nothing,
    diseases = nothing,
    demographics = nothing,
    interventions = nothing,
    analyzers = nothing,
    connectors = nothing,
    modules = nothing,
)
    # Build SimPars
    if pars === nothing
        pars = SimPars(n_agents=n_agents, start=start, stop=stop, dt=dt,
                       rand_seed=rand_seed, pop_scale=pop_scale,
                       use_aging=use_aging, verbose=verbose)
    end

    people = People(pars.n_agents; slot_scale=get_slot_scale())
    t = Timeline(start=pars.start, stop=pars.stop, dt=pars.dt)
    loop = Loop()
    results = Results()

    sim = Sim(
        pars, people, t, loop, results,
        OrderedDict{Symbol, AbstractNetwork}(),
        OrderedDict{Symbol, AbstractDisease}(),
        OrderedDict{Symbol, AbstractDemographics}(),
        OrderedDict{Symbol, AbstractIntervention}(),
        OrderedDict{Symbol, AbstractAnalyzer}(),
        OrderedDict{Symbol, AbstractConnector}(),
        OrderedDict{Symbol, AbstractModule}(),
        false, false,
    )

    # Add modules from keyword arguments
    _add_modules!(sim, networks, sim.networks)
    _add_modules!(sim, diseases, sim.diseases)
    _add_modules!(sim, demographics, sim.demographics)
    _add_modules!(sim, interventions, sim.interventions)
    _add_modules!(sim, analyzers, sim.analyzers)
    _add_modules!(sim, connectors, sim.connectors)
    _add_modules!(sim, modules, sim.extra_modules)

    return sim
end

"""Add modules from various input formats to the ordered dict."""
function _add_modules!(sim, spec, container)
    spec === nothing && return
    if spec isa AbstractModule
        container[module_name(spec)] = spec
    elseif spec isa AbstractVector
        for m in spec
            container[module_name(m)] = m
        end
    elseif spec isa AbstractDict
        for (k, v) in spec
            container[Symbol(k)] = v
        end
    end
    return
end

function Base.show(io::IO, s::Sim)
    status = s.complete ? "complete" : (s.initialized ? "initialized" : "created")
    print(io, "Sim($(s.pars.n_agents) agents, $(s.pars.start)→$(s.pars.stop), " *
              "dt=$(s.pars.dt), nets=$(length(s.networks)), dis=$(length(s.diseases)), " *
              "status=$status)")
end

"""
    all_modules(sim::Sim)

Iterate over all modules in the simulation in order:
networks, demographics, diseases, connectors, interventions, analyzers.
Returns pairs of (name, module).
"""
function all_modules(sim::Sim)
    mods = Pair{Symbol, AbstractModule}[]
    for (k, v) in sim.extra_modules;  push!(mods, k => v); end
    for (k, v) in sim.networks;       push!(mods, k => v); end
    for (k, v) in sim.demographics;   push!(mods, k => v); end
    for (k, v) in sim.diseases;       push!(mods, k => v); end
    for (k, v) in sim.connectors;     push!(mods, k => v); end
    for (k, v) in sim.interventions;  push!(mods, k => v); end
    for (k, v) in sim.analyzers;      push!(mods, k => v); end
    return mods
end

export Sim, all_modules

# ============================================================================
# Initialization
# ============================================================================

"""
    init!(sim::Sim)

Initialize the simulation. Follows Python starsim's 10-step init:
1. Initialize people
2. init_pre! all modules (registers states, sets up timelines)
3. init_post! all modules (seed infections, etc.)
4. Build integration loop
5. Initialize results

Returns `sim` for chaining.
"""
function init!(sim::Sim)
    sim.initialized && return sim

    # 1. Initialize people (pass rand_seed for per-seed population variation)
    init_people!(sim.people; use_aging=sim.pars.use_aging, rand_seed=sim.pars.rand_seed)

    # 2. init_pre! all modules (link to sim, register states)
    for (_, mod) in all_modules(sim)
        init_pre!(mod, sim)
    end

    # 3. init_post! all modules (seed infections, etc.)
    for (_, mod) in all_modules(sim)
        init_post!(mod, sim)
    end

    # 4. Build integration loop
    build_loop!(sim.loop, sim)

    # 5. Initialize sim-level results
    npts = sim.t.npts
    push!(sim.results, Result(:n_alive; npts=npts, label="Alive", scale=false))

    sim.initialized = true
    return sim
end

# ============================================================================
# Running
# ============================================================================

"""
    run!(sim::Sim; verbose=nothing, backend=:cpu)

Run the simulation. Initializes first if needed. Returns `sim` for chaining.

# Keywords
- `verbose::Union{Int, Nothing}` — override the sim's verbosity level
- `backend::Symbol` — execution backend:
  - `:cpu` (default) — standard CPU execution
  - `:gpu` / `:auto` — use the single loaded GPU backend extension
  - `:metal` — Apple Silicon GPU backend (requires `using Metal`)
  - `:cuda` — NVIDIA GPU backend (requires `using CUDA`)
  - `:amdgpu` — AMD GPU backend (requires `using AMDGPU`)

# Examples
```julia
# CPU (default)
sim = Sim(diseases=SIR(beta=0.1)) |> run!

# GPU — choose automatically if exactly one backend is loaded
sim = Sim(n_agents=1_000_000, diseases=SIR(beta=0.05),
          networks=RandomNet(n_contacts=10))
run!(sim; backend=:gpu)
```
"""
function run!(sim::Sim; verbose::Union{Int, Nothing}=nothing, backend::Symbol=:cpu)
    v = verbose === nothing ? sim.pars.verbose : verbose

    if backend in (:gpu, :auto, :metal, :cuda, :amdgpu)
        return run_gpu!(sim; verbose=v, backend=backend)
    end

    if !sim.initialized
        init!(sim)
    end

    if v >= 1
        println("Running simulation ($(sim.pars.n_agents) agents, " *
                "$(sim.t.npts) steps, dt=$(sim.pars.dt))")
    end

    run_loop!(sim.loop, sim; verbose=v)

    # Finalize all modules
    for (_, mod) in all_modules(sim)
        finalize!(mod, sim)
    end

    # Scale results
    if sim.pars.pop_scale != 1.0
        scale_results!(sim.results, sim.pars.pop_scale)
        for (_, mod) in all_modules(sim)
            scale_results!(module_results(mod), sim.pars.pop_scale)
        end
    end

    sim.complete = true
    return sim
end

const GPU_EXTENSIONS = Dict(
    :metal  => (:StarsimMetalExt,  "Metal.jl"),
    :cuda   => (:StarsimCUDAExt,   "CUDA.jl"),
    :amdgpu => (:StarsimAMDGPUExt, "AMDGPU.jl"),
)

_gpu_backend_names() = collect(keys(GPU_EXTENSIONS))

function _loaded_gpu_backends()
    loaded = Symbol[]
    for backend in _gpu_backend_names()
        extname, _ = GPU_EXTENSIONS[backend]
        Base.get_extension(Starsim, extname) === nothing || push!(loaded, backend)
    end
    return loaded
end

function _require_gpu_backend(backend::Symbol)
    if backend in (:gpu, :auto)
        loaded = _loaded_gpu_backends()
        isempty(loaded) && error(
            "GPU backend requires a loaded GPU package. Load one of: `using Metal`, `using CUDA`, or `using AMDGPU`."
        )
        length(loaded) == 1 && return only(loaded)
        choices = join([":$name" for name in loaded], ", ")
        error("Multiple GPU backends are loaded ($choices). Select one explicitly with `backend=:metal`, `:cuda`, or `:amdgpu`.")
    elseif haskey(GPU_EXTENSIONS, backend)
        extname, pkgname = GPU_EXTENSIONS[backend]
        Base.get_extension(Starsim, extname) === nothing &&
            error("GPU backend `:$backend` requires loading $pkgname first: `using $(replace(pkgname, ".jl" => ""))`")
        return backend
    else
        valid = join([":cpu", ":gpu", ":auto", ":metal", ":cuda", ":amdgpu"], ", ")
        error("Unknown backend `:$backend`. Valid backends are $valid.")
    end
end

"""
    to_gpu(sim; backend=:auto)

Convert an initialized simulation to GPU-backed arrays using the selected
GPU extension. With `backend=:auto` (or `:gpu`), exactly one loaded GPU
backend must be available.
"""
function to_gpu(sim::Sim; backend::Symbol=:auto)
    resolved = _require_gpu_backend(backend)
    return _to_gpu_backend(sim, Val(resolved))
end

function _to_gpu_backend(sim::Sim, ::Val{B}) where {B}
    error("GPU backend `:$B` is loaded but did not provide a `to_gpu` implementation.")
end

"""
    to_cpu(sim)

No-op on CPU simulations; GPU extensions provide `to_cpu(::GPUSim)`.
"""
to_cpu(sim::Sim) = sim

"""
    run_gpu!(sim; verbose=1, backend=:auto, kwargs...)

Run a simulation on the selected GPU backend. With `backend=:auto` (or
`:gpu`), exactly one loaded GPU backend must be available.
"""
function run_gpu!(sim::Sim; verbose::Int=1, backend::Symbol=:auto, kwargs...)
    resolved = _require_gpu_backend(backend)
    return _run_gpu_backend!(sim, Val(resolved); verbose=verbose, kwargs...)
end

function _run_gpu_backend!(sim::Sim, ::Val{B}; kwargs...) where {B}
    error("GPU backend `:$B` is loaded but did not provide a `run_gpu!` implementation.")
end

"""
    reset!(sim::Sim)

Reset the simulation to its pre-run state. Allows re-running with
different seeds or parameters.
"""
function reset!(sim::Sim)
    sim.initialized = false
    sim.complete = false
    sim.people = People(sim.pars.n_agents; slot_scale=get_slot_scale())
    sim.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    sim.loop = Loop()
    sim.results = Results()
    return sim
end

export init!, run!, run_gpu!, reset!

# ============================================================================
# Results access
# ============================================================================

"""
    to_dataframe(sim::Sim)

Convert all simulation results to a single DataFrame.
"""
function to_dataframe(sim::Sim)
    tvec = collect(sim.t.start:sim.t.dt:sim.t.stop - sim.t.dt)
    n = min(length(tvec), sim.t.npts)
    tvec = tvec[1:n]

    df = DataFrame(time=tvec)

    # Sim-level results
    for (name, result) in sim.results.data
        nn = min(n, length(result.values))
        df[!, name] = result.values[1:nn]
    end

    # Module results
    for (mod_name, mod) in all_modules(sim)
        for (res_name, result) in module_results(mod).data
            col_name = Symbol("$(mod_name)_$(res_name)")
            nn = min(n, length(result.values))
            df[!, col_name] = result.values[1:nn]
        end
    end

    return df
end

"""
    get_result(sim::Sim, module_name::Symbol, result_name::Symbol) → Vector{Float64}

Get a specific result from any module by name.
"""
function get_result(sim::Sim, mod_name::Symbol, result_name::Symbol)
    # Search all module containers
    for container in (sim.diseases, sim.networks, sim.demographics,
                      sim.interventions, sim.analyzers, sim.connectors,
                      sim.extra_modules)
        if haskey(container, mod_name)
            res = module_results(container[mod_name])
            if haskey(res, result_name)
                return res[result_name].values
            end
        end
    end
    error("Result $mod_name.$result_name not found")
end

"""
    get_result(sim::Sim, result_name::Symbol) → Vector{Float64}

Get a sim-level result.
"""
function get_result(sim::Sim, result_name::Symbol)
    if haskey(sim.results, result_name)
        return sim.results[result_name].values
    end
    # Search all modules
    for (_, mod) in all_modules(sim)
        res = module_results(mod)
        if haskey(res, result_name)
            return res[result_name].values
        end
    end
    error("Result $result_name not found")
end

export to_dataframe, get_result

# ============================================================================
# Convenience: demo
# ============================================================================

"""
    demo(; n_agents=1000, verbose=0, kwargs...)

Run a quick SIR demo simulation. Equivalent to Python's `ss.demo()`.

# Example
```julia
sim = demo()
```
"""
function demo(; n_agents::Int=1000, verbose::Int=0, kwargs...)
    n_contacts = 10
    beta = 0.5 / n_contacts
    sim = Sim(
        n_agents = n_agents,
        networks = RandomNet(n_contacts=n_contacts),
        diseases = SIR(beta=beta, dur_inf=4.0, init_prev=0.01),
        stop = 40.0,
        verbose = verbose;
        kwargs...
    )
    run!(sim)
    return sim
end

export demo

# ============================================================================
# Save / Load — serialization
# ============================================================================

"""
    save_sim(filename::AbstractString, sim::Sim)

Save a simulation to disk using Julia's `Serialization` module.
Matches Python starsim's `sim.save()` (which uses gzipped pickle).

# Arguments
- `filename` — path to save to (`.jls` extension recommended)
- `sim` — the Sim object to save

# Example
```julia
sim = Sim(diseases=SIR(beta=0.05))
run!(sim)
save_sim("my_sim.jls", sim)
```
"""
function save_sim(filename::AbstractString, sim::Sim)
    open(filename, "w") do io
        Serialization.serialize(io, sim)
    end
    return filename
end

"""
    load_sim(filename::AbstractString) → Sim

Load a simulation from disk. Inverse of [`save_sim`](@ref).

# Example
```julia
sim = load_sim("my_sim.jls")
```
"""
function load_sim(filename::AbstractString)
    sim = open(filename, "r") do io
        Serialization.deserialize(io)
    end
    sim isa Sim || error("Loaded object is not a Sim (got $(typeof(sim)))")
    return sim
end

"""
    to_json(sim::Sim; keys=[:pars, :results])

Export simulation parameters and/or results as a JSON-compatible `Dict`.
Matches Python starsim's `sim.to_json()`.

# Arguments
- `keys` — which sections to include: `:pars`, `:results`, or both

# Returns
A `Dict{String, Any}` suitable for JSON serialization.

# Example
```julia
sim = Sim(diseases=SIR(beta=0.05)); run!(sim)
d = to_json(sim)                   # Dict with pars + results
to_json(sim; filename="sim.json")  # write to file
```
"""
function to_json(sim::Sim; keys::Vector{Symbol}=[:pars, :results],
                 filename::Union{Nothing, AbstractString}=nothing)
    d = Dict{String, Any}()

    if :pars in keys
        p = sim.pars
        d["pars"] = Dict{String, Any}(
            "n_agents"   => p.n_agents,
            "start"      => p.start,
            "stop"       => p.stop,
            "dt"         => p.dt,
            "rand_seed"  => p.rand_seed,
            "pop_scale"  => p.pop_scale,
            "use_aging"  => p.use_aging,
            "verbose"    => p.verbose,
        )
    end

    if :results in keys && sim.complete
        res = Dict{String, Any}()
        timevec = [sim.pars.start + (t - 1) * sim.pars.dt for t in 1:sim.t.npts]
        res["time"] = timevec

        for (mod_name, mod) in all_modules(sim)
            mr = module_results(mod)
            for (rname, result) in mr.data
                k = "$(mod_name)_$(rname)"
                n = min(length(timevec), length(result.values))
                res[k] = result.values[1:n]
            end
        end
        d["results"] = res
    end

    if filename !== nothing
        open(filename, "w") do io
            JSON3.write(io, d)
        end
    end

    return d
end

export save_sim, load_sim, to_json

# ============================================================================
# summarize — summary statistics
# ============================================================================

"""
    summarize(sim::Sim)

Compute summary statistics for all results. Matches Python starsim's
`sim.summarize()`.

For each result, the aggregation method depends on the result name prefix:
- `n_*` → mean over time
- `new_*` → mean over time
- `cum_*` → last value
- `prevalence` → mean over time
- otherwise → mean over time

Returns an `OrderedDict{Symbol, Float64}`.

# Example
```julia
sim = Sim(diseases=SIR(beta=0.05)); run!(sim)
summary = summarize(sim)
```
"""
function summarize(sim::Sim)
    sim.complete || error("Simulation must be complete before summarizing")
    summary = OrderedDict{Symbol, Float64}()

    # Sim-level results
    for (name, r) in sim.results.data
        summary[name] = _summarize_result(r)
    end

    # Module results
    for (mod_name, mod) in all_modules(sim)
        for (res_name, r) in module_results(mod).data
            key = Symbol("$(mod_name)_$(res_name)")
            summary[key] = _summarize_result(r)
        end
    end

    return summary
end

function _summarize_result(r::Result)
    vals = r.values
    isempty(vals) && return NaN
    name = string(r.name)
    if startswith(name, "cum_")
        return vals[end]
    elseif startswith(name, "n_") || startswith(name, "new_") || name == "prevalence"
        return mean(vals)
    else
        return mean(vals)
    end
end

export summarize

# ============================================================================
# copy — deep copy
# ============================================================================

"""
    Base.copy(sim::Sim) → Sim

Create a deep copy of a simulation. Equivalent to Python's `sim.copy()`.
"""
Base.copy(sim::Sim) = deepcopy(sim)

# ============================================================================
# shrink! — reduce memory footprint
# ============================================================================

"""
    shrink!(sim::Sim; in_place::Bool=true)

Remove bulky data (people arrays, loop plan) to reduce memory usage.
Matches Python starsim's `sim.shrink()`.

After shrinking, the simulation cannot be re-run or have its people
queried, but results are preserved.

If `in_place=false`, returns a shrunken copy instead of modifying `sim`.

# Example
```julia
sim = Sim(diseases=SIR()); run!(sim)
shrink!(sim)
```
"""
function shrink!(sim::Sim; in_place::Bool=true)
    s = in_place ? sim : deepcopy(sim)

    # Clear people states (keep minimal metadata)
    s.people = People(s.pars.n_agents; slot_scale=s.people.slot_scale)

    # Clear loop plan
    s.loop = Loop()

    # Clear module states and distributions but keep results
    for (_, mod) in all_modules(s)
        md = module_data(mod)
        empty!(md.states)
        empty!(md.dists)
    end

    return s
end

export shrink!

# ============================================================================
# Simulation comparison utilities
# ============================================================================

"""
    diff_sims(sim1::Sim, sim2::Sim; keys=nothing) → Dict{Symbol, NamedTuple}

Compare two completed simulations and return a `Dict` mapping each shared
result key to `(max_diff=…, mean_diff=…)`. If `keys` is provided, only
those result keys are compared.

# Example
```julia
d = diff_sims(sim1, sim2)
d[:sir_prevalence].max_diff
```
"""
function diff_sims(sim1::Sim, sim2::Sim; keys=nothing)
    sim1.complete || error("sim1 is not complete")
    sim2.complete || error("sim2 is not complete")

    s1 = summarize_all_results(sim1)
    s2 = summarize_all_results(sim2)

    common = keys === nothing ? intersect(Base.keys(s1), Base.keys(s2)) : keys
    diffs = Dict{Symbol, NamedTuple{(:max_diff, :mean_diff), Tuple{Float64, Float64}}}()
    for k in common
        v1 = s1[k]
        v2 = s2[k]
        n = min(length(v1), length(v2))
        d = abs.(v1[1:n] .- v2[1:n])
        diffs[k] = (max_diff=maximum(d), mean_diff=mean(d))
    end
    return diffs
end

"""
    check_sims_match(sim1::Sim, sim2::Sim; rtol=0.01, atol=1e-8) → Bool

Check if two simulations match within tolerance. Prints mismatches and
returns `true` if all shared results match.
"""
function check_sims_match(sim1::Sim, sim2::Sim; rtol::Real=0.01, atol::Real=1e-8)
    diffs = diff_sims(sim1, sim2)
    all_match = true
    for (k, d) in diffs
        s1 = summarize_all_results(sim1)
        scale = max(maximum(abs.(s1[k])), 1e-16)
        rel = d.max_diff / scale
        if d.max_diff > atol && rel > rtol
            @warn "Mismatch in $k: max_diff=$(d.max_diff), rel=$(rel)"
            all_match = false
        end
    end
    return all_match
end

"""Collect all result vectors from a simulation keyed by `module_result` symbol."""
function summarize_all_results(sim::Sim)
    out = OrderedDict{Symbol, Vector{Float64}}()
    for (name, r) in sim.results.data
        out[name] = r.values
    end
    for (mod_name, mod) in all_modules(sim)
        for (res_name, r) in module_results(mod).data
            out[Symbol("$(mod_name)_$(res_name)")] = r.values
        end
    end
    return out
end

"""
    mock_sim(; n_agents=100, kwargs...) → Sim

Create a minimal SIR simulation for testing purposes.

# Example
```julia
sim = mock_sim()
run!(sim; verbose=0)
```
"""
function mock_sim(; n_agents::Int=100, kwargs...)
    Sim(; n_agents=n_agents, diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.1),
        networks=RandomNet(n_contacts=4), start=0.0, stop=10.0, rand_seed=1, kwargs...)
end

export diff_sims, check_sims_match, mock_sim
