"""
Analyzers for Starsim.jl.

Mirrors Python starsim's analyzer classes. Analyzers run each timestep
to collect custom metrics.
"""

"""
    AnalyzerData

Common mutable data for analyzer modules.
"""
mutable struct AnalyzerData
    mod::ModuleData
end

"""
    analyzer_data(a::AbstractAnalyzer) → AnalyzerData

Return the AnalyzerData. Concrete analyzers must implement this.
"""
function analyzer_data end

module_data(a::AbstractAnalyzer) = analyzer_data(a).mod

export AnalyzerData, analyzer_data

# ============================================================================
# FunctionAnalyzer — user-provided function
# ============================================================================

"""
    FunctionAnalyzer <: AbstractAnalyzer

Analyzer that runs a user-provided function each timestep.

# Keyword arguments
- `name::Symbol` — analyzer name (default `:func_analyzer`)
- `fn::Function` — function(sim, ti) to call each timestep

# Example
```julia
tracker = FunctionAnalyzer(fn = (sim, ti) -> begin
    println("Step \$ti: n_alive = \$(length(sim.people.auids))")
end)
```
"""
mutable struct FunctionAnalyzer <: AbstractAnalyzer
    data::AnalyzerData
    fn::Function
    collected::Vector{Any}
end

function FunctionAnalyzer(;
    name::Symbol = :func_analyzer,
    fn::Function = (sim, ti) -> nothing,
)
    md = ModuleData(name; label="Function analyzer")
    ad = AnalyzerData(md)
    FunctionAnalyzer(ad, fn, Any[])
end

analyzer_data(fa::FunctionAnalyzer) = fa.data

function init_pre!(fa::FunctionAnalyzer, sim)
    md = module_data(fa)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return fa
end

function step!(fa::FunctionAnalyzer, sim)
    ti = module_data(fa).t.ti
    result = fa.fn(sim, ti)
    if result !== nothing
        push!(fa.collected, result)
    end
    return fa
end

export FunctionAnalyzer

# ============================================================================
# Snapshot — captures sim state at specified times
# ============================================================================

"""
    Snapshot <: AbstractAnalyzer

Captures copies of simulation state at specified timepoints.

# Keyword arguments
- `name::Symbol` — analyzer name (default `:snapshot`)
- `years::Vector{Float64}` — years to capture (default: empty = capture every step)
"""
mutable struct Snapshot <: AbstractAnalyzer
    data::AnalyzerData
    years::Vector{Float64}
    snapshots::OrderedDict{Float64, Dict{Symbol, Any}}
end

function Snapshot(;
    name::Symbol = :snapshot,
    years::Vector{Float64} = Float64[],
)
    md = ModuleData(name; label="Snapshot")
    ad = AnalyzerData(md)
    Snapshot(ad, years, OrderedDict{Float64, Dict{Symbol, Any}}())
end

analyzer_data(s::Snapshot) = s.data

function init_pre!(s::Snapshot, sim)
    md = module_data(s)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return s
end

function step!(s::Snapshot, sim)
    md = module_data(s)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt

    should_capture = isempty(s.years) || any(y -> abs(year - y) < sim.pars.dt / 2, s.years)

    if should_capture
        snap = Dict{Symbol, Any}(
            :year => year,
            :ti => ti,
            :n_alive => length(sim.people.auids),
            :n_agents => sim.people.next_uid - 1,
        )
        s.snapshots[year] = snap
    end

    return s
end

export Snapshot

# ============================================================================
# InfectionLog — transmission chain tracking
# ============================================================================

"""
    TransmissionEvent

A single transmission event recorded in the infection log.

# Fields
- `source::Int` — UID of infecting agent (0 for seed infections)
- `target::Int` — UID of infected agent
- `t::Int` — timestep of infection
- `data::Dict{Symbol, Any}` — optional extra data (network, etc.)
"""
struct TransmissionEvent
    source::Int
    target::Int
    t::Int
    data::Dict{Symbol, Any}
end

TransmissionEvent(source::Int, target::Int, t::Int) =
    TransmissionEvent(source, target, t, Dict{Symbol, Any}())

"""
    InfectionLog <: AbstractAnalyzer

Record transmission chains for each disease. Matches Python starsim's
`infection_log` analyzer — activates infection logging in each disease
and collects the results after the simulation completes.

The log stores a `Vector{TransmissionEvent}` per disease and builds
a `SimpleDiGraph` (from Graphs.jl) of the transmission tree.

# Example
```julia
sim = Sim(
    n_agents = 1000,
    diseases = SIR(beta=0.1, init_prev=0.05),
    networks = RandomNet(n_contacts=10),
    analyzers = [InfectionLog()],
    stop = 30.0, verbose = 0,
)
run!(sim)
log = sim.analyzers[:infection_log]
log.events[:sir]   # Vector{TransmissionEvent}
log.graph[:sir]    # SimpleDiGraph
to_dataframe(log)  # DataFrame with source, target, t columns
```
"""
mutable struct InfectionLog <: AbstractAnalyzer
    data::AnalyzerData
    events::OrderedDict{Symbol, Vector{TransmissionEvent}}
    graph::OrderedDict{Symbol, SimpleDiGraph{Int}}
end

function InfectionLog(; name::Symbol=:infection_log)
    md = ModuleData(name; label="Infection log")
    ad = AnalyzerData(md)
    InfectionLog(ad,
        OrderedDict{Symbol, Vector{TransmissionEvent}}(),
        OrderedDict{Symbol, SimpleDiGraph{Int}}())
end

analyzer_data(il::InfectionLog) = il.data

function init_pre!(il::InfectionLog, sim)
    md = module_data(il)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return il
end

function step!(il::InfectionLog, sim)
    return il
end

"""
    finalize!(il::InfectionLog, sim)

Collect infection sources from each disease after simulation completes
and build transmission graphs.
"""
function finalize!(il::InfectionLog, sim)
    for (dis_name, dis) in sim.diseases
        dis isa AbstractInfection || continue
        sources = dis.infection.infection_sources

        events = Vector{TransmissionEvent}(undef, length(sources))
        @inbounds for i in eachindex(sources)
            target, source, t = sources[i]
            events[i] = TransmissionEvent(source, target, t)
        end

        il.events[dis_name] = events

        # Build DiGraph — nodes are UIDs, edges are transmissions
        max_uid = sim.people.next_uid - 1
        g = SimpleDiGraph(max_uid)
        for ev in events
            ev.source > 0 && add_edge!(g, ev.source, ev.target)
        end
        il.graph[dis_name] = g
    end

    return il
end

"""
    to_dataframe(il::InfectionLog; disease::Union{Nothing, Symbol}=nothing)

Convert infection log to a DataFrame. If `disease` is specified, return only
that disease's log; otherwise concatenate all diseases.
"""
function to_dataframe(il::InfectionLog; disease::Union{Nothing, Symbol}=nothing)
    dfs = DataFrame[]
    for (dname, events) in il.events
        disease !== nothing && dname != disease && continue
        isempty(events) && continue
        df = DataFrame(
            disease  = fill(dname, length(events)),
            source   = [ev.source for ev in events],
            target   = [ev.target for ev in events],
            t        = [ev.t for ev in events],
        )
        push!(dfs, df)
    end
    isempty(dfs) && return DataFrame(disease=Symbol[], source=Int[], target=Int[], t=Int[])
    result = vcat(dfs...)
    sort!(result, [:t, :source, :target])
    return result
end

export InfectionLog, TransmissionEvent
