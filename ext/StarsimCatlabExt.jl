"""
    StarsimCatlabExt

Category-theory-based composition for Starsim.jl using Catlab.jl.

Provides an ACSet schema for epidemiological models (states and transitions),
structured cospans (open epi-nets) for modular composition, and undirected
wiring diagram (UWD) composition following the CategoricalProjectionModels.jl
pattern.

# Key types

- `EpiNetData`: ACSet representing disease states and transitions
- `EpiSharerData`: Wraps a Starsim module exposing shared states as ports
- `compose_epi`: Composes modules via UWD, produces a configured `Sim`

# Example

```julia
using Starsim, Catlab

# Define modules as sharers
sir_sharer = EpiSharer(:sir,
    SIR(beta=0.05),
    ports=[:alive, :susceptible],
)
net_sharer = EpiSharer(:network,
    RandomNet(n_contacts=10),
    ports=[:alive],
)
demo_sharer = EpiSharer(:demo,
    [Births(birth_rate=20.0), Deaths(death_rate=15.0)],
    ports=[:alive],
)

# Compose via UWD
sim = compose_epi(
    [sir_sharer, net_sharer, demo_sharer];
    n_agents=5000, start=0.0, stop=50.0, dt=1.0,
)
run!(sim)
```
"""
module StarsimCatlabExt

using Starsim
using Catlab

# ============================================================================
# ACSet schema for epidemiological transition nets
# ============================================================================

"""
Define the schema for an epidemiological network (EpiNet).

Objects:
- `S` — compartmental states (Susceptible, Infected, Recovered, etc.)
- `T` — transitions between states (infection, recovery, death, etc.)

Morphisms:
- `src: T → S` — source state of a transition
- `tgt: T → S` — target state of a transition

Attributes:
- `sname: S → Name` — state name (Symbol)
- `tname: T → Name` — transition name (Symbol)
"""
@present SchEpiNet(FreeSchema) begin
    S::Ob
    T::Ob
    src::Hom(T, S)
    tgt::Hom(T, S)
    Name::AttrType
    sname::Attr(S, Name)
    tname::Attr(T, Name)
end

@acset_type EpiNetACSet(SchEpiNet, index=[:src, :tgt])

"""
    EpiNetData(states, transitions)

Construct an ACSet for an epidemiological network.

# Arguments
- `states::Vector{Symbol}` — names of compartmental states
- `transitions::Vector{Pair{Symbol, Pair{Symbol, Symbol}}}` — named transitions as
  `name => (source => target)` pairs

# Example
```julia
sir = EpiNetData(
    [:S, :I, :R],
    [:infection => (:S => :I), :recovery => (:I => :R)],
)
```
"""
function EpiNetData(states::Vector{Symbol}, transitions::Vector{<:Pair})
    net = EpiNetACSet{Symbol}()
    state_ids = Dict{Symbol, Int}()
    for s in states
        id = add_part!(net, :S; sname=s)
        state_ids[s] = id
    end
    for (tname, (s, t)) in transitions
        haskey(state_ids, s) || error("Unknown source state :$s in transition :$tname")
        haskey(state_ids, t) || error("Unknown target state :$t in transition :$tname")
        add_part!(net, :T; tname=tname, src=state_ids[s], tgt=state_ids[t])
    end
    return net
end

function Starsim.EpiNet(states::Vector{Symbol}, transitions::Vector{<:Pair})
    return EpiNetData(states, transitions)
end

# ============================================================================
# EpiSharer — wraps a Starsim module with ports for composition
# ============================================================================

"""
    EpiSharerData

An undirected open system wrapping one or more Starsim modules.

Following the `ProjectionSharer` pattern from CategoricalProjectionModels.jl,
each sharer exposes a set of ports (shared state names). When composed via
`oapply`, sharers connected at the same UWD junction share access to the
same underlying People object and its states.

# Fields
- `name::Symbol` — identifier for this sharer
- `modules::Vector{Starsim.AbstractModule}` — wrapped Starsim modules
- `ports::Vector{Symbol}` — names of shared states exposed as ports
- `category::Symbol` — module category (:disease, :network, :demographics, :connector, :intervention, :analyzer)
- `epinet::Union{EpiNetACSet, Nothing}` — optional transition diagram
"""
struct EpiSharerData
    name::Symbol
    modules::Vector{Starsim.AbstractModule}
    ports::Vector{Symbol}
    category::Symbol
    epinet::Union{EpiNetACSet{Symbol}, Nothing}
end

function _infer_category(mod::Starsim.AbstractModule)
    mod isa Starsim.AbstractDisease && return :disease
    mod isa Starsim.AbstractRoute && return :network
    mod isa Starsim.AbstractDemographics && return :demographics
    mod isa Starsim.AbstractConnector && return :connector
    mod isa Starsim.AbstractIntervention && return :intervention
    mod isa Starsim.AbstractAnalyzer && return :analyzer
    return :module
end

function _default_ports(mod::Starsim.AbstractModule)
    mod isa Starsim.AbstractDisease && return [:alive, :susceptible]
    mod isa Starsim.AbstractRoute && return [:alive]
    mod isa Starsim.AbstractDemographics && return [:alive]
    mod isa Starsim.AbstractConnector && return [:alive, :susceptible]
    mod isa Starsim.AbstractIntervention && return [:alive]
    mod isa Starsim.AbstractAnalyzer && return [:alive]
    return [:alive]
end

function _disease_epinet(mod::Starsim.AbstractInfection)
    name = Starsim.module_name(mod)
    if mod isa Starsim.SIR
        return EpiNetData(
            [:S, :I, :R],
            [:infection => (:S => :I), :recovery => (:I => :R)],
        )
    elseif mod isa Starsim.SIS
        return EpiNetData(
            [:S, :I],
            [:infection => (:S => :I), :recovery => (:I => :S)],
        )
    else
        return nothing
    end
end
_disease_epinet(mod::Starsim.AbstractModule) = nothing

"""
    Starsim.EpiSharer(name, module_or_modules; ports, epinet)

Create an EpiSharer wrapping Starsim module(s) for categorical composition.

# Arguments
- `name::Symbol` — identifier
- `mod` — a single `AbstractModule`, vector of modules, or vector of mixed types
- `ports::Vector{Symbol}` — shared state names (default: auto-inferred)
- `epinet` — optional transition diagram

# Example
```julia
sharer = EpiSharer(:sir, SIR(beta=0.05); ports=[:alive, :susceptible])
```
"""
function Starsim.EpiSharer(name::Symbol, mod::Starsim.AbstractModule;
                           ports::Vector{Symbol}=_default_ports(mod),
                           epinet=_disease_epinet(mod))
    cat = _infer_category(mod)
    EpiSharerData(name, [mod], ports, cat, epinet)
end

function Starsim.EpiSharer(name::Symbol, mods::Vector{<:Starsim.AbstractModule};
                           ports::Vector{Symbol}=Symbol[],
                           epinet=nothing)
    if isempty(ports)
        all_ports = Symbol[]
        for m in mods
            append!(all_ports, _default_ports(m))
        end
        ports = unique(all_ports)
    end
    cat = isempty(mods) ? :module : _infer_category(first(mods))
    EpiSharerData(name, mods, ports, cat, epinet)
end

# ============================================================================
# Composition via UWDs
# ============================================================================

"""
    Starsim.epi_uwd(sharers::Vector{EpiSharerData})

Build an undirected wiring diagram (UWD) that connects sharers at shared ports.

Junctions are created for each unique port name. Sharers that expose the same
port name are automatically wired to the same junction, ensuring they operate
on the same shared agent state.

Returns a `Catlab.WiringDiagrams.UndirectedWiringDiagram`.
"""
function Starsim.epi_uwd(sharers::Vector{EpiSharerData})
    # Collect all unique junction names
    all_ports = Symbol[]
    for s in sharers
        append!(all_ports, s.ports)
    end
    junction_names = unique(all_ports)
    junction_idx = Dict(name => i for (i, name) in enumerate(junction_names))

    n_junctions = length(junction_names)

    # Build UWD: start with 0 outer ports, add junctions manually
    uwd = UndirectedWiringDiagram(0)

    # Add junctions
    for _ in 1:n_junctions
        add_part!(uwd, :Junction)
    end

    # Track cumulative port count for set_junction! indexing
    port_offset = 0
    for (i, s) in enumerate(sharers)
        n_ports = length(s.ports)
        box_id = add_box!(uwd, n_ports)
        for (p, port_name) in enumerate(s.ports)
            j = junction_idx[port_name]
            set_junction!(uwd, port_offset + p, j)
        end
        port_offset += n_ports
    end

    # Expose all junctions as outer ports
    for j in 1:n_junctions
        add_part!(uwd, :OuterPort; outer_junction=j)
    end

    return uwd
end

# ============================================================================
# Compose into a Sim
# ============================================================================

"""
    Starsim.compose_epi(sharers; n_agents, start, stop, dt, rand_seed, kwargs...)

Compose EpiSharers into a fully configured `Sim`.

This is the main composition function. It:
1. Builds a UWD connecting sharers at shared ports
2. Validates that shared ports are compatible
3. Extracts all modules from sharers
4. Constructs a `Sim` with the modules properly wired

The composition is *additive*: modules at the same junction contribute
their effects to the same shared agent states (following the operadic
algebra from CategoricalProjectionModels.jl).

# Arguments
- `sharers::Vector{EpiSharerData}` — modules to compose
- `n_agents::Int` — number of agents
- `start, stop, dt` — simulation time parameters
- `rand_seed::Int` — random seed

# Returns
A configured `Starsim.Sim` ready for `run!`.

# Example
```julia
sim = compose_epi([sir_sharer, net_sharer, demo_sharer]; n_agents=5000)
run!(sim)
```
"""
function Starsim.compose_epi(sharers::Vector{EpiSharerData};
                             n_agents::Int=10_000,
                             start::Real=0.0,
                             stop::Real=10.0,
                             dt::Real=1.0,
                             rand_seed::Int=0,
                             verbose::Int=0,
                             kwargs...)
    # Build UWD for validation and documentation
    uwd = Starsim.epi_uwd(sharers)

    # Validate composition: all sharers with same port must be compatible
    _validate_composition(sharers, uwd)

    # Extract modules by category
    diseases = Starsim.AbstractDisease[]
    networks = Starsim.AbstractRoute[]
    demographics = Starsim.AbstractDemographics[]
    connectors = Starsim.AbstractConnector[]
    interventions = Starsim.AbstractIntervention[]
    analyzers = Starsim.AbstractAnalyzer[]

    for s in sharers
        for mod in s.modules
            if mod isa Starsim.AbstractDisease
                push!(diseases, mod)
            elseif mod isa Starsim.AbstractRoute
                push!(networks, mod)
            elseif mod isa Starsim.AbstractDemographics
                push!(demographics, mod)
            elseif mod isa Starsim.AbstractConnector
                push!(connectors, mod)
            elseif mod isa Starsim.AbstractIntervention
                push!(interventions, mod)
            elseif mod isa Starsim.AbstractAnalyzer
                push!(analyzers, mod)
            end
        end
    end

    # Build Sim with extracted modules
    sim = Starsim.Sim(;
        n_agents=n_agents,
        start=start,
        stop=stop,
        dt=dt,
        rand_seed=rand_seed,
        verbose=verbose,
        diseases=isempty(diseases) ? nothing : (length(diseases) == 1 ? diseases[1] : diseases),
        networks=isempty(networks) ? nothing : (length(networks) == 1 ? networks[1] : networks),
        demographics=isempty(demographics) ? nothing : (length(demographics) == 1 ? demographics[1] : demographics),
        connectors=isempty(connectors) ? nothing : (length(connectors) == 1 ? connectors[1] : connectors),
        interventions=isempty(interventions) ? nothing : (length(interventions) == 1 ? interventions[1] : interventions),
        analyzers=isempty(analyzers) ? nothing : (length(analyzers) == 1 ? analyzers[1] : analyzers),
        kwargs...,
    )

    return sim
end

"""
    _validate_composition(sharers, uwd)

Validate that the composition is well-formed:
- All sharers must have at least one port
- Sharers at the same junction must be compatible (e.g., disease + network sharing :alive)
- Disease sharers must have a network sharer sharing :alive
"""
function _validate_composition(sharers::Vector{EpiSharerData}, uwd)
    # Check all sharers have ports
    for s in sharers
        isempty(s.ports) && error("Sharer :$(s.name) has no ports — cannot compose")
    end

    # Check that disease sharers have a corresponding network
    has_disease = any(s -> s.category == :disease, sharers)
    has_network = any(s -> s.category == :network, sharers)

    if has_disease && !has_network
        @warn "Composition has disease module(s) but no network — transmission will not occur"
    end

    # Validate shared port compatibility
    port_owners = Dict{Symbol, Vector{Symbol}}()  # port => [sharer names]
    for s in sharers
        for p in s.ports
            owners = get!(port_owners, p, Symbol[])
            push!(owners, s.name)
        end
    end

    # :alive must be shared by all modules (it's the population)
    if haskey(port_owners, :alive)
        n_alive = length(port_owners[:alive])
        if n_alive < length(sharers)
            non_alive = [s.name for s in sharers if :alive ∉ s.ports]
            @warn "Modules $(non_alive) do not share :alive port — they may not interact with the population"
        end
    end

    return nothing
end

# ============================================================================
# Open EpiNet — structured cospans for hierarchical composition
# ============================================================================

"""
    Starsim.OpenEpiNet(net::EpiNetACSet, legs::Vector{Vector{Int}})

Create an open epidemiological network as a structured cospan.

Legs specify which states are exposed as ports for composition.
When two OpenEpiNets are composed, states at identified legs are merged.

# Arguments
- `net::EpiNetACSet` — the epi-net ACSet
- `legs::Vector{Vector{Int}}` — vectors of state indices forming each leg

# Example
```julia
sir_net = EpiNetData([:S, :I, :R], [:inf => (:S => :I), :rec => (:I => :R)])
open_sir = OpenEpiNet(sir_net, [[1]])  # Expose S as a port
```
"""
function Starsim.OpenEpiNet(net::EpiNetACSet{Symbol}, legs::Vector{Vector{Int}})
    n_states = nparts(net, :S)
    cospan_legs = [FinFunction(leg, n_states) for leg in legs]
    return (net=net, legs=cospan_legs)
end

# ============================================================================
# Conversion: EpiNet → Sim components
# ============================================================================

"""
    Starsim.to_sim(net::EpiNetACSet; n_agents, beta, dur_inf, kwargs...)

Convert an EpiNetACSet transition diagram into a Starsim Sim.

Automatically detects standard patterns (SIR, SIS, SEIR) and creates
the corresponding Starsim disease model.

# Example
```julia
net = EpiNet([:S, :I, :R], [:infection => (:S => :I), :recovery => (:I => :R)])
sim = to_sim(net; n_agents=1000, beta=0.1, networks=RandomNet(n_contacts=10))
```
"""
function Starsim.to_sim(net::EpiNetACSet{Symbol};
                        n_agents::Int=10_000,
                        beta::Real=0.05,
                        dur_inf::Real=10.0,
                        dur_exp::Real=5.0,
                        networks=nothing,
                        kwargs...)
    states = [subpart(net, i, :sname) for i in 1:nparts(net, :S)]
    transitions = [subpart(net, i, :tname) for i in 1:nparts(net, :T)]
    srcs = [subpart(net, i, :src) for i in 1:nparts(net, :T)]
    tgts = [subpart(net, i, :tgt) for i in 1:nparts(net, :T)]

    state_names = Set(states)

    # Detect SIR pattern: S→I→R
    if state_names == Set([:S, :I, :R])
        disease = Starsim.SIR(; beta=beta, dur_inf=dur_inf)
    elseif state_names == Set([:S, :I])
        disease = Starsim.SIS(; beta=beta, dur_inf=dur_inf)
    elseif state_names == Set([:S, :E, :I, :R])
        disease = Starsim.SEIR(; beta=beta, dur_inf=dur_inf, dur_exp=dur_exp)
    else
        error("Cannot auto-detect disease type from states: $state_names. " *
              "Use compose_epi() for custom models.")
    end

    return Starsim.Sim(;
        n_agents=n_agents,
        diseases=disease,
        networks=networks,
        kwargs...,
    )
end

# ============================================================================
# Pretty printing
# ============================================================================

function Base.show(io::IO, s::EpiSharerData)
    n_mods = length(s.modules)
    mod_str = n_mods == 1 ? "1 module" : "$n_mods modules"
    print(io, "EpiSharer(:$(s.name), $mod_str, ports=$(s.ports), category=:$(s.category))")
end

function Base.show(io::IO, ::MIME"text/plain", s::EpiSharerData)
    println(io, "EpiSharer :$(s.name)")
    println(io, "  Category: $(s.category)")
    println(io, "  Ports:    $(s.ports)")
    println(io, "  Modules:")
    for m in s.modules
        println(io, "    - $(Starsim.module_name(m)) ($(typeof(m)))")
    end
    if s.epinet !== nothing
        ns = nparts(s.epinet, :S)
        nt = nparts(s.epinet, :T)
        states = Symbol[subpart(s.epinet, i, :sname) for i in 1:ns]
        println(io, "  EpiNet:   $ns states $(states), $nt transitions")
    end
end

end # module StarsimCatlabExt
