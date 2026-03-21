"""
Contact networks for Starsim.jl.

Mirrors Python starsim's `networks.py`. Supports multiple storage backends:
- Parallel arrays (default, matches Python) — best for serial CPU
- Graphs.jl views — for graph algorithms (components, shortest paths)
- Sparse/dense adjacency matrices — for GPU and linear algebra

The transmission loop dispatches on the storage type, so users can
benchmark different backends without changing model code.
"""

# ============================================================================
# Edges — parallel array edge storage (default, matches Python)
# ============================================================================

"""
    Edges

Stores network edges as parallel arrays. Matches Python starsim's Edges
class with p1/p2/beta/acts arrays. This is the primary storage format.

# Fields
- `p1::Vector{Int}` — source agent UIDs
- `p2::Vector{Int}` — target agent UIDs
- `beta::Vector{Float64}` — per-edge transmission multiplier
- `acts::Vector{Float64}` — per-edge acts count for per-act compounding (default 1.0)
"""
mutable struct Edges
    p1::Vector{Int}
    p2::Vector{Int}
    beta::Vector{Float64}
    acts::Vector{Float64}
end

Edges() = Edges(Int[], Int[], Float64[], Float64[])

function Edges(p1::Vector{Int}, p2::Vector{Int})
    n = length(p1)
    Edges(p1, p2, ones(Float64, n), ones(Float64, n))
end

Base.length(e::Edges) = length(e.p1)
Base.isempty(e::Edges) = isempty(e.p1)
Base.show(io::IO, e::Edges) = print(io, "Edges(n=$(length(e)))")

"""Add edges with beta and acts."""
function add_edges!(e::Edges, p1::Vector{Int}, p2::Vector{Int},
                    beta::Vector{Float64}, acts::Vector{Float64})
    append!(e.p1, p1)
    append!(e.p2, p2)
    append!(e.beta, beta)
    append!(e.acts, acts)
    return e
end

"""Add edges with beta, default acts=1."""
function add_edges!(e::Edges, p1::Vector{Int}, p2::Vector{Int}, beta::Vector{Float64})
    add_edges!(e, p1, p2, beta, ones(Float64, length(p1)))
end

"""Add edges with default beta=1, acts=1."""
function add_edges!(e::Edges, p1::Vector{Int}, p2::Vector{Int})
    n = length(p1)
    add_edges!(e, p1, p2, ones(Float64, n), ones(Float64, n))
end

"""Remove edges at given indices."""
function remove_edges!(e::Edges, inds::Vector{Int})
    keep = setdiff(1:length(e), inds)
    e.p1 = e.p1[keep]
    e.p2 = e.p2[keep]
    e.beta = e.beta[keep]
    e.acts = e.acts[keep]
    return e
end

"""Clear all edges."""
function clear_edges!(e::Edges)
    empty!(e.p1)
    empty!(e.p2)
    empty!(e.beta)
    empty!(e.acts)
    return e
end

"""Find contact UIDs for a given set of source UIDs."""
function find_contacts(e::Edges, source_uids::UIDs)
    source_set = Set(source_uids.values)
    contacts = Set{Int}()
    for i in 1:length(e)
        if e.p1[i] in source_set
            push!(contacts, e.p2[i])
        end
        if e.p2[i] in source_set
            push!(contacts, e.p1[i])
        end
    end
    return UIDs(sort(collect(contacts)))
end

export Edges, add_edges!, remove_edges!, clear_edges!, find_contacts

# ============================================================================
# Graphs.jl interop — on-demand views
# ============================================================================

"""
    to_graph(e::Edges) -> SimpleGraph{Int}

Create a Graphs.jl `SimpleGraph` from edges. Useful for graph algorithms
(connected components, shortest paths, centrality, etc.).

This is a *copy*, not a live view — call again after edges change.

# Example
```julia
g = to_graph(network_edges(net))
components = connected_components(g)
degrees = degree(g)
```
"""
function to_graph(e::Edges)
    isempty(e) && return SimpleGraph(0)
    max_uid = max(maximum(e.p1), maximum(e.p2))
    g = SimpleGraph(max_uid)
    for i in 1:length(e)
        add_edge!(g, e.p1[i], e.p2[i])
    end
    return g
end

"""
    to_digraph(e::Edges) -> SimpleDiGraph{Int}

Create a directed graph from edges (p1 → p2).
"""
function to_digraph(e::Edges)
    isempty(e) && return SimpleDiGraph(0)
    max_uid = max(maximum(e.p1), maximum(e.p2))
    g = SimpleDiGraph(max_uid)
    for i in 1:length(e)
        add_edge!(g, e.p1[i], e.p2[i])
    end
    return g
end

"""Convenience: get contact degrees from edges via Graphs.jl."""
function contact_degrees(e::Edges)
    g = to_graph(e)
    return degree(g)
end

"""Convenience: connected components via Graphs.jl."""
function network_components(e::Edges)
    return connected_components(to_graph(e))
end

export to_graph, to_digraph, contact_degrees, network_components

# ============================================================================
# Matrix interop — sparse adjacency / weighted contact matrices
# ============================================================================

"""
    to_adjacency_matrix(e::Edges; n::Int=0, weighted::Bool=false) -> SparseMatrixCSC

Create a sparse adjacency matrix from edges. If `weighted`, entries are
the edge beta values; otherwise binary. Useful for GPU offload and
linear-algebra-based transmission.

# Example
```julia
A = to_adjacency_matrix(network_edges(net); weighted=true)
# GPU: A_gpu = Metal.MtlArray(Matrix(A))
```
"""
function to_adjacency_matrix(e::Edges; n::Int=0, weighted::Bool=false)
    isempty(e) && return spzeros(Float64, n, n)
    max_uid = n > 0 ? n : max(maximum(e.p1), maximum(e.p2))
    I = vcat(e.p1, e.p2)       # Symmetric
    J = vcat(e.p2, e.p1)
    V = weighted ? vcat(e.beta, e.beta) : ones(Float64, 2 * length(e))
    return sparse(I, J, V, max_uid, max_uid)
end

"""
    to_contact_matrix(e::Edges; n::Int=0) -> Matrix{Float64}

Create a dense contact matrix. Useful for small populations or GPU.
"""
function to_contact_matrix(e::Edges; n::Int=0)
    return Matrix(to_adjacency_matrix(e; n=n, weighted=true))
end

"""
    from_graph!(e::Edges, g::SimpleGraph; beta::Float64=1.0)

Populate edges from a Graphs.jl graph.
"""
function from_graph!(e::Edges, g::SimpleGraph; beta::Float64=1.0)
    clear_edges!(e)
    for edge in edges(g)
        push!(e.p1, src(edge))
        push!(e.p2, dst(edge))
        push!(e.beta, beta)
    end
    return e
end

export to_adjacency_matrix, to_contact_matrix, from_graph!

# ============================================================================
# Network base
# ============================================================================

"""
    NetworkData

Common mutable data for network modules.
"""
mutable struct NetworkData
    mod::ModuleData
    edges::Edges
    bidirectional::Bool
end

"""
    network_data(net::AbstractNetwork) -> NetworkData

Return the NetworkData for a network. Concrete networks must implement this.
"""
function network_data end

module_data(net::AbstractNetwork) = network_data(net).mod

"""
    network_edges(net::AbstractNetwork) -> Edges

Return the [`Edges`](@ref) container for network `net`.
"""
network_edges(net::AbstractNetwork) = network_data(net).edges

"""
    net_beta(net::AbstractNetwork, disease_beta::Float64)

Compute per-edge beta for transmission. Default: `edges.beta .* disease_beta`.
Override for sexual networks (acts-based).
"""
function net_beta(net::AbstractNetwork, disease_beta::Float64)
    return network_edges(net).beta .* disease_beta
end

export NetworkData, network_data, network_edges, net_beta

# ============================================================================
# RandomNet — matching Python ss.RandomNet
# ============================================================================

"""
    RandomNet <: AbstractNetwork

Random contact network. Matches Python `ss.RandomNet`: creates `n_contacts÷2`
edges per agent, with bidirectional transmission giving an effective
`n_contacts` contacts per agent.

# Note on n_contacts
From Python starsim: "n_contacts = 10 will create *5* edges per agent.
Since disease transmission usually occurs bidirectionally, this means
the effective number of contacts per agent is actually 10."

# Keyword arguments
- `name::Symbol` — network name (default `:random`)
- `n_contacts` — mean contacts per agent (default 10); `Int`, `Float64`, or `AbstractStarsimDist`

# Example
```julia
net = RandomNet(n_contacts=10)
```
"""
mutable struct RandomNet <: AbstractNetwork
    data::NetworkData
    n_contacts::Union{Int, Float64, AbstractStarsimDist}
    rng::StableRNG
end

function RandomNet(;
    name::Symbol = :random,
    n_contacts::Union{Int, Float64, AbstractStarsimDist} = 10,
)
    md = ModuleData(name; label="Random network")
    nd = NetworkData(md, Edges(), true)
    RandomNet(nd, n_contacts, StableRNG(0))
end

network_data(net::RandomNet) = net.data

function init_pre!(net::RandomNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    return net
end

"""
    update_edges!(net::RandomNet, people::People)

Regenerate random edges for the current timestep. Matches Python starsim's
approach: each agent produces `n_contacts÷2` half-edges, the full array
is shuffled to form (source, target) pairs. Transmission runs
bidirectionally over each edge, giving `n_contacts` effective contacts.

Self-loops are not removed (matching Python starsim). They are harmless
because an agent cannot be both infected and susceptible for the same disease.
"""
function update_edges!(net::RandomNet, people::People)
    edges = net.data.edges
    active = people.auids.values
    n::Int = length(active)
    n < 2 && (clear_edges!(edges); return net)

    nc_val::Int = net.n_contacts isa AbstractStarsimDist ?
        Int(round(mean(net.n_contacts.dist))) : Int(round(net.n_contacts))
    n_half::Int = max(1, nc_val ÷ 2)
    total::Int = n * n_half

    # Resize (no-op after first call since total is constant without demographics)
    resize!(edges.p1, total)
    resize!(edges.p2, total)
    resize!(edges.beta, total)

    # Build source directly into p1: each agent repeated n_half times
    @inbounds for k in 1:n
        a = active[k]
        base = (k - 1) * n_half
        for j in 1:n_half
            edges.p1[base + j] = a
        end
    end

    # Target (p2) = shuffled copy of source (matches Python's np.random.permutation)
    copyto!(edges.p2, edges.p1)
    shuffle!(net.rng, edges.p2)

    # Beta = 1.0 for all edges
    fill!(edges.beta, 1.0)

    return net
end

function step!(net::RandomNet, sim)
    update_edges!(net, sim.people)
    return net
end

export RandomNet, update_edges!

# ============================================================================
# MFNet — Male-Female sexual network
# ============================================================================

"""
    MFNet <: AbstractNetwork

Heterosexual partnership network (male-female pairs).

# Keyword arguments
- `name::Symbol` — default `:mf`
- `mean_dur::Float64` — mean partnership duration (default 5.0)
- `participation_rate::Float64` — fraction of adults in partnerships (default 0.5)
"""
mutable struct MFNet <: AbstractNetwork
    data::NetworkData
    mean_dur::Float64
    participation_rate::Float64
    rng::StableRNG
end

function MFNet(;
    name::Symbol = :mf,
    mean_dur::Real = 5.0,
    participation_rate::Real = 0.5,
    bidirectional::Bool = true,
)
    md = ModuleData(name; label="MF sexual network")
    nd = NetworkData(md, Edges(), bidirectional)
    MFNet(nd, Float64(mean_dur), Float64(participation_rate), StableRNG(0))
end

network_data(net::MFNet) = net.data

function init_pre!(net::MFNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    form_partnerships!(net, sim.people)
    return net
end

"""Form male-female partnerships."""
function form_partnerships!(net::MFNet, people::People)
    active = people.auids.values
    males = [u for u in active if !people.female.raw[u]]
    females = [u for u in active if people.female.raw[u]]

    n_pairs = min(
        Int(round(length(males) * net.participation_rate)),
        Int(round(length(females) * net.participation_rate))
    )

    if n_pairs > 0
        m_sample = males[randperm(net.rng, length(males))[1:min(n_pairs, length(males))]]
        f_sample = females[randperm(net.rng, length(females))[1:min(n_pairs, length(females))]]
        n_actual = min(length(m_sample), length(f_sample))
        add_edges!(net.data.edges, m_sample[1:n_actual], f_sample[1:n_actual])
    end
    return net
end

function step!(net::MFNet, sim)
    clear_edges!(net.data.edges)
    form_partnerships!(net, sim.people)
    return net
end

export MFNet, form_partnerships!

# ============================================================================
# MaternalNet — Mother-child network
# ============================================================================

"""
    MaternalNet <: AbstractNetwork

Mother-child contact network for vertical transmission.
"""
mutable struct MaternalNet <: AbstractNetwork
    data::NetworkData
end

function MaternalNet(; name::Symbol = :maternal)
    md = ModuleData(name; label="Maternal network")
    nd = NetworkData(md, Edges(), false)
    MaternalNet(nd)
end

network_data(net::MaternalNet) = net.data

export MaternalNet

# ============================================================================
# MixingPool — well-mixed population (non-network route)
# ============================================================================

"""
    MixingPool <: AbstractNetwork

Well-mixed population contact model. All agents can potentially
infect all others, weighted by a contact rate.

# Keyword arguments
- `name::Symbol` — default `:mixing`
- `contact_rate::Float64` — effective contacts per agent per dt (default 10.0)
"""
mutable struct MixingPool <: AbstractNetwork
    data::NetworkData
    contact_rate::Float64
    rng::StableRNG
end

function MixingPool(;
    name::Symbol = :mixing,
    contact_rate::Real = 10.0,
)
    md = ModuleData(name; label="Mixing pool")
    nd = NetworkData(md, Edges(), true)
    MixingPool(nd, Float64(contact_rate), StableRNG(0))
end

network_data(net::MixingPool) = net.data

function init_pre!(net::MixingPool, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    return net
end

function step!(net::MixingPool, sim)
    clear_edges!(net.data.edges)
    active = sim.people.auids.values
    n = length(active)
    n < 2 && return net

    n_contacts = Int(round(net.contact_rate * n / 2))
    p1 = [active[rand(net.rng, 1:n)] for _ in 1:n_contacts]
    p2 = [active[rand(net.rng, 1:n)] for _ in 1:n_contacts]
    add_edges!(net.data.edges, p1, p2)
    return net
end

export MixingPool

# ============================================================================
# StaticNet — fixed network from a Graphs.jl graph
# ============================================================================

"""
    StaticNet <: AbstractNetwork

Static network initialized from a Graphs.jl graph or generator function.
Edges are created once at initialization and do not change.

# Keyword arguments
- `name::Symbol` — default `:static`
- `graph_fn` — function `(n, rng) → SimpleGraph` that creates the network
- `n_contacts::Int` — mean contacts for Erdős-Rényi default (default 10)

# Example
```julia
# Erdős-Rényi
net = StaticNet(n_contacts=10)

# Custom graph generator
net = StaticNet(graph_fn=(n, rng) -> watts_strogatz(n, 4, 0.3))
```
"""
mutable struct StaticNet <: AbstractNetwork
    data::NetworkData
    graph_fn::Any
    n_contacts::Int
    rng::StableRNG
end

function StaticNet(;
    name::Symbol = :static,
    graph_fn = nothing,
    n_contacts::Int = 10,
)
    md = ModuleData(name; label="Static network")
    nd = NetworkData(md, Edges(), true)
    StaticNet(nd, graph_fn, n_contacts, StableRNG(0))
end

network_data(net::StaticNet) = net.data

function init_pre!(net::StaticNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    n = length(sim.people.auids)
    if net.graph_fn !== nothing
        g = net.graph_fn(n, net.rng)
    else
        # Default: Erdős-Rényi with target mean degree
        p_edge = net.n_contacts / max(1, n - 1)
        g = erdos_renyi(n, p_edge; seed=rand(net.rng, 1:typemax(Int32)))
    end

    from_graph!(net.data.edges, g)
    md.initialized = true
    return net
end

# Static networks don't update edges
function step!(net::StaticNet, sim)
    return net
end

export StaticNet

# ============================================================================
# MSMNet — Men who have sex with men network
# ============================================================================

"""
    MSMNet <: AbstractNetwork

Men-who-have-sex-with-men partnership network. Pairs only male agents.

# Keyword arguments
- `name::Symbol` — default `:msm`
- `participation_rate::Float64` — fraction of males in partnerships (default 0.3)
"""
mutable struct MSMNet <: AbstractNetwork
    data::NetworkData
    participation_rate::Float64
    rng::StableRNG
end

function MSMNet(;
    name::Symbol = :msm,
    participation_rate::Real = 0.3,
    bidirectional::Bool = true,
)
    md = ModuleData(name; label="MSM network")
    nd = NetworkData(md, Edges(), bidirectional)
    MSMNet(nd, Float64(participation_rate), StableRNG(0))
end

network_data(net::MSMNet) = net.data

function init_pre!(net::MSMNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    form_msm_partnerships!(net, sim.people)
    return net
end

"""Form male-male partnerships."""
function form_msm_partnerships!(net::MSMNet, people::People)
    active = people.auids.values
    males = [u for u in active if !people.female.raw[u]]

    n_pairs = Int(round(length(males) * net.participation_rate / 2))

    if n_pairs > 0 && length(males) >= 2
        perm = randperm(net.rng, length(males))
        n_actual = min(n_pairs, length(males) ÷ 2)
        p1 = males[perm[1:n_actual]]
        p2 = males[perm[n_actual+1:2*n_actual]]
        add_edges!(net.data.edges, p1, p2)
    end
    return net
end

function step!(net::MSMNet, sim)
    clear_edges!(net.data.edges)
    form_msm_partnerships!(net, sim.people)
    return net
end

export MSMNet, form_msm_partnerships!

# ============================================================================
# PrenatalNet — mother-to-unborn-child network
# ============================================================================

"""
    PrenatalNet <: AbstractNetwork

Network for prenatal (mother-to-unborn-child) transmission. Edges are managed
by the `Pregnancy` module — they are added when pregnancy starts and removed
at delivery. Matches Python starsim's `PrenatalNet`.

# Keyword arguments
- `name::Symbol` — default `:prenatal`
"""
mutable struct PrenatalNet <: AbstractNetwork
    data::NetworkData
end

function PrenatalNet(; name::Symbol = :prenatal)
    md = ModuleData(name; label="Prenatal network")
    nd = NetworkData(md, Edges(), false)
    PrenatalNet(nd)
end

network_data(net::PrenatalNet) = net.data

"""
    add_pairs!(net::PrenatalNet, mother_uids, unborn_uids)

Add edges between mothers and their unborn children. Called by the
Pregnancy module when pregnancies are initiated.
"""
function add_pairs!(net::PrenatalNet, mother_uids::Vector{Int}, unborn_uids::Vector{Int})
    n = length(mother_uids)
    n == 0 && return 0
    add_edges!(net.data.edges, mother_uids, unborn_uids)
    return n
end

function step!(net::PrenatalNet, sim)
    return net
end

export PrenatalNet, add_pairs!

# ============================================================================
# PostnatalNet — mother-to-infant network with duration
# ============================================================================

"""
    PostnatalNet <: AbstractNetwork

Network tracking postnatal contact between mothers and infants. Edges are
added at birth (via Pregnancy module) and persist for a specified duration,
after which they are automatically removed. Matches Python starsim's
`PostnatalNet`.

# Keyword arguments
- `name::Symbol` — default `:postnatal`
- `dur::Union{Nothing, Float64}` — edge duration in timesteps. `nothing` means edges persist indefinitely.
"""
mutable struct PostnatalNet <: AbstractNetwork
    data::NetworkData
    dur::Union{Nothing, Float64}
    edge_start::Vector{Float64}  # timestep when each edge was added
    rng::StableRNG
end

function PostnatalNet(;
    name::Symbol = :postnatal,
    dur::Union{Nothing, Real} = nothing,
)
    md = ModuleData(name; label="Postnatal network")
    nd = NetworkData(md, Edges(), false)
    PostnatalNet(nd, dur === nothing ? nothing : Float64(dur), Float64[], StableRNG(0))
end

network_data(net::PostnatalNet) = net.data

function init_pre!(net::PostnatalNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    return net
end

"""
    add_pairs!(net::PostnatalNet, mother_uids, infant_uids, ti)

Add edges between mothers and their newborns. Called by the Pregnancy
module at delivery.
"""
function add_pairs!(net::PostnatalNet, mother_uids::Vector{Int}, infant_uids::Vector{Int}, ti::Int)
    n = length(mother_uids)
    n == 0 && return 0
    add_edges!(net.data.edges, mother_uids, infant_uids)
    append!(net.edge_start, fill(Float64(ti), n))
    return n
end

"""Remove expired edges based on duration."""
function step!(net::PostnatalNet, sim)
    net.dur === nothing && return net
    ti = Float64(sim.loop.ti)
    edges = net.data.edges
    n = length(edges)
    n == 0 && return net

    # Keep only edges that haven't expired
    keep = Vector{Bool}(undef, n)
    dur = net.dur
    @inbounds for i in 1:n
        keep[i] = (ti - net.edge_start[i]) < dur
    end

    if !all(keep)
        edges.p1 = edges.p1[keep]
        edges.p2 = edges.p2[keep]
        edges.beta = edges.beta[keep]
        edges.acts = edges.acts[keep]
        net.edge_start = net.edge_start[keep]
    end

    return net
end

export PostnatalNet

# ============================================================================
# BreastfeedingNet — extends PostnatalNet for breastfeeding transmission
# ============================================================================

"""
    BreastfeedingNet <: AbstractNetwork

Network for breastfeeding transmission. Requires a `Pregnancy` module in the
simulation. Edges are added at birth if the mother is breastfeeding and
removed when breastfeeding ends. Matches Python starsim's `BreastfeedingNet`.

# Keyword arguments
- `name::Symbol` — default `:breastfeeding`
"""
mutable struct BreastfeedingNet <: AbstractNetwork
    data::NetworkData
    edge_start::Vector{Float64}
    rng::StableRNG
end

function BreastfeedingNet(; name::Symbol = :breastfeeding)
    md = ModuleData(name; label="Breastfeeding network")
    nd = NetworkData(md, Edges(), false)
    BreastfeedingNet(nd, Float64[], StableRNG(0))
end

network_data(net::BreastfeedingNet) = net.data

function init_pre!(net::BreastfeedingNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    return net
end

"""
    add_pairs!(net::BreastfeedingNet, mother_uids, infant_uids, ti)

Add breastfeeding edges. Called by the Pregnancy module at delivery.
"""
function add_pairs!(net::BreastfeedingNet, mother_uids::Vector{Int}, infant_uids::Vector{Int}, ti::Int)
    n = length(mother_uids)
    n == 0 && return 0
    add_edges!(net.data.edges, mother_uids, infant_uids)
    append!(net.edge_start, fill(Float64(ti), n))
    return n
end

function step!(net::BreastfeedingNet, sim)
    return net
end

export BreastfeedingNet

# ============================================================================
# HouseholdNet — household contact network
# ============================================================================

"""
    HouseholdNet <: AbstractNetwork

A household contact network. Agents are assigned to households, and all
members of the same household form symmetric pairwise contacts. Matches
Python starsim's `HouseholdNet`.

Household data is provided as a DataFrame with columns:
- `hh_id` — household identifier
- `ages` — comma-separated age strings (e.g. "72, 17, 30")
- `sexes` (optional) — comma-separated sex codes (1=male, 2=female)

# Keyword arguments
- `name::Symbol` — default `:household`
- `hh_data::Union{Nothing, DataFrame}` — household survey data
- `dynamic::Bool` — whether households evolve over time (default `false`)

# Example
```julia
using DataFrames
hh_data = DataFrame(
    hh_id = 1:3,
    ages = ["30, 5, 2", "45, 20", "60, 55, 30, 10"],
)
net = HouseholdNet(hh_data=hh_data)
sim = Sim(n_agents=100, networks=net, diseases=SIR(beta=0.1))
run!(sim)
```
"""
mutable struct HouseholdNet <: AbstractNetwork
    data::NetworkData
    hh_data::Union{Nothing, DataFrame}
    household_ids::Vector{Int}  # household ID per agent
    dynamic::Bool
    rng::StableRNG
end

function HouseholdNet(;
    name::Symbol = :household,
    hh_data::Union{Nothing, DataFrame} = nothing,
    dynamic::Bool = false,
)
    md = ModuleData(name; label="Household network")
    nd = NetworkData(md, Edges(), true)  # bidirectional
    HouseholdNet(nd, hh_data, Int[], dynamic, StableRNG(0))
end

network_data(net::HouseholdNet) = net.data

function init_pre!(net::HouseholdNet, sim)
    md = module_data(net)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true

    if net.hh_data !== nothing
        _assign_households!(net, sim.people)
    end

    return net
end

"""Assign agents to households from survey data and create pairwise contacts."""
function _assign_households!(net::HouseholdNet, people::People)
    hh = net.hh_data
    hh === nothing && error("HouseholdNet requires hh_data")
    n_rows = nrow(hh)
    n_rows > 0 || error("hh_data is empty")

    pop_size = length(people.auids.values)
    net.household_ids = zeros(Int, people.next_uid)  # indexed by UID

    p1_buf = Int[]
    p2_buf = Int[]
    n_remaining = pop_size
    cluster_id = 0
    uid_offset = 0

    has_sexes = hasproperty(hh, :sexes)

    while n_remaining > 0
        cluster_id += 1

        # Sample a random household from the data
        rand_row = rand(net.rng, 1:n_rows)
        age_str = string(hh[rand_row, :ages])
        ages = [parse(Float64, strip(s)) for s in split(age_str, ',')]
        cluster_size = min(length(ages), n_remaining)

        # Assign ages (and optionally sex) to agents
        for k in 1:cluster_size
            uid = uid_offset + k
            people.age.raw[uid] = ages[k] + rand(net.rng)  # add fractional year
            net.household_ids[uid] = cluster_id
        end

        if has_sexes
            sex_str = string(hh[rand_row, :sexes])
            sexes = [parse(Int, strip(s)) for s in split(sex_str, ',')]
            for k in 1:min(length(sexes), cluster_size)
                uid = uid_offset + k
                people.female.raw[uid] = (sexes[k] == 2)
            end
        end

        # Add symmetric pairwise contacts within the household
        for i in 1:cluster_size
            for j in (i+1):cluster_size
                push!(p1_buf, uid_offset + i)
                push!(p2_buf, uid_offset + j)
            end
        end

        uid_offset += cluster_size
        n_remaining -= cluster_size
    end

    if !isempty(p1_buf)
        add_edges!(net.data.edges, p1_buf, p2_buf)
    end

    return net
end

function step!(net::HouseholdNet, sim)
    return net
end

export HouseholdNet
