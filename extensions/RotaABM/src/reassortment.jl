"""
Reassortment connector — handles genetic reassortment in multi-strain simulations.
Port of Python `rotasim.reassortment.RotaReassortmentConnector`.
"""

# ============================================================================
# RotaReassortmentConnector
# ============================================================================

"""
    RotaReassortmentConnector <: AbstractConnector

Connector for rotavirus genetic reassortment between co-infected strains.
Co-infected hosts (≥ 2 active rotavirus infections) may generate
reassortant strains by mixing G and P antigenic segments.

# Keyword arguments
- `reassortment_prob::Float64` — per-host per-timestep probability (default 0.05)
"""
mutable struct RotaReassortmentConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    reassortment_prob::Float64
    rota_diseases::Vector{Rotavirus}
    gp_to_disease::Dict{Tuple{Int,Int}, Rotavirus}
    use_preferred_partners::Bool
    rng::StableRNG
end

function RotaReassortmentConnector(;
    name::Symbol = :rota_reassortment,
    reassortment_prob::Real = 0.05,
    use_preferred_partners::Bool = false,
)
    md = Starsim.ModuleData(name; label="Rota reassortment connector")
    cd = Starsim.ConnectorData(md)
    RotaReassortmentConnector(
        cd, Float64(reassortment_prob),
        Rotavirus[],
        Dict{Tuple{Int,Int}, Rotavirus}(),
        use_preferred_partners,
        StableRNG(0),
    )
end

Starsim.connector_data(c::RotaReassortmentConnector) = c.data

# ============================================================================
# Initialization
# ============================================================================

function Starsim.init_pre!(c::RotaReassortmentConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    c.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Detect Rotavirus diseases
    c.rota_diseases = Rotavirus[]
    c.gp_to_disease = Dict{Tuple{Int,Int}, Rotavirus}()
    for (_, dis) in sim.diseases
        if dis isa Rotavirus
            push!(c.rota_diseases, dis)
            c.gp_to_disease[(dis.G, dis.P)] = dis
        end
    end

    isempty(c.rota_diseases) && error("RotaReassortmentConnector requires at least one Rotavirus disease")

    # Results
    npts = md.t.npts
    Starsim.define_results!(c,
        Starsim.Result(:n_reassortments; npts=npts, label="Reassortment events"),
    )

    md.initialized = true
    return c
end

# ============================================================================
# Step
# ============================================================================

function Starsim.step!(c::RotaReassortmentConnector, sim)
    isempty(c.rota_diseases) && return c

    # Find co-infected hosts via immunity connector
    co_infected = _get_coinfected_hosts(c, sim)
    isempty(co_infected) && return c

    # Bernoulli draw for reassortment
    reassorting = Int[]
    for u in co_infected
        if rand(c.rng) < c.reassortment_prob
            push!(reassorting, u)
        end
    end
    isempty(reassorting) && return c

    # Generate and apply infection plans
    infection_plans = Tuple{Rotavirus, Int}[]
    for uid in reassorting
        plans = _get_reassortant_infections(c, uid)
        append!(infection_plans, plans)
    end

    total = _apply_infection_plans!(c, sim, infection_plans)

    md = Starsim.module_data(c)
    ti = md.t.ti
    if ti <= length(Starsim.module_results(c)[:n_reassortments].values)
        Starsim.module_results(c)[:n_reassortments][ti] += Float64(total)
    end

    return c
end

"""Find agents with ≥ 2 active rotavirus infections."""
function _get_coinfected_hosts(c::RotaReassortmentConnector, sim)
    # Use immunity connector's num_current_infections if available
    for (_, conn) in sim.connectors
        if conn isa RotaImmunityConnector
            active = sim.people.auids.values
            result = Int[]
            @inbounds for u in active
                if conn.num_current_infections.raw[u] >= 2.0
                    push!(result, u)
                end
            end
            return result
        end
    end

    # Fallback: count directly
    active = sim.people.auids.values
    result = Int[]
    for u in active
        count = 0
        for d in c.rota_diseases
            if d.infection.infected.raw[u]
                count += 1
            end
        end
        if count >= 2
            push!(result, u)
        end
    end
    return result
end

"""Generate reassortant infection plans for a single co-infected host."""
function _get_reassortant_infections(c::RotaReassortmentConnector, uid::Int)
    # Find active diseases in this host
    active_diseases = Rotavirus[]
    for d in c.rota_diseases
        if d.infection.infected.raw[uid]
            push!(active_diseases, d)
        end
    end
    length(active_diseases) < 2 && return Tuple{Rotavirus, Int}[]

    # Generate all GP combinations from parent strains
    parent_gps = [(d.G, d.P) for d in active_diseases]
    all_combos = generate_gp_reassortments(parent_gps; use_preferred_partners=c.use_preferred_partners)

    # Exclude parent combinations
    parent_set = Set(parent_gps)
    reassortant_combos = filter(gp -> !(gp in parent_set), all_combos)

    plans = Tuple{Rotavirus, Int}[]
    for (G, P) in reassortant_combos
        disease = get(c.gp_to_disease, (G, P), nothing)
        disease === nothing && continue
        # Don't re-infect if already infected
        disease.infection.infected.raw[uid] && continue
        push!(plans, (disease, uid))
    end
    return plans
end

"""Apply infection plans in batches per disease."""
function _apply_infection_plans!(c::RotaReassortmentConnector, sim, plans::Vector{Tuple{Rotavirus, Int}})
    isempty(plans) && return 0

    # Group by disease
    disease_uids = Dict{Rotavirus, Vector{Int}}()
    for (disease, uid) in plans
        v = get!(disease_uids, disease, Int[])
        push!(v, uid)
    end

    total = 0
    for (disease, uid_list) in disease_uids
        Starsim.set_prognoses!(disease, sim, Starsim.UIDs(uid_list))
        total += length(uid_list)
    end
    return total
end
