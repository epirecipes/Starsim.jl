"""
Bitmask-vectorized cross-strain immunity connector.
Port of Python `rotasim.immunity.RotaImmunityConnector`.

Uses bitwise operations on per-agent integer bitmasks to efficiently
compute strain-specific immunity without per-agent loops.
"""

# ============================================================================
# Bitmask utility functions
# ============================================================================

"""
    hamming_distance(a::Integer, b::Integer)

Compute the Hamming distance between two bitmasks (number of differing bits).
Used to quantify genetic distance between rotavirus strains.
"""
hamming_distance(a::Integer, b::Integer) = count_ones(xor(a, b))

"""
    strain_similarity(a::Integer, b::Integer)

Compute similarity between two strain bitmasks as fraction of shared bits.
Returns 1.0 for identical strains, 0.0 for completely different.
"""
function strain_similarity(a::Integer, b::Integer)
    union_bits = count_ones(a | b)
    union_bits == 0 && return 1.0
    shared_bits = count_ones(a & b)
    return shared_bits / union_bits
end

"""
    bitmask_from_gp(G::Int, P::Int, G_to_bit::Dict{Int,Int}, P_to_bit::Dict{Int,Int})

Create a combined GP bitmask from G and P genotypes using the given bit mappings.
"""
function bitmask_from_gp(G::Int, P::Int, G_to_bit::Dict{Int,Int}, P_to_bit::Dict{Int,Int})
    g_mask = haskey(G_to_bit, G) ? (Int64(1) << G_to_bit[G]) : Int64(0)
    p_mask = haskey(P_to_bit, P) ? (Int64(1) << P_to_bit[P]) : Int64(0)
    return g_mask | p_mask
end

"""
    match_type(disease_G::Int, disease_P::Int, exposed_G_mask::Int64, exposed_P_mask::Int64,
               exposed_GP_mask::Int64, G_to_bit, P_to_bit, GP_to_bit)

Determine immunity match type for a disease against an agent's exposure history.
Returns :homotypic, :partial_hetero, :complete_hetero, or :naive.
"""
function match_type(disease_G::Int, disease_P::Int,
                    exposed_GP::Int64, exposed_G::Int64, exposed_P::Int64,
                    G_to_bit::Dict{Int,Int}, P_to_bit::Dict{Int,Int},
                    GP_to_bit::Dict{Tuple{Int,Int},Int})
    gp_mask = Int64(1) << GP_to_bit[(disease_G, disease_P)]
    (exposed_GP & gp_mask) != 0 && return :homotypic

    g_mask = Int64(1) << G_to_bit[disease_G]
    p_mask = Int64(1) << P_to_bit[disease_P]
    has_g = (exposed_G & g_mask) != 0
    has_p = (exposed_P & p_mask) != 0
    (has_g || has_p) && return :partial_hetero

    (exposed_GP != 0 || exposed_G != 0 || exposed_P != 0) && return :complete_hetero
    return :naive
end

# ============================================================================
# RotaImmunityConnector
# ============================================================================

"""
    RotaImmunityConnector <: AbstractConnector

High-performance cross-strain immunity connector using bitmask vectorization.
Automatically detects all `Rotavirus` diseases in the simulation and manages
cross-strain immunity via bitwise operations on entire population arrays.

# Keyword arguments
- `homotypic_efficacy::Float64` — protection from same G,P strain (default 0.9)
- `partial_hetero_efficacy::Float64` — protection from shared G or P (default 0.5)
- `complete_hetero_efficacy::Float64` — protection from different G,P (default 0.3)
- `naive_efficacy::Float64` — baseline for naive individuals (default 0.0)
"""
mutable struct RotaImmunityConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData

    # Parameters
    homotypic_efficacy::Float64
    partial_hetero_efficacy::Float64
    complete_hetero_efficacy::Float64
    naive_efficacy::Float64

    # Per-agent bitmask states
    exposed_GP_bitmask::Starsim.StateVector{Int64, Vector{Int64}}
    exposed_G_bitmask::Starsim.StateVector{Int64, Vector{Int64}}
    exposed_P_bitmask::Starsim.StateVector{Int64, Vector{Int64}}
    has_immunity::Starsim.StateVector{Bool, Vector{Bool}}
    oldest_infection::Starsim.StateVector{Float64, Vector{Float64}}
    num_current_infections::Starsim.StateVector{Float64, Vector{Float64}}
    num_recovered_infections::Starsim.StateVector{Float64, Vector{Float64}}

    # Per-agent decay factor arrays (reused each step)
    homotypic_decay_factor::Starsim.StateVector{Float64, Vector{Float64}}
    partial_match_decay_factor::Starsim.StateVector{Float64, Vector{Float64}}
    final_decayed_factor::Starsim.StateVector{Float64, Vector{Float64}}

    # Per-strain decay states (one per unique GP pair)
    strain_decay_states::Dict{Tuple{Int,Int}, Starsim.StateVector{Float64, Vector{Float64}}}

    # Bitmask mappings (populated during init)
    rota_diseases::Vector{Rotavirus}
    unique_G::Vector{Int}
    unique_P::Vector{Int}
    unique_GP::Vector{Tuple{Int,Int}}
    G_to_bit::Dict{Int, Int}
    P_to_bit::Dict{Int, Int}
    GP_to_bit::Dict{Tuple{Int,Int}, Int}
    disease_G_masks::Dict{Symbol, Int64}
    disease_P_masks::Dict{Symbol, Int64}
    disease_GP_masks::Dict{Symbol, Int64}
end

function RotaImmunityConnector(;
    name::Symbol = :rota_immunity,
    homotypic_efficacy::Real     = 0.9,
    partial_hetero_efficacy::Real = 0.5,
    complete_hetero_efficacy::Real = 0.3,
    naive_efficacy::Real         = 0.0,
)
    md = Starsim.ModuleData(name; label="Rota immunity connector")
    cd = Starsim.ConnectorData(md)

    RotaImmunityConnector(
        cd,
        Float64(homotypic_efficacy),
        Float64(partial_hetero_efficacy),
        Float64(complete_hetero_efficacy),
        Float64(naive_efficacy),
        # Bitmask states
        Starsim.IntState(:exposed_GP_bitmask; default=0, label="Exposed GP bitmask"),
        Starsim.IntState(:exposed_G_bitmask; default=0, label="Exposed G bitmask"),
        Starsim.IntState(:exposed_P_bitmask; default=0, label="Exposed P bitmask"),
        Starsim.BoolState(:has_immunity; default=false, label="Has immunity"),
        Starsim.FloatState(:oldest_infection; default=NaN, label="Oldest infection time"),
        Starsim.FloatState(:num_current_infections; default=0.0, label="Current infections"),
        Starsim.FloatState(:num_recovered_infections; default=0.0, label="Recovered infections"),
        # Decay factors
        Starsim.FloatState(:homotypic_decay; default=0.0, label="Homotypic decay"),
        Starsim.FloatState(:partial_decay; default=0.0, label="Partial match decay"),
        Starsim.FloatState(:final_decay; default=0.0, label="Final decay factor"),
        # Per-strain decay states (populated during init)
        Dict{Tuple{Int,Int}, Starsim.StateVector{Float64, Vector{Float64}}}(),
        # Empty mappings
        Rotavirus[],
        Int[], Int[], Tuple{Int,Int}[],
        Dict{Int,Int}(), Dict{Int,Int}(), Dict{Tuple{Int,Int},Int}(),
        Dict{Symbol,Int64}(), Dict{Symbol,Int64}(), Dict{Symbol,Int64}(),
    )
end

Starsim.connector_data(c::RotaImmunityConnector) = c.data

# ============================================================================
# Initialization
# ============================================================================

function Starsim.init_pre!(c::RotaImmunityConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    # Auto-detect Rotavirus diseases
    c.rota_diseases = Rotavirus[]
    for (_, dis) in sim.diseases
        if dis isa Rotavirus
            push!(c.rota_diseases, dis)
        end
    end

    if isempty(c.rota_diseases)
        md.initialized = true
        return c
    end

    # Compute unique genotypes
    c.unique_G  = sort(unique(d.G for d in c.rota_diseases))
    c.unique_P  = sort(unique(d.P for d in c.rota_diseases))
    c.unique_GP = sort(unique((d.G, d.P) for d in c.rota_diseases))

    # Register base states
    base_states = [
        c.exposed_GP_bitmask, c.exposed_G_bitmask, c.exposed_P_bitmask,
        c.has_immunity, c.oldest_infection,
        c.num_current_infections, c.num_recovered_infections,
        c.homotypic_decay_factor, c.partial_match_decay_factor, c.final_decayed_factor,
    ]
    for s in base_states
        Starsim.add_module_state!(sim.people, s)
    end

    # Create and register per-strain decay states
    for gp in c.unique_GP
        sname = Symbol("G$(gp[1])P$(gp[2])_decay")
        sv = Starsim.FloatState(sname; default=0.0, label="Decay G$(gp[1])P$(gp[2])")
        Starsim.add_module_state!(sim.people, sv)
        c.strain_decay_states[gp] = sv
    end

    md.initialized = true
    return c
end

function Starsim.init_post!(c::RotaImmunityConnector, sim)
    isempty(c.rota_diseases) && return c

    # Build bitmask mappings
    c.G_to_bit  = Dict(g => i-1 for (i, g) in enumerate(c.unique_G))
    c.P_to_bit  = Dict(p => i-1 for (i, p) in enumerate(c.unique_P))
    c.GP_to_bit = Dict(gp => i-1 for (i, gp) in enumerate(c.unique_GP))

    # Pre-compute per-disease masks
    for disease in c.rota_diseases
        nm = Starsim.module_data(disease).name
        c.disease_G_masks[nm]  = Int64(1) << c.G_to_bit[disease.G]
        c.disease_P_masks[nm]  = Int64(1) << c.P_to_bit[disease.P]
        c.disease_GP_masks[nm] = Int64(1) << c.GP_to_bit[(disease.G, disease.P)]
    end

    return c
end

# ============================================================================
# Step — main immunity update
# ============================================================================

function Starsim.step!(c::RotaImmunityConnector, sim)
    isempty(c.rota_diseases) && return c
    _reset_decay_factors!(c, sim)
    _update_immunity_decay_factors!(c, sim)
    _calculate_disease_susceptibilities!(c, sim)
    return c
end

"""Reset all decay factor arrays to zero."""
function _reset_decay_factors!(c::RotaImmunityConnector, sim)
    fill!(c.final_decayed_factor.raw, 0.0)
    fill!(c.partial_match_decay_factor.raw, 0.0)
    fill!(c.homotypic_decay_factor.raw, 0.0)
    for (_, sv) in c.strain_decay_states
        fill!(sv.raw, 0.0)
    end
    return
end

"""
Update immunity decay factors for all diseases based on recovery times.
For each disease, agents who have recovered get an exponential decay
factor based on time since recovery and their individual waning rate.
"""
function _update_immunity_decay_factors!(c::RotaImmunityConnector, sim)
    dt_days = sim.pars.dt * 365.25
    active = sim.people.auids.values
    homo_raw = c.homotypic_decay_factor.raw

    for disease in c.rota_diseases
        infected_raw    = disease.infection.infected.raw
        ti_rec_raw      = disease.ti_recovered.raw
        waning_raw      = disease.waning_rate.raw
        ti_now          = Starsim.module_data(disease).t.ti

        gp = (disease.G, disease.P)
        strain_raw = c.strain_decay_states[gp].raw

        @inbounds for u in active
            if !infected_raw[u] && ti_rec_raw[u] < Inf && ti_rec_raw[u] > 0.0
                time_since_recovery_days = (Float64(ti_now) - ti_rec_raw[u]) * dt_days
                time_since_recovery_days <= 0.0 && continue

                decay_rate = waning_raw[u]
                decay_rate <= 0.0 && continue

                decay_factor = exp(-decay_rate * time_since_recovery_days)

                if decay_factor > strain_raw[u]
                    strain_raw[u] = decay_factor
                end
                if decay_factor > homo_raw[u]
                    homo_raw[u] = decay_factor
                end
            end
        end
    end
    return
end

"""
Calculate disease susceptibilities using bitmask matching.
For each disease, determines the type of immunity match for each agent
and applies the corresponding efficacy × decay factor.
"""
function _calculate_disease_susceptibilities!(c::RotaImmunityConnector, sim)
    active = sim.people.auids.values

    gp_bits_raw     = c.exposed_GP_bitmask.raw
    g_bits_raw      = c.exposed_G_bitmask.raw
    p_bits_raw      = c.exposed_P_bitmask.raw
    has_imm_raw     = c.has_immunity.raw
    homo_decay_raw  = c.homotypic_decay_factor.raw

    # Pre-extract all strain decay raw arrays into an indexable vector
    gp_keys = collect(keys(c.strain_decay_states))
    strain_raw_arrays = [c.strain_decay_states[gp].raw for gp in gp_keys]
    n_strains = length(gp_keys)

    for disease in c.rota_diseases
        nm = Starsim.module_data(disease).name
        gp_mask = c.disease_GP_masks[nm]
        g_mask  = c.disease_G_masks[nm]
        p_mask  = c.disease_P_masks[nm]

        # Pre-compute partial match indices and all-strain indices OUTSIDE the agent loop
        partial_indices = Int[]
        all_indices = collect(1:n_strains)
        for (idx, gp) in enumerate(gp_keys)
            if gp != (disease.G, disease.P) && (gp[1] == disease.G || gp[2] == disease.P)
                push!(partial_indices, idx)
            end
        end

        rel_sus_raw = disease.infection.rel_sus.raw

        @inbounds for u in active
            if !has_imm_raw[u]
                rel_sus_raw[u] = 1.0
                continue
            end

            has_exact = (gp_bits_raw[u] & gp_mask) != 0

            if has_exact
                efficacy = c.homotypic_efficacy
                decay = homo_decay_raw[u]
            else
                has_g = (g_bits_raw[u] & g_mask) != 0
                has_p = (p_bits_raw[u] & p_mask) != 0

                if has_g || has_p
                    efficacy = c.partial_hetero_efficacy
                    decay = 0.0
                    for idx in partial_indices
                        v = strain_raw_arrays[idx][u]
                        if v > decay
                            decay = v
                        end
                    end
                else
                    efficacy = c.complete_hetero_efficacy
                    decay = 0.0
                    for idx in all_indices
                        v = strain_raw_arrays[idx][u]
                        if v > decay
                            decay = v
                        end
                    end
                end
            end

            rel_sus_raw[u] = 1.0 - efficacy * decay
        end
    end
    return
end

# ============================================================================
# Record infection / recovery — called by Rotavirus disease
# ============================================================================

"""Record a new infection event for an agent."""
function record_infection!(c::RotaImmunityConnector, disease::Rotavirus, uid::Int)
    c.num_current_infections.raw[uid] += 1.0
    return
end

"""
Record a recovery event: update bitmasks and immunity flags.
Called from `step_state!` of Rotavirus.
"""
function record_recovery!(c::RotaImmunityConnector, disease::Rotavirus, uid::Int)
    # Update bitmasks
    G_bit  = Int64(1) << c.G_to_bit[disease.G]
    P_bit  = Int64(1) << c.P_to_bit[disease.P]
    GP_bit = Int64(1) << c.GP_to_bit[(disease.G, disease.P)]

    c.exposed_G_bitmask.raw[uid]  |= G_bit
    c.exposed_P_bitmask.raw[uid]  |= P_bit
    c.exposed_GP_bitmask.raw[uid] |= GP_bit

    # Mark as having immunity
    c.has_immunity.raw[uid] = true

    # Update infection counts
    c.num_current_infections.raw[uid]   -= 1.0
    c.num_recovered_infections.raw[uid] += 1.0

    # Track oldest infection time
    if isnan(c.oldest_infection.raw[uid])
        md = Starsim.module_data(disease)
        c.oldest_infection.raw[uid] = Float64(md.t.ti)
    end

    return
end
