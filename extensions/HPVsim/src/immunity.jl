"""
Cross-genotype immunity connector for HPV.

Manages immunity across HPV genotypes after infection clearance. Uses
a matrix-based approach where immunity[i,j] gives the protection that
prior infection with genotype i provides against genotype j.

Similar to RotaImmunityConnector but adapted for HPV's genotype structure
(no G/P segmentation — just named genotypes with risk-group relationships).
"""

# ============================================================================
# HPVImmunityConnector
# ============================================================================

"""
    HPVImmunityConnector <: AbstractConnector

Cross-genotype immunity connector for HPV. Automatically detects all
HPVGenotype diseases and manages immunity via a genotype × genotype matrix.

After clearance of a genotype infection, agents gain partial protection
against reinfection by the same and related genotypes. Immunity wanes
exponentially over time.

# Keyword arguments
- `imm_init::Float64` — mean initial immunity level after clearance (default 0.35,
  matching Python hpvsim `imm_init = beta_mean(0.35, 0.025)`)
- `own_imm_hr::Float64` — same-genotype cross-immunity factor for grouped types
  like hi5/ohr/lr (default 0.90, matching Python `own_imm_hr`)
- `partial_imm::Float64` — same risk-group cross-protection factor (default 0.50,
  matching Python `cross_imm_sus_high`)
- `cross_imm::Float64` — different risk-group cross-protection factor (default 0.30,
  matching Python `cross_imm_sus_med`)
- `waning_rate::Float64` — exponential waning rate per year (default 0.0,
  matching Python `use_waning=False`)
"""
mutable struct HPVImmunityConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData

    # Parameters
    imm_init::Float64       # Mean initial immunity level (Python: ~0.35)
    own_imm_hr::Float64     # Own-immunity factor for grouped genotypes (hi5, ohr, lr)
    partial_imm::Float64    # Cross-imm factor for same risk group (Python: cross_imm_sus_high)
    cross_imm::Float64      # Cross-imm factor for different risk group (Python: cross_imm_sus_med)
    waning_rate::Float64

    # Per-agent states
    has_immunity::Starsim.StateVector{Bool, Vector{Bool}}
    n_cleared::Starsim.StateVector{Float64, Vector{Float64}}

    # Per-genotype immunity level (drawn at clearance time)
    genotype_imm_level::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype immunity decay states
    genotype_decay::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype clearance times
    genotype_ti_cleared::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Mapping discovered at init
    hpv_diseases::Vector{HPVGenotype}
    genotype_names::Vector{Symbol}
    imm_matrix::Matrix{Float64}  # imm_matrix[source, target] = cross-immunity efficacy
end

"""Genotypes that use 1.0 own-immunity factor (like Python hpvsim's cross_immunity matrix)."""
const INDIVIDUAL_TYPE_GENOTYPES = Set([:hpv16, :hpv18])

function HPVImmunityConnector(;
    name::Symbol = :hpv_immunity,
    imm_init::Real = 0.35,
    own_imm_hr::Real = 0.90,
    partial_imm::Real = 0.50,
    cross_imm::Real = 0.30,
    waning_rate::Real = 0.0,
    # Legacy aliases
    own_imm::Union{Real, Nothing} = nothing,
)
    md = Starsim.ModuleData(name; label="HPV immunity connector")
    cd = Starsim.ConnectorData(md)

    # Support legacy own_imm kwarg
    hr = own_imm !== nothing ? Float64(own_imm) : Float64(own_imm_hr)

    HPVImmunityConnector(
        cd,
        Float64(imm_init),
        hr,
        Float64(partial_imm),
        Float64(cross_imm),
        Float64(waning_rate),
        Starsim.BoolState(:hpv_has_immunity; default=false, label="Has HPV immunity"),
        Starsim.FloatState(:hpv_n_cleared; default=0.0, label="HPV clearances"),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        HPVGenotype[],
        Symbol[],
        Matrix{Float64}(undef, 0, 0),
    )
end

Starsim.connector_data(c::HPVImmunityConnector) = c.data

# ============================================================================
# Risk group classification for cross-immunity
# ============================================================================

"""High-risk genotypes (includes both grouped and individual types)."""
const HIGH_RISK_GENOTYPES = Set([:hpv16, :hpv18, :hi5, :ohr, :hr,
                                  :hpv31, :hpv33, :hpv45, :hpv52, :hpv58])

"""Low-risk genotypes (includes both grouped and individual types)."""
const LOW_RISK_GENOTYPES = Set([:lr, :hpv6, :hpv11])

"""Check if two genotypes are in the same risk group."""
function same_risk_group(g1::Symbol, g2::Symbol)
    hr1 = g1 in HIGH_RISK_GENOTYPES
    hr2 = g2 in HIGH_RISK_GENOTYPES
    lr1 = g1 in LOW_RISK_GENOTYPES
    lr2 = g2 in LOW_RISK_GENOTYPES
    return (hr1 && hr2) || (lr1 && lr2)
end

# ============================================================================
# Initialization
# ============================================================================

function Starsim.init_pre!(c::HPVImmunityConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    # Auto-detect HPVGenotype diseases
    c.hpv_diseases = HPVGenotype[]
    c.genotype_names = Symbol[]
    for (_, dis) in sim.diseases
        if dis isa HPVGenotype
            push!(c.hpv_diseases, dis)
            push!(c.genotype_names, dis.genotype)
        end
    end

    if isempty(c.hpv_diseases)
        md.initialized = true
        return c
    end

    # Register base states
    for s in [c.has_immunity, c.n_cleared]
        Starsim.add_module_state!(sim.people, s)
    end

    # Create per-genotype decay, clearance-time, and immunity-level states
    for gn in c.genotype_names
        decay_name = Symbol("hpv_decay_", gn)
        sv_decay = Starsim.FloatState(decay_name; default=0.0, label="Decay $gn")
        Starsim.add_module_state!(sim.people, sv_decay)
        c.genotype_decay[gn] = sv_decay

        ti_name = Symbol("hpv_ti_cleared_", gn)
        sv_ti = Starsim.FloatState(ti_name; default=Inf, label="TI cleared $gn")
        Starsim.add_module_state!(sim.people, sv_ti)
        c.genotype_ti_cleared[gn] = sv_ti

        imm_name = Symbol("hpv_imm_level_", gn)
        sv_imm = Starsim.FloatState(imm_name; default=0.0, label="Imm level $gn")
        Starsim.add_module_state!(sim.people, sv_imm)
        c.genotype_imm_level[gn] = sv_imm
    end

    # Build cross-immunity matrix matching Python hpvsim's structure:
    # - hpv16/hpv18 have own-immunity factor = 1.0
    # - grouped types (hi5, ohr, lr, hr) use own_imm_hr (default 0.90)
    # - same risk group cross-immunity = partial_imm (Python: cross_imm_sus_high = 0.50)
    # - different risk group = cross_imm (Python: cross_imm_sus_med = 0.30)
    n_g = length(c.genotype_names)
    c.imm_matrix = zeros(Float64, n_g, n_g)
    for i in 1:n_g
        for j in 1:n_g
            gi = c.genotype_names[i]
            gj = c.genotype_names[j]
            if i == j
                # Own-immunity: 1.0 for individual types (hpv16, hpv18),
                # own_imm_hr for grouped types (matching Python)
                if gi in INDIVIDUAL_TYPE_GENOTYPES
                    c.imm_matrix[i, j] = 1.0
                else
                    c.imm_matrix[i, j] = c.own_imm_hr
                end
            elseif same_risk_group(gi, gj)
                c.imm_matrix[i, j] = c.partial_imm
            else
                c.imm_matrix[i, j] = c.cross_imm
            end
        end
    end

    md.initialized = true
    return c
end

# ============================================================================
# Step — update immunity and apply to susceptibility
# ============================================================================

function Starsim.step!(c::HPVImmunityConnector, sim)
    isempty(c.hpv_diseases) && return c
    _update_decay_factors!(c, sim)
    _apply_immunity!(c, sim)
    return c
end

"""Update exponential decay factors for all genotype immunities."""
function _update_decay_factors!(c::HPVImmunityConnector, sim)
    dt = sim.pars.dt
    active = sim.people.auids.values
    ti_now = Float64(Starsim.module_data(c).t.ti)

    for gn in c.genotype_names
        ti_cleared_raw = c.genotype_ti_cleared[gn].raw
        decay_raw = c.genotype_decay[gn].raw

        @inbounds for u in active
            tc = ti_cleared_raw[u]
            if tc < Inf && tc > 0.0
                time_since = (ti_now - tc) * dt  # in years
                if time_since > 0.0
                    decay_raw[u] = exp(-c.waning_rate * time_since)
                else
                    decay_raw[u] = 1.0
                end
            else
                decay_raw[u] = 0.0
            end
        end
    end
    return
end

"""Apply cross-immunity matrix to disease susceptibilities, scaled by per-agent immunity levels."""
function _apply_immunity!(c::HPVImmunityConnector, sim)
    active = sim.people.auids.values
    n_g = length(c.genotype_names)
    has_imm_raw = c.has_immunity.raw

    for (j, disease) in enumerate(c.hpv_diseases)
        rel_sus_raw = disease.infection.rel_sus.raw

        @inbounds for u in active
            has_imm_raw[u] || continue

            # Compute protection from all prior clearances (sum, clamped to 1.0)
            # Matches Python: sus_imm = sum(cross_imm[source, target] * nab_imm[source])
            total_protection = 0.0
            for i in 1:n_g
                gn_source = c.genotype_names[i]
                decay = c.genotype_decay[gn_source].raw[u]
                if decay > 0.0
                    imm_level = c.genotype_imm_level[gn_source].raw[u]
                    protection = c.imm_matrix[i, j] * imm_level * decay
                    total_protection += protection
                end
            end

            # Apply protection (clamp to [0, 1])
            total_protection = min(total_protection, 1.0)
            if total_protection > 0.0
                new_sus = 1.0 - total_protection
                rel_sus_raw[u] = min(rel_sus_raw[u], new_sus)
            end
        end
    end
    return
end

# ============================================================================
# Record events — called by HPVGenotype disease
# ============================================================================

"""Record a new infection event for an agent."""
function record_infection!(c::HPVImmunityConnector, disease::HPVGenotype, uid::Int)
    return  # Infections are tracked; immunity only granted on clearance
end

"""
Record a clearance event: update per-genotype clearance time, immunity level, and flags.
Called from `step_state!` of HPVGenotype.

Seroconversion is probabilistic: only a fraction of clearances lead to
protective immunity, determined by the genotype's `sero_prob` parameter.
On seroconversion, an immunity level is drawn from a Beta distribution
matching Python hpvsim's `imm_init = beta_mean(mean, var)`.
"""
function record_clearance!(c::HPVImmunityConnector, disease::HPVGenotype, uid::Int)
    gn = disease.genotype
    if !haskey(c.genotype_ti_cleared, gn)
        return
    end

    # Check seroconversion — not all clearances produce immunity
    sero_prob = disease.params.sero_prob
    if rand(disease.rng) >= sero_prob
        return  # No seroconversion; no immunity gained
    end

    # Update clearance time to current timestep
    md = Starsim.module_data(disease)
    c.genotype_ti_cleared[gn].raw[uid] = Float64(md.t.ti)

    # Draw immunity level from Beta distribution (matching Python imm_init)
    # Python uses beta_mean(par1=mean, par2=var): mean=0.35, var=0.025
    imm_mean = c.imm_init
    imm_var  = 0.025  # Fixed variance matching Python default
    if imm_mean > 0.0 && imm_var > 0.0 && imm_mean < 1.0
        # Convert mean/var to Beta(α, β) params
        alpha = imm_mean * (imm_mean * (1.0 - imm_mean) / imm_var - 1.0)
        beta_param = (1.0 - imm_mean) * (imm_mean * (1.0 - imm_mean) / imm_var - 1.0)
        if alpha > 0.0 && beta_param > 0.0
            level = rand(disease.rng, Distributions.Beta(alpha, beta_param))
        else
            level = imm_mean
        end
    else
        level = imm_mean
    end

    # Take max of existing and new immunity level (boosting on reinfection)
    existing_level = c.genotype_imm_level[gn].raw[uid]
    c.genotype_imm_level[gn].raw[uid] = max(existing_level, level)

    # Mark as having immunity
    c.has_immunity.raw[uid] = true
    c.n_cleared.raw[uid] += 1.0

    return
end
