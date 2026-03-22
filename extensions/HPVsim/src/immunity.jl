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

Two immunity channels are tracked (matching Python hpvsim):

- **Susceptibility immunity** (`sus_imm`): reduces probability of reinfection.
  Gated by seroconversion probability. Uses `peak_imm` / `nab_imm` and
  `cross_immunity_sus` matrix.
- **Severity immunity** (`sev_imm`): reduces duration of reinfections via
  `dur_precin *= (1 - sev_imm)`. Always set at clearance (no sero_prob gating).
  Uses `cell_imm` and `cross_immunity_sev` matrix.

# Keyword arguments
- `imm_init::Float64` — mean initial sus immunity level after clearance (default 0.35)
- `cell_imm_init::Float64` — mean initial cell immunity level (default 0.25)
- `own_imm_hr::Float64` — same-genotype factor for grouped types (default 0.90)
- `partial_imm::Float64` — same risk-group sus cross-protection (default 0.50)
- `cross_imm::Float64` — different risk-group sus cross-protection (default 0.30)
- `partial_imm_sev::Float64` — same risk-group sev cross-protection (default 0.70)
- `cross_imm_sev::Float64` — different risk-group sev cross-protection (default 0.50)
- `waning_rate::Float64` — exponential waning rate per year (default 0.0)
"""
mutable struct HPVImmunityConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData

    # Susceptibility immunity parameters
    imm_init::Float64       # Mean initial sus immunity level (Python: ~0.35)
    own_imm_hr::Float64     # Own-immunity factor for grouped genotypes (hi5, ohr, lr)
    partial_imm::Float64    # Sus cross-imm for same risk group (Python: cross_imm_sus_high)
    cross_imm::Float64      # Sus cross-imm for different risk group (Python: cross_imm_sus_med)
    waning_rate::Float64

    # Severity immunity parameters
    cell_imm_init::Float64  # Mean initial cell immunity level (Python: ~0.25)
    partial_imm_sev::Float64  # Sev cross-imm for same risk group (Python: cross_imm_sev_high)
    cross_imm_sev::Float64    # Sev cross-imm for different risk group (Python: cross_imm_sev_med)

    # Per-agent states
    has_immunity::Starsim.StateVector{Bool, Vector{Bool}}
    n_cleared::Starsim.StateVector{Float64, Vector{Float64}}

    # Per-genotype susceptibility immunity level (drawn at clearance, sero_prob gated)
    genotype_imm_level::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype cell immunity level (drawn at clearance, always set)
    genotype_cell_imm::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype computed sev_imm (updated each step)
    genotype_sev_imm::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype immunity decay states
    genotype_decay::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Per-genotype clearance times
    genotype_ti_cleared::Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}

    # Mapping discovered at init
    hpv_diseases::Vector{HPVGenotype}
    genotype_names::Vector{Symbol}
    imm_matrix::Matrix{Float64}      # sus cross-immunity matrix
    sev_matrix::Matrix{Float64}      # sev cross-immunity matrix
end

"""Genotypes that use 1.0 own-immunity factor (like Python hpvsim's cross_immunity matrix)."""
const INDIVIDUAL_TYPE_GENOTYPES = Set([:hpv16, :hpv18])

function HPVImmunityConnector(;
    name::Symbol = :hpv_immunity,
    imm_init::Real = 0.35,
    cell_imm_init::Real = 0.25,
    own_imm_hr::Real = 0.90,
    partial_imm::Real = 0.50,
    cross_imm::Real = 0.30,
    partial_imm_sev::Real = 0.70,
    cross_imm_sev::Real = 0.50,
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
        Float64(cell_imm_init),
        Float64(partial_imm_sev),
        Float64(cross_imm_sev),
        Starsim.BoolState(:hpv_has_immunity; default=false, label="Has HPV immunity"),
        Starsim.FloatState(:hpv_n_cleared; default=0.0, label="HPV clearances"),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        Dict{Symbol, Starsim.StateVector{Float64, Vector{Float64}}}(),
        HPVGenotype[],
        Symbol[],
        Matrix{Float64}(undef, 0, 0),
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

    # Create per-genotype states
    for gn in c.genotype_names
        # Decay factor for sus immunity
        decay_name = Symbol("hpv_decay_", gn)
        sv_decay = Starsim.FloatState(decay_name; default=0.0, label="Decay $gn")
        Starsim.add_module_state!(sim.people, sv_decay)
        c.genotype_decay[gn] = sv_decay

        # Time of clearance
        ti_name = Symbol("hpv_ti_cleared_", gn)
        sv_ti = Starsim.FloatState(ti_name; default=Inf, label="TI cleared $gn")
        Starsim.add_module_state!(sim.people, sv_ti)
        c.genotype_ti_cleared[gn] = sv_ti

        # Sus immunity level (peak_imm equivalent, sero_prob gated)
        imm_name = Symbol("hpv_imm_level_", gn)
        sv_imm = Starsim.FloatState(imm_name; default=0.0, label="Imm level $gn")
        Starsim.add_module_state!(sim.people, sv_imm)
        c.genotype_imm_level[gn] = sv_imm

        # Cell immunity level (always set at clearance, not sero_prob gated)
        cell_name = Symbol("hpv_cell_imm_", gn)
        sv_cell = Starsim.FloatState(cell_name; default=0.0, label="Cell imm $gn")
        Starsim.add_module_state!(sim.people, sv_cell)
        c.genotype_cell_imm[gn] = sv_cell

        # Computed sev_imm per genotype (updated each step)
        sev_name = Symbol("hpv_sev_imm_", gn)
        sv_sev = Starsim.FloatState(sev_name; default=0.0, label="Sev imm $gn")
        Starsim.add_module_state!(sim.people, sv_sev)
        c.genotype_sev_imm[gn] = sv_sev
    end

    # Build susceptibility cross-immunity matrix (cross_immunity_sus)
    n_g = length(c.genotype_names)
    c.imm_matrix = zeros(Float64, n_g, n_g)
    for i in 1:n_g
        for j in 1:n_g
            gi = c.genotype_names[i]
            gj = c.genotype_names[j]
            if i == j
                c.imm_matrix[i, j] = gi in INDIVIDUAL_TYPE_GENOTYPES ? 1.0 : c.own_imm_hr
            elseif gi in INDIVIDUAL_TYPE_GENOTYPES && gj in INDIVIDUAL_TYPE_GENOTYPES
                c.imm_matrix[i, j] = c.partial_imm
            else
                c.imm_matrix[i, j] = c.cross_imm
            end
        end
    end

    # Build severity cross-immunity matrix (cross_immunity_sev)
    c.sev_matrix = zeros(Float64, n_g, n_g)
    for i in 1:n_g
        for j in 1:n_g
            gi = c.genotype_names[i]
            gj = c.genotype_names[j]
            if i == j
                c.sev_matrix[i, j] = gi in INDIVIDUAL_TYPE_GENOTYPES ? 1.0 : c.own_imm_hr
            elseif gi in INDIVIDUAL_TYPE_GENOTYPES && gj in INDIVIDUAL_TYPE_GENOTYPES
                c.sev_matrix[i, j] = c.partial_imm_sev
            else
                c.sev_matrix[i, j] = c.cross_imm_sev
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

"""Apply cross-immunity matrices to disease susceptibilities and compute severity immunity."""
function _apply_immunity!(c::HPVImmunityConnector, sim)
    active = sim.people.auids.values
    n_g = length(c.genotype_names)
    has_imm_raw = c.has_immunity.raw

    # Apply sus_imm to rel_sus (same as before)
    for (j, disease) in enumerate(c.hpv_diseases)
        rel_sus_raw = disease.infection.rel_sus.raw

        # Reset rel_sus to 1.0 for immune agents before recomputing
        @inbounds for u in active
            if has_imm_raw[u]
                rel_sus_raw[u] = 1.0
            end
        end

        @inbounds for u in active
            has_imm_raw[u] || continue

            # sus_imm = sum(cross_imm_sus[source, target] * nab_imm[source])
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

            total_protection = min(total_protection, 1.0)
            if total_protection > 0.0
                rel_sus_raw[u] = 1.0 - total_protection
            end
        end
    end

    # Compute sev_imm for each genotype (cross_immunity_sev @ cell_imm)
    for (j, disease) in enumerate(c.hpv_diseases)
        gn_target = c.genotype_names[j]
        sev_imm_raw = c.genotype_sev_imm[gn_target].raw

        @inbounds for u in active
            if !has_imm_raw[u]
                sev_imm_raw[u] = 0.0
                continue
            end

            # sev_imm = sum(cross_imm_sev[source, target] * cell_imm[source])
            total_sev = 0.0
            for i in 1:n_g
                gn_source = c.genotype_names[i]
                cell_level = c.genotype_cell_imm[gn_source].raw[u]
                if cell_level > 0.0
                    total_sev += c.sev_matrix[i, j] * cell_level
                end
            end

            sev_imm_raw[u] = min(total_sev, 1.0)
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
Record a clearance event: update immunity levels and flags.
Called from `step_state!` of HPVGenotype.

Two immunity channels are updated:
- **peak_imm** (sus immunity): gated by seroconversion probability
- **cell_imm** (sev immunity): ALWAYS set (no sero_prob gating), matching Python
"""
function record_clearance!(c::HPVImmunityConnector, disease::HPVGenotype, uid::Int)
    gn = disease.genotype
    if !haskey(c.genotype_ti_cleared, gn)
        return
    end

    # Update clearance time to current timestep
    md = Starsim.module_data(disease)
    c.genotype_ti_cleared[gn].raw[uid] = Float64(md.t.ti)

    # --- Cell immunity (severity): ALWAYS set (no sero_prob gating) ---
    # Python: cell_imm[g, cleared_inds] = sample(cell_imm_init) * rel_imm
    cell_mean = c.cell_imm_init
    cell_var  = 0.025  # Fixed variance matching Python default
    if cell_mean > 0.0 && cell_var > 0.0 && cell_mean < 1.0
        alpha = cell_mean * (cell_mean * (1.0 - cell_mean) / cell_var - 1.0)
        beta_param = (1.0 - cell_mean) * (cell_mean * (1.0 - cell_mean) / cell_var - 1.0)
        if alpha > 0.0 && beta_param > 0.0
            cell_level = rand(disease.rng, Distributions.Beta(alpha, beta_param))
        else
            cell_level = cell_mean
        end
    else
        cell_level = cell_mean
    end
    # Take max of existing and new cell immunity level (boosting on reinfection)
    existing_cell = c.genotype_cell_imm[gn].raw[uid]

    # Python distinction: agents WITH prior immunity get cell_imm without rel_imm scaling
    # agents WITHOUT prior immunity get cell_imm * rel_imm
    # (rel_imm defaults to 1.0, so this only matters if rel_imm is customized)
    c.genotype_cell_imm[gn].raw[uid] = max(existing_cell, cell_level)

    # --- Susceptibility immunity ---
    # Python: agents WITH prior immunity (nab_imm > 0) ALWAYS get boosted (no sero_prob gating)
    # agents WITHOUT prior immunity are gated by sero_prob
    existing_level = c.genotype_imm_level[gn].raw[uid]
    has_prior_imm = existing_level > 0.0

    should_update = has_prior_imm || (rand(disease.rng) < disease.params.sero_prob)

    if should_update
        imm_mean = c.imm_init
        imm_var  = 0.025
        if imm_mean > 0.0 && imm_var > 0.0 && imm_mean < 1.0
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
        c.genotype_imm_level[gn].raw[uid] = max(existing_level, level)
    else
        # Agent didn't seroconvert; consume the RNG draw for consistency
        # (no-op on immunity)
    end

    # Mark as having immunity
    c.has_immunity.raw[uid] = true
    c.n_cleared.raw[uid] += 1.0

    return
end

"""
    get_sev_imm(sim, genotype::Symbol, uid::Int) → Float64

Look up the current severity immunity for an agent and genotype.
Returns 0.0 if no immunity connector is present.
"""
function get_sev_imm(sim, genotype::Symbol, uid::Int)
    for (_, conn) in sim.connectors
        if conn isa HPVImmunityConnector && haskey(conn.genotype_sev_imm, genotype)
            return conn.genotype_sev_imm[genotype].raw[uid]
        end
    end
    return 0.0
end
