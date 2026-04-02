"""
Diseases for Starsim.jl.

Mirrors Python starsim's `diseases.py`. Provides the Disease → Infection
hierarchy and built-in SIR/SIS models with full transmission logic.
"""

"""Exponential duration sample (discrete analogue: geometric). Memoryless recovery."""
_exponential_dur(mean::Float64, rng) = rand(rng, Exponential(mean))

"""Lognormal duration sample matching Python starsim's lognorm_ex(mean, std)."""
function _lognormal_dur(mean::Float64, std::Float64, rng)
    σ² = log(1 + (std / mean)^2)
    μ = log(mean) - σ² / 2
    return rand(rng, LogNormal(μ, sqrt(σ²)))
end

"""Sample a recovery duration using the specified distribution."""
function _sample_dur(dist::Symbol, mean::Float64, rng)
    if dist === :exponential
        return _exponential_dur(mean, rng)
    else  # :lognormal (Python starsim default)
        return _lognormal_dur(mean, 1.0, rng)
    end
end

"""Construct the recovery duration distribution for a disease."""
function _recovery_dist(dist::Symbol, mean::Float64)
    if dist === :exponential
        return Exponential(mean)
    else  # :lognormal (Python starsim default)
        σ² = log(1 + (1.0 / mean)^2)
        μ = log(mean) - σ² / 2
        return LogNormal(μ, sqrt(σ²))
    end
end

"""
    _sample_recovery_draws(disease, sim, uids::UIDs, ti::Int) -> Vector{Float64}

Sample recovery durations for the specified agents. In CRN mode this uses a
slot-indexed, per-timestep distribution so durations do not depend on call
order within the timestep.
"""
function _sample_recovery_draws(disease, sim, uids::UIDs, ti::Int)
    isempty(uids) && return Float64[]

    mean = disease.dur_inf / sim.pars.dt

    if crn_enabled()
        md = module_data(disease)
        dist = StarsimDist(Symbol(md.name, :_recovery), _recovery_dist(disease.recovery_dist, mean))
        init_dist!(dist; base_seed=sim.pars.rand_seed, trace=string(md.name, ".recovery"))
        set_slots!(dist, sim.people.slot)
        jump_dt!(dist, ti)
        return rvs(dist, uids)
    end

    draws = Vector{Float64}(undef, length(uids))
    @inbounds for i in eachindex(uids.values)
        draws[i] = _sample_dur(disease.recovery_dist, mean, disease.rng)
    end
    return draws
end

# ============================================================================
# Disease base
# ============================================================================

"""
    DiseaseData

Common mutable data for all diseases.
"""
mutable struct DiseaseData
    mod::ModuleData
    init_prev::Any       # Number, distribution, or function for initial prevalence
    beta::Any            # Dict{Symbol,Float64} mapping network_name → beta, or Float64
    beta_per_dt::Dict{Symbol, Float64}  # Computed per-dt beta
end

function DiseaseData(name::Symbol; init_prev=0.01, beta=0.05, label::String=string(name))
    md = ModuleData(name; label=label)
    beta_dict = beta isa Dict ? Dict{Symbol, Float64}(Symbol(k) => Float64(v) for (k,v) in beta) :
                Dict{Symbol, Float64}()
    DiseaseData(md, init_prev, beta, beta_dict)
end

"""
    disease_data(d::AbstractDisease) → DiseaseData

Return the DiseaseData for a disease. Concrete types must implement this.
"""
function disease_data end

module_data(d::AbstractDisease) = disease_data(d).mod

export DiseaseData, disease_data

# ============================================================================
# Infection — extends Disease with susceptibility/transmission states
# ============================================================================

"""
    InfectionData

Additional mutable data for infections (diseases with transmission).
"""
mutable struct InfectionData
    dd::DiseaseData

    # Core states
    susceptible::StateVector{Bool, Vector{Bool}}
    infected::StateVector{Bool, Vector{Bool}}
    ti_infected::StateVector{Float64, Vector{Float64}}

    # Relative susceptibility and transmissibility
    rel_sus::StateVector{Float64, Vector{Float64}}
    rel_trans::StateVector{Float64, Vector{Float64}}

    # Infection log
    infection_sources::Vector{Tuple{Int, Int, Int}}  # (target, source, timestep)

    # CRN-safe pairwise random number generator for transmission
    trans_rng::Union{MultiRandom, Nothing}
end

function InfectionData(name::Symbol; init_prev=0.01, beta=0.05, label::String=string(name))
    dd = DiseaseData(name; init_prev=init_prev, beta=beta, label=label)
    InfectionData(
        dd,
        BoolState(:susceptible; default=true, label="Susceptible"),
        BoolState(:infected; default=false, label="Infected"),
        FloatState(:ti_infected; default=Inf, label="Time infected"),
        FloatState(:rel_sus; default=1.0, label="Relative susceptibility"),
        FloatState(:rel_trans; default=1.0, label="Relative transmissibility"),
        Tuple{Int, Int, Int}[],
        nothing
    )
end

export InfectionData

# ============================================================================
# SIR — Susceptible-Infected-Recovered
# ============================================================================

"""
    SIR <: AbstractInfection

Standard SIR (Susceptible → Infected → Recovered) disease model.

# Keyword arguments
- `name::Symbol` — disease name (default `:sir`)
- `init_prev::Real` — initial prevalence (default 0.01)
- `beta::Union{Real, Dict}` — transmission rate (default 0.1)
- `dur_inf::Real` — mean duration of infection in years (default 6.0)
- `p_death::Real` — probability of death given infection (default 0.01)

# Example
```julia
sir = SIR(beta=0.1, dur_inf=10/365)
```
"""
mutable struct SIR <: AbstractInfection
    infection::InfectionData

    # SIR-specific states
    recovered::StateVector{Bool, Vector{Bool}}
    ti_recovered::StateVector{Float64, Vector{Float64}}

    # Parameters
    dur_inf::Float64
    p_death::Float64
    recovery_dist::Symbol  # :lognormal (Python-compatible) or :exponential (dt-invariant)

    rng::StableRNG
end

function SIR(;
    name::Symbol = :sir,
    init_prev::Real = 0.01,
    beta::Union{Real, Dict} = 0.1,
    dur_inf::Real = 6.0,
    p_death::Real = 0.01,
    recovery_dist::Symbol = :lognormal,
)
    inf = InfectionData(name; init_prev=init_prev, beta=beta, label="SIR")

    SIR(
        inf,
        BoolState(:recovered; default=false, label="Recovered"),
        FloatState(:ti_recovered; default=Inf, label="Time recovered"),
        Float64(dur_inf),
        Float64(p_death),
        recovery_dist,
        StableRNG(0)
    )
end

disease_data(d::SIR) = d.infection.dd
module_data(d::SIR) = d.infection.dd.mod

function init_pre!(d::SIR, sim)
    md = module_data(d)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    # Register states with People
    infection_states = [d.infection.susceptible, d.infection.infected,
                        d.infection.ti_infected, d.infection.rel_sus,
                        d.infection.rel_trans, d.recovered, d.ti_recovered]
    for s in infection_states
        add_module_state!(sim.people, s)
    end

    # Initialize CRN MultiRandom for transmission if CRN is enabled
    if crn_enabled()
        mr = MultiRandom(Symbol(md.name, :_transmission))
        init_dist!(mr; base_seed=sim.pars.rand_seed,
                   trace=string(md.name, ".transmission"))
        set_slots!(mr, sim.people.slot)
        d.infection.trans_rng = mr
    end

    # Compute beta per dt
    validate_beta!(d, sim)

    # Initialize results
    npts = md.t.npts
    define_results!(d,
        Result(:new_infections; npts=npts, label="New infections"),
        Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Result(:n_infected; npts=npts, label="Infected", scale=false),
        Result(:n_recovered; npts=npts, label="Recovered", scale=false),
        Result(:prevalence; npts=npts, label="Prevalence", scale=false),
    )

    md.initialized = true
    return d
end

"""Compute per-dt betas from the rate-based beta parameter."""
function validate_beta!(d::SIR, sim)
    dd = disease_data(d)
    dt = sim.pars.dt

    if dd.beta isa Real
        # Apply same beta to all networks
        for (name, _) in sim.networks
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(dd.beta) * dt)
        end
    elseif dd.beta isa Dict
        for (name, b) in dd.beta
            dd.beta_per_dt[Symbol(name)] = 1.0 - exp(-Float64(b) * dt)
        end
    end
    return d
end

function init_post!(d::SIR, sim)
    # Seed initial infections
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)

    # Choose agents to infect
    infect_uids = UIDs(active[randperm(d.rng, n)[1:n_infect]])

    # Set infected states
    d.infection.susceptible[infect_uids] = false
    d.infection.infected[infect_uids] = true
    d.infection.ti_infected[infect_uids] = 1.0  # Infected at first timestep

    draws = _sample_recovery_draws(d, sim, infect_uids, 1)
    for (i, u) in pairs(infect_uids.values)
        d.ti_recovered.raw[u] = 1.0 + draws[i]
    end

    return d
end

"""
    step_state!(d::SIR, sim)

Process state transitions: infected → recovered (or dead).
"""
function step_state!(d::SIR, sim)
    ti = module_data(d).t.ti
    ti_f = Float64(ti)
    active = sim.people.auids.values

    infected_raw = d.infection.infected.raw
    ti_rec_raw = d.ti_recovered.raw
    recovered_raw = d.recovered.raw

    if d.p_death > 0.0
        # With disease death: collect UIDs first, then process
        rec_uids = state_lte(d.ti_recovered, ti_f)
        alive_rec = intersect(rec_uids, uids(d.infection.infected))

        if !isempty(alive_rec)
            death_probs = rand(d.rng, length(alive_rec))
            death_mask = death_probs .< d.p_death
            death_uids = UIDs(alive_rec.values[death_mask])
            recover_uids = UIDs(alive_rec.values[.!death_mask])

            if !isempty(death_uids)
                request_death!(sim.people, death_uids, ti)
            end
            if !isempty(recover_uids)
                d.infection.infected[recover_uids] = false
                d.recovered[recover_uids] = true
            end
        end
    else
        # Common case: no death, zero-allocation loop
        @inbounds for u in active
            if infected_raw[u] && ti_rec_raw[u] <= ti_f
                infected_raw[u] = false
                recovered_raw[u] = true
            end
        end
    end

    return d
end

"""
    infect!(d::SIR, sim)

Compute transmission across all networks. When CRN is enabled and a
`MultiRandom` is available, uses pairwise XOR-combined random numbers
for CRN-safe transmission. Otherwise uses the standard per-edge RNG.
"""
function infect!(d::SIR, sim)
    md = module_data(d)
    ti = md.t.ti
    dd = disease_data(d)
    trans_rng = d.infection.trans_rng

    # Jump the MultiRandom to current timestep if CRN is active
    if trans_rng !== nothing
        jump_dt!(trans_rng, ti)
    end

    # Snapshot infected/susceptible state for synchronous update (matching Python)
    infected_snap = copy(d.infection.infected.raw)
    susceptible_snap = copy(d.infection.susceptible.raw)

    # Collect all candidate infections across networks
    all_targets = Int[]
    all_sources = Int[]

    for (net_name, net) in sim.networks
        edges = network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        if trans_rng !== nothing
            _find_infections_crn!(all_targets, all_sources, d, edges, beta_dt, trans_rng, net, infected_snap, susceptible_snap)
        else
            _find_infections!(all_targets, all_sources, d, edges, beta_dt, net, infected_snap, susceptible_snap)
        end
    end

    # Deduplicate targets (first occurrence wins, matching Python's unique())
    new_infections = 0
    seen = Set{Int}()
    for k in eachindex(all_targets)
        t = all_targets[k]
        if !(t in seen)
            push!(seen, t)
            _do_infection!(d, sim, t, all_sources[k], ti)
            new_infections += 1
        end
    end

    return new_infections
end

"""Collect infection candidates using snapshot state (synchronous update)."""
function _find_infections!(targets::Vector{Int}, sources::Vector{Int},
        d::SIR, edges::Edges, beta_dt::Float64, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)
    bidir = network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw
    rng = d.rng

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]

        # Try src → trg
        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rand(rng) < p
                push!(targets, trg)
                push!(sources, src)
            end
        end

        # Bidirectional: try trg → src
        if bidir
            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rand(rng) < p
                    push!(targets, src)
                    push!(sources, trg)
                end
            end
        end
    end
    return
end

"""CRN-safe infection candidate collection using snapshot state."""
function _find_infections_crn!(targets::Vector{Int}, sources::Vector{Int},
        d::SIR, edges::Edges, beta_dt::Float64, trans_rng::MultiRandom, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)

    # Batch draws for forward direction
    src_uids = UIDs(edges.p1)
    trg_uids = UIDs(edges.p2)
    rands_fwd = multi_rvs(trans_rng, src_uids, trg_uids)

    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw

    for i in 1:n_edges
        src = edges.p1[i]
        trg = edges.p2[i]
        edge_beta = edges.beta[i]

        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rands_fwd[i] < p
                push!(targets, trg)
                push!(sources, src)
            end
        end
    end

    # Bidirectional: reverse direction with swapped UIDs
    if network_data(net).bidirectional
        rands_rev = multi_rvs(trans_rng, trg_uids, src_uids)
        for i in 1:n_edges
            src = edges.p1[i]
            trg = edges.p2[i]
            edge_beta = edges.beta[i]

            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rands_rev[i] < p
                    push!(targets, src)
                    push!(sources, trg)
                end
            end
        end
    end

    return
end

"""Infect a single agent."""
function _do_infection!(d::SIR, sim, target::Int, source::Int, ti::Int)
    d.infection.susceptible.raw[target] = false
    d.infection.infected.raw[target] = true
    d.infection.ti_infected.raw[target] = Float64(ti)

    draw = _sample_recovery_draws(d, sim, UIDs([target]), ti)
    d.ti_recovered.raw[target] = Float64(ti) + draw[1]

    # Log infection
    push!(d.infection.infection_sources, (target, source, ti))
    return
end

function step!(d::SIR, sim)
    return infect!(d, sim)
end

function step_die!(d::SIR, death_uids::UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids] = false
    d.recovered[death_uids] = false
    return d
end

function update_results!(d::SIR, sim)
    md = module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw = d.infection.susceptible.raw
    inf_raw = d.infection.infected.raw
    rec_raw = d.recovered.raw

    n_sus = 0; n_inf = 0; n_rec = 0
    @inbounds for u in active
        n_sus += sus_raw[u]
        n_inf += inf_raw[u]
        n_rec += rec_raw[u]
    end

    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_infected][ti] = Float64(n_inf)
    md.results[:n_recovered][ti] = Float64(n_rec)

    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
    return d
end

function finalize!(d::SIR)
    md = module_data(d)
    # Calculate cumulative new infections per timestep from log
    for (target, source, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end

export SIR, infect!, validate_beta!

# ============================================================================
# SIS — Susceptible-Infected-Susceptible
# ============================================================================

"""
    SIS <: AbstractInfection

SIS disease model with waning immunity — matching Python starsim's SIS.

After recovery, agents gain temporary immunity (`imm_boost`) that wanes
exponentially at rate `waning` per year. Agents become re-susceptible as
immunity decays (rel_sus = max(0, 1 - immunity)).

# Keyword arguments
- `name::Symbol`: disease name (default `:sis`)
- `init_prev::Real`: initial prevalence fraction (default 0.01)
- `beta::Union{Real,Dict}`: transmission rate per year (default 0.05)
- `dur_inf::Real`: mean infectious duration in years (default 10.0)
- `waning::Real`: immunity waning rate per year (default 0.05)
- `imm_boost::Real`: immunity boost on infection (default 1.0)
- `p_death::Real`: probability of death on recovery (default 0.0)
- `recovery_dist::Symbol`: duration distribution (:lognormal or :exponential)
"""
mutable struct SIS <: AbstractInfection
    infection::InfectionData
    ti_recovered::StateVector{Float64, Vector{Float64}}
    immunity::StateVector{Float64, Vector{Float64}}
    dur_inf::Float64
    waning::Float64
    imm_boost::Float64
    p_death::Float64
    recovery_dist::Symbol
    rng::StableRNG
end

function SIS(;
    name::Symbol = :sis,
    init_prev::Real = 0.01,
    beta::Union{Real, Dict} = 0.05,
    dur_inf::Real = 10.0,
    waning::Real = 0.05,
    imm_boost::Real = 1.0,
    p_death::Real = 0.0,
    recovery_dist::Symbol = :lognormal,
)
    inf = InfectionData(name; init_prev=init_prev, beta=beta, label="SIS")
    SIS(inf,
        FloatState(:ti_recovered; default=Inf, label="Time recovered"),
        FloatState(:immunity; default=0.0, label="Immunity"),
        Float64(dur_inf), Float64(waning), Float64(imm_boost), Float64(p_death),
        recovery_dist,
        StableRNG(0))
end

disease_data(d::SIS) = d.infection.dd
module_data(d::SIS) = d.infection.dd.mod

function init_pre!(d::SIS, sim)
    md = module_data(d)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    # Register states
    for s in [d.infection.susceptible, d.infection.infected,
              d.infection.ti_infected, d.infection.rel_sus, d.infection.rel_trans,
              d.ti_recovered, d.immunity]
        add_module_state!(sim.people, s)
    end

    # Initialize CRN MultiRandom for transmission if CRN is enabled
    if crn_enabled()
        mr = MultiRandom(Symbol(md.name, :_transmission))
        init_dist!(mr; base_seed=sim.pars.rand_seed,
                   trace=string(md.name, ".transmission"))
        set_slots!(mr, sim.people.slot)
        d.infection.trans_rng = mr
    end

    # Compute beta per dt
    validate_beta!(d, sim)

    npts = md.t.npts
    define_results!(d,
        Result(:new_infections; npts=npts, label="New infections"),
        Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Result(:n_infected; npts=npts, label="Infected", scale=false),
        Result(:prevalence; npts=npts, label="Prevalence", scale=false),
        Result(:rel_sus; npts=npts, label="Relative susceptibility", scale=false),
    )

    md.initialized = true
    return d
end

function validate_beta!(d::SIS, sim)
    dd = disease_data(d)
    dt = sim.pars.dt
    if dd.beta isa Real
        for (name, _) in sim.networks
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(dd.beta) * dt)
        end
    elseif dd.beta isa Dict
        for (name, b) in dd.beta
            dd.beta_per_dt[Symbol(name)] = 1.0 - exp(-Float64(b) * dt)
        end
    end
    return d
end

function init_post!(d::SIS, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)

    infect_uids = UIDs(active[randperm(d.rng, n)[1:n_infect]])

    draws = _sample_recovery_draws(d, sim, infect_uids, 1)
    for (i, u) in pairs(infect_uids.values)
        d.infection.susceptible.raw[u] = false
        d.infection.infected.raw[u] = true
        d.infection.ti_infected.raw[u] = 1.0
        d.immunity.raw[u] += d.imm_boost
        d.ti_recovered.raw[u] = 1.0 + draws[i]
    end
    return d
end

function step_state!(d::SIS, sim)
    ti = module_data(d).t.ti
    ti_f = Float64(ti)

    # Recovery: infected agents whose ti_recovered <= ti
    for u in sim.people.auids.values
        @inbounds if d.infection.infected.raw[u]
            if d.ti_recovered.raw[u] <= ti_f
                if d.p_death > 0.0 && rand(d.rng) < d.p_death
                    request_death!(sim.people, UIDs([u]), ti)
                else
                    d.infection.infected.raw[u] = false
                    d.infection.susceptible.raw[u] = true
                end
            end
        end
    end

    # Update immunity (waning)
    _update_immunity!(d, sim)
    return d
end

"""Exponential immunity waning, matching Python's SIS.update_immunity."""
function _update_immunity!(d::SIS, sim)
    waning_prob = 1.0 - exp(-d.waning * sim.pars.dt)  # Rate → probability conversion
    imm_raw = d.immunity.raw
    rel_sus_raw = d.infection.rel_sus.raw
    @inbounds for u in sim.people.auids.values
        if imm_raw[u] > 0.0
            imm_raw[u] *= (1.0 - waning_prob)
            rel_sus_raw[u] = max(0.0, 1.0 - imm_raw[u])
        end
    end
    return
end

function step!(d::SIS, sim)
    return infect!(d, sim)
end

"""Transmission logic for SIS. Uses MultiRandom when CRN is enabled."""
function infect!(d::SIS, sim)
    md = module_data(d)
    ti = md.t.ti
    dd = disease_data(d)
    trans_rng = d.infection.trans_rng

    # Jump the MultiRandom to current timestep if CRN is active
    if trans_rng !== nothing
        jump_dt!(trans_rng, ti)
    end

    # Snapshot infected/susceptible state for synchronous update (matching Python)
    infected_snap = copy(d.infection.infected.raw)
    susceptible_snap = copy(d.infection.susceptible.raw)

    all_targets = Int[]
    all_sources = Int[]

    for (net_name, net) in sim.networks
        edges = network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        if trans_rng !== nothing
            _find_infections_crn_sis!(all_targets, d, edges, beta_dt, trans_rng, net, infected_snap, susceptible_snap)
        else
            _find_infections_sis!(all_targets, d, edges, beta_dt, net, infected_snap, susceptible_snap)
        end
    end

    # Deduplicate and apply (set_prognoses equivalent)
    new_targets = Int[]
    seen = Set{Int}()
    ti_f = Float64(ti)
    for t in all_targets
        if !(t in seen)
            push!(seen, t)
            push!(new_targets, t)
        end
    end

    draws = _sample_recovery_draws(d, sim, UIDs(new_targets), ti)
    for (i, t) in pairs(new_targets)
        d.infection.susceptible.raw[t] = false
        d.infection.infected.raw[t] = true
        d.infection.ti_infected.raw[t] = ti_f
        d.immunity.raw[t] += d.imm_boost
        d.ti_recovered.raw[t] = ti_f + draws[i]
    end

    return length(new_targets)
end

"""Collect SIS infection candidates using snapshot state."""
function _find_infections_sis!(targets::Vector{Int},
        d::SIS, edges::Edges, beta_dt::Float64, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)
    bidir = network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw
    rng = d.rng

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]

        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rand(rng) < p
                push!(targets, trg)
            end
        end

        if bidir
            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rand(rng) < p
                    push!(targets, src)
                end
            end
        end
    end
    return
end

"""CRN-safe SIS infection candidate collection using snapshot state."""
function _find_infections_crn_sis!(targets::Vector{Int},
        d::SIS, edges::Edges, beta_dt::Float64, trans_rng::MultiRandom, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)

    src_uids = UIDs(edges.p1)
    trg_uids = UIDs(edges.p2)
    rands_fwd = multi_rvs(trans_rng, src_uids, trg_uids)

    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw

    for i in 1:n_edges
        src = edges.p1[i]
        trg = edges.p2[i]
        edge_beta = edges.beta[i]

        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rands_fwd[i] < p
                push!(targets, trg)
            end
        end
    end

    if network_data(net).bidirectional
        rands_rev = multi_rvs(trans_rng, trg_uids, src_uids)
        for i in 1:n_edges
            src = edges.p1[i]
            trg = edges.p2[i]
            edge_beta = edges.beta[i]

            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rands_rev[i] < p
                    push!(targets, src)
                end
            end
        end
    end

    return
end

function step_die!(d::SIS, death_uids::UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids] = false
    return d
end

function update_results!(d::SIS, sim)
    md = module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw = d.infection.susceptible.raw
    inf_raw = d.infection.infected.raw
    rel_sus_raw = d.infection.rel_sus.raw

    n_sus = 0; n_inf = 0; sum_rel_sus = 0.0
    @inbounds for u in active
        n_sus += sus_raw[u]
        n_inf += inf_raw[u]
        sum_rel_sus += rel_sus_raw[u]
    end

    n_total = Float64(length(active))
    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_infected][ti] = Float64(n_inf)
    md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
    md.results[:rel_sus][ti] = n_total > 0.0 ? sum_rel_sus / n_total : 0.0
    return d
end

function finalize!(d::SIS)
    md = module_data(d)
    return d
end

export SIS

# ============================================================================
# SEIR — Susceptible-Exposed-Infected-Recovered
# ============================================================================

"""
    SEIR <: AbstractInfection

SEIR disease model with an exposed (latent) period before infectiousness.
Commonly used for diseases like measles, influenza, and COVID-19.

# Keyword arguments
- `name::Symbol` — disease name (default `:seir`)
- `init_prev::Real` — initial prevalence of infected (default 0.01)
- `beta::Union{Real, Dict}` — transmission rate (default 0.05)
- `dur_exp::Real` — mean duration of exposed/latent period (default 5.0 days)
- `dur_inf::Real` — mean duration of infectious period (default 10.0 days)
- `p_death::Real` — probability of death given infection (default 0.0)

# Example
```julia
seir = SEIR(beta=0.3, dur_exp=8.0, dur_inf=11.0)  # measles-like
```
"""
mutable struct SEIR <: AbstractInfection
    infection::InfectionData

    # SEIR-specific states
    exposed::StateVector{Bool, Vector{Bool}}
    recovered::StateVector{Bool, Vector{Bool}}
    ti_exposed::StateVector{Float64, Vector{Float64}}
    ti_recovered::StateVector{Float64, Vector{Float64}}

    # Parameters
    dur_exp::Float64
    dur_inf::Float64
    p_death::Float64
    recovery_dist::Symbol

    rng::StableRNG
end

function SEIR(;
    name::Symbol = :seir,
    init_prev::Real = 0.01,
    beta::Union{Real, Dict} = 0.05,
    dur_exp::Real = 5.0,
    dur_inf::Real = 10.0,
    p_death::Real = 0.0,
    recovery_dist::Symbol = :lognormal,
)
    inf = InfectionData(name; init_prev=init_prev, beta=beta, label="SEIR")

    SEIR(
        inf,
        BoolState(:exposed; default=false, label="Exposed"),
        BoolState(:recovered; default=false, label="Recovered"),
        FloatState(:ti_exposed; default=Inf, label="Time exposed"),
        FloatState(:ti_recovered; default=Inf, label="Time recovered"),
        Float64(dur_exp),
        Float64(dur_inf),
        Float64(p_death),
        recovery_dist,
        StableRNG(0)
    )
end

disease_data(d::SEIR) = d.infection.dd
module_data(d::SEIR) = d.infection.dd.mod

function init_pre!(d::SEIR, sim)
    md = module_data(d)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    infection_states = [d.infection.susceptible, d.infection.infected,
                        d.infection.ti_infected, d.infection.rel_sus,
                        d.infection.rel_trans, d.exposed, d.recovered,
                        d.ti_exposed, d.ti_recovered]
    for s in infection_states
        add_module_state!(sim.people, s)
    end

    # CRN MultiRandom
    if crn_enabled()
        mr = MultiRandom(Symbol(md.name, :_transmission))
        init_dist!(mr; base_seed=sim.pars.rand_seed,
                   trace=string(md.name, ".transmission"))
        set_slots!(mr, sim.people.slot)
        d.infection.trans_rng = mr
    end

    validate_beta!(d, sim)

    npts = md.t.npts
    define_results!(d,
        Result(:new_infections; npts=npts, label="New infections"),
        Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Result(:n_exposed; npts=npts, label="Exposed", scale=false),
        Result(:n_infected; npts=npts, label="Infected", scale=false),
        Result(:n_recovered; npts=npts, label="Recovered", scale=false),
        Result(:prevalence; npts=npts, label="Prevalence", scale=false),
    )

    md.initialized = true
    return d
end

function validate_beta!(d::SEIR, sim)
    dd = disease_data(d)
    dt = sim.pars.dt
    if dd.beta isa Real
        for (name, _) in sim.networks
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(dd.beta) * dt)
        end
    elseif dd.beta isa Dict
        for (name, b) in dd.beta
            dd.beta_per_dt[Symbol(name)] = 1.0 - exp(-Float64(b) * dt)
        end
    end
    return d
end

function init_post!(d::SEIR, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)

    infect_uids = UIDs(active[randperm(d.rng, n)[1:n_infect]])

    # Initially infected agents (already past exposed phase)
    d.infection.susceptible[infect_uids] = false
    d.infection.infected[infect_uids] = true
    d.infection.ti_infected[infect_uids] = 1.0

    draws = _sample_recovery_draws(d, sim, infect_uids, 1)
    for (i, u) in pairs(infect_uids.values)
        d.ti_recovered.raw[u] = 1.0 + draws[i]
    end

    return d
end

"""
    step_state!(d::SEIR, sim)

Process state transitions: exposed → infected, infected → recovered/dead.
"""
function step_state!(d::SEIR, sim)
    ti = module_data(d).t.ti
    ti_f = Float64(ti)
    dt = sim.pars.dt
    active = sim.people.auids.values

    exposed_raw = d.exposed.raw
    ti_exposed_raw = d.ti_exposed.raw
    infected_raw = d.infection.infected.raw
    ti_infected_raw = d.infection.ti_infected.raw
    ti_rec_raw = d.ti_recovered.raw
    recovered_raw = d.recovered.raw

    # Exposed → Infected (after latent period)
    dur_exp_ts = d.dur_exp / dt
    rng = d.rng
    new_infectious = Int[]
    @inbounds for u in active
        if exposed_raw[u]
            if (ti_f - ti_exposed_raw[u]) >= dur_exp_ts
                exposed_raw[u] = false
                infected_raw[u] = true
                ti_infected_raw[u] = ti_f
                push!(new_infectious, u)
            end
        end
    end

    draws = _sample_recovery_draws(d, sim, UIDs(new_infectious), ti)
    @inbounds for (i, u) in pairs(new_infectious)
        ti_rec_raw[u] = ti_f + draws[i]
    end

    # Infected → Recovered (or dead)
    if d.p_death > 0.0
        rec_uids = state_lte(d.ti_recovered, ti_f)
        alive_rec = intersect(rec_uids, uids(d.infection.infected))
        if !isempty(alive_rec)
            death_probs = rand(rng, length(alive_rec))
            death_mask = death_probs .< d.p_death
            death_uids = UIDs(alive_rec.values[death_mask])
            recover_uids = UIDs(alive_rec.values[.!death_mask])
            if !isempty(death_uids)
                request_death!(sim.people, death_uids, ti)
            end
            if !isempty(recover_uids)
                d.infection.infected[recover_uids] = false
                d.recovered[recover_uids] = true
            end
        end
    else
        @inbounds for u in active
            if infected_raw[u] && ti_rec_raw[u] <= ti_f
                infected_raw[u] = false
                recovered_raw[u] = true
            end
        end
    end

    return d
end

"""
    infect!(d::SEIR, sim)

Compute transmission. Newly infected agents enter the Exposed state.
"""
function infect!(d::SEIR, sim)
    md = module_data(d)
    ti = md.t.ti
    dd = disease_data(d)
    trans_rng = d.infection.trans_rng

    if trans_rng !== nothing
        jump_dt!(trans_rng, ti)
    end

    # Snapshot infected/susceptible state for synchronous update (matching Python)
    infected_snap = copy(d.infection.infected.raw)
    susceptible_snap = copy(d.infection.susceptible.raw)

    all_targets = Int[]
    all_sources = Int[]

    for (net_name, net) in sim.networks
        edges = network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        if trans_rng !== nothing
            _find_infections_crn_seir!(all_targets, all_sources, d, edges, beta_dt, trans_rng, net, infected_snap, susceptible_snap)
        else
            _find_infections_seir!(all_targets, all_sources, d, edges, beta_dt, net, infected_snap, susceptible_snap)
        end
    end

    # Deduplicate and apply
    new_infections = 0
    seen = Set{Int}()
    for k in eachindex(all_targets)
        t = all_targets[k]
        if !(t in seen)
            push!(seen, t)
            _do_exposure!(d, sim, t, all_sources[k], ti)
            new_infections += 1
        end
    end

    return new_infections
end

"""Collect SEIR infection candidates using snapshot state."""
function _find_infections_seir!(targets::Vector{Int}, sources::Vector{Int},
        d::SEIR, edges::Edges, beta_dt::Float64, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)
    bidir = network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw
    rng = d.rng

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]

        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rand(rng) < p
                push!(targets, trg)
                push!(sources, src)
            end
        end

        if bidir
            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rand(rng) < p
                    push!(targets, src)
                    push!(sources, trg)
                end
            end
        end
    end
    return
end

"""CRN-safe SEIR infection candidate collection using snapshot state."""
function _find_infections_crn_seir!(targets::Vector{Int}, sources::Vector{Int},
        d::SEIR, edges::Edges, beta_dt::Float64, trans_rng::MultiRandom, net,
        infected_snap::Vector{Bool}, susceptible_snap::Vector{Bool})
    n_edges = length(edges)

    src_uids = UIDs(edges.p1)
    trg_uids = UIDs(edges.p2)
    rands_fwd = multi_rvs(trans_rng, src_uids, trg_uids)

    rel_trans_raw = d.infection.rel_trans.raw
    rel_sus_raw = d.infection.rel_sus.raw

    for i in 1:n_edges
        src = edges.p1[i]
        trg = edges.p2[i]
        edge_beta = edges.beta[i]

        if infected_snap[src] && susceptible_snap[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rands_fwd[i] < p
                push!(targets, trg)
                push!(sources, src)
            end
        end
    end

    if network_data(net).bidirectional
        rands_rev = multi_rvs(trans_rng, trg_uids, src_uids)
        for i in 1:n_edges
            src = edges.p1[i]
            trg = edges.p2[i]
            edge_beta = edges.beta[i]

            if infected_snap[trg] && susceptible_snap[src]
                p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
                if rands_rev[i] < p
                    push!(targets, src)
                    push!(sources, trg)
                end
            end
        end
    end

    return
end

"""Move a susceptible agent to Exposed state."""
function _do_exposure!(d::SEIR, sim, target::Int, source::Int, ti::Int)
    d.infection.susceptible.raw[target] = false
    d.exposed.raw[target] = true
    d.ti_exposed.raw[target] = Float64(ti)
    push!(d.infection.infection_sources, (target, source, ti))
    return
end

function step!(d::SEIR, sim)
    return infect!(d, sim)
end

function step_die!(d::SEIR, death_uids::UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids] = false
    d.exposed[death_uids] = false
    d.recovered[death_uids] = false
    return d
end

function update_results!(d::SEIR, sim)
    md = module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw = d.infection.susceptible.raw
    exp_raw = d.exposed.raw
    inf_raw = d.infection.infected.raw
    rec_raw = d.recovered.raw

    n_sus = 0; n_exp = 0; n_inf = 0; n_rec = 0
    @inbounds for u in active
        n_sus += sus_raw[u]
        n_exp += exp_raw[u]
        n_inf += inf_raw[u]
        n_rec += rec_raw[u]
    end

    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_exposed][ti] = Float64(n_exp)
    md.results[:n_infected][ti] = Float64(n_inf)
    md.results[:n_recovered][ti] = Float64(n_rec)
    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_total > 0.0 ? (n_exp + n_inf) / n_total : 0.0
    return d
end

function finalize!(d::SEIR)
    md = module_data(d)
    for (target, source, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end

export SEIR
