"""
Rotavirus vaccination intervention with multi-dose schedule and cross-strain protection.
Port of Python `rotasim.interventions.RotaVaccination`.
"""

# ============================================================================
# RotaVaccination
# ============================================================================

"""
    RotaVaccination <: AbstractIntervention

Rotavirus vaccination with multi-dose schedule and cross-strain protection.

# Keyword arguments
- `start_year::Float64` — year to start vaccination
- `end_year::Float64` — year to stop (default Inf)
- `n_doses::Int` — number of doses (default 2)
- `dose_interval_days::Float64` — days between doses (default 28)
- `G_antigens::Vector{Int}` — covered G antigens (default [1])
- `P_antigens::Vector{Int}` — covered P antigens (default [8])
- `dose_effectiveness::Vector{Float64}` — effectiveness by dose
- `min_age_days::Float64` — minimum age in days (default 42)
- `max_age_days::Float64` — maximum age in days (default 365)
- `uptake_prob::Float64` — per-eligible probability (default 0.8)
- `waning_rate_mean::Float64` — mean waning time in days (default 365)
- `homotypic_efficacy::Float64` — efficacy multiplier for exact match (default 1.0)
- `partial_hetero_efficacy::Float64` — efficacy multiplier for partial match (default 0.6)
- `complete_hetero_efficacy::Float64` — efficacy multiplier for no match (default 0.3)
"""
mutable struct RotaVaccination <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData

    # Schedule parameters
    start_year::Float64
    end_year::Float64
    n_doses::Int
    dose_interval_days::Float64
    G_antigens::Vector{Int}
    P_antigens::Vector{Int}
    dose_effectiveness::Vector{Float64}
    min_age_days::Float64
    max_age_days::Float64
    uptake_prob::Float64
    waning_rate_mean::Float64
    homotypic_efficacy::Float64
    partial_hetero_efficacy::Float64
    complete_hetero_efficacy::Float64

    # Per-agent state
    doses_received::Starsim.StateVector{Int64, Vector{Int64}}
    last_dose_ti::Starsim.StateVector{Float64, Vector{Float64}}
    next_dose_due::Starsim.StateVector{Float64, Vector{Float64}}
    completed_schedule::Starsim.StateVector{Bool, Vector{Bool}}
    vx_waning_rate::Starsim.StateVector{Float64, Vector{Float64}}

    # Precomputed match efficacies (disease_name → efficacy)
    disease_match_efficacies::Dict{Symbol, Float64}
    covered_diseases::Vector{Rotavirus}

    rng::StableRNG
end

function RotaVaccination(;
    name::Symbol = :rota_vax,
    start_year::Real = 0.0,
    end_year::Real = Inf,
    n_doses::Int = 2,
    dose_interval_days::Real = 28.0,
    G_antigens::Vector{Int} = [1],
    P_antigens::Vector{Int} = [8],
    dose_effectiveness::Union{Nothing, Vector{Float64}} = nothing,
    min_age_days::Real = 42.0,
    max_age_days::Real = 365.0,
    uptake_prob::Real = 0.8,
    waning_rate_mean::Real = 365.0,
    homotypic_efficacy::Real = 1.0,
    partial_hetero_efficacy::Real = 0.6,
    complete_hetero_efficacy::Real = 0.3,
)
    md = Starsim.ModuleData(name; label="Rota vaccination")
    iv = Starsim.InterventionData(md, nothing, :rota)

    # Default dose effectiveness
    de = if dose_effectiveness === nothing
        if n_doses == 1
            [0.7]
        elseif n_doses == 2
            [0.6, 0.8]
        elseif n_doses == 3
            [0.5, 0.7, 0.85]
        else
            [0.4 + 0.4 * (i-1) / (n_doses-1) for i in 1:n_doses]
        end
    else
        dose_effectiveness
    end
    length(de) == n_doses || error("dose_effectiveness must have $n_doses values")

    RotaVaccination(
        iv,
        Float64(start_year), Float64(end_year), n_doses, Float64(dose_interval_days),
        G_antigens, P_antigens, de,
        Float64(min_age_days), Float64(max_age_days),
        Float64(uptake_prob), Float64(waning_rate_mean),
        Float64(homotypic_efficacy), Float64(partial_hetero_efficacy), Float64(complete_hetero_efficacy),
        Starsim.IntState(:vx_doses; default=0, label="Doses received"),
        Starsim.FloatState(:vx_last_ti; default=-Inf, label="Last dose time"),
        Starsim.FloatState(:vx_next_due; default=-Inf, label="Next dose due"),
        Starsim.BoolState(:vx_completed; default=false, label="Completed schedule"),
        Starsim.FloatState(:vx_waning_rate; default=0.0, label="Vaccine waning rate"),
        Dict{Symbol, Float64}(),
        Rotavirus[],
        StableRNG(0),
    )
end

Starsim.intervention_data(v::RotaVaccination) = v.iv

function Starsim.init_pre!(v::RotaVaccination, sim)
    md = Starsim.module_data(v)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    v.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Register states
    for s in [v.doses_received, v.last_dose_ti, v.next_dose_due, v.completed_schedule, v.vx_waning_rate]
        Starsim.add_module_state!(sim.people, s)
    end

    # Find covered diseases and precompute match efficacies
    v.covered_diseases = Rotavirus[]
    for (_, dis) in sim.diseases
        if dis isa Rotavirus
            push!(v.covered_diseases, dis)
            nm = Starsim.module_data(dis).name
            v.disease_match_efficacies[nm] = _compute_match_efficacy(v, dis)
        end
    end

    npts = md.t.npts
    Starsim.define_results!(v,
        Starsim.Result(:n_vaccinated; npts=npts, label="Agents vaccinated"),
    )

    md.initialized = true
    return v
end

function _compute_match_efficacy(v::RotaVaccination, disease::Rotavirus)
    is_homo = disease.G in v.G_antigens && disease.P in v.P_antigens
    is_homo && return v.homotypic_efficacy
    is_partial = disease.G in v.G_antigens || disease.P in v.P_antigens
    is_partial && return v.partial_hetero_efficacy
    return v.complete_hetero_efficacy
end

# ============================================================================
# Step
# ============================================================================

function Starsim.step!(v::RotaVaccination, sim)
    # Always update vaccine protection (waning)
    _update_vaccine_protection!(v, sim)

    md = Starsim.module_data(v)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    (year < v.start_year || year > v.end_year) && return v

    # Find eligible agents
    active = sim.people.auids.values
    eligible = Int[]
    min_age_years = v.min_age_days / 365.25
    max_age_years = v.max_age_days / 365.25
    dose_interval_ts = v.dose_interval_days / (sim.pars.dt * 365.25)

    @inbounds for u in active
        v.completed_schedule.raw[u] && continue
        age = sim.people.age.raw[u]
        (age < min_age_years || age > max_age_years) && continue

        doses = v.doses_received.raw[u]
        if doses == 0
            push!(eligible, u)
        elseif doses < v.n_doses && Float64(ti) >= v.next_dose_due.raw[u]
            push!(eligible, u)
        end
    end

    isempty(eligible) && return v

    # Random uptake
    vaccinated = Int[]
    for u in eligible
        if rand(v.rng) < v.uptake_prob
            push!(vaccinated, u)
        end
    end

    if !isempty(vaccinated)
        _vaccinate_agents!(v, sim, vaccinated)
    end

    if ti <= length(Starsim.module_results(v)[:n_vaccinated].values)
        Starsim.module_results(v)[:n_vaccinated][ti] = Float64(length(vaccinated))
    end

    return v
end

function _vaccinate_agents!(v::RotaVaccination, sim, uids::Vector{Int})
    ti = Starsim.module_data(v).t.ti
    dose_interval_ts = v.dose_interval_days / (sim.pars.dt * 365.25)

    for u in uids
        v.doses_received.raw[u] += 1
        v.last_dose_ti.raw[u] = Float64(ti)
        v.next_dose_due.raw[u] = Float64(ti) + dose_interval_ts

        if v.doses_received.raw[u] >= v.n_doses
            v.completed_schedule.raw[u] = true
        end

        # Set waning rate (1 / waning_duration_days)
        waning_days = max(1.0, v.waning_rate_mean + randn(v.rng) * v.waning_rate_mean * 0.3)
        v.vx_waning_rate.raw[u] = 1.0 / waning_days
    end
    return
end

"""Update vaccine protection with waning and apply to rel_sus."""
function _update_vaccine_protection!(v::RotaVaccination, sim)
    isempty(v.covered_diseases) && return
    dt_days = sim.pars.dt * 365.25
    active = sim.people.auids.values
    ti = Starsim.module_data(v).t.ti

    for disease in v.covered_diseases
        nm = Starsim.module_data(disease).name
        match_eff = v.disease_match_efficacies[nm]
        rel_sus_raw = disease.infection.rel_sus.raw

        @inbounds for u in active
            doses = v.doses_received.raw[u]
            doses <= 0 && continue

            # Dose-dependent effectiveness
            dose_idx = min(Int(doses), v.n_doses)
            dose_eff = v.dose_effectiveness[dose_idx]

            # Waning since last dose
            days_since = max(0.0, (Float64(ti) - v.last_dose_ti.raw[u]) * dt_days)
            waned_factor = exp(-v.vx_waning_rate.raw[u] * days_since)

            vx_eff = waned_factor * dose_eff * match_eff
            vx_sus = 1.0 - vx_eff

            # Use minimum susceptibility (most protective)
            if vx_sus < rel_sus_raw[u]
                rel_sus_raw[u] = vx_sus
            end
        end
    end
    return
end

"""
    get_vaccination_summary(v::RotaVaccination, sim)

Get summary statistics for the vaccination program.
"""
function get_vaccination_summary(v::RotaVaccination, sim)
    active = sim.people.auids.values
    n_total = length(active)
    n_any_dose = count(v.doses_received.raw[u] > 0 for u in active)
    n_completed = count(v.completed_schedule.raw[u] for u in active)
    dose_counts = Dict{Int,Int}()
    for d in 0:v.n_doses
        dose_counts[d] = count(v.doses_received.raw[u] == d for u in active)
    end
    mean_doses = n_any_dose > 0 ?
        sum(v.doses_received.raw[u] for u in active if v.doses_received.raw[u] > 0) / n_any_dose : 0.0
    return Dict{String,Any}(
        "total_agents"      => n_total,
        "received_any_dose" => n_any_dose,
        "completed_schedule"=> n_completed,
        "doses_by_number"   => dose_counts,
        "mean_doses"        => mean_doses,
    )
end

# ============================================================================
# Named vaccine constructors
# ============================================================================

"""
    Rotarix(; start_year, kwargs...)

Monovalent rotavirus vaccine (G1P[8]). Two oral doses at 6 and 10 weeks.
"""
function Rotarix(; start_year::Real=0.0, kwargs...)
    return RotaVaccination(;
        name               = :rotarix,
        start_year         = start_year,
        n_doses            = 2,
        dose_interval_days = 28.0,
        G_antigens         = [1],
        P_antigens         = [8],
        dose_effectiveness = [0.6, 0.85],
        min_age_days       = 42.0,
        max_age_days       = 365.0,
        homotypic_efficacy = 1.0,
        partial_hetero_efficacy = 0.6,
        complete_hetero_efficacy = 0.3,
        kwargs...,
    )
end

"""
    RotaTeq(; start_year, kwargs...)

Pentavalent rotavirus vaccine (G1, G2, G3, G4, P[8]). Three oral doses.
"""
function RotaTeq(; start_year::Real=0.0, kwargs...)
    return RotaVaccination(;
        name               = :rotateq,
        start_year         = start_year,
        n_doses            = 3,
        dose_interval_days = 28.0,
        G_antigens         = [1, 2, 3, 4],
        P_antigens         = [8],
        dose_effectiveness = [0.5, 0.7, 0.85],
        min_age_days       = 42.0,
        max_age_days       = 365.0,
        homotypic_efficacy = 1.0,
        partial_hetero_efficacy = 0.7,
        complete_hetero_efficacy = 0.35,
        kwargs...,
    )
end

"""
    Rotavac(; start_year, kwargs...)

Monovalent neonatal rotavirus vaccine (G9P[11]). Three oral doses.
"""
function Rotavac(; start_year::Real=0.0, kwargs...)
    return RotaVaccination(;
        name               = :rotavac,
        start_year         = start_year,
        n_doses            = 3,
        dose_interval_days = 28.0,
        G_antigens         = [9],
        P_antigens         = [11],
        dose_effectiveness = [0.4, 0.6, 0.75],
        min_age_days       = 42.0,
        max_age_days       = 365.0,
        homotypic_efficacy = 1.0,
        partial_hetero_efficacy = 0.5,
        complete_hetero_efficacy = 0.25,
        kwargs...,
    )
end
