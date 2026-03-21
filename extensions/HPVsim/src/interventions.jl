"""
HPV interventions: prophylactic vaccination, screening, treatment,
and therapeutic vaccination.

Ported from Python hpvsim/interventions.py and adapted to Starsim.jl's
intervention framework.
"""

# ============================================================================
# Treatment types
# ============================================================================

"""
    TreatmentType

Treatment modality for CIN lesions. Different types have different
stage-specific efficacies.
"""
@enum TreatmentType begin
    ABLATION   # Cryotherapy / thermal ablation — good for CIN1/2
    EXCISION   # LEEP / cold knife conization — better for CIN3+
    GENERIC    # Single efficacy for all stages
end

"""
    get_treatment_efficacy(ttype::TreatmentType, stage::Symbol) → Float64

Look up treatment efficacy by treatment type and CIN stage.
"""
function get_treatment_efficacy(ttype::TreatmentType, stage::Symbol)
    if ttype == ABLATION
        stage == :cin1   && return 0.90
        stage == :cin2   && return 0.85
        stage == :cin3   && return 0.75
        stage == :cancer && return 0.10
    elseif ttype == EXCISION
        stage == :cin1   && return 0.95
        stage == :cin2   && return 0.93
        stage == :cin3   && return 0.90
        stage == :cancer && return 0.50
    end
    return 0.85  # GENERIC default
end

# ============================================================================
# HPVVaccination — Prophylactic vaccination
# ============================================================================

"""
    HPVVaccination <: AbstractIntervention

Prophylactic HPV vaccination with genotype-specific efficacy and
multi-dose schedules.

# Keyword arguments
- `start_year::Float64` — year to start vaccination (default 0.0)
- `end_year::Float64` — year to stop (default Inf)
- `covered_genotypes::Vector{Symbol}` — genotypes with vaccine efficacy
- `genotype_efficacies::Dict{Symbol,Float64}` — per-genotype efficacy
- `n_doses::Int` — number of doses (default 2)
- `dose_efficacies::Vector{Float64}` — cumulative efficacy per dose
- `min_age::Float64` — minimum eligible age in years (default 9.0)
- `max_age::Float64` — maximum eligible age in years (default 14.0)
- `uptake_prob::Float64` — per-eligible acceptance probability (default 0.8)
- `waning_rate::Float64` — exponential waning rate per year (default 0.02)
- `sex::Symbol` — :female, :male, or :both (default :female)
"""
mutable struct HPVVaccination <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData

    # Schedule
    start_year::Float64
    end_year::Float64
    covered_genotypes::Vector{Symbol}
    genotype_efficacies::Dict{Symbol, Float64}
    n_doses::Int
    dose_efficacies::Vector{Float64}
    min_age::Float64
    max_age::Float64
    uptake_prob::Float64
    waning_rate::Float64
    sex::Symbol

    # Per-agent states
    vx_doses::Starsim.StateVector{Int64, Vector{Int64}}
    vx_last_ti::Starsim.StateVector{Float64, Vector{Float64}}
    vx_next_due::Starsim.StateVector{Float64, Vector{Float64}}
    vx_completed::Starsim.StateVector{Bool, Vector{Bool}}

    # Discovered diseases
    covered_diseases::Vector{HPVGenotype}
    disease_efficacies::Dict{Symbol, Float64}

    rng::StableRNG
end

function HPVVaccination(;
    name::Symbol = :hpv_vax,
    start_year::Real = 0.0,
    end_year::Real = Inf,
    covered_genotypes::Vector{Symbol} = [:hpv16, :hpv18],
    genotype_efficacies::Union{Dict{Symbol,Float64}, Nothing} = nothing,
    n_doses::Int = 2,
    dose_efficacies::Union{Nothing, Vector{Float64}} = nothing,
    min_age::Real = 9.0,
    max_age::Real = 14.0,
    uptake_prob::Real = 0.8,
    waning_rate::Real = 0.02,
    sex::Symbol = :female,
)
    md = Starsim.ModuleData(name; label="HPV vaccination")
    iv = Starsim.InterventionData(md, nothing, :hpv)

    # Default efficacies
    ge = if genotype_efficacies !== nothing
        genotype_efficacies
    else
        Dict{Symbol, Float64}(g => 0.95 for g in covered_genotypes)
    end

    de = if dose_efficacies !== nothing
        dose_efficacies
    else
        n_doses == 1 ? [0.85] :
        n_doses == 2 ? [0.85, 0.95] :
        [0.70, 0.90, 0.97]
    end

    HPVVaccination(
        iv,
        Float64(start_year), Float64(end_year),
        covered_genotypes, ge, n_doses, de,
        Float64(min_age), Float64(max_age),
        Float64(uptake_prob), Float64(waning_rate), sex,
        Starsim.IntState(:hpv_vx_doses; default=0, label="HPV vax doses"),
        Starsim.FloatState(:hpv_vx_last_ti; default=-Inf, label="HPV vax last time"),
        Starsim.FloatState(:hpv_vx_next_due; default=-Inf, label="HPV vax next due"),
        Starsim.BoolState(:hpv_vx_completed; default=false, label="HPV vax completed"),
        HPVGenotype[],
        Dict{Symbol, Float64}(),
        StableRNG(0),
    )
end

Starsim.intervention_data(v::HPVVaccination) = v.iv

function Starsim.init_pre!(v::HPVVaccination, sim)
    md = Starsim.module_data(v)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    v.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Register states
    for s in [v.vx_doses, v.vx_last_ti, v.vx_next_due, v.vx_completed]
        Starsim.add_module_state!(sim.people, s)
    end

    # Find covered diseases
    v.covered_diseases = HPVGenotype[]
    for (_, dis) in sim.diseases
        if dis isa HPVGenotype
            nm = dis.genotype
            if nm in v.covered_genotypes
                push!(v.covered_diseases, dis)
                v.disease_efficacies[Starsim.module_data(dis).name] = get(v.genotype_efficacies, nm, 0.0)
            end
        end
    end

    npts = md.t.npts
    Starsim.define_results!(v,
        Starsim.Result(:n_vaccinated; npts=npts, label="Agents vaccinated"),
        Starsim.Result(:n_doses_given; npts=npts, label="Doses given"),
    )

    md.initialized = true
    return v
end

function Starsim.step!(v::HPVVaccination, sim)
    # Always update vaccine protection (waning)
    _update_hpv_vaccine_protection!(v, sim)

    md = Starsim.module_data(v)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    (year < v.start_year || year > v.end_year) && return v

    # Find eligible agents
    active = sim.people.auids.values
    eligible = Int[]
    dose_interval_ts = 180.0 / (sim.pars.dt * 365.25)  # ~6 months between doses

    @inbounds for u in active
        v.vx_completed.raw[u] && continue

        # Sex eligibility
        if v.sex == :female && !sim.people.female.raw[u]
            continue
        elseif v.sex == :male && sim.people.female.raw[u]
            continue
        end

        age = sim.people.age.raw[u]
        (age < v.min_age || age > v.max_age) && continue

        doses = v.vx_doses.raw[u]
        if doses == 0
            push!(eligible, u)
        elseif doses < v.n_doses && Float64(ti) >= v.vx_next_due.raw[u]
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
        _give_hpv_doses!(v, sim, vaccinated)
    end

    if ti <= length(Starsim.module_results(v)[:n_vaccinated].values)
        Starsim.module_results(v)[:n_vaccinated][ti] = Float64(length(vaccinated))
        Starsim.module_results(v)[:n_doses_given][ti] = Float64(length(vaccinated))
    end

    return v
end

function _give_hpv_doses!(v::HPVVaccination, sim, uids::Vector{Int})
    ti = Starsim.module_data(v).t.ti
    dose_interval_ts = 180.0 / (sim.pars.dt * 365.25)

    for u in uids
        v.vx_doses.raw[u] += 1
        v.vx_last_ti.raw[u] = Float64(ti)
        v.vx_next_due.raw[u] = Float64(ti) + dose_interval_ts

        if v.vx_doses.raw[u] >= v.n_doses
            v.vx_completed.raw[u] = true
        end
    end
    return
end

"""Update vaccine protection with waning and apply to rel_sus.

Vaccine protection is SUBTRACTED from current rel_sus (additive with natural immunity),
matching Python hpvsim where vaccine NAb immunity is summed with natural immunity."""
function _update_hpv_vaccine_protection!(v::HPVVaccination, sim)
    isempty(v.covered_diseases) && return
    dt = sim.pars.dt
    active = sim.people.auids.values
    ti = Starsim.module_data(v).t.ti

    for disease in v.covered_diseases
        nm = Starsim.module_data(disease).name
        genotype_eff = get(v.disease_efficacies, nm, 0.0)
        rel_sus_raw = disease.infection.rel_sus.raw

        @inbounds for u in active
            doses = v.vx_doses.raw[u]
            doses <= 0 && continue

            dose_idx = min(Int(doses), v.n_doses)
            dose_eff = v.dose_efficacies[dose_idx]

            # Waning since last dose
            time_since = max(0.0, (Float64(ti) - v.vx_last_ti.raw[u]) * dt)
            waned_factor = exp(-v.waning_rate * time_since)

            vx_eff = waned_factor * dose_eff * genotype_eff
            # Additive: subtract vaccine protection from current rel_sus
            # (natural immunity already reduced rel_sus from 1.0 in connector step)
            rel_sus_raw[u] = max(0.0, rel_sus_raw[u] - vx_eff)
        end
    end
    return
end

# ============================================================================
# HPVScreening — Cervical cancer screening
# ============================================================================

"""
    HPVScreening <: AbstractIntervention

Cervical cancer screening with configurable test type, stage-dependent
sensitivity, and treatment cascade with selectable treatment modality.

# Keyword arguments
- `test_type::Symbol` — :pap, :hpv_dna, or :via (default :pap)
- `sensitivity::Float64` — test sensitivity for CIN2+ (default type-specific)
- `specificity::Float64` — test specificity (default type-specific)
- `sensitivity_cin1::Float64` — sensitivity for CIN1 (default type-specific)
- `sensitivity_cancer::Float64` — sensitivity for cancer (default type-specific)
- `start_year::Float64` — year to begin screening (default 0.0)
- `end_year::Float64` — year to end screening (default Inf)
- `screen_prob::Float64` — per-timestep screening probability (default 0.01)
- `min_age::Float64` — minimum screening age in years (default 25.0)
- `max_age::Float64` — maximum screening age in years (default 65.0)
- `treat_prob::Float64` — probability of treatment after positive screen (default 0.9)
- `treat_efficacy::Float64` — treatment cure rate for CIN (default 0.85, used if treatment_type=GENERIC)
- `treatment_type::TreatmentType` — ABLATION, EXCISION, or GENERIC (default GENERIC)
"""
mutable struct HPVScreening <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData

    test_type::Symbol
    sensitivity::Float64       # CIN2+ sensitivity
    specificity::Float64
    sensitivity_cin1::Float64  # Stage-specific sensitivities
    sensitivity_cancer::Float64
    start_year::Float64
    end_year::Float64
    screen_prob::Float64
    min_age::Float64
    max_age::Float64
    treat_prob::Float64
    treat_efficacy::Float64
    treatment_type::TreatmentType

    # Discovered diseases
    hpv_diseases::Vector{HPVGenotype}

    rng::StableRNG
end

function HPVScreening(;
    name::Symbol = :hpv_screening,
    test_type::Symbol = :pap,
    sensitivity::Union{Real, Nothing} = nothing,
    specificity::Union{Real, Nothing} = nothing,
    sensitivity_cin1::Union{Real, Nothing} = nothing,
    sensitivity_cancer::Union{Real, Nothing} = nothing,
    start_year::Real = 0.0,
    end_year::Real = Inf,
    screen_prob::Real = 0.01,
    min_age::Real = 25.0,
    max_age::Real = 65.0,
    treat_prob::Real = 0.9,
    treat_efficacy::Real = 0.85,
    treatment_type::TreatmentType = GENERIC,
)
    md = Starsim.ModuleData(name; label="HPV screening ($test_type)")
    iv = Starsim.InterventionData(md, nothing, :hpv)

    # Set test-specific defaults for CIN2+ sensitivity
    sens = if sensitivity !== nothing
        Float64(sensitivity)
    else
        test_type == :pap     ? 0.55 :
        test_type == :hpv_dna ? 0.95 :
        test_type == :via     ? 0.60 : 0.55
    end

    spec = if specificity !== nothing
        Float64(specificity)
    else
        test_type == :pap     ? 0.97 :
        test_type == :hpv_dna ? 0.90 :
        test_type == :via     ? 0.84 : 0.97
    end

    # Stage-specific sensitivity defaults
    sens_cin1 = if sensitivity_cin1 !== nothing
        Float64(sensitivity_cin1)
    else
        test_type == :pap     ? 0.30 :
        test_type == :hpv_dna ? 0.90 :
        test_type == :via     ? 0.40 : 0.30
    end

    sens_cancer = if sensitivity_cancer !== nothing
        Float64(sensitivity_cancer)
    else
        test_type == :pap     ? 0.95 :
        test_type == :hpv_dna ? 0.98 :
        test_type == :via     ? 0.85 : 0.95
    end

    HPVScreening(
        iv,
        test_type, sens, spec, sens_cin1, sens_cancer,
        Float64(start_year), Float64(end_year),
        Float64(screen_prob),
        Float64(min_age), Float64(max_age),
        Float64(treat_prob), Float64(treat_efficacy),
        treatment_type,
        HPVGenotype[],
        StableRNG(0),
    )
end

Starsim.intervention_data(s::HPVScreening) = s.iv

function Starsim.init_pre!(s::HPVScreening, sim)
    md = Starsim.module_data(s)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    s.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Find HPV diseases
    s.hpv_diseases = HPVGenotype[]
    for (_, dis) in sim.diseases
        if dis isa HPVGenotype
            push!(s.hpv_diseases, dis)
        end
    end

    npts = md.t.npts
    Starsim.define_results!(s,
        Starsim.Result(:n_screened; npts=npts, label="Agents screened"),
        Starsim.Result(:n_detected; npts=npts, label="CIN2+ detected"),
        Starsim.Result(:n_treated; npts=npts, label="Agents treated"),
    )

    md.initialized = true
    return s
end

function Starsim.step!(s::HPVScreening, sim)
    md = Starsim.module_data(s)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    (year < s.start_year || year > s.end_year) && return s

    active = sim.people.auids.values
    n_screened = 0
    n_detected = 0
    n_treated = 0

    # Screen females only in the eligible age range
    for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]
        (age < s.min_age || age > s.max_age) && continue
        rand(s.rng) >= s.screen_prob && continue

        n_screened += 1

        # Determine worst disease state across all HPV genotypes
        worst_stage = :none
        cin_disease = nothing
        for d in s.hpv_diseases
            if d.cancerous.raw[u]
                worst_stage = :cancer; cin_disease = d; break
            elseif d.cin3.raw[u]
                if worst_stage != :cancer
                    worst_stage = :cin3; cin_disease = d
                end
            elseif d.cin2.raw[u]
                if worst_stage ∉ (:cancer, :cin3)
                    worst_stage = :cin2; cin_disease = d
                end
            elseif d.cin1.raw[u]
                if worst_stage ∉ (:cancer, :cin3, :cin2)
                    worst_stage = :cin1; cin_disease = d
                end
            elseif d.infection.infected.raw[u]
                if worst_stage == :none
                    worst_stage = :infected; cin_disease = d
                end
            end
        end

        # Apply stage-dependent sensitivity
        detected = false
        if worst_stage == :cancer
            detected = rand(s.rng) < s.sensitivity_cancer
        elseif worst_stage in (:cin2, :cin3)
            detected = rand(s.rng) < s.sensitivity
        elseif worst_stage == :cin1
            detected = rand(s.rng) < s.sensitivity_cin1
        elseif worst_stage == :infected
            # Infection without CIN — depends on test type
            if s.test_type == :hpv_dna
                detected = rand(s.rng) < 0.85  # HPV DNA detects infection
            else
                detected = rand(s.rng) > s.specificity  # False positive
            end
        else
            # No disease — false positive
            has_infection = any(d -> d.infection.infected.raw[u], s.hpv_diseases)
            if has_infection && s.test_type == :hpv_dna
                detected = rand(s.rng) < 0.85
            end
        end

        if detected
            n_detected += 1
            if rand(s.rng) < s.treat_prob
                treated = _treat_cin!(s, sim, u)
                if treated
                    n_treated += 1
                end
            end
        end
    end

    if ti <= length(md.results[:n_screened].values)
        md.results[:n_screened][ti] = Float64(n_screened)
        md.results[:n_detected][ti] = Float64(n_detected)
        md.results[:n_treated][ti]  = Float64(n_treated)
    end

    return s
end

"""Treat CIN lesions for an agent — cure CIN across all genotypes using treatment type."""
function _treat_cin!(s::HPVScreening, sim, uid::Int)
    treated = false
    for d in s.hpv_diseases
        has_cin = d.cin1.raw[uid] || d.cin2.raw[uid] || d.cin3.raw[uid]
        has_cancer = d.cancerous.raw[uid]
        (has_cin || has_cancer) || continue

        # Determine stage-specific efficacy
        eff = if s.treatment_type == GENERIC
            s.treat_efficacy
        else
            stage = has_cancer ? :cancer : d.cin3.raw[uid] ? :cin3 :
                    d.cin2.raw[uid] ? :cin2 : :cin1
            get_treatment_efficacy(s.treatment_type, stage)
        end

        if rand(s.rng) < eff
            d.cin1.raw[uid] = false
            d.cin2.raw[uid] = false
            d.cin3.raw[uid] = false
            if has_cancer && s.treatment_type == EXCISION
                d.cancerous.raw[uid] = false  # Only excision can treat cancer
            end
            d.infection.infected.raw[uid] = false
            d.infection.susceptible.raw[uid] = true
            d.cleared.raw[uid] = true
            d.ti_cleared.raw[uid] = Float64(Starsim.module_data(d).t.ti)
            _notify_clearance(d, sim, uid)
            treated = true
        end
    end
    return treated
end

# ============================================================================
# HPVTherapeuticVaccine — treats existing infections via immune boost
# ============================================================================

"""
    HPVTherapeuticVaccine <: AbstractIntervention

Therapeutic HPV vaccination — boosts immune clearance of existing infections
and CIN lesions by reducing rel_sus and increasing clearance probability.

Unlike prophylactic vaccination, therapeutic vaccines target already-infected
agents to accelerate clearance of latent or active disease.

# Keyword arguments
- `start_year::Float64` — year to begin (default 0.0)
- `end_year::Float64` — year to end (default Inf)
- `covered_genotypes::Vector{Symbol}` — targeted genotypes
- `clearance_boost::Float64` — multiplicative boost to clearance rate (default 2.0)
- `sus_reduction::Float64` — reduction in susceptibility for cleared agents (default 0.8)
- `min_age::Float64` — minimum eligible age (default 18.0)
- `max_age::Float64` — maximum eligible age (default 45.0)
- `uptake_prob::Float64` — acceptance probability (default 0.5)
- `treat_cin::Bool` — whether to attempt CIN clearance (default true)
"""
mutable struct HPVTherapeuticVaccine <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData

    start_year::Float64
    end_year::Float64
    covered_genotypes::Vector{Symbol}
    clearance_boost::Float64
    sus_reduction::Float64
    min_age::Float64
    max_age::Float64
    uptake_prob::Float64
    treat_cin::Bool

    # Discovered diseases
    covered_diseases::Vector{HPVGenotype}

    # Per-agent tracking
    txvx_received::Starsim.StateVector{Bool, Vector{Bool}}

    rng::StableRNG
end

function HPVTherapeuticVaccine(;
    name::Symbol = :hpv_txvx,
    start_year::Real = 0.0,
    end_year::Real = Inf,
    covered_genotypes::Vector{Symbol} = [:hpv16, :hpv18],
    clearance_boost::Real = 2.0,
    sus_reduction::Real = 0.8,
    min_age::Real = 18.0,
    max_age::Real = 45.0,
    uptake_prob::Real = 0.5,
    treat_cin::Bool = true,
)
    md = Starsim.ModuleData(name; label="HPV therapeutic vaccine")
    iv = Starsim.InterventionData(md, nothing, :hpv)

    HPVTherapeuticVaccine(
        iv,
        Float64(start_year), Float64(end_year),
        covered_genotypes,
        Float64(clearance_boost),
        Float64(sus_reduction),
        Float64(min_age), Float64(max_age),
        Float64(uptake_prob), treat_cin,
        HPVGenotype[],
        Starsim.BoolState(:hpv_txvx_received; default=false, label="TxVx received"),
        StableRNG(0),
    )
end

Starsim.intervention_data(tv::HPVTherapeuticVaccine) = tv.iv

function Starsim.init_pre!(tv::HPVTherapeuticVaccine, sim)
    md = Starsim.module_data(tv)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    tv.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    Starsim.add_module_state!(sim.people, tv.txvx_received)

    # Find covered diseases
    tv.covered_diseases = HPVGenotype[]
    for (_, dis) in sim.diseases
        if dis isa HPVGenotype && dis.genotype in tv.covered_genotypes
            push!(tv.covered_diseases, dis)
        end
    end

    npts = md.t.npts
    Starsim.define_results!(tv,
        Starsim.Result(:n_txvx_given; npts=npts, label="TxVx doses given"),
        Starsim.Result(:n_txvx_cleared; npts=npts, label="TxVx-induced clearances"),
    )

    md.initialized = true
    return tv
end

function Starsim.step!(tv::HPVTherapeuticVaccine, sim)
    md = Starsim.module_data(tv)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    (year < tv.start_year || year > tv.end_year) && return tv

    active = sim.people.auids.values
    dt = sim.pars.dt
    n_given = 0
    n_cleared = 0

    for u in active
        tv.txvx_received.raw[u] && continue  # One-time treatment

        # Sex: therapeutic vaccine targets females (cervical disease)
        !sim.people.female.raw[u] && continue

        age = sim.people.age.raw[u]
        (age < tv.min_age || age > tv.max_age) && continue

        # Must be currently infected with a covered genotype
        is_infected = false
        for d in tv.covered_diseases
            if d.infection.infected.raw[u]
                is_infected = true
                break
            end
        end
        !is_infected && continue

        rand(tv.rng) >= tv.uptake_prob && continue

        # Administer therapeutic vaccine
        tv.txvx_received.raw[u] = true
        n_given += 1

        # Attempt to boost clearance for each covered infection
        for d in tv.covered_diseases
            d.infection.infected.raw[u] || continue

            # Boosted clearance probability
            gp = d.params
            base_clear = gp.clearance_rate_inf
            boosted_clear = min(1.0, base_clear * tv.clearance_boost)
            p_clear = 1.0 - (1.0 - boosted_clear)^dt

            if rand(tv.rng) < p_clear
                ti_f = Float64(ti)
                d.cin1.raw[u] = false
                d.cin2.raw[u] = false
                d.cin3.raw[u] = false
                d.infection.infected.raw[u] = false
                d.infection.susceptible.raw[u] = true
                d.cleared.raw[u] = true
                d.ti_cleared.raw[u] = ti_f
                _notify_clearance(d, sim, u)
                n_cleared += 1
            end

            # Reduce susceptibility for future reinfection
            new_sus = max(0.05, d.infection.rel_sus.raw[u] * (1.0 - tv.sus_reduction))
            d.infection.rel_sus.raw[u] = new_sus
        end
    end

    if ti <= length(md.results[:n_txvx_given].values)
        md.results[:n_txvx_given][ti] = Float64(n_given)
        md.results[:n_txvx_cleared][ti] = Float64(n_cleared)
    end

    return tv
end
