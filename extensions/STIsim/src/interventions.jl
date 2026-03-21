"""
STI interventions: testing, treatment, ART, VMMC, PrEP.
Port of Python `stisim.interventions`.
"""

# ============================================================================
# STITest — screening/testing intervention
# ============================================================================

"""
    STITest <: AbstractIntervention

Periodic screening and testing for STIs.

# Keyword arguments
- `disease_name::Symbol` — target disease (default :sti)
- `test_prob::Float64` — per-timestep test probability (default 0.01)
- `sensitivity::Float64` — test sensitivity (default 0.90)
- `start_year::Float64` — year to start testing (default 0.0)
- `end_year::Float64` — year to stop (default Inf)
"""
mutable struct STITest <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    disease_name::Symbol
    test_prob::Float64
    sensitivity::Float64
    start_year::Float64
    end_year::Float64
    rng::StableRNG
end

function STITest(;
    name::Symbol = :sti_test,
    disease_name::Symbol = :sti,
    test_prob::Real = 0.01,
    sensitivity::Real = 0.90,
    start_year::Real = 0.0,
    end_year::Real = Inf,
)
    md = Starsim.ModuleData(name; label="STI testing")
    iv = Starsim.InterventionData(md, nothing, disease_name)
    STITest(iv, disease_name, Float64(test_prob), Float64(sensitivity),
            Float64(start_year), Float64(end_year), StableRNG(0))
end

Starsim.intervention_data(t::STITest) = t.iv

function Starsim.init_pre!(t::STITest, sim)
    md = Starsim.module_data(t)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    t.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))
    npts = md.t.npts
    Starsim.define_results!(t,
        Starsim.Result(:n_tested; npts=npts, label="Agents tested"),
        Starsim.Result(:n_positive; npts=npts, label="Positive results"),
    )
    md.initialized = true
    return t
end

function Starsim.step!(t::STITest, sim)
    md = Starsim.module_data(t)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    (year < t.start_year || year > t.end_year) && return t
    haskey(sim.diseases, t.disease_name) || return t

    disease = sim.diseases[t.disease_name]
    active = sim.people.auids.values
    n_tested = 0
    n_positive = 0
    dt = sim.pars.dt

    if disease isa HIV
        # HIV testing: test undiagnosed infected agents
        @inbounds for u in active
            if disease.infection.infected.raw[u] && !disease.diagnosed.raw[u]
                if rand(t.rng) < t.test_prob * dt
                    n_tested += 1
                    if rand(t.rng) < t.sensitivity
                        disease.diagnosed.raw[u] = true
                        disease.ti_diagnosed.raw[u] = Float64(ti)
                        n_positive += 1
                    end
                end
            elseif !disease.infection.infected.raw[u]
                # Also test uninfected (they get negative results)
                if rand(t.rng) < t.test_prob * dt
                    n_tested += 1
                end
            end
        end
    elseif disease isa SEIS
        # SEIS testing: test infected agents (symptomatic or asymptomatic)
        @inbounds for u in active
            if disease.infection.infected.raw[u] || disease.exposed.raw[u]
                if rand(t.rng) < t.test_prob * dt
                    n_tested += 1
                    if rand(t.rng) < t.sensitivity
                        n_positive += 1
                    end
                end
            elseif disease.infection.susceptible.raw[u]
                if rand(t.rng) < t.test_prob * dt
                    n_tested += 1
                end
            end
        end
    end

    res = Starsim.module_results(t)
    if ti <= length(res[:n_tested].values)
        res[:n_tested][ti] = Float64(n_tested)
        res[:n_positive][ti] = Float64(n_positive)
    end
    return t
end

# ============================================================================
# STITreatment — treatment intervention
# ============================================================================

"""
    STITreatment <: AbstractIntervention

Treatment for diagnosed STI infections.
"""
mutable struct STITreatment <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    disease_name::Symbol
    treat_prob::Float64
    efficacy::Float64
    rng::StableRNG
end

function STITreatment(;
    name::Symbol = :sti_treat,
    disease_name::Symbol = :sti,
    treat_prob::Real = 0.90,
    efficacy::Real = 0.95,
)
    md = Starsim.ModuleData(name; label="STI treatment")
    iv = Starsim.InterventionData(md, nothing, disease_name)
    STITreatment(iv, disease_name, Float64(treat_prob), Float64(efficacy), StableRNG(0))
end

Starsim.intervention_data(t::STITreatment) = t.iv

function Starsim.init_pre!(t::STITreatment, sim)
    md = Starsim.module_data(t)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    t.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))
    npts = md.t.npts
    Starsim.define_results!(t,
        Starsim.Result(:n_treated; npts=npts, label="Agents treated"),
    )
    md.initialized = true
    return t
end

function Starsim.step!(t::STITreatment, sim)
    md = Starsim.module_data(t)
    ti = md.t.ti
    haskey(sim.diseases, t.disease_name) || return t

    disease = sim.diseases[t.disease_name]
    active = sim.people.auids.values
    n_treated = 0

    if disease isa SEIS
        # Treat symptomatic SEIS agents: cure the infection
        @inbounds for u in active
            if disease.infection.infected.raw[u] && disease.symptomatic.raw[u]
                if rand(t.rng) < t.treat_prob * sim.pars.dt
                    if rand(t.rng) < t.efficacy
                        disease.infection.infected.raw[u] = false
                        disease.infection.susceptible.raw[u] = true
                        disease.symptomatic.raw[u] = false
                        disease.exposed.raw[u] = false
                        disease.pid.raw[u] = false
                        n_treated += 1
                    end
                end
            end
        end
    elseif disease isa Syphilis
        # Treat any stage of syphilis: cure
        @inbounds for u in active
            if disease.infection.infected.raw[u] && !disease.treated.raw[u]
                if rand(t.rng) < t.treat_prob * sim.pars.dt
                    if rand(t.rng) < t.efficacy
                        disease.infection.infected.raw[u] = false
                        disease.infection.susceptible.raw[u] = true
                        disease.primary.raw[u] = false
                        disease.secondary.raw[u] = false
                        disease.early_latent.raw[u] = false
                        disease.late_latent.raw[u] = false
                        disease.tertiary.raw[u] = false
                        disease.treated.raw[u] = true
                        n_treated += 1
                    end
                end
            end
        end
    end

    res = Starsim.module_results(t)
    if haskey(res, :n_treated) && ti <= length(res[:n_treated].values)
        res[:n_treated][ti] = Float64(n_treated)
    end
    return t
end

# ============================================================================
# ART — Antiretroviral Therapy for HIV
# ============================================================================

"""
    ART <: AbstractIntervention

ART initiation for diagnosed HIV+ agents.

# Keyword arguments
- `start_year::Float64` — when ART becomes available
- `coverage::Float64` — target coverage (default 0.8)
- `initiation_rate::Float64` — per-timestep probability of starting ART (default 0.1)
"""
mutable struct ART <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    hiv_name::Symbol
    start_year::Float64
    coverage::Float64
    initiation_rate::Float64
    rng::StableRNG
end

function ART(;
    name::Symbol = :art,
    hiv_name::Symbol = :hiv,
    start_year::Real = 0.0,
    coverage::Real = 0.8,
    initiation_rate::Real = 0.1,
)
    md = Starsim.ModuleData(name; label="ART")
    iv = Starsim.InterventionData(md, nothing, hiv_name)
    ART(iv, hiv_name, Float64(start_year), Float64(coverage),
        Float64(initiation_rate), StableRNG(0))
end

Starsim.intervention_data(a::ART) = a.iv

function Starsim.init_pre!(a::ART, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    a.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))
    npts = md.t.npts
    Starsim.define_results!(a,
        Starsim.Result(:n_initiated; npts=npts, label="ART initiated"),
    )
    md.initialized = true
    return a
end

function Starsim.step!(a::ART, sim)
    md = Starsim.module_data(a)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    year < a.start_year && return a
    haskey(sim.diseases, a.hiv_name) || return a

    hiv = sim.diseases[a.hiv_name]::HIV
    active = sim.people.auids.values
    n_initiated = 0

    @inbounds for u in active
        if hiv.infection.infected.raw[u] && hiv.diagnosed.raw[u] && !hiv.on_art.raw[u]
            if rand(a.rng) < a.initiation_rate * sim.pars.dt
                hiv.on_art.raw[u] = true
                hiv.ti_art.raw[u] = Float64(ti)
                hiv.infection.rel_trans.raw[u] *= (1.0 - hiv.art_efficacy)
                n_initiated += 1
            end
        end
    end

    res = Starsim.module_results(a)
    if haskey(res, :n_initiated) && ti <= length(res[:n_initiated].values)
        res[:n_initiated][ti] = Float64(n_initiated)
    end
    return a
end

# ============================================================================
# VMMC — Voluntary Medical Male Circumcision
# ============================================================================

"""
    VMMC <: AbstractIntervention

Voluntary medical male circumcision — reduces HIV susceptibility in males.
"""
mutable struct VMMC <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    hiv_name::Symbol
    efficacy::Float64
    uptake_rate::Float64
    start_year::Float64
    circumcised::Starsim.StateVector{Bool, Vector{Bool}}
    rng::StableRNG
end

function VMMC(;
    name::Symbol = :vmmc,
    hiv_name::Symbol = :hiv,
    efficacy::Real = 0.6,
    uptake_rate::Real = 0.05,
    start_year::Real = 0.0,
)
    md = Starsim.ModuleData(name; label="VMMC")
    iv = Starsim.InterventionData(md, nothing, hiv_name)
    VMMC(iv, hiv_name, Float64(efficacy), Float64(uptake_rate), Float64(start_year),
         Starsim.BoolState(:circumcised; default=false, label="Circumcised"),
         StableRNG(0))
end

Starsim.intervention_data(v::VMMC) = v.iv

function Starsim.init_pre!(v::VMMC, sim)
    md = Starsim.module_data(v)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    v.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    Starsim.add_module_state!(sim.people, v.circumcised)

    npts = md.t.npts
    Starsim.define_results!(v,
        Starsim.Result(:n_circumcised; npts=npts, label="New circumcisions"),
        Starsim.Result(:n_total_circumcised; npts=npts, label="Total circumcised", scale=false),
    )
    md.initialized = true
    return v
end

function Starsim.step!(v::VMMC, sim)
    md = Starsim.module_data(v)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    year < v.start_year && return v

    active = sim.people.auids.values
    female_raw = sim.people.female.raw
    circ_raw = v.circumcised.raw
    dt = sim.pars.dt
    n_new = 0
    n_total = 0

    # Find HIV disease for susceptibility modification
    hiv = haskey(sim.diseases, v.hiv_name) ? sim.diseases[v.hiv_name] : nothing

    @inbounds for u in active
        if circ_raw[u]
            n_total += 1
            continue
        end
        if !female_raw[u]  # males only
            if rand(v.rng) < v.uptake_rate * dt
                circ_raw[u] = true
                n_new += 1
                n_total += 1
                # Reduce HIV susceptibility
                if hiv !== nothing && hiv isa HIV
                    hiv.infection.rel_sus.raw[u] *= (1.0 - v.efficacy)
                end
            end
        end
    end

    res = Starsim.module_results(v)
    if ti <= length(res[:n_circumcised].values)
        res[:n_circumcised][ti] = Float64(n_new)
        res[:n_total_circumcised][ti] = Float64(n_total)
    end
    return v
end

# ============================================================================
# PrEP — Pre-exposure prophylaxis
# ============================================================================

"""
    PrEP <: AbstractIntervention

Pre-exposure prophylaxis for HIV.
Reduces HIV susceptibility for HIV-negative agents.
"""
mutable struct PrEP <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    hiv_name::Symbol
    efficacy::Float64
    uptake_rate::Float64
    start_year::Float64
    on_prep::Starsim.StateVector{Bool, Vector{Bool}}
    rng::StableRNG
end

function PrEP(;
    name::Symbol = :prep,
    hiv_name::Symbol = :hiv,
    efficacy::Real = 0.86,
    uptake_rate::Real = 0.05,
    start_year::Real = 0.0,
)
    md = Starsim.ModuleData(name; label="PrEP")
    iv = Starsim.InterventionData(md, nothing, hiv_name)
    PrEP(iv, hiv_name, Float64(efficacy), Float64(uptake_rate), Float64(start_year),
         Starsim.BoolState(:on_prep; default=false, label="On PrEP"),
         StableRNG(0))
end

Starsim.intervention_data(p::PrEP) = p.iv

function Starsim.init_pre!(p::PrEP, sim)
    md = Starsim.module_data(p)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    p.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    Starsim.add_module_state!(sim.people, p.on_prep)

    npts = md.t.npts
    Starsim.define_results!(p,
        Starsim.Result(:n_initiated; npts=npts, label="PrEP initiated"),
        Starsim.Result(:n_on_prep; npts=npts, label="On PrEP", scale=false),
    )
    md.initialized = true
    return p
end

function Starsim.step!(p::PrEP, sim)
    md = Starsim.module_data(p)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt
    year < p.start_year && return p
    haskey(sim.diseases, p.hiv_name) || return p

    hiv = sim.diseases[p.hiv_name]
    hiv isa HIV || return p

    active = sim.people.auids.values
    prep_raw = p.on_prep.raw
    dt = sim.pars.dt
    n_initiated = 0
    n_on_prep = 0

    @inbounds for u in active
        if prep_raw[u]
            # Already on PrEP — check if they got infected (discontinue)
            if hiv.infection.infected.raw[u]
                prep_raw[u] = false
                hiv.infection.rel_sus.raw[u] /= (1.0 - p.efficacy)
            else
                n_on_prep += 1
            end
            continue
        end

        # Eligible: HIV-negative, susceptible
        if hiv.infection.susceptible.raw[u] && !hiv.infection.infected.raw[u]
            if rand(p.rng) < p.uptake_rate * dt
                prep_raw[u] = true
                hiv.infection.rel_sus.raw[u] *= (1.0 - p.efficacy)
                n_initiated += 1
                n_on_prep += 1
            end
        end
    end

    res = Starsim.module_results(p)
    if ti <= length(res[:n_initiated].values)
        res[:n_initiated][ti] = Float64(n_initiated)
        res[:n_on_prep][ti] = Float64(n_on_prep)
    end
    return p
end
