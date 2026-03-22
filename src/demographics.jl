"""
Demographics (births and deaths) for Starsim.jl.

Mirrors Python starsim's `demographics.py`. Provides `Births` and `Deaths`
modules with rate-based population dynamics.
"""

# ============================================================================
# Births
# ============================================================================

"""
    Births <: AbstractDemographics

Birth module that adds new agents to the population at a given rate.

# Keyword arguments
- `name::Symbol` — module name (default `:births`)
- `birth_rate::Real` — annual births per 1000 population (default 20.0)
- `init_age::Real` — age of newborns (default 0.0)
- `p_female::Real` — probability of female sex (default 0.5)

# Example
```julia
births = Births(birth_rate=25.0)
```
"""
mutable struct Births <: AbstractDemographics
    mod::ModuleData
    birth_rate::Float64
    init_age::Float64
    p_female::Float64
    rng::StableRNG
end

function Births(;
    name::Symbol = :births,
    birth_rate::Real = 20.0,
    init_age::Real = 0.0,
    p_female::Real = 0.5,
)
    md = ModuleData(name; label="Births")
    Births(md, Float64(birth_rate), Float64(init_age), Float64(p_female), StableRNG(0))
end

module_data(b::Births) = b.mod

function init_pre!(b::Births, sim)
    md = module_data(b)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    b.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    npts = md.t.npts
    define_results!(b,
        Result(:births; npts=npts, label="Births"),
        Result(:cbr; npts=npts, label="Crude birth rate", scale=false),
    )

    md.initialized = true
    return b
end

function step!(b::Births, sim)
    md = module_data(b)
    ti = md.t.ti
    dt = sim.pars.dt

    # Crude birth rate per 1000 per year, scaled to dt
    n_alive = Float64(length(sim.people.auids))
    annual_rate = b.birth_rate / 1000.0
    p_birth = 1.0 - exp(-annual_rate * dt)
    n_births = Int(round(n_alive * p_birth))

    if n_births > 0
        new_uids = grow!(sim.people, n_births)

        # Set sex of newborns
        for u in new_uids.values
            sim.people.female.raw[u] = rand(b.rng) < b.p_female
            sim.people.age.raw[u] = b.init_age
        end
    end

    # Update results
    if ti <= length(md.results[:births].values)
        md.results[:births][ti] = Float64(n_births)
        md.results[:cbr][ti] = n_alive > 0 ? (n_births / n_alive) * (1.0 / dt) * 1000.0 : 0.0
    end

    return b
end

export Births

# ============================================================================
# Deaths
# ============================================================================

"""
    Deaths <: AbstractDemographics

Background mortality module. Applies age-independent or age-dependent
death rates.

# Keyword arguments
- `name::Symbol` — module name (default `:deaths`)
- `death_rate::Real` — annual deaths per 1000 population (default 10.0)

# Example
```julia
deaths = Deaths(death_rate=8.0)
```
"""
mutable struct Deaths <: AbstractDemographics
    mod::ModuleData
    death_rate::Float64
    rng::StableRNG
end

function Deaths(;
    name::Symbol = :deaths,
    death_rate::Real = 10.0,
)
    md = ModuleData(name; label="Deaths")
    Deaths(md, Float64(death_rate), StableRNG(0))
end

module_data(d::Deaths) = d.mod

function init_pre!(d::Deaths, sim)
    md = module_data(d)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    npts = md.t.npts
    define_results!(d,
        Result(:deaths; npts=npts, label="Deaths"),
        Result(:cdr; npts=npts, label="Crude death rate", scale=false),
    )

    md.initialized = true
    return d
end

function step!(d::Deaths, sim)
    md = module_data(d)
    ti = md.t.ti
    dt = sim.pars.dt

    # Crude death rate per 1000 per year, scaled to dt
    active = sim.people.auids
    n_alive = Float64(length(active))
    annual_rate = d.death_rate / 1000.0
    p_death = 1.0 - exp(-annual_rate * dt)

    n_deaths = 0
    for u in active.values
        if rand(d.rng) < p_death
            request_death!(sim.people, UIDs([u]), ti)
            n_deaths += 1
        end
    end

    if ti <= length(md.results[:deaths].values)
        md.results[:deaths][ti] = Float64(n_deaths)
        md.results[:cdr][ti] = n_alive > 0 ? (n_deaths / n_alive) * (1.0 / dt) * 1000.0 : 0.0
    end

    return d
end

export Deaths

# ============================================================================
# Pregnancy
# ============================================================================

"""
    Pregnancy <: AbstractDemographics

Pregnancy module — tracks pregnant status and delivery.
(Simplified version; full version in FPsim.jl extension.)
"""
mutable struct Pregnancy <: AbstractDemographics
    mod::ModuleData
    pregnant::StateVector{Bool, Vector{Bool}}
    ti_delivery::StateVector{Float64, Vector{Float64}}
    dur_pregnancy::Float64
    fertility_rate::Float64
    rng::StableRNG
end

function Pregnancy(;
    name::Symbol = :pregnancy,
    fertility_rate::Real = 0.1,
    dur_pregnancy::Real = 0.75,  # 9 months in years
)
    md = ModuleData(name; label="Pregnancy")
    Pregnancy(
        md,
        BoolState(:pregnant; default=false),
        FloatState(:ti_delivery; default=Inf),
        Float64(dur_pregnancy),
        Float64(fertility_rate),
        StableRNG(0)
    )
end

module_data(p::Pregnancy) = p.mod

function init_pre!(preg::Pregnancy, sim)
    md = module_data(preg)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    preg.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    add_module_state!(sim.people, preg.pregnant)
    add_module_state!(sim.people, preg.ti_delivery)

    npts = md.t.npts
    define_results!(preg,
        Result(:new_pregnancies; npts=npts, label="New pregnancies"),
        Result(:deliveries; npts=npts, label="Deliveries"),
    )

    md.initialized = true
    return preg
end

function step!(preg::Pregnancy, sim)
    md = module_data(preg)
    ti = md.t.ti
    dt = sim.pars.dt

    # Check deliveries
    deliveries = 0
    for u in sim.people.auids.values
        if preg.pregnant.raw[u] && preg.ti_delivery.raw[u] <= Float64(ti)
            preg.pregnant.raw[u] = false
            preg.ti_delivery.raw[u] = Inf

            # Add newborn
            new_uids = grow!(sim.people, 1)
            for nu in new_uids.values
                sim.people.parent.raw[nu] = u
            end
            deliveries += 1
        end
    end

    # New pregnancies (simplified: eligible = alive, female, not pregnant, age 15-49)
    new_preg = 0
    p_conceive = 1.0 - exp(-preg.fertility_rate * dt)
    for u in sim.people.auids.values
        if sim.people.female.raw[u] && !preg.pregnant.raw[u]
            age = sim.people.age.raw[u]
            if 15.0 <= age <= 49.0 && rand(preg.rng) < p_conceive
                preg.pregnant.raw[u] = true
                dur_ts = preg.dur_pregnancy / dt
                preg.ti_delivery.raw[u] = Float64(ti) + dur_ts
                new_preg += 1
            end
        end
    end

    if ti <= length(md.results[:new_pregnancies].values)
        md.results[:new_pregnancies][ti] = Float64(new_preg)
        md.results[:deliveries][ti] = Float64(deliveries)
    end

    return preg
end

export Pregnancy
