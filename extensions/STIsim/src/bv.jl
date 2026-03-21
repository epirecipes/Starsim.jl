"""
Bacterial Vaginosis — risk-factor model (not a standard STI transmission).
Port of Python `stisim.diseases.bv.BV`.

BV is modeled as an NCD-like condition where risk factors (number of partners,
douching, etc.) increase acquisition probability. It is not directly transmitted
per-act but correlates with sexual behavior.
"""

"""
    BacterialVaginosis <: AbstractNCD

Bacterial vaginosis risk-factor model. Only affects female agents.
Acquisition depends on number of sexual partners and random risk factors.
Not transmitted per-act like other STIs.

# Keyword arguments
- `name::Symbol` — default `:bv`
- `init_prev::Real` — initial prevalence among females (default 0.25)
- `p_acquire::Real` — base per-timestep acquisition probability (default 0.02)
- `dur_inf::Real` — mean infection duration in years (default 0.5)
- `rel_risk_partners::Real` — relative risk per additional partner (default 1.3)
"""
mutable struct BacterialVaginosis <: Starsim.AbstractNCD
    data::Starsim.DiseaseData

    # States
    infected::Starsim.StateVector{Bool, Vector{Bool}}
    ti_infected::Starsim.StateVector{Float64, Vector{Float64}}
    ti_clearance::Starsim.StateVector{Float64, Vector{Float64}}

    # Parameters
    init_prev::Float64
    p_acquire::Float64
    dur_inf::Float64
    rel_risk_partners::Float64

    rng::StableRNG
end

function BacterialVaginosis(;
    name::Symbol             = :bv,
    init_prev::Real          = 0.25,
    p_acquire::Real          = 0.02,
    dur_inf::Real            = 0.5,
    rel_risk_partners::Real  = 1.3,
)
    dd = Starsim.DiseaseData(name; init_prev=Float64(init_prev), beta=0.0, label="Bacterial Vaginosis")

    BacterialVaginosis(
        dd,
        Starsim.BoolState(:infected; default=false, label="BV infected"),
        Starsim.FloatState(:ti_infected; default=Inf, label="Time BV infected"),
        Starsim.FloatState(:ti_clearance; default=Inf, label="Time BV clearance"),
        Float64(init_prev), Float64(p_acquire), Float64(dur_inf),
        Float64(rel_risk_partners),
        StableRNG(0),
    )
end

Starsim.disease_data(d::BacterialVaginosis) = d.data
Starsim.module_data(d::BacterialVaginosis)  = d.data.mod

function Starsim.init_pre!(d::BacterialVaginosis, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    for s in [d.infected, d.ti_infected, d.ti_clearance]
        Starsim.add_module_state!(sim.people, s)
    end

    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:n_infected; npts=npts, label="BV+", scale=false),
        Starsim.Result(:prevalence; npts=npts, label="BV prevalence", scale=false),
        Starsim.Result(:new_infections; npts=npts, label="New BV cases"),
    )

    md.initialized = true
    return d
end

function Starsim.init_post!(d::BacterialVaginosis, sim)
    people = sim.people
    active = people.auids.values
    dt = sim.pars.dt

    # Seed initial BV infections among females
    for u in active
        if people.female.raw[u] && rand(d.rng) < d.init_prev
            d.infected.raw[u] = true
            d.ti_infected.raw[u] = 1.0
            dur_ts = max(1.0, d.dur_inf / dt + randn(d.rng) * d.dur_inf * 0.3 / dt)
            d.ti_clearance.raw[u] = 1.0 + dur_ts
        end
    end
    return d
end

function Starsim.step_state!(d::BacterialVaginosis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti_f = Float64(ti)
    active = sim.people.auids.values
    dt = sim.pars.dt

    # Clearance
    @inbounds for u in active
        if d.infected.raw[u] && d.ti_clearance.raw[u] <= ti_f
            d.infected.raw[u] = false
        end
    end

    return d
end

function Starsim.step!(d::BacterialVaginosis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    active = sim.people.auids.values
    dt = sim.pars.dt
    new_infections = 0

    # Risk-factor based acquisition for susceptible females
    @inbounds for u in active
        if sim.people.female.raw[u] && !d.infected.raw[u]
            p = d.p_acquire * dt
            if rand(d.rng) < p
                d.infected.raw[u] = true
                d.ti_infected.raw[u] = Float64(ti)
                dur_ts = max(1.0, d.dur_inf / dt + randn(d.rng) * d.dur_inf * 0.3 / dt)
                d.ti_clearance.raw[u] = Float64(ti) + dur_ts
                new_infections += 1
            end
        end
    end

    return new_infections
end

function Starsim.step_die!(d::BacterialVaginosis, death_uids::Starsim.UIDs)
    d.infected[death_uids] = false
    return d
end

function Starsim.update_results!(d::BacterialVaginosis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_infected].values) && return d

    active = sim.people.auids.values
    n_inf = 0; n_female = 0
    @inbounds for u in active
        n_inf += d.infected.raw[u]
        n_female += sim.people.female.raw[u]
    end

    md.results[:n_infected][ti] = Float64(n_inf)
    md.results[:prevalence][ti] = n_female > 0 ? Float64(n_inf) / Float64(n_female) : 0.0
    return d
end

function Starsim.finalize!(d::BacterialVaginosis)
    return d
end
