"""
Rotavirus disease — individual strain-specific disease instances.
Port of Python `rotasim.rotavirus.Rotavirus(ss.Infection)`.

Each Rotavirus instance represents a specific (G, P) combination that
behaves as an independent SIRS disease in the simulation. Cross-strain
interactions are handled by connector classes.
"""

# ============================================================================
# Rotavirus <: AbstractInfection
# ============================================================================

"""
    Rotavirus <: AbstractInfection

Individual rotavirus strain as a separate disease instance (SIRS model).

# Keyword arguments
- `G::Int` — G genotype
- `P::Int` — P genotype
- `init_prev::Real` — initial prevalence (default 0.01)
- `beta::Real` — transmission rate per day (default 0.1); adjusted by fitness
- `dur_inf_mean::Real` — mean infection duration in days (default 7.0)
- `waning_rate_mean::Real` — mean waning rate denominator in days (default 180.0)
- `waning_rate_std::Real` — std of waning rate denominator in days (default 10.0)
- `name::Union{Symbol,Nothing}` — auto-generated as `Symbol("G\$(G)P\$(P)")` if nothing
"""
mutable struct Rotavirus <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # Genotype identity
    G::Int
    P::Int

    # SIRS states
    recovered::Starsim.StateVector{Bool, Vector{Bool}}
    ti_recovered::Starsim.StateVector{Float64, Vector{Float64}}
    waning_rate::Starsim.StateVector{Float64, Vector{Float64}}
    n_infections::Starsim.StateVector{Float64, Vector{Float64}}

    # Parameters (in days)
    dur_inf_mean::Float64
    waning_rate_mean::Float64
    waning_rate_std::Float64

    # Cached connector reference (set during init_pre!; typed Any to break circular dep)
    immunity_connector::Any

    rng::StableRNG
end

function Rotavirus(;
    G::Int,
    P::Int,
    init_prev::Real     = 0.01,
    beta::Real          = 0.1,
    dur_inf_mean::Real  = 7.0,
    waning_rate_mean::Real = 180.0,
    waning_rate_std::Real  = 10.0,
    name::Union{Symbol,Nothing} = nothing,
)
    nm = name === nothing ? Symbol("G$(G)P$(P)") : name
    # beta is per-day; convert to per-year for Starsim convention (dt is in years)
    beta_peryear = Float64(beta) * 365.25
    inf = Starsim.InfectionData(nm; init_prev=Float64(init_prev), beta=beta_peryear, label="Rotavirus G$(G)P$(P)")

    Rotavirus(
        inf, G, P,
        Starsim.BoolState(:recovered; default=false, label="Recovered"),
        Starsim.FloatState(:ti_recovered; default=Inf, label="Time recovered"),
        Starsim.FloatState(:waning_rate; default=0.0, label="Waning rate"),
        Starsim.FloatState(:n_infections; default=0.0, label="Infection count"),
        Float64(dur_inf_mean),
        Float64(waning_rate_mean),
        Float64(waning_rate_std),
        nothing,  # immunity_connector — set during init_pre!
        StableRNG(0),
    )
end

Starsim.disease_data(d::Rotavirus) = d.infection.dd
Starsim.module_data(d::Rotavirus)  = d.infection.dd.mod

"""Return (G, P) strain tuple."""
strain(d::Rotavirus) = (d.G, d.P)

# ============================================================================
# Lifecycle
# ============================================================================

function Starsim.init_pre!(d::Rotavirus, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Register states
    infection_states = [
        d.infection.susceptible, d.infection.infected,
        d.infection.ti_infected, d.infection.rel_sus,
        d.infection.rel_trans,
        d.recovered, d.ti_recovered, d.waning_rate, d.n_infections,
    ]
    for s in infection_states
        Starsim.add_module_state!(sim.people, s)
    end

    # Compute beta per dt
    Starsim.validate_beta!(d, sim)

    # Cache RotaImmunityConnector reference
    d.immunity_connector = nothing
    for (_, conn) in sim.connectors
        if conn isa RotaImmunityConnector
            d.immunity_connector = conn
            break
        end
    end

    # Results
    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:new_infections; npts=npts, label="New infections"),
        Starsim.Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Starsim.Result(:n_infected; npts=npts, label="Infected", scale=false),
        Starsim.Result(:n_recovered; npts=npts, label="Recovered", scale=false),
        Starsim.Result(:prevalence; npts=npts, label="Prevalence", scale=false),
        Starsim.Result(:new_recovered; npts=npts, label="New recoveries"),
    )

    md.initialized = true
    return d
end

function Starsim.validate_beta!(d::Rotavirus, sim)
    dd = Starsim.disease_data(d)
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

function Starsim.init_post!(d::Rotavirus, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)
    # If init_prev is 0, seed 0 infections
    if d.infection.dd.init_prev <= 0.0
        n_infect = 0
    end
    n_infect == 0 && return d

    infect_uids = Starsim.UIDs(active[randperm(d.rng, n)[1:n_infect]])
    d.infection.susceptible[infect_uids] = false
    d.infection.infected[infect_uids]    = true
    d.infection.ti_infected[infect_uids] = 1.0

    # Set recovery time (in timestep units)
    dt_days = sim.pars.dt * 365.25
    for u in infect_uids.values
        dur_days = max(1.0, d.dur_inf_mean + randn(d.rng) * d.dur_inf_mean * 0.3)
        dur_ts = dur_days / dt_days
        d.ti_recovered.raw[u] = 1.0 + dur_ts
    end

    # Increment n_infections
    for u in infect_uids.values
        d.n_infections.raw[u] += 1.0
    end

    return d
end

# ============================================================================
# Step — state transitions (SIRS)
# ============================================================================

function Starsim.step_state!(d::Rotavirus, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti_f = Float64(ti)
    active = sim.people.auids.values

    infected_raw    = d.infection.infected.raw
    ti_rec_raw      = d.ti_recovered.raw
    recovered_raw   = d.recovered.raw
    susceptible_raw = d.infection.susceptible.raw

    dt_days = sim.pars.dt * 365.25
    new_recovered = 0

    @inbounds for u in active
        if infected_raw[u] && ti_rec_raw[u] <= ti_f
            infected_raw[u]    = false
            recovered_raw[u]   = true
            susceptible_raw[u] = true  # SIRS: immediately susceptible again (rel_sus modulated by immunity connector)

            # Sample waning rate = 1 / duration (in days)
            waning_denom = d.waning_rate_mean + randn(d.rng) * d.waning_rate_std
            waning_denom = max(waning_denom, 1.0)
            d.waning_rate.raw[u] = 1.0 / waning_denom

            new_recovered += 1

            # Notify immunity connector (if present)
            _notify_recovery(d, sim, u)
        end
    end

    # Record new recoveries
    res = Starsim.module_results(d)
    if haskey(res, :new_recovered) && ti <= length(res[:new_recovered].values)
        res[:new_recovered][ti] = Float64(new_recovered)
    end

    return d
end

"""Notify the RotaImmunityConnector about a recovery event."""
function _notify_recovery(d::Rotavirus, sim, uid::Int)
    conn = d.immunity_connector
    conn === nothing && return
    record_recovery!(conn, d, uid)
    return
end

# ============================================================================
# Transmission (reuses Starsim SIR infect! pattern)
# ============================================================================

function Starsim.step!(d::Rotavirus, sim)
    return _infect_rota!(d, sim)
end

function _infect_rota!(d::Rotavirus, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dd = Starsim.disease_data(d)
    new_infections = 0

    for (net_name, net) in sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        new_infections += _infect_rota_standard!(d, sim, edges, beta_dt, ti, net)
    end

    # Record new infections directly
    res = Starsim.module_results(d)
    if haskey(res, :new_infections) && ti <= length(res[:new_infections].values)
        res[:new_infections][ti] += Float64(new_infections)
    end

    return new_infections
end

function _infect_rota_standard!(d::Rotavirus, sim, edges::Starsim.Edges, beta_dt::Float64, ti::Int, net)
    new_infections = 0
    n_edges = length(edges)
    bidir = Starsim.network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    infected_raw    = d.infection.infected.raw
    susceptible_raw = d.infection.susceptible.raw
    rel_trans_raw   = d.infection.rel_trans.raw
    rel_sus_raw     = d.infection.rel_sus.raw
    rng = d.rng

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]

        if infected_raw[src] && susceptible_raw[trg]
            p = rel_trans_raw[src] * rel_sus_raw[trg] * beta_dt * edge_beta
            if rand(rng) < p
                _do_rota_infection!(d, sim, trg, src, ti)
                new_infections += 1
            end
        end

        if bidir && infected_raw[trg] && susceptible_raw[src]
            p = rel_trans_raw[trg] * rel_sus_raw[src] * beta_dt * edge_beta
            if rand(rng) < p
                _do_rota_infection!(d, sim, src, trg, ti)
                new_infections += 1
            end
        end
    end
    return new_infections
end

function _do_rota_infection!(d::Rotavirus, sim, target::Int, source::Int, ti::Int)
    d.infection.susceptible.raw[target] = false
    d.infection.infected.raw[target]    = true
    d.infection.ti_infected.raw[target] = Float64(ti)

    # Sample infection duration and set recovery time
    dt_days = sim.pars.dt * 365.25
    dur_days = max(1.0, d.dur_inf_mean + randn(d.rng) * d.dur_inf_mean * 0.3)
    dur_ts = dur_days / dt_days
    d.ti_recovered.raw[target] = Float64(ti) + dur_ts

    # Increment infection count
    d.n_infections.raw[target] += 1.0

    # Notify immunity connector about new infection
    conn = d.immunity_connector
    if conn !== nothing
        record_infection!(conn, d, target)
    end

    return
end

# ============================================================================
# set_prognoses! — called by reassortment connector for direct infections
# ============================================================================

"""
    set_prognoses!(d::Rotavirus, sim, uids::Starsim.UIDs)

Set prognoses for agents becoming infected. Used by reassortment connector.
"""
function Starsim.set_prognoses!(d::Rotavirus, sim, uids::Starsim.UIDs)
    ti = Starsim.module_data(d).t.ti
    for u in uids.values
        _do_rota_infection!(d, sim, u, 0, ti)
    end
    return
end

# ============================================================================
# Death handling and results
# ============================================================================

function Starsim.step_die!(d::Rotavirus, death_uids::Starsim.UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids]    = false
    d.recovered[death_uids]             = false
    return d
end

function Starsim.update_results!(d::Rotavirus, sim)
    md = Starsim.module_data(d)
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
    md.results[:n_infected][ti]    = Float64(n_inf)
    md.results[:n_recovered][ti]   = Float64(n_rec)

    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
    return d
end

function Starsim.finalize!(d::Rotavirus)
    return d
end
