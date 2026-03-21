"""
Syphilis disease — multi-stage: Primary → Secondary → Early Latent → Late Latent → Tertiary.
Port of Python `stisim.diseases.syphilis.Syphilis`.
"""

# ============================================================================
# Syphilis <: AbstractInfection
# ============================================================================

"""
    Syphilis <: AbstractInfection

Multi-stage syphilis model.

Stages: Primary → Secondary → Early Latent → Late Latent → Tertiary
- Can reactivate from latent stages (early latent → secondary)
- Congenital transmission possible via maternal network
- Treatment cures at any stage
"""
mutable struct Syphilis <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # Stage states
    primary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_primary::Starsim.StateVector{Float64, Vector{Float64}}
    secondary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_secondary::Starsim.StateVector{Float64, Vector{Float64}}
    early_latent::Starsim.StateVector{Bool, Vector{Bool}}
    ti_early_latent::Starsim.StateVector{Float64, Vector{Float64}}
    late_latent::Starsim.StateVector{Bool, Vector{Bool}}
    ti_late_latent::Starsim.StateVector{Float64, Vector{Float64}}
    tertiary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_tertiary::Starsim.StateVector{Float64, Vector{Float64}}
    treated::Starsim.StateVector{Bool, Vector{Bool}}

    # Parameters
    beta_m2f::Float64
    rel_beta_f2m::Float64
    beta_m2m::Float64
    eff_condom::Float64
    dur_primary::Float64       # years
    dur_secondary::Float64
    dur_early_latent::Float64
    dur_late_latent::Float64
    p_reactivate::Float64
    p_tertiary::Float64
    rel_trans_primary::Float64
    rel_trans_secondary::Float64
    rel_trans_latent::Float64          # Baseline latent transmissibility (Python default 1.0)
    rel_trans_latent_half_life::Float64 # Half-life of latent transmissibility decay (years)
    rel_trans_tertiary::Float64

    rng::StableRNG
end

function Syphilis(;
    name::Symbol              = :syphilis,
    init_prev::Real           = 0.0,       # Python default: ss.bernoulli(p=0)
    beta_m2f::Real            = 0.1,       # Python default: 0.1
    rel_beta_f2m::Real        = 0.5,
    beta_m2m::Real            = 0.1,
    eff_condom::Real          = 0.0,       # Python default: 0.0
    dur_primary::Real         = 45/365,    # ~45 days
    dur_secondary::Real       = 120/365,   # ~4 months
    dur_early_latent::Real    = 1.0,       # ~1 year (matches Python dur_early ≈ 12-14 months)
    dur_late_latent::Real     = 19.0,      # Python: time_to_tertiary ~ Normal(20yr, 2yr) from latent onset
    p_reactivate::Real        = 0.35,      # Python default: 0.35
    p_tertiary::Real          = 0.35,      # Python default: 0.35
    rel_trans_primary::Real   = 1.0,       # Python default: 1.0
    rel_trans_secondary::Real = 1.0,       # Python default: 1.0
    rel_trans_latent::Real    = 1.0,       # Python default: 1.0 (decays exponentially)
    rel_trans_latent_half_life::Real = 1.0, # Python default: 1 year
    rel_trans_tertiary::Real  = 0.0,       # Python default: 0.0
)
    inf = Starsim.InfectionData(name; init_prev=Float64(init_prev), beta=Float64(beta_m2f), label="Syphilis")

    Syphilis(
        inf,
        Starsim.BoolState(:primary; default=false, label="Primary syphilis"),
        Starsim.FloatState(:ti_primary; default=Inf, label="Time primary"),
        Starsim.BoolState(:secondary; default=false, label="Secondary syphilis"),
        Starsim.FloatState(:ti_secondary; default=Inf, label="Time secondary"),
        Starsim.BoolState(:early_latent; default=false, label="Early latent"),
        Starsim.FloatState(:ti_early_latent; default=Inf, label="Time early latent"),
        Starsim.BoolState(:late_latent; default=false, label="Late latent"),
        Starsim.FloatState(:ti_late_latent; default=Inf, label="Time late latent"),
        Starsim.BoolState(:tertiary; default=false, label="Tertiary syphilis"),
        Starsim.FloatState(:ti_tertiary; default=Inf, label="Time tertiary"),
        Starsim.BoolState(:treated; default=false, label="Treated"),
        Float64(beta_m2f), Float64(rel_beta_f2m), Float64(beta_m2m),
        Float64(eff_condom),
        Float64(dur_primary), Float64(dur_secondary),
        Float64(dur_early_latent), Float64(dur_late_latent),
        Float64(p_reactivate), Float64(p_tertiary),
        Float64(rel_trans_primary), Float64(rel_trans_secondary),
        Float64(rel_trans_latent), Float64(rel_trans_latent_half_life),
        Float64(rel_trans_tertiary),
        StableRNG(0),
    )
end

Starsim.disease_data(d::Syphilis) = d.infection.dd
Starsim.module_data(d::Syphilis)  = d.infection.dd.mod

function Starsim.init_pre!(d::Syphilis, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    states = [
        d.infection.susceptible, d.infection.infected,
        d.infection.ti_infected, d.infection.rel_sus, d.infection.rel_trans,
        d.primary, d.ti_primary, d.secondary, d.ti_secondary,
        d.early_latent, d.ti_early_latent, d.late_latent, d.ti_late_latent,
        d.tertiary, d.ti_tertiary, d.treated,
    ]
    for s in states
        Starsim.add_module_state!(sim.people, s)
    end

    _validate_syph_beta!(d, sim)

    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:new_infections; npts=npts, label="New syphilis infections"),
        Starsim.Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Starsim.Result(:n_infected; npts=npts, label="Syphilis+", scale=false),
        Starsim.Result(:n_primary; npts=npts, label="Primary", scale=false),
        Starsim.Result(:n_secondary; npts=npts, label="Secondary", scale=false),
        Starsim.Result(:n_early_latent; npts=npts, label="Early latent", scale=false),
        Starsim.Result(:n_late_latent; npts=npts, label="Late latent", scale=false),
        Starsim.Result(:n_tertiary; npts=npts, label="Tertiary", scale=false),
        Starsim.Result(:prevalence; npts=npts, label="Syphilis prevalence", scale=false),
    )

    md.initialized = true
    return d
end

function _validate_syph_beta!(d::Syphilis, sim)
    dd = Starsim.disease_data(d)
    dt = sim.pars.dt
    for (name, _) in sim.networks
        if name == :mf || name == :structuredsexual
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2f) * dt)
        elseif name == :msm
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2m) * dt)
        else
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2f) * dt)
        end
    end
    return d
end

function Starsim.init_post!(d::Syphilis, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)
    d.infection.dd.init_prev <= 0.0 && return d

    infect_uids = Starsim.UIDs(active[randperm(d.rng, n)[1:n_infect]])
    for u in infect_uids.values
        _do_syph_infection!(d, sim, u, 0, 1)
    end
    return d
end

# ============================================================================
# State transitions
# ============================================================================

function Starsim.step_state!(d::Syphilis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti_f = Float64(ti)
    active = sim.people.auids.values
    dt = sim.pars.dt

    @inbounds for u in active
        if !d.infection.infected.raw[u]
            continue
        end

        # Primary → Secondary
        if d.primary.raw[u] && d.ti_secondary.raw[u] <= ti_f
            d.primary.raw[u] = false
            d.secondary.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_secondary
        end

        # Secondary → Early Latent
        if d.secondary.raw[u] && d.ti_early_latent.raw[u] <= ti_f
            d.secondary.raw[u] = false
            d.early_latent.raw[u] = true
            # Start latent transmissibility at baseline (will decay each timestep)
            d.infection.rel_trans.raw[u] = d.rel_trans_latent
        end

        # Early Latent → Late Latent or reactivation
        if d.early_latent.raw[u] && d.ti_late_latent.raw[u] <= ti_f
            if rand(d.rng) < d.p_reactivate
                d.early_latent.raw[u] = false
                d.secondary.raw[u] = true
                d.infection.rel_trans.raw[u] = d.rel_trans_secondary
                dur_sec_ts = max(1.0, d.dur_secondary / dt + randn(d.rng) * d.dur_secondary * 0.3 / dt)
                d.ti_early_latent.raw[u] = ti_f + dur_sec_ts
                dur_el_ts = max(1.0, d.dur_early_latent / dt + randn(d.rng) * d.dur_early_latent * 0.3 / dt)
                d.ti_late_latent.raw[u] = d.ti_early_latent.raw[u] + dur_el_ts
            else
                d.early_latent.raw[u] = false
                d.late_latent.raw[u] = true
            end
        end

        # Exponential decay of latent transmissibility (matching Python)
        if d.early_latent.raw[u] || d.late_latent.raw[u]
            dur_latent_years = (ti_f - d.ti_early_latent.raw[u]) * dt
            if d.rel_trans_latent_half_life > 0.0
                decay_rate = log(2.0) / d.rel_trans_latent_half_life
                d.infection.rel_trans.raw[u] = d.rel_trans_latent * exp(-decay_rate * dur_latent_years)
            else
                d.infection.rel_trans.raw[u] = d.rel_trans_latent
            end
        end

        # Late Latent → Tertiary (one-time check, matching Python)
        if d.late_latent.raw[u] && d.ti_tertiary.raw[u] <= ti_f
            if rand(d.rng) < d.p_tertiary
                d.late_latent.raw[u] = false
                d.tertiary.raw[u] = true
                d.infection.rel_trans.raw[u] = d.rel_trans_tertiary
            else
                d.ti_tertiary.raw[u] = Inf  # Failed check; stay in late latent permanently
            end
        end
    end

    return d
end

# ============================================================================
# Transmission
# ============================================================================

function Starsim.step!(d::Syphilis, sim)
    return _infect_syph!(d, sim)
end

function _infect_syph!(d::Syphilis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dd = Starsim.disease_data(d)
    new_infections = 0

    for (net_name, net) in sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue
        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue
        new_infections += _infect_syph_edges!(d, sim, edges, beta_dt, ti, net)
    end
    return new_infections
end

function _infect_syph_edges!(d::Syphilis, sim, edges::Starsim.Edges, beta_dt::Float64, ti::Int, net)
    new_infections = 0
    bidir = Starsim.network_data(net).bidirectional
    p1 = edges.p1; p2 = edges.p2; eb = edges.beta; ea = edges.acts
    inf_raw = d.infection.infected.raw
    sus_raw = d.infection.susceptible.raw
    rt_raw  = d.infection.rel_trans.raw
    rs_raw  = d.infection.rel_sus.raw
    female_raw = sim.people.female.raw
    rng = d.rng
    dt = sim.pars.dt

    @inbounds for i in 1:length(edges)
        src = p1[i]; trg = p2[i]; edge_beta = eb[i]; acts = ea[i]
        if inf_raw[src] && sus_raw[trg]
            ba = _get_directional_beta(female_raw[src], female_raw[trg], d.beta_m2f, d.beta_m2f*d.rel_beta_f2m, d.beta_m2m)
            beta_per_act = 1.0 - exp(-ba * dt)
            eff_beta = clamp(beta_per_act * rt_raw[src] * rs_raw[trg], 0.0, 1.0)
            p = (1.0 - (1.0 - eff_beta)^acts) * edge_beta
            if rand(rng) < p
                _do_syph_infection!(d, sim, trg, src, ti)
                new_infections += 1
            end
        end
        if bidir && inf_raw[trg] && sus_raw[src]
            ba = _get_directional_beta(female_raw[trg], female_raw[src], d.beta_m2f, d.beta_m2f*d.rel_beta_f2m, d.beta_m2m)
            beta_per_act = 1.0 - exp(-ba * dt)
            eff_beta = clamp(beta_per_act * rt_raw[trg] * rs_raw[src], 0.0, 1.0)
            p = (1.0 - (1.0 - eff_beta)^acts) * edge_beta
            if rand(rng) < p
                _do_syph_infection!(d, sim, src, trg, ti)
                new_infections += 1
            end
        end
    end
    return new_infections
end

function _do_syph_infection!(d::Syphilis, sim, target::Int, source::Int, ti::Int)
    dt = sim.pars.dt
    d.infection.susceptible.raw[target] = false
    d.infection.infected.raw[target] = true
    d.infection.ti_infected.raw[target] = Float64(ti)

    # Start at primary
    d.primary.raw[target] = true
    d.ti_primary.raw[target] = Float64(ti)
    d.infection.rel_trans.raw[target] = d.rel_trans_primary

    # Set transition times
    dur_pri_ts = max(1.0, d.dur_primary / dt + randn(d.rng) * d.dur_primary * 0.3 / dt)
    d.ti_secondary.raw[target] = Float64(ti) + dur_pri_ts

    dur_sec_ts = max(1.0, d.dur_secondary / dt + randn(d.rng) * d.dur_secondary * 0.3 / dt)
    d.ti_early_latent.raw[target] = d.ti_secondary.raw[target] + dur_sec_ts

    dur_el_ts = max(1.0, d.dur_early_latent / dt + randn(d.rng) * d.dur_early_latent * 0.3 / dt)
    d.ti_late_latent.raw[target] = d.ti_early_latent.raw[target] + dur_el_ts

    # Python: time_to_tertiary ~ Normal(20yr, 2yr) → relative SD = 0.1
    dur_ll_ts = max(1.0, d.dur_late_latent / dt + randn(d.rng) * d.dur_late_latent * 0.1 / dt)
    d.ti_tertiary.raw[target] = d.ti_late_latent.raw[target] + dur_ll_ts

    push!(d.infection.infection_sources, (target, source, ti))
    return
end

function Starsim.set_prognoses!(d::Syphilis, sim, uids::Starsim.UIDs)
    ti = Starsim.module_data(d).t.ti
    for u in uids.values
        _do_syph_infection!(d, sim, u, 0, ti)
    end
    return
end

function Starsim.step_die!(d::Syphilis, death_uids::Starsim.UIDs)
    for f in [d.infection.susceptible, d.infection.infected, d.primary, d.secondary,
              d.early_latent, d.late_latent, d.tertiary, d.treated]
        f[death_uids] = false
    end
    return d
end

function Starsim.update_results!(d::Syphilis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    n_sus=0; n_inf=0; n_pri=0; n_sec=0; n_el=0; n_ll=0; n_tert=0
    @inbounds for u in active
        n_sus  += d.infection.susceptible.raw[u]
        n_inf  += d.infection.infected.raw[u]
        n_pri  += d.primary.raw[u]
        n_sec  += d.secondary.raw[u]
        n_el   += d.early_latent.raw[u]
        n_ll   += d.late_latent.raw[u]
        n_tert += d.tertiary.raw[u]
    end

    md.results[:n_susceptible][ti]  = Float64(n_sus)
    md.results[:n_infected][ti]     = Float64(n_inf)
    md.results[:n_primary][ti]      = Float64(n_pri)
    md.results[:n_secondary][ti]    = Float64(n_sec)
    md.results[:n_early_latent][ti] = Float64(n_el)
    md.results[:n_late_latent][ti]  = Float64(n_ll)
    md.results[:n_tertiary][ti]     = Float64(n_tert)

    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
    return d
end

function Starsim.finalize!(d::Syphilis)
    md = Starsim.module_data(d)
    for (_, _, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end
