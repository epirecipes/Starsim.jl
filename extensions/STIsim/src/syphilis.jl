"""
Syphilis disease — multi-stage: Exposed → Primary → Secondary → Latent (early/late) → Tertiary.
Port of Python `stisim.diseases.syphilis.Syphilis` (v1.5.0).
"""

# ============================================================================
# Syphilis <: AbstractInfection
# ============================================================================

"""
    Syphilis <: AbstractInfection

Multi-stage syphilis model matching Python stisim v1.5.0.

Stages: Exposed → Primary → Secondary → Latent (early/late) → Tertiary
- Exposed stage is non-infectious incubation (~50 days)
- Latent transmissibility decays with half-life of 1 year
- Reactivation from latent effectively disabled (matching Python bug)
- Treatment cures at any stage
"""
mutable struct Syphilis <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # Stage states
    exposed::Starsim.StateVector{Bool, Vector{Bool}}
    ti_exposed::Starsim.StateVector{Float64, Vector{Float64}}
    primary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_primary::Starsim.StateVector{Float64, Vector{Float64}}
    secondary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_secondary::Starsim.StateVector{Float64, Vector{Float64}}
    latent::Starsim.StateVector{Bool, Vector{Bool}}
    ti_latent::Starsim.StateVector{Float64, Vector{Float64}}
    dur_early::Starsim.StateVector{Float64, Vector{Float64}}   # per-agent early latent duration (timesteps)
    early_latent::Starsim.StateVector{Bool, Vector{Bool}}       # dynamically computed from dur_early
    late_latent::Starsim.StateVector{Bool, Vector{Bool}}
    tertiary::Starsim.StateVector{Bool, Vector{Bool}}
    ti_tertiary::Starsim.StateVector{Float64, Vector{Float64}}
    ti_dead::Starsim.StateVector{Float64, Vector{Float64}}
    treated::Starsim.StateVector{Bool, Vector{Bool}}

    # Parameters (durations in years, matching Python's raw values before .to(dt))
    beta_m2f::Float64
    rel_beta_f2m::Float64
    beta_m2m::Float64
    eff_condom::Float64
    # dur_exposed: Normal(50 days, 10 days) — mean/std in years
    dur_exposed_mean::Float64
    dur_exposed_std::Float64
    # dur_primary: Normal(6 weeks, 1 week)
    dur_primary_mean::Float64
    dur_primary_std::Float64
    # dur_secondary: LogNormal_ex(3.6 months, 1.5 months) — mean/std of actual dist
    dur_secondary_mean::Float64
    dur_secondary_std::Float64
    # dur_early: Uniform(12 months, 14 months)
    dur_early_low::Float64
    dur_early_high::Float64
    # Reactivation and tertiary
    p_reactivate::Float64
    time_to_reactivate_mean::Float64    # lognorm_ex mean (years)
    time_to_reactivate_std::Float64
    p_tertiary::Float64
    time_to_tertiary_mean::Float64      # Normal mean (years)
    time_to_tertiary_std::Float64
    p_death::Float64
    time_to_death_mean::Float64         # lognorm_ex mean (years)
    time_to_death_std::Float64
    # Transmissibility by stage
    rel_trans_primary::Float64
    rel_trans_secondary::Float64
    rel_trans_latent::Float64
    rel_trans_latent_half_life::Float64  # years (converted to timesteps in step_state)
    rel_trans_tertiary::Float64

    rng::StableRNG
end

# Helper: draw from lognormal parameterized by actual mean/std (matching Python lognorm_ex)
function _draw_lognorm_ex(rng::StableRNG, mean::Float64, std::Float64)
    σ² = log(1.0 + (std / mean)^2)
    μ  = log(mean) - σ² / 2.0
    return rand(rng, LogNormal(μ, sqrt(σ²)))
end

function Syphilis(;
    name::Symbol              = :syphilis,
    init_prev::Real           = 0.0,
    beta_m2f::Real            = 0.1,
    rel_beta_f2m::Real        = 0.5,
    beta_m2m::Real            = 0.1,
    eff_condom::Real          = 0.0,
    # Duration parameters match Python stisim v1.5.0 defaults (stored in years)
    dur_exposed_mean::Real    = 50/365,    # Normal(50 days, 10 days)
    dur_exposed_std::Real     = 10/365,
    dur_primary_mean::Real    = 42/365,    # Normal(6 weeks, 1 week)
    dur_primary_std::Real     = 7/365,
    dur_secondary_mean::Real  = 3.6/12,    # LogNormal_ex(3.6 months, 1.5 months)
    dur_secondary_std::Real   = 1.5/12,
    dur_early_low::Real       = 12/12,     # Uniform(12 months, 14 months)
    dur_early_high::Real      = 14/12,
    p_reactivate::Real        = 0.35,
    time_to_reactivate_mean::Real = 1.0,   # lognorm_ex(1 year, 1 year)
    time_to_reactivate_std::Real  = 1.0,
    p_tertiary::Real          = 0.35,
    time_to_tertiary_mean::Real = 20.0,    # Normal(20 years, 2 years)
    time_to_tertiary_std::Real  = 2.0,
    p_death::Real             = 0.05,
    time_to_death_mean::Real  = 5.0,       # lognorm_ex(5 years, 5 years)
    time_to_death_std::Real   = 5.0,
    rel_trans_primary::Real   = 1.0,
    rel_trans_secondary::Real = 1.0,
    rel_trans_latent::Real    = 1.0,
    rel_trans_latent_half_life::Real = 1.0,
    rel_trans_tertiary::Real  = 0.0,
)
    inf = Starsim.InfectionData(name; init_prev=Float64(init_prev), beta=Float64(beta_m2f), label="Syphilis")

    Syphilis(
        inf,
        Starsim.BoolState(:exposed; default=false, label="Exposed syphilis"),
        Starsim.FloatState(:ti_exposed; default=Inf, label="Time exposed"),
        Starsim.BoolState(:primary; default=false, label="Primary syphilis"),
        Starsim.FloatState(:ti_primary; default=Inf, label="Time primary"),
        Starsim.BoolState(:secondary; default=false, label="Secondary syphilis"),
        Starsim.FloatState(:ti_secondary; default=Inf, label="Time secondary"),
        Starsim.BoolState(:latent; default=false, label="Latent syphilis"),
        Starsim.FloatState(:ti_latent; default=Inf, label="Time latent"),
        Starsim.FloatState(:dur_early; default=Inf, label="Duration early latent (ts)"),
        Starsim.BoolState(:early_latent; default=false, label="Early latent"),
        Starsim.BoolState(:late_latent; default=false, label="Late latent"),
        Starsim.BoolState(:tertiary; default=false, label="Tertiary syphilis"),
        Starsim.FloatState(:ti_tertiary; default=Inf, label="Time tertiary"),
        Starsim.FloatState(:ti_dead; default=Inf, label="Time syphilis death"),
        Starsim.BoolState(:treated; default=false, label="Treated"),
        Float64(beta_m2f), Float64(rel_beta_f2m), Float64(beta_m2m),
        Float64(eff_condom),
        Float64(dur_exposed_mean), Float64(dur_exposed_std),
        Float64(dur_primary_mean), Float64(dur_primary_std),
        Float64(dur_secondary_mean), Float64(dur_secondary_std),
        Float64(dur_early_low), Float64(dur_early_high),
        Float64(p_reactivate),
        Float64(time_to_reactivate_mean), Float64(time_to_reactivate_std),
        Float64(p_tertiary),
        Float64(time_to_tertiary_mean), Float64(time_to_tertiary_std),
        Float64(p_death),
        Float64(time_to_death_mean), Float64(time_to_death_std),
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
        d.exposed, d.ti_exposed,
        d.primary, d.ti_primary, d.secondary, d.ti_secondary,
        d.latent, d.ti_latent, d.dur_early,
        d.early_latent, d.late_latent,
        d.tertiary, d.ti_tertiary, d.ti_dead, d.treated,
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
        Starsim.Result(:n_exposed; npts=npts, label="Exposed", scale=false),
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

    # Python: hl = self.pars.rel_trans_latent_half_life → years(1) acts as raw 1.0
    # Python does NOT convert to timesteps; decay_rate = log(2)/1.0 = 0.693
    hl = d.rel_trans_latent_half_life

    @inbounds for u in active
        if !d.infection.infected.raw[u]
            continue
        end

        # Exposed → Primary (matching Python step_state order)
        if d.exposed.raw[u] && d.ti_primary.raw[u] <= ti_f
            d.exposed.raw[u] = false
            d.primary.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_primary
        end

        # Primary → Secondary
        if d.primary.raw[u] && d.ti_secondary.raw[u] <= ti_f
            d.primary.raw[u] = false
            d.secondary.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_secondary
        end

        # Secondary → Latent
        if d.secondary.raw[u] && d.ti_latent.raw[u] <= ti_f
            d.secondary.raw[u] = false
            d.latent.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_latent
        end

        # Latent transmissibility: exponential decay with half-life
        # Python: decay_rate = log(2) / hl; dur = ti - ti_latent (both in timestep units)
        if d.latent.raw[u]
            dur_latent_ts = ti_f - d.ti_latent.raw[u]
            if hl > 0.0
                decay_rate = log(2.0) / hl
                d.infection.rel_trans.raw[u] = d.rel_trans_latent * exp(-decay_rate * dur_latent_ts)
            else
                d.infection.rel_trans.raw[u] = d.rel_trans_latent
            end

            # Dynamically compute early/late latent from per-agent dur_early
            if dur_latent_ts <= d.dur_early.raw[u]
                d.early_latent.raw[u] = true
                d.late_latent.raw[u] = false
            else
                d.early_latent.raw[u] = false
                d.late_latent.raw[u] = true
            end
        end

        # Latent → Tertiary
        if d.latent.raw[u] && d.ti_tertiary.raw[u] <= ti_f
            d.latent.raw[u] = false
            d.early_latent.raw[u] = false
            d.late_latent.raw[u] = false
            d.tertiary.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_tertiary
        end

        # Tertiary → Death
        if d.tertiary.raw[u] && d.ti_dead.raw[u] <= ti_f
            d.infection.infected.raw[u] = false
            d.tertiary.raw[u] = false
            sim.people.alive.raw[u] = false
            sim.people.ti_dead.raw[u] = ti_f
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

    # SNAPSHOT state arrays before edge loop (matching Python's synchronous update)
    n_agents = length(d.infection.infected.raw)
    inf_snap = copy(d.infection.infected.raw)
    sus_snap = copy(d.infection.susceptible.raw)
    rt_snap  = copy(d.infection.rel_trans.raw)
    rs_snap  = copy(d.infection.rel_sus.raw)

    @inbounds for u in 1:n_agents
        if !inf_snap[u];  rt_snap[u] = 0.0; end
        if !sus_snap[u]; rs_snap[u] = 0.0; end
    end

    female_raw = sim.people.female.raw
    rng = d.rng
    dt = sim.pars.dt

    @inbounds for i in 1:length(edges)
        src = p1[i]; trg = p2[i]; edge_beta = eb[i]; acts = ea[i]
        if inf_snap[src] && sus_snap[trg]
            ba = _get_directional_beta(female_raw[src], female_raw[trg], d.beta_m2f, d.beta_m2f*d.rel_beta_f2m, d.beta_m2m)
            net_beta_val = (1.0 - (1.0 - ba)^acts) * edge_beta
            p = clamp(rt_snap[src] * rs_snap[trg] * net_beta_val, 0.0, 1.0)
            if rand(rng) < p
                _do_syph_infection!(d, sim, trg, src, ti)
                sus_snap[trg] = false  # Prevent re-infection (matches Python dedup)
                new_infections += 1
            end
        end
        if bidir && inf_snap[trg] && sus_snap[src]
            ba = _get_directional_beta(female_raw[trg], female_raw[src], d.beta_m2f, d.beta_m2f*d.rel_beta_f2m, d.beta_m2m)
            net_beta_val = (1.0 - (1.0 - ba)^acts) * edge_beta
            p = clamp(rt_snap[trg] * rs_snap[src] * net_beta_val, 0.0, 1.0)
            if rand(rng) < p
                _do_syph_infection!(d, sim, src, trg, ti)
                sus_snap[src] = false  # Prevent re-infection (matches Python dedup)
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

    # Start at exposed (non-infectious incubation period)
    d.exposed.raw[target] = true
    d.ti_exposed.raw[target] = Float64(ti)
    d.infection.rel_trans.raw[target] = 0.0  # Not infectious during incubation

    # Exposed → Primary: Normal(50 days, 10 days), converted to timesteps, rounded
    dur_exp_ts = round(max(1.0, rand(d.rng, Normal(d.dur_exposed_mean / dt, d.dur_exposed_std / dt))))
    d.ti_primary.raw[target] = Float64(ti) + dur_exp_ts

    # Primary → Secondary: Normal(6 weeks, 1 week), converted to timesteps, rounded
    dur_pri_ts = round(max(1.0, rand(d.rng, Normal(d.dur_primary_mean / dt, d.dur_primary_std / dt))))
    d.ti_secondary.raw[target] = d.ti_primary.raw[target] + dur_pri_ts

    # Per-agent dur_early: Uniform(12 months, 14 months) in timesteps, rounded
    d.dur_early.raw[target] = round(rand(d.rng, Uniform(d.dur_early_low / dt, d.dur_early_high / dt)))

    # Secondary → Latent: LogNormal_ex(3.6 months, 1.5 months) in timesteps, rounded
    dur_sec_ts = round(max(1.0, _draw_lognorm_ex(d.rng, d.dur_secondary_mean / dt, d.dur_secondary_std / dt)))
    d.ti_latent.raw[target] = d.ti_secondary.raw[target] + dur_sec_ts

    # Latent prognoses: reactivation/tertiary/stay (matching Python set_latent_prognoses)
    # Python's reactivation condition `(ti_latent >= ti)` is never satisfiable,
    # so reactivation never fires. Pre-scheduling all branching at infection.
    will_reactivate = rand(d.rng) < d.p_reactivate
    if will_reactivate
        # Draw reactivation time (consumed for RNG parity) but it never fires
        _draw_lognorm_ex(d.rng, d.time_to_reactivate_mean / dt, d.time_to_reactivate_std / dt)
        d.ti_tertiary.raw[target] = Inf
        d.ti_dead.raw[target] = Inf
    else
        will_tertiary = rand(d.rng) < d.p_tertiary
        if will_tertiary
            dur_tert_ts = round(max(1.0, rand(d.rng, Normal(d.time_to_tertiary_mean / dt, d.time_to_tertiary_std / dt))))
            d.ti_tertiary.raw[target] = d.ti_latent.raw[target] + dur_tert_ts
            if rand(d.rng) < d.p_death
                dur_death_ts = round(max(1.0, _draw_lognorm_ex(d.rng, d.time_to_death_mean / dt, d.time_to_death_std / dt)))
                d.ti_dead.raw[target] = d.ti_tertiary.raw[target] + dur_death_ts
            else
                d.ti_dead.raw[target] = Inf
            end
        else
            d.ti_tertiary.raw[target] = Inf
            d.ti_dead.raw[target] = Inf
        end
    end

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
    for f in [d.infection.susceptible, d.infection.infected,
              d.exposed, d.primary, d.secondary,
              d.latent, d.early_latent, d.late_latent, d.tertiary, d.treated]
        f[death_uids] = false
    end
    for u in death_uids.values
        d.ti_dead.raw[u] = Inf
    end
    return d
end

function Starsim.update_results!(d::Syphilis, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    n_sus=0; n_inf=0; n_exp=0; n_pri=0; n_sec=0; n_el=0; n_ll=0; n_tert=0
    @inbounds for u in active
        n_sus  += d.infection.susceptible.raw[u]
        n_inf  += d.infection.infected.raw[u]
        n_exp  += d.exposed.raw[u]
        n_pri  += d.primary.raw[u]
        n_sec  += d.secondary.raw[u]
        n_el   += d.early_latent.raw[u]
        n_ll   += d.late_latent.raw[u]
        n_tert += d.tertiary.raw[u]
    end

    md.results[:n_susceptible][ti]  = Float64(n_sus)
    md.results[:n_infected][ti]     = Float64(n_inf)
    md.results[:n_exposed][ti]      = Float64(n_exp)
    md.results[:n_primary][ti]      = Float64(n_pri)
    md.results[:n_secondary][ti]    = Float64(n_sec)
    md.results[:n_early_latent][ti] = Float64(n_el)
    md.results[:n_late_latent][ti]  = Float64(n_ll)
    md.results[:n_tertiary][ti]     = Float64(n_tert)

    # Prevalence among sexually active adults (matching Python's definition)
    n_sa = 0; n_inf_sa = 0
    age_raw = sim.people.age.raw
    debut_raw = nothing
    for (_, net) in sim.networks
        if net isa StructuredSexual
            debut_raw = getfield(net, :debut_state).raw
            break
        end
    end
    @inbounds for u in active
        age_u = age_raw[u]
        if age_u >= 15.0 && age_u <= 50.0 && (debut_raw === nothing || age_u > debut_raw[u])
            n_sa += 1
            n_inf_sa += d.infection.infected.raw[u]
        end
    end
    md.results[:prevalence][ti] = n_sa > 0 ? Float64(n_inf_sa) / Float64(n_sa) : 0.0
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
