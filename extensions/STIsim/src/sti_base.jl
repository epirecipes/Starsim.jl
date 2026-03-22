"""
Base STI disease — SEIS pattern with per-act transmission.
Port of Python `stisim.diseases.sti.BaseSTI` and `stisim.diseases.sti.SEIS`.

The core STI transmission differs from basic SIR by using per-act transmission:
  effective_beta = 1 - (1 - beta_per_act * (1 - condom_eff * condom_use))^n_acts
Bidirectional with separate beta_m2f, beta_f2m, beta_m2m.
"""

# ============================================================================
# Lognormal sampling helper (same as in hiv.jl)
# ============================================================================

"""
Sample from an *explicit* lognormal with the given mean and std.
Converts to underlying normal parameters: given lognormal mean μ and std σ,
  σ²_n = log(1 + (σ/μ)²),  μ_n = log(μ) - σ²_n/2.
"""
function _seis_lognorm_ex_sample(rng, mean_val::Float64, std_val::Float64)
    mean_val <= 0.0 && return mean_val
    std_val <= 0.0 && return mean_val
    cv = std_val / mean_val
    sigma2 = log(1.0 + cv^2)
    mu = log(mean_val) - sigma2 / 2.0
    return exp(mu + sqrt(sigma2) * randn(rng))
end

# ============================================================================
# SEIS <: AbstractInfection  — base SEIS disease for simple STIs
# ============================================================================

"""
    SEIS <: AbstractInfection

SEIS (Susceptible-Exposed-Infected-Susceptible) base model for STIs.
After clearance, agents return to susceptible. Supports:
- Per-act transmission with condom efficacy
- Direction-specific betas (m2f, f2m, m2m)
- Exposed period before infectiousness
- Symptomatic vs asymptomatic branches with sex-specific durations
- Sex-specific clearance durations (lognormal)
- PID (pelvic inflammatory disease) for females

# Keyword arguments
- `name::Symbol` — disease name
- `init_prev::Real` — initial prevalence (default 0.01)
- `beta_m2f::Real` — per-act male-to-female beta (default 0.05)
- `rel_beta_f2m::Real` — relative female-to-male beta (default 0.5)
- `beta_m2m::Real` — per-act MSM beta (default 0.05)
- `eff_condom::Real` — condom efficacy (default 0.9)
- `dur_exp::Real` — exposed duration in years (default 1/52, ~1 week)
- `p_symp_f::Real` — probability symptomatic for females (default 0.375)
- `p_symp_m::Real` — probability symptomatic for males (default 0.375)
- `dur_inf::Real` — mean infection duration in years (fallback, default 1.0)
- `dur_inf_std::Real` — std of infection duration in years (fallback, default 0.1)
- `dur_asymp2clear_f::Real` — female asymptomatic→clearance mean (years; -1 = use dur_inf)
- `dur_asymp2clear_f_std::Real` — female asymptomatic→clearance std (years)
- `dur_asymp2clear_m::Real` — male asymptomatic→clearance mean (years; -1 = use dur_inf)
- `dur_asymp2clear_m_std::Real` — male asymptomatic→clearance std (years)
- `dur_symp2clear_f::Real` — female symptomatic→clearance mean (years; -1 = use dur_inf)
- `dur_symp2clear_f_std::Real` — female symptomatic→clearance std (years)
- `dur_symp2clear_m::Real` — male symptomatic→clearance mean (years; -1 = use dur_inf)
- `dur_symp2clear_m_std::Real` — male symptomatic→clearance std (years)
- `dur_presymp_f::Real` — female presymptomatic period mean (years; -1 = use dur_exp-based)
- `dur_presymp_f_std::Real` — female presymptomatic period std (years)
- `dur_presymp_m::Real` — male presymptomatic period mean (years; -1 = use dur_exp-based)
- `dur_presymp_m_std::Real` — male presymptomatic period std (years)
- `p_pid::Real` — probability of PID for infected females (default 0.2)
"""
mutable struct SEIS <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # SEIS-specific states
    exposed::Starsim.StateVector{Bool, Vector{Bool}}
    ti_exposed::Starsim.StateVector{Float64, Vector{Float64}}
    symptomatic::Starsim.StateVector{Bool, Vector{Bool}}
    ti_symptomatic::Starsim.StateVector{Float64, Vector{Float64}}
    ti_clearance::Starsim.StateVector{Float64, Vector{Float64}}
    pid::Starsim.StateVector{Bool, Vector{Bool}}
    ti_pid::Starsim.StateVector{Float64, Vector{Float64}}
    n_infections::Starsim.StateVector{Float64, Vector{Float64}}

    # Parameters
    beta_m2f::Float64
    rel_beta_f2m::Float64
    beta_m2m::Float64
    eff_condom::Float64
    dur_exp::Float64
    p_symp_f::Float64
    p_symp_m::Float64
    dur_inf::Float64
    dur_inf_std::Float64
    p_pid::Float64

    # Sex-specific clearance durations (years); -1.0 = use dur_inf fallback
    dur_asymp2clear_f::Float64
    dur_asymp2clear_f_std::Float64
    dur_asymp2clear_m::Float64
    dur_asymp2clear_m_std::Float64
    dur_symp2clear_f::Float64
    dur_symp2clear_f_std::Float64
    dur_symp2clear_m::Float64
    dur_symp2clear_m_std::Float64

    # Sex-specific presymptomatic period (years); -1.0 = use dur_exp-based fallback
    dur_presymp_f::Float64
    dur_presymp_f_std::Float64
    dur_presymp_m::Float64
    dur_presymp_m_std::Float64

    # Age range for prevalence computation (matching Python)
    age_lo::Float64
    age_hi::Float64

    rng::StableRNG
end

# Alias for clarity
const BaseSTI = SEIS

function SEIS(;
    name::Symbol       = :sti,
    init_prev::Real    = 0.01,
    beta_m2f::Real     = 0.05,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.05,
    eff_condom::Real   = 0.9,
    dur_exp::Real      = 1/52,
    p_symp_f::Real     = 0.375,
    p_symp_m::Real     = 0.375,
    dur_inf::Real      = 1.0,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.2,
    dur_asymp2clear_f::Real     = -1.0,
    dur_asymp2clear_f_std::Real = -1.0,
    dur_asymp2clear_m::Real     = -1.0,
    dur_asymp2clear_m_std::Real = -1.0,
    dur_symp2clear_f::Real      = -1.0,
    dur_symp2clear_f_std::Real  = -1.0,
    dur_symp2clear_m::Real      = -1.0,
    dur_symp2clear_m_std::Real  = -1.0,
    dur_presymp_f::Real         = -1.0,
    dur_presymp_f_std::Real     = -1.0,
    dur_presymp_m::Real         = -1.0,
    dur_presymp_m_std::Real     = -1.0,
    age_lo::Real                = 15.0,
    age_hi::Real                = 65.0,
    label::String      = "SEIS STI",
)
    inf = Starsim.InfectionData(name; init_prev=Float64(init_prev), beta=Float64(beta_m2f), label=label)

    SEIS(
        inf,
        Starsim.BoolState(:exposed; default=false, label="Exposed"),
        Starsim.FloatState(:ti_exposed; default=Inf, label="Time exposed"),
        Starsim.BoolState(:symptomatic; default=false, label="Symptomatic"),
        Starsim.FloatState(:ti_symptomatic; default=Inf, label="Time symptomatic"),
        Starsim.FloatState(:ti_clearance; default=Inf, label="Time of clearance"),
        Starsim.BoolState(:pid; default=false, label="PID"),
        Starsim.FloatState(:ti_pid; default=Inf, label="Time of PID"),
        Starsim.FloatState(:n_infections; default=0.0, label="Infection count"),
        Float64(beta_m2f),
        Float64(rel_beta_f2m),
        Float64(beta_m2m),
        Float64(eff_condom),
        Float64(dur_exp),
        Float64(p_symp_f),
        Float64(p_symp_m),
        Float64(dur_inf),
        Float64(dur_inf_std),
        Float64(p_pid),
        Float64(dur_asymp2clear_f),
        Float64(dur_asymp2clear_f_std),
        Float64(dur_asymp2clear_m),
        Float64(dur_asymp2clear_m_std),
        Float64(dur_symp2clear_f),
        Float64(dur_symp2clear_f_std),
        Float64(dur_symp2clear_m),
        Float64(dur_symp2clear_m_std),
        Float64(dur_presymp_f),
        Float64(dur_presymp_f_std),
        Float64(dur_presymp_m),
        Float64(dur_presymp_m_std),
        Float64(age_lo),
        Float64(age_hi),
        StableRNG(0),
    )
end

Starsim.disease_data(d::SEIS) = d.infection.dd
Starsim.module_data(d::SEIS)  = d.infection.dd.mod

# ============================================================================
# Lifecycle
# ============================================================================

function Starsim.init_pre!(d::SEIS, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Register states
    states = [
        d.infection.susceptible, d.infection.infected,
        d.infection.ti_infected, d.infection.rel_sus, d.infection.rel_trans,
        d.exposed, d.ti_exposed, d.symptomatic, d.ti_symptomatic,
        d.ti_clearance, d.pid, d.ti_pid, d.n_infections,
    ]
    for s in states
        Starsim.add_module_state!(sim.people, s)
    end

    # Compute per-dt betas for all networks
    _validate_sti_beta!(d, sim)

    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:new_infections; npts=npts, label="New infections"),
        Starsim.Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Starsim.Result(:n_exposed; npts=npts, label="Exposed", scale=false),
        Starsim.Result(:n_infected; npts=npts, label="Infected", scale=false),
        Starsim.Result(:n_symptomatic; npts=npts, label="Symptomatic", scale=false),
        Starsim.Result(:n_pid; npts=npts, label="PID", scale=false),
        Starsim.Result(:prevalence; npts=npts, label="Prevalence", scale=false),
        Starsim.Result(:incidence; npts=npts, label="Incidence", scale=false),
    )

    md.initialized = true
    return d
end

"""Compute per-dt betas using per-act formula for STI networks."""
function _validate_sti_beta!(d::SEIS, sim)
    dd = Starsim.disease_data(d)
    dt = sim.pars.dt

    for (name, _) in sim.networks
        if name == :mf || name == :structuredsexual
            # Use per-act formula for sexual networks: store beta_m2f as base
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2f) * dt)
        elseif name == :msm
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2m) * dt)
        elseif name == :maternal
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2f) * 0.5 * dt)
        else
            dd.beta_per_dt[name] = 1.0 - exp(-Float64(d.beta_m2f) * dt)
        end
    end
    return d
end

function Starsim.init_post!(d::SEIS, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    n_infect = max(1, Int(round(d.infection.dd.init_prev * n)))
    n_infect = min(n_infect, n)

    if d.infection.dd.init_prev <= 0.0
        return d
    end

    infect_uids = Starsim.UIDs(active[randperm(d.rng, n)[1:n_infect]])
    ti = 1
    for u in infect_uids.values
        _do_seis_infection!(d, sim, u, 0, ti)
    end

    return d
end

# ============================================================================
# State transitions
# ============================================================================

function Starsim.step_state!(d::SEIS, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti_f = Float64(ti)
    active = sim.people.auids.values
    dt = sim.pars.dt

    exposed_raw     = d.exposed.raw
    ti_exposed_raw  = d.ti_exposed.raw
    infected_raw    = d.infection.infected.raw
    ti_inf_raw      = d.infection.ti_infected.raw
    susceptible_raw = d.infection.susceptible.raw
    symptomatic_raw = d.symptomatic.raw
    ti_symp_raw     = d.ti_symptomatic.raw
    ti_clear_raw    = d.ti_clearance.raw
    pid_raw         = d.pid.raw
    ti_pid_raw      = d.ti_pid.raw
    female_raw      = sim.people.female.raw

    # Exposed → Infected transition
    @inbounds for u in active
        if exposed_raw[u] && ti_inf_raw[u] <= ti_f
            exposed_raw[u] = false
            infected_raw[u] = true
        end
    end

    # Infected → Cleared (return to susceptible)
    @inbounds for u in active
        if infected_raw[u] && ti_clear_raw[u] <= ti_f
            infected_raw[u] = false
            susceptible_raw[u] = true
            symptomatic_raw[u] = false
            pid_raw[u] = false
        end
    end

    # Symptomatic onset
    @inbounds for u in active
        if infected_raw[u] && !symptomatic_raw[u] && ti_symp_raw[u] <= ti_f
            symptomatic_raw[u] = true
        end
    end

    # PID onset for females
    @inbounds for u in active
        if infected_raw[u] && female_raw[u] && !pid_raw[u] && ti_pid_raw[u] <= ti_f
            pid_raw[u] = true
        end
    end

    return d
end

# ============================================================================
# Per-act transmission
# ============================================================================

function Starsim.step!(d::SEIS, sim)
    return _infect_sti!(d, sim)
end

function _infect_sti!(d::SEIS, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dd = Starsim.disease_data(d)
    new_infections = 0

    for (net_name, net) in sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        new_infections += _infect_sti_edges!(d, sim, edges, beta_dt, ti, net, net_name)
    end

    return new_infections
end

"""Per-act STI transmission across edges with direction-specific betas."""
function _infect_sti_edges!(d::SEIS, sim, edges::Starsim.Edges, beta_dt::Float64, ti::Int, net, net_name::Symbol)
    new_infections = 0
    n_edges = length(edges)
    bidir = Starsim.network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    ea = edges.acts

    # SNAPSHOT state arrays before edge loop (matching Python's synchronous update:
    # rel_trans/rel_sus are computed once; new infections don't affect this step)
    n_agents = length(d.infection.infected.raw)
    infected_snap    = copy(d.infection.infected.raw)
    exposed_snap     = copy(d.exposed.raw)
    susceptible_snap = copy(d.infection.susceptible.raw)
    rel_trans_snap   = copy(d.infection.rel_trans.raw)
    rel_sus_snap     = copy(d.infection.rel_sus.raw)

    # Mask: non-infectious get rel_trans=0, non-susceptible get rel_sus=0
    @inbounds for u in 1:n_agents
        if !infected_snap[u]
            rel_trans_snap[u] = 0.0
        end
        if !susceptible_snap[u]
            rel_sus_snap[u] = 0.0
        end
    end

    female_raw      = sim.people.female.raw
    rng = d.rng

    beta_m2f = d.beta_m2f
    beta_f2m = d.beta_m2f * d.rel_beta_f2m
    beta_m2m = d.beta_m2m
    eff_condom = d.eff_condom
    dt = sim.pars.dt

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]
        acts = ea[i]

        # src → trg
        if (infected_snap[src] || exposed_snap[src]) && susceptible_snap[trg] && !exposed_snap[trg]
            if infected_snap[src]  # Only infectious (not exposed) can transmit
                beta_act = _get_directional_beta(female_raw[src], female_raw[trg], beta_m2f, beta_f2m, beta_m2m)
                net_beta_val = (1.0 - (1.0 - beta_act)^acts) * edge_beta
                p = clamp(rel_trans_snap[src] * rel_sus_snap[trg] * net_beta_val, 0.0, 1.0)
                if rand(rng) < p
                    _do_seis_infection!(d, sim, trg, src, ti)
                    susceptible_snap[trg] = false  # Prevent re-infection (matches Python dedup)
                    new_infections += 1
                end
            end
        end

        # trg → src (bidirectional)
        if bidir && (infected_snap[trg] || exposed_snap[trg]) && susceptible_snap[src] && !exposed_snap[src]
            if infected_snap[trg]
                beta_act = _get_directional_beta(female_raw[trg], female_raw[src], beta_m2f, beta_f2m, beta_m2m)
                net_beta_val = (1.0 - (1.0 - beta_act)^acts) * edge_beta
                p = clamp(rel_trans_snap[trg] * rel_sus_snap[src] * net_beta_val, 0.0, 1.0)
                if rand(rng) < p
                    _do_seis_infection!(d, sim, src, trg, ti)
                    susceptible_snap[src] = false  # Prevent re-infection (matches Python dedup)
                    new_infections += 1
                end
            end
        end
    end
    return new_infections
end

"""Get directional beta based on sex of source and target."""
function _get_directional_beta(src_female::Bool, trg_female::Bool, beta_m2f, beta_f2m, beta_m2m)
    if !src_female && trg_female
        return beta_m2f      # male → female
    elseif src_female && !trg_female
        return beta_f2m      # female → male
    elseif !src_female && !trg_female
        return beta_m2m      # male → male
    else
        return beta_m2f * 0.5  # female → female (rare, reduced)
    end
end

"""Infect a single agent with SEIS prognosis (sex-specific lognormal clearance)."""
function _do_seis_infection!(d::SEIS, sim, target::Int, source::Int, ti::Int)
    dt = sim.pars.dt
    female = sim.people.female.raw[target]
    rng = d.rng

    d.infection.susceptible.raw[target] = false
    d.exposed.raw[target] = true
    d.ti_exposed.raw[target] = Float64(ti)

    # Wipe previous dates (matching Python's wipe_dates)
    d.ti_symptomatic.raw[target] = Inf
    d.ti_clearance.raw[target] = Inf
    d.ti_pid.raw[target] = Inf

    # Time to become infectious (exposed → infected)
    if d.dur_exp > 0.0
        dur_exp_ts = d.dur_exp / dt
        ti_inf = Float64(ti) + max(1.0, dur_exp_ts + randn(rng) * dur_exp_ts * 0.3)
    else
        ti_inf = Float64(ti)  # Immediate infectiousness (e.g. gonorrhea)
    end
    d.infection.ti_infected.raw[target] = ti_inf

    # Symptomatic?
    p_symp = female ? d.p_symp_f : d.p_symp_m
    is_symp = rand(rng) < p_symp

    if is_symp
        # Presymptomatic period (sex-specific lognormal if available)
        presymp_mean = female ? d.dur_presymp_f : d.dur_presymp_m
        presymp_std  = female ? d.dur_presymp_f_std : d.dur_presymp_m_std
        if presymp_mean >= 0.0
            presymp_dur_years = _seis_lognorm_ex_sample(rng, presymp_mean, presymp_std)
            presymp_ts = max(0.0, presymp_dur_years / dt)
        else
            # Fallback: use dur_exp-based heuristic
            dur_exp_ts = d.dur_exp / dt
            presymp_ts = max(1.0, dur_exp_ts * 0.5 + randn(rng) * dur_exp_ts * 0.2)
        end
        d.ti_symptomatic.raw[target] = ti_inf + presymp_ts

        # Symptomatic clearance (sex-specific lognormal if available)
        clear_mean = female ? d.dur_symp2clear_f : d.dur_symp2clear_m
        clear_std  = female ? d.dur_symp2clear_f_std : d.dur_symp2clear_m_std
        if clear_mean >= 0.0
            clear_dur_years = _seis_lognorm_ex_sample(rng, clear_mean, clear_std)
            clear_ts = max(1.0, clear_dur_years / dt)
            d.ti_clearance.raw[target] = d.ti_symptomatic.raw[target] + clear_ts
        else
            # Fallback: use dur_inf
            dur_ts = max(1.0, d.dur_inf / dt + randn(rng) * (d.dur_inf_std / dt))
            d.ti_clearance.raw[target] = ti_inf + dur_ts
        end
    else
        # Asymptomatic clearance (sex-specific lognormal if available)
        clear_mean = female ? d.dur_asymp2clear_f : d.dur_asymp2clear_m
        clear_std  = female ? d.dur_asymp2clear_f_std : d.dur_asymp2clear_m_std
        if clear_mean >= 0.0
            clear_dur_years = _seis_lognorm_ex_sample(rng, clear_mean, clear_std)
            clear_ts = max(1.0, clear_dur_years / dt)
            d.ti_clearance.raw[target] = ti_inf + clear_ts
        else
            # Fallback: use dur_inf
            dur_ts = max(1.0, d.dur_inf / dt + randn(rng) * (d.dur_inf_std / dt))
            d.ti_clearance.raw[target] = ti_inf + dur_ts
        end
    end

    # PID for females
    if female && rand(rng) < d.p_pid
        pid_delay = max(1.0, 6.0 / (dt * 52) + randn(rng) * 2.0 / (dt * 52))
        d.ti_pid.raw[target] = ti_inf + pid_delay
    end

    d.n_infections.raw[target] += 1.0
    push!(d.infection.infection_sources, (target, source, ti))
    return
end

function Starsim.set_prognoses!(d::SEIS, sim, uids::Starsim.UIDs)
    ti = Starsim.module_data(d).t.ti
    for u in uids.values
        _do_seis_infection!(d, sim, u, 0, ti)
    end
    return
end

# ============================================================================
# Death handling and results
# ============================================================================

function Starsim.step_die!(d::SEIS, death_uids::Starsim.UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids]    = false
    d.exposed[death_uids]               = false
    d.symptomatic[death_uids]           = false
    d.pid[death_uids]                   = false
    return d
end

function Starsim.update_results!(d::SEIS, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw  = d.infection.susceptible.raw
    exp_raw  = d.exposed.raw
    inf_raw  = d.infection.infected.raw
    symp_raw = d.symptomatic.raw
    pid_raw  = d.pid.raw

    n_sus = 0; n_exp = 0; n_inf = 0; n_symp = 0; n_pid = 0
    @inbounds for u in active
        n_sus  += sus_raw[u]
        n_exp  += exp_raw[u]
        n_inf  += inf_raw[u]
        n_symp += symp_raw[u]
        n_pid  += pid_raw[u]
    end

    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_exposed][ti]     = Float64(n_exp)
    md.results[:n_infected][ti]    = Float64(n_inf)
    md.results[:n_symptomatic][ti] = Float64(n_symp)
    md.results[:n_pid][ti]         = Float64(n_pid)

    # Prevalence among adults (matching Python's age_range)
    age_raw = sim.people.age.raw
    age_lo = d.age_lo
    age_hi = d.age_hi
    n_adults = 0; n_inf_adults = 0
    @inbounds for u in active
        a = age_raw[u]
        if a >= age_lo && a <= age_hi
            n_adults += 1
            n_inf_adults += inf_raw[u]
        end
    end
    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_adults > 0 ? Float64(n_inf_adults) / Float64(n_adults) : 0.0
    md.results[:incidence][ti]  = n_total > 0.0 ? Float64(n_exp) / n_total : 0.0
    return d
end

function Starsim.finalize!(d::SEIS)
    md = Starsim.module_data(d)
    for (_, _, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end
