"""
HIV disease — Acute → Chronic/Latent → AIDS → Death.
Port of Python `stisim.diseases.hiv.HIV`.

Natural history follows CD4 dynamics:
- Acute: rapid CD4 decline, high transmissibility
- Latent/Chronic: stable CD4 (~500), lower transmissibility
- Falling/AIDS: CD4 declines to 0, high transmissibility, death
- ART: suppresses viral load, restores CD4, reduces transmissibility

NOTE: Python stisim diseases have dt='month' — all STI diseases step at monthly
resolution even when the sim/network dt is weekly. This Julia implementation gates
step_state!, step!, and update_results! to only execute on monthly boundaries.
"""

const _HIV_DISEASE_DT = 1.0 / 12.0  # Monthly (in years)

"""Sample from a lognormal specified by mean and std (matching Python ss.lognorm_ex)."""
function _lognorm_ex_sample(rng, mean_val::Float64, std_val::Float64)
    mean_val <= 0.0 && return mean_val
    cv = std_val / mean_val
    sigma2 = log(1.0 + cv^2)
    mu = log(mean_val) - sigma2 / 2.0
    return exp(mu + sqrt(sigma2) * randn(rng))
end

"""Convert sim step index to monthly disease step index (matching Python's disease ti)."""
function _sim_ti_to_monthly(sim_ti::Int, sim_dt::Float64)
    return floor(Int, Float64(sim_ti) * sim_dt * 12.0)
end

"""Check if the current sim step is a monthly boundary for the HIV disease."""
function _hiv_should_step(sim)
    sim_dt = sim.pars.dt
    ti = sim.loop.ti
    ti <= 1 && return true
    current_time = Float64(ti) * sim_dt
    prev_time = Float64(ti - 1) * sim_dt
    return floor(Int, current_time / _HIV_DISEASE_DT) != floor(Int, prev_time / _HIV_DISEASE_DT)
end

# ============================================================================
# HIV <: AbstractInfection
# ============================================================================

"""
    HIV <: AbstractInfection

HIV disease model with CD4 dynamics and ART.

# Keyword arguments
- `name::Symbol` — default `:hiv`
- `init_prev::Real` — initial prevalence (default 0.05)
- `beta_m2f::Real` — per-act male-to-female beta (default 0.05)
- `rel_beta_f2m::Real` — relative female-to-male (default 0.5)
- `beta_m2m::Real` — per-act MSM beta (default 0.05)
- `eff_condom::Real` — condom efficacy (default 0.9)
- `cd4_start_mean::Real` — mean initial CD4 (default 800)
- `cd4_latent_mean::Real` — mean latent CD4 (default 500)
- `dur_acute::Real` — acute duration in years (default 0.25)
- `dur_latent::Real` — latent duration in years (default 10.0)
- `dur_falling::Real` — falling/AIDS duration in years (default 3.0)
- `rel_trans_acute::Real` — transmission multiplier during acute (default 6.0)
- `rel_trans_falling::Real` — transmission multiplier during falling (default 8.0)
- `include_aids_deaths::Bool` — whether AIDS causes death (default true)
- `art_efficacy::Real` — ART efficacy at reducing transmission (default 0.96)
"""
mutable struct HIV <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # HIV-specific states
    acute::Starsim.StateVector{Bool, Vector{Bool}}
    ti_acute::Starsim.StateVector{Float64, Vector{Float64}}
    latent::Starsim.StateVector{Bool, Vector{Bool}}
    ti_latent::Starsim.StateVector{Float64, Vector{Float64}}
    falling::Starsim.StateVector{Bool, Vector{Bool}}
    ti_falling::Starsim.StateVector{Float64, Vector{Float64}}
    ti_zero::Starsim.StateVector{Float64, Vector{Float64}}
    ti_dead::Starsim.StateVector{Float64, Vector{Float64}}

    # CD4 states
    cd4::Starsim.StateVector{Float64, Vector{Float64}}
    cd4_start::Starsim.StateVector{Float64, Vector{Float64}}
    cd4_latent::Starsim.StateVector{Float64, Vector{Float64}}
    cd4_nadir::Starsim.StateVector{Float64, Vector{Float64}}

    # Treatment states
    on_art::Starsim.StateVector{Bool, Vector{Bool}}
    ti_art::Starsim.StateVector{Float64, Vector{Float64}}
    ti_stop_art::Starsim.StateVector{Float64, Vector{Float64}}
    diagnosed::Starsim.StateVector{Bool, Vector{Bool}}
    ti_diagnosed::Starsim.StateVector{Float64, Vector{Float64}}

    # Parameters
    beta_m2f::Float64
    rel_beta_f2m::Float64
    beta_m2m::Float64
    eff_condom::Float64
    cd4_start_mean::Float64
    cd4_latent_mean::Float64
    dur_acute::Float64
    dur_latent::Float64
    dur_falling::Float64
    rel_trans_acute::Float64
    rel_trans_acute_std::Float64
    rel_trans_falling::Float64
    rel_trans_falling_std::Float64
    include_aids_deaths::Bool
    art_efficacy::Float64
    art_cd4_growth::Float64

    rng::StableRNG
end

function HIV(;
    name::Symbol            = :hiv,
    init_prev::Real         = 0.05,
    beta_m2f::Real          = 0.05,
    rel_beta_f2m::Real      = 0.5,
    beta_m2m::Real          = 0.05,
    eff_condom::Real        = 0.9,
    cd4_start_mean::Real    = 800.0,
    cd4_latent_mean::Real   = 500.0,
    dur_acute::Real         = 0.25,
    dur_latent::Real        = 10.0,
    dur_falling::Real       = 3.0,
    rel_trans_acute::Real   = 6.0,
    rel_trans_acute_std::Real = 0.5,   # Python: ss.normal(loc=6, scale=0.5)
    rel_trans_falling::Real = 8.0,
    rel_trans_falling_std::Real = 0.5, # Python: ss.normal(loc=8, scale=0.5)
    include_aids_deaths::Bool = true,
    art_efficacy::Real      = 0.96,
    art_cd4_growth::Real    = 0.1,
)
    inf = Starsim.InfectionData(name; init_prev=Float64(init_prev), beta=Float64(beta_m2f), label="HIV")

    HIV(
        inf,
        # Natural history states
        Starsim.BoolState(:acute; default=false, label="Acute HIV"),
        Starsim.FloatState(:ti_acute; default=Inf, label="Time acute"),
        Starsim.BoolState(:latent; default=false, label="Latent HIV"),
        Starsim.FloatState(:ti_latent; default=Inf, label="Time latent"),
        Starsim.BoolState(:falling; default=false, label="Falling CD4"),
        Starsim.FloatState(:ti_falling; default=Inf, label="Time falling"),
        Starsim.FloatState(:ti_zero; default=Inf, label="Time CD4 zero"),
        Starsim.FloatState(:ti_dead; default=Inf, label="Time HIV death"),
        # CD4 states
        Starsim.FloatState(:cd4; default=0.0, label="CD4 count"),
        Starsim.FloatState(:cd4_start; default=0.0, label="CD4 start"),
        Starsim.FloatState(:cd4_latent; default=0.0, label="CD4 latent"),
        Starsim.FloatState(:cd4_nadir; default=1000.0, label="CD4 nadir"),
        # Treatment states
        Starsim.BoolState(:on_art; default=false, label="On ART"),
        Starsim.FloatState(:ti_art; default=Inf, label="Time ART start"),
        Starsim.FloatState(:ti_stop_art; default=Inf, label="Time ART stop"),
        Starsim.BoolState(:diagnosed; default=false, label="Diagnosed"),
        Starsim.FloatState(:ti_diagnosed; default=Inf, label="Time diagnosed"),
        # Parameters
        Float64(beta_m2f), Float64(rel_beta_f2m), Float64(beta_m2m),
        Float64(eff_condom), Float64(cd4_start_mean), Float64(cd4_latent_mean),
        Float64(dur_acute), Float64(dur_latent), Float64(dur_falling),
        Float64(rel_trans_acute), Float64(rel_trans_acute_std),
        Float64(rel_trans_falling), Float64(rel_trans_falling_std),
        include_aids_deaths, Float64(art_efficacy), Float64(art_cd4_growth),
        StableRNG(0),
    )
end

Starsim.disease_data(d::HIV) = d.infection.dd
Starsim.module_data(d::HIV)  = d.infection.dd.mod

# ============================================================================
# Lifecycle
# ============================================================================

function Starsim.init_pre!(d::HIV, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    states = [
        d.infection.susceptible, d.infection.infected,
        d.infection.ti_infected, d.infection.rel_sus, d.infection.rel_trans,
        d.acute, d.ti_acute, d.latent, d.ti_latent,
        d.falling, d.ti_falling, d.ti_zero, d.ti_dead,
        d.cd4, d.cd4_start, d.cd4_latent, d.cd4_nadir,
        d.on_art, d.ti_art, d.ti_stop_art,
        d.diagnosed, d.ti_diagnosed,
    ]
    for s in states
        Starsim.add_module_state!(sim.people, s)
    end

    _validate_hiv_beta!(d, sim)

    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:new_infections; npts=npts, label="New HIV infections"),
        Starsim.Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Starsim.Result(:n_infected; npts=npts, label="HIV+", scale=false),
        Starsim.Result(:n_acute; npts=npts, label="Acute", scale=false),
        Starsim.Result(:n_latent; npts=npts, label="Latent", scale=false),
        Starsim.Result(:n_falling; npts=npts, label="Falling", scale=false),
        Starsim.Result(:n_on_art; npts=npts, label="On ART", scale=false),
        Starsim.Result(:n_diagnosed; npts=npts, label="Diagnosed", scale=false),
        Starsim.Result(:prevalence; npts=npts, label="HIV prevalence", scale=false),
        Starsim.Result(:new_deaths; npts=npts, label="HIV deaths"),
        Starsim.Result(:mean_cd4; npts=npts, label="Mean CD4", scale=false),
    )

    md.initialized = true
    return d
end

function _validate_hiv_beta!(d::HIV, sim)
    dd = Starsim.disease_data(d)
    dt = sim.pars.dt
    for (name, _) in sim.networks
        if name == :mf || name == :structuredsexual
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

function Starsim.init_post!(d::HIV, sim)
    people = sim.people
    active = people.auids.values

    # Initialize CD4 for all agents
    for u in active
        cd4_val = d.cd4_start_mean + randn(d.rng) * 50.0
        d.cd4.raw[u] = max(100.0, cd4_val)
        d.cd4_start.raw[u] = d.cd4.raw[u]
        d.cd4_latent.raw[u] = max(100.0, d.cd4_latent_mean + randn(d.rng) * 50.0)
    end

    # Seed initial infections with Bernoulli sampling (matching Python: init_prev.filter())
    n = length(active)
    if d.infection.dd.init_prev <= 0.0
        return d
    end
    # Stochastic Bernoulli seeding — each agent independently has probability init_prev
    infect_list = Int[]
    for u in active
        if rand(d.rng) < d.infection.dd.init_prev
            push!(infect_list, u)
        end
    end
    if isempty(infect_list)
        push!(infect_list, active[1])  # At least one
    end
    infect_uids = Starsim.UIDs(infect_list)
    dt = sim.pars.dt
    # Python: ss.uniform(low=-10*12, high=-5) → -120 to -5 in MONTHLY timesteps
    # We pass ti as sim step, _do_hiv_infection! converts to monthly internally
    # So generate monthly offsets and convert back to sim steps for the ti argument
    ti_monthly_low  = -10 * 12  # -120 months
    ti_monthly_high = -5        # -5 months
    for u in infect_uids.values
        monthly_offset = rand(d.rng, ti_monthly_low:ti_monthly_high)
        # Convert monthly offset to sim step: monthly_offset months = monthly_offset/12 years = monthly_offset/12/dt sim steps
        ti_init = round(Int, Float64(monthly_offset) / 12.0 / dt)
        _do_hiv_infection!(d, sim, u, 0, ti_init)
    end

    # Advance initial cases to correct disease stage based on past infection time
    # ti values are now in monthly disease steps; compare against monthly step 0
    for u in infect_uids.values
        # Acute → Latent (if past ti_latent)
        if d.acute.raw[u] && d.ti_latent.raw[u] <= 0.0
            d.acute.raw[u] = false
            d.latent.raw[u] = true
            d.infection.rel_trans.raw[u] = 1.0
            d.cd4.raw[u] = d.cd4_latent.raw[u]
        end
        # Latent → Falling (if past ti_falling)
        if d.latent.raw[u] && d.ti_falling.raw[u] <= 0.0
            d.latent.raw[u] = false
            d.falling.raw[u] = true
            d.infection.rel_trans.raw[u] = d.rel_trans_falling
        end
        # Falling → Dead (if past ti_zero)
        if d.falling.raw[u] && d.ti_zero.raw[u] <= 0.0
            d.infection.infected.raw[u] = false
            d.falling.raw[u] = false
            sim.people.alive.raw[u] = false
            sim.people.ti_dead.raw[u] = 1.0
        end
    end

    return d
end

# ============================================================================
# State transitions — HIV natural history
# ============================================================================

function Starsim.step_state!(d::HIV, sim)
    # Only step at monthly boundaries (matching Python stisim dt='month')
    _hiv_should_step(sim) || return d

    md = Starsim.module_data(d)
    ti = md.t.ti
    sim_dt = sim.pars.dt
    # Monthly disease step index (matching Python's self.ti for disease)
    monthly_ti = Float64(_sim_ti_to_monthly(ti, sim_dt))
    active = sim.people.auids.values
    new_deaths = 0

    acute_raw   = d.acute.raw
    latent_raw  = d.latent.raw
    falling_raw = d.falling.raw
    infected_raw = d.infection.infected.raw
    on_art_raw  = d.on_art.raw

    # --- Phase 1: State transitions (matching Python order) ---
    # All ti_* values are in monthly disease steps
    @inbounds for u in active
        infected_raw[u] || continue

        # Acute → Latent
        if acute_raw[u] && d.ti_latent.raw[u] <= monthly_ti
            acute_raw[u] = false
            latent_raw[u] = true
        end

        # Latent → Falling
        if latent_raw[u] && !on_art_raw[u] && d.ti_falling.raw[u] <= monthly_ti
            latent_raw[u] = false
            falling_raw[u] = true
        end
    end

    # --- Phase 2: CD4 updates (BEFORE rel_trans, matching Python) ---
    # All durations are in monthly steps → decline_per_step is per month → fires monthly → correct
    @inbounds for u in active
        infected_raw[u] || continue

        if on_art_raw[u]
            _update_cd4_art!(d, u, monthly_ti, sim_dt)
        elseif acute_raw[u]
            _update_cd4_acute!(d, u, monthly_ti)
        elseif latent_raw[u]
            # Python: self.cd4[untreated_latent.uids] = self.cd4_latent[...]
            d.cd4.raw[u] = d.cd4_latent.raw[u]
        elseif falling_raw[u]
            _update_cd4_falling!(d, u, monthly_ti)
        end

        # Track nadir (for anyone not on ART)
        if !on_art_raw[u] && d.cd4.raw[u] < d.cd4_nadir.raw[u]
            d.cd4_nadir.raw[u] = d.cd4.raw[u]
        end
    end

    # --- Phase 3: Update rel_trans (AFTER CD4, matching Python update_transmission) ---
    @inbounds for u in active
        infected_raw[u] || continue

        if acute_raw[u]
            d.infection.rel_trans.raw[u] = d.rel_trans_acute + randn(d.rng) * d.rel_trans_acute_std
        elseif d.cd4.raw[u] < 200.0  # AIDS condition
            d.infection.rel_trans.raw[u] = d.rel_trans_falling + randn(d.rng) * d.rel_trans_falling_std
        else
            d.infection.rel_trans.raw[u] = 1.0
        end
    end

    # --- Phase 4: Deaths (matching Python order) ---
    @inbounds for u in active
        infected_raw[u] || continue

        # CD4-based mortality (matching Python's make_p_hiv_death)
        if !on_art_raw[u] && sim.people.alive.raw[u]
            cd4_val = d.cd4.raw[u]
            annual_rate = if cd4_val >= 1000.0
                0.003
            elseif cd4_val >= 500.0
                0.003
            elseif cd4_val >= 350.0
                0.005
            elseif cd4_val >= 200.0
                0.01
            elseif cd4_val >= 50.0
                0.05
            else
                0.300
            end
            p_death = 1.0 - exp(-annual_rate * _HIV_DISEASE_DT)  # Monthly probability
            if rand(d.rng) < p_death
                Starsim.request_death!(sim.people, Starsim.UIDs([u]), ti)
                new_deaths += 1
                continue
            end
        end

        # AIDS death (CD4 reaches zero) — ti_dead is in monthly steps
        if d.include_aids_deaths && !on_art_raw[u] && d.ti_dead.raw[u] <= monthly_ti
            Starsim.request_death!(sim.people, Starsim.UIDs([u]), ti)
            new_deaths += 1
        end
    end

    # Record deaths
    res = Starsim.module_results(d)
    if haskey(res, :new_deaths) && ti <= length(res[:new_deaths].values)
        res[:new_deaths][ti] = Float64(new_deaths)
    end

    return d
end

"""Update CD4 during acute phase — linear decline from cd4_start to cd4_latent.
ti_acute and ti_latent are in monthly disease steps, so dur is in months.
Fires once per month → decline_per_step is per month → correct."""
function _update_cd4_acute!(d::HIV, u::Int, monthly_ti::Float64)
    acute_start = d.ti_acute.raw[u]
    acute_end = d.ti_latent.raw[u]
    dur = acute_end - acute_start  # in monthly steps
    dur <= 0.0 && return
    decline_per_step = (d.cd4_start.raw[u] - d.cd4_latent.raw[u]) / dur
    d.cd4.raw[u] = max(1.0, d.cd4.raw[u] - decline_per_step)
    return
end

"""Update CD4 during falling phase — linear decline to ~0.
ti_falling and ti_zero are in monthly disease steps, so dur is in months.
Fires once per month → decline_per_step is per month → correct."""
function _update_cd4_falling!(d::HIV, u::Int, monthly_ti::Float64)
    falling_start = d.ti_falling.raw[u]
    falling_end = d.ti_zero.raw[u]
    dur = falling_end - falling_start  # in monthly steps
    dur <= 0.0 && return
    cd4_end = 1.0  # Match Python: cd4_end = 1 to avoid divide by zero
    decline_per_step = (d.cd4_latent.raw[u] - cd4_end) / dur
    d.cd4.raw[u] = max(cd4_end, d.cd4.raw[u] - decline_per_step)
    return
end

"""Update CD4 on ART — logistic growth toward cd4_start."""
function _update_cd4_art!(d::HIV, u::Int, monthly_ti::Float64, sim_dt::Float64)
    cd4_max = 1000.0
    cd4_current = d.cd4.raw[u]
    growth = d.art_cd4_growth
    # Logistic growth: dCD4/dt = growth * CD4 * (1 - CD4/cd4_max)
    cd4_new = cd4_current + growth * cd4_current * (1.0 - cd4_current / cd4_max)
    d.cd4.raw[u] = min(cd4_max, max(1.0, cd4_new))
    return
end

# ============================================================================
# Transmission — direction-specific betas
# ============================================================================

function Starsim.step!(d::HIV, sim)
    # Only step at monthly boundaries (matching Python stisim dt='month')
    _hiv_should_step(sim) || return 0
    return _infect_hiv!(d, sim)
end

function _infect_hiv!(d::HIV, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dd = Starsim.disease_data(d)
    new_infections = 0

    for (net_name, net) in sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue

        new_infections += _infect_hiv_edges!(d, sim, edges, beta_dt, ti, net, net_name)
    end

    return new_infections
end

function _infect_hiv_edges!(d::HIV, sim, edges::Starsim.Edges, beta_dt::Float64, ti::Int, net, net_name::Symbol)
    new_infections = 0
    n_edges = length(edges)
    bidir = Starsim.network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    eb = edges.beta
    ea = edges.acts

    # SNAPSHOT state arrays before edge loop (matching Python: rel_trans/rel_sus
    # are computed once before the loop; newly infected agents do NOT become
    # infectious within the same step)
    n_agents = length(d.infection.infected.raw)
    infected_snap    = copy(d.infection.infected.raw)
    susceptible_snap = copy(d.infection.susceptible.raw)
    rel_trans_snap   = copy(d.infection.rel_trans.raw)
    rel_sus_snap     = copy(d.infection.rel_sus.raw)

    # Python also masks: rel_trans = infectious * rel_trans, rel_sus = susceptible * rel_sus
    @inbounds for u in 1:n_agents
        if !infected_snap[u]
            rel_trans_snap[u] = 0.0
        end
        if !susceptible_snap[u]
            rel_sus_snap[u] = 0.0
        end
    end

    on_art_raw      = d.on_art.raw
    female_raw      = sim.people.female.raw
    rng = d.rng

    beta_m2f = d.beta_m2f
    beta_f2m = d.beta_m2f * d.rel_beta_f2m
    beta_m2m = d.beta_m2m
    dt = sim.pars.dt
    art_eff = d.art_efficacy

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        edge_beta = eb[i]
        acts = ea[i]

        if infected_snap[src] && susceptible_snap[trg]
            beta_act = _get_directional_beta(female_raw[src], female_raw[trg], beta_m2f, beta_f2m, beta_m2m)
            trans_mult = on_art_raw[src] ? (1.0 - art_eff) : 1.0
            net_beta_val = (1.0 - (1.0 - beta_act)^acts) * edge_beta
            p = clamp(rel_trans_snap[src] * trans_mult * rel_sus_snap[trg] * net_beta_val, 0.0, 1.0)
            if rand(rng) < p
                _do_hiv_infection!(d, sim, trg, src, ti)
                susceptible_snap[trg] = false  # Prevent re-infection from other edges (matches Python dedup)
                new_infections += 1
            end
        end

        if bidir && infected_snap[trg] && susceptible_snap[src]
            beta_act = _get_directional_beta(female_raw[trg], female_raw[src], beta_m2f, beta_f2m, beta_m2m)
            trans_mult = on_art_raw[trg] ? (1.0 - art_eff) : 1.0
            net_beta_val = (1.0 - (1.0 - beta_act)^acts) * edge_beta
            p = clamp(rel_trans_snap[trg] * trans_mult * rel_sus_snap[src] * net_beta_val, 0.0, 1.0)
            if rand(rng) < p
                _do_hiv_infection!(d, sim, src, trg, ti)
                susceptible_snap[src] = false  # Prevent re-infection from other edges (matches Python dedup)
                new_infections += 1
            end
        end
    end
    return new_infections
end

"""Infect a single agent with HIV and set full prognosis.

All disease timing (ti_acute, ti_latent, ti_falling, ti_zero, ti_dead) is stored
in MONTHLY disease timestep indices, matching Python stisim where dt='month'.
Durations are sampled in months:
  dur_acute  ~ lognorm(mean=3mo, std=1mo),  truncated to int months
  dur_latent ~ lognorm(mean=120mo, std=36mo), truncated to int months
  dur_falling~ lognorm(mean=36mo, std=12mo), truncated to int months
"""
function _do_hiv_infection!(d::HIV, sim, target::Int, source::Int, ti::Int)
    sim_dt = sim.pars.dt
    # Convert sim step ti to monthly disease step
    monthly_ti = _sim_ti_to_monthly(ti, sim_dt)

    d.infection.susceptible.raw[target] = false
    d.infection.infected.raw[target] = true
    d.infection.ti_infected.raw[target] = Float64(ti)  # sim-step for infection log

    # Acute phase
    d.acute.raw[target] = true
    d.ti_acute.raw[target] = Float64(monthly_ti)
    d.infection.rel_trans.raw[target] = d.rel_trans_acute + randn(d.rng) * d.rel_trans_acute_std

    # CD4 at infection start
    cd4_init = d.cd4_start_mean + randn(d.rng) * 50.0
    d.cd4_start.raw[target] = max(100.0, cd4_init)
    d.cd4.raw[target] = d.cd4_start.raw[target]
    cd4_lat = max(100.0, d.cd4_latent_mean + randn(d.rng) * 50.0)
    d.cd4_latent.raw[target] = cd4_lat

    # Duration of acute in MONTHS — Python: ss.lognorm_ex(months(3), months(1))
    # d.dur_acute is in years (0.25), convert mean to months for sampling
    dur_acute_months = _lognorm_ex_sample(d.rng, d.dur_acute * 12.0, d.dur_acute * 12.0 / 3.0)
    dur_acute_int = max(1, floor(Int, dur_acute_months))
    d.ti_latent.raw[target] = Float64(monthly_ti + dur_acute_int)

    # Duration of latent in MONTHS — Python: ss.lognorm_ex(years(10), years(3))
    dur_latent_months = _lognorm_ex_sample(d.rng, d.dur_latent * 12.0, d.dur_latent * 12.0 * 0.3)
    dur_latent_int = max(1, floor(Int, dur_latent_months))
    d.ti_falling.raw[target] = d.ti_latent.raw[target] + Float64(dur_latent_int)

    # Duration of falling in MONTHS — Python: ss.lognorm_ex(years(3), years(1))
    dur_falling_months = _lognorm_ex_sample(d.rng, d.dur_falling * 12.0, d.dur_falling * 12.0 / 3.0)
    dur_falling_int = max(1, floor(Int, dur_falling_months))
    d.ti_zero.raw[target] = d.ti_falling.raw[target] + Float64(dur_falling_int)
    d.ti_dead.raw[target] = d.ti_zero.raw[target]

    push!(d.infection.infection_sources, (target, source, ti))
    return
end

function Starsim.set_prognoses!(d::HIV, sim, uids::Starsim.UIDs)
    ti = Starsim.module_data(d).t.ti
    for u in uids.values
        _do_hiv_infection!(d, sim, u, 0, ti)
    end
    return
end

# ============================================================================
# Death handling and results
# ============================================================================

function Starsim.step_die!(d::HIV, death_uids::Starsim.UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids]    = false
    d.acute[death_uids]                 = false
    d.latent[death_uids]                = false
    d.falling[death_uids]               = false
    d.on_art[death_uids]                = false
    d.diagnosed[death_uids]             = false
    return d
end

function Starsim.update_results!(d::HIV, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw  = d.infection.susceptible.raw
    inf_raw  = d.infection.infected.raw
    acute_raw = d.acute.raw
    latent_raw = d.latent.raw
    falling_raw = d.falling.raw
    art_raw = d.on_art.raw
    diag_raw = d.diagnosed.raw
    cd4_raw = d.cd4.raw

    n_sus = 0; n_inf = 0; n_acute = 0; n_latent = 0; n_falling = 0
    n_art = 0; n_diag = 0; cd4_sum = 0.0; n_cd4 = 0
    @inbounds for u in active
        n_sus     += sus_raw[u]
        n_inf     += inf_raw[u]
        n_acute   += acute_raw[u]
        n_latent  += latent_raw[u]
        n_falling += falling_raw[u]
        n_art     += art_raw[u]
        n_diag    += diag_raw[u]
        if inf_raw[u]
            cd4_sum += cd4_raw[u]
            n_cd4 += 1
        end
    end

    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_infected][ti]    = Float64(n_inf)
    md.results[:n_acute][ti]       = Float64(n_acute)
    md.results[:n_latent][ti]      = Float64(n_latent)
    md.results[:n_falling][ti]     = Float64(n_falling)
    md.results[:n_on_art][ti]      = Float64(n_art)
    md.results[:n_diagnosed][ti]   = Float64(n_diag)

    n_total = Float64(length(active))
    # HIV overrides BaseSTI prevalence back to total population:
    # "Recalculate prevalence so it's for the whole population"
    md.results[:prevalence][ti] = n_total > 0.0 ? Float64(n_inf) / n_total : 0.0
    md.results[:mean_cd4][ti]   = n_cd4 > 0 ? cd4_sum / n_cd4 : 0.0
    return d
end

function Starsim.finalize!(d::HIV)
    md = Starsim.module_data(d)
    for (_, _, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end
