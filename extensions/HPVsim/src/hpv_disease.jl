"""
HPV disease model — individual genotype-specific disease instances.

Each HPVGenotype instance represents a specific HPV genotype that behaves
as an independent disease in the simulation, with the progression pathway:

  Susceptible → Infected → CIN1 → CIN2 → CIN3 → Cancer → (Death)

Cross-genotype interactions (immunity) are handled by HPVImmunityConnector.

Two progression modes are supported:
- **Rate-based** (`use_duration_model=false`, default): per-timestep transition
  probabilities from genotype parameters.
- **Duration-based** (`use_duration_model=true`): at infection, pre-CIN and CIN
  durations are sampled from LogNormal distributions. CIN development and cancer
  probabilities are computed from logistic functions (matching Python hpvsim).

Key differences from the Python hpvsim:
- Python uses 2D arrays [genotype, person]; here each genotype is a separate disease
- Cancer progression only affects females (males can transmit but do not progress)
- Cancer mortality is an absorbing sink with configurable annual mortality rate
"""

# ============================================================================
# HPVGenotype <: AbstractInfection
# ============================================================================

"""
    HPVGenotype <: AbstractInfection

Individual HPV genotype as a separate disease instance with CIN progression.

# Keyword arguments
- `genotype::Symbol` — genotype name (e.g., :hpv16)
- `init_prev::Real` — initial infection prevalence (default from genotype params)
- `beta::Real` — base transmission rate (adjusted by rel_beta)
- `params::GenotypeParams` — natural history parameters
- `use_duration_model::Bool` — use duration-based progression (default false)
"""
mutable struct HPVGenotype <: Starsim.AbstractInfection
    infection::Starsim.InfectionData

    # Genotype identity
    genotype::Symbol
    params::GenotypeParams

    # Progression mode
    use_duration_model::Bool

    # CIN progression states (females only progress; tracked for all for indexing)
    cin1::Starsim.StateVector{Bool, Vector{Bool}}
    cin2::Starsim.StateVector{Bool, Vector{Bool}}
    cin3::Starsim.StateVector{Bool, Vector{Bool}}
    cancerous::Starsim.StateVector{Bool, Vector{Bool}}

    # Timing states
    ti_cin1::Starsim.StateVector{Float64, Vector{Float64}}
    ti_cin2::Starsim.StateVector{Float64, Vector{Float64}}
    ti_cin3::Starsim.StateVector{Float64, Vector{Float64}}
    ti_cancer::Starsim.StateVector{Float64, Vector{Float64}}
    ti_cleared::Starsim.StateVector{Float64, Vector{Float64}}

    # Duration-based model states (sampled at infection time)
    dur_precin::Starsim.StateVector{Float64, Vector{Float64}}
    dur_cin::Starsim.StateVector{Float64, Vector{Float64}}
    rel_sev::Starsim.StateVector{Float64, Vector{Float64}}

    # Pre-determined outcomes (duration model only)
    will_cin::Starsim.StateVector{Bool, Vector{Bool}}
    will_cancer::Starsim.StateVector{Bool, Vector{Bool}}
    date_cin::Starsim.StateVector{Float64, Vector{Float64}}
    date_cancer::Starsim.StateVector{Float64, Vector{Float64}}
    date_clearance::Starsim.StateVector{Float64, Vector{Float64}}

    # Tracking states
    cleared::Starsim.StateVector{Bool, Vector{Bool}}
    n_infections::Starsim.StateVector{Float64, Vector{Float64}}

    rng::StableRNG
end

function HPVGenotype(;
    genotype::Symbol = :hpv16,
    init_prev::Union{Real, Nothing} = nothing,
    beta::Real = DEFAULT_BETA,
    params::Union{GenotypeParams, Nothing} = nothing,
    name::Union{Symbol, Nothing} = nothing,
    use_duration_model::Bool = true,
)
    gp = params === nothing ? get_genotype_params(genotype) : params
    nm = name === nothing ? gp.name : name
    prev = init_prev === nothing ? 0.01 : Float64(init_prev)

    # Adjust beta by genotype's relative transmissibility
    adjusted_beta = Float64(beta) * gp.rel_beta
    inf = Starsim.InfectionData(nm; init_prev=prev, beta=adjusted_beta, label="HPV $(gp.name)")

    HPVGenotype(
        inf,
        gp.name,
        gp,
        use_duration_model,
        # CIN states
        Starsim.BoolState(:cin1; default=false, label="CIN1"),
        Starsim.BoolState(:cin2; default=false, label="CIN2"),
        Starsim.BoolState(:cin3; default=false, label="CIN3"),
        Starsim.BoolState(:cancerous; default=false, label="Cancerous"),
        # Timing states
        Starsim.FloatState(:ti_cin1; default=Inf, label="Time CIN1"),
        Starsim.FloatState(:ti_cin2; default=Inf, label="Time CIN2"),
        Starsim.FloatState(:ti_cin3; default=Inf, label="Time CIN3"),
        Starsim.FloatState(:ti_cancer; default=Inf, label="Time cancer"),
        Starsim.FloatState(:ti_cleared; default=Inf, label="Time cleared"),
        # Duration-based model states
        Starsim.FloatState(:dur_precin; default=0.0, label="Pre-CIN duration"),
        Starsim.FloatState(:dur_cin; default=0.0, label="CIN duration"),
        Starsim.FloatState(:rel_sev; default=1.0, label="Relative severity"),
        # Pre-determined outcomes
        Starsim.BoolState(:will_cin; default=false, label="Will develop CIN"),
        Starsim.BoolState(:will_cancer; default=false, label="Will develop cancer"),
        Starsim.FloatState(:date_cin; default=Inf, label="Scheduled CIN date"),
        Starsim.FloatState(:date_cancer; default=Inf, label="Scheduled cancer date"),
        Starsim.FloatState(:date_clearance; default=Inf, label="Scheduled clearance date"),
        # Tracking
        Starsim.BoolState(:cleared; default=false, label="Cleared"),
        Starsim.FloatState(:n_infections; default=0.0, label="Infection count"),
        StableRNG(0),
    )
end

Starsim.disease_data(d::HPVGenotype) = d.infection.dd
Starsim.module_data(d::HPVGenotype)  = d.infection.dd.mod

# ============================================================================
# Lifecycle
# ============================================================================

"""Reset rel_sus to 1.0 each timestep so immunity/vaccination can be reapplied cleanly."""
function Starsim.start_step!(d::HPVGenotype, sim)
    # Call base start_step! for jump distributions
    md = Starsim.module_data(d)
    for dist in md.dists
        Starsim.jump_dt!(dist, md.t.ti)
    end
    # Reset rel_sus so immunity and vaccination are recalculated fresh
    active = sim.people.auids.values
    rel_sus_raw = d.infection.rel_sus.raw
    @inbounds for u in active
        rel_sus_raw[u] = 1.0
    end
    return d
end

function Starsim.init_pre!(d::HPVGenotype, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    # Register all states with People
    all_states = [
        d.infection.susceptible, d.infection.infected,
        d.infection.ti_infected, d.infection.rel_sus,
        d.infection.rel_trans,
        d.cin1, d.cin2, d.cin3, d.cancerous,
        d.ti_cin1, d.ti_cin2, d.ti_cin3, d.ti_cancer, d.ti_cleared,
        d.dur_precin, d.dur_cin, d.rel_sev,
        d.will_cin, d.will_cancer, d.date_cin, d.date_cancer, d.date_clearance,
        d.cleared, d.n_infections,
    ]
    for s in all_states
        Starsim.add_module_state!(sim.people, s)
    end

    # Compute beta per dt
    Starsim.validate_beta!(d, sim)

    # Results
    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:new_infections; npts=npts, label="New infections"),
        Starsim.Result(:n_susceptible; npts=npts, label="Susceptible", scale=false),
        Starsim.Result(:n_infected; npts=npts, label="Infected", scale=false),
        Starsim.Result(:n_cin1; npts=npts, label="CIN1", scale=false),
        Starsim.Result(:n_cin2; npts=npts, label="CIN2", scale=false),
        Starsim.Result(:n_cin3; npts=npts, label="CIN3", scale=false),
        Starsim.Result(:n_cancerous; npts=npts, label="Cancerous", scale=false),
        Starsim.Result(:n_cleared; npts=npts, label="Cleared", scale=false),
        Starsim.Result(:n_cancer_deaths; npts=npts, label="Cancer deaths"),
        Starsim.Result(:prevalence; npts=npts, label="Prevalence", scale=false),
        Starsim.Result(:cin_prevalence; npts=npts, label="CIN prevalence", scale=false),
    )

    md.initialized = true
    return d
end

function Starsim.validate_beta!(d::HPVGenotype, sim)
    dd = Starsim.disease_data(d)
    # Store per-act beta (= beta * rel_beta, already computed in constructor).
    # Direction multiplier, condom effects, and acts exponentiation are applied
    # per-edge in the transmission loop, matching Python hpvsim's FOI formula.
    if dd.beta isa Real
        for (name, _) in sim.networks
            dd.beta_per_dt[name] = Float64(dd.beta)
        end
    elseif dd.beta isa Dict
        for (name, b) in dd.beta
            dd.beta_per_dt[Symbol(name)] = Float64(b)
        end
    end
    return d
end

function Starsim.init_post!(d::HPVGenotype, sim)
    people = sim.people
    active = people.auids.values
    rng = d.rng

    # Initialize individual severity for all agents (drawn once)
    _init_rel_sev!(d, active)

    # --- Initial infection seeding (coordinated across genotypes) ---
    # Python hpvsim draws TOTAL HPV status once, then assigns ONE genotype
    # per agent. To match this, only the FIRST genotype to init seeds ALL
    # genotypes. Subsequent genotypes skip seeding.

    # Check if init_prev is zero for ALL genotypes — skip seeding entirely
    all_genotypes_for_check = HPVGenotype[d]
    for (_, dis) in sim.diseases
        dis isa HPVGenotype && dis !== d && push!(all_genotypes_for_check, dis)
    end
    all_zero_prev = all(gd -> gd.infection.dd.init_prev <= 0.0, all_genotypes_for_check)
    if all_zero_prev
        return d
    end

    other_hpv = [dis for (_, dis) in sim.diseases if dis isa HPVGenotype && dis !== d]
    any_already_seeded = any(dis -> any(i -> dis.infection.infected.raw[i], active), other_hpv)

    if any_already_seeded
        # Another genotype already seeded; skip (our agents were assigned above)
        return d
    end

    # This is the first genotype — seed ALL genotypes
    all_genotypes = HPVGenotype[d]
    for (_, dis) in sim.diseases
        dis isa HPVGenotype && dis !== d && push!(all_genotypes, dis)
    end
    n_genotypes = length(all_genotypes)

    # Draw total HPV status using age-structured prevalence (scale=1.0)
    age_brackets = DEFAULT_INIT_PREV_AGE_BRACKETS
    prev_m = DEFAULT_INIT_PREV_MALE
    prev_f = DEFAULT_INIT_PREV_FEMALE

    hpv_positive = Int[]
    @inbounds for u in active
        age = people.age.raw[u]
        is_female = people.female.raw[u]

        bracket_prev = 0.0
        for bi in eachindex(age_brackets)
            if age < age_brackets[bi]
                bracket_prev = is_female ? prev_f[bi] : prev_m[bi]
                break
            end
        end

        if bracket_prev > 0.0 && rand(rng) < bracket_prev
            push!(hpv_positive, u)
        end
    end

    # Assign ONE genotype per HPV+ agent (uniformly at random)
    genotype_assignments = [Int[] for _ in 1:n_genotypes]
    for u in hpv_positive
        gi = rand(rng, 1:n_genotypes)
        push!(genotype_assignments[gi], u)
    end

    # Infect agents in each genotype
    dt = sim.pars.dt
    for (gi, gd) in enumerate(all_genotypes)
        uids_for_genotype = genotype_assignments[gi]
        if !isempty(uids_for_genotype)
            for u in uids_for_genotype
                gd.infection.susceptible.raw[u] = false
                gd.infection.infected.raw[u]    = true
                gd.infection.ti_infected.raw[u] = 1.0
                gd.n_infections.raw[u] = 1.0
            end

            # Set prognoses for duration model (sex-specific)
            if gd.use_duration_model
                for u in uids_for_genotype
                    if people.female.raw[u]
                        _set_prognosis!(gd, u, 1, dt, sim)
                    else
                        _set_male_prognosis!(gd, u, 1, dt)
                    end
                end
            end
        end
    end

    return d
end

"""Initialize individual relative severity (heterogeneity factor)."""
function _init_rel_sev!(d::HPVGenotype, active::Vector{Int})
    @inbounds for u in active
        # Draw from truncated normal centered at 1.0 with small variance
        d.rel_sev.raw[u] = max(0.2, 1.0 + 0.3 * randn(d.rng))
    end
    return
end

# ============================================================================
# Prognosis at infection (duration-based model)
# ============================================================================

"""
Set disease trajectory for an agent at infection time (duration model).

Samples pre-CIN and CIN durations, computes CIN/cancer probabilities
using logf2 functions, and schedules future transition dates.

For reinfections, severity immunity (`sev_imm`) shortens `dur_precin`:
`dur_precin *= (1 - sev_imm)`, matching Python hpvsim.
"""
function _set_prognosis!(d::HPVGenotype, uid::Int, ti::Int, dt::Float64, sim)
    gp = d.params
    rng = d.rng

    # Sample pre-CIN duration (returned in years by sample_lognormal_duration)
    dp = sample_lognormal_duration(rng, gp.dur_precin_par1, gp.dur_precin_par2)

    # Apply severity immunity reduction (matches Python: dur_precin *= (1 - sev_imm))
    sev_imm = get_sev_imm(sim, d.genotype, uid)
    if sev_imm > 0.0
        dp *= (1.0 - sev_imm)
    end
    d.dur_precin.raw[uid] = dp

    # Compute CIN probability from duration and individual severity
    rsev = d.rel_sev.raw[uid]
    cin_prob = compute_cin_prob(dp, rsev, gp.cin_fn_k, gp.cin_fn_x_infl; ttc=gp.cin_fn_ttc)

    if rand(rng) < cin_prob
        d.will_cin.raw[uid] = true
        # Schedule CIN date (current ti + randround(dur_precin/dt), matching Python)
        d.date_cin.raw[uid] = Float64(ti) + _randround(rng, dp / dt)

        # Sample CIN duration (returned in years)
        dc = sample_lognormal_duration(rng, gp.dur_cin_par1, gp.dur_cin_par2)
        d.dur_cin.raw[uid] = dc

        # Compute cancer probability from CIN duration
        cancer_prob = compute_cancer_prob(dc, rsev, gp.cin_fn_k, gp.cin_fn_x_infl,
                                          gp.cancer_fn_transform_prob; ttc=gp.cin_fn_ttc)
        if rand(rng) < cancer_prob
            d.will_cancer.raw[uid] = true
            d.date_cancer.raw[uid] = d.date_cin.raw[uid] + _randround(rng, dc / dt)
        end

        # Schedule clearance at date_exposed + randround((dur_precin + dur_cin) / dt)
        # matching Python: date_clearance = date_exposed + sc.randround(time_to_clear / dt)
        total_dur = dp + dc
        d.date_clearance.raw[uid] = Float64(ti) + _randround(rng, total_dur / dt)
    else
        # No CIN: clearance at dur_precin (matching Python: t + randround(dur_precin / dt))
        d.will_cin.raw[uid] = false
        d.date_clearance.raw[uid] = Float64(ti) + _randround(rng, dp / dt)
    end
    return
end

# ============================================================================
# Step — state transitions (CIN progression and clearance)
# ============================================================================

function Starsim.step_state!(d::HPVGenotype, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dt = sim.pars.dt  # Timestep in years
    active = sim.people.auids.values

    if d.use_duration_model
        _step_duration_model!(d, sim, active, ti, dt)
    else
        _step_rate_model!(d, sim, active, ti, dt)
    end

    # Cancer mortality (both models)
    _step_cancer_mortality!(d, sim, active, ti, dt)

    return d
end

"""Rate-based progression (original model)."""
function _step_rate_model!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    _step_cin3_rate!(d, sim, active, ti, dt)
    _step_cin2_rate!(d, sim, active, ti, dt)
    _step_cin1_rate!(d, sim, active, ti, dt)
    _step_infected_rate!(d, sim, active, ti, dt)
    return
end

"""Duration-based progression (Python hpvsim reference model).

Matches Python's date-based clearance using pre-computed date_clearance
(set at infection time with stochastic rounding, matching sc.randround).
"""
function _step_duration_model!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    ti_f = Float64(ti)
    gp = d.params

    @inbounds for u in active
        d.infection.infected.raw[u] || continue
        d.cancerous.raw[u] && continue

        # === Non-CIN and male agents: clear at pre-computed date_clearance ===
        if !d.will_cin.raw[u] || (d.will_cin.raw[u] && !sim.people.female.raw[u])
            if ti_f >= d.date_clearance.raw[u]
                _clear_infection!(d, u, ti_f)
                _notify_clearance(d, sim, u)
                continue
            end
        end

        # === Female CIN agents ===
        if d.will_cin.raw[u] && sim.people.female.raw[u]
            # Mark CIN onset (for tracking)
            if !d.cin1.raw[u] && !d.cin2.raw[u] && !d.cin3.raw[u]
                if ti_f >= d.date_cin.raw[u]
                    d.cin1.raw[u] = true
                    d.ti_cin1.raw[u] = ti_f
                end
            end

            # CIN staging (for internal tracking; doesn't affect clearance timing)
            if d.cin1.raw[u] && !d.cin2.raw[u] && !d.cin3.raw[u]
                time_in_cin = (ti_f - d.ti_cin1.raw[u]) * dt
                severity = logf2(time_in_cin * d.rel_sev.raw[u], gp.cin_fn_k, gp.cin_fn_x_infl; ttc=gp.cin_fn_ttc)
                if severity > 1.0/3.0
                    d.cin1.raw[u] = false
                    d.cin2.raw[u] = true
                    d.ti_cin2.raw[u] = ti_f
                end
            end
            if d.cin2.raw[u] && !d.cin3.raw[u]
                time_in_cin = (ti_f - d.ti_cin1.raw[u]) * dt
                severity = logf2(time_in_cin * d.rel_sev.raw[u], gp.cin_fn_k, gp.cin_fn_x_infl; ttc=gp.cin_fn_ttc)
                if severity > 2.0/3.0
                    d.cin2.raw[u] = false
                    d.cin3.raw[u] = true
                    d.ti_cin3.raw[u] = ti_f
                end
            end

            # Cancer progression (date-based)
            if d.will_cancer.raw[u] && ti_f >= d.date_cancer.raw[u]
                d.cin1.raw[u] = false
                d.cin2.raw[u] = false
                d.cin3.raw[u] = false
                d.cancerous.raw[u] = true
                d.ti_cancer.raw[u] = ti_f
                continue
            end

            # Date-based clearance for CIN agents WITHOUT cancer
            # Uses pre-computed date_clearance (set at prognosis time with randround)
            if !d.will_cancer.raw[u]
                if ti_f >= d.date_clearance.raw[u]
                    d.cin1.raw[u] = false
                    d.cin2.raw[u] = false
                    d.cin3.raw[u] = false
                    _clear_infection!(d, u, ti_f)
                    _notify_clearance(d, sim, u)
                    continue
                end
            end
        end
    end
    return
end

# ============================================================================
# Rate-based progression functions (original model)
# ============================================================================

"""Progress/clear infected agents (no CIN yet)."""
function _step_infected_rate!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    gp = d.params
    infected_raw   = d.infection.infected.raw
    cin1_raw       = d.cin1.raw
    ti_cin1_raw    = d.ti_cin1.raw
    female_raw     = sim.people.female.raw
    ti_f = Float64(ti)

    p_clear = 1.0 - (1.0 - gp.clearance_rate_inf)^dt
    p_prog  = 1.0 - (1.0 - gp.prog_rate_cin1)^dt

    @inbounds for u in active
        infected_raw[u] || continue
        cin1_raw[u] && continue
        d.cin2.raw[u] && continue
        d.cin3.raw[u] && continue
        d.cancerous.raw[u] && continue

        r = rand(d.rng)
        if r < p_clear
            _clear_infection!(d, u, ti_f)
            _notify_clearance(d, sim, u)
        elseif female_raw[u] && r < p_clear + p_prog
            cin1_raw[u] = true
            ti_cin1_raw[u] = ti_f
        end
    end
    return
end

"""Progress/clear CIN1 agents."""
function _step_cin1_rate!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    gp = d.params
    cin1_raw    = d.cin1.raw
    cin2_raw    = d.cin2.raw
    ti_cin2_raw = d.ti_cin2.raw
    ti_f = Float64(ti)

    p_clear = 1.0 - (1.0 - gp.clearance_rate_cin1)^dt
    p_prog  = 1.0 - (1.0 - gp.prog_rate_cin2)^dt

    @inbounds for u in active
        cin1_raw[u] || continue
        cin2_raw[u] && continue

        r = rand(d.rng)
        if r < p_clear
            cin1_raw[u] = false
            _clear_infection!(d, u, ti_f)
            _notify_clearance(d, sim, u)
        elseif r < p_clear + p_prog
            cin1_raw[u] = false
            cin2_raw[u] = true
            ti_cin2_raw[u] = ti_f
        end
    end
    return
end

"""Progress/clear CIN2 agents."""
function _step_cin2_rate!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    gp = d.params
    cin2_raw    = d.cin2.raw
    cin3_raw    = d.cin3.raw
    ti_cin3_raw = d.ti_cin3.raw
    ti_f = Float64(ti)

    p_clear = 1.0 - (1.0 - gp.clearance_rate_cin2)^dt
    p_prog  = 1.0 - (1.0 - gp.prog_rate_cin3)^dt

    @inbounds for u in active
        cin2_raw[u] || continue
        cin3_raw[u] && continue

        r = rand(d.rng)
        if r < p_clear
            cin2_raw[u] = false
            _clear_infection!(d, u, ti_f)
            _notify_clearance(d, sim, u)
        elseif r < p_clear + p_prog
            cin2_raw[u] = false
            cin3_raw[u] = true
            ti_cin3_raw[u] = ti_f
        end
    end
    return
end

"""Progress/clear CIN3 agents."""
function _step_cin3_rate!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    gp = d.params
    cin3_raw      = d.cin3.raw
    cancer_raw    = d.cancerous.raw
    ti_cancer_raw = d.ti_cancer.raw
    ti_f = Float64(ti)

    p_clear = 1.0 - (1.0 - gp.clearance_rate_cin3)^dt
    p_prog  = 1.0 - (1.0 - gp.cancer_rate)^dt

    @inbounds for u in active
        cin3_raw[u] || continue
        cancer_raw[u] && continue

        r = rand(d.rng)
        if r < p_clear
            cin3_raw[u] = false
            _clear_infection!(d, u, ti_f)
            _notify_clearance(d, sim, u)
        elseif r < p_clear + p_prog
            cin3_raw[u] = false
            cancer_raw[u] = true
            ti_cancer_raw[u] = ti_f
        end
    end
    return
end

# ============================================================================
# Cancer mortality
# ============================================================================

"""
Apply cancer mortality. Agents with cancer face an annual mortality rate.
Deaths are logged but actual removal is handled by the Starsim demographics module.
"""
function _step_cancer_mortality!(d::HPVGenotype, sim, active::Vector{Int}, ti::Int, dt::Float64)
    gp = d.params
    gp.cancer_mortality_rate <= 0.0 && return

    cancer_raw = d.cancerous.raw
    p_death = 1.0 - (1.0 - gp.cancer_mortality_rate)^dt
    md = Starsim.module_data(d)
    n_deaths = 0

    @inbounds for u in active
        cancer_raw[u] || continue
        if rand(d.rng) < p_death
            # Schedule death via the people system if demographics supports it
            if sim.people.ti_dead.raw[u] == Inf || isnan(sim.people.ti_dead.raw[u])
                sim.people.ti_dead.raw[u] = Float64(ti)
            end
            n_deaths += 1
        end
    end

    if ti <= length(md.results[:n_cancer_deaths].values)
        md.results[:n_cancer_deaths][ti] = Float64(n_deaths)
    end
    return
end

# ============================================================================
# Infection helper — clear and notify
# ============================================================================

"""Clear infection for an agent — reset all CIN states."""
function _clear_infection!(d::HPVGenotype, u::Int, ti_f::Float64)
    d.infection.infected.raw[u]    = false
    d.infection.susceptible.raw[u] = true
    d.cin1.raw[u] = false
    d.cin2.raw[u] = false
    d.cin3.raw[u] = false
    d.cleared.raw[u] = true
    d.ti_cleared.raw[u] = ti_f
    # Reset duration model prognosis
    d.will_cin.raw[u] = false
    d.will_cancer.raw[u] = false
    d.date_cin.raw[u] = Inf
    d.date_cancer.raw[u] = Inf
    d.date_clearance.raw[u] = Inf
    return
end

"""Notify the HPVImmunityConnector about a clearance event.

Only FEMALES get immunity on clearance, matching Python hpvsim where
`update_peak_immunity` is only called for `f_cleared_inds`.
"""
function _notify_clearance(d::HPVGenotype, sim, uid::Int)
    # Males don't get immunity on clearance (matching Python)
    sim.people.female.raw[uid] || return

    for (_, conn) in sim.connectors
        if conn isa HPVImmunityConnector
            record_clearance!(conn, d, uid)
            return
        end
    end
    return
end

# ============================================================================
# Transmission (reuses Starsim SIR infect! pattern)
# ============================================================================

function Starsim.step!(d::HPVGenotype, sim)
    return _infect_hpv!(d, sim)
end

function _infect_hpv!(d::HPVGenotype, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dd = Starsim.disease_data(d)
    dt = sim.pars.dt
    new_infections = 0

    # Snapshot infected status at start of timestep (matching Python:
    # inf = people.infectious.copy()). Agents infected during this step
    # cannot transmit until the NEXT timestep.
    n_raw = length(d.infection.infected.raw)
    infected_snapshot = copy(d.infection.infected.raw)

    for (net_name, net) in sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        per_act_beta = get(dd.beta_per_dt, net_name, 0.0)
        per_act_beta == 0.0 && continue

        new_infections += _infect_hpv_standard!(d, sim, edges, per_act_beta, dt, ti, net, infected_snapshot)
    end

    return new_infections
end

"""HPV transmission matching Python hpvsim FOI formula.

Edge beta stores age-scaled acts_per_year. The per-act transmission probability
is computed per-edge incorporating direction multiplier and condom effects
BEFORE exponentiation, matching:
  p_per_act = beta * rel_beta * trans_mult * (1 - condom_eff)
  p_per_dt  = 1 - (1 - p_per_act)^(acts * dt)
  p_final   = p_per_dt * rel_trans * rel_sus
"""
function _infect_hpv_standard!(d::HPVGenotype, sim, edges::Starsim.Edges, per_act_beta::Float64, dt::Float64, ti::Int, net, infected_snapshot::Vector{Bool})
    new_infections = 0
    n_edges = length(edges)
    bidir = Starsim.network_data(net).bidirectional

    p1 = edges.p1
    p2 = edges.p2
    acts_per_year = edges.beta  # Edge beta stores age-scaled acts/year

    susceptible_raw = d.infection.susceptible.raw
    rel_trans_raw   = d.infection.rel_trans.raw
    rel_sus_raw     = d.infection.rel_sus.raw
    female_raw      = sim.people.female.raw
    rng = d.rng

    # Get per-edge condom factors from network
    edge_condom_factors = if net isa HPVSexualNet && !isempty(net._edge_condom_factors)
        net._edge_condom_factors
    else
        nothing
    end

    @inbounds for i in 1:n_edges
        src = p1[i]
        trg = p2[i]
        acts = acts_per_year[i] * dt  # Acts this timestep
        cf = edge_condom_factors !== nothing && i <= length(edge_condom_factors) ? edge_condom_factors[i] : 1.0

        # src → trg (use snapshot for infectiousness check)
        if infected_snapshot[src] && susceptible_raw[trg]
            # transm2f=3.69: male-to-female has higher per-act probability
            # transf2m=1.0: female-to-male is baseline
            # When source is female (F→M): use 1.0; when source is male (M→F): use 3.69
            trans_mult = female_raw[src] ? 1.0 : M2F_TRANS_RATIO
            p_per_act = min(per_act_beta * trans_mult * cf, 1.0)

            # Split into whole + fractional acts (matching Python)
            whole_acts = floor(Int, acts)
            frac_acts = acts - whole_acts

            p_not_whole = whole_acts > 0 ? (1.0 - p_per_act)^whole_acts : 1.0
            p_not_frac = 1.0 - frac_acts * p_per_act
            p_per_dt = 1.0 - p_not_whole * p_not_frac

            p = rel_trans_raw[src] * rel_sus_raw[trg] * p_per_dt
            if rand(rng) < p
                _do_hpv_infection!(d, sim, trg, src, ti)
                new_infections += 1
            end
        end

        # Bidirectional: trg → src (use snapshot for infectiousness check)
        if bidir && infected_snapshot[trg] && susceptible_raw[src]
            # trg is now the source: if female→male use 1.0, if male→female use 3.69
            trans_mult = female_raw[trg] ? 1.0 : M2F_TRANS_RATIO
            p_per_act = min(per_act_beta * trans_mult * cf, 1.0)

            whole_acts = floor(Int, acts)
            frac_acts = acts - whole_acts
            p_not_whole = whole_acts > 0 ? (1.0 - p_per_act)^whole_acts : 1.0
            p_not_frac = 1.0 - frac_acts * p_per_act
            p_per_dt = 1.0 - p_not_whole * p_not_frac

            p = rel_trans_raw[trg] * rel_sus_raw[src] * p_per_dt
            if rand(rng) < p
                _do_hpv_infection!(d, sim, src, trg, ti)
                new_infections += 1
            end
        end
    end
    return new_infections
end

function _do_hpv_infection!(d::HPVGenotype, sim, target::Int, source::Int, ti::Int)
    d.infection.susceptible.raw[target] = false
    d.infection.infected.raw[target]    = true
    d.infection.ti_infected.raw[target] = Float64(ti)
    d.n_infections.raw[target] += 1.0

    # Reset cleared state
    d.cleared.raw[target] = false

    # Initialize rel_sev if not yet set
    if d.rel_sev.raw[target] <= 0.0
        d.rel_sev.raw[target] = max(0.2, 1.0 + 0.3 * randn(d.rng))
    end

    # Log infection
    push!(d.infection.infection_sources, (target, source, ti))

    # Set prognosis: females get full CIN pathway, males get short clearance
    # (matching Python: set_prognoses for females, dur_infection_male for males)
    if d.use_duration_model
        dt = sim.pars.dt
        if sim.people.female.raw[target]
            _set_prognosis!(d, target, ti, dt, sim)
        else
            _set_male_prognosis!(d, target, ti, dt)
        end
    end

    # Notify immunity connector about new infection
    for (_, conn) in sim.connectors
        if conn isa HPVImmunityConnector
            record_infection!(conn, d, target)
            break
        end
    end

    return
end

"""Set male infection prognosis: short clearance, no CIN pathway.

Males clear HPV much faster than females (~1 month vs months/years).
Matching Python: `dur_infection = sample(**dur_infection_male)`
"""
function _set_male_prognosis!(d::HPVGenotype, uid::Int, ti::Int, dt::Float64=0.25)
    # Males: short infection duration with randround (matching Python: np.ceil(dur/dt))
    dur = sample_lognormal_duration(d.rng, DUR_INFECTION_MALE_PAR1, DUR_INFECTION_MALE_PAR2)
    d.dur_precin.raw[uid] = dur
    d.dur_cin.raw[uid] = 0.0
    d.will_cin.raw[uid] = false
    d.will_cancer.raw[uid] = false
    # Python uses np.ceil for males: date_clearance = date_infectious + ceil(dur/dt)
    d.date_clearance.raw[uid] = Float64(ti) + ceil(dur / dt)
    return
end

# ============================================================================
# set_prognoses! — for direct infections (e.g., from reactivation)
# ============================================================================

function Starsim.set_prognoses!(d::HPVGenotype, sim, uids::Starsim.UIDs)
    ti = Starsim.module_data(d).t.ti
    for u in uids.values
        _do_hpv_infection!(d, sim, u, 0, ti)
    end
    return
end

# ============================================================================
# Death handling and results
# ============================================================================

function Starsim.step_die!(d::HPVGenotype, death_uids::Starsim.UIDs)
    d.infection.susceptible[death_uids] = false
    d.infection.infected[death_uids]    = false
    d.cin1[death_uids]      = false
    d.cin2[death_uids]      = false
    d.cin3[death_uids]      = false
    d.cancerous[death_uids] = false
    d.cleared[death_uids]   = false
    return d
end

function Starsim.update_results!(d::HPVGenotype, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    ti > length(md.results[:n_susceptible].values) && return d

    active = sim.people.auids.values
    sus_raw  = d.infection.susceptible.raw
    inf_raw  = d.infection.infected.raw
    c1_raw   = d.cin1.raw
    c2_raw   = d.cin2.raw
    c3_raw   = d.cin3.raw
    ca_raw   = d.cancerous.raw
    cl_raw   = d.cleared.raw

    n_sus = 0; n_inf = 0; n_c1 = 0; n_c2 = 0; n_c3 = 0; n_ca = 0; n_cl = 0
    @inbounds for u in active
        n_sus += sus_raw[u]
        n_inf += inf_raw[u]
        n_c1  += c1_raw[u]
        n_c2  += c2_raw[u]
        n_c3  += c3_raw[u]
        n_ca  += ca_raw[u]
        n_cl  += cl_raw[u]
    end

    md.results[:n_susceptible][ti] = Float64(n_sus)
    md.results[:n_infected][ti]    = Float64(n_inf)
    md.results[:n_cin1][ti]        = Float64(n_c1)
    md.results[:n_cin2][ti]        = Float64(n_c2)
    md.results[:n_cin3][ti]        = Float64(n_c3)
    md.results[:n_cancerous][ti]   = Float64(n_ca)
    md.results[:n_cleared][ti]     = Float64(n_cl)

    n_total = Float64(length(active))
    md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
    n_cin_total = Float64(n_c1 + n_c2 + n_c3)
    md.results[:cin_prevalence][ti] = n_total > 0.0 ? n_cin_total / n_total : 0.0
    return d
end

function Starsim.finalize!(d::HPVGenotype)
    md = Starsim.module_data(d)
    for (_, _, ti) in d.infection.infection_sources
        if ti > 0 && ti <= length(md.results[:new_infections].values)
            md.results[:new_infections][ti] += 1.0
        end
    end
    return d
end
