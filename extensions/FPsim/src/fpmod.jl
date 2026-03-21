"""
FPmod — Reproductive lifecycle state machine connector.
Core connector that manages pregnancy, breastfeeding, postpartum,
contraception interaction, and birth outcomes.
"""

# ============================================================================
# FPmod <: AbstractConnector
# ============================================================================

"""
    FPmod <: AbstractConnector

Reproductive lifecycle state machine. Manages:
- Sexual debut (fated age)
- Conception and pregnancy (with contraceptive efficacy)
- Birth outcomes (live birth, miscarriage, stillbirth, abortion)
- Breastfeeding, postpartum, and LAM
- Fecundity by age, parity, and individual variation
- Newborn population growth

# Keyword arguments
- `pars::FPPars` — family planning parameters
"""
mutable struct FPmod <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    pars::FPPars

    # Core reproductive states
    sexually_active::Starsim.StateVector{Bool, Vector{Bool}}
    pregnant::Starsim.StateVector{Bool, Vector{Bool}}
    ti_pregnant::Starsim.StateVector{Float64, Vector{Float64}}
    postpartum::Starsim.StateVector{Bool, Vector{Bool}}
    breastfeeding::Starsim.StateVector{Bool, Vector{Bool}}
    parity::Starsim.StateVector{Float64, Vector{Float64}}
    months_postpartum::Starsim.StateVector{Float64, Vector{Float64}}
    gestation_month::Starsim.StateVector{Float64, Vector{Float64}}
    ti_birth::Starsim.StateVector{Float64, Vector{Float64}}

    # Contraception
    on_contra::Starsim.StateVector{Bool, Vector{Bool}}
    method_idx::Starsim.StateVector{Float64, Vector{Float64}}

    # Sexual debut
    sexual_debut::Starsim.StateVector{Bool, Vector{Bool}}
    fated_debut::Starsim.StateVector{Float64, Vector{Float64}}

    # Individual fecundity
    personal_fecundity::Starsim.StateVector{Float64, Vector{Float64}}
    fertile::Starsim.StateVector{Bool, Vector{Bool}}

    # Lactational amenorrhea
    lam::Starsim.StateVector{Bool, Vector{Bool}}

    # Sampled durations per-agent
    dur_pregnancy_state::Starsim.StateVector{Float64, Vector{Float64}}
    dur_breastfeed_state::Starsim.StateVector{Float64, Vector{Float64}}

    # Methods reference for efficacy lookup
    methods::Vector{Method}

    rng::StableRNG
end

function FPmod(;
    name::Symbol = :fpmod,
    pars::Union{FPPars, Nothing} = nothing,
    methods::Union{Vector{Method}, Nothing} = nothing,
)
    md = Starsim.ModuleData(name; label="Reproductive lifecycle")
    cd = Starsim.ConnectorData(md)

    fp = pars === nothing ? load_location_data(:generic) : pars
    m = methods === nothing ? load_methods() : methods

    FPmod(
        cd, fp,
        # Core states
        Starsim.BoolState(:sexually_active; default=false, label="Sexually active"),
        Starsim.BoolState(:pregnant; default=false, label="Pregnant"),
        Starsim.FloatState(:ti_pregnant; default=Inf, label="Time pregnant"),
        Starsim.BoolState(:postpartum; default=false, label="Postpartum"),
        Starsim.BoolState(:breastfeeding; default=false, label="Breastfeeding"),
        Starsim.FloatState(:parity; default=0.0, label="Parity"),
        Starsim.FloatState(:months_postpartum; default=0.0, label="Months PP"),
        Starsim.FloatState(:gestation_month; default=0.0, label="Gestation month"),
        Starsim.FloatState(:ti_birth; default=Inf, label="Time of birth"),
        # Contraception
        Starsim.BoolState(:on_contra; default=false, label="On contraception"),
        Starsim.FloatState(:method_idx; default=0.0, label="Method index"),
        # Sexual debut
        Starsim.BoolState(:sexual_debut; default=false, label="Sexual debut"),
        Starsim.FloatState(:fated_debut; default=Inf, label="Fated debut age"),
        # Individual fecundity
        Starsim.FloatState(:personal_fecundity; default=1.0, label="Personal fecundity"),
        Starsim.BoolState(:fertile; default=true, label="Fertile"),
        # LAM
        Starsim.BoolState(:lam; default=false, label="Using LAM"),
        # Durations
        Starsim.FloatState(:dur_pregnancy_state; default=9.0, label="Pregnancy duration"),
        Starsim.FloatState(:dur_breastfeed_state; default=24.0, label="Breastfeeding duration"),
        # Methods
        m,
        # RNG
        StableRNG(0),
    )
end

Starsim.connector_data(c::FPmod) = c.data

# ============================================================================
# Initialization
# ============================================================================

function Starsim.init_pre!(c::FPmod, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    c.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    all_states = [
        c.sexually_active, c.pregnant, c.ti_pregnant, c.postpartum,
        c.breastfeeding, c.parity, c.months_postpartum, c.gestation_month,
        c.ti_birth, c.on_contra, c.method_idx, c.sexual_debut,
        c.fated_debut, c.personal_fecundity, c.fertile, c.lam,
        c.dur_pregnancy_state, c.dur_breastfeed_state,
    ]
    for s in all_states
        Starsim.add_module_state!(sim.people, s)
    end

    npts = md.t.npts
    Starsim.define_results!(c,
        Starsim.Result(:n_pregnant; npts=npts, label="Pregnant", scale=false),
        Starsim.Result(:n_births; npts=npts, label="Births"),
        Starsim.Result(:n_miscarriages; npts=npts, label="Miscarriages"),
        Starsim.Result(:n_stillbirths; npts=npts, label="Stillbirths"),
        Starsim.Result(:n_abortions; npts=npts, label="Abortions"),
        Starsim.Result(:n_maternal_deaths; npts=npts, label="Maternal deaths"),
        Starsim.Result(:n_infant_deaths; npts=npts, label="Infant deaths"),
        Starsim.Result(:n_sexually_active; npts=npts, label="Sexually active", scale=false),
        Starsim.Result(:n_on_contra; npts=npts, label="On contraception", scale=false),
    )

    md.initialized = true
    return c
end

function Starsim.init_post!(c::FPmod, sim)
    people = sim.people
    active = people.auids.values
    pars = c.pars

    for u in active
        if people.female.raw[u]
            age = people.age.raw[u]

            # Draw personal fecundity (uniform [fecundity_low, fecundity_high])
            c.personal_fecundity.raw[u] = pars.fecundity_low +
                rand(c.rng) * (pars.fecundity_high - pars.fecundity_low)

            # Primary infertility
            c.fertile.raw[u] = rand(c.rng) >= pars.primary_infertility

            # Draw fated sexual debut age
            c.fated_debut.raw[u] = draw_debut_age(c.rng, pars)

            # Initialize sexual activity for women past debut age
            if age >= c.fated_debut.raw[u]
                c.sexual_debut.raw[u] = true
                idx = int_age_clip(age)
                sa = !isempty(pars.sexual_activity) && idx <= length(pars.sexual_activity) ?
                    pars.sexual_activity[idx] : 0.5
                if rand(c.rng) < sa
                    c.sexually_active.raw[u] = true
                end
            end

            # Seed some initial pregnancies for reproductive-age women
            if age >= pars.method_age && age < pars.age_limit_fecundity && c.sexually_active.raw[u]
                fecundity = age_lookup(pars.age_fecundity, age, 0.0) * c.personal_fecundity.raw[u]
                if rand(c.rng) < fecundity * 0.05  # small initial seed
                    c.pregnant.raw[u] = true
                    c.ti_pregnant.raw[u] = 1.0
                    c.gestation_month.raw[u] = rand(c.rng) * 8.0  # random gestation stage
                    c.dur_pregnancy_state.raw[u] = 9.0
                end
            end

            # Seed some initial postpartum women
            if age >= 20 && age < pars.age_limit_fecundity && !c.pregnant.raw[u]
                if rand(c.rng) < 0.05  # ~5% postpartum at initialization
                    c.postpartum.raw[u] = true
                    c.breastfeeding.raw[u] = true
                    c.months_postpartum.raw[u] = rand(c.rng) * Float64(pars.dur_postpartum)
                    c.parity.raw[u] = max(c.parity.raw[u], 1.0)
                    bf_dur = max(1.0, pars.dur_breastfeeding_mean + randn(c.rng) * pars.dur_breastfeeding_std)
                    c.dur_breastfeed_state.raw[u] = bf_dur
                end
            end
        end
    end
    return c
end

# ============================================================================
# Step logic
# ============================================================================

function Starsim.step!(c::FPmod, sim)
    md = Starsim.module_data(c)
    ti = md.t.ti
    dt = sim.pars.dt
    dt_months = dt * MPY
    year = sim.pars.start + (ti - 1) * dt
    active = sim.people.auids.values
    pars = c.pars

    n_births = 0
    n_miscarriages = 0
    n_stillbirths = 0
    n_abortions = 0
    n_maternal_deaths = 0
    n_infant_deaths = 0
    births_to_add = 0

    @inbounds for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]

        # ---- 1. Sexual debut ----
        if !c.sexual_debut.raw[u]
            if age >= c.fated_debut.raw[u]
                c.sexual_debut.raw[u] = true
                c.sexually_active.raw[u] = true
            end
            continue  # not yet active, nothing to do
        end

        # Update sexual activity for non-pregnant, non-postpartum, debuted women
        # Python re-evaluates each timestep: women can toggle active↔inactive
        if !c.pregnant.raw[u] && !c.postpartum.raw[u] && c.sexual_debut.raw[u]
            sa = age_lookup(pars.sexual_activity, age, 0.0)
            c.sexually_active.raw[u] = rand(c.rng) < sa
        end

        # ---- 2. Handle ongoing pregnancy ----
        if c.pregnant.raw[u]
            c.gestation_month.raw[u] += dt_months

            # First trimester miscarriage check — fires once when gest crosses end_first_tri
            gest = c.gestation_month.raw[u]
            prev_gest = gest - dt_months
            if prev_gest < pars.end_first_tri && gest >= pars.end_first_tri
                mis_rate = age_lookup(pars.miscarriage_rates, age, 0.12)
                if rand(c.rng) < mis_rate
                    n_miscarriages += 1
                    _end_pregnancy!(c, u)
                    continue
                end
            end

            # Delivery check
            if gest >= c.dur_pregnancy_state.raw[u]
                # Maternal mortality
                mat_mort = interp_year(pars.maternal_mort_years, pars.maternal_mort_probs, year) *
                           pars.maternal_mortality_factor
                if mat_mort > 0 && rand(c.rng) < mat_mort
                    n_maternal_deaths += 1
                    _end_pregnancy!(c, u)
                    Starsim.request_death!(sim.people, Starsim.UIDs([u]), ti)
                    continue
                end

                # Stillbirth check
                still_prob = interp_year(pars.stillbirth_years, pars.stillbirth_probs, year)
                if rand(c.rng) < still_prob
                    n_stillbirths += 1
                    _start_postpartum!(c, u, pars, ti)
                else
                    # Live birth
                    n_births += 1
                    births_to_add += 1
                    c.parity.raw[u] += 1.0

                    # Twins check
                    if rand(c.rng) < pars.twins_prob
                        n_births += 1
                        births_to_add += 1
                        c.parity.raw[u] += 1.0
                    end

                    # Infant mortality (tracked but doesn't affect mother)
                    inf_mort = interp_year(pars.infant_mort_years, pars.infant_mort_probs, year)
                    if rand(c.rng) < inf_mort
                        n_infant_deaths += 1
                        births_to_add -= 1  # infant doesn't survive
                    end

                    _start_postpartum!(c, u, pars, ti)
                    c.breastfeeding.raw[u] = true
                    c.lam.raw[u] = true

                    # Draw breastfeeding duration
                    bf_dur = max(1.0, pars.dur_breastfeeding_mean +
                                      randn(c.rng) * pars.dur_breastfeeding_std)
                    c.dur_breastfeed_state.raw[u] = bf_dur
                end

                c.pregnant.raw[u] = false
                c.gestation_month.raw[u] = 0.0
            end
            continue
        end

        # ---- 3. Handle postpartum ----
        if c.postpartum.raw[u]
            c.months_postpartum.raw[u] += dt_months

            # LAM check
            if c.lam.raw[u]
                mo = Int(floor(c.months_postpartum.raw[u]))
                if mo >= pars.max_lam_dur || !c.breastfeeding.raw[u]
                    c.lam.raw[u] = false
                elseif !isempty(pars.lam_rates)
                    lam_idx = min(mo + 1, length(pars.lam_rates))
                    if rand(c.rng) > pars.lam_rates[lam_idx]
                        c.lam.raw[u] = false
                    end
                end
            end

            # Breastfeeding duration check
            if c.breastfeeding.raw[u] && c.months_postpartum.raw[u] >= c.dur_breastfeed_state.raw[u]
                c.breastfeeding.raw[u] = false
                c.lam.raw[u] = false
            end

            # Postpartum sexual activity (matches Python spacing_pref logic)
            # Postpartum sexual activity — re-evaluated each step (matches Python)
            begin
                mo = Int(floor(c.months_postpartum.raw[u]))
                if !isempty(pars.pp_percent_active)
                    pp_idx = min(mo + 1, length(pars.pp_percent_active))
                    pp_sa = pars.pp_percent_active[pp_idx]
                else
                    pp_sa = mo >= 6 ? 0.7 : 0.3
                end
                # Apply spacing preference if available (matches Python)
                if !isempty(pars.spacing_weights) && pars.spacing_interval > 0
                    sp_bin = min(Int(floor(c.months_postpartum.raw[u] / pars.spacing_interval)),
                                pars.spacing_n_bins)
                    sp_idx = min(sp_bin + 1, length(pars.spacing_weights))
                    pp_sa *= pars.spacing_weights[sp_idx]
                end
                c.sexually_active.raw[u] = rand(c.rng) < pp_sa
            end

            # End postpartum period
            if c.months_postpartum.raw[u] >= pars.dur_postpartum
                c.postpartum.raw[u] = false
                c.breastfeeding.raw[u] = false
                c.lam.raw[u] = false
            end

            # If still under LAM protection, skip conception
            c.lam.raw[u] && continue
        end

        # ---- 4. Conception ----
        if c.sexually_active.raw[u] && c.fertile.raw[u] &&
           !c.pregnant.raw[u] && age >= pars.method_age && age < pars.age_limit_fecundity

            # Base fecundity
            fecundity = age_lookup(pars.age_fecundity, age, 0.0) * c.personal_fecundity.raw[u]

            # Nulliparous reduction
            if c.parity.raw[u] == 0.0
                fecundity *= age_lookup(pars.fecundity_ratio_nullip, age, 1.0)
            end

            # Exposure factors
            exposure = pars.exposure_factor
            exposure *= exposure_age_factor(pars, age)
            exposure *= exposure_parity_factor(pars, c.parity.raw[u])

            # Contraceptive efficacy
            rel_sus = 1.0
            if c.on_contra.raw[u]
                midx = Int(c.method_idx.raw[u])
                if midx >= 1 && midx <= length(c.methods)
                    rel_sus *= (1.0 - c.methods[midx].efficacy)
                end
            end

            # LAM efficacy (for women past initial LAM check but still breastfeeding)
            if c.lam.raw[u]
                rel_sus *= (1.0 - pars.LAM_efficacy)
            end

            # Build annual probability, then convert to per-timestep
            # Matches Python: probperyear(raw_probs).to_prob(dt)
            raw_prob = fecundity * exposure * rel_sus
            raw_prob = clamp(raw_prob, 0.0, 1.0)
            p_conceive = prob_per_timestep(raw_prob, dt)

            if rand(c.rng) < p_conceive
                # Abortion check
                if rand(c.rng) < pars.abortion_prob
                    n_abortions += 1
                else
                    # Make pregnant
                    c.pregnant.raw[u] = true
                    c.ti_pregnant.raw[u] = Float64(ti)
                    c.gestation_month.raw[u] = 0.0
                    c.dur_pregnancy_state.raw[u] = pars.dur_pregnancy_low
                    # Stop contraception
                    c.on_contra.raw[u] = false
                    c.method_idx.raw[u] = 0.0
                    c.postpartum.raw[u] = false
                    c.lam.raw[u] = false
                end
            end
        end
    end

    # ---- Record step-level results ----
    res = Starsim.module_results(c)
    if ti <= length(res[:n_births].values)
        res[:n_births][ti] = Float64(n_births)
        res[:n_miscarriages][ti] = Float64(n_miscarriages)
        res[:n_stillbirths][ti] = Float64(n_stillbirths)
        res[:n_abortions][ti] = Float64(n_abortions)
        res[:n_maternal_deaths][ti] = Float64(n_maternal_deaths)
        res[:n_infant_deaths][ti] = Float64(n_infant_deaths)
    end

    # ---- Add newborns to population ----
    if births_to_add > 0
        births_to_add = max(births_to_add, 0)
        new_uids = Starsim.grow!(sim.people, births_to_add)
        # Initialize newborn states
        for u in new_uids.values
            c.fated_debut.raw[u] = draw_debut_age(c.rng, pars)
            c.personal_fecundity.raw[u] = pars.fecundity_low +
                rand(c.rng) * (pars.fecundity_high - pars.fecundity_low)
            c.fertile.raw[u] = rand(c.rng) >= pars.primary_infertility
        end
    end

    return c
end

# ============================================================================
# Helper functions
# ============================================================================

"""Reset pregnancy state."""
function _end_pregnancy!(c::FPmod, u::Int)
    c.pregnant.raw[u] = false
    c.gestation_month.raw[u] = 0.0
    return nothing
end

"""Transition to postpartum state."""
function _start_postpartum!(c::FPmod, u::Int, pars::FPPars, ti::Int)
    c.postpartum.raw[u] = true
    c.months_postpartum.raw[u] = 0.0
    c.ti_birth.raw[u] = Float64(ti)
    c.sexually_active.raw[u] = false  # temporarily inactive postpartum
    return nothing
end

# ============================================================================
# Results update
# ============================================================================

function Starsim.update_results!(c::FPmod, sim)
    md = Starsim.module_data(c)
    ti = md.t.ti
    ti > length(md.results[:n_pregnant].values) && return c

    active = sim.people.auids.values
    n_preg = 0; n_sa = 0; n_contra = 0
    @inbounds for u in active
        n_preg += c.pregnant.raw[u]
        n_sa += c.sexually_active.raw[u]
        n_contra += c.on_contra.raw[u]
    end

    md.results[:n_pregnant][ti] = Float64(n_preg)
    md.results[:n_sexually_active][ti] = Float64(n_sa)
    md.results[:n_on_contra][ti] = Float64(n_contra)
    return c
end

# ============================================================================
# Contraception connector
# ============================================================================

"""
    Contraception <: AbstractIntervention

Contraceptive method intervention. Assigns, switches, and discontinues
methods for eligible women based on method mix probabilities.
"""
mutable struct Contraception <: Starsim.AbstractIntervention
    iv::Starsim.InterventionData
    methods::Vector{Method}
    method_mix::MethodMix
    initiation_rate::Float64    # Annual probability of starting contraception for non-users
    initial_cpr::Float64        # Initial CPR at sim start (among eligible women)
    rng::StableRNG
end

function Contraception(;
    name::Symbol = :contraception,
    methods::Union{Vector{Method}, Nothing} = nothing,
    method_mix::Union{MethodMix, Nothing} = nothing,
    initiation_rate::Float64 = 0.10,
    initial_cpr::Float64 = 0.25,
)
    md = Starsim.ModuleData(name; label="Contraception")
    iv = Starsim.InterventionData(md, nothing, :none)
    m = methods === nothing ? load_methods() : methods
    mm = method_mix === nothing ? DEFAULT_METHOD_MIX : method_mix
    Contraception(iv, m, mm, initiation_rate, initial_cpr, StableRNG(0))
end

Starsim.intervention_data(c::Contraception) = c.iv

function Starsim.init_pre!(c::Contraception, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    c.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    npts = md.t.npts
    Starsim.define_results!(c,
        Starsim.Result(:n_initiations; npts=npts, label="New users"),
        Starsim.Result(:n_discontinuations; npts=npts, label="Discontinued"),
        Starsim.Result(:n_switches; npts=npts, label="Switched method"),
    )

    md.initialized = true
    return c
end

function Starsim.init_post!(c::Contraception, sim)
    # Seed initial contraceptive users based on method mix
    fpmod = _find_fpmod(sim)
    fpmod === nothing && return c

    pars = fpmod.pars
    active = sim.people.auids.values
    methods = c.methods
    mix = c.method_mix

    for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]
        age < pars.method_age && continue
        !fpmod.sexually_active.raw[u] && continue
        fpmod.pregnant.raw[u] && continue
        fpmod.postpartum.raw[u] && continue

        # Initial uptake probability (matches location CPR data)
        if rand(c.rng) < c.initial_cpr
            midx = sample_method(c.rng, methods, mix)
            fpmod.on_contra.raw[u] = true
            fpmod.method_idx.raw[u] = Float64(midx)
        end
    end
    return c
end

function Starsim.step!(c::Contraception, sim)
    fpmod = _find_fpmod(sim)
    fpmod === nothing && return c

    md = Starsim.module_data(c)
    ti = md.t.ti
    dt = sim.pars.dt
    active = sim.people.auids.values
    pars = fpmod.pars
    methods = c.methods
    mix = c.method_mix

    n_init = 0; n_disc = 0; n_switch = 0

    @inbounds for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]
        age < pars.method_age && continue
        !fpmod.sexually_active.raw[u] && continue
        fpmod.pregnant.raw[u] && continue

        if fpmod.on_contra.raw[u]
            # Discontinuation check
            midx = Int(fpmod.method_idx.raw[u])
            if midx >= 1 && midx <= length(methods)
                disc_prob = prob_per_timestep(methods[midx].discontinuation, dt)
                if rand(c.rng) < disc_prob
                    # Most women who discontinue switch to another method (matching Python)
                    if rand(c.rng) < mix.switch_prob
                        new_midx = sample_method(c.rng, methods, mix)
                        fpmod.method_idx.raw[u] = Float64(new_midx)
                        n_switch += 1
                    else
                        fpmod.on_contra.raw[u] = false
                        fpmod.method_idx.raw[u] = 0.0
                        n_disc += 1
                    end
                end
            end
        else
            # Initiation check for non-users (higher rate for postpartum women)
            base_init = prob_per_timestep(c.initiation_rate, dt)
            # Postpartum women re-initiate faster (matching Python ti_contra mechanism)
            init_prob = if fpmod.postpartum.raw[u] && fpmod.months_postpartum.raw[u] < 3.0
                min(base_init * 3.0, 0.5)  # boosted initiation early postpartum
            else
                base_init
            end
            if rand(c.rng) < init_prob
                midx = sample_method(c.rng, methods, mix)
                fpmod.on_contra.raw[u] = true
                fpmod.method_idx.raw[u] = Float64(midx)
                n_init += 1
            end
        end
    end

    res = Starsim.module_results(c)
    if ti <= length(res[:n_initiations].values)
        res[:n_initiations][ti] = Float64(n_init)
        res[:n_discontinuations][ti] = Float64(n_disc)
        res[:n_switches][ti] = Float64(n_switch)
    end

    return c
end

"""Find the FPmod connector in the simulation."""
function _find_fpmod(sim)
    for (_, c) in sim.connectors
        c isa FPmod && return c
    end
    return nothing
end
