"""
Location-specific demographics for HPVsim, matching Python hpvsim exactly.

Python hpvsim uses:
- Year-specific crude birth rates (CBR per 1000) from UN data
- Year-specific, age/sex-specific mortality rates (mx) from UN life tables
- dt_demog=1.0 gating: demographics fire once per year, not every dt step
- Initial age distribution from location-specific population pyramid
- Births add agents at age 0
- Deaths remove agents via Bernoulli draw with age/sex/year-specific probability
"""

using StableRNGs

# ============================================================================
# HPVLocationDemographics — replaces simple Births + HPVDeaths
# ============================================================================

"""
    HPVLocationDemographics <: Starsim.AbstractDemographics

Location-specific demographics matching Python hpvsim exactly.
Handles both births and deaths in a single module with dt_demog gating.
"""
mutable struct HPVLocationDemographics <: Starsim.AbstractDemographics
    mod::Starsim.ModuleData
    # Data
    birth_years::Vector{Int}
    birth_rates::Vector{Float64}     # CBR per 1000
    death_years::Vector{Int}
    death_f::Vector{Vector{Float64}} # female mx by age bin, per year
    death_m::Vector{Vector{Float64}} # male mx by age bin, per year
    death_age_bins::Vector{Float64}  # age bin edges for death rates
    age_dist::Vector{Tuple{Int,Int,Int}}  # (age_min, age_max, count) for init
    # Migration data
    use_migration::Bool
    pop_trend_years::Vector{Int}
    pop_trend_sizes::Vector{Float64}
    age_trend_years::Vector{Int}
    age_trend_male::Vector{Vector{Float64}}   # [year_idx][age] counts
    age_trend_female::Vector{Vector{Float64}} # [year_idx][age] counts
    pop_scale::Float64  # n_agents / data_pop at start
    # Parameters
    dt_demog::Float64
    rel_birth::Float64
    rel_death::Float64
    sex_ratio::Float64
    # Internal
    _step_counter::Int
    _update_freq::Int
    rng::StableRNG
end

function HPVLocationDemographics(;
    name::Symbol = :location_demographics,
    location::Symbol = :nigeria,
    dt_demog::Real = 1.0,
    rel_birth::Real = 1.0,
    rel_death::Real = 1.0,
    sex_ratio::Real = 0.5,
    use_migration::Bool = true,
)
    md = Starsim.ModuleData(name; label="Location demographics ($location)")

    # Load data for location
    # Note: age_dist is set to a placeholder here; the correct year-specific
    # distribution is selected in init_pre! using sim.pars.start, matching
    # Python hpvsim which calls get_age_distribution(location, year=sim['start']).
    if location == :nigeria
        birth_years = NIGERIA_BIRTH_YEARS
        birth_rates = NIGERIA_BIRTH_RATES
        death_years = NIGERIA_DEATH_YEARS
        death_f = NIGERIA_DEATH_F
        death_m = NIGERIA_DEATH_M
        death_age_bins = DEATH_RATE_AGE_BINS
        age_dist = NIGERIA_AGE_DIST_2000  # Default to 2000; overridden in init_pre!
        pop_trend_years = NIGERIA_POP_TREND_YEARS
        pop_trend_sizes = NIGERIA_POP_TREND_SIZES
        age_trend_years = NIGERIA_AGE_TREND_YEARS
        age_trend_male = NIGERIA_AGE_TREND_MALE
        age_trend_female = NIGERIA_AGE_TREND_FEMALE
    else
        error("Location $location not supported; only :nigeria is available")
    end

    HPVLocationDemographics(
        md,
        birth_years, birth_rates,
        death_years, death_f, death_m, death_age_bins,
        age_dist,
        use_migration,
        pop_trend_years, pop_trend_sizes,
        age_trend_years, age_trend_male, age_trend_female,
        0.0,  # pop_scale — computed during init
        Float64(dt_demog), Float64(rel_birth), Float64(rel_death), Float64(sex_ratio),
        0, 1,
        StableRNG(0),
    )
end

Starsim.module_data(d::HPVLocationDemographics) = d.mod

# ============================================================================
# Initialization — age distribution from location data
# ============================================================================

function Starsim.init_pre!(d::HPVLocationDemographics, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    d.rng = StableRNG(sim.pars.rand_seed + hash(:location_demographics))
    d._step_counter = 0
    d._update_freq = max(1, Int(round(d.dt_demog / sim.pars.dt)))

    # Select age distribution for the sim start year (matching Python:
    # age_data = hpdata.get_age_distribution(location, year=sim['start']))
    d.age_dist = get_nigeria_age_dist(sim.pars.start)

    # Compute pop_scale: n_agents / data_pop(start_year)
    if d.use_migration && !isempty(d.pop_trend_years)
        data_pop0 = _interp_pop(d, Float64(sim.pars.start))
        d.pop_scale = Float64(sim.pars.n_agents) / data_pop0
    end

    # Set initial ages from location-specific age pyramid
    _init_ages_from_pyramid!(d, sim)

    return d
end

"""Initialize agent ages from location-specific age distribution (matching Python)."""
function _init_ages_from_pyramid!(d::HPVLocationDemographics, sim)
    people = sim.people
    active = people.auids.values
    n = length(active)
    rng = d.rng
    dt = sim.pars.dt

    # Build probability distribution from age pyramid
    age_dist = d.age_dist
    n_bins = length(age_dist)
    probs = Float64[Float64(ad[3]) for ad in age_dist]
    total = sum(probs)
    probs ./= total

    # Multinomial sampling: assign each agent to an age bin
    cum_probs = cumsum(probs)
    for u in active
        r = rand(rng)
        bi = 1
        @inbounds for j in 1:n_bins
            if r <= cum_probs[j]
                bi = j
                break
            end
        end

        age_min = Float64(age_dist[bi][1])
        age_max = Float64(age_dist[bi][2])
        age_range = age_max - age_min

        # Uniformly distribute within bin, rounded to dt (matching Python dt_round_age)
        n_steps = max(1, Int(floor(age_range / dt)))
        age = age_min + rand(rng, 0:n_steps-1) * dt

        people.age.raw[u] = age
    end

    return
end

# ============================================================================
# Step — births and deaths with dt_demog gating
# ============================================================================

function Starsim.step!(d::HPVLocationDemographics, sim)
    d._step_counter += 1

    # Only apply demographics every update_freq steps (matching Python dt_demog)
    if (d._step_counter - 1) % d._update_freq != 0
        return d
    end

    ti = sim.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt

    # Apply deaths first (matching Python order)
    _apply_death_rates!(d, sim, year)

    # Then births
    _apply_births!(d, sim, year)

    # Then migration (matching Python order)
    if d.use_migration
        _check_migration!(d, sim, year)
    end

    return d
end

# ============================================================================
# Deaths — year/age/sex-specific mortality
# ============================================================================

"""Apply age/sex-specific death rates matching Python hpvsim exactly."""
function _apply_death_rates!(d::HPVLocationDemographics, sim, year::Float64)
    people = sim.people
    rng = d.rng

    # Find nearest year in death rate data
    yr_int = Int(round(year))
    nearest_idx = 1
    min_diff = abs(d.death_years[1] - yr_int)
    for i in 2:length(d.death_years)
        diff = abs(d.death_years[i] - yr_int)
        if diff < min_diff
            min_diff = diff
            nearest_idx = i
        end
    end

    f_rates = d.death_f[nearest_idx]
    m_rates = d.death_m[nearest_idx]
    age_bins = d.death_age_bins
    n_bins = length(age_bins)

    death_uids = Int[]

    @inbounds for u in people.auids.values
        age = people.age.raw[u]

        # np.digitize(age, bins) - 1: find bin index
        bi = n_bins  # default to last bin
        for j in n_bins:-1:1
            if age >= age_bins[j]
                bi = j
                break
            end
        end
        # Clamp to valid range
        bi = min(bi, length(f_rates))

        # Get death probability = mx * dt_demog * rel_death
        is_female = people.female.raw[u]
        mx = is_female ? f_rates[bi] : m_rates[bi]
        death_prob = mx * d.dt_demog * d.rel_death

        # age > 100 → death_prob = 1.0 (matching Python)
        if age > 100.0
            death_prob = 1.0
        end

        # Bernoulli draw
        if rand(rng) < death_prob
            push!(death_uids, u)
        end
    end

    # Remove dead agents
    if !isempty(death_uids)
        ti = Starsim.module_data(d).t.ti
        Starsim.request_death!(sim.people, Starsim.UIDs(death_uids), ti)
    end

    return
end

# ============================================================================
# Births — year-specific crude birth rate
# ============================================================================

"""Apply births using year-specific CBR matching Python hpvsim exactly."""
function _apply_births!(d::HPVLocationDemographics, sim, year::Float64)
    people = sim.people
    rng = d.rng
    n_alive = length(people.auids.values)

    # Interpolate CBR for current year (matching Python np.interp)
    cbr = _interp_birth_rate(d, year)

    # new_births = randround(rel_birth * cbr * dt_demog / 1000 * n_alive)
    expected = d.rel_birth * cbr * d.dt_demog / 1000.0 * n_alive
    n_births = _randround(rng, expected)

    n_births <= 0 && return

    # Add new agents via grow! (age defaults to 0.0)
    new_uids = Starsim.grow!(people, n_births)

    # Set sex for new agents (matching Python sex_ratio)
    for u in new_uids.values
        is_female = rand(rng) < d.sex_ratio
        people.female.raw[u] = is_female
    end

    return
end

"""Linearly interpolate birth rate for a given year (matching numpy.interp)."""
function _interp_birth_rate(d::HPVLocationDemographics, year::Float64)
    years = d.birth_years
    rates = d.birth_rates
    n = length(years)

    year <= Float64(years[1]) && return rates[1]
    year >= Float64(years[n]) && return rates[n]

    # Find bracketing years
    for i in 1:n-1
        y1 = Float64(years[i])
        y2 = Float64(years[i+1])
        if y1 <= year <= y2
            frac = (year - y1) / (y2 - y1)
            return rates[i] + frac * (rates[i+1] - rates[i])
        end
    end

    return rates[n]
end

"""Randomly round to floor or ceil based on fractional part (matching sciris randround)."""
function _randround(rng, x::Float64)
    fl = floor(Int, x)
    frac = x - fl
    return rand(rng) < frac ? fl + 1 : fl
end

# ============================================================================
# Migration — match population to expected age/sex distribution
# ============================================================================

"""Interpolate total population for a given year from population trend data."""
function _interp_pop(d::HPVLocationDemographics, year::Float64)
    years = d.pop_trend_years
    sizes = d.pop_trend_sizes
    n = length(years)
    year <= Float64(years[1]) && return sizes[1]
    year >= Float64(years[n]) && return sizes[n]
    for i in 1:n-1
        y1, y2 = Float64(years[i]), Float64(years[i+1])
        if y1 <= year <= y2
            frac = (year - y1) / (y2 - y1)
            return sizes[i] + frac * (sizes[i+1] - sizes[i])
        end
    end
    return sizes[n]
end

"""Find nearest year index in age trend data."""
function _nearest_age_trend_idx(d::HPVLocationDemographics, year::Float64)
    yr = Int(round(year))
    best_idx = 1
    best_diff = abs(d.age_trend_years[1] - yr)
    for i in 2:length(d.age_trend_years)
        diff = abs(d.age_trend_years[i] - yr)
        if diff < best_diff
            best_diff = diff
            best_idx = i
        end
    end
    return best_idx
end

"""
    _check_migration!(d, sim, year)

Adjust population via migration to match expected age/sex distribution.
Matching Python hpvsim's check_migration method.
"""
function _check_migration!(d::HPVLocationDemographics, sim, year::Float64)
    people = sim.people
    rng = d.rng

    isempty(d.pop_trend_years) && return
    isempty(d.age_trend_years) && return

    # Skip if outside data range
    yr_first = Float64(d.pop_trend_years[1])
    yr_last = Float64(d.pop_trend_years[end])
    (year < yr_first || year > yr_last) && return

    scale = d.pop_scale  # n_agents / data_pop at start
    scale <= 0.0 && return

    # Get expected age distribution for this year
    ati = _nearest_age_trend_idx(d, year)
    expected_male = d.age_trend_male[ati]
    expected_female = d.age_trend_female[ati]
    n_ages = length(expected_male)  # 101 (ages 0-100)

    # Process each sex separately (matching Python)
    for is_female_sex in [false, true]
        expected = is_female_sex ? expected_female : expected_male

        # Count current ALIVE agents by integer age
        # Skip agents already scheduled to die (ti_dead <= current ti),
        # matching Python where remove_people sets alive=False before migration runs
        current_ti = Float64(Starsim.module_data(d).t.ti)
        count_ages = zeros(Int, n_ages)
        age_uids = [Int[] for _ in 1:n_ages]  # UIDs per age bin
        for u in people.auids.values
            people.female.raw[u] == is_female_sex || continue
            people.ti_dead.raw[u] <= current_ti && continue  # already dead
            age_int = min(Int(floor(people.age.raw[u])), n_ages - 1)
            ai = age_int + 1  # 1-indexed
            count_ages[ai] += 1
            push!(age_uids[ai], u)
        end

        # Compute difference: positive = need immigration, negative = need emigration
        # Match Python: (expected_float - count_int).astype(int) truncates toward zero
        difference = [Int(trunc(expected[ai] * scale - count_ages[ai])) for ai in 1:n_ages]

        # Immigration: add agents at ages where we have too few
        for ai in 1:n_ages
            n_to_add = difference[ai]
            n_to_add <= 0 && continue
            new_uids = Starsim.grow!(people, n_to_add)
            age_val = Float64(ai - 1)  # age = 0-indexed
            for u in new_uids.values
                people.age.raw[u] = age_val
                people.female.raw[u] = is_female_sex
            end
        end

        # Emigration: remove agents at ages where we have too many
        for ai in 1:n_ages
            n_to_remove = -difference[ai]
            n_to_remove <= 0 && continue
            n_to_remove = min(n_to_remove, length(age_uids[ai]))
            n_to_remove <= 0 && continue
            # Sample agents to remove (matching Python's choose_w with uniform probs)
            uids_pool = age_uids[ai]
            if n_to_remove >= length(uids_pool)
                remove_these = uids_pool
            else
                # Random selection without replacement
                perm = randperm(rng, length(uids_pool))
                remove_these = uids_pool[perm[1:n_to_remove]]
            end
            ti = Starsim.module_data(d).t.ti
            Starsim.request_death!(people, Starsim.UIDs(remove_these), ti)
        end
    end

    return
end
