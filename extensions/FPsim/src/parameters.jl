"""
Parameter definitions for FPsim.jl.
Port of Python fpsim `parameters.py` and `defaults.py`.
"""

"""
    FPPars

Family planning parameters. All durations default to monthly timestep units.
"""
Base.@kwdef mutable struct FPPars
    # Age limits
    method_age::Float64 = 15.0
    age_limit_fecundity::Float64 = 50.0
    max_age::Float64 = 99.0

    # Durations (in months)
    end_first_tri::Int = 3
    dur_pregnancy_low::Float64 = 9.0    # months
    dur_pregnancy_high::Float64 = 9.0   # months
    dur_breastfeeding_mean::Float64 = 24.0  # months
    dur_breastfeeding_std::Float64 = 6.0    # months
    dur_postpartum::Int = 35  # Updated from data
    max_lam_dur::Int = 5
    short_int_months::Int = 24

    # Conception
    LAM_efficacy::Float64 = 0.98
    primary_infertility::Float64 = 0.05

    # Calibration-tunable
    maternal_mortality_factor::Float64 = 1.0
    fecundity_low::Float64 = 0.7
    fecundity_high::Float64 = 1.1
    exposure_factor::Float64 = 1.0

    # Exposure splines (age-indexed and parity-indexed)
    exposure_age_ages::Vector{Float64} = Float64[0, 5, 10, 12.5, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    exposure_age_vals::Vector{Float64} = ones(13)
    exposure_parity_parities::Vector{Float64} = Float64[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20]
    exposure_parity_vals::Vector{Float64} = Float64[1, 1, 1, 1, 1, 1, 1, 0.8, 0.5, 0.3, 0.15, 0.10, 0.05, 0.01]

    # Location data (populated by load_location_data)
    abortion_prob::Float64 = 0.0
    twins_prob::Float64 = 0.0

    # Age-indexed arrays (0:MAX_AGE indexed by int_age+1)
    age_fecundity::Vector{Float64} = Float64[]
    fecundity_ratio_nullip::Vector{Float64} = Float64[]
    miscarriage_rates::Vector{Float64} = Float64[]
    sexual_activity::Vector{Float64} = Float64[]

    # Debut age distribution
    debut_ages::Vector{Float64} = Float64[]
    debut_probs::Vector{Float64} = Float64[]

    # Lactational amenorrhea
    lam_months::Vector{Int} = Int[]
    lam_rates::Vector{Float64} = Float64[]

    # Postpartum sexual activity
    pp_months::Vector{Int} = Int[]
    pp_percent_active::Vector{Float64} = Float64[]

    # Spacing preference
    spacing_months::Vector{Float64} = Float64[]
    spacing_weights::Vector{Float64} = Float64[]
    spacing_interval::Float64 = 12.0
    spacing_n_bins::Int = 36

    # Mortality (year-indexed)
    maternal_mort_years::Vector{Float64} = Float64[]
    maternal_mort_probs::Vector{Float64} = Float64[]
    infant_mort_years::Vector{Float64} = Float64[]
    infant_mort_probs::Vector{Float64} = Float64[]
    stillbirth_years::Vector{Float64} = Float64[]
    stillbirth_probs::Vector{Float64} = Float64[]

    # Infant mortality age adjustment
    infant_mort_ages::Vector{Float64} = Float64[]
    infant_mort_age_probs::Vector{Float64} = Float64[]

    # Stillbirth age adjustment
    stillbirth_rate_ages::Vector{Float64} = Float64[]
    stillbirth_rate_age_probs::Vector{Float64} = Float64[]

    # Death rates by age/sex (for background mortality)
    death_rate_ages::Vector{Float64} = Float64[]
    death_rate_female::Vector{Float64} = Float64[]
    death_rate_male::Vector{Float64} = Float64[]

    # Age pyramid for initialization
    age_pyramid_ages::Vector{Float64} = Float64[]
    age_pyramid_male::Vector{Float64} = Float64[]
    age_pyramid_female::Vector{Float64} = Float64[]
end

"""
    interpolate_exposure(ages, vals, query_ages)

Create an interpolation for exposure factor lookup.
"""
function interpolate_exposure(ages::Vector{Float64}, vals::Vector{Float64}, query_age::Real)
    query_age = clamp(Float64(query_age), ages[1], ages[end])
    itp = linear_interpolation(ages, vals; extrapolation_bc=Flat())
    return itp(query_age)
end

"""
    exposure_age_factor(pars, age)

Look up the exposure factor for a given age.
"""
function exposure_age_factor(pars::FPPars, age::Real)
    isempty(pars.exposure_age_ages) && return 1.0
    return interpolate_exposure(pars.exposure_age_ages, pars.exposure_age_vals, age)
end

"""
    exposure_parity_factor(pars, parity)

Look up the exposure factor for a given parity.
"""
function exposure_parity_factor(pars::FPPars, parity::Real)
    isempty(pars.exposure_parity_parities) && return 1.0
    p = clamp(Float64(parity), 0.0, Float64(MAX_PARITY))
    return interpolate_exposure(pars.exposure_parity_parities, pars.exposure_parity_vals, p)
end

"""Integer-clipped age index (1-based, clamped to 1:100)."""
int_age_clip(age::Real) = clamp(Int(floor(age)) + 1, 1, MAX_AGE + 1)


"""
    prob_per_timestep(annual_prob, dt)

Convert an annual probability to a per-timestep probability.
Uses: p_dt = 1 - (1 - p_annual)^dt
"""
function prob_per_timestep(annual_prob::Real, dt::Real)
    annual_prob <= 0.0 && return 0.0
    annual_prob >= 1.0 && return 1.0
    return 1.0 - (1.0 - annual_prob)^dt
end

"""
    interp_year(years, vals, year)

Interpolate a year-indexed time series at a given year.
"""
function interp_year(years::Vector{Float64}, vals::Vector{Float64}, year::Real)
    isempty(years) && return 0.0
    length(years) == 1 && return vals[1]
    y = clamp(Float64(year), years[1], years[end])
    itp = linear_interpolation(years, vals; extrapolation_bc=Flat())
    return itp(y)
end

"""
    age_lookup(arr, age, default)

Look up a value from a 0:MAX_AGE array (1-indexed by int_age_clip).
"""
function age_lookup(arr::Vector{Float64}, age::Real, default::Float64=0.0)
    isempty(arr) && return default
    idx = int_age_clip(age)
    idx > length(arr) && return default
    return arr[idx]
end

"""
    interp_to_ages(knots_ages, knots_vals)

Interpolate sparse age knots into a full 0:MAX_AGE array.
"""
function interp_to_ages(knots_ages::Vector{Float64}, knots_vals::Vector{Float64})
    out = zeros(MAX_AGE + 1)
    isempty(knots_ages) && return out
    itp = linear_interpolation(knots_ages, knots_vals; extrapolation_bc=Flat())
    for age in 0:MAX_AGE
        a = clamp(Float64(age), knots_ages[1], knots_ages[end])
        out[age + 1] = itp(a)
    end
    return out
end

"""
    draw_debut_age(rng, pars)

Sample a sexual debut age from the debut age distribution.
"""
function draw_debut_age(rng::AbstractRNG, pars::FPPars)
    if isempty(pars.debut_ages) || isempty(pars.debut_probs)
        return 15.0 + rand(rng) * 10.0  # uniform 15-25
    end
    r = rand(rng)
    cum = 0.0
    for i in eachindex(pars.debut_probs)
        cum += pars.debut_probs[i]
        if r < cum
            return pars.debut_ages[i]
        end
    end
    return pars.debut_ages[end]
end
