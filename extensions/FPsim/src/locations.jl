"""
Location-specific demographic data for FPsim.
Supports generic defaults and CSV-based data loading for Kenya and other locations.
"""

"""Valid location names."""
const VALID_LOCATIONS = [:generic, :kenya, :senegal, :ethiopia]

"""
    load_location_data(location::Symbol=:generic) → FPPars

Load location-specific demographic parameters.
"""
function load_location_data(location::Symbol=:generic)
    if location == :generic
        return _load_generic_data()
    else
        loc_dir = joinpath(DATA_DIR, string(location))
        if isdir(loc_dir)
            return _load_csv_location(location)
        else
            @warn "Location data directory not found for $location, using generic"
            return _load_generic_data()
        end
    end
end

# ============================================================================
# Generic (hardcoded) defaults
# ============================================================================

function _load_generic_data()
    pars = FPPars()

    # Age-specific fecundity (0-99)
    pars.age_fecundity = zeros(MAX_AGE + 1)
    for age in MIN_AGE:MAX_AGE_PREG
        idx = age + 1
        if age < 20
            pars.age_fecundity[idx] = 0.3
        elseif age < 25
            pars.age_fecundity[idx] = 0.8
        elseif age < 30
            pars.age_fecundity[idx] = 1.0
        elseif age < 35
            pars.age_fecundity[idx] = 0.9
        elseif age < 40
            pars.age_fecundity[idx] = 0.6
        elseif age < 45
            pars.age_fecundity[idx] = 0.2
        else
            pars.age_fecundity[idx] = 0.05
        end
    end

    # Sexual activity by age
    pars.sexual_activity = zeros(MAX_AGE + 1)
    for age in MIN_AGE:65
        idx = age + 1
        if age < 20
            pars.sexual_activity[idx] = 0.4
        elseif age < 50
            pars.sexual_activity[idx] = 0.8
        else
            pars.sexual_activity[idx] = 0.3
        end
    end

    # Death rates (crude, per year)
    pars.death_rate_ages = Float64[0, 1, 5, 15, 50, 70, 99]
    pars.death_rate_female = Float64[0.05, 0.005, 0.001, 0.002, 0.01, 0.05, 0.2]
    pars.death_rate_male = Float64[0.06, 0.006, 0.001, 0.003, 0.015, 0.07, 0.25]

    # Miscarriage rates by age
    pars.miscarriage_rates = zeros(MAX_AGE + 1)
    for age in MIN_AGE:MAX_AGE_PREG
        idx = age + 1
        if age < 20
            pars.miscarriage_rates[idx] = 0.15
        elseif age < 35
            pars.miscarriage_rates[idx] = 0.12
        elseif age < 40
            pars.miscarriage_rates[idx] = 0.20
        else
            pars.miscarriage_rates[idx] = 0.35
        end
    end

    # Fecundity ratio for nulliparous women (1.0 = no reduction)
    pars.fecundity_ratio_nullip = ones(MAX_AGE + 1)

    # Maternal mortality
    pars.maternal_mort_probs = Float64[0.0003]
    pars.maternal_mort_years = Float64[2000.0]

    # Stillbirth probability
    pars.stillbirth_probs = Float64[0.02]
    pars.stillbirth_years = Float64[2000.0]

    # Infant mortality
    pars.infant_mort_probs = Float64[0.04]
    pars.infant_mort_years = Float64[2000.0]

    # Abortion and twins
    pars.abortion_prob = 0.08
    pars.twins_prob = 0.012

    return pars
end

# ============================================================================
# CSV-based loading (Kenya and other locations)
# ============================================================================

function _load_csv_location(location::Symbol)
    pars = FPPars()
    shared_dir = joinpath(DATA_DIR, "shared")
    loc_dir = joinpath(DATA_DIR, string(location))

    # --- Shared data ---
    _load_shared_data!(pars, shared_dir)

    # --- Location-specific data ---
    _load_location_overrides!(pars, loc_dir)

    return pars
end

function _load_shared_data!(pars::FPPars, shared_dir::AbstractString)
    # Age fecundity (values are percentages 0-100, convert to 0-1)
    af_file = joinpath(shared_dir, "age_fecundity.csv")
    if isfile(af_file)
        df = CSV.read(af_file, DataFrame)
        pars.age_fecundity = interp_to_ages(Float64.(df.bin), Float64.(df.f) ./ 100.0)
    end

    # Fecundity ratio for nulliparous women (already 0-1)
    fn_file = joinpath(shared_dir, "fecundity_ratio_nullip.csv")
    if isfile(fn_file)
        df = CSV.read(fn_file, DataFrame)
        pars.fecundity_ratio_nullip = interp_to_ages(Float64.(df.age), Float64.(df.prob))
    end

    # Miscarriage rates (already 0-1)
    mis_file = joinpath(shared_dir, "miscarriage.csv")
    if isfile(mis_file)
        df = CSV.read(mis_file, DataFrame)
        pars.miscarriage_rates = interp_to_ages(Float64.(df.age), Float64.(df.prob))
    end

    return pars
end

function _load_location_overrides!(pars::FPPars, loc_dir::AbstractString)
    # Sexual activity (percentages 0-100 → 0-1)
    sa_file = joinpath(loc_dir, "sexually_active.csv")
    if isfile(sa_file)
        df = CSV.read(sa_file, DataFrame)
        pars.sexual_activity = interp_to_ages(Float64.(df.age), Float64.(df.probs) ./ 100.0)
    else
        # Default sexual activity
        pars.sexual_activity = zeros(MAX_AGE + 1)
        for age in MIN_AGE:65
            idx = age + 1
            pars.sexual_activity[idx] = age < 20 ? 0.4 : (age < 50 ? 0.8 : 0.3)
        end
    end

    # Debut age distribution
    da_file = joinpath(loc_dir, "debut_age.csv")
    if isfile(da_file)
        df = CSV.read(da_file, DataFrame)
        pars.debut_ages = Float64.(df.age)
        pars.debut_probs = Float64.(df.probs)
        # Normalize
        s = sum(pars.debut_probs)
        s > 0 && (pars.debut_probs ./= s)
    end

    # Scalar probabilities (abortion, twins)
    sc_file = joinpath(loc_dir, "scalar_probs.csv")
    if isfile(sc_file)
        df = CSV.read(sc_file, DataFrame)
        for row in eachrow(df)
            if row.param == "abortion_prob"
                pars.abortion_prob = Float64(row.prob)
            elseif row.param == "twins_prob"
                pars.twins_prob = Float64(row.prob)
            end
        end
    end

    # Maternal mortality (per 100,000 → probability)
    mm_file = joinpath(loc_dir, "maternal_mortality.csv")
    if isfile(mm_file)
        df = CSV.read(mm_file, DataFrame)
        pars.maternal_mort_years = Float64.(df.year)
        pars.maternal_mort_probs = Float64.(df.probs) ./ 100_000.0
    end

    # Infant mortality (per 1,000 → probability)
    im_file = joinpath(loc_dir, "infant_mortality.csv")
    if isfile(im_file)
        df = CSV.read(im_file, DataFrame)
        pars.infant_mort_years = Float64.(df.year)
        pars.infant_mort_probs = Float64.(df.probs) ./ 1000.0
    end

    # Stillbirth rates (per 1,000 → probability)
    sb_file = joinpath(loc_dir, "stillbirths.csv")
    if isfile(sb_file)
        df = CSV.read(sb_file, DataFrame)
        pars.stillbirth_years = Float64.(df.year)
        pars.stillbirth_probs = Float64.(df.probs) ./ 1000.0
    end

    # LAM (lactational amenorrhea) rates by month postpartum
    lam_file = joinpath(loc_dir, "lam.csv")
    if isfile(lam_file)
        df = CSV.read(lam_file, DataFrame)
        pars.lam_months = Int.(df.month)
        pars.lam_rates = Float64.(df.rate)
    end

    # Breastfeeding statistics
    bf_file = joinpath(loc_dir, "bf_stats.csv")
    if isfile(bf_file)
        df = CSV.read(bf_file, DataFrame)
        for row in eachrow(df)
            if row.parameter == "mean"
                pars.dur_breastfeeding_mean = Float64(row.value)
            elseif row.parameter == "sd"
                pars.dur_breastfeeding_std = Float64(row.value)
            end
        end
    end

    # Postpartum sexual activity
    pp_file = joinpath(loc_dir, "sexually_active_pp.csv")
    if isfile(pp_file)
        df = CSV.read(pp_file, DataFrame)
        pars.pp_months = Int.(df.month)
        pars.pp_percent_active = Float64.(df.probs)
    end

    # Birth spacing preferences
    sp_file = joinpath(loc_dir, "birth_spacing_pref.csv")
    if isfile(sp_file)
        df = CSV.read(sp_file, DataFrame)
        pars.spacing_months = Float64.(df.month)
        pars.spacing_weights = Float64.(df.weights)
    end

    # Background mortality by age/sex
    mort_file = joinpath(loc_dir, "mortality_prob.csv")
    if isfile(mort_file)
        df = CSV.read(mort_file, DataFrame)
        pars.death_rate_ages = Float64.(df.age)
        pars.death_rate_female = Float64.(df.female)
        pars.death_rate_male = Float64.(df.male)
    end

    # Age pyramid for initialization
    pyr_file = joinpath(loc_dir, "age_pyramid.csv")
    if isfile(pyr_file)
        df = CSV.read(pyr_file, DataFrame)
        pars.age_pyramid_ages = Float64.(df.age)
        pars.age_pyramid_male = Float64.(df.male)
        pars.age_pyramid_female = Float64.(df.female)
    end

    # Fill defaults for empty arrays
    if isempty(pars.age_fecundity)
        pars.age_fecundity = zeros(MAX_AGE + 1)
    end
    if isempty(pars.fecundity_ratio_nullip)
        pars.fecundity_ratio_nullip = ones(MAX_AGE + 1)
    end
    if isempty(pars.miscarriage_rates)
        pars.miscarriage_rates = zeros(MAX_AGE + 1)
    end
    if isempty(pars.stillbirth_probs)
        pars.stillbirth_probs = Float64[0.02]
        pars.stillbirth_years = Float64[2000.0]
    end
    if isempty(pars.maternal_mort_probs)
        pars.maternal_mort_probs = Float64[0.0003]
        pars.maternal_mort_years = Float64[2000.0]
    end
    if isempty(pars.infant_mort_probs)
        pars.infant_mort_probs = Float64[0.04]
        pars.infant_mort_years = Float64[2000.0]
    end

    return pars
end
