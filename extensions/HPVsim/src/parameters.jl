"""
Default parameters for HPVsim.jl.

Contains genotype-specific natural history parameters ported from
Python hpvsim/defaults.py and hpvsim/parameters.py.

Two progression models are supported:
- **Rate-based** (default): fixed annual transition probabilities per CIN stage.
- **Duration-based** (Python reference): at infection, sample pre-CIN and CIN
  durations from LogNormal distributions. CIN development probability is
  computed via a logistic function (`logf2`). Cancer probability uses the
  integral of that function (`cin_integral` method).
"""

# ============================================================================
# Logistic progression functions (from Python hpvsim/utils.py)
# ============================================================================

"""
    logf2(x, k, x_infl; ttc=25.0, y_max=1.0)

Logistic function constrained to pass through (0,0) and (ttc, y_max).
Ported from Python hpvsim `utils.logf2`.

# Arguments
- `x` — duration (years) or array of durations
- `k` — growth rate parameter
- `x_infl` — inflection point
- `ttc` — time to cancer (x value at which curve reaches y_max)
- `y_max` — upper bound
"""
function logf2(x::Real, k::Real, x_infl::Real; ttc::Real=25.0, y_max::Real=1.0)
    u_asymp, l_asymp = _get_asymptotes(k, x_infl, ttc)
    val = l_asymp + (u_asymp - l_asymp) / (1.0 + exp(-k * (x - x_infl)))
    return clamp(val, 0.0, Float64(y_max))
end

function logf2(x::AbstractVector{<:Real}, k::Real, x_infl::Real; ttc::Real=25.0, y_max::Real=1.0)
    return [logf2(xi, k, x_infl; ttc=ttc, y_max=y_max) for xi in x]
end

"""Compute asymptotes for logf2 (s=1 case of logf3). Numerically stable form."""
function _get_asymptotes(k::Real, x_infl::Real, ttc::Real)
    alpha = 1.0 + exp(k * x_infl)
    beta  = 1.0 + exp(-k * (ttc - x_infl))
    l_asymp = beta / (beta - alpha)
    u_asymp = (alpha - 1.0) * beta / (alpha - beta)
    return u_asymp, l_asymp
end

"""
    indef_intlogf2(x, k, x_infl; ttc=25.0, y_max=1.0)

Indefinite integral of `logf2`. Used internally by `intlogf2`.
"""
function indef_intlogf2(x::Real, k::Real, x_infl::Real; ttc::Real=25.0, y_max::Real=1.0)
    u_asymp, l_asymp = _get_asymptotes(k, x_infl, ttc)
    # Integrate: l + (u-l)/(1+exp(-k*(t-c))) dt
    # = l*t + (u-l)/k * log(1 + exp(k*(t-c))) + const
    # Use numerically stable form
    val = l_asymp * x + (u_asymp - l_asymp) / k * log(1.0 + exp(k * (x - x_infl)))
    return val * y_max
end

"""
    intlogf2(upper, k, x_infl; ttc=25.0, y_max=1.0)

Definite integral of `logf2` from 0 to `upper`. Values beyond `ttc`
contribute linearly (severity = 1). Ported from Python hpvsim `utils.intlogf2`.
"""
function intlogf2(upper::Real, k::Real, x_infl::Real; ttc::Real=25.0, y_max::Real=1.0)
    lim = min(Float64(ttc), Float64(upper))
    val_at_0   = indef_intlogf2(0.0, k, x_infl; ttc=ttc, y_max=y_max)
    val_at_lim = indef_intlogf2(lim, k, x_infl; ttc=ttc, y_max=y_max)
    integral = val_at_lim - val_at_0
    if upper > ttc
        integral += Float64(upper) - Float64(ttc)  # excess beyond ttc
    end
    return integral
end

"""
    compute_cin_prob(dur_precin, rel_sev, k, x_infl; ttc=25.0)

Compute CIN development probability given pre-CIN infection duration.
Maps to Python `compute_severity` with `logf2` form.
"""
function compute_cin_prob(dur_precin::Real, rel_sev::Real, k::Real, x_infl::Real; ttc::Real=25.0)
    t_eff = dur_precin * rel_sev
    return logf2(t_eff, k, x_infl; ttc=ttc)
end

"""
    compute_cancer_prob(dur_cin, rel_sev, k, x_infl, transform_prob; ttc=25.0)

Compute cancer probability from CIN duration using the cin-integral method.
Maps to Python `compute_severity` with `cin_integral` method.
"""
function compute_cancer_prob(dur_cin::Real, rel_sev::Real, k::Real, x_infl::Real,
                             transform_prob::Real; ttc::Real=25.0)
    t_eff = dur_cin * rel_sev
    sev = intlogf2(t_eff, k, x_infl; ttc=ttc)
    return 1.0 - (1.0 - transform_prob)^(sev^2)
end

"""
    sample_lognormal_duration(rng, par1, par2)

Sample a duration from a LogNormal distribution parameterized as in Python
hpvsim: `par1` is the scale (mean of the actual distribution when σ is small),
`par2` is the shape parameter σ (std dev of underlying normal).
Uses `scipy.stats.lognorm(s=par2, scale=par1)` convention.
"""
function sample_lognormal_duration(rng, par1::Real, par2::Real)
    par1 <= 0.0 && return 0.0
    par2 <= 0.0 && return Float64(par1)
    # scipy lognorm(s=sigma, scale=scale): underlying normal has μ=log(scale), σ=s
    mu = log(Float64(par1))
    sigma = Float64(par2)
    return rand(rng, Distributions.LogNormal(mu, sigma))
end

# ============================================================================
# Genotype parameter structure
# ============================================================================

"""
    GenotypeParams

Natural history parameters for a single HPV genotype. Supports both rate-based
(simple) and duration-based (Python-reference) progression models.

# Rate-based fields (simple per-timestep model)
- `prog_rate_cin1..3` — annual progression probabilities
- `clearance_rate_inf..cin3` — annual clearance probabilities

# Duration-based fields (Python hpvsim reference model)
- `dur_precin_par1/par2` — LogNormal parameters for pre-CIN duration
- `dur_cin_par1/par2` — LogNormal parameters for CIN duration
- `cin_fn_k/x_infl/ttc` — logf2 CIN development function parameters
- `cancer_fn_transform_prob` — cancer transformation probability
"""
Base.@kwdef struct GenotypeParams
    name::Symbol

    # Relative transmissibility (HPV-16 = 1.0 reference)
    rel_beta::Float64               = 1.0

    # --- Rate-based progression parameters ---
    dur_precin_mean::Float64        = 3.0     # Mean pre-CIN infectious period (years)
    dur_precin_std::Float64         = 2.0     # Std dev of pre-CIN duration
    dur_cin_mean::Float64           = 5.0     # Mean CIN stage duration (years)
    dur_cin_std::Float64            = 3.0     # Std dev of CIN duration
    prog_rate_cin1::Float64         = 0.10    # Annual P(CIN1 → CIN2)
    prog_rate_cin2::Float64         = 0.05    # Annual P(CIN2 → CIN3)
    prog_rate_cin3::Float64         = 0.03    # Annual P(CIN3 → Cancer)
    clearance_rate_inf::Float64     = 0.25    # Annual P(clearance from infection)
    clearance_rate_cin1::Float64    = 0.15    # Annual P(clearance from CIN1)
    clearance_rate_cin2::Float64    = 0.10    # Annual P(clearance from CIN2)
    clearance_rate_cin3::Float64    = 0.05    # Annual P(clearance from CIN3)

    # Immunity
    own_imm::Float64                = 0.90    # Same-genotype immunity efficacy
    sero_prob::Float64              = 0.75    # Probability of seroconversion

    # --- Duration-based progression parameters (Python hpvsim reference) ---
    dur_precin_par1::Float64        = 3.0     # LogNormal par1 for pre-CIN duration
    dur_precin_par2::Float64        = 9.0     # LogNormal par2 (variance)
    dur_cin_par1::Float64           = 5.0     # LogNormal par1 for CIN duration
    dur_cin_par2::Float64           = 20.0    # LogNormal par2 (variance)
    cin_fn_k::Float64               = 0.3     # logf2 growth rate
    cin_fn_x_infl::Float64          = 0.0     # logf2 inflection point
    cin_fn_ttc::Float64             = 50.0    # logf2 time-to-cancer
    cancer_fn_transform_prob::Float64 = 2.0e-3 # Cancer transformation probability

    # Cancer outcomes
    cancer_rate::Float64            = 0.005   # Annual P(CIN3 → Cancer) — separate from CIN2→CIN3
    cancer_mortality_rate::Float64  = 0.05    # Annual cancer mortality rate
end

# ============================================================================
# Default genotype parameters (from Python hpvsim/defaults.py)
# ============================================================================

"""Default parameters for HPV-16 (high-risk, most oncogenic).

Rate-based CIN rates calibrated to match effective CIN dynamics of
Python hpvsim's duration-based model (logf2 severity + lognormal durations).
"""
const HPV16_PARAMS = GenotypeParams(
    name                     = :hpv16,
    rel_beta                 = 1.0,
    dur_precin_mean          = 3.0,
    dur_precin_std           = 2.0,
    dur_cin_mean             = 5.0,
    dur_cin_std              = 3.0,
    prog_rate_cin1           = 0.08,
    prog_rate_cin2           = 0.04,
    prog_rate_cin3           = 0.025,
    clearance_rate_inf       = 0.25,
    clearance_rate_cin1      = 0.18,
    clearance_rate_cin2      = 0.12,
    clearance_rate_cin3      = 0.06,
    own_imm                  = 0.90,
    sero_prob                = 0.75,
    dur_precin_par1          = 3.0,
    dur_precin_par2          = 9.0,
    dur_cin_par1             = 5.0,
    dur_cin_par2             = 20.0,
    cin_fn_k                 = 0.3,
    cin_fn_x_infl            = 0.0,
    cin_fn_ttc               = 50.0,
    cancer_fn_transform_prob = 2.0e-3,
    cancer_rate              = 0.005,
    cancer_mortality_rate    = 0.065,
)

"""Default parameters for HPV-18 (high-risk)."""
const HPV18_PARAMS = GenotypeParams(
    name                     = :hpv18,
    rel_beta                 = 0.75,
    dur_precin_mean          = 2.5,
    dur_precin_std           = 1.5,
    dur_cin_mean             = 5.0,
    dur_cin_std              = 3.0,
    prog_rate_cin1           = 0.065,
    prog_rate_cin2           = 0.032,
    prog_rate_cin3           = 0.017,
    clearance_rate_inf       = 0.30,
    clearance_rate_cin1      = 0.22,
    clearance_rate_cin2      = 0.14,
    clearance_rate_cin3      = 0.07,
    own_imm                  = 0.90,
    sero_prob                = 0.56,
    dur_precin_par1          = 2.5,
    dur_precin_par2          = 9.0,
    dur_cin_par1             = 5.0,
    dur_cin_par2             = 20.0,
    cin_fn_k                 = 0.25,
    cin_fn_x_infl            = 0.0,
    cin_fn_ttc               = 50.0,
    cancer_fn_transform_prob = 2.0e-3,
    cancer_rate              = 0.003,
    cancer_mortality_rate    = 0.065,
)

"""Default parameters for high-risk types in 9-valent vaccine (grouped: 31, 33, 45, 52, 58)."""
const HI5_PARAMS = GenotypeParams(
    name                     = :hi5,
    rel_beta                 = 0.90,
    dur_precin_mean          = 2.5,
    dur_precin_std           = 1.5,
    dur_cin_mean             = 4.5,
    dur_cin_std              = 3.0,
    prog_rate_cin1           = 0.056,
    prog_rate_cin2           = 0.028,
    prog_rate_cin3           = 0.012,
    clearance_rate_inf       = 0.35,
    clearance_rate_cin1      = 0.24,
    clearance_rate_cin2      = 0.17,
    clearance_rate_cin3      = 0.085,
    own_imm                  = 0.90,
    sero_prob                = 0.60,
    dur_precin_par1          = 2.5,
    dur_precin_par2          = 9.0,
    dur_cin_par1             = 4.5,
    dur_cin_par2             = 20.0,
    cin_fn_k                 = 0.2,
    cin_fn_x_infl            = 0.0,
    cin_fn_ttc               = 50.0,
    cancer_fn_transform_prob = 1.5e-3,
    cancer_rate              = 0.002,
    cancer_mortality_rate    = 0.055,
)

"""Default parameters for other high-risk types not in 9-valent vaccine (grouped: 35, 39, 51, 56, 59)."""
const OHR_PARAMS = GenotypeParams(
    name                     = :ohr,
    rel_beta                 = 0.70,
    dur_precin_mean          = 2.0,
    dur_precin_std           = 1.5,
    dur_cin_mean             = 4.0,
    dur_cin_std              = 2.5,
    prog_rate_cin1           = 0.04,
    prog_rate_cin2           = 0.02,
    prog_rate_cin3           = 0.008,
    clearance_rate_inf       = 0.40,
    clearance_rate_cin1      = 0.30,
    clearance_rate_cin2      = 0.22,
    clearance_rate_cin3      = 0.12,
    own_imm                  = 0.90,
    sero_prob                = 0.50,
    dur_precin_par1          = 2.5,
    dur_precin_par2          = 9.0,
    dur_cin_par1             = 4.5,
    dur_cin_par2             = 20.0,
    cin_fn_k                 = 0.2,
    cin_fn_x_infl            = 0.0,
    cin_fn_ttc               = 50.0,
    cancer_fn_transform_prob = 1.5e-3,
    cancer_rate              = 0.001,
    cancer_mortality_rate    = 0.055,
)

"""Default parameters for low-risk types (HPV-6, HPV-11; cause warts, not cancer)."""
const LR_PARAMS = GenotypeParams(
    name                     = :lr,
    rel_beta                 = 0.60,
    dur_precin_mean          = 1.5,
    dur_precin_std           = 1.0,
    dur_cin_mean             = 3.0,
    dur_cin_std              = 2.0,
    prog_rate_cin1           = 0.016,
    prog_rate_cin2           = 0.004,
    prog_rate_cin3           = 0.0008,
    clearance_rate_inf       = 0.50,
    clearance_rate_cin1      = 0.42,
    clearance_rate_cin2      = 0.30,
    clearance_rate_cin3      = 0.18,
    own_imm                  = 0.90,
    sero_prob                = 0.40,
    dur_precin_par1          = 2.0,
    dur_precin_par2          = 10.0,
    dur_cin_par1             = 0.1,
    dur_cin_par2             = 0.1,
    cin_fn_k                 = 0.01,
    cin_fn_x_infl            = 0.0,
    cin_fn_ttc               = 100.0,
    cancer_fn_transform_prob = 1.0e-6,
    cancer_rate              = 0.0001,
    cancer_mortality_rate    = 0.02,
)

"""Aggregate parameters for all high-risk types (31–59). Use instead of hi5+ohr."""
const HR_PARAMS = GenotypeParams(
    name                     = :hr,
    rel_beta                 = 0.90,
    dur_precin_mean          = 2.0,
    dur_precin_std           = 1.5,
    dur_cin_mean             = 4.0,
    dur_cin_std              = 2.5,
    prog_rate_cin1           = 0.048,
    prog_rate_cin2           = 0.024,
    prog_rate_cin3           = 0.010,
    clearance_rate_inf       = 0.37,
    clearance_rate_cin1      = 0.26,
    clearance_rate_cin2      = 0.19,
    clearance_rate_cin3      = 0.10,
    own_imm                  = 0.90,
    sero_prob                = 0.60,
    dur_precin_par1          = 2.0,
    dur_precin_par2          = 10.0,
    dur_cin_par1             = 4.0,
    dur_cin_par2             = 4.0,
    cin_fn_k                 = 0.15,
    cin_fn_x_infl            = 10.0,
    cin_fn_ttc               = 50.0,
    cancer_fn_transform_prob = 1.0e-3,
    cancer_rate              = 0.002,
    cancer_mortality_rate    = 0.05,
)

# ============================================================================
# Individual genotype parameters (expanded registry)
# ============================================================================

"""HPV-31 (high-risk, 9-valent vaccine)."""
const HPV31_PARAMS = GenotypeParams(
    name = :hpv31, rel_beta = 0.85,
    dur_precin_mean = 2.5, dur_precin_std = 1.5, dur_cin_mean = 4.5, dur_cin_std = 3.0,
    prog_rate_cin1 = 0.056, prog_rate_cin2 = 0.028, prog_rate_cin3 = 0.012,
    clearance_rate_inf = 0.35, clearance_rate_cin1 = 0.24, clearance_rate_cin2 = 0.17, clearance_rate_cin3 = 0.085,
    own_imm = 0.90, sero_prob = 0.60,
    dur_precin_par1 = 2.5, dur_precin_par2 = 9.0, dur_cin_par1 = 4.5, dur_cin_par2 = 20.0,
    cin_fn_k = 0.2, cin_fn_x_infl = 0.0, cin_fn_ttc = 50.0,
    cancer_fn_transform_prob = 1.5e-3, cancer_rate = 0.002, cancer_mortality_rate = 0.055,
)

"""HPV-33 (high-risk, 9-valent vaccine)."""
const HPV33_PARAMS = GenotypeParams(
    name = :hpv33, rel_beta = 0.85,
    dur_precin_mean = 2.5, dur_precin_std = 1.5, dur_cin_mean = 4.5, dur_cin_std = 3.0,
    prog_rate_cin1 = 0.056, prog_rate_cin2 = 0.028, prog_rate_cin3 = 0.012,
    clearance_rate_inf = 0.35, clearance_rate_cin1 = 0.24, clearance_rate_cin2 = 0.17, clearance_rate_cin3 = 0.085,
    own_imm = 0.90, sero_prob = 0.60,
    dur_precin_par1 = 2.5, dur_precin_par2 = 9.0, dur_cin_par1 = 4.5, dur_cin_par2 = 20.0,
    cin_fn_k = 0.2, cin_fn_x_infl = 0.0, cin_fn_ttc = 50.0,
    cancer_fn_transform_prob = 1.5e-3, cancer_rate = 0.002, cancer_mortality_rate = 0.055,
)

"""HPV-45 (high-risk, 9-valent vaccine)."""
const HPV45_PARAMS = GenotypeParams(
    name = :hpv45, rel_beta = 0.90,
    dur_precin_mean = 2.5, dur_precin_std = 1.5, dur_cin_mean = 5.0, dur_cin_std = 3.0,
    prog_rate_cin1 = 0.064, prog_rate_cin2 = 0.032, prog_rate_cin3 = 0.015,
    clearance_rate_inf = 0.32, clearance_rate_cin1 = 0.23, clearance_rate_cin2 = 0.16, clearance_rate_cin3 = 0.07,
    own_imm = 0.90, sero_prob = 0.62,
    dur_precin_par1 = 2.5, dur_precin_par2 = 9.0, dur_cin_par1 = 5.0, dur_cin_par2 = 20.0,
    cin_fn_k = 0.22, cin_fn_x_infl = 0.0, cin_fn_ttc = 50.0,
    cancer_fn_transform_prob = 1.8e-3, cancer_rate = 0.003, cancer_mortality_rate = 0.060,
)

"""HPV-52 (high-risk, 9-valent vaccine)."""
const HPV52_PARAMS = GenotypeParams(
    name = :hpv52, rel_beta = 0.85,
    dur_precin_mean = 2.5, dur_precin_std = 1.5, dur_cin_mean = 4.5, dur_cin_std = 3.0,
    prog_rate_cin1 = 0.048, prog_rate_cin2 = 0.024, prog_rate_cin3 = 0.010,
    clearance_rate_inf = 0.37, clearance_rate_cin1 = 0.26, clearance_rate_cin2 = 0.18, clearance_rate_cin3 = 0.10,
    own_imm = 0.90, sero_prob = 0.58,
    dur_precin_par1 = 2.5, dur_precin_par2 = 9.0, dur_cin_par1 = 4.5, dur_cin_par2 = 20.0,
    cin_fn_k = 0.18, cin_fn_x_infl = 0.0, cin_fn_ttc = 50.0,
    cancer_fn_transform_prob = 1.4e-3, cancer_rate = 0.002, cancer_mortality_rate = 0.050,
)

"""HPV-58 (high-risk, 9-valent vaccine)."""
const HPV58_PARAMS = GenotypeParams(
    name = :hpv58, rel_beta = 0.85,
    dur_precin_mean = 2.5, dur_precin_std = 1.5, dur_cin_mean = 4.5, dur_cin_std = 3.0,
    prog_rate_cin1 = 0.048, prog_rate_cin2 = 0.024, prog_rate_cin3 = 0.010,
    clearance_rate_inf = 0.37, clearance_rate_cin1 = 0.26, clearance_rate_cin2 = 0.18, clearance_rate_cin3 = 0.10,
    own_imm = 0.90, sero_prob = 0.58,
    dur_precin_par1 = 2.5, dur_precin_par2 = 9.0, dur_cin_par1 = 4.5, dur_cin_par2 = 20.0,
    cin_fn_k = 0.18, cin_fn_x_infl = 0.0, cin_fn_ttc = 50.0,
    cancer_fn_transform_prob = 1.4e-3, cancer_rate = 0.002, cancer_mortality_rate = 0.050,
)

"""HPV-6 (low-risk, causes genital warts)."""
const HPV6_PARAMS = GenotypeParams(
    name = :hpv6, rel_beta = 0.65,
    dur_precin_mean = 1.5, dur_precin_std = 1.0, dur_cin_mean = 3.0, dur_cin_std = 2.0,
    prog_rate_cin1 = 0.012, prog_rate_cin2 = 0.0024, prog_rate_cin3 = 0.0004,
    clearance_rate_inf = 0.55, clearance_rate_cin1 = 0.46, clearance_rate_cin2 = 0.34, clearance_rate_cin3 = 0.22,
    own_imm = 0.90, sero_prob = 0.45,
    dur_precin_par1 = 2.0, dur_precin_par2 = 10.0, dur_cin_par1 = 0.1, dur_cin_par2 = 0.1,
    cin_fn_k = 0.01, cin_fn_x_infl = 0.0, cin_fn_ttc = 100.0,
    cancer_fn_transform_prob = 1.0e-6, cancer_rate = 0.0001, cancer_mortality_rate = 0.01,
)

"""HPV-11 (low-risk, causes genital warts)."""
const HPV11_PARAMS = GenotypeParams(
    name = :hpv11, rel_beta = 0.60,
    dur_precin_mean = 1.5, dur_precin_std = 1.0, dur_cin_mean = 3.0, dur_cin_std = 2.0,
    prog_rate_cin1 = 0.012, prog_rate_cin2 = 0.0024, prog_rate_cin3 = 0.0004,
    clearance_rate_inf = 0.52, clearance_rate_cin1 = 0.42, clearance_rate_cin2 = 0.30, clearance_rate_cin3 = 0.18,
    own_imm = 0.90, sero_prob = 0.40,
    dur_precin_par1 = 2.0, dur_precin_par2 = 10.0, dur_cin_par1 = 0.1, dur_cin_par2 = 0.1,
    cin_fn_k = 0.01, cin_fn_x_infl = 0.0, cin_fn_ttc = 100.0,
    cancer_fn_transform_prob = 1.0e-6, cancer_rate = 0.0001, cancer_mortality_rate = 0.01,
)

"""Registry of default genotype parameters (grouped + individual types)."""
const GENOTYPE_REGISTRY = Dict{Symbol, GenotypeParams}(
    # Standard grouped genotypes
    :hpv16 => HPV16_PARAMS,
    :hpv18 => HPV18_PARAMS,
    :hi5   => HI5_PARAMS,
    :ohr   => OHR_PARAMS,
    :lr    => LR_PARAMS,
    :hr    => HR_PARAMS,
    # Individual types
    :hpv31 => HPV31_PARAMS,
    :hpv33 => HPV33_PARAMS,
    :hpv45 => HPV45_PARAMS,
    :hpv52 => HPV52_PARAMS,
    :hpv58 => HPV58_PARAMS,
    :hpv6  => HPV6_PARAMS,
    :hpv11 => HPV11_PARAMS,
)

"""
    get_genotype_params(name::Symbol) → GenotypeParams

Look up default parameters for a named genotype.
"""
function get_genotype_params(name::Symbol)
    key = Symbol(lowercase(string(name)))
    haskey(GENOTYPE_REGISTRY, key) || error("Unknown genotype: $name. Available: $(keys(GENOTYPE_REGISTRY))")
    return GENOTYPE_REGISTRY[key]
end

"""
    list_genotypes() → Vector{Symbol}

Return all registered genotype names.
"""
list_genotypes() = collect(keys(GENOTYPE_REGISTRY))

"""
    is_high_risk(name::Symbol) → Bool

Check if a genotype is in the high-risk oncogenic category.
"""
function is_high_risk(name::Symbol)
    return name in (:hpv16, :hpv18, :hi5, :ohr, :hr,
                     :hpv31, :hpv33, :hpv45, :hpv52, :hpv58)
end

"""
    is_low_risk(name::Symbol) → Bool

Check if a genotype is in the low-risk category.
"""
function is_low_risk(name::Symbol)
    return name in (:lr, :hpv6, :hpv11)
end

# ============================================================================
# Default simulation parameters
# ============================================================================

"""Default base transmission probability per act."""
const DEFAULT_BETA = 0.25

"""Male-to-female transmission multiplier relative to female-to-male."""
const M2F_TRANS_RATIO = 3.69

"""Default cross-immunity matrix entries (matching Python hpvsim)."""
const DEFAULT_CROSS_IMMUNITY = Dict{Symbol, Float64}(
    :imm_init => 0.35,  # Mean initial immunity level (Python: imm_init beta(0.35, 0.025))
    :own_hr   => 0.90,  # Same genotype for grouped types (Python: own_imm_hr)
    :partial  => 0.50,  # Same risk group (Python: cross_imm_sus_high)
    :cross    => 0.30,  # Different risk group (Python: cross_imm_sus_med)
)

"""Default vaccination parameters."""
const DEFAULT_VAX_PARAMS = Dict{String, Any}(
    "bivalent_genotypes"   => [:hpv16, :hpv18],
    "quadrivalent_genotypes" => [:hpv16, :hpv18, :hpv6, :hpv11],
    "nonavalent_genotypes" => [:hpv16, :hpv18, :hpv31, :hpv33, :hpv45, :hpv52, :hpv58, :hpv6, :hpv11],
    "nonavalent_grouped"   => [:hpv16, :hpv18, :hi5, :lr],
    "dose_efficacies_1dose" => [0.85],
    "dose_efficacies_2dose" => [0.85, 0.95],
    "dose_efficacies_3dose" => [0.70, 0.90, 0.97],
)

"""Default screening parameters — sensitivity varies by test type and disease state."""
const DEFAULT_SCREENING_PARAMS = Dict{String, Any}(
    # Pap / LBC
    "pap_sensitivity_cin1"      => 0.30,
    "pap_sensitivity_cin2"      => 0.55,
    "pap_sensitivity_cin3"      => 0.75,
    "pap_sensitivity_cancer"    => 0.95,
    "pap_specificity"           => 0.97,
    # HPV DNA
    "hpv_dna_sensitivity_inf"   => 0.85,
    "hpv_dna_sensitivity_cin1"  => 0.90,
    "hpv_dna_sensitivity_cin2plus" => 0.95,
    "hpv_dna_specificity"       => 0.90,
    # VIA
    "via_sensitivity_cin1"      => 0.40,
    "via_sensitivity_cin2plus"  => 0.60,
    "via_specificity"           => 0.84,
)

"""Default treatment parameters — efficacy by treatment type and disease stage."""
const DEFAULT_TREATMENT_PARAMS = Dict{String, Any}(
    "ablation_efficacy_cin1"    => 0.90,
    "ablation_efficacy_cin2"    => 0.85,
    "ablation_efficacy_cin3"    => 0.75,
    "excision_efficacy_cin1"    => 0.95,
    "excision_efficacy_cin2"    => 0.93,
    "excision_efficacy_cin3"    => 0.90,
    "excision_efficacy_cancer"  => 0.50,
)
