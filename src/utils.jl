"""
Utility functions for Starsim.jl.
"""

"""
    standardize_netkey(key)

Standardize a network key by stripping common suffixes.
"""
function standardize_netkey(key::Union{Symbol, AbstractString})
    s = lowercase(string(key))
    for suffix in ("net", "network")
        if endswith(s, suffix) && length(s) > length(suffix)
            s = s[1:end-length(suffix)]
        end
    end
    return Symbol(s)
end

"""
    warn_starsim(msg::String)

Issue a warning with a Starsim prefix.
"""
function warn_starsim(msg::String)
    @warn "Starsim: $msg"
end

export standardize_netkey, warn_starsim

# ============================================================================
# Rate/probability conversions — matches Python ss.rate_prob / ss.time_prob
# ============================================================================

"""
    rate_prob(rate::Real, dt::Real=1.0)

Convert a rate to a probability: `p = 1 - exp(-rate * dt)`.
Matches Python `ss.rate_prob()`.
"""
rate_prob(rate::Real, dt::Real=1.0) = 1.0 - exp(-rate * dt)

"""
    prob_rate(prob::Real, dt::Real=1.0)

Convert a probability to a rate: `rate = -log(1 - p) / dt`.
Matches Python `ss.prob_rate()`.
"""
prob_rate(prob::Real, dt::Real=1.0) = -log(1.0 - clamp(prob, 0.0, 1.0 - eps())) / dt

"""
    time_prob(prob::Real, dt_old::Real, dt_new::Real)

Convert a time-dependent probability from one timestep to another:
`p_new = 1 - (1 - p)^(dt_new / dt_old)`.
Matches Python `ss.time_prob()`.
"""
time_prob(prob::Real, dt_old::Real, dt_new::Real) = 1.0 - (1.0 - clamp(prob, 0.0, 1.0))^(dt_new / dt_old)

export rate_prob, prob_rate, time_prob
