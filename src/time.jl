"""
Time system for Starsim.jl.

Mirrors Python starsim's `time.py`. Provides duration/rate parameter types
with unit conversions and a Timeline that tracks simulation progress.
"""

# ============================================================================
# Duration types
# ============================================================================

"""
    Duration{U} <: AbstractDuration

A duration with a time unit. The value is stored in the unit specified,
and converted to simulation timestep units via `to_dt`.

# Fields
- `value::Float64` — numeric value in the specified unit
- `unit::Symbol` — one of `:days`, `:weeks`, `:months`, `:years`
"""
struct Duration <: AbstractDuration
    value::Float64
    unit::Symbol

    function Duration(value::Real, unit::Symbol)
        unit in (:days, :weeks, :months, :years) ||
            error("Unknown duration unit: $unit. Use :days, :weeks, :months, or :years")
        new(Float64(value), unit)
    end
end

"""Duration in days."""
days(v::Real) = Duration(v, :days)

"""Duration in weeks."""
weeks(v::Real) = Duration(v, :weeks)

"""Duration in months (1 month = 1/12 year)."""
months(v::Real) = Duration(v, :months)

"""Duration in years."""
years(v::Real) = Duration(v, :years)

Base.show(io::IO, d::Duration) = print(io, "$(d.value) $(d.unit)")

"""
    to_years(d::Duration)

Convert a duration to years.
"""
function to_years(d::Duration)
    d.unit == :years  && return d.value
    d.unit == :months && return d.value / 12.0
    d.unit == :weeks  && return d.value / 52.1775
    d.unit == :days   && return d.value / 365.25
end

"""
    to_days(d::Duration)

Convert a duration to days.
"""
function to_days(d::Duration)
    return to_years(d) * 365.25
end

"""
    to_dt(d::Duration, dt::Float64)

Convert a duration to simulation timestep units.
`dt` is the simulation timestep in years.
"""
function to_dt(d::Duration, dt::Float64)
    return to_years(d) / dt
end

export Duration, days, weeks, months, years, to_years, to_days, to_dt

# ============================================================================
# Rate types
# ============================================================================

"""
    Rate <: AbstractRate

A rate parameter with time units. Rates can be converted to
per-timestep probabilities via `to_prob`.

# Fields
- `value::Float64` — rate value
- `unit::Symbol` — time unit (`:perday`, `:perweek`, `:permonth`, `:peryear`)
"""
struct Rate <: AbstractRate
    value::Float64
    unit::Symbol

    function Rate(value::Real, unit::Symbol)
        unit in (:perday, :perweek, :permonth, :peryear) ||
            error("Unknown rate unit: $unit")
        new(Float64(value), unit)
    end
end

"""Rate per day."""
perday(v::Real) = Rate(v, :perday)

"""Rate per week."""
perweek(v::Real) = Rate(v, :perweek)

"""Rate per month."""
permonth(v::Real) = Rate(v, :permonth)

"""Rate per year."""
peryear(v::Real) = Rate(v, :peryear)

Base.show(io::IO, r::Rate) = print(io, "$(r.value) $(r.unit)")

"""
    to_peryear(r::Rate)

Convert a rate to per-year.
"""
function to_peryear(r::Rate)
    r.unit == :peryear  && return r.value
    r.unit == :permonth && return r.value * 12.0
    r.unit == :perweek  && return r.value * 52.1775
    r.unit == :perday   && return r.value * 365.25
end

"""
    to_prob(r::Rate, dt::Float64)

Convert a rate to a per-timestep probability using the
exponential formula: `p = 1 - exp(-rate * dt)`.
`dt` is the simulation timestep in years.
"""
function to_prob(r::Rate, dt::Float64)
    rate_per_year = to_peryear(r)
    return 1.0 - exp(-rate_per_year * dt)
end

"""
    to_prob(val::Real, dt::Float64)

For scalar beta values (already a probability), return as-is multiplied by dt.
"""
function to_prob(val::Real, dt::Float64)
    return Float64(val) * dt
end

export Rate, perday, perweek, permonth, peryear, to_peryear, to_prob

# ============================================================================
# Timeline
# ============================================================================

"""
    Timeline

Tracks simulation time, providing the time vector, current index,
and current time value.

# Fields
- `start::Float64` — simulation start (years)
- `stop::Float64` — simulation end (years)
- `dt::Float64` — timestep (years)
- `npts::Int` — number of timepoints
- `tvec::Vector{Float64}` — time vector
- `ti::Int` — current time index (1-based)
"""
mutable struct Timeline
    start::Float64
    stop::Float64
    dt::Float64
    npts::Int
    tvec::Vector{Float64}
    ti::Int
end

"""
    Timeline(; start=0.0, stop=10.0, dt=1.0)

Create a timeline from start to stop with the given timestep.
All values are in years by default.

# Example
```julia
t = Timeline(start=2020.0, stop=2030.0, dt=1/12)
```
"""
function Timeline(; start::Real=0.0, stop::Real=10.0, dt::Union{Real,Duration}=1.0)
    dt_years = dt isa Duration ? to_years(dt) : Float64(dt)
    start_f = Float64(start)
    stop_f = Float64(stop)
    tvec = collect(start_f:dt_years:stop_f)
    if isempty(tvec) || tvec[end] < stop_f - dt_years/2
        push!(tvec, stop_f)
    end
    npts = length(tvec)
    return Timeline(start_f, stop_f, dt_years, npts, tvec, 1)
end

"""Current simulation time."""
now(t::Timeline) = t.tvec[t.ti]

"""Advance the timeline by one step."""
function advance!(t::Timeline)
    t.ti += 1
    return t
end

"""Reset timeline to the start."""
function reset!(t::Timeline)
    t.ti = 1
    return t
end

"""Check if the simulation is complete."""
is_done(t::Timeline) = t.ti > t.npts

Base.show(io::IO, t::Timeline) = print(io, "Timeline($(t.start)→$(t.stop), dt=$(t.dt), ti=$(t.ti)/$(t.npts))")
Base.length(t::Timeline) = t.npts

export Timeline, now, advance!, is_done
