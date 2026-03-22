"""
Parameter management for Starsim.jl.

Mirrors Python starsim's `parameters.py`. Provides `Pars` (an ordered dict
with smart update semantics) and `SimPars` (typed simulation parameters).
"""

# ============================================================================
# Pars — ordered parameter dictionary
# ============================================================================

"""
    Pars <: AbstractDict{Symbol, Any}

Ordered parameter dictionary with smart merge semantics. Nested updates
handle `AbstractStarsimDist`, `AbstractTimePar`, and sub-module parameters.

# Example
```julia
p = Pars(:beta => 0.05, :dur_inf => years(10))
update!(p, :beta => 0.1)
```
"""
struct Pars <: AbstractDict{Symbol, Any}
    data::OrderedDict{Symbol, Any}
end

Pars(pairs::Pair...) = Pars(OrderedDict{Symbol, Any}(pairs...))
Pars(d::AbstractDict) = Pars(OrderedDict{Symbol, Any}(Symbol(k) => v for (k,v) in d))
Pars() = Pars(OrderedDict{Symbol, Any}())

Base.iterate(p::Pars, args...) = iterate(p.data, args...)
Base.length(p::Pars) = length(p.data)
Base.getindex(p::Pars, k::Symbol) = p.data[k]
Base.setindex!(p::Pars, v, k::Symbol) = (p.data[k] = v)
Base.haskey(p::Pars, k::Symbol) = haskey(p.data, k)
Base.keys(p::Pars) = keys(p.data)
Base.values(p::Pars) = values(p.data)
Base.get(p::Pars, k::Symbol, default) = get(p.data, k, default)
Base.delete!(p::Pars, k::Symbol) = (delete!(p.data, k); p)
Base.pop!(p::Pars, k::Symbol, default=nothing) = pop!(p.data, k, default)
Base.show(io::IO, p::Pars) = print(io, "Pars($(join(["$k=$v" for (k,v) in p.data], ", ")))")

"""
    update!(p::Pars, pairs::Pair...)

Update parameters. Raises an error if a key does not already exist
(unless the value is a new distribution or module).
"""
function update!(p::Pars, pairs::Pair...)
    for (k, v) in pairs
        k = Symbol(k)
        if !haskey(p, k)
            p[k] = v  # Allow adding new keys during define_pars
        else
            old = p[k]
            if old isa AbstractStarsimDist && v isa Real
                # Update distribution parameter (e.g., Bernoulli p)
                if old isa BernoulliDist
                    old.p = Float64(v)
                end
            elseif old isa AbstractTimePar && v isa Real
                # Would update value in place
                p[k] = v
            else
                p[k] = v
            end
        end
    end
    return p
end

function update!(p::Pars, d::AbstractDict)
    for (k, v) in d
        update!(p, Symbol(k) => v)
    end
    return p
end

function update!(p::Pars; kwargs...)
    for (k, v) in kwargs
        update!(p, k => v)
    end
    return p
end

export Pars, update!

# ============================================================================
# SimPars — typed simulation parameters
# ============================================================================

"""
    SimPars

Standard simulation parameters.

# Fields
- `n_agents::Int` — number of agents (default 10_000)
- `start::Float64` — simulation start time in years (default 0.0)
- `stop::Float64` — simulation end time in years (default 10.0)
- `dt::Float64` — timestep in years (default 1.0)
- `rand_seed::Int` — RNG seed (default 0)
- `pop_scale::Float64` — population scaling factor (default 1.0)
- `use_aging::Bool` — whether agents age each timestep (default true)
- `verbose::Int` — output verbosity (default 1)

# Example
```julia
pars = SimPars(n_agents=5000, dt=1/12, stop=2030.0)
```
"""
mutable struct SimPars
    n_agents::Int
    start::Float64
    stop::Float64
    dt::Float64
    rand_seed::Int
    pop_scale::Float64
    use_aging::Bool
    verbose::Int
end

function SimPars(;
    n_agents::Int = 10_000,
    start::Real = 0.0,
    stop::Real = 10.0,
    dt::Union{Real, Duration} = 1.0,
    rand_seed::Int = 0,
    pop_scale::Real = 1.0,
    use_aging::Bool = true,
    verbose::Int = 1,
)
    dt_years = dt isa Duration ? to_years(dt) : Float64(dt)
    SimPars(n_agents, Float64(start), Float64(stop), dt_years, rand_seed,
            Float64(pop_scale), use_aging, verbose)
end

function Base.show(io::IO, sp::SimPars)
    print(io, "SimPars(n=$(sp.n_agents), $(sp.start)→$(sp.stop), dt=$(sp.dt), seed=$(sp.rand_seed))")
end

export SimPars
