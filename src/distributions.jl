"""
Distribution wrappers for Starsim.jl.

Wraps `Distributions.jl` with deterministic seeding, per-timestep jumping,
slot-based per-agent draws, and CRN-safe pairwise random number combining,
mirroring Python starsim's `distributions.py`.
"""

# ============================================================================
# StarsimDist — wrapper around Distributions.jl
# ============================================================================

"""
    StarsimDist{D<:Distribution} <: AbstractStarsimDist

Wraps a `Distributions.jl` distribution with Starsim-specific RNG management.

# Fields
- `dist::D` — the underlying distribution
- `name::Symbol` — unique name for seeding
- `rng::StableRNG` — deterministic RNG
- `seed::UInt64` — seed derived from name + base seed
- `slots::Union{Nothing, StateVector{Int64, Vector{Int64}}}` — reference to People's slot array (for CRN)
- `initialized::Bool`

# Example
```julia
d = StarsimDist(:dur_inf, Normal(10.0, 2.0))
init_dist!(d, base_seed=42)
vals = rvs(d, 100)  # draw 100 samples
```
"""
mutable struct StarsimDist{D<:Distribution} <: AbstractStarsimDist
    dist::D
    name::Symbol
    rng::StableRNG
    seed::UInt64
    slots::Union{Nothing, StateVector{Int64, Vector{Int64}}}
    initialized::Bool
end

function StarsimDist(name::Symbol, dist::D) where {D<:Distribution}
    StarsimDist{D}(dist, name, StableRNG(0), UInt64(0), nothing, false)
end

function Base.show(io::IO, d::StarsimDist)
    print(io, "StarsimDist(:$(d.name), $(d.dist), init=$(d.initialized))")
end

"""
    init_dist!(d::StarsimDist; base_seed::Int=0, trace::String="")

Initialize the distribution's RNG from a deterministic seed derived
from the trace path and base seed.
"""
function init_dist!(d::StarsimDist; base_seed::Int=0, trace::String="")
    # Deterministic seed from name hash + base
    h = hash(isempty(trace) ? string(d.name) : trace)
    d.seed = UInt64(h) ⊻ UInt64(base_seed)
    d.rng = StableRNG(d.seed)
    d.initialized = true
    return d
end

"""
    jump_dt!(d::StarsimDist, ti::Int)

Jump the RNG to a deterministic state for timestep `ti`.
Ensures reproducibility regardless of call order within a timestep.
"""
function jump_dt!(d::StarsimDist, ti::Int)
    # Re-seed to a deterministic state for this timestep
    d.rng = StableRNG(d.seed + UInt64(ti) * UInt64(1000))
    return d
end

export StarsimDist, init_dist!, jump_dt!

# ============================================================================
# Drawing random values
# ============================================================================

"""
    rvs(d::StarsimDist, n::Int)

Draw `n` random values from the distribution.

# Example
```julia
d = StarsimDist(:test, Normal(0, 1))
init_dist!(d, base_seed=42)
vals = rvs(d, 100)
```
"""
function rvs(d::StarsimDist, n::Int)
    d.initialized || error("Distribution :$(d.name) not initialized. Call init_dist! first.")
    return rand(d.rng, d.dist, n)
end

"""
    rvs(d::StarsimDist, uids::UIDs)

Draw random values for the given UIDs. When CRN is enabled and slots are
available, draws `max(slot)+1` values and indexes by slot, ensuring that
adding/removing agents does not shift other agents' draws.
"""
function rvs(d::StarsimDist, uids::UIDs)
    if crn_enabled() && d.slots !== nothing && !isempty(uids)
        return _rvs_crn(d, uids)
    end
    return rvs(d, length(uids))
end

"""
    _rvs_crn(d::StarsimDist, uids::UIDs)

CRN-safe draw: generate `max(slot)+1` values, then index by each agent's slot.
"""
function _rvs_crn(d::StarsimDist, uids::UIDs)
    slots = d.slots
    agent_slots = slots.raw[uids.values]
    max_slot = maximum(agent_slots)
    all_draws = rand(d.rng, d.dist, max_slot)
    return all_draws[agent_slots]
end

"""
    rvs(d::StarsimDist)

Draw a single scalar random value.
"""
function rvs(d::StarsimDist)
    d.initialized || error("Distribution :$(d.name) not initialized.")
    return rand(d.rng, d.dist)
end

"""
    set_slots!(d::StarsimDist, slots::StateVector{Int64, Vector{Int64}})

Link the distribution to the People's slot state for CRN-safe draws.
"""
function set_slots!(d::StarsimDist, slots::StateVector{Int64, Vector{Int64}})
    d.slots = slots
    return d
end

export rvs, set_slots!

# ============================================================================
# Bernoulli filter — special case for boolean sampling
# ============================================================================

"""
    BernoulliDist <: AbstractStarsimDist

Bernoulli distribution wrapper with a `filter` method that returns UIDs
of agents who "pass" the trial. Central to initial prevalence seeding
and per-timestep probability sampling.

# Example
```julia
b = BernoulliDist(:init_prev, 0.01)
init_dist!(b, base_seed=42)
infected_uids = filter(b, all_uids)
```
"""
mutable struct BernoulliDist <: AbstractStarsimDist
    name::Symbol
    p::Float64
    rng::StableRNG
    seed::UInt64
    slots::Union{Nothing, StateVector{Int64, Vector{Int64}}}
    initialized::Bool
end

BernoulliDist(name::Symbol, p::Real) = BernoulliDist(name, Float64(p), StableRNG(0), UInt64(0), nothing, false)

function Base.show(io::IO, b::BernoulliDist)
    print(io, "BernoulliDist(:$(b.name), p=$(b.p))")
end

function init_dist!(b::BernoulliDist; base_seed::Int=0, trace::String="")
    h = hash(isempty(trace) ? string(b.name) : trace)
    b.seed = UInt64(h) ⊻ UInt64(base_seed)
    b.rng = StableRNG(b.seed)
    b.initialized = true
    return b
end

function jump_dt!(b::BernoulliDist, ti::Int)
    b.rng = StableRNG(b.seed + UInt64(ti) * UInt64(1000))
    return b
end

"""
    filter(b::BernoulliDist, uids::UIDs)

Return UIDs that pass a Bernoulli trial with probability `b.p`.
When CRN is enabled and slots are linked, uses slot-based indexing.
"""
function Base.filter(b::BernoulliDist, uids::UIDs)
    b.initialized || error("BernoulliDist :$(b.name) not initialized.")
    n = length(uids)
    n == 0 && return UIDs()

    if crn_enabled() && b.slots !== nothing
        agent_slots = b.slots.raw[uids.values]
        max_slot = maximum(agent_slots)
        all_rands = rand(b.rng, max_slot)
        rands = all_rands[agent_slots]
    else
        rands = rand(b.rng, n)
    end

    mask = rands .< b.p
    return UIDs(uids.values[mask])
end

"""
    rvs(b::BernoulliDist, n::Int)

Draw `n` Bernoulli samples (returns Bool vector).
"""
function rvs(b::BernoulliDist, n::Int)
    b.initialized || error("BernoulliDist :$(b.name) not initialized.")
    return rand(b.rng, n) .< b.p
end

function rvs(b::BernoulliDist, uids::UIDs)
    if crn_enabled() && b.slots !== nothing && !isempty(uids)
        agent_slots = b.slots.raw[uids.values]
        max_slot = maximum(agent_slots)
        all_rands = rand(b.rng, max_slot)
        rands = all_rands[agent_slots]
        return rands .< b.p
    end
    return rvs(b, length(uids))
end

"""
    set_slots!(b::BernoulliDist, slots::StateVector{Int64, Vector{Int64}})

Link the distribution to the People's slot state for CRN-safe draws.
"""
function set_slots!(b::BernoulliDist, slots::StateVector{Int64, Vector{Int64}})
    b.slots = slots
    return b
end

export BernoulliDist

# ============================================================================
# MultiRandom — CRN-safe pairwise random numbers for transmission
# ============================================================================

"""
    MultiRandom <: AbstractStarsimDist

Generates CRN-safe pairwise random numbers for transmission edges.
Each dimension (e.g., source, target) has its own independent `StarsimDist`.
Results are combined via XOR to produce a single Uniform(0,1) value per edge.

From the Starsim CRN paper (2409.02086v2): for an edge (i,j), compute
`u_ij = xor(u_i * u_j, u_i - u_j) / typemax(UInt64)` where u_i, u_j
are per-agent random UInt64s. This ensures adding/removing agents
doesn't shift randomness for other pairs.

# Fields
- `dists::Vector{StarsimDist}` — one distribution per dimension (Uniform(0,1))
- `name::Symbol` — identifier
- `seed::UInt64` — base seed
- `initialized::Bool`

# Example
```julia
mr = MultiRandom(:transmission)
init_dist!(mr, base_seed=42)
probs = multi_rvs(mr, source_uids, target_uids)
```
"""
mutable struct MultiRandom <: AbstractStarsimDist
    dists::Vector{StarsimDist}
    name::Symbol
    seed::UInt64
    initialized::Bool
end

function MultiRandom(name::Symbol; n_dims::Int=2)
    dists = [StarsimDist(Symbol(name, :_dim, i), Uniform(0.0, 1.0)) for i in 1:n_dims]
    MultiRandom(dists, name, UInt64(0), false)
end

function Base.show(io::IO, mr::MultiRandom)
    print(io, "MultiRandom(:$(mr.name), dims=$(length(mr.dists)), init=$(mr.initialized))")
end

function init_dist!(mr::MultiRandom; base_seed::Int=0, trace::String="")
    h = hash(isempty(trace) ? string(mr.name) : trace)
    mr.seed = UInt64(h) ⊻ UInt64(base_seed)
    for (i, d) in enumerate(mr.dists)
        # Use modular arithmetic to stay in Int range
        sub_seed = Int(mr.seed % UInt64(typemax(Int64)) ⊻ UInt64(i * 7919))
        init_dist!(d; base_seed=sub_seed,
                   trace=isempty(trace) ? string(mr.name, "_dim", i) : string(trace, "_dim", i))
    end
    mr.initialized = true
    return mr
end

function jump_dt!(mr::MultiRandom, ti::Int)
    for d in mr.dists
        jump_dt!(d, ti)
    end
    return mr
end

"""
    set_slots!(mr::MultiRandom, slots::StateVector{Int64, Vector{Int64}})

Link all underlying distributions to the People's slot state.
"""
function set_slots!(mr::MultiRandom, slots::StateVector{Int64, Vector{Int64}})
    for d in mr.dists
        set_slots!(d, slots)
    end
    return mr
end

"""
    combine_rvs(rvs_list::Vector{Vector{Float64}}) → Vector{Float64}

Combine multiple random-value vectors into a single pseudo-uniform vector
using XOR combining. Matches Python starsim's `combine_rvs`:
  1. Reinterpret each float64 as UInt64 (bit-level)
  2. Compute `xor(a * b, a - b)` in unsigned integer arithmetic (wrapping)
  3. Normalize by `typemax(UInt64)`

# Arguments
- `rvs_list` — vector of Float64 vectors, all the same length

# Returns
- `Vector{Float64}` — combined values in [0, 1)
"""
function combine_rvs(rvs_list::Vector{Vector{Float64}})
    length(rvs_list) == 0 && return Float64[]
    length(rvs_list) == 1 && return rvs_list[1]

    n = length(rvs_list[1])
    result = Vector{Float64}(undef, n)
    u = rvs_list[1]
    v = rvs_list[2]

    # Match Python: reinterpret floats as UInt64 FIRST, then do integer
    # arithmetic (wrapping multiply/subtract), then XOR and normalize.
    # Python: rand_ints = rvs.view(uint64); xor(rand_ints*rand_ints2, rand_ints-rand_ints2)
    @inbounds for i in 1:n
        a = reinterpret(UInt64, u[i])
        b = reinterpret(UInt64, v[i])
        combined = xor(a * b, a - b)  # UInt64 wrapping arithmetic
        result[i] = Float64(combined) / Float64(typemax(UInt64))
    end

    # Chain additional dimensions if more than 2
    for k in 3:length(rvs_list)
        w = rvs_list[k]
        @inbounds for i in 1:n
            a = reinterpret(UInt64, result[i])
            b = reinterpret(UInt64, w[i])
            combined = xor(a * b, a - b)
            result[i] = Float64(combined) / Float64(typemax(UInt64))
        end
    end

    return result
end

"""
    multi_rvs(mr::MultiRandom, uid_lists::Vector{UIDs}) → Vector{Float64}

Draw CRN-safe pairwise random values. Each `UIDs` vector corresponds
to one dimension of agents (e.g., sources and targets). Each dimension
draws from its own RNG, then results are XOR-combined.

# Arguments
- `mr` — MultiRandom with one dist per dimension
- `uid_lists` — one UIDs vector per dimension, all same length

# Returns
- `Vector{Float64}` — combined random values in [0, 1)
"""
function multi_rvs(mr::MultiRandom, uid_lists::AbstractVector{<:UIDs})
    mr.initialized || error("MultiRandom :$(mr.name) not initialized.")
    length(uid_lists) == length(mr.dists) ||
        error("Expected $(length(mr.dists)) UID lists, got $(length(uid_lists))")
    isempty(uid_lists[1]) && return Float64[]

    rvs_list = [rvs(mr.dists[i], uid_lists[i]) for i in 1:length(mr.dists)]
    return combine_rvs(rvs_list)
end

"""
    multi_rvs(mr::MultiRandom, sources::UIDs, targets::UIDs) → Vector{Float64}

Convenience method for 2D pairwise random values (common transmission case).
"""
function multi_rvs(mr::MultiRandom, sources::UIDs, targets::UIDs)
    return multi_rvs(mr, [sources, targets])
end

export MultiRandom, combine_rvs, multi_rvs

# ============================================================================
# Convenience constructors matching Python API
# ============================================================================

"""Create a Bernoulli distribution for prevalence seeding."""
bernoulli(; p::Real) = BernoulliDist(:bernoulli, p)
bernoulli(p::Real) = BernoulliDist(:bernoulli, p)

"""
    ss_random()

Uniform(0, 1) random distribution. Matches Python `ss.random()`.
"""
ss_random() = StarsimDist(:random, Uniform(0.0, 1.0))

"""
    ss_normal(; loc, scale)

Normal distribution wrapper. `loc` can be `Real` or `Duration`.
"""
function ss_normal(; loc::Union{Real, Duration}=0.0, scale::Real=1.0)
    if loc isa Duration
        return StarsimDist(:normal, Normal(to_years(loc), scale * to_years(Duration(1.0, loc.unit))))
    else
        return StarsimDist(:normal, Normal(Float64(loc), Float64(scale)))
    end
end

"""
    ss_lognormal(; mean, sigma)

Lognormal distribution (explicit parameterization via mean and sigma).
"""
ss_lognormal(; mean::Real, sigma::Real) = StarsimDist(:lognormal, LogNormal(log(mean), sigma))

"""
    ss_lognormal_im(; loc, scale)

Lognormal distribution (implicit parameterization: log-space μ and σ).
Matches Python `ss.lognorm_im()`.
"""
ss_lognormal_im(; loc::Real=0.0, scale::Real=1.0) = StarsimDist(:lognormal_im, LogNormal(Float64(loc), Float64(scale)))

"""Uniform distribution."""
ss_uniform(; low::Real=0.0, high::Real=1.0) = StarsimDist(:uniform, Uniform(low, high))

"""Poisson distribution."""
ss_poisson(; lam::Real) = StarsimDist(:poisson, Poisson(lam))

"""Exponential distribution."""
ss_exponential(; scale::Real=1.0) = StarsimDist(:exponential, Exponential(scale))

"""
    ss_beta(; a, b)

Beta distribution. Matches Python `ss.beta_dist()`.
"""
ss_beta(; a::Real=1.0, b::Real=1.0) = StarsimDist(:beta, Beta(a, b))

"""
    ss_beta_mean(; mean, n)

Beta distribution parameterized by mean and sample-size `n`.
Converts to `Beta(mean*n, (1-mean)*n)`. Matches Python `ss.beta_mean()`.
"""
function ss_beta_mean(; mean::Real=0.5, n::Real=10.0)
    a = mean * n
    b = (1.0 - mean) * n
    StarsimDist(:beta_mean, Beta(a, b))
end

"""
    ss_weibull(; c, scale)

Weibull distribution. `c` = shape, `scale` = scale. Matches Python `ss.weibull()`.
"""
ss_weibull(; c::Real=1.0, scale::Real=1.0) = StarsimDist(:weibull, Weibull(c, scale))

"""
    ss_gamma(; a, scale)

Gamma distribution. `a` = shape, `scale` = scale. Matches Python `ss.gamma()`.
"""
ss_gamma(; a::Real=1.0, scale::Real=1.0) = StarsimDist(:gamma, Gamma(a, scale))

"""
    ss_nbinom(; n, p)

Negative binomial distribution. Matches Python `ss.nbinom()`.
"""
ss_nbinom(; n::Real=1.0, p::Real=0.5) = StarsimDist(:nbinom, NegativeBinomial(n, p))

"""
    ss_constant(; value)

Constant "distribution" — always returns the same value.
Matches Python `ss.constant()`.
"""
mutable struct ConstantDist <: AbstractStarsimDist
    name::Symbol
    value::Float64
    initialized::Bool
end

ConstantDist(name::Symbol, value::Real) = ConstantDist(name, Float64(value), false)

function init_dist!(d::ConstantDist; base_seed::Int=0, trace::String="")
    d.initialized = true
    return d
end
jump_dt!(d::ConstantDist, ti::Int) = d
set_slots!(d::ConstantDist, slots) = d
rvs(d::ConstantDist, n::Int) = fill(d.value, n)
rvs(d::ConstantDist, uids::UIDs) = fill(d.value, length(uids))
rvs(d::ConstantDist) = d.value

ss_constant(; value::Real=0.0) = ConstantDist(:constant, value)

"""
    ss_choice(; a, p)

Random choice from discrete options `a` with optional probabilities `p`.
Matches Python `ss.choice()`.
"""
mutable struct ChoiceDist <: AbstractStarsimDist
    name::Symbol
    options::Vector{Float64}
    weights::Vector{Float64}
    rng::StableRNG
    seed::UInt64
    initialized::Bool
end

function ChoiceDist(name::Symbol, options::AbstractVector, weights::AbstractVector=Float64[])
    w = isempty(weights) ? ones(Float64, length(options)) ./ length(options) : Float64.(weights)
    ChoiceDist(name, Float64.(options), w, StableRNG(0), UInt64(0), false)
end

function init_dist!(d::ChoiceDist; base_seed::Int=0, trace::String="")
    h = hash(isempty(trace) ? string(d.name) : trace)
    d.seed = UInt64(h) ⊻ UInt64(base_seed)
    d.rng = StableRNG(d.seed)
    d.initialized = true
    return d
end

function jump_dt!(d::ChoiceDist, ti::Int)
    d.rng = StableRNG(d.seed + UInt64(ti) * UInt64(1000))
    return d
end

set_slots!(d::ChoiceDist, slots) = d

function rvs(d::ChoiceDist, n::Int)
    d.initialized || error("ChoiceDist not initialized.")
    cat = Categorical(d.weights ./ sum(d.weights))
    indices = rand(d.rng, cat, n)
    return d.options[indices]
end
rvs(d::ChoiceDist, uids::UIDs) = rvs(d, length(uids))
rvs(d::ChoiceDist) = rvs(d, 1)[1]

ss_choice(; a::AbstractVector, p::AbstractVector=Float64[]) = ChoiceDist(:choice, a, p)

"""
    ss_dur(; mean)

Duration distribution — alias for exponential with rate `1/mean`.
Matches Python `ss.dur()`.
"""
ss_dur(; mean::Real) = ss_exponential(scale=Float64(mean))

"""
    ss_randint(; low=0, high=10)

Random integer distribution — uniform choice from `low:high`.
Matches Python `ss.randint()`.
"""
ss_randint(; low::Int=0, high::Int=10) = ChoiceDist(:randint, collect(Float64, low:high))

export bernoulli, ss_random, ss_normal, ss_lognormal, ss_lognormal_im, ss_uniform,
       ss_poisson, ss_exponential, ss_beta, ss_beta_mean, ss_weibull, ss_gamma,
       ss_nbinom, ss_constant, ss_choice, ConstantDist, ChoiceDist,
       ss_dur, ss_randint

# ============================================================================
# Dists container — manages all distributions in a module tree
# ============================================================================

"""
    DistsContainer

Collects and manages all `AbstractStarsimDist` instances across a
simulation's module tree. Handles bulk initialization and per-timestep
RNG jumping.
"""
mutable struct DistsContainer
    dists::OrderedDict{String, AbstractStarsimDist}
    base_seed::Int
end

DistsContainer(; base_seed::Int=0) = DistsContainer(OrderedDict{String, AbstractStarsimDist}(), base_seed)

"""
    register_dist!(dc::DistsContainer, trace::String, dist::AbstractStarsimDist)

Register a distribution with its trace path for seeding.
"""
function register_dist!(dc::DistsContainer, trace::String, dist::AbstractStarsimDist)
    dc.dists[trace] = dist
    return dc
end

"""
    init_dists!(dc::DistsContainer)

Initialize all registered distributions with deterministic seeds.
"""
function init_dists!(dc::DistsContainer)
    for (trace, dist) in dc.dists
        init_dist!(dist; base_seed=dc.base_seed, trace=trace)
    end
    return dc
end

"""
    jump_all!(dc::DistsContainer, ti::Int)

Jump all distributions to the state for timestep `ti`.
"""
function jump_all!(dc::DistsContainer, ti::Int)
    for dist in values(dc.dists)
        jump_dt!(dist, ti)
    end
    return dc
end

export DistsContainer, register_dist!, init_dists!, jump_all!
