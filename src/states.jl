"""
State management arrays for agents.

Mirrors Python starsim's `arrays.py`. Each state wraps a raw array with
active-UID-based views and supports dynamic growth as agents are born.

All state types are parametric on the array backend `A` so the same code
works on CPU (`Vector`) or GPU (`MtlVector`).
"""

# ============================================================================
# UIDs — typed container for agent unique identifiers
# ============================================================================

"""
    UIDs{A<:AbstractVector{Int}} <: AbstractVector{Int}

Sorted vector of agent unique identifiers. Supports set operations
(`intersect`, `union`, `setdiff`, `symdiff`) and concatenation.

# Constructors
```julia
UIDs()                     # empty
UIDs([1, 3, 5])            # from vector
UIDs(1:10)                 # from range
```
"""
struct UIDs{A<:AbstractVector{Int}} <: AbstractVector{Int}
    values::A
    UIDs{A}(v::A) where {A<:AbstractVector{Int}} = new{A}(v)
end

UIDs() = UIDs{Vector{Int}}(Int[])
UIDs(v::A) where {A<:AbstractVector{Int}} = UIDs{A}(v)
UIDs(r::AbstractRange{Int}) = UIDs(collect(r))
UIDs(x::Int) = UIDs([x])

Base.size(u::UIDs) = size(u.values)
Base.getindex(u::UIDs, i::Int) = u.values[i]
Base.getindex(u::UIDs, i) = UIDs(u.values[i])
Base.length(u::UIDs) = length(u.values)
Base.iterate(u::UIDs, args...) = iterate(u.values, args...)
Base.IndexStyle(::Type{<:UIDs}) = IndexLinear()
Base.similar(u::UIDs, ::Type{T}, dims::Dims) where T = similar(u.values, T, dims)
Base.show(io::IO, u::UIDs) = print(io, "UIDs($(u.values))")
Base.convert(::Type{UIDs}, v::AbstractVector{Int}) = UIDs(v)

"""Concatenate multiple UIDs."""
function Base.vcat(a::UIDs, b::UIDs)
    UIDs(vcat(a.values, b.values))
end

"""Concatenate any number of UIDs."""
function uids_cat(args::UIDs...)
    UIDs(vcat([a.values for a in args]...))
end

"""Set intersection of UIDs."""
Base.intersect(a::UIDs, b::UIDs) = UIDs(intersect(a.values, b.values))

"""Set union of UIDs."""
Base.union(a::UIDs, b::UIDs) = UIDs(union(a.values, b.values))

"""Set difference of UIDs (elements in `a` not in `b`)."""
Base.setdiff(a::UIDs, b::UIDs) = UIDs(setdiff(a.values, b.values))

"""Symmetric difference of UIDs."""
Base.symdiff(a::UIDs, b::UIDs) = UIDs(symdiff(a.values, b.values))

"""Unique UIDs, preserving order."""
function Base.unique(u::UIDs)
    UIDs(unique(u.values))
end

"""Check if a UID is contained."""
Base.in(x::Int, u::UIDs) = in(x, u.values)

"""Check if empty."""
Base.isempty(u::UIDs) = isempty(u.values)

export UIDs, uids_cat

# ============================================================================
# StateVector — core agent state array
# ============================================================================

"""
    StateVector{T, A<:AbstractVector{T}} <: AbstractState

Wraps a raw array of agent data with active-UID views and dynamic growth.

The `raw` field stores data for *all* agents (including dead/removed).
The `values` property returns only data for currently active agents,
filtered by `auids` (active UIDs).

# Fields
- `name::Symbol` — state name (e.g., `:infected`)
- `label::String` — human-readable label
- `default` — default value for new agents (scalar, function `n -> values`, or Distribution)
- `nan_val` — sentinel for missing data
- `raw::A` — underlying array (all agents, including dead)
- `len_used::Int` — number of populated slots
- `len_tot::Int` — total allocated capacity
- `auids_ref::Ref` — reference to the active UIDs (shared with People)
- `initialized::Bool`

# Indexing
- `state[i::Int]` — index into `values` (active agents)
- `state[u::UIDs]` — index into `raw` by UID
- `state[b::BoolState]` — index by boolean mask → UIDs
"""
mutable struct StateVector{T, A<:AbstractVector{T}} <: AbstractState
    name::Symbol
    label::String
    default::Any
    nan_val::T
    raw::A
    len_used::Int
    len_tot::Int
    auids_ref::Base.RefValue{UIDs{Vector{Int}}}
    initialized::Bool
end

function Base.show(io::IO, s::StateVector{T}) where T
    n = s.initialized ? length(get_auids(s)) : 0
    print(io, "StateVector{$T}(:$(s.name), n_active=$n, capacity=$(s.len_tot))")
end

"""Get active UIDs from the shared reference."""
get_auids(s::StateVector) = s.auids_ref[]

"""Get values for active agents only."""
function active_values(s::StateVector)
    auids = get_auids(s)
    return s.raw[auids.values]
end

"""Number of active agents."""
Base.length(s::StateVector) = s.initialized ? length(get_auids(s)) : 0

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

"""
    FloatState(name; label, default, nan_val)

Create a floating-point state array.

# Example
```julia
age = FloatState(:age; default=0.0, label="Age")
```
"""
function FloatState(name::Symbol; label::String=string(name),
                    default=FLOAT_NAN, nan_val::Float64=FLOAT_NAN)
    StateVector{Float64, Vector{Float64}}(
        name, label, default, nan_val,
        Float64[], 0, 0, Ref(UIDs()), false
    )
end

"""
    IntState(name; label, default, nan_val)

Create an integer state array.

# Example
```julia
n_contacts = IntState(:n_contacts; default=0)
```
"""
function IntState(name::Symbol; label::String=string(name),
                  default=INT_NAN, nan_val::Int64=INT_NAN)
    StateVector{Int64, Vector{Int64}}(
        name, label, default, nan_val,
        Int64[], 0, 0, Ref(UIDs()), false
    )
end

"""
    BoolState(name; label, default)

Create a boolean state array. BoolStates automatically generate
result tracking (e.g., `n_infected` for a state named `:infected`).

# Example
```julia
infected = BoolState(:infected; label="Infected")
```
"""
function BoolState(name::Symbol; label::String=string(name), default::Bool=false)
    StateVector{Bool, Vector{Bool}}(
        name, label, default, false,
        Bool[], 0, 0, Ref(UIDs()), false
    )
end

"""
    IndexState(name; label)

Create an index/UID state array (integers, NaN sentinel = INT_NAN).
Used for `uid` and `slot` arrays that are not regular agent states.
"""
function IndexState(name::Symbol; label::String=string(name))
    StateVector{Int64, Vector{Int64}}(
        name, label, INT_NAN, INT_NAN,
        Int64[], 0, 0, Ref(UIDs()), false
    )
end

export StateVector, FloatState, IntState, BoolState, IndexState,
       active_values, get_auids

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

"""
    init_state!(s::StateVector, n::Int, auids_ref)

Initialize a state vector for `n` agents. Links to the shared `auids_ref`.
"""
function init_state!(s::StateVector{T}, n::Int, auids_ref) where T
    s.auids_ref = auids_ref
    s.len_used = n
    s.len_tot = n
    s.raw = fill(s.nan_val, n)
    s.initialized = true
    return s
end

"""
    init_vals!(s::StateVector, uids::UIDs)

Populate state values for the given UIDs using the `default` field.
Handles scalar defaults, callable defaults `f(n)`, and Distributions.
"""
function init_vals!(s::StateVector{T}, uids::UIDs) where T
    n = length(uids)
    n == 0 && return s
    d = s.default
    if d isa Number
        s.raw[uids.values] .= T(d)
    elseif d isa Function
        vals = d(n)
        s.raw[uids.values] .= T.(vals)
    elseif d isa AbstractStarsimDist
        # Will be handled after distribution system is initialized
    else
        s.raw[uids.values] .= T(d)
    end
    return s
end

export init_state!, init_vals!

# ---------------------------------------------------------------------------
# Growth (dynamic resizing for births)
# ---------------------------------------------------------------------------

"""
    grow!(s::StateVector{T}, new_uids::UIDs; new_vals=nothing) where T

Extend the state array to accommodate new agents. Uses 50%
over-allocation to amortize resize cost, matching the Python strategy.

# Arguments
- `new_uids` — UIDs of the new agents
- `new_vals` — optional values to assign (otherwise uses `default`)
"""
function grow!(s::StateVector{T}, new_uids::UIDs; new_vals=nothing) where T
    n_new = length(new_uids)
    n_new == 0 && return s
    max_uid = maximum(new_uids.values)

    # Ensure capacity
    if max_uid > s.len_tot
        n_grow = max(max_uid - s.len_tot, s.len_tot ÷ 2, n_new)
        new_size = s.len_tot + n_grow
        old_raw = s.raw
        s.raw = fill(s.nan_val, new_size)
        s.raw[1:s.len_tot] .= old_raw
        s.len_tot = new_size
    end

    s.len_used += n_new

    # Set values
    if new_vals !== nothing
        s.raw[new_uids.values] .= T.(new_vals)
    else
        init_vals!(s, new_uids)
    end
    return s
end

export grow!

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

# Index by UIDs → raw array access
function Base.getindex(s::StateVector, u::UIDs)
    return s.raw[u.values]
end

# Set by UIDs → raw array access
function Base.setindex!(s::StateVector, val, u::UIDs)
    s.raw[u.values] .= val
    return s
end

# Index by Int → active agents
function Base.getindex(s::StateVector, i::Int)
    auids = get_auids(s)
    return s.raw[auids.values[i]]
end

# Slice → active agents
function Base.getindex(s::StateVector, r::AbstractRange)
    auids = get_auids(s)
    return s.raw[auids.values[r]]
end

# Full slice → all active values
function Base.getindex(s::StateVector, ::Colon)
    return active_values(s)
end

# Set by Colon → all active agents
function Base.setindex!(s::StateVector, val, ::Colon)
    auids = get_auids(s)
    s.raw[auids.values] .= val
    return s
end

# ---------------------------------------------------------------------------
# Boolean operations on StateVector{Bool} — return UIDs
# ---------------------------------------------------------------------------

"""
    uids(s::StateVector{Bool})

Return UIDs of agents where the boolean state is `true` (active agents only).
"""
function uids(s::StateVector{Bool})
    auids = get_auids(s)
    mask = s.raw[auids.values]
    return UIDs(auids.values[mask])
end

"""
    false_uids(s::StateVector{Bool})

Return UIDs of active agents where the boolean state is `false`.
"""
function false_uids(s::StateVector{Bool})
    auids = get_auids(s)
    mask = s.raw[auids.values]
    return UIDs(auids.values[.!mask])
end

export uids, false_uids

# ---------------------------------------------------------------------------
# Comparison operations — return StateVector{Bool}-like UIDs
# ---------------------------------------------------------------------------

"""Compare state values against a scalar/array, returning UIDs where true."""
function compare_uids(s::StateVector, op::Function, val)
    auids = get_auids(s)
    mask = op.(s.raw[auids.values], val)
    return UIDs(auids.values[mask])
end

"""UIDs where `state .> val`."""
state_gt(s::StateVector, val) = compare_uids(s, >, val)

"""UIDs where `state .< val`."""
state_lt(s::StateVector, val) = compare_uids(s, <, val)

"""UIDs where `state .>= val`."""
state_gte(s::StateVector, val) = compare_uids(s, >=, val)

"""UIDs where `state .<= val`."""
state_lte(s::StateVector, val) = compare_uids(s, <=, val)

"""UIDs where `state .== val`."""
state_eq(s::StateVector, val) = compare_uids(s, ==, val)

"""UIDs where `state .!= val`."""
state_neq(s::StateVector, val) = compare_uids(s, !=, val)

"""UIDs where bool state is true AND comparison holds."""
function state_and_cmp(bool_state::StateVector{Bool}, cmp_state::StateVector, op::Function, val)
    auids = get_auids(bool_state)
    bmask = bool_state.raw[auids.values]
    cmask = op.(cmp_state.raw[auids.values], val)
    return UIDs(auids.values[bmask .& cmask])
end

export state_gt, state_lt, state_gte, state_lte, state_eq, state_neq, state_and_cmp

# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

"""Set state to scalar for given UIDs."""
function set_state!(s::StateVector{T}, u::UIDs, val) where T
    s.raw[u.values] .= T(val)
    return s
end

"""Element-wise multiply state for given UIDs."""
function mul_state!(s::StateVector{T}, u::UIDs, factor) where T
    s.raw[u.values] .*= T(factor)
    return s
end

export set_state!, mul_state!
