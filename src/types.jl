"""
Abstract type hierarchy for Starsim.jl.

All simulation components inherit from these abstract types, enabling
multiple dispatch for the module lifecycle (init, step, finalize).
"""

# ---------------------------------------------------------------------------
# Module hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractModule

Root abstract type for all Starsim modules. Every simulation component
(disease, network, intervention, etc.) is a subtype of `AbstractModule`.
"""
abstract type AbstractModule end

"""
    AbstractDemographics <: AbstractModule

Demographics modules handle births, deaths, aging, and migration.
"""
abstract type AbstractDemographics <: AbstractModule end

"""
    AbstractRoute <: AbstractModule

Base type for all transmission routes (contact networks, mixing pools, etc.).
"""
abstract type AbstractRoute <: AbstractModule end

"""
    AbstractNetwork <: AbstractRoute

Contact networks where agents form explicit pairwise edges.
"""
abstract type AbstractNetwork <: AbstractRoute end

"""
    AbstractDisease <: AbstractModule

Base type for all disease models (infectious and non-communicable).
"""
abstract type AbstractDisease <: AbstractModule end

"""
    AbstractInfection <: AbstractDisease

Infectious diseases with network-based transmission, directional beta,
and susceptibility/infectiousness modifiers.
"""
abstract type AbstractInfection <: AbstractDisease end

"""
    AbstractNCD <: AbstractDisease

Non-communicable diseases with risk-factor-based acquisition.
"""
abstract type AbstractNCD <: AbstractDisease end

"""
    AbstractIntervention <: AbstractModule

Interventions that modify simulation state (vaccines, treatments, screening).
"""
abstract type AbstractIntervention <: AbstractModule end

"""
    AbstractAnalyzer <: AbstractModule

Analyzers that track custom results during the simulation.
"""
abstract type AbstractAnalyzer <: AbstractModule end

"""
    AbstractConnector <: AbstractModule

Connectors that mediate interactions between modules (e.g., co-infection effects).
"""
abstract type AbstractConnector <: AbstractModule end

"""
    AbstractProduct

Products administered by interventions (vaccines, diagnostics, treatments).
"""
abstract type AbstractProduct end

# ---------------------------------------------------------------------------
# State hierarchy
# ---------------------------------------------------------------------------

"""
    AbstractState

Base type for agent state arrays. Subtypes are parametric on the array
backend `A <: AbstractVector` to support CPU/GPU execution.
"""
abstract type AbstractState end

"""
    AbstractFloatState <: AbstractState

Floating-point agent state (e.g., age, time of infection).
"""
abstract type AbstractFloatState <: AbstractState end

"""
    AbstractIntState <: AbstractState

Integer agent state (e.g., counts, indices).
"""
abstract type AbstractIntState <: AbstractState end

"""
    AbstractBoolState <: AbstractState

Boolean agent state (e.g., infected, alive). `BoolState` subtypes
automatically generate result tracking (`n_infected`, etc.).
"""
abstract type AbstractBoolState <: AbstractState end

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

"""
    AbstractResult

Base type for simulation result arrays.
"""
abstract type AbstractResult end

# ---------------------------------------------------------------------------
# Time types
# ---------------------------------------------------------------------------

"""
    AbstractTimePar

Base type for time-aware parameters (durations, rates, probabilities).
"""
abstract type AbstractTimePar end

"""
    AbstractDuration <: AbstractTimePar

Duration parameters with time units (days, weeks, months, years).
"""
abstract type AbstractDuration <: AbstractTimePar end

"""
    AbstractRate <: AbstractTimePar

Rate parameters (per-time, probability-per-time, frequency-per-time).
"""
abstract type AbstractRate <: AbstractTimePar end

# ---------------------------------------------------------------------------
# Distribution types
# ---------------------------------------------------------------------------

"""
    AbstractStarsimDist

Base type for Starsim distribution wrappers around `Distributions.jl`.
"""
abstract type AbstractStarsimDist end

# ---------------------------------------------------------------------------
# Module map — maps category symbols to abstract types
# ---------------------------------------------------------------------------

"""
    MODULE_MAP

Maps module category names to their abstract supertypes.
This is the source of truth about module types.
"""
const MODULE_MAP = OrderedDict{Symbol, Type}(
    :demographics  => AbstractDemographics,
    :networks      => AbstractRoute,
    :diseases      => AbstractDisease,
    :connectors    => AbstractConnector,
    :interventions => AbstractIntervention,
    :analyzers     => AbstractAnalyzer,
)

"""
    MODULE_CATEGORIES

List of module category names in execution order.
"""
const MODULE_CATEGORIES = collect(keys(MODULE_MAP))

# ---------------------------------------------------------------------------
# Module registry for string-based lookup
# ---------------------------------------------------------------------------

"""
    MODULE_REGISTRY

Global registry mapping lowercase names to module types.
Populated by [`register_module!`](@ref) and used by [`find_module`](@ref).
"""
const MODULE_REGISTRY = Dict{Symbol, Type}()

"""
    register_module!(T::Type{<:AbstractModule})

Register a module type so it can be looked up by name string.

# Example
```julia
register_module!(SIR)
# Now find_module(:sir) returns SIR
```
"""
function register_module!(T::Type{<:AbstractModule})
    MODULE_REGISTRY[Symbol(lowercase(string(nameof(T))))] = T
    return nothing
end

"""
    find_module(name::Symbol)

Look up a registered module type by its lowercase name.

# Example
```julia
T = find_module(:sir)  # returns SIR
```
"""
function find_module(name::Symbol)
    key = Symbol(lowercase(string(name)))
    haskey(MODULE_REGISTRY, key) || error("Unknown module: $name. Registered: $(keys(MODULE_REGISTRY))")
    return MODULE_REGISTRY[key]
end

# ---------------------------------------------------------------------------
# Numeric type aliases
# ---------------------------------------------------------------------------

"""Default float type for agent states."""
const SS_FLOAT = Float64

"""Default integer type for agent states."""
const SS_INT = Int64

"""Default boolean type for agent states."""
const SS_BOOL = Bool

"""Sentinel value for integer NaN (missing data in integer arrays)."""
const INT_NAN = typemin(Int64)

"""Sentinel value for float NaN."""
const FLOAT_NAN = NaN

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

export AbstractModule, AbstractDemographics, AbstractRoute, AbstractNetwork,
       AbstractDisease, AbstractInfection, AbstractNCD,
       AbstractIntervention, AbstractAnalyzer, AbstractConnector, AbstractProduct,
       AbstractState, AbstractFloatState, AbstractIntState, AbstractBoolState,
       AbstractResult, AbstractTimePar, AbstractDuration, AbstractRate,
       AbstractStarsimDist,
       MODULE_MAP, MODULE_CATEGORIES, MODULE_REGISTRY,
       register_module!, find_module,
       SS_FLOAT, SS_INT, SS_BOOL, INT_NAN, FLOAT_NAN
