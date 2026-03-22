"""
Interventions for Starsim.jl.

Mirrors Python starsim's `interventions.py`. Provides delivery mechanisms
(routine, campaign) that apply products to eligible agents.
"""

# ============================================================================
# Intervention base
# ============================================================================

"""
    InterventionData

Common mutable data for intervention modules.
"""
mutable struct InterventionData
    mod::ModuleData
    product::Union{AbstractProduct, Nothing}
    disease_name::Symbol
end

"""
    intervention_data(iv::AbstractIntervention) → InterventionData

Return the InterventionData. Concrete intervention types must implement this.
"""
function intervention_data end

module_data(iv::AbstractIntervention) = intervention_data(iv).mod

export InterventionData, intervention_data

# ============================================================================
# RoutineDelivery
# ============================================================================

"""
    RoutineDelivery <: AbstractIntervention

Routine intervention delivery — applies a product to eligible agents at
a specified coverage rate each timestep.

# Keyword arguments
- `name::Symbol` — intervention name (default `:routine`)
- `product::AbstractProduct` — the product to deliver
- `disease_name::Symbol` — target disease (default `:sir`)
- `prob::Float64` — per-timestep delivery probability (default 0.01)
- `start_year::Float64` — year to start delivering (default 0.0)
- `end_year::Float64` — year to stop delivering (default Inf)
- `eligibility::Function` — function(sim, uids) → eligible UIDs (default: all alive)

# Example
```julia
vx = Vx(efficacy=0.9)
routine = RoutineDelivery(product=vx, prob=0.02, disease_name=:sir)
```
"""
mutable struct RoutineDelivery <: AbstractIntervention
    iv::InterventionData
    prob::Float64
    start_year::Float64
    end_year::Float64
    eligibility::Function
    rng::StableRNG
end

function RoutineDelivery(;
    name::Symbol = :routine,
    product::AbstractProduct = Vx(),
    disease_name::Symbol = :sir,
    prob::Real = 0.01,
    start_year::Real = 0.0,
    end_year::Real = Inf,
    eligibility::Function = (sim, uids) -> uids,
)
    md = ModuleData(name; label="Routine delivery")
    iv = InterventionData(md, product, disease_name)
    RoutineDelivery(iv, Float64(prob), Float64(start_year), Float64(end_year),
                    eligibility, StableRNG(0))
end

intervention_data(r::RoutineDelivery) = r.iv

function init_pre!(r::RoutineDelivery, sim)
    md = module_data(r)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    r.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    # Init product RNG
    if r.iv.product !== nothing
        r.iv.product.rng = StableRNG(hash(r.iv.product.name) ⊻ sim.pars.rand_seed)
    end

    npts = md.t.npts
    define_results!(r,
        Result(:n_delivered; npts=npts, label="Number delivered"),
    )

    md.initialized = true
    return r
end

function step!(r::RoutineDelivery, sim)
    md = module_data(r)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt

    # Check if we're in the active period
    if year < r.start_year || year > r.end_year
        return r
    end

    # Find eligible agents
    eligible = r.eligibility(sim, sim.people.auids)
    isempty(eligible) && return r

    # Deliver to random subset based on prob
    delivered = Int[]
    for u in eligible.values
        if rand(r.rng) < r.prob
            push!(delivered, u)
        end
    end

    # Apply product
    if !isempty(delivered) && r.iv.product !== nothing && haskey(sim.diseases, r.iv.disease_name)
        disease = sim.diseases[r.iv.disease_name]
        administer!(r.iv.product, UIDs(delivered), disease)
    end

    if ti <= length(md.results[:n_delivered].values)
        md.results[:n_delivered][ti] = Float64(length(delivered))
    end

    return r
end

export RoutineDelivery

# ============================================================================
# CampaignDelivery
# ============================================================================

"""
    CampaignDelivery <: AbstractIntervention

Campaign intervention — delivers a product in specific years to a fraction
of the eligible population.

# Keyword arguments
- `name::Symbol` — intervention name (default `:campaign`)
- `product::AbstractProduct` — the product to deliver
- `disease_name::Symbol` — target disease (default `:sir`)
- `years::Vector{Float64}` — years to conduct campaigns
- `coverage::Float64` — fraction of eligible population to cover (default 0.5)
- `eligibility::Function` — function(sim, uids) → eligible UIDs

# Example
```julia
vx = Vx(efficacy=0.95)
campaign = CampaignDelivery(product=vx, years=[2025.0, 2030.0], coverage=0.8)
```
"""
mutable struct CampaignDelivery <: AbstractIntervention
    iv::InterventionData
    years::Vector{Float64}
    coverage::Float64
    eligibility::Function
    rng::StableRNG
end

function CampaignDelivery(;
    name::Symbol = :campaign,
    product::AbstractProduct = Vx(),
    disease_name::Symbol = :sir,
    years::Vector{Float64} = Float64[],
    coverage::Real = 0.5,
    eligibility::Function = (sim, uids) -> uids,
)
    md = ModuleData(name; label="Campaign delivery")
    iv = InterventionData(md, product, disease_name)
    CampaignDelivery(iv, years, Float64(coverage), eligibility, StableRNG(0))
end

intervention_data(c::CampaignDelivery) = c.iv

function init_pre!(c::CampaignDelivery, sim)
    md = module_data(c)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    c.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    if c.iv.product !== nothing
        c.iv.product.rng = StableRNG(hash(c.iv.product.name) ⊻ sim.pars.rand_seed)
    end

    npts = md.t.npts
    define_results!(c,
        Result(:n_delivered; npts=npts, label="Number delivered"),
    )

    md.initialized = true
    return c
end

function step!(c::CampaignDelivery, sim)
    md = module_data(c)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt

    # Check if this is a campaign year (within dt tolerance)
    is_campaign = any(y -> abs(year - y) < sim.pars.dt / 2, c.years)
    !is_campaign && return c

    # Find eligible agents
    eligible = c.eligibility(sim, sim.people.auids)
    isempty(eligible) && return c

    # Select coverage fraction
    n_deliver = max(1, Int(round(length(eligible) * c.coverage)))
    perm = randperm(c.rng, length(eligible))
    delivered = UIDs(eligible.values[perm[1:min(n_deliver, length(eligible))]])

    # Apply product
    if !isempty(delivered) && c.iv.product !== nothing && haskey(sim.diseases, c.iv.disease_name)
        disease = sim.diseases[c.iv.disease_name]
        administer!(c.iv.product, delivered, disease)
    end

    if ti <= length(md.results[:n_delivered].values)
        md.results[:n_delivered][ti] = Float64(length(delivered))
    end

    return c
end

"""
    FunctionIntervention <: AbstractIntervention

A simple intervention that calls a user-provided function each timestep.

# Example
```julia
iv = FunctionIntervention(fn = (sim, ti) -> begin
    # Custom logic here
end)
```
"""
mutable struct FunctionIntervention <: AbstractIntervention
    iv::InterventionData
    fn::Function
end

function FunctionIntervention(;
    name::Symbol = :func_iv,
    fn::Function = (sim, ti) -> nothing,
)
    md = ModuleData(name; label="Function intervention")
    iv = InterventionData(md, nothing, :none)
    FunctionIntervention(iv, fn)
end

intervention_data(fi::FunctionIntervention) = fi.iv

function init_pre!(fi::FunctionIntervention, sim)
    md = module_data(fi)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return fi
end

function step!(fi::FunctionIntervention, sim)
    ti = module_data(fi).t.ti
    fi.fn(sim, ti)
    return fi
end

export CampaignDelivery, FunctionIntervention

# ============================================================================
# Convenience constructors matching Python API
# ============================================================================

"""Create a routine vaccination delivery. Matches Python `ss.routine_vx()`."""
routine_vx(; prob=0.02, efficacy=0.9, kwargs...) = RoutineDelivery(; product=Vx(efficacy=efficacy), prob=prob, kwargs...)

"""Create a campaign vaccination delivery. Matches Python `ss.campaign_vx()`."""
campaign_vx(; years, coverage=0.8, efficacy=0.9, kwargs...) = CampaignDelivery(; product=Vx(efficacy=efficacy), years=years, coverage=coverage, kwargs...)

"""Alias for `routine_vx`. Matches Python `ss.simple_vx()`."""
simple_vx(; prob=0.02, efficacy=0.9, kwargs...) = RoutineDelivery(; product=Vx(efficacy=efficacy), prob=prob, kwargs...)

"""Create a routine screening delivery. Matches Python `ss.routine_screening()`."""
routine_screening(; prob=0.1, sensitivity=0.95, specificity=0.99, kwargs...) = RoutineDelivery(; product=Dx(sensitivity=sensitivity, specificity=specificity), prob=prob, kwargs...)

"""Create a campaign screening delivery. Matches Python `ss.campaign_screening()`."""
campaign_screening(; years, coverage=0.5, sensitivity=0.95, specificity=0.99, kwargs...) = CampaignDelivery(; product=Dx(sensitivity=sensitivity, specificity=specificity), years=years, coverage=coverage, kwargs...)

export routine_vx, campaign_vx, simple_vx, routine_screening, campaign_screening
