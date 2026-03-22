"""
Intervention products for Starsim.jl.

Mirrors Python starsim's product classes: diagnostic (Dx), treatment (Tx),
and vaccination (Vx) products.
"""

# ============================================================================
# Product base
# ============================================================================

# AbstractProduct is defined in types.jl

"""
    Dx <: AbstractProduct

Diagnostic product — tests agents and returns results.

# Keyword arguments
- `name::Symbol` — product name (default `:dx`)
- `sensitivity::Float64` — probability of true positive (default 1.0)
- `specificity::Float64` — probability of true negative (default 1.0)
"""
mutable struct Dx <: AbstractProduct
    name::Symbol
    label::String
    sensitivity::Float64
    specificity::Float64
    rng::StableRNG
end

function Dx(;
    name::Symbol = :dx,
    label::String = "Diagnostic",
    sensitivity::Real = 1.0,
    specificity::Real = 1.0,
)
    Dx(name, label, Float64(sensitivity), Float64(specificity), StableRNG(0))
end

"""
    administer!(dx::Dx, uids::UIDs, disease) → diagnosed::UIDs

Apply diagnostic to agents. Returns UIDs of agents who test positive.
"""
function administer!(dx::Dx, uids::UIDs, disease)
    diagnosed = Int[]
    for u in uids.values
        if disease.infection.infected.raw[u]
            # True positive
            if rand(dx.rng) < dx.sensitivity
                push!(diagnosed, u)
            end
        else
            # False positive
            if rand(dx.rng) > dx.specificity
                push!(diagnosed, u)
            end
        end
    end
    return UIDs(diagnosed)
end

export Dx, administer!

"""
    Tx <: AbstractProduct

Treatment product — applies treatment effect to agents.

# Keyword arguments
- `name::Symbol` — product name (default `:tx`)
- `efficacy::Float64` — treatment efficacy (default 1.0)
"""
mutable struct Tx <: AbstractProduct
    name::Symbol
    label::String
    efficacy::Float64
    rng::StableRNG
end

function Tx(;
    name::Symbol = :tx,
    label::String = "Treatment",
    efficacy::Real = 1.0,
)
    Tx(name, label, Float64(efficacy), StableRNG(0))
end

"""
    administer!(tx::Tx, uids::UIDs, disease) → treated::UIDs

Apply treatment. Returns UIDs of successfully treated agents.
"""
function administer!(tx::Tx, uids::UIDs, disease)
    treated = Int[]
    for u in uids.values
        if rand(tx.rng) < tx.efficacy && disease.infection.infected.raw[u]
            # Cure: set infected to false, susceptible to true
            disease.infection.infected.raw[u] = false
            disease.infection.susceptible.raw[u] = true
            push!(treated, u)
        end
    end
    return UIDs(treated)
end

export Tx

"""
    Vx <: AbstractProduct

Vaccination product — reduces susceptibility.

# Keyword arguments
- `name::Symbol` — product name (default `:vx`)
- `efficacy::Float64` — vaccine efficacy (default 0.9)
"""
mutable struct Vx <: AbstractProduct
    name::Symbol
    label::String
    efficacy::Float64
    rng::StableRNG
end

function Vx(;
    name::Symbol = :vx,
    label::String = "Vaccine",
    efficacy::Real = 0.9,
)
    Vx(name, label, Float64(efficacy), StableRNG(0))
end

"""
    administer!(vx::Vx, uids::UIDs, disease) → vaccinated::UIDs

Apply vaccine. Reduces `rel_sus` for susceptible agents.
"""
function administer!(vx::Vx, uids::UIDs, disease)
    vaccinated = Int[]
    for u in uids.values
        if disease.infection.susceptible.raw[u]
            if rand(vx.rng) < vx.efficacy
                disease.infection.rel_sus.raw[u] *= (1.0 - vx.efficacy)
                push!(vaccinated, u)
            end
        end
    end
    return UIDs(vaccinated)
end

export Vx
