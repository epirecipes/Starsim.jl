"""
Connectors for Starsim.jl.

Mirrors Python starsim's `connectors.py`. Connectors mediate cross-module
interactions such as seasonality, coinfection, and cross-disease immunity.
"""

"""
    ConnectorData

Common mutable data for connector modules.
"""
mutable struct ConnectorData
    mod::ModuleData
end

"""
    connector_data(c::AbstractConnector) → ConnectorData

Return the ConnectorData. Concrete connectors must implement this.
"""
function connector_data end

module_data(c::AbstractConnector) = connector_data(c).mod

export ConnectorData, connector_data

# ============================================================================
# Seasonality — modulates transmission by time of year
# ============================================================================

"""
    Seasonality <: AbstractConnector

Modulates disease transmission rates with a seasonal (cosine) curve.

# Keyword arguments
- `name::Symbol` — connector name (default `:seasonality`)
- `disease_name::Symbol` — target disease (default `:sir`)
- `amplitude::Float64` — seasonal amplitude (0–1, default 0.3)
- `peak_day::Int` — day of year for peak transmission (default 1 = Jan 1)

# Example
```julia
seas = Seasonality(disease_name=:sir, amplitude=0.5, peak_day=180)
```
"""
mutable struct Seasonality <: AbstractConnector
    data::ConnectorData
    disease_name::Symbol
    amplitude::Float64
    peak_day::Int
end

function Seasonality(;
    name::Symbol = :seasonality,
    disease_name::Symbol = :sir,
    amplitude::Real = 0.3,
    peak_day::Real = 1,
)
    md = ModuleData(name; label="Seasonality")
    cd = ConnectorData(md)
    Seasonality(cd, disease_name, Float64(amplitude), Int(peak_day))
end

connector_data(s::Seasonality) = s.data

function init_pre!(s::Seasonality, sim)
    md = module_data(s)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return s
end

"""Compute seasonal multiplier for the current time."""
function seasonal_factor(s::Seasonality, year_frac::Float64)
    day_of_year = (year_frac - floor(year_frac)) * 365.0
    phase = 2π * (day_of_year - s.peak_day) / 365.0
    return 1.0 + s.amplitude * cos(phase)
end

function step!(s::Seasonality, sim)
    md = module_data(s)
    ti = md.t.ti
    year = sim.pars.start + (ti - 1) * sim.pars.dt

    factor = seasonal_factor(s, year)

    # Modulate rel_trans for the target disease
    if haskey(sim.diseases, s.disease_name)
        disease = sim.diseases[s.disease_name]
        if hasproperty(disease, :infection) || hasfield(typeof(disease), :infection)
            active = sim.people.auids.values
            for u in active
                if disease.infection.infected.raw[u]
                    disease.infection.rel_trans.raw[u] *= factor
                end
            end
        end
    end

    return s
end

export Seasonality, seasonal_factor

# ============================================================================
# CoinfectionConnector — cross-disease immunity/susceptibility
# ============================================================================

"""
    CoinfectionConnector <: AbstractConnector

Modifies susceptibility or transmissibility based on coinfection status.

# Keyword arguments
- `name::Symbol` — connector name
- `disease1::Symbol` — first disease
- `disease2::Symbol` — second disease
- `rel_sus_if_infected::Float64` — multiplier on disease2 susceptibility if infected with disease1
- `rel_trans_if_infected::Float64` — multiplier on disease1 transmissibility if infected with disease2
"""
mutable struct CoinfectionConnector <: AbstractConnector
    data::ConnectorData
    disease1::Symbol
    disease2::Symbol
    rel_sus_if_infected::Float64
    rel_trans_if_infected::Float64
end

function CoinfectionConnector(;
    name::Symbol = :coinfection,
    disease1::Symbol = :hiv,
    disease2::Symbol = :gonorrhea,
    rel_sus_if_infected::Real = 2.0,
    rel_trans_if_infected::Real = 1.5,
)
    md = ModuleData(name; label="Coinfection connector")
    cd = ConnectorData(md)
    CoinfectionConnector(cd, disease1, disease2,
                         Float64(rel_sus_if_infected), Float64(rel_trans_if_infected))
end

connector_data(c::CoinfectionConnector) = c.data

function init_pre!(c::CoinfectionConnector, sim)
    md = module_data(c)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function step!(c::CoinfectionConnector, sim)
    # Modify susceptibility/transmissibility based on coinfection
    if haskey(sim.diseases, c.disease1) && haskey(sim.diseases, c.disease2)
        d1 = sim.diseases[c.disease1]
        d2 = sim.diseases[c.disease2]

        for u in sim.people.auids.values
            # If infected with disease1, modify susceptibility to disease2
            if d1.infection.infected.raw[u]
                d2.infection.rel_sus.raw[u] *= c.rel_sus_if_infected
            end
            # If infected with disease2, modify transmissibility of disease1
            if d2.infection.infected.raw[u]
                d1.infection.rel_trans.raw[u] *= c.rel_trans_if_infected
            end
        end
    end

    return c
end

export CoinfectionConnector
