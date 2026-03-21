"""
FPSim convenience constructor.
Creates a family planning simulation with reproductive lifecycle.
"""

"""
    FPSim(; n_agents, start, stop, dt, location, kwargs...)

Convenience constructor for a family planning simulation.

# Keyword arguments
- `n_agents::Int` — number of agents (default 1000)
- `start::Real` — start year (default 2000.0)
- `stop::Real` — stop year (default 2020.0)
- `dt::Real` — timestep in years (default 1/12, monthly)
- `rand_seed::Int` — RNG seed (default 0)
- `location::Symbol` — location for demographic data (default :generic)
- `pars::Union{FPPars, Nothing}` — custom parameters
- `use_contraception::Bool` — include contraception (default false)
- `initiation_rate::Float64` — annual contraception initiation rate (default 0.10)
- `analyzers` — optional analyzers
"""
function FPSim(;
    n_agents::Int = 1000,
    start::Real = 2000.0,
    stop::Real = 2020.0,
    dt::Real = 1/12,
    rand_seed::Int = 0,
    location::Symbol = :generic,
    pars::Union{FPPars, Nothing} = nothing,
    use_contraception::Bool = false,
    initiation_rate::Float64 = 0.10,
    analyzers = nothing,
    kwargs...,
)
    fp_pars = pars === nothing ? load_location_data(location) : pars

    # Load methods for this location
    methods = load_methods(; location=location)

    # Build connectors
    connectors = Starsim.AbstractConnector[FPmod(pars=fp_pars, methods=methods)]

    # Build interventions
    interventions = Starsim.AbstractIntervention[]
    if use_contraception
        mm = load_method_mix(; location=location, methods=methods)
        push!(interventions, Contraception(;
            methods=methods,
            method_mix=mm,
            initiation_rate=initiation_rate,
        ))
    end

    sim = Starsim.Sim(;
        n_agents   = n_agents,
        start      = start,
        stop       = stop,
        dt         = dt,
        rand_seed  = rand_seed,
        connectors = connectors,
        interventions = isempty(interventions) ? nothing : interventions,
        analyzers  = analyzers,
        kwargs...,
    )

    return sim
end
