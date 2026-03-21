"""
STISim convenience constructor.
Creates a complete STI simulation with sexual networks.
"""

"""
    STISim(; diseases, n_agents, start, stop, dt, kwargs...)

Convenience constructor for an STI simulation.

# Keyword arguments
- `diseases` — list of disease specs (default: [HIV()])
- `n_agents::Int` — number of agents (default 5000)
- `start::Real` — start year (default 2000.0)
- `stop::Real` — stop year (default 2050.0)
- `dt::Real` — timestep in years (default 1/52, weekly)
- `rand_seed::Int` — RNG seed (default 0)
- `network` — custom network (default: StructuredSexual())
- `connectors` — coinfection connectors
- `interventions` — interventions
- `analyzers` — analyzers
"""
function STISim(;
    diseases = nothing,
    n_agents::Int = 5000,
    start::Real = 2000.0,
    stop::Real = 2050.0,
    dt::Real = 1/52,
    rand_seed::Int = 0,
    network = nothing,
    connectors = nothing,
    interventions = nothing,
    analyzers = nothing,
    kwargs...,
)
    # Default diseases
    if diseases === nothing
        diseases = [HIV()]
    end

    # Default network
    if network === nothing
        network = StructuredSexual()
    end

    sim = Starsim.Sim(;
        n_agents   = n_agents,
        start      = start,
        stop       = stop,
        dt         = dt,
        rand_seed  = rand_seed,
        networks   = network,
        diseases   = diseases,
        connectors = connectors,
        interventions = interventions,
        analyzers  = analyzers,
        kwargs...,
    )

    return sim
end
