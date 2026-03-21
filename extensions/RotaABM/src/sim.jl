"""
RotaSim convenience constructor.
Port of Python `rotasim.rotasim.Sim`.
"""

"""
    RotaSim(; scenario, base_beta, n_agents, stop, rand_seed, kwargs...)

Convenience constructor that creates a full multi-strain rotavirus
simulation from a scenario definition.

# Keyword arguments
- `scenario` — name string or Dict (default "baseline")
- `base_beta::Float64` — base transmission rate per day (default 0.1)
- `override_fitness` — override fitness values
- `override_prevalence` — override prevalence values
- `override_strains` — add/modify strains
- `use_preferred_partners::Bool` — filter reassortments (default false)
- `n_agents::Int` — number of agents (default 5000)
- `start::Real` — start year (default 0.0)
- `stop::Real` — stop year (default 1.0, i.e. 365 days)
- `dt` — timestep: `Starsim.days(1)` (default), or Real in years (e.g. 1/365.25)
- `rand_seed::Int` — RNG seed (default 0)
- Other keyword arguments passed to `Starsim.Sim`
"""
function RotaSim(;
    scenario = "baseline",
    base_beta::Real = 0.1,
    override_fitness = nothing,
    override_prevalence = nothing,
    override_strains = nothing,
    use_preferred_partners::Bool = false,
    n_agents::Int = 5000,
    start::Real = 0.0,
    stop::Real = 1.0,
    dt::Union{Real, Starsim.Duration} = Starsim.days(1),
    rand_seed::Int = 0,
    connectors = nothing,
    analyzers = nothing,
    networks = nothing,
    kwargs...,
)
    # Validate and process scenario
    validated = validate_scenario(scenario)
    final_scenario = apply_scenario_overrides(
        validated;
        override_fitness   = override_fitness,
        override_prevalence = override_prevalence,
        override_strains   = override_strains,
    )

    # Create diseases from scenario
    diseases = _create_strain_diseases(final_scenario, base_beta, use_preferred_partners)

    # Default connectors
    if connectors === nothing
        connectors = [
            RotaImmunityConnector(),
            RotaReassortmentConnector(; use_preferred_partners=use_preferred_partners),
        ]
    end

    # Default network
    if networks === nothing
        networks = Starsim.RandomNet(n_contacts=10)
    end

    sim = Starsim.Sim(;
        n_agents   = n_agents,
        start      = start,
        stop       = stop,
        dt         = dt,
        rand_seed  = rand_seed,
        networks   = networks,
        diseases   = diseases,
        connectors = connectors,
        analyzers  = analyzers,
        kwargs...,
    )

    return sim
end

"""Create all Rotavirus disease instances from scenario data."""
function _create_strain_diseases(
    scenario::Dict,
    base_beta::Real,
    use_preferred_partners::Bool,
)
    initial_strains = collect(keys(scenario["strains"]))
    gp_combos = generate_gp_reassortments(initial_strains; use_preferred_partners=use_preferred_partners)

    diseases = Rotavirus[]
    for (G, P) in gp_combos
        strain_key = (G, P)
        if haskey(scenario["strains"], strain_key)
            data = scenario["strains"][strain_key]
            fitness    = data["fitness"]
            prevalence = data["prevalence"]
        else
            fitness    = get(scenario, "default_fitness", 1.0)
            prevalence = 0.0
        end

        adjusted_beta = Float64(base_beta) * Float64(fitness)

        push!(diseases, Rotavirus(;
            G = G, P = P,
            init_prev = prevalence,
            beta      = adjusted_beta,
        ))
    end

    return diseases
end
