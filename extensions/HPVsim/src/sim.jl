"""
HPVSim convenience constructor.
Creates a complete multi-genotype HPV simulation from a genotype list.
"""

"""
    HPVSim(; genotypes, beta, n_agents, start, stop, dt, kwargs...)

Convenience constructor for a multi-genotype HPV simulation.

# Keyword arguments
- `genotypes` — list of GenotypeDef or Symbol genotype names (default: DEFAULT_GENOTYPES)
- `beta::Float64` — base transmission rate (default 0.25)
- `n_agents::Int` — number of agents (default 5000)
- `start::Real` — start year (default 2000.0)
- `stop::Real` — stop year (default 2060.0)
- `dt::Real` — timestep in years (default 0.25, quarterly)
- `rand_seed::Int` — RNG seed (default 0)
- `use_immunity::Bool` — include HPVImmunityConnector (default true)
- `immunity_kwargs` — keyword arguments for HPVImmunityConnector
- `network` — custom network (default: HPVNet())
- `interventions` — optional interventions
- `analyzers` — optional analyzers
- `connectors` — additional connectors (immunity is auto-added if use_immunity=true)
"""
function HPVSim(;
    genotypes = DEFAULT_GENOTYPES,
    beta::Real = DEFAULT_BETA,
    n_agents::Int = 5000,
    start::Real = 2000.0,
    stop::Real = 2060.0,
    dt::Real = 0.25,
    rand_seed::Int = 0,
    use_immunity::Bool = true,
    immunity_kwargs::Dict = Dict(),
    network = nothing,
    interventions = nothing,
    analyzers = nothing,
    connectors = nothing,
    kwargs...,
)
    # Parse genotype specifications
    genotype_defs = _parse_genotypes(genotypes)

    # Create disease instances
    diseases = HPVGenotype[]
    for gd in genotype_defs
        gp = get_genotype_params(gd.name)
        # Apply overrides
        push!(diseases, HPVGenotype(;
            genotype = gd.name,
            init_prev = gd.init_prev,
            beta = Float64(beta),
            params = gp,
        ))
    end

    # Default network
    if network === nothing
        network = HPVNet()
    end

    # Build connectors
    all_connectors = Starsim.AbstractConnector[]
    if use_immunity
        push!(all_connectors, HPVImmunityConnector(; immunity_kwargs...))
    end
    if connectors !== nothing
        if connectors isa AbstractVector
            append!(all_connectors, connectors)
        else
            push!(all_connectors, connectors)
        end
    end

    sim = Starsim.Sim(;
        n_agents   = n_agents,
        start      = start,
        stop       = stop,
        dt         = dt,
        rand_seed  = rand_seed,
        networks   = network,
        diseases   = diseases,
        connectors = isempty(all_connectors) ? nothing : all_connectors,
        interventions = interventions,
        analyzers  = analyzers,
        kwargs...,
    )

    return sim
end

"""Parse genotype specifications into GenotypeDef list."""
function _parse_genotypes(genotypes)
    if genotypes isa Vector{GenotypeDef}
        return genotypes
    elseif genotypes isa Vector{Symbol}
        return [GenotypeDef(g) for g in genotypes]
    elseif genotypes isa GenotypeDef
        return [genotypes]
    elseif genotypes isa Symbol
        return [GenotypeDef(genotypes)]
    else
        error("genotypes must be Vector{GenotypeDef}, Vector{Symbol}, GenotypeDef, or Symbol")
    end
end
