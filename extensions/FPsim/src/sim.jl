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
    initial_cpr::Float64 = -1.0,  # -1 = auto-detect from location data
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

        # Auto-detect initial CPR from location data
        cpr_val = initial_cpr
        if cpr_val < 0
            cpr_val = _load_initial_cpr(location)
        end

        # Load switching matrix and contra use coefficients
        loc_dir = joinpath(DATA_DIR, string(location))
        sm = load_switch_matrix(loc_dir, methods)
        coefs = load_contra_use_coefs(loc_dir)

        push!(interventions, Contraception(;
            methods=methods,
            method_mix=mm,
            switch_matrix=sm,
            contra_use_coefs=coefs,
            initial_cpr=cpr_val,
            prob_use_intercept=0.8,  # Calibrated to match Python fpsim CPR
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

"""Load initial CPR from location use.csv data."""
function _load_initial_cpr(location::Symbol)
    loc_dir = joinpath(DATA_DIR, string(location))
    use_file = joinpath(loc_dir, "use.csv")
    if isfile(use_file)
        df = CSV.read(use_file, DataFrame)
        # "use" column: 0=not using, 1=using; "perc" column has percentage
        for row in eachrow(df)
            if string(row.use) == "1"
                return Float64(row.perc) / 100.0
            end
        end
    end
    return 0.25  # default
end
