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
        # Derive initiation rate to sustain the target CPR at equilibrium
        # Equilibrium: init_rate * (1 - CPR) = avg_disc_rate * CPR
        # => init_rate = avg_disc_rate * CPR / (1 - CPR)
        avg_disc = 0.0
        n_active = 0
        for (i, name) in enumerate(mm.method_names)
            idx = findfirst(m -> m.name == name, methods)
            if idx !== nothing
                avg_disc += mm.mix_probs[i] * methods[idx].discontinuation
                n_active += 1
            end
        end
        target_init = if cpr_val > 0 && cpr_val < 1 && avg_disc > 0
            avg_disc * cpr_val / (1 - cpr_val)
        else
            initiation_rate
        end
        push!(interventions, Contraception(;
            methods=methods,
            method_mix=mm,
            initiation_rate=target_init,
            initial_cpr=cpr_val,
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
