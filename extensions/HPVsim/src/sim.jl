"""
HPVSim convenience constructor and demographics support.
Creates a complete multi-genotype HPV simulation from a genotype list.
"""

using StableRNGs

# ============================================================================
# Age-specific deaths — realistic mortality for HPV modeling
# ============================================================================

"""
    HPVDeaths <: Starsim.AbstractDemographics

Deaths with age-specific mortality rates. Uses a simple exponential
Gompertz-like model: death rate increases exponentially with age above
a threshold. This produces realistic demographic turnover matching
developing-country age structures (e.g., Nigeria).

Default parameters:
- base_rate=10.0/1000/yr (background mortality for young adults)
- age_threshold=50.0 (age above which mortality accelerates)
- age_scale=15.0 (doubling time in years for mortality above threshold)
"""
mutable struct HPVDeaths <: Starsim.AbstractDemographics
    mod::Starsim.ModuleData
    base_rate::Float64
    age_threshold::Float64
    age_scale::Float64
    rng::StableRNG
end

function HPVDeaths(;
    name::Symbol = :hpv_deaths,
    base_rate::Real = 10.0,
    age_threshold::Real = 50.0,
    age_scale::Real = 15.0,
)
    md = Starsim.ModuleData(name; label="Age-specific deaths")
    HPVDeaths(md, Float64(base_rate), Float64(age_threshold), Float64(age_scale), StableRNG(0))
end

Starsim.module_data(d::HPVDeaths) = d.mod

function Starsim.init_pre!(d::HPVDeaths, sim)
    md = Starsim.module_data(d)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    d.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    npts = md.t.npts
    Starsim.define_results!(d,
        Starsim.Result(:deaths; npts=npts, label="Deaths"),
        Starsim.Result(:cdr; npts=npts, label="Crude death rate", scale=false),
    )
    md.initialized = true
    return d
end

function Starsim.step!(d::HPVDeaths, sim)
    md = Starsim.module_data(d)
    ti = md.t.ti
    dt = sim.pars.dt

    active = sim.people.auids
    n_alive = Float64(length(active))
    base_annual = d.base_rate / 1000.0

    n_deaths = 0
    @inbounds for u in active.values
        age = sim.people.age.raw[u]
        # Gompertz-like: rate increases exponentially above threshold
        if age > d.age_threshold
            excess = age - d.age_threshold
            rate = base_annual * exp(log(2.0) * excess / d.age_scale)
        else
            rate = base_annual
        end
        p_death = 1.0 - exp(-rate * dt)
        if rand(d.rng) < p_death
            Starsim.request_death!(sim.people, Starsim.UIDs([u]), ti)
            n_deaths += 1
        end
    end

    if ti <= length(md.results[:deaths].values)
        md.results[:deaths][ti] = Float64(n_deaths)
        md.results[:cdr][ti] = n_alive > 0 ? (n_deaths / n_alive) * (1.0 / dt) * 1000.0 : 0.0
    end

    return d
end

"""Set initial age distribution using the stable age distribution for a growing population.

For a population with constant birth rate b, the stable age distribution is:
  n(a) ∝ exp(-r*a) * S(a)
where r = birth_rate - death_rate is the population growth rate and S(a) is the
survival function. This produces a young-skewed distribution matching developing
countries like Nigeria (median age ~18-20 years).
"""
function Starsim.init_post!(d::HPVDeaths, sim)
    active = sim.people.auids.values
    base_annual = d.base_rate / 1000.0

    # Find the birth rate from the Births module to compute growth rate
    growth_rate = 0.0
    for (_, demo) in sim.demographics
        if demo isa Starsim.Births
            growth_rate = demo.birth_rate / 1000.0 - base_annual
            break
        end
    end
    growth_rate = max(growth_rate, 0.0)

    for u in active
        age = _sample_stable_age(d.rng, base_annual, d.age_threshold, d.age_scale, growth_rate)
        sim.people.age.raw[u] = age
    end
    return d
end

"""Sample an age from the stable age distribution of a growing population with Gompertz mortality.

The effective hazard is h(a) + r where h(a) is the Gompertz mortality hazard
and r is the population growth rate.
"""
function _sample_stable_age(rng::StableRNG, base_rate::Float64, threshold::Float64, scale::Float64, growth_rate::Float64)
    # Effective rate = base_rate + growth_rate (accounts for population pyramid shape)
    eff_rate = base_rate + growth_rate
    u = rand(rng)
    log_u = log(u)

    # Cumulative hazard H(a) for the effective rate
    # For a <= threshold: H(a) = eff_rate * a
    H_at_threshold = eff_rate * threshold

    if -log_u <= H_at_threshold
        return -log_u / eff_rate
    else
        # Above threshold: Gompertz mortality accelerates, plus growth rate
        # h(a) = base_rate * 2^((a-threshold)/scale) + growth_rate  for a > threshold
        # H(a) = H_at_threshold + growth_rate*(a-threshold) + (scale/ln2)*base_rate*(2^((a-threshold)/scale) - 1)
        # This doesn't have a clean closed form, so use Newton's method
        remaining = -log_u - H_at_threshold
        a = threshold  # start guess
        k = scale / log(2.0) * base_rate
        for _ in 1:50
            x = a - threshold
            gompertz_H = k * (exp(log(2.0) * x / scale) - 1.0)
            total_H = growth_rate * x + gompertz_H
            if abs(total_H - remaining) < 1e-8
                break
            end
            # Derivative: growth_rate + base_rate * 2^(x/scale)
            deriv = growth_rate + base_rate * exp(log(2.0) * x / scale)
            a -= (total_H - remaining) / deriv
            a = max(a, threshold)
        end
        return a
    end
end

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
- `use_demographics::Bool` — include births/deaths (default true, matching Python hpvsim)
- `birth_rate::Real` — crude birth rate per 1000/year (default 35.0, Nigeria-like)
- `death_rate::Real` — base death rate per 1000/year (default 10.0; age-specific increases apply above age 50)
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
    use_demographics::Bool = true,
    birth_rate::Real = 35.0,
    death_rate::Real = 10.0,
    immunity_kwargs::Dict = Dict(),
    network = nothing,
    interventions = nothing,
    analyzers = nothing,
    connectors = nothing,
    demographics = nothing,
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

    # Build demographics (births + deaths for population turnover)
    all_demographics = Starsim.AbstractDemographics[]
    if demographics !== nothing
        if demographics isa AbstractVector
            append!(all_demographics, demographics)
        else
            push!(all_demographics, demographics)
        end
    elseif use_demographics
        push!(all_demographics, Starsim.Births(birth_rate=Float64(birth_rate)))
        push!(all_demographics, HPVDeaths(base_rate=Float64(death_rate)))
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
        demographics = isempty(all_demographics) ? nothing : all_demographics,
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
