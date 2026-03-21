"""
Scenarios and utility functions for multi-strain rotavirus simulations.
Port of Python `rotasim.utils`.
"""

# ============================================================================
# Scenarios — unified strain/fitness/prevalence definitions
# ============================================================================

const SCENARIOS = Dict{String, Dict{String, Any}}(
    "simple" => Dict{String, Any}(
        "description" => "Simple two-strain scenario — G1P8 and G2P4 with equal fitness",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.01),
            (2, 4) => Dict("fitness" => 1.0, "prevalence" => 0.01),
        ),
        "default_fitness" => 1.0,
    ),
    "baseline" => Dict{String, Any}(
        "description" => "Baseline scenario — common global strains with equal fitness",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.015),
            (2, 4) => Dict("fitness" => 1.0, "prevalence" => 0.008),
            (3, 8) => Dict("fitness" => 1.0, "prevalence" => 0.007),
        ),
        "default_fitness" => 1.0,
    ),
    "realistic_competition" => Dict{String, Any}(
        "description" => "G1P8 dominant with realistic strain competition",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.015),
            (2, 4) => Dict("fitness" => 0.2, "prevalence" => 0.008),
            (3, 8) => Dict("fitness" => 0.4, "prevalence" => 0.007),
            (4, 8) => Dict("fitness" => 0.5, "prevalence" => 0.005),
        ),
        "default_fitness" => 0.05,
    ),
    "balanced_competition" => Dict{String, Any}(
        "description" => "G1P8 dominant with moderate balanced competition",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.015),
            (2, 4) => Dict("fitness" => 0.6, "prevalence" => 0.008),
            (3, 8) => Dict("fitness" => 0.9, "prevalence" => 0.007),
            (4, 8) => Dict("fitness" => 0.9, "prevalence" => 0.005),
        ),
        "default_fitness" => 0.2,
    ),
    "high_diversity" => Dict{String, Any}(
        "description" => "High diversity with 12 strains and varied fitness",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8)  => Dict("fitness" => 1.0,  "prevalence" => 0.012),
            (2, 4)  => Dict("fitness" => 0.7,  "prevalence" => 0.007),
            (3, 8)  => Dict("fitness" => 0.85, "prevalence" => 0.005),
            (4, 8)  => Dict("fitness" => 0.88, "prevalence" => 0.004),
            (9, 8)  => Dict("fitness" => 0.95, "prevalence" => 0.003),
            (12, 8) => Dict("fitness" => 0.93, "prevalence" => 0.003),
            (9, 6)  => Dict("fitness" => 0.85, "prevalence" => 0.002),
            (12, 6) => Dict("fitness" => 0.90, "prevalence" => 0.002),
            (9, 4)  => Dict("fitness" => 0.90, "prevalence" => 0.002),
            (1, 6)  => Dict("fitness" => 0.6,  "prevalence" => 0.002),
            (2, 8)  => Dict("fitness" => 0.6,  "prevalence" => 0.002),
            (2, 6)  => Dict("fitness" => 0.6,  "prevalence" => 0.002),
        ),
        "default_fitness" => 0.4,
    ),
    "low_diversity" => Dict{String, Any}(
        "description" => "Low diversity with 4 main competitive strains",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 0.98, "prevalence" => 0.020),
            (2, 4) => Dict("fitness" => 0.7,  "prevalence" => 0.012),
            (3, 8) => Dict("fitness" => 0.8,  "prevalence" => 0.008),
            (4, 8) => Dict("fitness" => 0.8,  "prevalence" => 0.005),
        ),
        "default_fitness" => 0.5,
    ),
    "emergence_scenario" => Dict{String, Any}(
        "description" => "Scenario for studying strain emergence with weak background",
        "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
            (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.015),
            (2, 4) => Dict("fitness" => 0.4, "prevalence" => 0.005),
            (3, 8) => Dict("fitness" => 0.7, "prevalence" => 0.003),
        ),
        "default_fitness" => 0.05,
    ),
)

const PREFERRED_PARTNERS = Dict{Int, Vector{Int}}(
    1  => [6, 8],
    2  => [4, 6, 8],
    3  => [6, 8],
    4  => [8],
    9  => [4, 6, 8],
    12 => [6, 8],
)

# ============================================================================
# Utility functions
# ============================================================================

"""
    generate_gp_reassortments(initial_strains; use_preferred_partners=false)

Generate all possible (G, P) combinations from initial strains.
"""
function generate_gp_reassortments(
    initial_strains::Vector{Tuple{Int,Int}};
    use_preferred_partners::Bool = false,
)
    isempty(initial_strains) && error("initial_strains cannot be empty")

    unique_G = sort(unique(g for (g, _) in initial_strains))
    unique_P = sort(unique(p for (_, p) in initial_strains))

    combos = Tuple{Int,Int}[]
    if use_preferred_partners
        for g in unique_G
            haskey(PREFERRED_PARTNERS, g) || continue
            for p in unique_P
                p in PREFERRED_PARTNERS[g] || continue
                push!(combos, (g, p))
            end
        end
    else
        for g in unique_G, p in unique_P
            push!(combos, (g, p))
        end
    end
    return combos
end

"""List available built-in scenarios."""
list_scenarios() = Dict(k => v["description"] for (k, v) in SCENARIOS)

"""Get a scenario by name."""
function get_scenario(name::AbstractString)
    haskey(SCENARIOS, name) || error("Unknown scenario '$name'. Available: $(keys(SCENARIOS))")
    return deepcopy(SCENARIOS[name])
end

"""
    validate_scenario(scenario)

Validate and return a scenario dict. Accepts a name string or a Dict.
"""
function validate_scenario(scenario::AbstractString)
    return get_scenario(scenario)
end

function validate_scenario(scenario::Dict)
    haskey(scenario, "strains") || error("Custom scenario must contain 'strains' key")
    strains = scenario["strains"]
    strains isa Dict || error("Scenario 'strains' must be a Dict")
    isempty(strains) && error("Scenario must contain at least one strain")
    for (strain, data) in strains
        strain isa Tuple{Int,Int} || error("Strain key must be (G,P) tuple, got $strain")
        data isa Dict || error("Strain data must be Dict for strain $strain")
        haskey(data, "fitness") && haskey(data, "prevalence") ||
            error("Strain data must contain 'fitness' and 'prevalence' for strain $strain")
    end
    if !haskey(scenario, "default_fitness")
        scenario["default_fitness"] = 1.0
    end
    return scenario
end

"""
    apply_scenario_overrides(scenario; override_fitness, override_prevalence, override_strains)

Apply override parameters to a (deep-copied) scenario.
"""
function apply_scenario_overrides(
    scenario::Dict;
    override_fitness = nothing,
    override_prevalence = nothing,
    override_strains = nothing,
)
    result = deepcopy(scenario)

    if override_strains !== nothing
        for (strain, data) in override_strains
            result["strains"][strain] = deepcopy(data)
        end
    end

    if override_fitness !== nothing
        if override_fitness isa Real
            for strain in keys(result["strains"])
                result["strains"][strain]["fitness"] = Float64(override_fitness)
            end
        elseif override_fitness isa Dict
            for (strain, f) in override_fitness
                if haskey(result["strains"], strain)
                    result["strains"][strain]["fitness"] = Float64(f)
                end
            end
        end
    end

    if override_prevalence !== nothing
        if override_prevalence isa Real
            for strain in keys(result["strains"])
                result["strains"][strain]["prevalence"] = Float64(override_prevalence)
            end
        elseif override_prevalence isa Dict
            for (strain, p) in override_prevalence
                if haskey(result["strains"], strain)
                    result["strains"][strain]["prevalence"] = Float64(p)
                end
            end
        end
    end

    return result
end

# ============================================================================
# Vaccination scenario presets
# ============================================================================

const VACCINATION_SCENARIOS = Dict{String, Dict{String, Any}}(
    "rotarix_baseline" => Dict{String, Any}(
        "description" => "Rotarix (monovalent G1P8) on baseline strain scenario",
        "scenario"    => "baseline",
        "vaccine"     => "rotarix",
        "start_year"  => 0.25,
        "uptake_prob" => 0.8,
    ),
    "rotateq_baseline" => Dict{String, Any}(
        "description" => "RotaTeq (pentavalent) on baseline strain scenario",
        "scenario"    => "baseline",
        "vaccine"     => "rotateq",
        "start_year"  => 0.25,
        "uptake_prob" => 0.8,
    ),
    "rotarix_high_diversity" => Dict{String, Any}(
        "description" => "Rotarix on high-diversity strain scenario",
        "scenario"    => "high_diversity",
        "vaccine"     => "rotarix",
        "start_year"  => 0.25,
        "uptake_prob" => 0.8,
    ),
    "rotateq_high_diversity" => Dict{String, Any}(
        "description" => "RotaTeq on high-diversity strain scenario",
        "scenario"    => "high_diversity",
        "vaccine"     => "rotateq",
        "start_year"  => 0.25,
        "uptake_prob" => 0.8,
    ),
    "no_vaccination" => Dict{String, Any}(
        "description" => "Baseline scenario with no vaccination (control)",
        "scenario"    => "baseline",
        "vaccine"     => nothing,
        "start_year"  => nothing,
        "uptake_prob" => 0.0,
    ),
)

"""List available vaccination scenarios."""
list_vaccination_scenarios() = Dict(k => v["description"] for (k, v) in VACCINATION_SCENARIOS)

"""Get a vaccination scenario by name."""
function get_vaccination_scenario(name::AbstractString)
    haskey(VACCINATION_SCENARIOS, name) || error("Unknown vaccination scenario '$name'. Available: $(keys(VACCINATION_SCENARIOS))")
    return deepcopy(VACCINATION_SCENARIOS[name])
end
