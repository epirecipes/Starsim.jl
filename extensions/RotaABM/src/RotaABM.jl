"""
    RotaABM

Multi-strain rotavirus agent-based model built on Starsim.jl.
Port of the Python `rotasim` package with bitmask-vectorized immunity.
"""
module RotaABM

using Starsim
using Random
using StableRNGs
using Distributions
using Statistics
using OrderedCollections

# Scenarios and utilities (no dependencies on other RotaABM types)
include("scenarios.jl")

# Disease
include("rotavirus.jl")

# Connectors
include("immunity.jl")
include("reassortment.jl")

# Interventions
include("interventions.jl")

# Analyzers
include("analyzers.jl")

# Convenience simulation constructor
include("sim.jl")

export Rotavirus, strain, RotaImmunityConnector, RotaReassortmentConnector,
       RotaVaccination, Rotarix, RotaTeq, Rotavac,
       StrainStats, EventStats, AgeStats,
       RotaSim,
       SCENARIOS, PREFERRED_PARTNERS, VACCINATION_SCENARIOS,
       generate_gp_reassortments, list_scenarios, get_scenario,
       validate_scenario, apply_scenario_overrides,
       list_vaccination_scenarios, get_vaccination_scenario,
       get_vaccination_summary,
       hamming_distance, strain_similarity, bitmask_from_gp, match_type

end # module RotaABM
