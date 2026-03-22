"""
    HPVsim

Multi-genotype HPV and cervical cancer model built on Starsim.jl.
Port of the Python `hpvsim` package with Starsim-native architecture.

Each HPV genotype is modeled as an independent `AbstractInfection` disease
with CIN progression (Normal → Infected → CIN1 → CIN2 → CIN3 → Cancer).
Cross-genotype immunity is handled by `HPVImmunityConnector`.
"""
module HPVsim

using Starsim
using Random
using StableRNGs
using Distributions
using Statistics
using OrderedCollections

# Parameters and defaults
include("parameters.jl")

# Genotype definitions
include("genotypes.jl")

# Disease model (depends on parameters — forward reference to HPVImmunityConnector)
include("hpv_disease.jl")

# Immunity connector (depends on HPVGenotype)
include("immunity.jl")

# Network helpers
include("networks.jl")

# Interventions (depends on HPVGenotype, HPVImmunityConnector)
include("interventions.jl")

# Convenience simulation constructor
include("sim.jl")

export HPVGenotype, GenotypeParams, GenotypeDef,
       HPVImmunityConnector,
       HPVVaccination, HPVScreening, HPVTherapeuticVaccine,
       HPVDeaths,
       TreatmentType, ABLATION, EXCISION, GENERIC,
       get_treatment_efficacy,
       HPVSexualNet, HPVNet, HPVSim,
       get_genotype_params, list_genotypes, is_high_risk, is_low_risk,
       genotype_names, genotype_index,
       logf2, intlogf2, indef_intlogf2,
       compute_cin_prob, compute_cancer_prob,
       sample_lognormal_duration,
       DEFAULT_GENOTYPES, SIMPLE_GENOTYPES, FULL_GENOTYPES,
       BIVALENT_GENOTYPES, NONAVALENT_GENOTYPES, INDIVIDUAL_HR_GENOTYPES,
       DEFAULT_BETA, M2F_TRANS_RATIO,
       HPV16_PARAMS, HPV18_PARAMS, HI5_PARAMS, OHR_PARAMS, LR_PARAMS, HR_PARAMS,
       HPV31_PARAMS, HPV33_PARAMS, HPV45_PARAMS, HPV52_PARAMS, HPV58_PARAMS,
       HPV6_PARAMS, HPV11_PARAMS,
       GENOTYPE_REGISTRY,
       DEFAULT_CROSS_IMMUNITY, DEFAULT_VAX_PARAMS,
       DEFAULT_SCREENING_PARAMS, DEFAULT_TREATMENT_PARAMS,
       AGE_MIXING_BINS, MARITAL_MIXING_MATRIX, CASUAL_MIXING_MATRIX,
       default_layer_probs

end # module HPVsim
