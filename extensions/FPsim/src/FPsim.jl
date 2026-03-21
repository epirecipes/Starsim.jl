"""
    FPsim

Family planning agent-based model built on Starsim.jl.
Port of the Python `fpsim` package for reproductive lifecycle simulation
with contraceptive method choice and location-specific demographics.
"""
module FPsim

using Starsim
using Random
using StableRNGs
using Distributions
using Statistics
using OrderedCollections
using CSV
using DataFrames
using Interpolations

# Data directory
const DATA_DIR = joinpath(@__DIR__, "..", "data")

# Global constants matching Python fpsim defaults
const MPY = 12       # Months per year
const MIN_AGE = 15   # Minimum age for contraception
const MAX_AGE = 99   # Maximum age
const MAX_AGE_PREG = 50  # Maximum age for pregnancy
const MAX_PARITY = 20    # Maximum parity tracked

# Parameters and data loading
include("parameters.jl")
include("locations.jl")

# Contraceptive methods
include("methods.jl")

# Core connector: FPmod (reproductive lifecycle state machine)
include("fpmod.jl")

# Analyzers
include("analyzers.jl")

# Convenience simulation constructor
include("sim.jl")

export # Parameters
       FPPars, prob_per_timestep, interp_year,
       # Locations
       load_location_data, VALID_LOCATIONS,
       # Methods
       Method, MethodMix, load_methods, load_method_mix,
       method_by_name, method_index, sample_method,
       # Connectors
       FPmod, Contraception,
       # Analyzers
       FPAnalyzer, ASFR_BINS,
       # Sim
       FPSim

end # module FPsim
