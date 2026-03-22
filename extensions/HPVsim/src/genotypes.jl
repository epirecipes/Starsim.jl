"""
HPV genotype definitions — convenience constructors for creating
genotype-specific disease instances from named defaults.
"""

"""
    GenotypeDef

Lightweight descriptor for defining a simulation's genotype composition.
Holds the genotype name, optional parameter overrides, and initial prevalence.
"""
struct GenotypeDef
    name::Symbol
    init_prev::Float64
    overrides::Dict{Symbol, Any}
end

function GenotypeDef(name::Symbol; init_prev::Real=0.01, kwargs...)
    overrides = Dict{Symbol, Any}(k => v for (k, v) in kwargs)
    GenotypeDef(name, Float64(init_prev), overrides)
end

# ============================================================================
# Pre-defined genotype configurations
# ============================================================================

"""Single-genotype configuration for quick testing."""
const SIMPLE_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=1.0),
]

"""Standard 3-genotype configuration: HPV-16, HPV-18, hi5 (other high-risk).

init_prev is each genotype's share of the total age-structured prevalence.
Python hpvsim assigns each initially infected agent to one genotype uniformly
at random, so each genotype gets ~1/n_genotypes of the total prevalence.
"""
const DEFAULT_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=1.0/3),
    GenotypeDef(:hpv18; init_prev=1.0/3),
    GenotypeDef(:hi5;   init_prev=1.0/3),
]

"""Full 5-genotype configuration including low-risk types."""
const FULL_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=1.0/5),
    GenotypeDef(:hpv18; init_prev=1.0/5),
    GenotypeDef(:hi5;   init_prev=1.0/5),
    GenotypeDef(:ohr;   init_prev=1.0/5),
    GenotypeDef(:lr;    init_prev=1.0/5),
]

"""Bivalent vaccine-matched genotypes (HPV-16/18)."""
const BIVALENT_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=0.5),
    GenotypeDef(:hpv18; init_prev=0.5),
]

"""9-valent vaccine-matched individual genotypes + low-risk."""
const NONAVALENT_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=1.0/9),
    GenotypeDef(:hpv18; init_prev=1.0/9),
    GenotypeDef(:hpv31; init_prev=1.0/9),
    GenotypeDef(:hpv33; init_prev=1.0/9),
    GenotypeDef(:hpv45; init_prev=1.0/9),
    GenotypeDef(:hpv52; init_prev=1.0/9),
    GenotypeDef(:hpv58; init_prev=1.0/9),
    GenotypeDef(:hpv6;  init_prev=1.0/9),
    GenotypeDef(:hpv11; init_prev=1.0/9),
]

"""Individual high-risk genotypes (without low-risk)."""
const INDIVIDUAL_HR_GENOTYPES = [
    GenotypeDef(:hpv16; init_prev=1.0/7),
    GenotypeDef(:hpv18; init_prev=1.0/7),
    GenotypeDef(:hpv31; init_prev=1.0/7),
    GenotypeDef(:hpv33; init_prev=1.0/7),
    GenotypeDef(:hpv45; init_prev=1.0/7),
    GenotypeDef(:hpv52; init_prev=1.0/7),
    GenotypeDef(:hpv58; init_prev=1.0/7),
]

# ============================================================================
# Utility functions
# ============================================================================

"""
    genotype_index(name::Symbol, genotypes::Vector{GenotypeDef}) → Int

Find the index of a genotype in a list, or 0 if not found.
"""
function genotype_index(name::Symbol, genotypes::Vector{GenotypeDef})
    for (i, g) in enumerate(genotypes)
        g.name == name && return i
    end
    return 0
end

"""
    genotype_names(genotypes::Vector{GenotypeDef}) → Vector{Symbol}

Extract genotype name symbols from a list of GenotypeDefs.
"""
genotype_names(genotypes::Vector{GenotypeDef}) = [g.name for g in genotypes]
