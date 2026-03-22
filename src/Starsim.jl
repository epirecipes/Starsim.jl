"""
    Starsim

Agent-based modeling framework for simulating disease transmission among agents
via dynamic contact networks. Julia port of the Python
[Starsim](https://starsim.org) framework.

# Overview

Starsim.jl provides a modular architecture where all simulation components
inherit from abstract module types. The six module categories are:

- **Networks** — how agents contact each other
- **Demographics** — births, deaths, aging
- **Diseases** — transmission and progression
- **Interventions** — vaccines, treatments, screening
- **Analyzers** — custom result tracking
- **Connectors** — cross-disease interactions

# Quick start

```julia
using Starsim

sim = Sim(
    n_agents = 5_000,
    networks = RandomNet(n_contacts=10),
    diseases = SIR(beta=0.05),
)
run!(sim)
```
"""
module Starsim

using Random
using StableRNGs
using Distributions
using Statistics
using LinearAlgebra
using SparseArrays
using OrderedCollections
using DataFrames
using Graphs
using Printf
using Serialization
using JSON3
using FunctionWrappers: FunctionWrapper
using RecipesBase

# Re-export StableRNG for custom disease definitions
export StableRNG

# ---- Layer 0: Abstract types and constants ----
include("types.jl")

# ---- Layer 0.5: Global settings (CRN options, verbosity) ----
include("settings.jl")

# ---- Layer 1: Core data structures (no module deps) ----
include("utils.jl")
include("states.jl")
include("time.jl")
include("distributions.jl")

# ---- Layer 2: Parameter and result containers ----
include("parameters.jl")
include("results.jl")

# ---- Layer 3: People (depends on states) ----
include("people.jl")

# ---- Layer 4: Module base (depends on people, pars, results) ----
include("modules.jl")

# ---- Layer 5: Concrete modules (all depend on module base) ----
include("networks.jl")
include("products.jl")
include("interventions.jl")
include("analyzers.jl")
include("connectors.jl")
include("demographics.jl")
include("diseases.jl")

# ---- Layer 6: Simulation engine (depends on all modules) ----
include("loop.jl")
include("sim.jl")
include("run.jl")
include("calibration.jl")
include("plotting.jl")

# ---- Extension stubs (implementations in ext/) ----
"""Compute derivative of a summary result w.r.t. a parameter (ForwardDiff extension)."""
function sensitivity end

"""Compute per-timestep derivatives of a result w.r.t. a parameter (ForwardDiff extension)."""
function sensitivity_timeseries end

"""Compute objective and gradient for calibration (ForwardDiff extension)."""
function gradient_objective end

"""Build an Optimization.jl problem from a Calibration (Optimization extension)."""
function build_optproblem end

"""Run calibration via Optimization.jl solvers (Optimization extension)."""
function run_optimization! end

"""Plot all auto-plot results from a Sim or reduced MultiSim (Makie extension)."""
function plot_sim end

"""Plot results for a single disease module (Makie extension)."""
function plot_disease end

"""Overlay results from multiple simulations on shared axes (Makie extension)."""
function plot_comparison end

"""Compute sensitivity via Enzyme reverse-mode AD (Enzyme extension)."""
function enzyme_sensitivity end

"""Convert simulation state arrays to GPU (Metal extension)."""
function to_gpu end

"""Convert simulation state arrays back to CPU (Metal extension)."""
function to_cpu end

"""Plot simulation results using Makie (Makie extension)."""
function plot_sim end

"""Plot a single disease's results using Makie (Makie extension)."""
function plot_disease end

"""Plot comparison of multiple simulations using Makie (Makie extension)."""
function plot_comparison end

"""Define an epidemiological network ACSet (Catlab extension)."""
function EpiNet end

"""Create an open epidemiological network (structured cospan) for composition (Catlab extension)."""
function OpenEpiNet end

"""Wrap a Starsim module as an EpiSharer for categorical composition (Catlab extension)."""
function EpiSharer end

"""Compose EpiSharers according to an undirected wiring diagram (Catlab extension)."""
function compose_epi end

"""Convert a composed EpiNet into a Starsim Sim (Catlab extension)."""
function to_sim end

"""Build an undirected wiring diagram for composing epidemiological modules (Catlab extension)."""
function epi_uwd end

export plot_sim, plot_disease, plot_comparison
export sensitivity, sensitivity_timeseries, gradient_objective, build_optproblem, run_optimization!
export enzyme_sensitivity, to_gpu, to_cpu, plot_sim, plot_disease, plot_comparison
export EpiNet, OpenEpiNet, EpiSharer, compose_epi, to_sim, epi_uwd

end # module Starsim
