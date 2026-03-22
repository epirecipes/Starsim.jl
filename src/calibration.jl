"""
Calibration for Starsim.jl.

Mirrors Python starsim's `calibration.py`. Provides model fitting via
derivative-free optimization (like Optuna) and gradient-based methods
via Julia's AD ecosystem.
"""

using Statistics

# ============================================================================
# CalibPar — calibration parameter specification
# ============================================================================

"""
    CalibPar

Specification for a single calibration parameter.

# Fields
- `name::Symbol` — parameter name
- `low::Float64` — lower bound
- `high::Float64` — upper bound
- `initial::Float64` — starting value
- `module_name::Symbol` — target module (e.g., `:sir`)
- `par_name::Symbol` — parameter within the module (e.g., `:beta`)
"""
struct CalibPar
    name::Symbol
    low::Float64
    high::Float64
    initial::Float64
    module_name::Symbol
    par_name::Symbol
end

function CalibPar(name::Symbol;
                  low::Real, high::Real,
                  initial::Union{Real, Nothing}=nothing,
                  module_name::Symbol=:sir,
                  par_name::Symbol=name)
    init = initial === nothing ? (low + high) / 2 : Float64(initial)
    CalibPar(name, Float64(low), Float64(high), init, module_name, par_name)
end

export CalibPar

# ============================================================================
# CalibComponent — likelihood component
# ============================================================================

"""
    CalibComponent

A single likelihood/objective component for calibration.

# Fields
- `name::Symbol` — component name
- `target_data::Vector{Float64}` — observed data
- `sim_result::Symbol` — result key in the simulation
- `disease_name::Symbol` — disease module name
- `weight::Float64` — weight in the composite objective
- `loss_fn::Function` — loss(simulated, observed) → Float64
"""
struct CalibComponent
    name::Symbol
    target_data::Vector{Float64}
    sim_result::Symbol
    disease_name::Symbol
    weight::Float64
    loss_fn::Function
end

function CalibComponent(name::Symbol;
                        target_data::Vector{Float64},
                        sim_result::Symbol,
                        disease_name::Symbol = :sir,
                        weight::Real = 1.0,
                        loss_fn::Function = mse_loss)
    CalibComponent(name, target_data, sim_result, disease_name, Float64(weight), loss_fn)
end

"""Default MSE loss function."""
function mse_loss(simulated::Vector{Float64}, observed::Vector{Float64})
    n = min(length(simulated), length(observed))
    n == 0 && return Inf
    return sum((simulated[1:n] .- observed[1:n]).^2) / n
end

"""Negative log-likelihood under Normal distribution."""
function normal_loss(simulated::Vector{Float64}, observed::Vector{Float64}; sigma::Float64=1.0)
    n = min(length(simulated), length(observed))
    n == 0 && return Inf
    return sum((simulated[1:n] .- observed[1:n]).^2) / (2 * sigma^2) + n * log(sigma)
end

export CalibComponent, mse_loss, normal_loss

# ============================================================================
# Calibration
# ============================================================================

"""
    Calibration

Model calibration workflow. Fits simulation parameters to data using
derivative-free or gradient-based optimization.

# Keyword arguments
- `sim::Sim` — base simulation
- `calib_pars::Vector{CalibPar}` — parameters to calibrate
- `components::Vector{CalibComponent}` — objective components
- `n_trials::Int` — number of optimization trials (default 100)
- `n_runs::Int` — runs per trial for averaging (default 1)

# Example
```julia
calib = Calibration(
    sim = Sim(diseases=SIR(beta=0.05)),
    calib_pars = [CalibPar(:beta; low=0.01, high=0.2, module_name=:sir)],
    components = [CalibComponent(:prev; target_data=data, sim_result=:prevalence, disease_name=:sir)],
)
run!(calib)
best = calib.best_pars
```
"""
mutable struct Calibration
    base_sim::Sim
    calib_pars::Vector{CalibPar}
    components::Vector{CalibComponent}
    n_trials::Int
    n_runs::Int

    # Results
    best_pars::Dict{Symbol, Float64}
    best_objective::Float64
    trial_history::Vector{Tuple{Dict{Symbol, Float64}, Float64}}
    complete::Bool
end

function Calibration(;
    sim::Sim,
    calib_pars::Vector{CalibPar},
    components::Vector{CalibComponent},
    n_trials::Int = 100,
    n_runs::Int = 1,
)
    Calibration(
        sim, calib_pars, components, n_trials, n_runs,
        Dict{Symbol, Float64}(), Inf,
        Tuple{Dict{Symbol, Float64}, Float64}[],
        false
    )
end

function Base.show(io::IO, c::Calibration)
    status = c.complete ? "complete (obj=$(round(c.best_objective, digits=4)))" : "created"
    print(io, "Calibration($(length(c.calib_pars)) pars, $(length(c.components)) components, " *
              "$(c.n_trials) trials, $status)")
end

"""
    apply_pars!(sim::Sim, pars::Dict{Symbol, Float64}, calib_pars::Vector{CalibPar})

Apply calibration parameters to a simulation.
"""
function apply_pars!(sim::Sim, pars::Dict{Symbol, Float64}, calib_pars::Vector{CalibPar})
    for cp in calib_pars
        val = pars[cp.name]
        mod_name = cp.module_name

        # Apply to disease module
        if haskey(sim.diseases, mod_name)
            disease = sim.diseases[mod_name]
            dd = disease_data(disease)
            if cp.par_name == :beta
                dd.beta = val
            else
                dd.mod.pars[cp.par_name] = val
            end
        end
    end
    return sim
end

"""
    compute_objective(sim::Sim, components::Vector{CalibComponent}) → Float64

Compute the total objective (loss) for a simulation run.
"""
function compute_objective(sim::Sim, components::Vector{CalibComponent})
    total = 0.0
    for comp in components
        if haskey(sim.diseases, comp.disease_name)
            res = module_results(sim.diseases[comp.disease_name])
            if haskey(res, comp.sim_result)
                sim_vals = res[comp.sim_result].values
                loss = comp.loss_fn(sim_vals, comp.target_data)
                total += comp.weight * loss
            else
                total += 1e10  # Penalize missing results
            end
        else
            total += 1e10
        end
    end
    return total
end

"""
    run!(calib::Calibration; verbose::Int=1)

Run calibration using random search (simple derivative-free optimization).
"""
function run!(calib::Calibration; verbose::Int=1)
    rng = StableRNG(calib.base_sim.pars.rand_seed + 9999)

    if verbose >= 1
        println("Starting calibration ($(calib.n_trials) trials, " *
                "$(length(calib.calib_pars)) parameters)")
    end

    for trial in 1:calib.n_trials
        # Sample parameters
        pars = Dict{Symbol, Float64}()
        for cp in calib.calib_pars
            pars[cp.name] = cp.low + rand(rng) * (cp.high - cp.low)
        end

        # Run simulation(s)
        total_obj = 0.0
        for run_i in 1:calib.n_runs
            sim = deepcopy(calib.base_sim)
            sim.pars.rand_seed = calib.base_sim.pars.rand_seed + trial * 100 + run_i
            apply_pars!(sim, pars, calib.calib_pars)
            reset!(sim)
            run!(sim; verbose=0)
            total_obj += compute_objective(sim, calib.components)
        end
        avg_obj = total_obj / calib.n_runs

        push!(calib.trial_history, (pars, avg_obj))

        if avg_obj < calib.best_objective
            calib.best_objective = avg_obj
            calib.best_pars = copy(pars)
            if verbose >= 2
                println("  Trial $trial: new best = $(round(avg_obj, digits=6)) ($pars)")
            end
        end
    end

    calib.complete = true
    if verbose >= 1
        println("Calibration complete. Best objective: $(round(calib.best_objective, digits=6))")
        println("Best parameters: $(calib.best_pars)")
    end

    return calib
end

export Calibration, apply_pars!, compute_objective
