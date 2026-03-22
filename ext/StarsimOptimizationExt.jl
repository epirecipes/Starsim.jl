"""
    StarsimOptimizationExt

Extension providing Optimization.jl-based calibration backend for Starsim.jl.
Supports derivative-free methods (NelderMead, ParticleSwarm) and gradient-based
methods when combined with ForwardDiff or Enzyme.

# Usage
```julia
using Optimization, OptimizationOptimJL
calib = Calibration(sim=sim, calib_pars=cps, components=comps)
run_optimization!(calib; algorithm=Optim.NelderMead(), maxiters=200)
```
"""
module StarsimOptimizationExt

using Starsim
using Optimization

# ============================================================================
# Build Optimization.jl problem from Calibration
# ============================================================================

"""
    build_optproblem(calib::Starsim.Calibration;
                     adtype=Optimization.AutoFiniteDiff()) → OptimizationProblem

Convert a Starsim `Calibration` to an `Optimization.OptimizationProblem`.

The objective function runs the simulation with given parameters and computes
the composite loss from all `CalibComponent`s.
"""
function Starsim.build_optproblem(calib::Starsim.Calibration;
                                   adtype = Optimization.AutoFiniteDiff())
    # Parameter bounds
    lb = [cp.low for cp in calib.calib_pars]
    ub = [cp.high for cp in calib.calib_pars]
    x0 = [cp.initial for cp in calib.calib_pars]

    function objective(x, p)
        pars_dict = Dict{Symbol, Float64}()
        for (i, cp) in enumerate(calib.calib_pars)
            pars_dict[cp.name] = x[i]
        end
        total = 0.0
        for run_i in 1:calib.n_runs
            sim = deepcopy(calib.base_sim)
            sim.pars.rand_seed = calib.base_sim.pars.rand_seed + run_i
            Starsim.apply_pars!(sim, pars_dict, calib.calib_pars)
            Starsim.reset!(sim)
            Starsim.run!(sim; verbose=0)
            total += Starsim.compute_objective(sim, calib.components)
        end
        return total / calib.n_runs
    end

    optf = OptimizationFunction(objective, adtype)
    prob = OptimizationProblem(optf, x0; lb=lb, ub=ub)
    return prob
end

"""
    run_optimization!(calib::Starsim.Calibration;
                      algorithm = nothing,
                      adtype = Optimization.AutoFiniteDiff(),
                      maxiters::Int = calib.n_trials,
                      verbose::Int = 1,
                      kwargs...) → Calibration

Run calibration using Optimization.jl solvers.

If `algorithm` is not provided, defaults to `Optim.NelderMead()` (requires
OptimizationOptimJL to be loaded).

# Example
```julia
using Optimization, OptimizationOptimJL
run_optimization!(calib; algorithm=Optim.NelderMead(), maxiters=200)
```
"""
function Starsim.run_optimization!(calib::Starsim.Calibration;
                                    algorithm = nothing,
                                    adtype = Optimization.AutoFiniteDiff(),
                                    maxiters::Int = calib.n_trials,
                                    verbose::Int = 1,
                                    kwargs...)
    prob = Starsim.build_optproblem(calib; adtype=adtype)

    if algorithm === nothing
        error("Please provide an optimization algorithm. " *
              "Example: `using OptimizationOptimJL; run_optimization!(calib, algorithm=Optim.NelderMead())`")
    end

    if verbose >= 1
        println("Starting Optimization.jl calibration ($(maxiters) iters, " *
                "$(length(calib.calib_pars)) parameters)")
    end

    sol = Optimization.solve(prob, algorithm; maxiters=maxiters, kwargs...)

    # Store results
    calib.best_objective = sol.objective
    for (i, cp) in enumerate(calib.calib_pars)
        calib.best_pars[cp.name] = sol.u[i]
    end
    calib.complete = true

    if verbose >= 1
        println("Calibration complete. Best objective: $(round(sol.objective, digits=6))")
        println("Best parameters: $(calib.best_pars)")
    end

    return calib
end

# ============================================================================
# Stub for build_optproblem in the main module
# ============================================================================

end # module
