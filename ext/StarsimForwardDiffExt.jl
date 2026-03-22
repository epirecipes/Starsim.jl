"""
    StarsimForwardDiffExt

Extension providing sensitivity analysis and gradient-based calibration for
Starsim.jl via ForwardDiff.jl.

Since agent-based simulations involve stochastic branching and discrete events,
we use central finite differences for parameter sensitivity (common random number
seeds ensure the only variation is from the parameter perturbation). ForwardDiff
is used for smooth calibration objectives when the mapping from parameters to
objective is locally smooth.

# Features
- `sensitivity(sim, par_name; ...)` — ∂(summary(result))/∂(parameter) via central FD
- `sensitivity_timeseries(sim, par_name; ...)` — per-timestep sensitivity
- `gradient_objective(calib, x)` — objective + gradient for calibration
"""
module StarsimForwardDiffExt

using Starsim
using ForwardDiff

# ============================================================================
# Helper: run sim with a parameter set to a specific value
# ============================================================================

function _run_with_par(sim::Starsim.Sim, par_name::Symbol, module_name::Symbol, x::Float64)
    sim_copy = deepcopy(sim)
    d = sim_copy.diseases[module_name]
    dd = Starsim.disease_data(d)
    if par_name == :beta
        dd.beta = x
    else
        dd.mod.pars[par_name] = x
    end
    Starsim.reset!(sim_copy)
    Starsim.run!(sim_copy; verbose=0)
    return Starsim.module_results(sim_copy.diseases[module_name])
end

# ============================================================================
# Sensitivity analysis (central finite differences with CRN)
# ============================================================================

"""
    sensitivity(sim::Starsim.Sim, par_name::Symbol;
                module_name::Symbol=:sir, result::Symbol=:prevalence,
                summary::Function=maximum, h::Float64=1e-6) → Float64

Compute ∂(summary(result))/∂(parameter) using central finite differences.

The same random seed is used for both the +h and -h simulations (common random
numbers), so the derivative estimate captures the effect of the parameter change
alone, not stochastic noise.

# Example
```julia
using ForwardDiff
sim = Sim(diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
          networks=RandomNet(n_contacts=10), start=0.0, stop=100.0, dt=1.0, rand_seed=42)
init!(sim); run!(sim; verbose=0)
dsdb = sensitivity(sim, :beta; module_name=:sir, result=:prevalence, summary=maximum)
```
"""
function Starsim.sensitivity(sim::Starsim.Sim, par_name::Symbol;
                              module_name::Symbol = :sir,
                              result::Symbol = :prevalence,
                              summary::Function = maximum,
                              h::Float64 = 1e-3)
    disease = sim.diseases[module_name]
    dd = Starsim.disease_data(disease)
    x0 = par_name == :beta ? Float64(dd.beta) : Float64(dd.mod.pars[par_name])

    res_plus  = _run_with_par(sim, par_name, module_name, x0 + h)
    res_minus = _run_with_par(sim, par_name, module_name, x0 - h)

    return (summary(res_plus[result].values) - summary(res_minus[result].values)) / (2h)
end

"""
    sensitivity_timeseries(sim::Starsim.Sim, par_name::Symbol;
                           module_name::Symbol=:sir, result::Symbol=:prevalence,
                           h::Float64=1e-3) → Vector{Float64}

Compute per-timestep ∂result[t]/∂parameter using central finite differences.
Returns a vector of length `npts`.
"""
function Starsim.sensitivity_timeseries(sim::Starsim.Sim, par_name::Symbol;
                                         module_name::Symbol = :sir,
                                         result::Symbol = :prevalence,
                                         h::Float64 = 1e-3)
    disease = sim.diseases[module_name]
    dd = Starsim.disease_data(disease)
    x0 = par_name == :beta ? Float64(dd.beta) : Float64(dd.mod.pars[par_name])

    res_plus  = _run_with_par(sim, par_name, module_name, x0 + h)
    res_minus = _run_with_par(sim, par_name, module_name, x0 - h)

    return (res_plus[result].values .- res_minus[result].values) ./ (2h)
end

# ============================================================================
# Gradient-based calibration objective (finite differences per component)
# ============================================================================

"""
    gradient_objective(calib::Starsim.Calibration, x::Vector{Float64};
                       h::Float64=1e-6) → (Float64, Vector{Float64})

Compute calibration objective and its gradient w.r.t. the parameter vector
using central finite differences. Each parameter is perturbed independently.

Returns `(objective_value, gradient_vector)`.
"""
function Starsim.gradient_objective(calib::Starsim.Calibration, x::Vector{Float64};
                                     h::Float64 = 1e-6)
    function eval_obj(params::Vector{Float64})
        pars_dict = Dict{Symbol, Float64}()
        for (i, cp) in enumerate(calib.calib_pars)
            pars_dict[cp.name] = params[i]
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

    val = eval_obj(x)
    grad = similar(x)
    for i in eachindex(x)
        x_plus = copy(x); x_plus[i] += h
        x_minus = copy(x); x_minus[i] -= h
        grad[i] = (eval_obj(x_plus) - eval_obj(x_minus)) / (2h)
    end
    return val, grad
end

end # module
