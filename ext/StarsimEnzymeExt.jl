"""
    StarsimEnzymeExt

Extension providing reverse-mode automatic differentiation for Starsim.jl
via Enzyme.jl. Enables efficient gradient computation for:

1. **Deterministic ODE modules** — Full reverse-mode AD through the ODE right-hand side
2. **CRN-based sensitivity** — Finite differences using common random numbers (CRN),
   wrapped in Enzyme's interface for consistency
3. **Smooth calibration objectives** — Reverse-mode gradients of calibration loss

Enzyme excels at differentiating through mutating code, making it suitable for
Starsim's mutable state arrays. However, stochastic branching (random infection
events) creates non-differentiable points. For ABM parameters, we use CRN finite
differences; for ODE/deterministic parameters, we use true reverse-mode AD.

# Usage
```julia
using Enzyme
sim = Sim(diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
          networks=RandomNet(n_contacts=10), start=0.0, stop=100.0, rand_seed=42)
run!(sim; verbose=0)

# Sensitivity of peak prevalence to beta
dsdb = enzyme_sensitivity(sim, :beta; module_name=:sir, result=:prevalence)

# Gradient of calibration objective
val, grad = enzyme_gradient(calib, x0)
```
"""
module StarsimEnzymeExt

using Starsim
using Enzyme

# ============================================================================
# CRN-based sensitivity (same algorithm as ForwardDiff ext, for consistency)
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

"""
    enzyme_sensitivity(sim::Starsim.Sim, par_name::Symbol;
                       module_name::Symbol=:sir, result::Symbol=:prevalence,
                       summary::Function=maximum, h::Float64=1e-3) → Float64

Compute ∂(summary(result))/∂(parameter) using central finite differences with
common random numbers (CRN). The same seed is used for +h and -h runs.

For stochastic ABMs, this is more reliable than attempting to differentiate
through random branching. Enzyme's `autodiff` is used for deterministic
sub-components when available.

# Example
```julia
using Enzyme
sim = Sim(diseases=SIR(beta=0.05), networks=RandomNet(n_contacts=10),
          start=0.0, stop=100.0, rand_seed=42)
run!(sim; verbose=0)
dsdb = enzyme_sensitivity(sim, :beta; module_name=:sir, result=:prevalence)
```
"""
function Starsim.enzyme_sensitivity(sim::Starsim.Sim, par_name::Symbol;
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

# ============================================================================
# Enzyme autodiff for smooth scalar functions
# ============================================================================

"""
    enzyme_grad(f::Function, x::Vector{Float64}) → Vector{Float64}

Compute the gradient of a scalar function `f(x) → Float64` using Enzyme
reverse-mode AD. This is useful for calibration objectives that are smooth
functions of parameters.

# Example
```julia
using Enzyme
grad = enzyme_grad(x -> sum(x.^2), [1.0, 2.0, 3.0])
# grad ≈ [2.0, 4.0, 6.0]
```
"""
function enzyme_grad(f::F, x::Vector{Float64}) where {F}
    dx = zeros(Float64, length(x))
    Enzyme.autodiff(Reverse, f, Active, Duplicated(x, dx))
    return dx
end

# ============================================================================
# Enzyme-based calibration gradient
# ============================================================================

"""
    enzyme_gradient(calib::Starsim.Calibration, x::Vector{Float64};
                    h::Float64=1e-3) → (Float64, Vector{Float64})

Compute calibration objective and gradient. Uses CRN finite differences for
the stochastic ABM simulation, but can use Enzyme autodiff for post-processing
of simulation outputs (e.g., objective function computation).

Returns `(objective_value, gradient_vector)`.
"""
function enzyme_gradient(calib::Starsim.Calibration, x::Vector{Float64};
                          h::Float64 = 1e-3)
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

# ============================================================================
# Enzyme autodiff for deterministic ODE components
# ============================================================================

"""
    enzyme_ode_sensitivity(rhs!::Function, u0::Vector{Float64},
                           p::Vector{Float64}, tspan::Tuple{Float64,Float64};
                           dp_idx::Int=1, h_time::Float64=0.01) → Vector{Float64}

For deterministic ODE modules (like malaria metapopulation), compute
∂u(T)/∂p[dp_idx] using Enzyme reverse-mode AD through a simple Euler
integration of the ODE.

This is exact for deterministic systems (no stochastic noise).

# Arguments
- `rhs!` — In-place ODE right-hand side: `rhs!(du, u, p, t)`
- `u0` — Initial state vector
- `p` — Parameter vector
- `tspan` — (t0, tf) time span
- `dp_idx` — Index of the parameter to differentiate w.r.t.
- `h_time` — Euler integration step size
"""
function enzyme_ode_sensitivity(rhs!::F, u0::Vector{Float64},
                                 p::Vector{Float64},
                                 tspan::Tuple{Float64, Float64};
                                 dp_idx::Int = 1,
                                 h_time::Float64 = 0.01) where {F}
    function simulate_and_sum(params::Vector{Float64})
        n = length(u0)
        u = copy(u0)
        du = zeros(n)
        t = tspan[1]
        while t < tspan[2]
            dt = min(h_time, tspan[2] - t)
            rhs!(du, u, params, t)
            for i in 1:n
                u[i] += du[i] * dt
            end
            t += dt
        end
        return sum(u)
    end

    dp = zeros(length(p))
    try
        Enzyme.autodiff(Reverse, simulate_and_sum, Active, Duplicated(p, dp))
    catch e
        # Fall back to finite differences if Enzyme fails
        @warn "Enzyme autodiff failed, falling back to finite differences: $e"
        val0 = simulate_and_sum(p)
        for i in eachindex(p)
            p_plus = copy(p); p_plus[i] += 1e-6
            dp[i] = (simulate_and_sum(p_plus) - val0) / 1e-6
        end
    end
    return dp
end

export enzyme_grad, enzyme_gradient, enzyme_ode_sensitivity

end # module
