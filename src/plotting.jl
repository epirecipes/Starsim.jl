"""
Plot recipes for Starsim.jl using RecipesBase.

Provides `plot(sim)` and `plot(msim)` when `using Plots`.
Uses RecipesBase so the core package stays lightweight
(no Plots/Makie dependency required).
"""

using RecipesBase

# ============================================================================
# Helper to collect plottable results
# ============================================================================

function _collect_plot_results(sim::Sim)
    plot_results = Pair{Symbol, Result}[]
    for (mod_name, mod) in all_modules(sim)
        for (res_name, r) in module_results(mod).data
            key = Symbol("$(mod_name)_$(res_name)")
            r.auto_plot && push!(plot_results, key => r)
        end
    end
    for (name, r) in sim.results.data
        name == :timevec && continue
        r.auto_plot && push!(plot_results, name => r)
    end
    return plot_results
end

function _sim_tvec(sim::Sim)
    return [sim.pars.start + (t - 1) * sim.pars.dt for t in 1:sim.t.npts]
end

# ============================================================================
# Sim plot recipe
# ============================================================================

@recipe function f(sim::Sim)
    sim.complete || error("Simulation must be complete before plotting")

    plot_results = _collect_plot_results(sim)
    isempty(plot_results) && return

    tvec = _sim_tvec(sim)

    layout := length(plot_results)
    for (i, (key, r)) in enumerate(plot_results)
        n = min(length(tvec), length(r.values))
        @series begin
            subplot := i
            label := r.label
            xlabel := "Time"
            ylabel := r.label
            title := string(key)
            linewidth := 2
            tvec[1:n], r.values[1:n]
        end
    end
end

# ============================================================================
# MultiSim plot recipe (reduced results)
# ============================================================================

@recipe function f(msim::MultiSim)
    msim.which === :reduced || error("MultiSim must be reduced before plotting. Call reduce!(msim) first.")

    first_sim = msim.sims[1]
    tvec = _sim_tvec(first_sim)
    keys_to_plot = collect(keys(msim.reduced))
    isempty(keys_to_plot) && return

    layout := length(keys_to_plot)
    for (i, key) in enumerate(keys_to_plot)
        rr = msim.reduced[key]
        n = min(length(tvec), length(rr.values))

        @series begin
            subplot := i
            seriestype := :line
            ribbon := (rr.values[1:n] .- rr.low[1:n], rr.high[1:n] .- rr.values[1:n])
            fillalpha := 0.2
            label := string(key)
            xlabel := "Time"
            ylabel := string(key)
            title := string(key)
            linewidth := 2
            tvec[1:n], rr.values[1:n]
        end
    end
end

export plot
