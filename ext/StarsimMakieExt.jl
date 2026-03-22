"""
    StarsimMakieExt

Makie.jl plotting extension for Starsim.jl. Provides interactive, publication-quality
plots for simulation results using the Makie ecosystem (CairoMakie, GLMakie, WGLMakie).

# Functions
- `plot_sim(sim)` — grid of all auto-plot results
- `plot_sim(msim)` — reduced MultiSim results with confidence bands
- `plot_disease(sim, disease_name)` — single disease result panel
- `plot_comparison(sims; labels)` — overlay multiple sims on shared axes
"""
module StarsimMakieExt

using Starsim
using Makie

# ============================================================================
# Color palette — professional defaults
# ============================================================================

const STARSIM_COLORS = [
    Makie.RGBf(0.122, 0.467, 0.706),  # blue
    Makie.RGBf(1.000, 0.498, 0.055),  # orange
    Makie.RGBf(0.173, 0.627, 0.173),  # green
    Makie.RGBf(0.839, 0.153, 0.157),  # red
    Makie.RGBf(0.580, 0.404, 0.741),  # purple
    Makie.RGBf(0.549, 0.337, 0.294),  # brown
    Makie.RGBf(0.890, 0.467, 0.761),  # pink
    Makie.RGBf(0.498, 0.498, 0.498),  # gray
    Makie.RGBf(0.737, 0.741, 0.133),  # olive
    Makie.RGBf(0.090, 0.745, 0.812),  # cyan
]

function _get_color(i::Int)
    return STARSIM_COLORS[mod1(i, length(STARSIM_COLORS))]
end

# ============================================================================
# Grid layout helpers
# ============================================================================

"""Compute a roughly square grid layout for `n` panels."""
function _grid_dims(n::Int)
    ncols = ceil(Int, sqrt(n))
    nrows = ceil(Int, n / ncols)
    return nrows, ncols
end

# ============================================================================
# plot_sim(sim::Sim) — grid of auto-plot results
# ============================================================================

"""
    plot_sim(sim::Starsim.Sim; size=(900, 700), kwargs...) → Figure

Plot all auto-plot results from a completed simulation. Each result gets its
own Axis in a grid layout.

# Example
```julia
using CairoMakie
sim = Sim(diseases=SIR(beta=0.05), n_agents=5000)
run!(sim)
fig = plot_sim(sim)
```
"""
function Starsim.plot_sim(sim::Starsim.Sim; size=(900, 700), kwargs...)
    sim.complete || error("Simulation must be complete before plotting")

    plot_results = Starsim._collect_plot_results(sim)
    isempty(plot_results) && error("No auto-plot results found")

    tvec = Starsim._sim_tvec(sim)
    nresults = length(plot_results)
    nrows, ncols = _grid_dims(nresults)

    fig = Figure(; size=size, kwargs...)

    for (i, (key, r)) in enumerate(plot_results)
        row = div(i - 1, ncols) + 1
        col = mod1(i, ncols)
        ax = Axis(fig[row, col];
                  title=string(key),
                  xlabel="Time",
                  ylabel=r.label,
                  xgridvisible=true,
                  ygridvisible=true,
                  xgridstyle=:dash,
                  ygridstyle=:dash,
                  xgridcolor=(:black, 0.12),
                  ygridcolor=(:black, 0.12))
        n = min(length(tvec), length(r.values))
        lines!(ax, tvec[1:n], r.values[1:n];
               color=_get_color(i), linewidth=2, label=r.label)
    end

    return fig
end

# ============================================================================
# plot_sim(msim::MultiSim) — reduced results with bands
# ============================================================================

"""
    plot_sim(msim::Starsim.MultiSim; size=(900, 700), kwargs...) → Figure

Plot reduced MultiSim results. Each result is shown as a central line with
a shaded confidence band (from `low` to `high`).

The MultiSim must be reduced first via `reduce!(msim)`.

# Example
```julia
using CairoMakie
msim = MultiSim(Sim(diseases=SIR(beta=0.05)), n_runs=10)
run!(msim); reduce!(msim)
fig = plot_sim(msim)
```
"""
function Starsim.plot_sim(msim::Starsim.MultiSim; size=(900, 700), kwargs...)
    msim.which === :reduced || error("MultiSim must be reduced before plotting. Call reduce!(msim) first.")

    first_sim = msim.sims[1]
    tvec = Starsim._sim_tvec(first_sim)
    keys_to_plot = collect(keys(msim.reduced))
    isempty(keys_to_plot) && error("No reduced results found")

    nresults = length(keys_to_plot)
    nrows, ncols = _grid_dims(nresults)

    fig = Figure(; size=size, kwargs...)

    for (i, key) in enumerate(keys_to_plot)
        rr = msim.reduced[key]
        row = div(i - 1, ncols) + 1
        col = mod1(i, ncols)
        ax = Axis(fig[row, col];
                  title=string(key),
                  xlabel="Time",
                  ylabel=string(key),
                  xgridvisible=true,
                  ygridvisible=true,
                  xgridstyle=:dash,
                  ygridstyle=:dash,
                  xgridcolor=(:black, 0.12),
                  ygridcolor=(:black, 0.12))
        n = min(length(tvec), length(rr.values))
        color = _get_color(i)

        # Confidence band
        band!(ax, tvec[1:n], rr.low[1:n], rr.high[1:n];
              color=(color, 0.2))
        # Central line
        lines!(ax, tvec[1:n], rr.values[1:n];
               color=color, linewidth=2, label=string(key))
    end

    return fig
end

# ============================================================================
# plot_disease — single disease panel
# ============================================================================

"""
    plot_disease(sim::Starsim.Sim, disease_name::Symbol; size=(800, 500), kwargs...) → Figure

Plot results for a single disease module. Shows prevalence, n_infected, and
new_infections (when available) on separate axes.

# Example
```julia
using CairoMakie
sim = Sim(diseases=SIR(beta=0.05), n_agents=5000); run!(sim)
fig = plot_disease(sim, :sir)
```
"""
function Starsim.plot_disease(sim::Starsim.Sim, disease_name::Symbol; size=(800, 500), kwargs...)
    sim.complete || error("Simulation must be complete before plotting")
    haskey(sim.diseases, disease_name) || error("Disease :$disease_name not found. Available: $(collect(keys(sim.diseases)))")

    disease = sim.diseases[disease_name]
    res = Starsim.module_results(disease)
    tvec = Starsim._sim_tvec(sim)

    # Collect results to plot
    result_pairs = Pair{Symbol, Starsim.Result}[]
    for (rname, r) in res.data
        push!(result_pairs, rname => r)
    end
    isempty(result_pairs) && error("No results found for disease :$disease_name")

    nresults = length(result_pairs)
    fig = Figure(; size=size, kwargs...)

    for (i, (rname, r)) in enumerate(result_pairs)
        ax = Axis(fig[i, 1];
                  title="$(disease_name): $(rname)",
                  xlabel= i == nresults ? "Time" : "",
                  ylabel=r.label,
                  xgridvisible=true,
                  ygridvisible=true,
                  xgridstyle=:dash,
                  ygridstyle=:dash,
                  xgridcolor=(:black, 0.12),
                  ygridcolor=(:black, 0.12))
        n = min(length(tvec), length(r.values))
        lines!(ax, tvec[1:n], r.values[1:n];
               color=_get_color(i), linewidth=2, label=r.label)

        # Hide x-axis tick labels for all but the bottom panel
        if i < nresults
            hidexdecorations!(ax; grid=false, ticks=false)
        end
    end

    Label(fig[0, 1], string(disease_name); fontsize=18, font=:bold)
    return fig
end

# ============================================================================
# plot_comparison — overlay multiple sims
# ============================================================================

"""
    plot_comparison(sims::Vector{Starsim.Sim};
                    labels=nothing, size=(900, 700), kwargs...) → Figure

Overlay results from multiple simulations on shared axes. Useful for comparing
parameter sweeps or scenario analyses.

Each simulation is plotted in a different color. Only results that are common
across all sims and marked `auto_plot` are shown.

# Example
```julia
using CairoMakie
sim1 = Sim(diseases=SIR(beta=0.03), n_agents=5000); run!(sim1)
sim2 = Sim(diseases=SIR(beta=0.08), n_agents=5000); run!(sim2)
fig = plot_comparison([sim1, sim2]; labels=["β=0.03", "β=0.08"])
```
"""
function Starsim.plot_comparison(sims::Vector{Starsim.Sim};
                                  labels=nothing, size=(900, 700), kwargs...)
    isempty(sims) && error("Must provide at least one simulation")
    all(s -> s.complete, sims) || error("All simulations must be complete before plotting")

    nsims = length(sims)
    if labels === nothing
        labels = ["Sim $i" for i in 1:nsims]
    end
    length(labels) == nsims || error("Number of labels ($(length(labels))) must match number of sims ($nsims)")

    # Find common auto-plot result keys
    all_keys = [Set(k for (k, _) in Starsim._collect_plot_results(s)) for s in sims]
    common_keys = intersect(all_keys...)
    isempty(common_keys) && error("No common auto-plot results found across simulations")

    # Use first sim's result ordering
    ref_results = Starsim._collect_plot_results(sims[1])
    ordered_keys = [k for (k, _) in ref_results if k in common_keys]

    nresults = length(ordered_keys)
    nrows, ncols = _grid_dims(nresults)

    fig = Figure(; size=size, kwargs...)

    for (panel_idx, key) in enumerate(ordered_keys)
        row = div(panel_idx - 1, ncols) + 1
        col = mod1(panel_idx, ncols)

        # Get label from first sim's result
        rlabel = string(key)
        for (k, r) in ref_results
            if k == key
                rlabel = r.label
                break
            end
        end

        ax = Axis(fig[row, col];
                  title=string(key),
                  xlabel="Time",
                  ylabel=rlabel,
                  xgridvisible=true,
                  ygridvisible=true,
                  xgridstyle=:dash,
                  ygridstyle=:dash,
                  xgridcolor=(:black, 0.12),
                  ygridcolor=(:black, 0.12))

        for (sim_idx, sim) in enumerate(sims)
            tvec = Starsim._sim_tvec(sim)
            pr = Starsim._collect_plot_results(sim)
            for (k, r) in pr
                if k == key
                    n = min(length(tvec), length(r.values))
                    lines!(ax, tvec[1:n], r.values[1:n];
                           color=_get_color(sim_idx), linewidth=2,
                           label=labels[sim_idx])
                    break
                end
            end
        end

        # Only show legend on the first panel
        if panel_idx == 1
            axislegend(ax; position=:rt, framevisible=false, labelsize=11)
        end
    end

    return fig
end

# ============================================================================
# Makie recipe for Sim via convert_arguments
# ============================================================================

"""
Makie recipe that enables `lines(sim)` to plot the first auto-plot result,
or `series(sim)` to plot all results as a matrix.
"""
function Makie.convert_arguments(P::Type{<:Lines}, sim::Starsim.Sim)
    sim.complete || error("Simulation must be complete before plotting")
    plot_results = Starsim._collect_plot_results(sim)
    isempty(plot_results) && error("No auto-plot results found")

    tvec = Starsim._sim_tvec(sim)
    _, r = first(plot_results)
    n = min(length(tvec), length(r.values))
    return Makie.convert_arguments(P, tvec[1:n], r.values[1:n])
end

function Makie.convert_arguments(P::Type{<:Series}, sim::Starsim.Sim)
    sim.complete || error("Simulation must be complete before plotting")
    plot_results = Starsim._collect_plot_results(sim)
    isempty(plot_results) && error("No auto-plot results found")

    tvec = Starsim._sim_tvec(sim)
    npts = length(tvec)
    nres = length(plot_results)
    mat = zeros(npts, nres)
    for (j, (_, r)) in enumerate(plot_results)
        n = min(npts, length(r.values))
        mat[1:n, j] = r.values[1:n]
    end
    return Makie.convert_arguments(P, tvec, mat)
end

end # module StarsimMakieExt
