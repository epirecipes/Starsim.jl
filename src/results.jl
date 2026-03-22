"""
Result storage for Starsim.jl.

Mirrors Python starsim's `results.py`. Each `Result` wraps a time-series
array, and `Results` is an ordered collection of results.
"""

# ============================================================================
# Result
# ============================================================================

"""
    Result <: AbstractResult

A single time-series result from a simulation module.

# Fields
- `name::Symbol` — result name (e.g., `:new_infections`)
- `label::String` — human-readable label
- `module_name::Symbol` — owning module
- `values::Vector{Float64}` — time-series data
- `scale::Bool` — whether to scale by `pop_scale` during finalization
- `auto_plot::Bool` — include in default plots
- `summarize_by::Symbol` — aggregation method (`:sum`, `:mean`, `:last`)
"""
mutable struct Result <: AbstractResult
    name::Symbol
    label::String
    module_name::Symbol
    values::Vector{Float64}
    scale::Bool
    auto_plot::Bool
    summarize_by::Symbol
end

function Result(name::Symbol;
                label::String=string(name),
                module_name::Symbol=:sim,
                npts::Int=0,
                dtype::Type=Float64,
                scale::Bool=true,
                auto_plot::Bool=true,
                summarize_by::Symbol=:sum)
    Result(name, label, module_name, zeros(dtype, npts), scale, auto_plot, summarize_by)
end

Base.show(io::IO, r::Result) = print(io, "Result(:$(r.name), n=$(length(r.values)))")

"""Get result value at time index `ti`."""
Base.getindex(r::Result, ti::Int) = r.values[ti]

"""Set result value at time index `ti`."""
Base.setindex!(r::Result, val, ti::Int) = (r.values[ti] = val)

"""Number of timepoints."""
Base.length(r::Result) = length(r.values)

export Result

# ============================================================================
# Results — ordered collection
# ============================================================================

"""
    Results

Ordered dictionary of `Result` objects for a module or simulation.

# Example
```julia
res = Results()
push!(res, Result(:prevalence; npts=100, scale=false))
res[:prevalence][1] = 0.05
```
"""
struct Results
    data::OrderedDict{Symbol, Result}
end

Results() = Results(OrderedDict{Symbol, Result}())

Base.getindex(r::Results, k::Symbol) = r.data[k]
Base.setindex!(r::Results, v::Result, k::Symbol) = (r.data[k] = v)
Base.haskey(r::Results, k::Symbol) = haskey(r.data, k)
Base.keys(r::Results) = keys(r.data)
Base.values(r::Results) = values(r.data)
Base.iterate(r::Results, args...) = iterate(r.data, args...)
Base.length(r::Results) = length(r.data)

function Base.show(io::IO, r::Results)
    print(io, "Results($(join(keys(r.data), ", ")))")
end

"""Add a Result to the collection."""
function Base.push!(r::Results, result::Result)
    r.data[result.name] = result
    return r
end

"""
    to_dataframe(res::Results, timevec::Vector{Float64})

Convert results to a DataFrame with a `time` column.
"""
function to_dataframe(res::Results, timevec::Vector{Float64})
    df = DataFrame(time=timevec)
    for (name, result) in res.data
        n = min(length(timevec), length(result.values))
        df[!, name] = result.values[1:n]
    end
    return df
end

"""
    summarize(res::Results)

Compute summary statistics for each result based on its `summarize_by` field.
"""
function summarize(res::Results)
    summary = OrderedDict{Symbol, Float64}()
    for (name, r) in res.data
        vals = r.values
        if r.summarize_by == :sum
            summary[name] = sum(vals)
        elseif r.summarize_by == :mean
            summary[name] = mean(vals)
        elseif r.summarize_by == :last
            summary[name] = isempty(vals) ? NaN : vals[end]
        else
            summary[name] = sum(vals)
        end
    end
    return summary
end

"""
    scale_results!(res::Results, factor::Float64)

Scale results that have `scale=true` by the given factor.
"""
function scale_results!(res::Results, factor::Float64)
    for r in values(res.data)
        if r.scale
            r.values .*= factor
        end
    end
    return res
end

export Results, to_dataframe, summarize, scale_results!
