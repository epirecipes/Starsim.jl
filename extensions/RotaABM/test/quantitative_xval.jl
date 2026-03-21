#!/usr/bin/env julia
#=
Quantitative cross-validation: Julia RotaABM vs Python rotasim.
Runs identical scenarios across 20 seeds and saves summary statistics as JSON.

Usage:
    cd code/Starsim.jl
    julia --project=extensions/RotaABM extensions/RotaABM/test/quantitative_xval.jl
=#

starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using RotaABM
using Printf
using Statistics

const RESULTS_FILE = joinpath(@__DIR__, "quantitative_xval_julia.json")
const N_SEEDS  = 20
const N_AGENTS = 10_000
const DUR_YEARS = 5

# ── Helpers ────────────────────────────────────────────────────────────────

function attack_rate(d, n_agents)
    return sum(Starsim.module_results(d)[:new_infections].values) / n_agents
end

function peak_prevalence(d)
    return maximum(Starsim.module_results(d)[:prevalence].values)
end

function mean_prevalence(d)
    prev = Starsim.module_results(d)[:prevalence].values
    return sum(prev) / length(prev)
end

function equilibrium_prevalence(d; last_years=2, dt=1/365.25)
    prev = Starsim.module_results(d)[:prevalence].values
    n_steps_last = Int(round(last_years / dt))
    n_steps_last = min(n_steps_last, length(prev))
    tail = prev[end-n_steps_last+1:end]
    return sum(tail) / length(tail)
end

function total_new_infections(sim)
    total = 0.0
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        total += sum(Starsim.module_results(d)[:new_infections].values)
    end
    return total
end

function count_active_strains(sim)
    n = 0
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        sum(Starsim.module_results(d)[:new_infections].values) > 0 && (n += 1)
    end
    return n
end

function reassortment_count(sim)
    for (_, c) in sim.connectors
        c isa RotaReassortmentConnector || continue
        r = Starsim.module_results(c)
        haskey(r, :n_reassortments) || continue
        return sum(r[:n_reassortments].values)
    end
    return 0.0
end

# ── Scenario 1: Single strain SEIR ────────────────────────────────────────

function run_scenario1(seed)
    sim = RotaSim(
        scenario = Dict{String,Any}(
            "strains" => Dict{Tuple{Int,Int}, Dict{String,Float64}}(
                (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.01)
            ),
            "default_fitness" => 1.0
        ),
        n_agents  = N_AGENTS,
        stop      = Float64(DUR_YEARS),
        dt        = Starsim.days(1),
        rand_seed = seed,
    )
    run!(sim; verbose=0)

    d = sim.diseases[:G1P8]
    return Dict{String,Float64}(
        "attack_rate"    => attack_rate(d, N_AGENTS),
        "peak_prev"      => peak_prevalence(d),
        "mean_prev"      => mean_prevalence(d),
        "equil_prev"     => equilibrium_prevalence(d; last_years=2, dt=sim.pars.dt),
    )
end

# ── Scenario 2: Multi-strain ──────────────────────────────────────────────

function run_scenario2(seed)
    sim = RotaSim(
        scenario  = "simple",
        n_agents  = N_AGENTS,
        stop      = Float64(DUR_YEARS),
        dt        = Starsim.days(1),
        rand_seed = seed,
    )
    run!(sim; verbose=0)

    # Total prevalence time series
    npts = length(Starsim.module_results(first(values(sim.diseases)))[:prevalence].values)
    total_prev = zeros(npts)
    per_strain = Dict{String, Vector{Float64}}()
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        prev_vals = Starsim.module_results(d)[:prevalence].values
        total_prev .+= prev_vals
        per_strain["G$(d.G)P$(d.P)"] = copy(prev_vals)
    end

    total_peak = maximum(total_prev)
    total_mean = sum(total_prev) / length(total_prev)
    last_n = Int(round(2.0 / sim.pars.dt))
    last_n = min(last_n, npts)
    total_equil = sum(total_prev[end-last_n+1:end]) / last_n

    result = Dict{String,Float64}(
        "total_attack_rate" => total_new_infections(sim) / N_AGENTS,
        "total_peak_prev"   => total_peak,
        "total_mean_prev"   => total_mean,
        "total_equil_prev"  => total_equil,
        "active_strains"    => Float64(count_active_strains(sim)),
        "reassortments"     => Float64(reassortment_count(sim)),
    )

    # Per-strain attack rates
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        nm = "G$(d.G)P$(d.P)"
        result["$(nm)_attack_rate"] = attack_rate(d, N_AGENTS)
        result["$(nm)_peak_prev"]   = peak_prevalence(d)
    end

    return result
end

# ── Statistics ─────────────────────────────────────────────────────────────

function summarize(values::Vector{Float64})
    n = length(values)
    m = mean(values)
    s = n > 1 ? std(values) : 0.0
    se = s / sqrt(n)
    ci95 = 1.96 * se
    return Dict("mean" => m, "std" => s, "ci95" => ci95, "lo" => m - ci95, "hi" => m + ci95)
end

# ── Minimal JSON writer ───────────────────────────────────────────────────

function dict_to_json(d::Dict, indent=0)
    buf = IOBuffer()
    _write_json(buf, d, indent)
    return String(take!(buf))
end

function _write_json(io, d::Dict, indent)
    pad = " " ^ indent
    println(io, "{")
    ks = sort(collect(keys(d)))
    for (i, k) in enumerate(ks)
        print(io, pad, "  \"", k, "\": ")
        _write_json(io, d[k], indent + 2)
        i < length(ks) && print(io, ",")
        println(io)
    end
    print(io, pad, "}")
end
_write_json(io, v::Bool, _)            = print(io, v ? "true" : "false")
_write_json(io, v::AbstractFloat, _)   = @printf(io, "%.6f", v)
_write_json(io, v::Integer, _)         = print(io, v)
_write_json(io, v::AbstractString, _)  = print(io, "\"", v, "\"")
function _write_json(io, v::AbstractVector, indent)
    print(io, "[")
    for (i, x) in enumerate(v)
        _write_json(io, x, indent)
        i < length(v) && print(io, ", ")
    end
    print(io, "]")
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    println("Julia RotaABM — Quantitative Cross-Validation")
    println("N_SEEDS=$N_SEEDS  N_AGENTS=$N_AGENTS  DUR=$DUR_YEARS years")
    println("="^60)

    seeds = collect(1:N_SEEDS)

    # Scenario 1: Single strain
    println("\nScenario 1: Single strain G1P8")
    s1_results = [run_scenario1(s) for s in seeds]
    s1_keys = sort(collect(keys(s1_results[1])))
    s1_summary = Dict{String,Any}()
    for k in s1_keys
        vals = Float64[r[k] for r in s1_results]
        s = summarize(vals)
        s1_summary[k] = s
        @printf("  %-20s: %.4f ± %.4f  [%.4f, %.4f]\n", k, s["mean"], s["ci95"], s["lo"], s["hi"])
    end

    # Scenario 2: Multi-strain
    println("\nScenario 2: Multi-strain (simple)")
    s2_results = [run_scenario2(s) for s in seeds]
    s2_all_keys = Set{String}()
    for r in s2_results
        union!(s2_all_keys, keys(r))
    end
    s2_keys = sort(collect(s2_all_keys))
    s2_summary = Dict{String,Any}()
    for k in s2_keys
        vals = Float64[get(r, k, NaN) for r in s2_results]
        vals = filter(!isnan, vals)
        isempty(vals) && continue
        s = summarize(vals)
        s2_summary[k] = s
        @printf("  %-20s: %.4f ± %.4f  [%.4f, %.4f]\n", k, s["mean"], s["ci95"], s["lo"], s["hi"])
    end

    # Save results
    results = Dict{String,Any}(
        "n_seeds"  => N_SEEDS,
        "n_agents" => N_AGENTS,
        "dur_years" => DUR_YEARS,
        "scenario1" => s1_summary,
        "scenario2" => s2_summary,
    )
    open(RESULTS_FILE, "w") do f
        write(f, dict_to_json(results))
    end
    println("\nResults saved to: $RESULTS_FILE")
end

main()
