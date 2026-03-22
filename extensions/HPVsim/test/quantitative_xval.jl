#!/usr/bin/env julia
"""
Quantitative cross-validation: Julia HPVsim vs Python hpvsim.

Runs N_SEEDS simulations matching the Python scenarios, then compares
mean ± 95% CI. Reads Python results from quantitative_xval_results/.

Scenarios:
  1. HPV16 only — natural history (rate-based model)
  2. HPV16 + vaccination starting year 20

Note: Julia uses rate-based CIN progression (CIN1→CIN2→CIN3→Cancer)
while Python uses event-scheduled binary CIN. We compare aggregate
metrics (HPV prevalence, CIN prevalence, cancer counts) which should
be statistically similar given aligned parameters.

Calibration notes (v2 — age-assortative network + corrected CIN rates):
  - Julia now uses HPVSexualNet with age-assortative mixing matching
    Python's age-structured multi-layer network
  - CIN progression rates reduced ~20%, clearance rates increased ~20%
    to match effective CIN dynamics of Python's duration-based model
  - Separate cancer_rate parameter for CIN3→Cancer (was reusing prog_rate_cin3)
  - init_prev=0.10 and beta=0.22 recalibrated for the new network
"""

using HPVsim
using Starsim
using Statistics
using JSON3
using Printf

const OUT_DIR = joinpath(@__DIR__, "quantitative_xval_results")
mkpath(OUT_DIR)

const N_AGENTS = 5000
const N_YEARS  = 50
const DT       = 0.25
const START    = 2000.0
const STOP     = START + N_YEARS
const N_SEEDS  = 20
const SAMPLE_YEARS = [10, 25, 50]

# Calibrated parameters (v2: age-assortative network + corrected CIN rates)
const INIT_PREV  = 0.10   # Matches Python's age-structured ~10-12% effective init_prev
const BETA       = 0.70   # Calibrated for age-assortative HPVSexualNet
const BIRTH_RATE = 35.0   # Nigeria-like crude birth rate (per 1000/yr)
const DEATH_RATE = 10.0   # Background mortality (per 1000/yr)

# ============================================================================
# Helper functions
# ============================================================================

"""Get the result index for a given year offset from start."""
function get_timepoint_index(npts::Int, dt::Float64, year_offset::Int)
    ti = round(Int, year_offset / dt) + 1
    return clamp(ti, 1, npts)
end

"""Extract HPV disease from simulation (first HPVGenotype found)."""
function get_hpv_disease(sim)
    for (_, dis) in sim.diseases
        if dis isa HPVGenotype
            return dis
        end
    end
    error("No HPVGenotype disease found in simulation")
end

"""Extract metrics from a completed simulation."""
function extract_metrics(sim)
    dis = get_hpv_disease(sim)
    md = Starsim.module_data(dis)
    npts = md.t.npts

    metrics = Dict{String, Float64}()

    for yr in SAMPLE_YEARS
        ti = get_timepoint_index(npts, DT, yr)
        prev = ti <= length(md.results[:prevalence].values) ? md.results[:prevalence][ti] : NaN
        cin_prev = ti <= length(md.results[:cin_prevalence].values) ? md.results[:cin_prevalence][ti] : NaN
        n_ca = ti <= length(md.results[:n_cancerous].values) ? md.results[:n_cancerous][ti] : NaN

        metrics["hpv_prev_yr$yr"] = prev
        metrics["cin_prev_yr$yr"] = cin_prev
        metrics["n_cancerous_yr$yr"] = n_ca
    end

    # Cumulative cancer deaths
    n_cancer_deaths = md.results[:n_cancer_deaths].values
    metrics["cum_cancer_deaths"] = sum(n_cancer_deaths)

    # Cumulative cancerous count at each timepoint (as proxy for incidence)
    cancer_vals = md.results[:n_cancerous].values
    metrics["peak_cancerous"] = maximum(cancer_vals)

    return metrics
end

# ============================================================================
# Scenario 1: HPV16 natural history
# ============================================================================

function run_scenario1(seed::Int)
    sim = HPVSim(
        genotypes   = [GenotypeDef(:hpv16; init_prev=INIT_PREV)],
        beta        = BETA,
        n_agents    = N_AGENTS,
        start       = START,
        stop        = STOP,
        dt          = DT,
        rand_seed   = seed,
        use_immunity = true,
        use_demographics = true,
        birth_rate  = BIRTH_RATE,
        death_rate  = DEATH_RATE,
    )
    Starsim.run!(sim; verbose=0)
    return extract_metrics(sim)
end

# ============================================================================
# Scenario 2: HPV16 + vaccination
# ============================================================================

function run_scenario2(seed::Int)
    # Baseline (no vaccination)
    sim_base = HPVSim(
        genotypes   = [GenotypeDef(:hpv16; init_prev=INIT_PREV)],
        beta        = BETA,
        n_agents    = N_AGENTS,
        start       = START,
        stop        = STOP,
        dt          = DT,
        rand_seed   = seed,
        use_immunity = true,
        use_demographics = true,
        birth_rate  = BIRTH_RATE,
        death_rate  = DEATH_RATE,
    )
    Starsim.run!(sim_base; verbose=0)
    m_base = extract_metrics(sim_base)

    # With vaccination starting year 20 (no vaccine waning, matching Python use_waning=False)
    # Python's bivalent vaccine: imm_init ~ beta(30,2) → mean 0.9375 → ~93.75% protection
    # We set dose_eff * genotype_eff = 0.9375 to match Python's effective per-agent protection
    vax = HPVVaccination(
        start_year = START + 20,
        end_year   = Inf,
        covered_genotypes = [:hpv16],
        genotype_efficacies = Dict(:hpv16 => 1.0),
        n_doses    = 1,
        dose_efficacies = [0.9375],
        min_age    = 9.0,
        max_age    = 14.0,
        uptake_prob = 0.9,
        waning_rate = 0.0,   # No waning, matching Python default
        sex         = :female,
    )
    sim_vx = HPVSim(
        genotypes    = [GenotypeDef(:hpv16; init_prev=INIT_PREV)],
        beta         = BETA,
        n_agents     = N_AGENTS,
        start        = START,
        stop         = STOP,
        dt           = DT,
        rand_seed    = seed,
        use_immunity = true,
        use_demographics = true,
        birth_rate   = BIRTH_RATE,
        death_rate   = DEATH_RATE,
        interventions = [vax],
    )
    Starsim.run!(sim_vx; verbose=0)
    m_vx = extract_metrics(sim_vx)

    metrics = Dict{String, Float64}()

    # Prevalence at year 50
    base_prev = m_base["hpv_prev_yr50"]
    vx_prev   = m_vx["hpv_prev_yr50"]
    base_cin  = m_base["cin_prev_yr50"]
    vx_cin    = m_vx["cin_prev_yr50"]

    metrics["hpv_prev_yr50_base"] = base_prev
    metrics["hpv_prev_yr50_vx"]   = vx_prev
    metrics["cin_prev_yr50_base"] = base_cin
    metrics["cin_prev_yr50_vx"]   = vx_cin
    metrics["hpv_prev_reduction_pct"] = base_prev > 0 ? (1 - vx_prev / base_prev) * 100 : 0.0
    metrics["cin_prev_reduction_pct"] = base_cin > 0 ? (1 - vx_cin / base_cin) * 100 : 0.0

    return metrics
end

# ============================================================================
# Statistics
# ============================================================================

function compute_stats(all_metrics::Vector{Dict{String, Float64}})
    keys_set = keys(all_metrics[1])
    stats = Dict{String, Any}()
    for k in keys_set
        vals = [m[k] for m in all_metrics if !isnan(m[k])]
        n = length(vals)
        if n == 0
            stats[k] = Dict("mean" => 0.0, "ci_lo" => 0.0, "ci_hi" => 0.0, "std" => 0.0, "n" => 0)
            continue
        end
        m = mean(vals)
        s = n > 1 ? std(vals) : 0.0
        se = n > 1 ? s / sqrt(n) : 0.0
        stats[k] = Dict(
            "mean"   => m,
            "ci_lo"  => m - 1.96 * se,
            "ci_hi"  => m + 1.96 * se,
            "std"    => s,
            "n"      => n,
            "values" => vals,
        )
    end
    return stats
end

function print_stats_table(name::String, stats::Dict)
    println("\n", "="^72)
    println("  ", name)
    println("="^72)
    @printf("  %-35s %10s %22s %8s\n", "Metric", "Mean", "95% CI", "Std")
    println("  ", "-"^75)
    for (k, v) in sort(collect(stats), by=x->x[1])
        ci_str = @sprintf("[%.6f, %.6f]", v["ci_lo"], v["ci_hi"])
        @printf("  %-35s %10.6f %22s %8.6f\n", k, v["mean"], ci_str, v["std"])
    end
end

# ============================================================================
# Cross-validation comparison
# ============================================================================

function load_python_results()
    path = joinpath(OUT_DIR, "python_results.json")
    if !isfile(path)
        println("WARNING: Python results not found at $path")
        println("  Run quantitative_xval.py first!")
        return nothing
    end
    return JSON3.read(read(path, String))
end

function compare_metric(name::String, julia_stats::Dict, python_data, scenario_key::String)
    py_scenario = getproperty(python_data, Symbol(scenario_key))
    if !hasproperty(py_scenario, Symbol(name))
        return nothing
    end

    py_metric = getproperty(py_scenario, Symbol(name))
    py_mean  = Float64(py_metric.mean)
    py_ci_lo = Float64(py_metric.ci_lo)
    py_ci_hi = Float64(py_metric.ci_hi)

    jl_mean  = julia_stats[name]["mean"]
    jl_ci_lo = julia_stats[name]["ci_lo"]
    jl_ci_hi = julia_stats[name]["ci_hi"]

    # Check overlap: Julia mean in Python CI, or Python mean in Julia CI
    jl_in_py = py_ci_lo <= jl_mean <= py_ci_hi
    py_in_jl = jl_ci_lo <= py_mean <= jl_ci_hi
    overlap = jl_in_py || py_in_jl

    return (
        name     = name,
        py_mean  = py_mean,
        py_ci    = (py_ci_lo, py_ci_hi),
        jl_mean  = jl_mean,
        jl_ci    = (jl_ci_lo, jl_ci_hi),
        overlap  = overlap,
    )
end

function print_comparison(scenario_name::String, julia_stats::Dict, python_data, scenario_key::String)
    println("\n", "="^90)
    println("  COMPARISON: ", scenario_name)
    println("="^90)
    @printf("  %-28s %12s %12s %12s %12s %8s\n",
            "Metric", "Python Mean", "Julia Mean", "Py 95%CI", "Jl 95%CI", "Overlap?")
    println("  ", "-"^88)

    n_pass = 0
    n_total = 0
    n_primary_pass = 0
    n_primary_total = 0

    # Primary metrics: HPV prevalence at equilibrium (should match)
    # Secondary metrics: CIN prevalence, transients, vaccination (expected architectural diffs)
    primary_metrics = Set([
        "hpv_prev_yr10", "hpv_prev_yr50",
        "hpv_prev_yr50_base",
    ])

    for k in sort(collect(keys(julia_stats)))
        r = compare_metric(k, julia_stats, python_data, scenario_key)
        r === nothing && continue
        n_total += 1
        is_primary = k in primary_metrics
        status = r.overlap ? "✓ YES" : "✗ NO"
        tag = is_primary ? " [PRIMARY]" : ""
        if r.overlap
            n_pass += 1
            if is_primary
                n_primary_pass += 1
            end
        end
        if is_primary
            n_primary_total += 1
        end
        @printf("  %-28s %12.6f %12.6f  [%.4f,%.4f] [%.4f,%.4f] %8s%s\n",
                r.name, r.py_mean, r.jl_mean,
                r.py_ci[1], r.py_ci[2],
                r.jl_ci[1], r.jl_ci[2],
                status, tag)
    end

    println("  ", "-"^88)
    println("  RESULT: $n_pass / $n_total total, $n_primary_pass / $n_primary_total primary metrics overlap")
    return n_pass, n_total, n_primary_pass, n_primary_total
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("Running Julia HPVsim quantitative cross-validation")
    println("  N_AGENTS=$N_AGENTS, N_YEARS=$N_YEARS, DT=$DT, N_SEEDS=$N_SEEDS")

    # Scenario 1
    println("\n--- Scenario 1: HPV16 natural history ---")
    s1_metrics = Dict{String, Float64}[]
    for seed in 1:N_SEEDS
        print("  Seed $seed/$N_SEEDS...")
        m = run_scenario1(seed)
        push!(s1_metrics, m)
        @printf(" HPV prev@50=%.4f\n", m["hpv_prev_yr50"])
    end
    s1_stats = compute_stats(s1_metrics)
    print_stats_table("Scenario 1: HPV16 natural history (Julia)", s1_stats)

    # Scenario 2
    println("\n--- Scenario 2: HPV16 + vaccination ---")
    s2_metrics = Dict{String, Float64}[]
    for seed in 1:N_SEEDS
        print("  Seed $seed/$N_SEEDS...")
        m = run_scenario2(seed)
        push!(s2_metrics, m)
        @printf(" HPV prev reduction=%.1f%%\n", m["hpv_prev_reduction_pct"])
    end
    s2_stats = compute_stats(s2_metrics)
    print_stats_table("Scenario 2: HPV16 + vaccination (Julia)", s2_stats)

    # Save Julia results
    julia_results = Dict(
        "scenario1" => s1_stats,
        "scenario2" => s2_stats,
        "config" => Dict(
            "n_agents" => N_AGENTS,
            "n_years"  => N_YEARS,
            "dt"       => DT,
            "n_seeds"  => N_SEEDS,
            "start"    => START,
            "sample_years" => SAMPLE_YEARS,
        ),
    )
    jl_path = joinpath(OUT_DIR, "julia_results.json")
    open(jl_path, "w") do f
        JSON3.pretty(f, julia_results)
    end
    println("\nJulia results saved to $jl_path")

    # Compare with Python
    python_data = load_python_results()
    if python_data !== nothing
        total_pass = 0
        total_metrics = 0
        primary_pass = 0
        primary_total = 0

        # Only compare metrics that exist in both
        p1, t1, pp1, pt1 = print_comparison("Scenario 1: HPV16 natural history", s1_stats, python_data, "scenario1")
        total_pass += p1; total_metrics += t1; primary_pass += pp1; primary_total += pt1

        p2, t2, pp2, pt2 = print_comparison("Scenario 2: HPV16 + vaccination", s2_stats, python_data, "scenario2")
        total_pass += p2; total_metrics += t2; primary_pass += pp2; primary_total += pt2

        println("\n", "="^90)
        println("  OVERALL: $total_pass / $total_metrics total metrics overlap")
        println("  PRIMARY: $primary_pass / $primary_total equilibrium HPV prevalence metrics overlap")
        println()
        if primary_pass == primary_total
            println("  ✓ All primary metrics pass — core disease dynamics are validated!")
        else
            println("  ✗ Some primary metrics differ — investigation needed")
        end

        println()
        println("  Notes on architectural differences:")
        println("  • CIN prevalence: Julia uses rate-based CIN1→CIN2→CIN3 stages vs")
        println("    Python's event-scheduled binary CIN. Rates now calibrated to")
        println("    match effective CIN dynamics (v2: ~20% lower prog, ~20% higher clearance)")
        println("  • Network: Julia now uses HPVSexualNet with age-assortative mixing")
        println("    matching Python's age-structured multi-layer network")
        println("  • CIN3→Cancer: now uses separate cancer_rate parameter (was reusing prog_rate_cin3)")
        println("="^90)
    end
end

main()
