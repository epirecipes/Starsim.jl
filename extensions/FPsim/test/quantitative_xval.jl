"""
Quantitative cross-validation: Julia FPsim.
Runs N_SEEDS simulations with Kenya parameters and saves summary statistics.
Must match quantitative_xval.py configuration exactly.
"""

# Setup load paths
starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using FPsim
using Statistics
using JSON3
using Printf

# Configuration — must match Python exactly
const N_SEEDS    = 20
const N_AGENTS   = 5000
const START_YEAR = 2000.0
const STOP_YEAR  = 2020.0
const LOCATION   = :kenya
const DT_VAL     = 1/12

const OUTFILE = joinpath(@__DIR__, "xval_julia_results.json")

function get_fpmod(sim)
    for (_, c) in sim.connectors
        c isa FPmod && return c
    end
    return nothing
end

"""Compute annual births from per-step births array."""
function compute_annual_births(births::Vector{Float64}; mpy::Int=12)
    n = length(births)
    annual = Float64[]
    for start in 1:mpy:(n - mpy + 1)
        push!(annual, sum(births[start:start+mpy-1]))
    end
    return annual
end

"""Compute TFR for a specific annual window using birth counts and women counts."""
function compute_tfr(fpmod, sim, year_idx::Int)
    mpy = 12
    ti_start = (year_idx - 1) * mpy + 1  # 1-based
    ti_end = min(ti_start + mpy - 1, length(Starsim.module_results(fpmod)[:n_births].values))
    ti_start > ti_end && return NaN

    active = sim.people.auids.values
    age_bins = Float64[15, 20, 25, 30, 35, 40, 45, 50]
    n_bins = length(age_bins) - 1
    births_by_bin = zeros(n_bins)
    women_by_bin = zeros(n_bins)

    # Count women who just gave birth in this window by age bin
    for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]
        for b in 1:n_bins
            if age >= age_bins[b] && age < age_bins[b+1]
                women_by_bin[b] += 1.0
                break
            end
        end
    end

    # Use stored births data from FPmod results
    res = Starsim.module_results(fpmod)
    total_births_window = 0.0
    for ti in ti_start:ti_end
        total_births_window += res[:n_births][ti]
    end

    # Distribute births proportionally by women count (simplified)
    # For a more accurate approach, we'd need per-age birth tracking
    total_women = sum(women_by_bin)
    total_women == 0 && return 0.0

    tfr = 0.0
    for b in 1:n_bins
        frac = women_by_bin[b] / total_women
        bin_births = total_births_window * frac
        asfr = women_by_bin[b] > 0 ? (bin_births / women_by_bin[b]) * (1.0 / DT_VAL) * 1000.0 : 0.0
        bin_width = age_bins[b+1] - age_bins[b]
        tfr += asfr * bin_width / 1000.0
    end
    return tfr
end

"""Run one simulation and return summary metrics."""
function run_one_seed(seed::Int)
    sim = FPSim(
        n_agents   = N_AGENTS,
        start      = START_YEAR,
        stop       = STOP_YEAR,
        dt         = DT_VAL,
        rand_seed  = seed,
        location   = LOCATION,
        use_contraception = true,
        analyzers  = [FPAnalyzer()],
    )
    Starsim.run!(sim; verbose=0)

    fpmod = get_fpmod(sim)
    fpmod === nothing && error("FPmod not found in simulation")
    res = Starsim.module_results(fpmod)

    npts = length(res[:n_births].values)
    births_arr = Float64[res[:n_births][ti] for ti in 1:npts]
    annual_births = compute_annual_births(births_arr)

    total_births = sum(births_arr)
    total_miscarriages = sum(Float64[res[:n_miscarriages][ti] for ti in 1:npts])
    total_stillbirths = sum(Float64[res[:n_stillbirths][ti] for ti in 1:npts])
    total_abortions = sum(Float64[res[:n_abortions][ti] for ti in 1:npts])
    total_mat_deaths = sum(Float64[res[:n_maternal_deaths][ti] for ti in 1:npts])
    total_inf_deaths = sum(Float64[res[:n_infant_deaths][ti] for ti in 1:npts])

    # Population
    final_pop = length(sim.people.auids.values)

    # CPR and TFR from analyzer
    analyzer = nothing
    for (_, a) in sim.analyzers
        a isa FPAnalyzer && (analyzer = a; break)
    end

    cpr = 0.0
    mean_tfr = NaN
    tfr_yr5 = NaN; tfr_yr10 = NaN; tfr_yr15 = NaN; tfr_yr19 = NaN

    if analyzer !== nothing
        ares = Starsim.module_results(analyzer)
        # CPR at end
        cpr = ares[:cpr][npts]

        # TFR at specific years
        tfr_vals = Float64[ares[:tfr][ti] for ti in 1:npts]

        # TFR at year X = average of monthly TFR values in that year
        function avg_tfr_year(yr)
            ti_s = (yr - 1) * 12 + 1
            ti_e = min(yr * 12, npts)
            ti_s > npts && return NaN
            vals = tfr_vals[ti_s:ti_e]
            return mean(vals)
        end

        tfr_yr5  = avg_tfr_year(5)
        tfr_yr10 = avg_tfr_year(10)
        tfr_yr15 = avg_tfr_year(15)
        tfr_yr19 = avg_tfr_year(19)

        # Mean TFR over stable period (year 3 to end)
        stable_tfr = Float64[]
        for yr in 3:Int(STOP_YEAR - START_YEAR)
            t = avg_tfr_year(yr)
            !isnan(t) && push!(stable_tfr, t)
        end
        mean_tfr = isempty(stable_tfr) ? NaN : mean(stable_tfr)
    end

    # Birth rate per 1000 women-years (last 5 years)
    last5_births = sum(births_arr[max(1, npts-59):npts])
    # Estimate women 15-49 at end
    n_women_15_49 = 0
    for u in sim.people.auids.values
        if sim.people.female.raw[u]
            age = sim.people.age.raw[u]
            if age >= 15 && age < 50
                n_women_15_49 += 1
            end
        end
    end
    birth_rate = n_women_15_49 > 0 ? (last5_births / 5.0) / n_women_15_49 * 1000 : 0.0

    # Pregnancy rate (conceptions per year, average over stable period)
    total_conceptions = total_births + total_miscarriages + total_stillbirths + total_abortions
    n_stable_years = max(STOP_YEAR - START_YEAR - 2, 1)
    preg_rate_annual = total_conceptions / n_stable_years

    # Per-pregnancy outcome ratios (architecture-independent)
    live_birth_ratio = total_conceptions > 0 ? total_births / total_conceptions : 0.0
    miscarriage_ratio = total_conceptions > 0 ? total_miscarriages / total_conceptions : 0.0
    stillbirth_ratio = total_conceptions > 0 ? total_stillbirths / total_conceptions : 0.0
    abortion_ratio = total_conceptions > 0 ? total_abortions / total_conceptions : 0.0

    # Crude birth rate and births per year
    crude_birth_rate = (total_births / 20.0) / N_AGENTS * 1000
    births_per_year = sum(births_arr[max(1, 25):end]) / max(STOP_YEAR - START_YEAR - 2, 1)

    return Dict{String, Any}(
        "seed" => seed,
        "total_births" => total_births,
        "total_miscarriages" => total_miscarriages,
        "total_stillbirths" => total_stillbirths,
        "total_abortions" => total_abortions,
        "total_mat_deaths" => total_mat_deaths,
        "total_inf_deaths" => total_inf_deaths,
        "annual_births" => annual_births,
        "final_pop" => final_pop,
        "cpr" => cpr,
        "birth_rate_per1000" => birth_rate,
        "mean_tfr" => mean_tfr,
        "preg_rate_annual" => preg_rate_annual,
        "live_birth_ratio" => live_birth_ratio,
        "miscarriage_ratio" => miscarriage_ratio,
        "stillbirth_ratio" => stillbirth_ratio,
        "abortion_ratio" => abortion_ratio,
        "crude_birth_rate" => crude_birth_rate,
        "births_per_year" => births_per_year,
        "tfr_yr5" => tfr_yr5,
        "tfr_yr10" => tfr_yr10,
        "tfr_yr15" => tfr_yr15,
        "tfr_yr19" => tfr_yr19,
    )
end

function main()
    println("Running Julia FPsim cross-validation: $N_SEEDS seeds × $N_AGENTS agents")
    println("Location: $LOCATION, Period: $START_YEAR-$STOP_YEAR")

    all_results = Dict{String, Any}[]
    for seed in 1:N_SEEDS
        @printf("  Seed %d/%d...", seed, N_SEEDS)
        result = run_one_seed(seed)
        push!(all_results, result)
        tfr_str = isnan(result["mean_tfr"]) ? "N/A" : @sprintf("%.2f", result["mean_tfr"])
        @printf(" births=%.0f, TFR=%s\n", result["total_births"], tfr_str)
    end

    # Compute summary statistics
    metrics = ["total_births", "total_miscarriages", "total_stillbirths", "total_abortions",
               "total_mat_deaths", "total_inf_deaths", "final_pop", "cpr",
               "birth_rate_per1000", "mean_tfr", "preg_rate_annual",
               "live_birth_ratio", "miscarriage_ratio", "stillbirth_ratio", "abortion_ratio",
               "crude_birth_rate", "births_per_year",
               "tfr_yr5", "tfr_yr10", "tfr_yr15", "tfr_yr19"]

    summary = Dict{String, Any}()
    for m in metrics
        vals = Float64[]
        for r in all_results
            v = r[m]
            if v isa Number && !isnan(v)
                push!(vals, Float64(v))
            end
        end
        if !isempty(vals)
            mn = mean(vals)
            sd = std(vals; corrected=true)
            ci95 = 1.96 * sd / sqrt(length(vals))
            summary[m] = Dict("mean" => mn, "std" => sd, "ci95" => ci95, "n" => length(vals),
                              "lo" => mn - ci95, "hi" => mn + ci95)
        end
    end

    output = Dict{String, Any}(
        "config" => Dict("n_seeds" => N_SEEDS, "n_agents" => N_AGENTS, "start" => START_YEAR,
                         "stop" => STOP_YEAR, "location" => string(LOCATION), "dt" => DT_VAL),
        "summary" => summary,
        "raw" => all_results,
    )

    open(OUTFILE, "w") do f
        JSON3.write(f, output)
    end

    # Print table
    println("\n", "="^80)
    println("Julia FPsim Cross-Validation Summary ($N_SEEDS seeds)")
    println("="^80)
    @printf("%-25s %10s %10s %10s\n", "Metric", "Mean", "± 95%CI", "Std")
    println("-"^55)
    for m in metrics
        if haskey(summary, m)
            s = summary[m]
            @printf("%-25s %10.3f %10.3f %10.3f\n", m, s["mean"], s["ci95"], s["std"])
        end
    end

    println("\nResults saved to $OUTFILE")
    return output
end

main()
