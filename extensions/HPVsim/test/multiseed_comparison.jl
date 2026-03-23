"""
Multi-seed trajectory comparison: Julia HPVsim vs Python hpvsim.

Runs N_SEEDS simulations for each scenario, computes mean trajectories,
then loads Python multi-seed results and computes correlation and level ratio.

Usage:
  # First: python multiseed_comparison.py
  # Then:  julia --project=.. multiseed_comparison.jl
"""

using HPVsim
using Starsim
using Statistics
using JSON3
using Printf

const OUT_DIR   = joinpath(@__DIR__, "cross_validation_results")
const N_AGENTS  = 5_000
const SIM_START = 2000.0
const SIM_STOP  = 2050.0
const SIM_DT    = 0.25
const N_SEEDS   = 50

# ── Helpers ──

function extract_hpv_prev(sim)
    """Extract per-genotype HPV prevalence trajectories."""
    results = Dict{String, Vector{Float64}}()
    total_prev = nothing
    total_cin  = nothing

    for (name, dis) in sim.diseases
        md   = Starsim.module_data(dis)
        prev = copy(md.results[:prevalence].values)
        cin  = copy(md.results[:cin_prevalence].values)

        if total_prev === nothing
            total_prev = zeros(length(prev))
            total_cin  = zeros(length(prev))
        end
        total_prev .+= prev
        total_cin  .+= cin

        sname = String(name)
        results["hpv_prev_$sname"] = prev
        results["cin_prev_$sname"] = cin
    end

    n_genotypes = length(sim.diseases)
    results["hpv_prev"] = total_prev ./ n_genotypes
    results["cin_prev"] = total_cin ./ n_genotypes
    return results
end


function subsample_annual(traj; dt=SIM_DT)
    """Subsample quarterly Julia trajectory to annual (end-of-year points).
    Julia records every dt step; Python records annually at step resfreq-1.
    Both have run 4 dynamics steps per year, so indices 4,8,12,... in Julia."""
    step = round(Int, 1.0 / dt)  # 4 for dt=0.25
    return traj[step:step:end]   # indices 4, 8, 12, ...
end


function correlation(x, y)
    n = min(length(x), length(y))
    x = x[1:n]
    y = y[1:n]
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx < 1e-15 || sy < 1e-15
        return NaN
    end
    return mean((x .- mx) .* (y .- my)) / (sx * sy)
end


function level_ratio(jl_traj, py_traj; skip=5)
    """Compute mean(Julia)/mean(Python), skipping first `skip` annual points."""
    n = min(length(jl_traj), length(py_traj))
    s = min(skip + 1, n)
    jl_mean = mean(jl_traj[s:n])
    py_mean = mean(py_traj[s:n])
    return py_mean > 0 ? jl_mean / py_mean : NaN
end


# ── Scenario 1: Single HPV16 ──

function run_scenario1()
    println("\n", "="^60)
    println("  Scenario 1: Single HPV16 — $N_SEEDS seeds")
    println("="^60)

    all_hpv_prev = Vector{Vector{Float64}}()
    all_cin_prev = Vector{Vector{Float64}}()

    for seed in 1:N_SEEDS
        sim = HPVSim(
            genotypes    = [GenotypeDef(:hpv16; init_prev=0.05)],
            n_agents     = N_AGENTS,
            start        = SIM_START,
            stop         = SIM_STOP,
            dt           = SIM_DT,
            rand_seed    = seed,
            use_immunity = true,
            location     = :nigeria,
            verbose      = 0,
        )
        Starsim.run!(sim)
        res = extract_hpv_prev(sim)
        push!(all_hpv_prev, res["hpv_prev"])
        push!(all_cin_prev, res["cin_prev"])

        if seed % 10 == 0
            println("  Seed $seed/$N_SEEDS done")
        end
    end

    # Compute means (quarterly)
    npts = length(all_hpv_prev[1])
    mean_hpv = zeros(npts)
    mean_cin = zeros(npts)
    for i in 1:N_SEEDS
        mean_hpv .+= all_hpv_prev[i]
        mean_cin .+= all_cin_prev[i]
    end
    mean_hpv ./= N_SEEDS
    mean_cin ./= N_SEEDS

    # Subsample to annual
    annual_hpv = subsample_annual(mean_hpv)
    annual_cin = subsample_annual(mean_cin)

    return Dict(
        "hpv_prev_quarterly" => mean_hpv,
        "cin_prev_quarterly" => mean_cin,
        "hpv_prev_annual"    => annual_hpv,
        "cin_prev_annual"    => annual_cin,
    )
end


# ── Scenario 2: Two-genotype HPV16+18 ──

function run_scenario2()
    println("\n", "="^60)
    println("  Scenario 2: HPV16+18 — $N_SEEDS seeds")
    println("="^60)

    all_hpv_prev      = Vector{Vector{Float64}}()
    all_cin_prev      = Vector{Vector{Float64}}()
    all_hpv16_prev    = Vector{Vector{Float64}}()
    all_hpv18_prev    = Vector{Vector{Float64}}()
    all_cin16_prev    = Vector{Vector{Float64}}()
    all_cin18_prev    = Vector{Vector{Float64}}()

    for seed in 1:N_SEEDS
        sim = HPVSim(
            genotypes = [
                GenotypeDef(:hpv16; init_prev=0.05),
                GenotypeDef(:hpv18; init_prev=0.05),
            ],
            n_agents     = N_AGENTS,
            start        = SIM_START,
            stop         = SIM_STOP,
            dt           = SIM_DT,
            rand_seed    = seed,
            use_immunity = true,
            location     = :nigeria,
            verbose      = 0,
        )
        Starsim.run!(sim)
        res = extract_hpv_prev(sim)

        push!(all_hpv_prev,   res["hpv_prev"])
        push!(all_cin_prev,   res["cin_prev"])
        push!(all_hpv16_prev, res["hpv_prev_hpv16"])
        push!(all_hpv18_prev, res["hpv_prev_hpv18"])
        push!(all_cin16_prev, res["cin_prev_hpv16"])
        push!(all_cin18_prev, res["cin_prev_hpv18"])

        if seed % 10 == 0
            println("  Seed $seed/$N_SEEDS done")
        end
    end

    function compute_mean(vecs)
        npts = length(vecs[1])
        m = zeros(npts)
        for v in vecs
            m .+= v
        end
        return m ./ length(vecs)
    end

    mean_hpv   = compute_mean(all_hpv_prev)
    mean_cin   = compute_mean(all_cin_prev)
    mean_hpv16 = compute_mean(all_hpv16_prev)
    mean_hpv18 = compute_mean(all_hpv18_prev)
    mean_cin16 = compute_mean(all_cin16_prev)
    mean_cin18 = compute_mean(all_cin18_prev)

    return Dict(
        "hpv_prev_annual"       => subsample_annual(mean_hpv),
        "cin_prev_annual"       => subsample_annual(mean_cin),
        "hpv_prev_hpv16_annual" => subsample_annual(mean_hpv16),
        "hpv_prev_hpv18_annual" => subsample_annual(mean_hpv18),
        "cin_prev_hpv16_annual" => subsample_annual(mean_cin16),
        "cin_prev_hpv18_annual" => subsample_annual(mean_cin18),
    )
end


# ── Compare ──

function compare_trajectories(label, jl_annual, py_annual; skip=5)
    n = min(length(jl_annual), length(py_annual))
    s = min(skip + 1, n)
    jl = jl_annual[s:n]
    py = py_annual[s:n]
    r = correlation(jl, py)
    ratio = level_ratio(jl_annual, py_annual; skip=skip)
    status = abs(1.0 - ratio) ≤ 0.02 && r > 0.98 ? "✅" : "⚠️"
    @printf("  %-25s  r=%.4f  ratio=%.4f  (n=%d pts)  %s\n", label, r, ratio, length(jl), status)
    return (r=r, ratio=ratio)
end


function main()
    # ── Run Julia scenarios ──
    jl1 = run_scenario1()
    jl2 = run_scenario2()

    # ── Load Python multi-seed results ──
    py1_path = joinpath(OUT_DIR, "multiseed_single_hpv16_python.json")
    py2_path = joinpath(OUT_DIR, "multiseed_twogen_hpv16_18_python.json")

    if !isfile(py1_path) || !isfile(py2_path)
        println("\n⚠️  Python multi-seed results not found.")
        println("   Run: python multiseed_comparison.py first")
        return
    end

    py1 = JSON3.read(read(py1_path, String), Dict{String, Any})
    py2 = JSON3.read(read(py2_path, String), Dict{String, Any})

    py1_mean = py1["mean"]
    py2_mean = py2["mean"]

    # ── Alignment diagnostic: test all subsampling approaches for Scenario 1 HPV prev ──
    println("\n--- Alignment diagnostic (Scenario 1 HPV prevalence) ---")
    py_hpv = Float64.(py1_mean["hpv_prev"])
    jl_quarterly = jl1["hpv_prev_quarterly"]
    nq = length(jl_quarterly)
    println("  Julia quarterly: $nq pts, Python annual: $(length(py_hpv)) pts")

    for (label, sub) in [
        ("1:4:end",  jl_quarterly[1:4:end]),
        ("2:4:end",  jl_quarterly[2:4:end]),
        ("3:4:end",  jl_quarterly[3:4:end]),
        ("4:4:end",  jl_quarterly[4:4:end]),
        ("5:4:end",  jl_quarterly[5:4:end]),
    ]
        n = min(length(sub), length(py_hpv))
        r10 = correlation(sub[6:n], py_hpv[6:n])
        rat = mean(sub[6:n]) / mean(py_hpv[6:n])
        @printf("    %-10s  n=%d  r=%.4f  ratio=%.4f  jl[1]=%.4f py[1]=%.4f\n",
                label, n, r10, rat, sub[1], py_hpv[1])
    end
    println("---")

    # ── Report ──
    println("\n", "="^70)
    println("  TRAJECTORY COMPARISON: Julia vs Python ($(N_SEEDS)-seed means)")
    println("  ms_agent_ratio=1 in Python (structurally equivalent to Julia)")
    println("="^70)

    println("\n  Scenario 1: Single HPV16")
    println("  ", "-"^60)
    compare_trajectories("HPV prevalence",
        jl1["hpv_prev_annual"],
        Float64.(py1_mean["hpv_prev"]))
    compare_trajectories("CIN prevalence",
        jl1["cin_prev_annual"],
        Float64.(py1_mean["cin_prev"]))

    println("\n  Scenario 2: Two-genotype HPV16+18")
    println("  ", "-"^60)
    compare_trajectories("HPV prev (total)",
        jl2["hpv_prev_annual"],
        Float64.(py2_mean["hpv_prev"]))
    compare_trajectories("HPV16 prev",
        jl2["hpv_prev_hpv16_annual"],
        Float64.(py2_mean["hpv_prev_hpv16"]))
    compare_trajectories("HPV18 prev",
        jl2["hpv_prev_hpv18_annual"],
        Float64.(py2_mean["hpv_prev_hpv18"]))
    compare_trajectories("CIN prev (total)",
        jl2["cin_prev_annual"],
        Float64.(py2_mean["cin_prev"]))
    compare_trajectories("CIN16 prev",
        jl2["cin_prev_hpv16_annual"],
        Float64.(py2_mean["cin_prev_hpv16"]))
    compare_trajectories("CIN18 prev",
        jl2["cin_prev_hpv18_annual"],
        Float64.(py2_mean["cin_prev_hpv18"]))

    println("\n  Target: r > 0.98 AND ratio ∈ [0.99, 1.01]")
    println("="^70)
end

main()
