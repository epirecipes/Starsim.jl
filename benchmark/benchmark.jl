"""
Starsim.jl vs Python starsim benchmark.

Two phases:

  1. Timing benchmark — runs an SIR sim at several population sizes in both
     Julia and Python, collects wall-clock timing and peak memory, prints a
     side-by-side comparison.

  2. Distributional validation — runs the same sim with many independent
     seeds and computes the maximum mean discrepancy (MMD²_u, RBF kernel,
     median bandwidth) between the joint distribution of summary statistics
     (peak prevalence, time of peak, attack rate) produced by each
     implementation. A permutation test gives a p-value, and a pure-Julia
     (split-half) MMD is reported as the within-implementation null
     reference. If Julia is faithfully reproducing the Python dynamics,
     the cross-implementation MMD should be statistically indistinguishable
     from the within-implementation MMD.

Usage (from the Starsim.jl project root):

    julia --project=. benchmark/benchmark.jl

Requires the Python venv at test/python_ref/.venv/ with `starsim` installed.

Outputs a JSON file at benchmark/results.json that the README is generated
from.
"""

using Starsim
using Printf
using Random
using Statistics
using Dates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

const SIZES         = [10_000, 50_000, 100_000, 200_000]
const TIMING_REPS   = 5
const VALIDATION_N  = 5_000     # smaller so 200 reps fit in a few minutes
const VALIDATION_R  = 200       # replicates per implementation
const VALIDATION_T  = 200.0     # stop time for validation runs (epidemic peaks well within this)
const PERM_REPS     = 2_000     # permutation reps for MMD p-value
const RESULTS_PATH  = joinpath(@__DIR__, "results.json")

# Common SIR parameters used everywhere
const PARS = (n_contacts=10, beta=0.05, dur_inf=10.0, init_prev=0.01)

# ---------------------------------------------------------------------------
# Julia run helpers
# ---------------------------------------------------------------------------

function jl_sim(n; seed, stop)
    Sim(n_agents=n, networks=RandomNet(n_contacts=PARS.n_contacts),
        diseases=SIR(beta=PARS.beta, dur_inf=PARS.dur_inf, init_prev=PARS.init_prev),
        dt=1.0, stop=stop, rand_seed=seed, verbose=0)
end

function jl_time_run(n; seed, stop=365.0)
    sim = jl_sim(n; seed=seed, stop=stop)
    GC.gc()
    t0 = time()
    run!(sim)
    return time() - t0
end

function jl_summary_run(n; seed, stop)
    sim = jl_sim(n; seed=seed, stop=stop)
    run!(sim)
    n_inf = get_result(sim, :sir, :n_infected)
    n_rec = get_result(sim, :sir, :n_recovered)
    peak_prev    = maximum(n_inf) / n
    time_of_peak = Float64(argmax(n_inf) - 1)   # ti index → days (dt=1)
    attack_rate  = n_rec[end] / n
    return (peak_prev, time_of_peak, attack_rate)
end

function jl_memory(n; seed=1)
    GC.gc(); GC.gc()
    mem0 = Base.gc_live_bytes()
    sim = jl_sim(n; seed=seed, stop=365.0)
    init!(sim)
    mem1 = Base.gc_live_bytes()
    run!(sim)
    mem2 = Base.gc_live_bytes()
    sim_alive = sim  # keep root so GC doesn't reclaim before measure
    return max(mem1 - mem0, mem2 - mem0) / 1024^2
end

# ---------------------------------------------------------------------------
# Python helpers — single subprocess that does both timing and summaries
# ---------------------------------------------------------------------------

function python_path()
    venv = joinpath(@__DIR__, "..", "test", "python_ref", ".venv")
    py = joinpath(venv, "bin", "python")
    isfile(py) || error("Python venv not found at $venv — run: `python -m venv $venv && $py -m pip install starsim`")
    return py
end

function py_run(sizes, timing_reps, val_n, val_reps, val_stop)
    py = python_path()
    out_path = joinpath(@__DIR__, ".python_out.json")
    script = """
import starsim as ss, time, json, tracemalloc
sizes        = $(collect(sizes))
timing_reps  = $(timing_reps)
val_n        = $(val_n)
val_reps     = $(val_reps)
val_stop     = $(val_stop)
N_CONTACTS, BETA, DUR_INF, INIT_PREV = $(PARS.n_contacts), $(PARS.beta), $(PARS.dur_inf), $(PARS.init_prev)

def make_sim(n, seed, stop):
    return ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=N_CONTACTS),
                  diseases=ss.SIR(beta=BETA, dur_inf=DUR_INF, init_prev=INIT_PREV),
                  dt=1, start=0, stop=stop, rand_seed=seed, verbose=0)

timing = {}
for n in sizes:
    times = []
    for r in range(timing_reps):
        sim = make_sim(n, 42+r, 365)
        t0 = time.perf_counter(); sim.run(); times.append(time.perf_counter()-t0)
    times.sort()
    tracemalloc.start()
    sim = make_sim(n, 42, 365); sim.run()
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    timing[str(n)] = {'min': times[0], 'med': times[len(times)//2],
                      'mem_peak_mb': peak/1024/1024}

val = []
for r in range(val_reps):
    sim = make_sim(val_n, 1000+r, val_stop)
    sim.run()
    sir = sim.diseases.sir
    n_inf = sir.results.n_infected.values
    n_rec = sir.results.n_recovered.values
    val.append([float(n_inf.max())/val_n,
                float(n_inf.argmax()),
                float(n_rec[-1])/val_n])

with open('$(out_path)', 'w') as f:
    json.dump({'timing': timing, 'val': val}, f)
"""
    run(`$py -c $script`)
    raw = read(out_path, String)
    rm(out_path; force=true)
    return parse_python_json(raw)
end

function parse_python_json(s)
    # Tiny ad-hoc parser tailored to the exact shape we emit:
    #   {"timing": {"<n>": {"min": ..., "med": ..., "mem_peak_mb": ...}, ...},
    #    "val":    [[a,b,c], [a,b,c], ...]}
    timing = Dict{Int,Dict{String,Float64}}()
    tmatch = match(r"\"timing\"\s*:\s*\{(.*?)\}\s*,\s*\"val\""s, s)
    tmatch === nothing && error("could not find timing block in:\n$s")
    for blk in eachmatch(r"\"(\d+)\"\s*:\s*\{([^\}]*)\}", tmatch[1])
        n = parse(Int, blk[1])
        d = Dict{String,Float64}()
        for fm in eachmatch(r"\"(\w+)\"\s*:\s*([\d.eE+-]+)", blk[2])
            d[fm[1]] = parse(Float64, fm[2])
        end
        timing[n] = d
    end

    vmatch = match(r"\"val\"\s*:\s*\[(.*)\]\s*\}\s*$"s, s)
    vmatch === nothing && error("could not find val block")
    val = NTuple{3,Float64}[]
    for m in eachmatch(r"\[\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*,\s*([\d.eE+-]+)\s*\]", vmatch[1])
        push!(val, (parse(Float64, m[1]), parse(Float64, m[2]), parse(Float64, m[3])))
    end
    return (timing=timing, val=val)
end

# ---------------------------------------------------------------------------
# Maximum mean discrepancy (RBF, unbiased)
# ---------------------------------------------------------------------------

"""
    standardize(X, Y) -> (Xz, Yz)

Z-score columns using pooled mean/std so the RBF bandwidth is on a sane
scale across mixed-unit summary features.
"""
function standardize(X::Matrix{Float64}, Y::Matrix{Float64})
    Z = vcat(X, Y)
    μ = mean(Z, dims=1)
    σ = std(Z, dims=1)
    σ[σ .== 0] .= 1.0
    return (X .- μ) ./ σ, (Y .- μ) ./ σ
end

"""
    pairwise_sqdist(A, B) -> Matrix{Float64}

Squared Euclidean distance between every row of A and every row of B.
"""
function pairwise_sqdist(A::Matrix{Float64}, B::Matrix{Float64})
    aa = sum(abs2, A; dims=2)
    bb = sum(abs2, B; dims=2)
    return aa .+ bb' .- 2 .* (A * B')
end

"""
    median_heuristic_sigma(Z) -> σ²

RBF bandwidth = median pairwise distance over the pooled sample.
"""
function median_heuristic_sigma(Z::Matrix{Float64})
    D = pairwise_sqdist(Z, Z)
    n = size(D, 1)
    # off-diagonal entries
    vals = Float64[]
    @inbounds for i in 1:n, j in i+1:n
        push!(vals, D[i,j])
    end
    return max(median(vals), 1e-12)
end

"""
    mmd2_unbiased(X, Y; sigma2) -> MMD²_u

Unbiased MMD² estimator with RBF kernel k(x,y) = exp(-||x-y||² / σ²).
"""
function mmd2_unbiased(X::Matrix{Float64}, Y::Matrix{Float64}; sigma2::Float64)
    m, n = size(X, 1), size(Y, 1)
    Kxx = exp.(-pairwise_sqdist(X, X) ./ sigma2)
    Kyy = exp.(-pairwise_sqdist(Y, Y) ./ sigma2)
    Kxy = exp.(-pairwise_sqdist(X, Y) ./ sigma2)
    sum_xx = (sum(Kxx) - sum(Kxx[i,i] for i in 1:m)) / (m*(m-1))
    sum_yy = (sum(Kyy) - sum(Kyy[i,i] for i in 1:n)) / (n*(n-1))
    sum_xy = sum(Kxy) / (m*n)
    return sum_xx + sum_yy - 2*sum_xy
end

"""
    mmd_permutation_test(X, Y; sigma2, B) -> (mmd_obs, p_value)

Permutation test: shuffle the pooled dataset B times, recompute MMD, and
report the fraction of permuted statistics ≥ the observed value.
"""
function mmd_permutation_test(X::Matrix{Float64}, Y::Matrix{Float64}; sigma2::Float64, B::Int=2000)
    m, n = size(X, 1), size(Y, 1)
    Z = vcat(X, Y)
    obs = mmd2_unbiased(X, Y; sigma2=sigma2)
    geq = 0
    rng = Random.Xoshiro(0xC0FFEE)
    for _ in 1:B
        perm = randperm(rng, m+n)
        Xp = Z[perm[1:m], :]
        Yp = Z[perm[m+1:end], :]
        mp = mmd2_unbiased(Xp, Yp; sigma2=sigma2)
        if mp >= obs
            geq += 1
        end
    end
    return obs, (geq + 1) / (B + 1)
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    println(repeat("=", 84))
    println("  Starsim.jl benchmark — ", Dates.now())
    println(repeat("=", 84))
    println("Configuration:")
    println("  Sizes             = $(SIZES)")
    println("  Timing reps       = $(TIMING_REPS)")
    println("  Validation n      = $(VALIDATION_N), reps = $(VALIDATION_R), stop = $(VALIDATION_T)")
    println("  Permutation reps  = $(PERM_REPS)")
    println()

    # --- Warmup Julia ---
    print("Warming up Julia... ")
    jl_time_run(1_000; seed=1)
    println("done.")

    # --- Julia timing ---
    println("\nJulia timing pass:")
    jl_timing = Dict{Int,Dict{String,Float64}}()
    for n in SIZES
        times = [jl_time_run(n; seed=42+i) for i in 1:TIMING_REPS]
        sort!(times)
        med = times[TIMING_REPS ÷ 2 + 1]
        jl_timing[n] = Dict("min"=>times[1], "med"=>med)
        @printf("  n=%-7d  min=%.3fs  med=%.3fs\n", n, times[1], med)
    end

    # --- Julia memory ---
    println("\nJulia memory pass:")
    for n in SIZES
        mem = jl_memory(n)
        jl_timing[n]["mem_mb"] = mem
        @printf("  n=%-7d  %.1f MB\n", n, mem)
    end

    # --- Julia validation summaries ---
    @printf("\nJulia validation: %d reps at n=%d...\n", VALIDATION_R, VALIDATION_N)
    jl_val = Matrix{Float64}(undef, VALIDATION_R, 3)
    t0 = time()
    for r in 1:VALIDATION_R
        jl_val[r, :] .= jl_summary_run(VALIDATION_N; seed=1000+r, stop=VALIDATION_T)
        if r % 50 == 0
            @printf("  %3d / %d done\n", r, VALIDATION_R)
        end
    end
    @printf("  total wall time: %.1fs\n", time()-t0)

    # --- Python (timing + validation) ---
    println("\nRunning Python (timing + $(VALIDATION_R) validation reps at n=$(VALIDATION_N))...")
    t0 = time()
    py = py_run(SIZES, TIMING_REPS, VALIDATION_N, VALIDATION_R, VALIDATION_T)
    @printf("  total wall time: %.1fs\n", time()-t0)
    py_val = Matrix{Float64}([t[i] for t in py.val, i in 1:3])

    # --- MMD distributional validation ---
    println("\n", repeat("=", 84))
    println("  Distributional validation (MMD², RBF kernel, median bandwidth)")
    println(repeat("=", 84))

    # Standardize, then compute observed MMD between Julia and Python.
    Xs, Ys = standardize(jl_val, py_val)
    sigma2 = median_heuristic_sigma(vcat(Xs, Ys))
    @printf("\n  Pooled bandwidth σ² (median heuristic): %.4f\n", sigma2)

    println("\n  Computing cross-implementation MMD (Julia vs Python)...")
    mmd_xy, p_xy = mmd_permutation_test(Xs, Ys; sigma2=sigma2, B=PERM_REPS)
    @printf("    MMD²_u(Julia, Python) = %+.5f   permutation p = %.4f\n", mmd_xy, p_xy)

    # Within-Julia null reference: split Julia into two halves of equal size
    # and compute MMD between them. Should be statistically indistinguishable
    # from zero for any reasonable implementation.
    half = VALIDATION_R ÷ 2
    Xs_a = Xs[1:half, :]
    Xs_b = Xs[half+1:2*half, :]
    println("\n  Computing within-Julia MMD (split-half null reference)...")
    mmd_jj, p_jj = mmd_permutation_test(Xs_a, Xs_b; sigma2=sigma2, B=PERM_REPS)
    @printf("    MMD²_u(Julia_A, Julia_B) = %+.5f   permutation p = %.4f\n", mmd_jj, p_jj)

    # And a within-Python null for symmetry
    Ys_a = Ys[1:half, :]
    Ys_b = Ys[half+1:2*half, :]
    println("\n  Computing within-Python MMD (split-half null reference)...")
    mmd_pp, p_pp = mmd_permutation_test(Ys_a, Ys_b; sigma2=sigma2, B=PERM_REPS)
    @printf("    MMD²_u(Python_A, Python_B) = %+.5f   permutation p = %.4f\n", mmd_pp, p_pp)

    # --- Per-feature summary stats ---
    println("\n  Per-feature mean ± std:")
    feature_names = ("peak_prev", "time_of_peak", "attack_rate")
    @printf("    %-15s %20s %20s\n", "feature", "Julia", "Python")
    for (i, name) in enumerate(feature_names)
        @printf("    %-15s  %8.4f ± %.4f    %8.4f ± %.4f\n",
                name, mean(jl_val[:, i]), std(jl_val[:, i]),
                       mean(py_val[:, i]), std(py_val[:, i]))
    end

    # --- Final timing report ---
    println()
    println(repeat("=", 84))
    println("  Timing summary (SIR, dt=1, stop=365, n_contacts=10, beta=0.05, dur_inf=10)")
    println(repeat("=", 84))
    @printf("  %-10s │ %10s %10s │ %10s %10s │ %8s\n",
            "n_agents", "Julia med", "Julia min", "Python med", "Python min", "Speedup")
    println("  ", repeat("─", 80))
    for n in SIZES
        jmed = jl_timing[n]["med"]; jmin = jl_timing[n]["min"]
        pmed = py.timing[n]["med"]; pmin = py.timing[n]["min"]
        @printf("  %-10d │ %8.3fs %8.3fs │ %8.3fs %8.3fs │ %6.1fx\n",
                n, jmed, jmin, pmed, pmin, pmed / jmed)
    end
    println("  ", repeat("─", 80))
    n = SIZES[end]
    jl_aps = n * 365 / jl_timing[n]["med"]
    py_aps = n * 365 / py.timing[n]["med"]
    @printf("  Throughput at n=%d: Julia %.1fM agent-ts/s, Python %.1fM agent-ts/s\n",
            n, jl_aps/1e6, py_aps/1e6)

    # --- Memory ---
    println()
    println(repeat("=", 60))
    println("  Memory")
    println(repeat("=", 60))
    @printf("  %-10s │ %12s │ %12s │ %8s\n",
            "n_agents", "Julia (MB)", "Python (MB)", "Ratio")
    println("  ", repeat("─", 56))
    for n in SIZES
        jmem = jl_timing[n]["mem_mb"]
        pmem = py.timing[n]["mem_peak_mb"]
        @printf("  %-10d │ %10.1f │ %10.1f │ %6.1fx\n", n, jmem, pmem, pmem / jmem)
    end

    # --- Persist ---
    write_results(RESULTS_PATH, jl_timing, py.timing, jl_val, py_val,
                  mmd_xy, p_xy, mmd_jj, p_jj, mmd_pp, p_pp, sigma2)
    println("\nResults written to $(RESULTS_PATH)")
end

function write_results(path, jl_timing, py_timing, jl_val, py_val,
                       mmd_xy, p_xy, mmd_jj, p_jj, mmd_pp, p_pp, sigma2)
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"date\": \"$(Dates.now())\",")
        println(io, "  \"config\": {")
        println(io, "    \"sizes\": $(SIZES),")
        println(io, "    \"timing_reps\": $(TIMING_REPS),")
        println(io, "    \"validation_n\": $(VALIDATION_N),")
        println(io, "    \"validation_reps\": $(VALIDATION_R),")
        println(io, "    \"validation_stop\": $(VALIDATION_T),")
        println(io, "    \"perm_reps\": $(PERM_REPS)")
        println(io, "  },")
        println(io, "  \"timing\": {")
        for (i, n) in enumerate(SIZES)
            j = jl_timing[n]; p = py_timing[n]
            sep = i == length(SIZES) ? "" : ","
            @printf(io, "    \"%d\": {\"jl_med\": %.6f, \"jl_min\": %.6f, \"jl_mem_mb\": %.3f, \"py_med\": %.6f, \"py_min\": %.6f, \"py_mem_mb\": %.3f}%s\n",
                    n, j["med"], j["min"], j["mem_mb"], p["med"], p["min"], p["mem_peak_mb"], sep)
        end
        println(io, "  },")
        println(io, "  \"validation\": {")
        @printf(io, "    \"sigma2\": %.6f,\n", sigma2)
        @printf(io, "    \"mmd_julia_python\":   %.8f,\n", mmd_xy)
        @printf(io, "    \"p_julia_python\":     %.4f,\n", p_xy)
        @printf(io, "    \"mmd_julia_split\":    %.8f,\n", mmd_jj)
        @printf(io, "    \"p_julia_split\":      %.4f,\n", p_jj)
        @printf(io, "    \"mmd_python_split\":   %.8f,\n", mmd_pp)
        @printf(io, "    \"p_python_split\":     %.4f,\n", p_pp)
        println(io, "    \"jl_summary\": {")
        for (i, name) in enumerate(("peak_prev", "time_of_peak", "attack_rate"))
            sep = i == 3 ? "" : ","
            @printf(io, "      \"%s\": {\"mean\": %.6f, \"std\": %.6f}%s\n",
                    name, mean(jl_val[:,i]), std(jl_val[:,i]), sep)
        end
        println(io, "    },")
        println(io, "    \"py_summary\": {")
        for (i, name) in enumerate(("peak_prev", "time_of_peak", "attack_rate"))
            sep = i == 3 ? "" : ","
            @printf(io, "      \"%s\": {\"mean\": %.6f, \"std\": %.6f}%s\n",
                    name, mean(py_val[:,i]), std(py_val[:,i]), sep)
        end
        println(io, "    }")
        println(io, "  }")
        println(io, "}")
    end
end

main()
