"""
Starsim.jl vs Python starsim benchmark.

Runs an SIR simulation at several population sizes in both Julia and Python,
then prints a side-by-side comparison table with timing and memory usage.

Usage (from the Starsim.jl project root):

    julia --project=. benchmark/benchmark.jl

Requires the Python venv at test/python_ref/.venv/ with starsim installed.
"""

using Starsim
using Printf

# ---------------------------------------------------------------------------
# Julia benchmark
# ---------------------------------------------------------------------------

function bench_julia(n_agents; seed=42)
    sim = Sim(n_agents=n_agents, networks=RandomNet(n_contacts=10),
              diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
              dt=1.0, stop=365.0, rand_seed=seed, verbose=0)
    GC.gc()
    mem_before = Base.gc_live_bytes()
    t0 = time()
    run!(sim)
    elapsed = time() - t0
    mem_after = Base.gc_live_bytes()
    mem_mb = (mem_after - mem_before) / 1024^2
    return elapsed, mem_mb
end

function measure_julia_memory(n_agents; seed=42)
    GC.gc(); GC.gc()
    mem0 = Base.gc_live_bytes()
    sim = Sim(n_agents=n_agents, networks=RandomNet(n_contacts=10),
              diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
              dt=1.0, stop=365.0, rand_seed=seed, verbose=0)
    init!(sim)
    mem1 = Base.gc_live_bytes()
    alloc_mb = (mem1 - mem0) / 1024^2
    run!(sim)
    GC.gc()
    mem2 = Base.gc_live_bytes()
    peak_mb = (mem2 - mem0) / 1024^2
    return alloc_mb, peak_mb
end

# ---------------------------------------------------------------------------
# Python benchmark (called via the venv interpreter)
# ---------------------------------------------------------------------------

function bench_python(sizes; n_repeats=3)
    venv = joinpath(@__DIR__, "..", "test", "python_ref", ".venv")
    python = joinpath(venv, "bin", "python")
    isfile(python) || error("Python venv not found at $venv — run: python -m venv $venv && $python -m pip install starsim")

    script = """
import starsim as ss, time, json, sys, tracemalloc
sizes = $(sizes)
n_repeats = $(n_repeats)
results = {}
for n in sizes:
    times = []
    for r in range(n_repeats):
        sim = ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=10),
                     diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
                     dt=1, start=0, stop=365, rand_seed=42+r, verbose=0)
        t0 = time.perf_counter()
        sim.run()
        times.append(time.perf_counter() - t0)
    times.sort()
    # Memory: measure once with tracemalloc
    tracemalloc.start()
    sim = ss.Sim(n_agents=n, networks=ss.RandomNet(n_contacts=10),
                 diseases=ss.SIR(beta=0.05, dur_inf=10, init_prev=0.01),
                 dt=1, start=0, stop=365, rand_seed=42, verbose=0)
    sim.run()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results[n] = {"min": times[0], "med": times[len(times)//2],
                  "mem_cur_mb": current/1024/1024, "mem_peak_mb": peak/1024/1024}
json.dump(results, sys.stdout)
"""
    output = read(`$python -c $script`, String)
    return JSON_parse(output)
end

# Minimal JSON parser (avoids adding a dependency)
function JSON_parse(s)
    result = Dict{Int,Dict{String,Float64}}()
    s = strip(s, ['{', '}', ' '])
    for block in split(s, r"\},\s*")
        block = strip(block, ['{', '}', ' '])
        # Extract key
        km = match(r"\"(\d+)\"\s*:\s*\{(.+)", block)
        km === nothing && continue
        n = parse(Int, km[1])
        inner = km[2]
        d = Dict{String,Float64}()
        for fm in eachmatch(r"\"(\w+)\"\s*:\s*([\d.eE+-]+)", inner)
            d[fm[1]] = parse(Float64, fm[2])
        end
        result[n] = d
    end
    return result
end

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    n_repeats = 5

    # Warmup Julia
    bench_julia(100)

    # --- Julia timing ---
    println("Running Julia benchmarks...")
    jl = Dict{Int,Dict{String,Float64}}()
    for n in sizes
        times = Float64[]
        for r in 1:n_repeats
            t, _ = bench_julia(n; seed=42+r)
            push!(times, t)
        end
        sort!(times)
        jl[n] = Dict("min" => times[1], "med" => times[n_repeats ÷ 2 + 1])
        @printf("  n=%d done (med=%.2fs)\n", n, jl[n]["med"])
    end

    # --- Julia memory ---
    println("Measuring Julia memory...")
    for n in sizes
        alloc, peak = measure_julia_memory(n)
        jl[n]["mem_mb"] = max(alloc, peak)
        @printf("  n=%d: %.1f MB\n", n, jl[n]["mem_mb"])
    end

    # --- Python ---
    println("Running Python benchmarks...")
    py = bench_python(sizes; n_repeats=n_repeats)

    # --- Timing report ---
    println()
    println("=" ^ 84)
    println("  Starsim.jl vs Python starsim — SIR benchmark")
    println("  (dt=1, stop=365, n_contacts=10, beta=0.05, dur_inf=10, init_prev=0.01)")
    println("=" ^ 84)
    @printf("%-10s │ %10s %10s │ %10s %10s │ %8s\n",
            "n_agents", "Julia med", "Julia min", "Python med", "Python min", "Speedup")
    println("─" ^ 84)

    for n in sizes
        jmed = jl[n]["med"]
        jmin = jl[n]["min"]
        pmed = py[n]["med"]
        pmin = py[n]["min"]
        speedup = pmed / jmed
        @printf("%-10d │ %8.3fs %8.3fs │ %8.3fs %8.3fs │ %6.1fx\n",
                n, jmed, jmin, pmed, pmin, speedup)
    end
    println("─" ^ 84)

    n = sizes[end]
    jl_aps = n * 365 / jl[n]["med"]
    py_aps = n * 365 / py[n]["med"]
    @printf("Throughput at n=%d:  Julia %.1fM agent-ts/s  Python %.1fM agent-ts/s\n",
            n, jl_aps / 1e6, py_aps / 1e6)

    # --- Memory report ---
    println()
    println("=" ^ 60)
    println("  Memory usage")
    println("=" ^ 60)
    @printf("%-10s │ %12s │ %12s │ %8s\n",
            "n_agents", "Julia (MB)", "Python (MB)", "Ratio")
    println("─" ^ 60)

    for n in sizes
        jmem = jl[n]["mem_mb"]
        pmem = get(py[n], "mem_peak_mb", NaN)
        ratio = pmem / jmem
        @printf("%-10d │ %10.1f │ %10.1f │ %6.1fx\n",
                n, jmem, pmem, ratio)
    end
    println("─" ^ 60)
    println()
end

main()
