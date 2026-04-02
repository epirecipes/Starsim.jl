"""
    Metal GPU Benchmark — Large-scale SIR with static network caching

Tests GPU vs CPU at 1M–5M agents with cached edges (uploaded once).
This isolates the GPU kernel throughput from the per-step edge transfer
overhead that dominated the smaller benchmarks.

Usage:
    julia --project=. benchmark/benchmark_metal_large.jl
"""

using Starsim
using Metal
using Printf
using Statistics

const MetalExt = Base.get_extension(Starsim, :StarsimMetalExt)
using .MetalExt: gpu_step_state!, gpu_transmit!, gpu_transmit_cached!,
                 sync_to_gpu!, cache_edges!, uncache_edges!, gpu_step_fused!

# ============================================================================
# GPU loop — static network (edges cached on GPU)
# ============================================================================

function run_gpu_cached!(sim::Starsim.Sim; track_results::Bool=false)
    init!(sim)

    # Generate edges once on CPU
    for (_, net) in sim.networks
        Starsim.step!(net, sim)
    end

    gsim = to_gpu(sim; backend=:metal)
    cache_edges!(gsim)  # upload edges to GPU once

    npts = sim.t.npts
    disease_name = first(keys(sim.diseases))

    for ti in 1:npts
        sim.loop.ti = ti

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        # Recovery on GPU
        gpu_step_state!(gsim, disease_name; current_ti=ti)

        # Transmission on GPU (uses cached edges — no CPU→GPU edge transfer)
        gpu_transmit_cached!(gsim, disease_name; current_ti=ti)

        if track_results
            Starsim.to_cpu(gsim)
            Starsim.update_people_results!(sim.people, ti, sim.results)
            for (_, mod) in Starsim.all_modules(sim)
                Starsim.update_results!(mod, sim)
            end
            if ti < npts
                sync_to_gpu!(gsim)
            end
        end

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end
    end

    Starsim.to_cpu(gsim)
    sim.complete = true
    return sim
end

# ============================================================================
# GPU loop — dynamic network (re-upload edges each step, for comparison)
# ============================================================================

function run_gpu_dynamic!(sim::Starsim.Sim)
    init!(sim)
    gsim = to_gpu(sim; backend=:metal)

    npts = sim.t.npts
    disease_name = first(keys(sim.diseases))

    for ti in 1:npts
        sim.loop.ti = ti

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        gpu_step_state!(gsim, disease_name; current_ti=ti)

        for (_, net) in sim.networks
            Starsim.step!(net, sim)
        end

        gpu_transmit!(gsim, disease_name; current_ti=ti)

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end
    end

    Starsim.to_cpu(gsim)
    sim.complete = true
    return sim
end

# ============================================================================
# GPU loop — fused kernels with GPU-side RNG (zero per-step CPU→GPU transfer)
# ============================================================================

function run_gpu_fused!(sim::Starsim.Sim; track_results::Bool=false)
    init!(sim)

    # Generate edges once on CPU
    for (_, net) in sim.networks
        Starsim.step!(net, sim)
    end

    gsim = to_gpu(sim; backend=:metal)
    cache_edges!(gsim)

    npts = sim.t.npts
    disease_name = first(keys(sim.diseases))

    for ti in 1:npts
        sim.loop.ti = ti

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        # Single fused call: recovery + transmission with GPU-side RNG
        gpu_step_fused!(gsim, disease_name; current_ti=ti)

        if track_results
            Starsim.to_cpu(gsim)
            Starsim.update_people_results!(sim.people, ti, sim.results)
            for (_, mod) in Starsim.all_modules(sim)
                Starsim.update_results!(mod, sim)
            end
            if ti < npts
                sync_to_gpu!(gsim)
            end
        end

        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end
    end

    Starsim.to_cpu(gsim)
    sim.complete = true
    return sim
end

# ============================================================================
# Benchmark
# ============================================================================

function make_sim(n_agents; npts=100)
    dt = 1/365
    beta = 20.0
    dur_inf = 0.05  # ~18 days, R0 ≈ 10
    dur = npts * dt
    Sim(n_agents=n_agents, diseases=SIR(beta=beta, dur_inf=dur_inf, init_prev=0.01),
        networks=RandomNet(n_contacts=10),
        start=2000.0, stop=2000.0 + dur, dt=dt, verbose=0)
end

function benchmark_scale(n_agents; npts=100, n_runs=3)
    # --- CPU baseline ---
    run!(make_sim(n_agents; npts=npts); verbose=0)  # warmup
    GC.gc()

    cpu_times = Float64[]
    for _ in 1:n_runs
        t = @elapsed run!(make_sim(n_agents; npts=npts); verbose=0)
        push!(cpu_times, t)
        GC.gc()
    end

    # --- GPU with cached static edges (no per-step results) ---
    run_gpu_cached!(make_sim(n_agents; npts=npts))  # warmup
    GC.gc()

    gpu_cached_times = Float64[]
    for _ in 1:n_runs
        t = @elapsed run_gpu_cached!(make_sim(n_agents; npts=npts))
        push!(gpu_cached_times, t)
        GC.gc()
    end

    # --- GPU fused kernels with GPU-side RNG ---
    run_gpu_fused!(make_sim(n_agents; npts=npts))  # warmup
    GC.gc()

    gpu_fused_times = Float64[]
    for _ in 1:n_runs
        t = @elapsed run_gpu_fused!(make_sim(n_agents; npts=npts))
        push!(gpu_fused_times, t)
        GC.gc()
    end

    # --- GPU with cached edges + per-step results ---
    run_gpu_cached!(make_sim(n_agents; npts=npts); track_results=true)  # warmup
    GC.gc()

    gpu_cached_results_times = Float64[]
    for _ in 1:n_runs
        t = @elapsed run_gpu_cached!(make_sim(n_agents; npts=npts); track_results=true)
        push!(gpu_cached_results_times, t)
        GC.gc()
    end

    # --- GPU dynamic (per-step edge upload) ---
    run_gpu_dynamic!(make_sim(n_agents; npts=npts))  # warmup
    GC.gc()

    gpu_dynamic_times = Float64[]
    for _ in 1:n_runs
        t = @elapsed run_gpu_dynamic!(make_sim(n_agents; npts=npts))
        push!(gpu_dynamic_times, t)
        GC.gc()
    end

    cpu_t = median(cpu_times)
    gpu_ct = median(gpu_cached_times)
    gpu_ft = median(gpu_fused_times)
    gpu_crt = median(gpu_cached_results_times)
    gpu_dt = median(gpu_dynamic_times)
    ats = n_agents * npts / 1e6  # million agent-timesteps

    return (
        n_agents = n_agents,
        npts = npts,
        cpu = cpu_t,
        gpu_cached = gpu_ct,
        gpu_fused = gpu_ft,
        gpu_cached_results = gpu_crt,
        gpu_dynamic = gpu_dt,
        spd_cached = cpu_t / gpu_ct,
        spd_fused = cpu_t / gpu_ft,
        spd_cached_results = cpu_t / gpu_crt,
        spd_dynamic = cpu_t / gpu_dt,
        cpu_mats = ats / cpu_t,
        gpu_cached_mats = ats / gpu_ct,
        gpu_fused_mats = ats / gpu_ft,
        gpu_cached_results_mats = ats / gpu_crt,
        gpu_dynamic_mats = ats / gpu_dt,
    )
end

function format_n(n::Int)
    n >= 1_000_000 ? "$(n ÷ 1_000_000)M" :
    n >= 1_000 ? "$(n ÷ 1_000)K" : string(n)
end

function main()
    println("=" ^ 100)
    println("  Starsim.jl Metal GPU Benchmark — Large-scale SIR")
    println("  Device: ", Metal.current_device())
    println("  Model: SIR, beta=20, dur_inf=0.05yr (~18d), init_prev=1%, dt=daily")
    println("=" ^ 100)
    println()

    # Start smaller to warmup, then go big
    agent_counts = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    npts = 100  # 100 daily steps

    results = []
    for n in agent_counts
        @printf("Benchmarking %5s agents (%d steps) ...\n", format_n(n), npts)
        r = benchmark_scale(n; npts=npts)
        push!(results, r)
        @printf("  CPU=%6.2fs  Cached=%6.2fs  Fused=%6.2fs  Cch+Res=%6.2fs  Dynamic=%6.2fs\n",
                r.cpu, r.gpu_cached, r.gpu_fused, r.gpu_cached_results, r.gpu_dynamic)
        @printf("  Speedup: cached=%.2fx  fused=%.2fx  cached+res=%.2fx  dynamic=%.2fx\n\n",
                r.spd_cached, r.spd_fused, r.spd_cached_results, r.spd_dynamic)
    end

    # Summary table
    println("=" ^ 130)
    @printf("%-6s │ %8s  %9s  %9s  %9s  %9s │ %7s  %7s  %7s  %7s │ %7s  %7s  %7s\n",
            "Agents", "CPU (s)", "GPU cache", "GPU fused", "GPU c+res", "GPU dyn",
            "Spd cch", "Spd fus", "Spd c+r", "Spd dyn",
            "CPU M/s", "Cch M/s", "Fus M/s")
    println("─" ^ 130)
    for r in results
        @printf("%-6s │ %8.2f  %9.2f  %9.2f  %9.2f  %9.2f │ %6.2fx  %6.2fx  %6.2fx  %6.2fx │ %7.1f  %7.1f  %7.1f\n",
                format_n(r.n_agents),
                r.cpu, r.gpu_cached, r.gpu_fused, r.gpu_cached_results, r.gpu_dynamic,
                r.spd_cached, r.spd_fused, r.spd_cached_results, r.spd_dynamic,
                r.cpu_mats, r.gpu_cached_mats, r.gpu_fused_mats)
    end
    println("=" ^ 130)

    # Correctness check: fused vs CPU at 100K
    println("\nCorrectness check (100K agents, fused kernels, 100 steps):")
    sim_cpu = make_sim(100_000; npts=100)
    run!(sim_cpu; verbose=0)
    sim_fused = make_sim(100_000; npts=100)
    run_gpu_fused!(sim_fused; track_results=true)

    cpu_prev = Starsim.module_results(sim_cpu.diseases[:sir]).data[:prevalence].values
    fused_prev = Starsim.module_results(sim_fused.diseases[:sir]).data[:prevalence].values
    @printf("  CPU  peak=%.3f final=%.4f\n", maximum(cpu_prev), cpu_prev[end])
    @printf("  Fused peak=%.3f final=%.4f\n", maximum(fused_prev), fused_prev[end])

    # Also check cached vs fused (should be very close)
    sim_cached = make_sim(100_000; npts=100)
    run_gpu_cached!(sim_cached; track_results=true)
    cached_prev = Starsim.module_results(sim_cached.diseases[:sir]).data[:prevalence].values
    @printf("  Cached peak=%.3f final=%.4f\n", maximum(cached_prev), cached_prev[end])
end

main()
