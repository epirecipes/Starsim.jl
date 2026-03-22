"""
    Metal GPU Benchmark — CPU vs GPU SIR simulation

Compares full SIR simulation performance between:
1. CPU-only (standard `run!`)
2. GPU-accelerated (network step on CPU, disease dynamics on Metal GPU)

The GPU path mirrors the CPU loop order:
  - step_state! (recovery) → GPU kernel
  - networks.step (edge rebuild) → CPU
  - diseases.step (transmission) → GPU kernel
  - update_results! → CPU (requires to_cpu download)

Usage:
    julia --project=. benchmark/benchmark_metal.jl
"""

using Starsim
using Metal
using Printf
using Statistics

# Import GPU functions from the extension module
const MetalExt = Base.get_extension(Starsim, :StarsimMetalExt)
using .MetalExt: GPUSim, gpu_step!, gpu_step_state!, gpu_transmit!, gpu_waning!, sync_to_gpu!

# ============================================================================
# GPU simulation loop — mirrors CPU loop order
# ============================================================================

"""
Run an SIR simulation with disease dynamics offloaded to Metal GPU.
Returns (sim, elapsed_seconds).

Follows the exact Python starsim loop order:
  5. diseases.step_state (recovery)   → GPU
  6. connectors.step                  → (none for basic SIR)
  7. networks.step (edge rebuild)     → CPU
  8. interventions.step               → (none for basic SIR)
  9. diseases.step (transmission)     → GPU
  10-12. die, results, update        → CPU
"""
function run_gpu_sir!(sim::Starsim.Sim)
    init!(sim)
    gsim = Starsim.to_gpu(sim)

    npts = sim.t.npts
    disease_name = first(keys(sim.diseases))

    for ti in 1:npts
        sim.loop.ti = ti

        # Step 2: start_step! for all modules (advances RNG distributions)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        # Step 5: Disease state transitions (recovery) on GPU
        gpu_step_state!(gsim, disease_name; current_ti=ti)

        # Step 7: Network edge rebuild on CPU
        for (_, net) in sim.networks
            Starsim.step!(net, sim)
        end

        # Step 9: Disease transmission on GPU
        gpu_transmit!(gsim, disease_name; current_ti=ti)

        # Steps 10-12: Download state for results bookkeeping on CPU
        Starsim.to_cpu(gsim)
        Starsim.update_people_results!(sim.people, ti, sim.results)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.update_results!(mod, sim)
        end

        # Step 14: finish_step! for all modules (advance module timestep)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end

        # Step 15: people.finish_step (aging, dead removal)
        Starsim.finish_step!(sim.people, sim.pars.dt, sim.pars.use_aging)

        # Re-upload for next iteration
        if ti < npts
            sync_to_gpu!(gsim)
        end
    end

    sim.complete = true
    return sim
end

"""
Run an SIR simulation with GPU, but only sync back to CPU at the end
(no per-step results tracking). This shows the GPU's upper bound throughput.

Still follows the correct Python loop order:
  5. step_state → 7. networks → 9. transmit
"""
function run_gpu_sir_nosync!(sim::Starsim.Sim)
    init!(sim)
    gsim = Starsim.to_gpu(sim)

    npts = sim.t.npts
    disease_name = first(keys(sim.diseases))

    for ti in 1:npts
        sim.loop.ti = ti

        # start_step! for all modules
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        # Step 5: Disease state transitions (recovery) on GPU
        gpu_step_state!(gsim, disease_name; current_ti=ti)

        # Step 7: Network edge rebuild on CPU
        for (_, net) in sim.networks
            Starsim.step!(net, sim)
        end

        # Step 9: Disease transmission on GPU
        gpu_transmit!(gsim, disease_name; current_ti=ti)

        # finish_step! for all modules (advance timestep)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end
    end

    Starsim.to_cpu(gsim)
    sim.complete = true
    return sim
end

# ============================================================================
# Benchmark harness
# ============================================================================

function benchmark_single(n_agents::Int; n_warmup=1, n_runs=3, npts=200)
    # Daily timesteps with beta/dur_inf tuned for R0 ≈ 10
    dt = 1/365
    beta = 20.0       # high beta needed for daily dt: R0 ≈ p*contacts*dur_ts ≈ 0.053*10*18 ≈ 10
    dur_inf = 0.05    # ~18 days
    dur = npts * dt

    make_sim() = Sim(n_agents=n_agents, diseases=SIR(beta=beta, dur_inf=dur_inf, init_prev=0.01),
                     networks=RandomNet(n_contacts=10),
                     start=2000.0, stop=2000.0 + dur, dt=dt, verbose=0)

    # --- CPU baseline ---
    for _ in 1:n_warmup
        run!(make_sim(); verbose=0)
    end
    GC.gc()

    cpu_times = Float64[]
    cpu_sim = nothing
    for _ in 1:n_runs
        sim = make_sim()
        t = @elapsed run!(sim; verbose=0)
        push!(cpu_times, t)
        cpu_sim = sim
    end

    # --- GPU with per-step sync (full results) ---
    for _ in 1:n_warmup
        run_gpu_sir!(make_sim())
    end
    GC.gc()

    gpu_sync_times = Float64[]
    gpu_sync_sim = nothing
    for _ in 1:n_runs
        sim = make_sim()
        t = @elapsed run_gpu_sir!(sim)
        push!(gpu_sync_times, t)
        gpu_sync_sim = sim
    end

    # --- GPU without per-step sync (throughput ceiling) ---
    for _ in 1:n_warmup
        run_gpu_sir_nosync!(make_sim())
    end
    GC.gc()

    gpu_nosync_times = Float64[]
    gpu_nosync_sim = nothing
    for _ in 1:n_runs
        sim = make_sim()
        t = @elapsed run_gpu_sir_nosync!(sim)
        push!(gpu_nosync_times, t)
        gpu_nosync_sim = sim
    end

    # --- Memory ---
    cpu_mem = @allocated begin
        run!(make_sim(); verbose=0)
    end

    gpu_mem = @allocated begin
        run_gpu_sir!(make_sim())
    end

    return (
        n_agents     = n_agents,
        npts         = npts,
        cpu_time     = median(cpu_times),
        gpu_sync     = median(gpu_sync_times),
        gpu_nosync   = median(gpu_nosync_times),
        cpu_mem_mb   = cpu_mem / 1e6,
        gpu_mem_mb   = gpu_mem / 1e6,
        speedup_sync   = median(cpu_times) / median(gpu_sync_times),
        speedup_nosync = median(cpu_times) / median(gpu_nosync_times),
        cpu_ats      = n_agents * npts / median(cpu_times) / 1e6,
        gpu_sync_ats = n_agents * npts / median(gpu_sync_times) / 1e6,
        gpu_nosync_ats = n_agents * npts / median(gpu_nosync_times) / 1e6,
    )
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 80)
    println("  Starsim.jl Metal GPU Benchmark — SIR model")
    println("  Device: ", Metal.current_device())
    println("=" ^ 80)
    println()

    agent_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    npts = 200  # 200 daily steps

    results = []
    for n in agent_counts
        @printf("Benchmarking n_agents=%7d ...", n)
        r = benchmark_single(n; npts=npts)
        push!(results, r)
        @printf(" CPU=%.4fs  GPU(sync)=%.4fs  GPU(nosync)=%.4fs  speedup=%.1fx / %.1fx\n",
                r.cpu_time, r.gpu_sync, r.gpu_nosync, r.speedup_sync, r.speedup_nosync)
    end

    # Summary table
    println()
    println("=" ^ 105)
    @printf("%-10s │ %10s  %10s  %10s │ %8s  %8s │ %9s  %9s  %9s\n",
            "Agents", "CPU (s)", "GPU sync", "GPU nosync", "Spd sync", "Spd nosn", "CPU M a/s", "GPUs M/s", "GPUn M/s")
    println("─" ^ 105)
    for r in results
        @printf("%-10s │ %10.4f  %10.4f  %10.4f │ %7.1fx  %7.1fx │ %9.1f  %9.1f  %9.1f\n",
                format_agents(r.n_agents),
                r.cpu_time, r.gpu_sync, r.gpu_nosync,
                r.speedup_sync, r.speedup_nosync,
                r.cpu_ats, r.gpu_sync_ats, r.gpu_nosync_ats)
    end
    println("=" ^ 105)

    # Memory table
    println()
    println("Memory usage:")
    @printf("%-10s │ %10s  %10s\n", "Agents", "CPU (MB)", "GPU (MB)")
    println("─" ^ 35)
    for r in results
        @printf("%-10s │ %10.1f  %10.1f\n",
                format_agents(r.n_agents), r.cpu_mem_mb, r.gpu_mem_mb)
    end
    println()

    # Check correctness — GPU should produce a valid SIR epidemic
    println("Correctness check (2000 agents, 200 daily steps, beta=20, dur_inf=0.05):")
    dt_check = 1/365
    npts_check = 200
    dur_check = npts_check * dt_check
    sim_cpu = Sim(n_agents=2000, diseases=SIR(beta=20.0, dur_inf=0.05, init_prev=0.01),
                  networks=RandomNet(n_contacts=10),
                  start=2000.0, stop=2000.0 + dur_check, dt=dt_check, verbose=0)
    run!(sim_cpu; verbose=0)
    sim_gpu = Sim(n_agents=2000, diseases=SIR(beta=20.0, dur_inf=0.05, init_prev=0.01),
                  networks=RandomNet(n_contacts=10),
                  start=2000.0, stop=2000.0 + dur_check, dt=dt_check, verbose=0)
    run_gpu_sir!(sim_gpu)

    cpu_res = Starsim.module_results(sim_cpu.diseases[:sir])
    gpu_res = Starsim.module_results(sim_gpu.diseases[:sir])

    cpu_prev = cpu_res.data[:prevalence].values
    gpu_prev = gpu_res.data[:prevalence].values
    println("  CPU peak prevalence: $(round(maximum(cpu_prev), digits=4))")
    println("  GPU peak prevalence: $(round(maximum(gpu_prev), digits=4))")
    println("  CPU final prevalence: $(round(cpu_prev[end], digits=4))")
    println("  GPU final prevalence: $(round(gpu_prev[end], digits=4))")
    println("  Both show epidemic: CPU=$(maximum(cpu_prev) > 0.01) GPU=$(maximum(gpu_prev) > 0.01)")
end

function format_agents(n::Int)
    if n >= 1_000_000
        return "$(n ÷ 1_000_000)M"
    elseif n >= 1_000
        return "$(n ÷ 1_000)K"
    else
        return string(n)
    end
end

main()
