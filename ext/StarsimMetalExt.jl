"""
    StarsimMetalExt

Extension providing Apple Silicon GPU support for Starsim.jl via Metal.jl.

Offloads compute-heavy agent-state operations (transmission, recovery) to the
GPU while keeping structurally dynamic operations (network rewiring, births,
deaths) on the CPU.

# Architecture — what runs where

**GPU (MtlVector, Float32/UInt8)**
- Agent state arrays: `alive`, `age`, `female`
- Disease state arrays: `susceptible`, `infected`, `recovered`,
  `rel_sus`, `rel_trans`, `ti_infected`, `ti_recovered`
- Transmission kernel (per-edge probability evaluation)
- State-transition kernel (recovery, waning immunity)

**CPU (Vector, Float64/Bool)**
- Network edge lists (`p1`, `p2`, `beta`) — rebuilt each timestep
- People management (births, deaths, UID tracking, dynamic resizing)
- Results bookkeeping and infection-source logging
- All non-disease modules (demographics, interventions, analyzers)

# Type conventions
- Metal does **not** support Float64 natively → all GPU floats are `Float32`
- Metal has no native `Bool` → GPU booleans are stored as `UInt8` (0x00/0x01)
- Edge indices are `Int32` on GPU (sufficient for up to 2 billion agents)

# Usage
```julia
using Starsim, Metal

sim = Sim(n_agents=100_000, diseases=SIR(beta=0.05), networks=RandomNet())
init!(sim)
gsim = to_gpu(sim)                     # Upload state arrays to GPU

for ti in 1:sim.t.npts
    for (_, net) in sim.networks       # Network updates stay on CPU
        step!(net, sim)
    end
    gpu_step!(gsim, :sir; current_ti=ti)  # Transmission + recovery on GPU
end

sim = to_cpu(gsim)                     # Download results to CPU
```

# Limitations
- Infection-source logging is skipped on GPU (no dynamic allocation in kernels)
- Disease-induced death (`p_death > 0`) is not handled in GPU kernels; use
  CPU fallback or call `to_cpu` before death processing
- CRN (common random numbers) mode is CPU-only; GPU uses pre-generated randoms
- Agent births/deaths between GPU steps require `sync_to_gpu!` to re-upload
"""
module StarsimMetalExt

using Starsim
using Metal

# ============================================================================
# GPU array container types
# ============================================================================

"""
    GPUPeopleArrays

GPU-resident mirror of People state arrays.

Only the arrays needed for disease dynamics are uploaded. Index states (`uid`,
`slot`) and management fields (`ti_dead`, `ti_removed`, `scale`, `parent`)
stay on CPU because they are only used in demographic operations.
"""
struct GPUPeopleArrays
    alive::MtlVector{UInt8}
    age::MtlVector{Float32}
    female::MtlVector{UInt8}
    n::Int
end

"""
    GPUDiseaseArrays

GPU-resident mirror of disease state arrays for one `AbstractInfection`.

Supports SIR (with `recovered`/`ti_recovered`) and SIS (those fields are
`nothing`). All Float64 values are stored as Float32; Bool as UInt8.
"""
struct GPUDiseaseArrays
    susceptible::MtlVector{UInt8}
    infected::MtlVector{UInt8}
    recovered::Union{MtlVector{UInt8}, Nothing}
    ti_infected::MtlVector{Float32}
    ti_recovered::Union{MtlVector{Float32}, Nothing}
    rel_sus::MtlVector{Float32}
    rel_trans::MtlVector{Float32}
    n::Int
end

"""
    GPUSim

Wrapper pairing a CPU `Sim` with GPU-mirrored state arrays. Created by
[`to_gpu`](@ref), converted back with [`to_cpu`](@ref).

The CPU `Sim` remains the source of truth for networks, demographics,
results, and all non-disease module state. The `GPUDiseaseArrays` hold
the authoritative disease state while on GPU.
"""
struct GPUSim
    sim::Starsim.Sim
    people::GPUPeopleArrays
    diseases::Dict{Symbol, GPUDiseaseArrays}
end

function Base.show(io::IO, g::GPUSim)
    nd = length(g.diseases)
    names = join(keys(g.diseases), ", ")
    print(io, "GPUSim(n_agents=$(g.people.n), diseases=[$names])")
end

# ============================================================================
# CPU → GPU conversion
# ============================================================================

function _people_to_gpu(people::Starsim.People)
    n = people.alive.len_tot
    GPUPeopleArrays(
        MtlVector{UInt8}(UInt8.(people.alive.raw)),
        MtlVector{Float32}(Float32.(people.age.raw)),
        MtlVector{UInt8}(UInt8.(people.female.raw)),
        n,
    )
end

function _sir_to_gpu(d::Starsim.SIR)
    n = d.infection.infected.len_tot
    GPUDiseaseArrays(
        MtlVector{UInt8}(UInt8.(d.infection.susceptible.raw)),
        MtlVector{UInt8}(UInt8.(d.infection.infected.raw)),
        MtlVector{UInt8}(UInt8.(d.recovered.raw)),
        MtlVector{Float32}(Float32.(d.infection.ti_infected.raw)),
        MtlVector{Float32}(Float32.(d.ti_recovered.raw)),
        MtlVector{Float32}(Float32.(d.infection.rel_sus.raw)),
        MtlVector{Float32}(Float32.(d.infection.rel_trans.raw)),
        n,
    )
end

function _sis_to_gpu(d::Starsim.SIS)
    n = d.infection.infected.len_tot
    GPUDiseaseArrays(
        MtlVector{UInt8}(UInt8.(d.infection.susceptible.raw)),
        MtlVector{UInt8}(UInt8.(d.infection.infected.raw)),
        nothing,
        MtlVector{Float32}(Float32.(d.infection.ti_infected.raw)),
        nothing,
        MtlVector{Float32}(Float32.(d.infection.rel_sus.raw)),
        MtlVector{Float32}(Float32.(d.infection.rel_trans.raw)),
        n,
    )
end

"""
    to_gpu(sim::Starsim.Sim) → GPUSim

Copy a CPU simulation's state arrays to Metal GPU memory.

The simulation must be initialized (`init!(sim)`) before calling this.
Returns a [`GPUSim`](@ref) that pairs the original `Sim` with GPU arrays.
Network edge lists are NOT copied (they are rebuilt each timestep on CPU).

Currently supports `SIR` and `SIS` diseases. Unsupported disease types
are skipped with a warning.
"""
function Starsim.to_gpu(sim::Starsim.Sim)
    sim.initialized || error("Sim must be initialized before to_gpu(). Call init!(sim) first.")

    gpu_people = _people_to_gpu(sim.people)
    gpu_diseases = Dict{Symbol, GPUDiseaseArrays}()

    for (name, disease) in sim.diseases
        if disease isa Starsim.SIR
            gpu_diseases[name] = _sir_to_gpu(disease)
        elseif disease isa Starsim.SIS
            gpu_diseases[name] = _sis_to_gpu(disease)
        else
            @warn "GPU acceleration not yet implemented for $(typeof(disease)); skipping :$name"
        end
    end

    return GPUSim(sim, gpu_people, gpu_diseases)
end

# ============================================================================
# GPU → CPU conversion
# ============================================================================

function _people_from_gpu!(people::Starsim.People, gpu::GPUPeopleArrays)
    n = min(length(people.alive.raw), gpu.n)
    cpu_alive  = Array(gpu.alive)
    cpu_age    = Array(gpu.age)
    cpu_female = Array(gpu.female)
    @inbounds for i in 1:n
        people.alive.raw[i]  = cpu_alive[i] != 0x00
        people.age.raw[i]    = Float64(cpu_age[i])
        people.female.raw[i] = cpu_female[i] != 0x00
    end
    return people
end

function _sir_from_gpu!(d::Starsim.SIR, gpu::GPUDiseaseArrays)
    n = min(length(d.infection.infected.raw), gpu.n)
    cpu_sus       = Array(gpu.susceptible)
    cpu_inf       = Array(gpu.infected)
    cpu_rec       = Array(gpu.recovered)
    cpu_ti_inf    = Array(gpu.ti_infected)
    cpu_ti_rec    = Array(gpu.ti_recovered)
    cpu_rel_sus   = Array(gpu.rel_sus)
    cpu_rel_trans = Array(gpu.rel_trans)
    @inbounds for i in 1:n
        d.infection.susceptible.raw[i] = cpu_sus[i] != 0x00
        d.infection.infected.raw[i]    = cpu_inf[i] != 0x00
        d.recovered.raw[i]             = cpu_rec[i] != 0x00
        d.infection.ti_infected.raw[i] = Float64(cpu_ti_inf[i])
        d.ti_recovered.raw[i]          = Float64(cpu_ti_rec[i])
        d.infection.rel_sus.raw[i]     = Float64(cpu_rel_sus[i])
        d.infection.rel_trans.raw[i]   = Float64(cpu_rel_trans[i])
    end
    return d
end

function _sis_from_gpu!(d::Starsim.SIS, gpu::GPUDiseaseArrays)
    n = min(length(d.infection.infected.raw), gpu.n)
    cpu_sus       = Array(gpu.susceptible)
    cpu_inf       = Array(gpu.infected)
    cpu_ti_inf    = Array(gpu.ti_infected)
    cpu_rel_sus   = Array(gpu.rel_sus)
    cpu_rel_trans = Array(gpu.rel_trans)
    @inbounds for i in 1:n
        d.infection.susceptible.raw[i] = cpu_sus[i] != 0x00
        d.infection.infected.raw[i]    = cpu_inf[i] != 0x00
        d.infection.ti_infected.raw[i] = Float64(cpu_ti_inf[i])
        d.infection.rel_sus.raw[i]     = Float64(cpu_rel_sus[i])
        d.infection.rel_trans.raw[i]   = Float64(cpu_rel_trans[i])
    end
    return d
end

"""
    to_cpu(gsim::GPUSim) → Sim

Copy GPU state arrays back to the CPU simulation. Modifies the original
`Sim` in-place and returns it.

Float32 → Float64 and UInt8 → Bool conversions are applied automatically.
Call this after GPU steps to read back results or before CPU-only operations
(death processing, results update, etc.).
"""
function Starsim.to_cpu(gsim::GPUSim)
    _people_from_gpu!(gsim.sim.people, gsim.people)
    for (name, gpu_dis) in gsim.diseases
        disease = gsim.sim.diseases[name]
        if disease isa Starsim.SIR
            _sir_from_gpu!(disease, gpu_dis)
        elseif disease isa Starsim.SIS
            _sis_from_gpu!(disease, gpu_dis)
        end
    end
    return gsim.sim
end

# No-op when the sim is already on CPU
Starsim.to_cpu(sim::Starsim.Sim) = sim

# ============================================================================
# Metal kernels — transmission
# ============================================================================

# Each thread processes one directed edge (src → trg). If the source is
# infected and the target is susceptible, the kernel evaluates the per-edge
# transmission probability and flags new infections.
#
# Thread safety: multiple threads may write `0x01` to the same target's
# `new_infected` slot. This is a benign race — all threads write the same
# idempotent value, so no atomics are needed.

function _transmission_kernel!(
    new_infected, susceptible, infected, rel_trans, rel_sus,
    p1, p2, edge_beta, beta_dt, rng_vals, n_edges,
)
    i = thread_position_in_grid_1d()
    i > n_edges && return
    src = p1[i]
    trg = p2[i]
    if infected[src] == 0x01 && susceptible[trg] == 0x01
        prob = beta_dt * rel_trans[src] * rel_sus[trg] * edge_beta[i]
        if rng_vals[i] < prob
            new_infected[trg] = 0x01
        end
    end
    return
end

# ============================================================================
# Metal kernels — apply new infections
# ============================================================================

# After the transmission kernel marks `new_infected`, these kernels flip the
# actual state arrays. Separate variants for SIR (sets ti_recovered) and SIS
# (no recovery time tracking) avoid branching on `nothing` inside a kernel.

function _apply_infections_sir_kernel!(
    susceptible, infected, ti_infected, ti_recovered,
    new_infected, current_ti, dur_inf_ts, jitter, n,
)
    i = thread_position_in_grid_1d()
    i > n && return
    if new_infected[i] == 0x01
        susceptible[i]  = 0x00
        infected[i]     = 0x01
        ti_infected[i]  = current_ti
        ti_recovered[i] = current_ti + dur_inf_ts + jitter[i]
    end
    return
end

function _apply_infections_sis_kernel!(
    susceptible, infected, ti_infected,
    new_infected, current_ti, n,
)
    i = thread_position_in_grid_1d()
    i > n && return
    if new_infected[i] == 0x01
        susceptible[i] = 0x00
        infected[i]    = 0x01
        ti_infected[i] = current_ti
    end
    return
end

# ============================================================================
# Metal kernels — state transitions (recovery / waning)
# ============================================================================

# SIR recovery: infected → recovered when ti_recovered <= current_ti
function _sir_recovery_kernel!(infected, recovered, ti_recovered, current_ti, n)
    i = thread_position_in_grid_1d()
    i > n && return
    if infected[i] == 0x01 && ti_recovered[i] <= current_ti
        infected[i]  = 0x00
        recovered[i] = 0x01
    end
    return
end

# SIS recovery: infected → susceptible when (current_ti - ti_infected) >= dur
function _sis_recovery_kernel!(infected, susceptible, ti_infected, current_ti, dur_inf_ts, n)
    i = thread_position_in_grid_1d()
    i > n && return
    if infected[i] == 0x01 && (current_ti - ti_infected[i]) >= dur_inf_ts
        infected[i]    = 0x00
        susceptible[i] = 0x01
    end
    return
end

# Waning immunity: recovered → susceptible when (current_ti - ti_recovered) >= wane_dur
# Useful for SIRS-like extensions built on top of SIR.
function _waning_kernel!(recovered, susceptible, ti_recovered, current_ti, wane_dur, n)
    i = thread_position_in_grid_1d()
    i > n && return
    if recovered[i] == 0x01 && (current_ti - ti_recovered[i]) >= wane_dur
        recovered[i]   = 0x00
        susceptible[i] = 0x01
    end
    return
end

# ============================================================================
# High-level GPU step functions
# ============================================================================

const METAL_THREADS = 256  # Threads per threadgroup (Apple GPU wavefront = 32)

"""
    gpu_step_state!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Run GPU-accelerated state transitions (recovery) for one disease.

- **SIR**: infected → recovered when `ti_recovered ≤ current_ti`
- **SIS**: infected → susceptible when infection duration has elapsed

This replaces the CPU `step_state!` call. Disease-induced death (`p_death`)
is **not** handled on GPU; use `to_cpu` and the CPU code path for that.
"""
function gpu_step_state!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gpu_dis = gsim.diseases[disease_name]
    disease = gsim.sim.diseases[disease_name]
    n = Int32(gpu_dis.n)
    groups = cld(n, METAL_THREADS)
    ti_f = Float32(current_ti)

    if disease isa Starsim.SIR && gpu_dis.recovered !== nothing
        @metal threads=METAL_THREADS groups=groups _sir_recovery_kernel!(
            gpu_dis.infected, gpu_dis.recovered, gpu_dis.ti_recovered,
            ti_f, n,
        )
    elseif disease isa Starsim.SIS
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
        @metal threads=METAL_THREADS groups=groups _sis_recovery_kernel!(
            gpu_dis.infected, gpu_dis.susceptible, gpu_dis.ti_infected,
            ti_f, dur_inf_ts, n,
        )
    end

    Metal.synchronize()
    return gsim
end

"""
    gpu_transmit!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Run GPU-accelerated transmission for one disease across all networks.

# Algorithm
1. Read edge arrays (`p1`, `p2`, `beta`) from the CPU sim (already updated
   by the CPU network `step!`) and copy them to GPU as `Int32`/`Float32`.
2. Generate uniform random numbers on CPU and upload to GPU.
3. Launch the transmission kernel — each thread evaluates one directed edge.
4. For bidirectional networks, launch a second pass with swapped endpoints.
5. Apply new infections via a second kernel that flips state arrays.

# Notes
- Random numbers are generated on CPU (Metal has no built-in GPU RNG).
- Infection-source logging is skipped (no dynamic allocation on GPU).
- Recovery times for SIR are set with pre-computed jitter during the
  apply-infections kernel.
"""
function gpu_transmit!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gpu_dis = gsim.diseases[disease_name]
    disease = gsim.sim.diseases[disease_name]
    dd = Starsim.disease_data(disease)
    n_agents = Int32(gpu_dis.n)
    ti_f = Float32(current_ti)

    # Accumulate new infections across all networks
    new_infected = Metal.zeros(UInt8, gpu_dis.n)

    for (net_name, net) in gsim.sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue
        beta_dt_f32 = Float32(beta_dt)

        n_edges = length(edges)
        n_edges_i32 = Int32(n_edges)
        groups_e = cld(n_edges, METAL_THREADS)

        # Upload edge arrays to GPU (edges are regenerated each timestep)
        gpu_p1 = MtlVector{Int32}(Int32.(edges.p1))
        gpu_p2 = MtlVector{Int32}(Int32.(edges.p2))
        gpu_eb = MtlVector{Float32}(Float32.(edges.beta))

        # Pre-generate random numbers on CPU, upload to GPU
        rng_fwd = MtlVector{Float32}(rand(Float32, n_edges))

        # Forward direction: p1[i] → p2[i]
        @metal threads=METAL_THREADS groups=groups_e _transmission_kernel!(
            new_infected, gpu_dis.susceptible, gpu_dis.infected,
            gpu_dis.rel_trans, gpu_dis.rel_sus,
            gpu_p1, gpu_p2, gpu_eb, beta_dt_f32, rng_fwd, n_edges_i32,
        )

        # Reverse direction for bidirectional networks: p2[i] → p1[i]
        if Starsim.network_data(net).bidirectional
            rng_rev = MtlVector{Float32}(rand(Float32, n_edges))
            @metal threads=METAL_THREADS groups=groups_e _transmission_kernel!(
                new_infected, gpu_dis.susceptible, gpu_dis.infected,
                gpu_dis.rel_trans, gpu_dis.rel_sus,
                gpu_p2, gpu_p1, gpu_eb, beta_dt_f32, rng_rev, n_edges_i32,
            )
        end
    end

    Metal.synchronize()

    # Apply accumulated new infections to state arrays
    groups_a = cld(n_agents, METAL_THREADS)

    if disease isa Starsim.SIR && gpu_dis.ti_recovered !== nothing
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
        # Per-agent jitter for recovery time (uniform in [0, 0.5 * dur_inf_ts])
        jitter = MtlVector{Float32}(rand(Float32, gpu_dis.n) .* (dur_inf_ts * 0.5f0))
        @metal threads=METAL_THREADS groups=groups_a _apply_infections_sir_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            gpu_dis.ti_recovered, new_infected, ti_f, dur_inf_ts, jitter, n_agents,
        )
    elseif disease isa Starsim.SIS
        @metal threads=METAL_THREADS groups=groups_a _apply_infections_sis_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            new_infected, ti_f, n_agents,
        )
    end

    Metal.synchronize()
    return gsim
end

"""
    gpu_step!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Combined GPU disease step: state transitions (recovery) followed by
transmission. Equivalent to calling `step_state!` then `step!` for one
disease, but both execute on the Metal GPU.

Network `step!` must still run on CPU before each `gpu_step!` call, since
edge arrays are dynamically regenerated each timestep.

# Example
```julia
sim = Sim(n_agents=100_000, diseases=SIR(beta=0.05), networks=RandomNet())
init!(sim)
gsim = to_gpu(sim)

for ti in 1:sim.t.npts
    # 1. Network rewiring on CPU
    for (_, net) in sim.networks
        step!(net, sim)
    end

    # 2. Disease dynamics on GPU
    for name in keys(gsim.diseases)
        gpu_step!(gsim, name; current_ti=ti)
    end
end

# 3. Download final state
sim = to_cpu(gsim)
```

# What is NOT handled on GPU
- Infection-source logging (requires dynamic `push!`)
- Disease-induced death (`p_death > 0`; use CPU fallback)
- CRN (common random numbers) pairwise draws
- Results accumulation (`update_results!`)
"""
function gpu_step!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gpu_step_state!(gsim, disease_name; current_ti=current_ti)
    gpu_transmit!(gsim, disease_name; current_ti=current_ti)
    return gsim
end

"""
    gpu_waning!(gsim::GPUSim, disease_name::Symbol; current_ti::Int, wane_dur::Float32)

GPU kernel for waning immunity (recovered → susceptible). Useful for
SIRS-like models built on top of `SIR`.

Agents in the `recovered` state whose recovery time is at least `wane_dur`
timesteps in the past transition back to `susceptible`.
"""
function gpu_waning!(gsim::GPUSim, disease_name::Symbol; current_ti::Int, wane_dur::Float32)
    gpu_dis = gsim.diseases[disease_name]
    gpu_dis.recovered === nothing && return gsim
    gpu_dis.ti_recovered === nothing && return gsim

    n = Int32(gpu_dis.n)
    groups = cld(n, METAL_THREADS)
    ti_f = Float32(current_ti)

    @metal threads=METAL_THREADS groups=groups _waning_kernel!(
        gpu_dis.recovered, gpu_dis.susceptible, gpu_dis.ti_recovered,
        ti_f, wane_dur, n,
    )
    Metal.synchronize()
    return gsim
end

# ============================================================================
# Re-upload utility
# ============================================================================

"""
    sync_to_gpu!(gsim::GPUSim)

Re-upload CPU state arrays to the existing GPU buffers. Call this after
CPU-side operations (births, deaths, interventions) have modified the
CPU sim's state arrays and you want to continue GPU stepping.

Uses `copyto!` to overwrite existing MtlVector buffers without reallocating.
"""
function sync_to_gpu!(gsim::GPUSim)
    people = gsim.sim.people
    n_p = min(length(people.alive.raw), gsim.people.n)
    copyto!(gsim.people.alive,  UInt8.(people.alive.raw[1:n_p]))
    copyto!(gsim.people.age,    Float32.(people.age.raw[1:n_p]))
    copyto!(gsim.people.female, UInt8.(people.female.raw[1:n_p]))

    for (name, gpu_dis) in gsim.diseases
        disease = gsim.sim.diseases[name]
        if disease isa Starsim.SIR
            _sync_infection_to_gpu!(disease.infection, gpu_dis)
            n = min(length(disease.recovered.raw), gpu_dis.n)
            copyto!(gpu_dis.recovered,   UInt8.(disease.recovered.raw[1:n]))
            copyto!(gpu_dis.ti_recovered, Float32.(disease.ti_recovered.raw[1:n]))
        elseif disease isa Starsim.SIS
            _sync_infection_to_gpu!(disease.infection, gpu_dis)
        end
    end
    return gsim
end

function _sync_infection_to_gpu!(inf::Starsim.InfectionData, gpu::GPUDiseaseArrays)
    n = min(length(inf.infected.raw), gpu.n)
    copyto!(gpu.susceptible, UInt8.(inf.susceptible.raw[1:n]))
    copyto!(gpu.infected,    UInt8.(inf.infected.raw[1:n]))
    copyto!(gpu.ti_infected, Float32.(inf.ti_infected.raw[1:n]))
    copyto!(gpu.rel_sus,     Float32.(inf.rel_sus.raw[1:n]))
    copyto!(gpu.rel_trans,   Float32.(inf.rel_trans.raw[1:n]))
    return
end

# ============================================================================
# Exports
# ============================================================================

export GPUSim, GPUPeopleArrays, GPUDiseaseArrays
export gpu_step!, gpu_step_state!, gpu_transmit!, gpu_waning!, sync_to_gpu!

end # module
