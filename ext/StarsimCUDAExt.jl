"""
    StarsimCUDAExt

Extension providing NVIDIA GPU support for Starsim.jl via CUDA.jl.

Offloads compute-heavy agent-state operations (transmission, recovery) to the
GPU while keeping structurally dynamic operations (network rewiring, births,
deaths) on the CPU.

# Architecture — what runs where

**GPU (CuVector, Float32/UInt8)**
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
- All GPU floats are `Float32` for performance consistency
- GPU booleans are stored as `UInt8` (0x00/0x01)
- Edge indices are `Int32` on GPU (sufficient for up to 2 billion agents)

# Usage
```julia
using Starsim, CUDA

sim = Sim(n_agents=100_000, diseases=SIR(beta=0.05), networks=RandomNet())
init!(sim)
gsim = to_gpu(sim)

for ti in 1:sim.t.npts
    # Step 5: State transitions (recovery) on GPU
    gpu_step_state!(gsim, :sir; current_ti=ti)

    # Step 7: Network rewiring on CPU
    for (_, net) in sim.networks
        step!(net, sim)
    end

    # Step 9: Transmission on GPU
    gpu_transmit!(gsim, :sir; current_ti=ti)
end

# Download final state
sim = to_cpu(gsim)
```

# Limitations
- Infection-source logging is skipped on GPU (no dynamic allocation in kernels)
- Disease-induced death (`p_death > 0`) is not handled in GPU kernels; use
  CPU fallback or call `to_cpu` before death processing
- Agent births/deaths between GPU steps require `sync_to_gpu!` to re-upload
"""
module StarsimCUDAExt

using Starsim
using CUDA
using Distributions

# ============================================================================
# Constants
# ============================================================================

const CUDA_THREADS = 256
const CRN_DT_JUMP  = UInt32(1000)   # matches Python starsim: dt_jump_size = 1000

# ============================================================================
# GPU array container types
# ============================================================================

"""
    GPUPeopleArrays

GPU-resident mirror of People state arrays.

Only the arrays needed for disease dynamics and CRN are uploaded.
Index states (`uid`) and management fields (`ti_dead`, `ti_removed`,
`scale`, `parent`) stay on CPU because they are only used in demographic
operations. The `slot` array is uploaded for CRN-safe per-agent RNG indexing.
"""
struct GPUPeopleArrays
    alive::CuVector{UInt8}
    age::CuVector{Float32}
    female::CuVector{UInt8}
    slot::CuVector{Int32}    # CRN slot indices (1-based)
    n::Int
end

"""
    GPUDiseaseArrays

GPU-resident mirror of disease state arrays for one `AbstractInfection`.

Supports SIR (with `recovered`/`ti_recovered`), SIS (those fields are
`nothing`), and SEIR (adds `exposed`/`ti_exposed`). All Float64 values
are stored as Float32; Bool as UInt8.
"""
struct GPUDiseaseArrays
    susceptible::CuVector{UInt8}
    infected::CuVector{UInt8}
    recovered::Union{CuVector{UInt8}, Nothing}
    ti_infected::CuVector{Float32}
    ti_recovered::Union{CuVector{Float32}, Nothing}
    rel_sus::CuVector{Float32}
    rel_trans::CuVector{Float32}
    n::Int
    # SEIR-specific (nothing for SIR/SIS)
    exposed::Union{CuVector{UInt8}, Nothing}
    ti_exposed::Union{CuVector{Float32}, Nothing}
end

"""
    GPUSim

Wrapper pairing a CPU `Sim` with GPU-mirrored state arrays. Created by
[`to_gpu`](@ref), converted back with [`to_cpu`](@ref).

The CPU `Sim` remains the source of truth for networks, demographics,
results, and all non-disease module state. The `GPUDiseaseArrays` hold
the authoritative disease state while on GPU.

Pre-allocated GPU edge buffers (`edge_p1`, `edge_p2`, etc.) are reused
each timestep via `copyto!` to avoid per-step CuVector allocation.
"""
mutable struct GPUSim
    sim::Starsim.Sim
    people::GPUPeopleArrays
    diseases::Dict{Symbol, GPUDiseaseArrays}
    # Pre-allocated edge buffers (sized to max expected edges)
    edge_p1::CuVector{Int32}
    edge_p2::CuVector{Int32}
    edge_beta::CuVector{Float32}
    rng_buf::CuVector{Float32}
    new_infected::CuVector{UInt8}
    jitter_buf::CuVector{Float32}
    snap_infected::CuVector{UInt8}      # Snapshot buffer for synchronous transmission
    snap_susceptible::CuVector{UInt8}   # Snapshot buffer for synchronous transmission
    edge_capacity::Int
    # Cached edges for static network mode
    cached_edges::Bool
    cached_n_edges::Int
    cached_bidirectional::Bool
    cached_beta_dt::Dict{Symbol, Float32}
    # GPU-side RNG state (xorshift32 per-thread seeds)
    rng_seeds::CuVector{UInt32}
    # CRN support: base seeds (per-agent), reset each timestep
    crn_mode::Bool
    rng_seeds_base::CuVector{UInt32}   # per-agent base seeds (deterministic from sim seed)
    rng_seeds_agent::CuVector{UInt32}  # per-agent current seeds (reset each timestep)
end

function Base.show(io::IO, g::GPUSim)
    nd = length(g.diseases)
    names = join(keys(g.diseases), ", ")
    print(io, "GPUSim(n_agents=$(g.people.n), diseases=[$names], edge_cap=$(g.edge_capacity))")
end

# ============================================================================
# CPU → GPU conversion
# ============================================================================

function _people_to_gpu(people::Starsim.People)
    n = people.alive.len_tot
    GPUPeopleArrays(
        CuVector{UInt8}(UInt8.(people.alive.raw)),
        CuVector{Float32}(Float32.(people.age.raw)),
        CuVector{UInt8}(UInt8.(people.female.raw)),
        CuVector{Int32}(Int32.(people.slot.raw)),
        n,
    )
end

function _sir_to_gpu(d::Starsim.SIR)
    n = d.infection.infected.len_tot
    GPUDiseaseArrays(
        CuVector{UInt8}(UInt8.(d.infection.susceptible.raw)),
        CuVector{UInt8}(UInt8.(d.infection.infected.raw)),
        CuVector{UInt8}(UInt8.(d.recovered.raw)),
        CuVector{Float32}(Float32.(d.infection.ti_infected.raw)),
        CuVector{Float32}(Float32.(d.ti_recovered.raw)),
        CuVector{Float32}(Float32.(d.infection.rel_sus.raw)),
        CuVector{Float32}(Float32.(d.infection.rel_trans.raw)),
        n,
        nothing,  # exposed (SIR has none)
        nothing,  # ti_exposed
    )
end

function _sis_to_gpu(d::Starsim.SIS)
    n = d.infection.infected.len_tot
    GPUDiseaseArrays(
        CuVector{UInt8}(UInt8.(d.infection.susceptible.raw)),
        CuVector{UInt8}(UInt8.(d.infection.infected.raw)),
        nothing,  # no recovered state (SIS cycles back to susceptible)
        CuVector{Float32}(Float32.(d.infection.ti_infected.raw)),
        CuVector{Float32}(Float32.(d.ti_recovered.raw)),  # SIS uses ti_recovered for timing
        CuVector{Float32}(Float32.(d.infection.rel_sus.raw)),
        CuVector{Float32}(Float32.(d.infection.rel_trans.raw)),
        n,
        nothing,  # exposed (SIS has none)
        nothing,  # ti_exposed
    )
end

function _seir_to_gpu(d::Starsim.SEIR)
    n = d.infection.infected.len_tot
    GPUDiseaseArrays(
        CuVector{UInt8}(UInt8.(d.infection.susceptible.raw)),
        CuVector{UInt8}(UInt8.(d.infection.infected.raw)),
        CuVector{UInt8}(UInt8.(d.recovered.raw)),
        CuVector{Float32}(Float32.(d.infection.ti_infected.raw)),
        CuVector{Float32}(Float32.(d.ti_recovered.raw)),
        CuVector{Float32}(Float32.(d.infection.rel_sus.raw)),
        CuVector{Float32}(Float32.(d.infection.rel_trans.raw)),
        n,
        # SEIR-specific fields
        CuVector{UInt8}(UInt8.(d.exposed.raw)),
        CuVector{Float32}(Float32.(d.ti_exposed.raw)),
    )
end

"""
    to_gpu(sim::Starsim.Sim) → GPUSim

Copy a CPU simulation's state arrays to CUDA GPU memory.

The simulation must be initialized (`init!(sim)`) before calling this.
Returns a [`GPUSim`](@ref) that pairs the original `Sim` with GPU arrays.
Network edge lists are NOT copied (they are rebuilt each timestep on CPU).

Currently supports `SIR`, `SIS`, and `SEIR` diseases. Unsupported disease types
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
        elseif disease isa Starsim.SEIR
            gpu_diseases[name] = _seir_to_gpu(disease)
        else
            @warn "GPU acceleration not yet implemented for $(typeof(disease)); skipping :$name"
        end
    end

    # Estimate edge buffer size: n_agents * max_contacts * 2 (safety margin)
    n = gpu_people.n
    max_contacts = 20  # conservative default
    for (_, net) in sim.networks
        nd = Starsim.network_data(net)
        if hasproperty(nd.mod, :n_contacts)
            max_contacts = max(max_contacts, nd.mod.n_contacts)
        end
    end
    edge_cap = n * max_contacts
    edge_p1 = CuVector{Int32}(zeros(Int32, edge_cap))
    edge_p2 = CuVector{Int32}(zeros(Int32, edge_cap))
    edge_beta = CuVector{Float32}(zeros(Float32, edge_cap))
    rng_buf = CuVector{Float32}(zeros(Float32, edge_cap))
    new_infected = CUDA.zeros(UInt8, n)
    jitter_buf = CuVector{Float32}(zeros(Float32, n))
    snap_infected = CUDA.zeros(UInt8, n)
    snap_susceptible = CUDA.zeros(UInt8, n)

    # Deterministic seeding: derive per-edge seeds from sim's rand_seed
    base_seed = UInt32(sim.pars.rand_seed & 0xffffffff)
    rng_seed_count = max(edge_cap, n)
    edge_seeds = Vector{UInt32}(undef, rng_seed_count)
    for i in 1:rng_seed_count
        # Hash base_seed with index to get deterministic unique per-thread seed
        s = base_seed ⊻ UInt32((i * 2654435761) & 0xffffffff)  # Knuth multiplicative hash
        s = s == UInt32(0) ? UInt32(1) : s  # xorshift32 cannot have seed=0
        edge_seeds[i] = s
    end
    rng_seeds = CuVector{UInt32}(edge_seeds)

    # CRN: per-agent base seeds (deterministic from sim seed + agent slot)
    use_crn = Starsim.crn_enabled()
    agent_base_seeds = Vector{UInt32}(undef, n)
    for i in 1:n
        slot_i = Int32(sim.people.slot.raw[i])
        s = base_seed ⊻ UInt32((slot_i * 2654435761) & 0xffffffff)
        s = s == UInt32(0) ? UInt32(1) : s
        agent_base_seeds[i] = s
    end
    rng_seeds_base = CuVector{UInt32}(agent_base_seeds)
    rng_seeds_agent = CuVector{UInt32}(copy(agent_base_seeds))

    return GPUSim(sim, gpu_people, gpu_diseases,
                  edge_p1, edge_p2, edge_beta, rng_buf,
                  new_infected, jitter_buf, snap_infected, snap_susceptible, edge_cap,
                  false, 0, false, Dict{Symbol, Float32}(),
                  rng_seeds,
                  use_crn, rng_seeds_base, rng_seeds_agent)
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
    cpu_ti_rec    = gpu.ti_recovered !== nothing ? Array(gpu.ti_recovered) : nothing
    @inbounds for i in 1:n
        d.infection.susceptible.raw[i] = cpu_sus[i] != 0x00
        d.infection.infected.raw[i]    = cpu_inf[i] != 0x00
        d.infection.ti_infected.raw[i] = Float64(cpu_ti_inf[i])
        d.infection.rel_sus.raw[i]     = Float64(cpu_rel_sus[i])
        d.infection.rel_trans.raw[i]   = Float64(cpu_rel_trans[i])
        if cpu_ti_rec !== nothing
            d.ti_recovered.raw[i] = Float64(cpu_ti_rec[i])
        end
    end
    return d
end

function _seir_from_gpu!(d::Starsim.SEIR, gpu::GPUDiseaseArrays)
    n = min(length(d.infection.infected.raw), gpu.n)
    cpu_sus       = Array(gpu.susceptible)
    cpu_inf       = Array(gpu.infected)
    cpu_rec       = Array(gpu.recovered)
    cpu_exp       = Array(gpu.exposed)
    cpu_ti_inf    = Array(gpu.ti_infected)
    cpu_ti_rec    = Array(gpu.ti_recovered)
    cpu_ti_exp    = Array(gpu.ti_exposed)
    cpu_rel_sus   = Array(gpu.rel_sus)
    cpu_rel_trans = Array(gpu.rel_trans)
    @inbounds for i in 1:n
        d.infection.susceptible.raw[i] = cpu_sus[i] != 0x00
        d.infection.infected.raw[i]    = cpu_inf[i] != 0x00
        d.recovered.raw[i]             = cpu_rec[i] != 0x00
        d.exposed.raw[i]               = cpu_exp[i] != 0x00
        d.infection.ti_infected.raw[i] = Float64(cpu_ti_inf[i])
        d.ti_recovered.raw[i]          = Float64(cpu_ti_rec[i])
        d.ti_exposed.raw[i]            = Float64(cpu_ti_exp[i])
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
        elseif disease isa Starsim.SEIR
            _seir_from_gpu!(disease, gpu_dis)
        end
    end
    return gsim.sim
end

# No-op when the sim is already on CPU
Starsim.to_cpu(sim::Starsim.Sim) = sim

# ============================================================================
# CUDA kernels — transmission
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
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
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
# CUDA kernels — apply new infections
# ============================================================================

# After the transmission kernel marks `new_infected`, these kernels flip the
# actual state arrays. Separate variants for SIR (sets ti_recovered) and SIS
# (no recovery time tracking) avoid branching on `nothing` inside a kernel.

function _apply_infections_sir_kernel!(
    susceptible, infected, ti_infected, ti_recovered,
    new_infected, current_ti, exp_dur, n,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if new_infected[i] == 0x01
        susceptible[i]  = 0x00
        infected[i]     = 0x01
        ti_infected[i]  = current_ti
        ti_recovered[i] = current_ti + exp_dur[i]
    end
    return
end

function _apply_infections_sis_kernel!(
    susceptible, infected, ti_infected, ti_recovered,
    new_infected, current_ti, exp_dur, n,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if new_infected[i] == 0x01
        susceptible[i]  = 0x00
        infected[i]     = 0x01
        ti_infected[i]  = current_ti
        ti_recovered[i] = current_ti + exp_dur[i]
    end
    return
end

# ============================================================================
# CUDA kernels — state transitions (recovery / waning)
# ============================================================================

# SIR recovery: infected → recovered when ti_recovered <= current_ti
function _sir_recovery_kernel!(infected, recovered, ti_recovered, current_ti, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if infected[i] == 0x01 && ti_recovered[i] <= current_ti
        infected[i]  = 0x00
        recovered[i] = 0x01
    end
    return
end

# SIS recovery: infected → susceptible when ti_recovered <= current_ti
# Uses per-agent ti_recovered (lognormal durations), matching CPU SIS exactly.
function _sis_recovery_ti_kernel!(infected, susceptible, ti_recovered, current_ti, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if infected[i] == 0x01 && ti_recovered[i] <= current_ti
        infected[i]    = 0x00
        susceptible[i] = 0x01
    end
    return
end

# SIS recovery: infected → susceptible when (current_ti - ti_infected) >= dur
function _sis_recovery_kernel!(infected, susceptible, ti_infected, current_ti, dur_inf_ts, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
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
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if recovered[i] == 0x01 && (current_ti - ti_recovered[i]) >= wane_dur
        recovered[i]   = 0x00
        susceptible[i] = 0x01
    end
    return
end

# SEIR: exposed → infected when latent period has elapsed, with recovery time set
function _seir_exposure_kernel!(exposed, infected, ti_exposed, ti_infected, ti_recovered, current_ti, dur_exp_ts, recovery_dur, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if exposed[i] == 0x01 && (current_ti - ti_exposed[i]) >= dur_exp_ts
        exposed[i]      = 0x00
        infected[i]     = 0x01
        ti_infected[i]  = current_ti
        ti_recovered[i] = current_ti + recovery_dur[i]
    end
    return
end

# SEIR apply new exposures: susceptible → exposed
function _apply_exposures_seir_kernel!(
    susceptible, exposed, ti_exposed,
    new_infected, current_ti, n,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if new_infected[i] == 0x01
        susceptible[i] = 0x00
        exposed[i]     = 0x01
        ti_exposed[i]  = current_ti
    end
    return
end

# ============================================================================
# Fused kernels with GPU-side xorshift32 RNG
# ============================================================================

# Xorshift32 PRNG — fast, stateful, good enough for Monte Carlo transmission.
# Each thread maintains its own seed in `rng_seeds[i]`, mutated in-place.
# Returns a Float32 in [0, 1).
@inline function _xorshift32(state::UInt32)
    state ⊻= state << UInt32(13)
    state ⊻= state >> UInt32(17)
    state ⊻= state << UInt32(5)
    return state
end

@inline function _u32_to_f32(state::UInt32)
    # Map UInt32 → Float32 in [0, 1): divide by 2^32
    return Float32(state) * Float32(2.3283064e-10)  # 1/2^32
end

# ============================================================================
# CRN kernels — per-agent seeds reset and pairwise XOR combining
# ============================================================================

# Reset per-agent seeds to a deterministic state for timestep `ti`.
# Matches Python starsim: seed = base_seed + ti * 1000
function _crn_reset_seeds_kernel!(rng_seeds_agent, rng_seeds_base, ti_jump, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    base = rng_seeds_base[i]
    s = base + ti_jump
    # Ensure non-zero (xorshift32 fixpoint)
    if s == UInt32(0)
        s = UInt32(1)
    end
    rng_seeds_agent[i] = s
    return
end

# Pairwise XOR combining: for edge (src, trg), draw from each agent's
# independent stream and XOR-combine to produce a single Float32.
# Mirrors the CPU MultiRandom.combine_rvs logic.
@inline function _crn_pairwise_rng(seed_src::UInt32, seed_trg::UInt32)
    # Advance both streams
    s1 = _xorshift32(seed_src)
    s2 = _xorshift32(seed_trg)
    # XOR combine (product bits ⊻ difference bits) — GPU Float32 variant
    f1 = _u32_to_f32(s1)
    f2 = _u32_to_f32(s2)
    prod_bits = reinterpret(UInt32, f1 * f2)
    diff_bits = reinterpret(UInt32, f1 - f2)
    combined = prod_bits ⊻ diff_bits
    return _u32_to_f32(combined), s1, s2
end

# ============================================================================
# CRN-aware fused transmission kernels
# ============================================================================

# CRN variant of fused SIR transmission: uses per-agent slot-indexed seeds
# with pairwise XOR combining. Reads from snapshot buffers, writes to new_infected.
function _crn_fused_transmit_sir_kernel!(
    new_infected, snap_infected, snap_susceptible, rel_trans, rel_sus,
    p1, p2, edge_beta, beta_dt,
    rng_seeds_agent, slot,
    n_edges,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_edges && return

    src = p1[i]
    trg = p2[i]

    if snap_infected[src] == 0x01 && snap_susceptible[trg] == 0x01
        prob = beta_dt * rel_trans[src] * rel_sus[trg] * edge_beta[i]

        # CRN pairwise random number from src and trg agent seeds
        seed_src = rng_seeds_agent[src]
        seed_trg = rng_seeds_agent[trg]
        rng_val, new_s1, new_s2 = _crn_pairwise_rng(seed_src, seed_trg)

        # Write back advanced seeds (benign race if agent appears in multiple edges)
        rng_seeds_agent[src] = new_s1
        rng_seeds_agent[trg] = new_s2

        if rng_val < prob
            new_infected[trg] = 0x01
        end
    end
    return
end

# CRN variant of fused SIS transmission — reads snapshots, writes new_infected
function _crn_fused_transmit_sis_kernel!(
    new_infected, snap_infected, snap_susceptible, rel_trans, rel_sus,
    p1, p2, edge_beta, beta_dt,
    rng_seeds_agent, slot,
    n_edges,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_edges && return

    src = p1[i]
    trg = p2[i]

    if snap_infected[src] == 0x01 && snap_susceptible[trg] == 0x01
        prob = beta_dt * rel_trans[src] * rel_sus[trg] * edge_beta[i]

        seed_src = rng_seeds_agent[src]
        seed_trg = rng_seeds_agent[trg]
        rng_val, new_s1, new_s2 = _crn_pairwise_rng(seed_src, seed_trg)
        rng_seeds_agent[src] = new_s1
        rng_seeds_agent[trg] = new_s2

        if rng_val < prob
            new_infected[trg] = 0x01
        end
    end
    return
end

# ============================================================================
# Non-CRN fused transmission kernels (original — per-edge seeds)
# ============================================================================

# Fused transmission kernel: generates its own random numbers on GPU,
# evaluates transmission against snapshot state, and flags new infections.
# Eliminates: CPU rand generation, CPU→GPU rng upload.
function _fused_transmit_sir_kernel!(
    new_infected, snap_infected, snap_susceptible, rel_trans, rel_sus,
    p1, p2, edge_beta, beta_dt, rng_seeds, n_edges,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_edges && return

    src = p1[i]
    trg = p2[i]

    if snap_infected[src] == 0x01 && snap_susceptible[trg] == 0x01
        prob = beta_dt * rel_trans[src] * rel_sus[trg] * edge_beta[i]

        # GPU-side random number
        seed = rng_seeds[i]
        seed = _xorshift32(seed)
        rng_val = _u32_to_f32(seed)

        # Write back mutated seed
        rng_seeds[i] = seed

        if rng_val < prob
            new_infected[trg] = 0x01
        end
    end
    return
end

# SIS variant — reads snapshots, writes new_infected
function _fused_transmit_sis_kernel!(
    new_infected, snap_infected, snap_susceptible, rel_trans, rel_sus,
    p1, p2, edge_beta, beta_dt, rng_seeds, n_edges,
)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n_edges && return

    src = p1[i]
    trg = p2[i]

    if snap_infected[src] == 0x01 && snap_susceptible[trg] == 0x01
        prob = beta_dt * rel_trans[src] * rel_sus[trg] * edge_beta[i]

        seed = rng_seeds[i]
        seed = _xorshift32(seed)
        rng_val = _u32_to_f32(seed)
        rng_seeds[i] = seed

        if rng_val < prob
            new_infected[trg] = 0x01
        end
    end
    return
end

# Fused recovery + apply kernel for SIR: per-agent, checks recovery AND
# could do other per-agent work in a single pass.
function _fused_recovery_sir_kernel!(infected, recovered, ti_recovered, current_ti, n)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > n && return
    if infected[i] == 0x01 && ti_recovered[i] <= current_ti
        infected[i]  = 0x00
        recovered[i] = 0x01
    end
    return
end

# ============================================================================
# High-level GPU step functions
# ============================================================================

"""
    gpu_step_state!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Run GPU-accelerated state transitions (recovery) for one disease.

- **SIR**: infected → recovered when `ti_recovered ≤ current_ti`
- **SIS**: infected → susceptible when infection duration has elapsed
- **SEIR**: exposed → infected when latent period has elapsed,
            then infected → recovered when `ti_recovered ≤ current_ti`

This replaces the CPU `step_state!` call. Disease-induced death (`p_death`)
is **not** handled on GPU; use `to_cpu` and the CPU code path for that.
"""
function gpu_step_state!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gpu_dis = gsim.diseases[disease_name]
    disease = gsim.sim.diseases[disease_name]
    n = Int32(gpu_dis.n)
    groups = cld(n, CUDA_THREADS)
    ti_f = Float32(current_ti)

    if disease isa Starsim.SEIR && gpu_dis.exposed !== nothing
        dt = gsim.sim.pars.dt
        dur_exp_ts = Float32(disease.dur_exp / dt)
        dur_inf_ts = Float32(disease.dur_inf / dt)

        # Pre-sample recovery durations for E→I transitions
        cpu_dur = _sample_recovery_durations(disease, dur_inf_ts, Int(n))
        copyto!(gsim.jitter_buf, 1, cpu_dur, 1, Int(n))

        # Exposed → Infected (latent period elapsed, with recovery time)
        @cuda threads=CUDA_THREADS blocks=groups _seir_exposure_kernel!(
            gpu_dis.exposed, gpu_dis.infected, gpu_dis.ti_exposed,
            gpu_dis.ti_infected, gpu_dis.ti_recovered, ti_f, dur_exp_ts,
            gsim.jitter_buf, n,
        )

        # Infected → Recovered
        if gpu_dis.recovered !== nothing
            @cuda threads=CUDA_THREADS blocks=groups _sir_recovery_kernel!(
                gpu_dis.infected, gpu_dis.recovered, gpu_dis.ti_recovered,
                ti_f, n,
            )
        end
    elseif disease isa Starsim.SIR && gpu_dis.recovered !== nothing
        @cuda threads=CUDA_THREADS blocks=groups _sir_recovery_kernel!(
            gpu_dis.infected, gpu_dis.recovered, gpu_dis.ti_recovered,
            ti_f, n,
        )
    elseif disease isa Starsim.SIS
        # SIS uses ti_recovered (lognormal durations), same as SIR
        if gpu_dis.ti_recovered !== nothing
            @cuda threads=CUDA_THREADS blocks=groups _sis_recovery_ti_kernel!(
                gpu_dis.infected, gpu_dis.susceptible, gpu_dis.ti_recovered,
                ti_f, n,
            )
        else
            dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
            @cuda threads=CUDA_THREADS blocks=groups _sis_recovery_kernel!(
                gpu_dis.infected, gpu_dis.susceptible, gpu_dis.ti_infected,
                ti_f, dur_inf_ts, n,
            )
        end
    end

    CUDA.synchronize()
    return gsim
end

"""Ensure edge buffers can hold at least `n` elements, growing if needed."""
function _ensure_edge_capacity!(gsim::GPUSim, n::Int)
    if n > gsim.edge_capacity
        new_cap = max(n, gsim.edge_capacity * 2)
        gsim.edge_p1 = CuVector{Int32}(zeros(Int32, new_cap))
        gsim.edge_p2 = CuVector{Int32}(zeros(Int32, new_cap))
        gsim.edge_beta = CuVector{Float32}(zeros(Float32, new_cap))
        gsim.rng_buf = CuVector{Float32}(zeros(Float32, new_cap))
        gsim.edge_capacity = new_cap
    end
    return
end

"""
Sample `n` recovery durations on CPU matching the disease's recovery_dist.
Returns a Vector{Float32} of duration values (in timesteps).
"""
function _sample_recovery_durations(disease, dur_inf_ts::Float32, n::Int)
    mean_f = Float64(dur_inf_ts)
    dist = disease.recovery_dist
    buf = Vector{Float32}(undef, n)
    rng = disease.rng
    if dist === :exponential
        d = Distributions.Exponential(mean_f)
        @inbounds for i in 1:n
            buf[i] = Float32(rand(rng, d))
        end
    else  # :lognormal (default, matches Python)
        std_f = 1.0
        σ² = log(1 + (std_f / mean_f)^2)
        μ = log(mean_f) - σ² / 2
        d = Distributions.LogNormal(μ, sqrt(σ²))
        @inbounds for i in 1:n
            buf[i] = Float32(rand(rng, d))
        end
    end
    return buf
end

"""
    gpu_transmit!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Run GPU-accelerated transmission for one disease across all networks.

Uses pre-allocated GPU edge buffers to avoid per-timestep CuVector
allocation. Buffers grow automatically if edge count exceeds capacity.
"""
function gpu_transmit!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gpu_dis = gsim.diseases[disease_name]
    disease = gsim.sim.diseases[disease_name]
    dd = Starsim.disease_data(disease)
    n_agents = Int32(gpu_dis.n)
    ti_f = Float32(current_ti)

    # Snapshot infected/susceptible for synchronous transmission (matching CPU)
    copyto!(gsim.snap_infected, gpu_dis.infected)
    copyto!(gsim.snap_susceptible, gpu_dis.susceptible)

    # Reset new_infected buffer
    CUDA.fill!(gsim.new_infected, 0x00)

    for (net_name, net) in gsim.sim.networks
        edges = Starsim.network_edges(net)
        isempty(edges) && continue

        beta_dt = get(dd.beta_per_dt, net_name, 0.0)
        beta_dt == 0.0 && continue
        beta_dt_f32 = Float32(beta_dt)

        n_edges = length(edges)
        n_edges_i32 = Int32(n_edges)
        groups_e = cld(n_edges, CUDA_THREADS)

        # Ensure buffers are large enough
        _ensure_edge_capacity!(gsim, n_edges)

        # Copy edge data into pre-allocated GPU buffers (no allocation)
        cpu_p1 = Int32.(edges.p1)
        cpu_p2 = Int32.(edges.p2)
        cpu_eb = Float32.(edges.beta)
        copyto!(gsim.edge_p1, 1, cpu_p1, 1, n_edges)
        copyto!(gsim.edge_p2, 1, cpu_p2, 1, n_edges)
        copyto!(gsim.edge_beta, 1, cpu_eb, 1, n_edges)

        # Generate random numbers and copy to pre-allocated buffer
        cpu_rng = rand(Float32, n_edges)
        copyto!(gsim.rng_buf, 1, cpu_rng, 1, n_edges)

        # Forward direction: p1[i] → p2[i] — reads from snapshots
        @cuda threads=CUDA_THREADS blocks=groups_e _transmission_kernel!(
            gsim.new_infected, gsim.snap_susceptible, gsim.snap_infected,
            gpu_dis.rel_trans, gpu_dis.rel_sus,
            gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
            beta_dt_f32, gsim.rng_buf, n_edges_i32,
        )

        # Reverse direction for bidirectional networks
        if Starsim.network_data(net).bidirectional
            cpu_rng_rev = rand(Float32, n_edges)
            copyto!(gsim.rng_buf, 1, cpu_rng_rev, 1, n_edges)
            @cuda threads=CUDA_THREADS blocks=groups_e _transmission_kernel!(
                gsim.new_infected, gsim.snap_susceptible, gsim.snap_infected,
                gpu_dis.rel_trans, gpu_dis.rel_sus,
                gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
                beta_dt_f32, gsim.rng_buf, n_edges_i32,
            )
        end
    end

    CUDA.synchronize()

    # Apply accumulated new infections
    groups_a = cld(n_agents, CUDA_THREADS)

    if disease isa Starsim.SEIR && gpu_dis.exposed !== nothing
        # SEIR: new infections go to Exposed state (not directly Infected)
        @cuda threads=CUDA_THREADS blocks=groups_a _apply_exposures_seir_kernel!(
            gpu_dis.susceptible, gpu_dis.exposed, gpu_dis.ti_exposed,
            gsim.new_infected, ti_f, n_agents,
        )
    elseif disease isa Starsim.SIR && gpu_dis.ti_recovered !== nothing
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
        # Sample recovery durations matching CPU distribution
        cpu_dur = _sample_recovery_durations(disease, dur_inf_ts, gpu_dis.n)
        copyto!(gsim.jitter_buf, 1, cpu_dur, 1, gpu_dis.n)
        @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sir_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            gpu_dis.ti_recovered, gsim.new_infected, ti_f,
            gsim.jitter_buf, n_agents,
        )
    elseif disease isa Starsim.SIS
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
        cpu_dur = _sample_recovery_durations(disease, dur_inf_ts, gpu_dis.n)
        copyto!(gsim.jitter_buf, 1, cpu_dur, 1, gpu_dis.n)
        @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sis_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            gpu_dis.ti_recovered, gsim.new_infected, ti_f,
            gsim.jitter_buf, n_agents,
        )
    end

    CUDA.synchronize()
    return gsim
end

"""
    gpu_step!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Combined GPU disease step: state transitions (recovery) followed by
transmission. Equivalent to calling `step_state!` then `step!` for one
disease, but both execute on the CUDA GPU.

⚠ **Loop ordering**: In the Python starsim loop, `diseases.step_state`
(step 5) runs BEFORE `networks.step` (step 7), which runs BEFORE
`diseases.step` i.e. transmission (step 9). This convenience function
bundles steps 5 and 9 together, which is only correct when the network
does not change between recovery and transmission within a single timestep.

For full fidelity with the Python loop order, call `gpu_step_state!` and
`gpu_transmit!` separately with `networks.step` in between:

```julia
for ti in 1:sim.t.npts
    # Step 5: State transitions (recovery)
    gpu_step_state!(gsim, :sir; current_ti=ti)

    # Step 7: Network rewiring on CPU
    for (_, net) in sim.networks
        step!(net, sim)
    end

    # Step 9: Transmission
    gpu_transmit!(gsim, :sir; current_ti=ti)
end
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
    groups = cld(n, CUDA_THREADS)
    ti_f = Float32(current_ti)

    @cuda threads=CUDA_THREADS blocks=groups _waning_kernel!(
        gpu_dis.recovered, gpu_dis.susceptible, gpu_dis.ti_recovered,
        ti_f, wane_dur, n,
    )
    CUDA.synchronize()
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

Uses `copyto!` to overwrite existing CuVector buffers without reallocating.
"""
function sync_to_gpu!(gsim::GPUSim)
    people = gsim.sim.people
    n_p = min(length(people.alive.raw), gsim.people.n)
    copyto!(gsim.people.alive,  UInt8.(people.alive.raw[1:n_p]))
    copyto!(gsim.people.age,    Float32.(people.age.raw[1:n_p]))
    copyto!(gsim.people.female, UInt8.(people.female.raw[1:n_p]))
    copyto!(gsim.people.slot,   Int32.(people.slot.raw[1:n_p]))

    for (name, gpu_dis) in gsim.diseases
        disease = gsim.sim.diseases[name]
        if disease isa Starsim.SIR
            _sync_infection_to_gpu!(disease.infection, gpu_dis)
            n = min(length(disease.recovered.raw), gpu_dis.n)
            copyto!(gpu_dis.recovered,   UInt8.(disease.recovered.raw[1:n]))
            copyto!(gpu_dis.ti_recovered, Float32.(disease.ti_recovered.raw[1:n]))
        elseif disease isa Starsim.SIS
            _sync_infection_to_gpu!(disease.infection, gpu_dis)
            if gpu_dis.ti_recovered !== nothing
                n = min(length(disease.ti_recovered.raw), gpu_dis.n)
                copyto!(gpu_dis.ti_recovered, Float32.(disease.ti_recovered.raw[1:n]))
            end
        elseif disease isa Starsim.SEIR
            _sync_infection_to_gpu!(disease.infection, gpu_dis)
            n = min(length(disease.recovered.raw), gpu_dis.n)
            copyto!(gpu_dis.recovered,    UInt8.(disease.recovered.raw[1:n]))
            copyto!(gpu_dis.ti_recovered, Float32.(disease.ti_recovered.raw[1:n]))
            copyto!(gpu_dis.exposed,      UInt8.(disease.exposed.raw[1:n]))
            copyto!(gpu_dis.ti_exposed,   Float32.(disease.ti_exposed.raw[1:n]))
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
# Fused GPU step — single-pass recovery + transmission with GPU-side RNG
# ============================================================================

"""
    gpu_reset_rng!(gsim::GPUSim; current_ti::Int)

Reset per-agent RNG seeds to a deterministic state for timestep `ti`.
Called at the start of each timestep in CRN mode to ensure reproducibility
regardless of the order operations are called within a timestep.

Matches the Python starsim pattern: `seed = base_seed + ti * dt_jump_size`.
"""
function gpu_reset_rng!(gsim::GPUSim; current_ti::Int)
    n = Int32(gsim.people.n)
    ti_jump = UInt32(current_ti) * CRN_DT_JUMP
    groups = cld(Int(n), CUDA_THREADS)
    @cuda threads=CUDA_THREADS blocks=groups _crn_reset_seeds_kernel!(
        gsim.rng_seeds_agent, gsim.rng_seeds_base, ti_jump, n,
    )
    CUDA.synchronize()
    return gsim
end

"""
    gpu_step_fused!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

All-GPU disease step using fused kernels and GPU-side xorshift32 RNG.
Eliminates all per-step CPU→GPU random number transfers for transmission.

Uses synchronous (snapshot+batch) transmission matching the CPU code:
  1. Recovery kernel runs first
  2. Infected/susceptible arrays are snapshot-copied on GPU
  3. Fused transmission kernels read from snapshots, write to new_infected buffer
  4. Apply kernel batch-applies infections with exponential recovery durations

When CRN mode is active (`crn_enabled()` was true at `to_gpu` time):
  - Resets per-agent seeds deterministically for this timestep
  - Uses pairwise XOR combining of source/target agent RNG streams
  - Produces reproducible results given the same `sim.pars.rand_seed`

When CRN is off, uses per-edge seeds (faster, non-reproducible).

Requires cached edges: call `cache_edges!(gsim)` first.
"""
function gpu_step_fused!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gsim.cached_edges || error("gpu_step_fused! requires cached edges. Call cache_edges!(gsim) first.")

    gpu_dis = gsim.diseases[disease_name]
    disease = gsim.sim.diseases[disease_name]
    n_agents = Int32(gpu_dis.n)
    ti_f = Float32(current_ti)
    n_edges = Int32(gsim.cached_n_edges)
    groups_a = cld(Int(n_agents), CUDA_THREADS)
    groups_e = cld(Int(n_edges), CUDA_THREADS)

    beta_dt_f32 = get(gsim.cached_beta_dt, disease_name, Float32(0))

    # CRN: reset per-agent seeds deterministically for this timestep
    if gsim.crn_mode
        ti_jump = UInt32(current_ti) * CRN_DT_JUMP
        @cuda threads=CUDA_THREADS blocks=groups_a _crn_reset_seeds_kernel!(
            gsim.rng_seeds_agent, gsim.rng_seeds_base, ti_jump, n_agents,
        )
    else
        # Non-CRN: ensure per-edge seed array is large enough
        if length(gsim.rng_seeds) < n_edges
            gsim.rng_seeds = CuVector{UInt32}(rand(UInt32(1):UInt32(0xfffffffe), Int(n_edges)))
        end
    end

    if disease isa Starsim.SIR && gpu_dis.recovered !== nothing
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)

        # Kernel 1: Recovery (per-agent)
        @cuda threads=CUDA_THREADS blocks=groups_a _fused_recovery_sir_kernel!(
            gpu_dis.infected, gpu_dis.recovered, gpu_dis.ti_recovered,
            ti_f, n_agents,
        )

        if beta_dt_f32 > Float32(0)
            # Snapshot infected/susceptible for synchronous transmission
            copyto!(gsim.snap_infected, gpu_dis.infected)
            copyto!(gsim.snap_susceptible, gpu_dis.susceptible)
            CUDA.fill!(gsim.new_infected, 0x00)

            if gsim.crn_mode
                # CRN path: pairwise XOR combining — reads snapshots, writes new_infected
                @cuda threads=CUDA_THREADS blocks=groups_e _crn_fused_transmit_sir_kernel!(
                    gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                    gpu_dis.rel_trans, gpu_dis.rel_sus,
                    gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
                    beta_dt_f32, gsim.rng_seeds_agent, gsim.people.slot,
                    n_edges,
                )
                if gsim.cached_bidirectional
                    @cuda threads=CUDA_THREADS blocks=groups_e _crn_fused_transmit_sir_kernel!(
                        gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                        gpu_dis.rel_trans, gpu_dis.rel_sus,
                        gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
                        beta_dt_f32, gsim.rng_seeds_agent, gsim.people.slot,
                        n_edges,
                    )
                end
            else
                # Non-CRN path: per-edge seeds — reads snapshots, writes new_infected
                @cuda threads=CUDA_THREADS blocks=groups_e _fused_transmit_sir_kernel!(
                    gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                    gpu_dis.rel_trans, gpu_dis.rel_sus,
                    gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
                    beta_dt_f32, gsim.rng_seeds, n_edges,
                )
                if gsim.cached_bidirectional
                    @cuda threads=CUDA_THREADS blocks=groups_e _fused_transmit_sir_kernel!(
                        gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                        gpu_dis.rel_trans, gpu_dis.rel_sus,
                        gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
                        beta_dt_f32, gsim.rng_seeds, n_edges,
                    )
                end
            end

            CUDA.synchronize()

            # Apply infections with lognormal recovery duration (matching CPU)
            cpu_dur = _sample_recovery_durations(disease, dur_inf_ts, Int(n_agents))
            copyto!(gsim.jitter_buf, 1, cpu_dur, 1, Int(n_agents))
            @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sir_kernel!(
                gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
                gpu_dis.ti_recovered, gsim.new_infected, ti_f,
                gsim.jitter_buf, n_agents,
            )
        end
    elseif disease isa Starsim.SIS
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)

        @cuda threads=CUDA_THREADS blocks=groups_a _sis_recovery_kernel!(
            gpu_dis.infected, gpu_dis.susceptible, gpu_dis.ti_infected,
            ti_f, dur_inf_ts, n_agents,
        )

        if beta_dt_f32 > Float32(0)
            # Snapshot infected/susceptible for synchronous transmission
            copyto!(gsim.snap_infected, gpu_dis.infected)
            copyto!(gsim.snap_susceptible, gpu_dis.susceptible)
            CUDA.fill!(gsim.new_infected, 0x00)

            if gsim.crn_mode
                @cuda threads=CUDA_THREADS blocks=groups_e _crn_fused_transmit_sis_kernel!(
                    gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                    gpu_dis.rel_trans, gpu_dis.rel_sus,
                    gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
                    beta_dt_f32, gsim.rng_seeds_agent, gsim.people.slot,
                    n_edges,
                )
                if gsim.cached_bidirectional
                    @cuda threads=CUDA_THREADS blocks=groups_e _crn_fused_transmit_sis_kernel!(
                        gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                        gpu_dis.rel_trans, gpu_dis.rel_sus,
                        gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
                        beta_dt_f32, gsim.rng_seeds_agent, gsim.people.slot,
                        n_edges,
                    )
                end
            else
                @cuda threads=CUDA_THREADS blocks=groups_e _fused_transmit_sis_kernel!(
                    gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                    gpu_dis.rel_trans, gpu_dis.rel_sus,
                    gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
                    beta_dt_f32, gsim.rng_seeds, n_edges,
                )
                if gsim.cached_bidirectional
                    @cuda threads=CUDA_THREADS blocks=groups_e _fused_transmit_sis_kernel!(
                        gsim.new_infected, gsim.snap_infected, gsim.snap_susceptible,
                        gpu_dis.rel_trans, gpu_dis.rel_sus,
                        gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
                        beta_dt_f32, gsim.rng_seeds, n_edges,
                    )
                end
            end

            CUDA.synchronize()

            # Apply infections
            @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sis_kernel!(
                gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
                gsim.new_infected, ti_f, n_agents,
            )
        end
    end

    CUDA.synchronize()
    return gsim
end

# ============================================================================
# Static network caching — upload edges once, reuse every timestep
# ============================================================================

"""
    cache_edges!(gsim::GPUSim)

Snapshot the current CPU edge arrays to GPU and mark them as cached.
Subsequent calls to `gpu_transmit!` will skip per-step edge upload and
reuse these cached GPU buffers — eliminating the main CPU→GPU bottleneck.

Call this after the first `network.step!` to capture representative edges.
For truly static networks (edges don't change), this gives the GPU's
peak throughput.

To uncache: `uncache_edges!(gsim)`.
"""
function cache_edges!(gsim::GPUSim)
    total_edges = 0
    bidir = false

    for (net_name, net) in gsim.sim.networks
        edges = Starsim.network_edges(net)
        n_edges = length(edges)
        total_edges += n_edges
        bidir = bidir || Starsim.network_data(net).bidirectional
    end

    _ensure_edge_capacity!(gsim, total_edges)

    # Upload edges and cache beta_per_dt for each disease
    offset = 0
    for (net_name, net) in gsim.sim.networks
        edges = Starsim.network_edges(net)
        n_edges = length(edges)
        n_edges == 0 && continue

        cpu_p1 = Int32.(edges.p1)
        cpu_p2 = Int32.(edges.p2)
        cpu_eb = Float32.(edges.beta)
        copyto!(gsim.edge_p1, offset + 1, cpu_p1, 1, n_edges)
        copyto!(gsim.edge_p2, offset + 1, cpu_p2, 1, n_edges)
        copyto!(gsim.edge_beta, offset + 1, cpu_eb, 1, n_edges)
        offset += n_edges

        # Cache beta_per_dt for each disease
        for (dname, disease) in gsim.sim.diseases
            dd = Starsim.disease_data(disease)
            bdt = get(dd.beta_per_dt, net_name, 0.0)
            gsim.cached_beta_dt[dname] = Float32(bdt)
        end
    end

    # Ensure rng_buf is large enough
    if length(gsim.rng_buf) < total_edges
        gsim.rng_buf = CuVector{Float32}(zeros(Float32, total_edges))
    end

    gsim.cached_edges = true
    gsim.cached_n_edges = total_edges
    gsim.cached_bidirectional = bidir
    CUDA.synchronize()
    return gsim
end

"""Uncache edges, reverting to per-step upload."""
function uncache_edges!(gsim::GPUSim)
    gsim.cached_edges = false
    gsim.cached_n_edges = 0
    return gsim
end

"""
    gpu_transmit_cached!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)

Transmission using cached GPU edges. No CPU→GPU edge transfer — only
random number generation and kernel launch. This is the fast path for
static or slowly-changing networks.
"""
function gpu_transmit_cached!(gsim::GPUSim, disease_name::Symbol; current_ti::Int)
    gsim.cached_edges || error("Edges not cached. Call cache_edges!(gsim) first.")

    gpu_dis = gsim.diseases[disease_name]
    n_agents = Int32(gpu_dis.n)
    ti_f = Float32(current_ti)
    n_edges = gsim.cached_n_edges
    n_edges_i32 = Int32(n_edges)
    groups_e = cld(n_edges, CUDA_THREADS)

    beta_dt_f32 = get(gsim.cached_beta_dt, disease_name, Float32(0))
    beta_dt_f32 == Float32(0) && return gsim

    # Snapshot infected/susceptible for synchronous transmission
    copyto!(gsim.snap_infected, gpu_dis.infected)
    copyto!(gsim.snap_susceptible, gpu_dis.susceptible)

    # Reset new_infected
    fill!(gsim.new_infected, 0x00)

    # Only need to upload random numbers (CPU → GPU)
    cpu_rng = rand(Float32, n_edges)
    copyto!(gsim.rng_buf, 1, cpu_rng, 1, n_edges)

    # Forward transmission — reads from snapshots
    @cuda threads=CUDA_THREADS blocks=groups_e _transmission_kernel!(
        gsim.new_infected, gsim.snap_susceptible, gsim.snap_infected,
        gpu_dis.rel_trans, gpu_dis.rel_sus,
        gsim.edge_p1, gsim.edge_p2, gsim.edge_beta,
        beta_dt_f32, gsim.rng_buf, n_edges_i32,
    )

    # Reverse direction
    if gsim.cached_bidirectional
        cpu_rng_rev = rand(Float32, n_edges)
        copyto!(gsim.rng_buf, 1, cpu_rng_rev, 1, n_edges)
        @cuda threads=CUDA_THREADS blocks=groups_e _transmission_kernel!(
            gsim.new_infected, gsim.snap_susceptible, gsim.snap_infected,
            gpu_dis.rel_trans, gpu_dis.rel_sus,
            gsim.edge_p2, gsim.edge_p1, gsim.edge_beta,
            beta_dt_f32, gsim.rng_buf, n_edges_i32,
        )
    end

    CUDA.synchronize()

    # Apply infections
    groups_a = cld(n_agents, CUDA_THREADS)
    disease = gsim.sim.diseases[disease_name]

    if disease isa Starsim.SIR && gpu_dis.ti_recovered !== nothing
        dur_inf_ts = Float32(disease.dur_inf / gsim.sim.pars.dt)
        cpu_dur = _sample_recovery_durations(disease, dur_inf_ts, gpu_dis.n)
        copyto!(gsim.jitter_buf, 1, cpu_dur, 1, gpu_dis.n)
        @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sir_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            gpu_dis.ti_recovered, gsim.new_infected, ti_f,
            gsim.jitter_buf, n_agents,
        )
    elseif disease isa Starsim.SIS
        @cuda threads=CUDA_THREADS blocks=groups_a _apply_infections_sis_kernel!(
            gpu_dis.susceptible, gpu_dis.infected, gpu_dis.ti_infected,
            gsim.new_infected, ti_f, n_agents,
        )
    end

    CUDA.synchronize()
    return gsim
end

# ============================================================================
# High-level run_gpu! — drop-in GPU backend for run!(sim; backend=:gpu)
# ============================================================================

"""
Apply SIS immunity boost to agents newly infected this step.

Called after `to_cpu` in the results round-trip. Compares current state
(from GPU) with previous state to detect new infections and applies
`imm_boost` to their immunity level.
"""
function _gpu_sis_immunity_boost!(sim, disease_names)
    for dname in disease_names
        disease = sim.diseases[dname]
        if disease isa Starsim.SIS
            imm_raw = disease.immunity.raw
            ti_inf_raw = disease.infection.ti_infected.raw
            ti = Float64(sim.loop.ti)
            @inbounds for u in sim.people.auids.values
                # Agent was newly infected this step if ti_infected == current_ti
                if disease.infection.infected.raw[u] && ti_inf_raw[u] == ti
                    imm_raw[u] += disease.imm_boost
                    disease.infection.rel_sus.raw[u] = max(0.0, 1.0 - imm_raw[u])
                end
            end
        end
    end
    return
end

"""
Handle SIS immunity waning on CPU for GPU simulation.

SIS has immunity that wanes each step, modifying `rel_sus`. Since this
requires per-agent float arithmetic on the `immunity` array (which is
NOT on GPU), we need a GPU→CPU sync to run the waning, then re-upload.
"""
function _gpu_sis_immunity_waning!(gsim::GPUSim, sim, disease_names)
    for dname in disease_names
        disease = sim.diseases[dname]
        if disease isa Starsim.SIS
            gpu_dis = gsim.diseases[dname]
            n = gpu_dis.n

            # First, sync GPU infected state to CPU so we know current state
            cpu_inf = Array(gpu_dis.infected)
            @inbounds for i in 1:min(length(disease.infection.infected.raw), n)
                disease.infection.infected.raw[i] = cpu_inf[i] != 0x00
            end
            # Also sync susceptible
            cpu_sus = Array(gpu_dis.susceptible)
            @inbounds for i in 1:min(length(disease.infection.susceptible.raw), n)
                disease.infection.susceptible.raw[i] = cpu_sus[i] != 0x00
            end

            # Run immunity waning on CPU (matches _update_immunity!)
            dt = sim.pars.dt
            waning_prob = 1.0 - exp(-disease.waning * dt)
            imm_raw = disease.immunity.raw
            rel_sus_raw = disease.infection.rel_sus.raw
            @inbounds for u in sim.people.auids.values
                if imm_raw[u] > 0.0
                    imm_raw[u] *= (1.0 - waning_prob)
                    rel_sus_raw[u] = max(0.0, 1.0 - imm_raw[u])
                end
            end

            # Re-upload rel_sus to GPU
            copyto!(gpu_dis.rel_sus, Float32.(rel_sus_raw[1:n]))
        end
    end
    return
end

"""
    Starsim.run_gpu!(sim::Sim; verbose::Int=1, cache_edges::Bool=false)

GPU backend for `run!`. Called automatically when `run!(sim; backend=:gpu)`.

Follows the exact same 16-step loop order as the CPU path:

1. `start_step!` for all modules (jump distributions)
2. `step_state!` (recovery) on GPU
3. `networks.step` on CPU (regenerate edges)
4. `diseases.step` (transmission) on GPU with fresh edges uploaded each step
5. `to_cpu` → `update_results!` → `sync_to_gpu!` (results round-trip)
6. `finish_step!` for all modules

# Keyword arguments
- `verbose::Int=1`: 0=silent, 1=summary, 2=per-step progress
- `cache_edges::Bool=false`: if `true`, edges are generated once and reused
  every step (only correct for truly static networks). Default `false` re-steps
  networks on CPU each timestep matching the CPU loop exactly.

# Example
```julia
using CUDA  # triggers GPU extension loading
sim = Sim(n_agents=1_000_000, diseases=[SIR(beta=0.05, dur_inf=10.0)],
          networks=[RandomNet(n_contacts=10)], stop=365.0)
run!(sim; backend=:gpu)
```
"""
function Starsim.run_gpu!(sim::Starsim.Sim; verbose::Int=1, cache_edges::Bool=false)
    if !sim.initialized
        Starsim.init!(sim)
    end

    n_agents = sim.pars.n_agents
    npts = sim.t.npts

    if verbose >= 1
        println("Running simulation on GPU ($(n_agents) agents, $(npts) steps, " *
                "dt=$(sim.pars.dt), device=$(CUDA.device()))")
    end

    gsim = Starsim.to_gpu(sim)
    disease_names = collect(keys(sim.diseases))

    # For cached-edge mode, generate edges once and upload
    if cache_edges
        for (_, net) in sim.networks
            Starsim.step!(net, sim)
        end
        cache_edges!(gsim)
    end

    for ti in 1:npts
        sim.loop.ti = ti

        if verbose >= 2
            year = sim.pars.start + (ti - 1) * sim.pars.dt
            println("  Step $ti / $npts (year=$(round(year, digits=2)))")
        end

        # Step 2: Module lifecycle — start_step (jump distributions)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.start_step!(mod, sim)
        end

        # Step 5: Disease state transitions (recovery) on GPU
        for dname in disease_names
            gpu_step_state!(gsim, dname; current_ti=ti)
        end

        # SIS immunity waning: requires CPU roundtrip to update rel_sus
        # This must happen after recovery and before transmission
        _gpu_sis_immunity_waning!(gsim, sim, disease_names)

        # Step 7: Network rewiring on CPU (regenerates edges)
        if !cache_edges
            for (_, net) in sim.networks
                Starsim.step!(net, sim)
            end
        end

        # Step 9: Transmission on GPU (uploads fresh edges each step)
        for dname in disease_names
            if cache_edges && gsim.cached_edges
                gpu_transmit_cached!(gsim, dname; current_ti=ti)
            else
                gpu_transmit!(gsim, dname; current_ti=ti)
            end
        end

        # Steps 10-12: Results tracking
        _has_sis = any(sim.diseases[dname] isa Starsim.SIS for dname in disease_names)

        if _has_sis
            # SIS requires full GPU→CPU round-trip for immunity tracking
            Starsim.to_cpu(gsim)
            _gpu_sis_immunity_boost!(sim, disease_names)
            Starsim.update_people_results!(sim.people, ti, sim.results)
            for (_, mod) in Starsim.all_modules(sim)
                Starsim.update_results!(mod, sim)
            end
            if ti < npts
                sync_to_gpu!(gsim)
            end
        else
            # Fast path: count results on GPU (no full state transfer)
            n_alive = Int(sum(gsim.people.alive))
            sim.results[:n_alive][ti] = Float64(n_alive)

            for dname in disease_names
                gpu_dis = gsim.diseases[dname]
                disease = sim.diseases[dname]
                md = Starsim.module_data(disease)
                ti > length(md.results[:n_infected].values) && continue

                n_sus = Int(sum(gpu_dis.susceptible))
                n_inf = Int(sum(gpu_dis.infected))

                md.results[:n_susceptible][ti] = Float64(n_sus)
                md.results[:n_infected][ti] = Float64(n_inf)

                if gpu_dis.recovered !== nothing
                    n_rec = Int(sum(gpu_dis.recovered))
                    md.results[:n_recovered][ti] = Float64(n_rec)
                end

                if gpu_dis.exposed !== nothing
                    n_exp = Int(sum(gpu_dis.exposed))
                    md.results[:n_exposed][ti] = Float64(n_exp)
                end

                n_total = Float64(n_alive)
                md.results[:prevalence][ti] = n_total > 0.0 ? n_inf / n_total : 0.0
            end
            # No sync_to_gpu! needed — GPU state unchanged
        end

        # Step 14: Module lifecycle — finish_step
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.finish_step!(mod, sim)
        end
    end

    # Final GPU→CPU sync
    Starsim.to_cpu(gsim)

    # Finalize all modules
    for (_, mod) in Starsim.all_modules(sim)
        Starsim.finalize!(mod, sim)
    end

    # Scale results
    if sim.pars.pop_scale != 1.0
        Starsim.scale_results!(sim.results, sim.pars.pop_scale)
        for (_, mod) in Starsim.all_modules(sim)
            Starsim.scale_results!(Starsim.module_results(mod), sim.pars.pop_scale)
        end
    end

    if verbose >= 1
        println("GPU simulation complete ($npts steps)")
    end

    sim.complete = true
    return sim
end

# ============================================================================
# Exports
# ============================================================================

export GPUSim, GPUPeopleArrays, GPUDiseaseArrays
export gpu_step!, gpu_step_state!, gpu_transmit!, gpu_waning!, sync_to_gpu!
export cache_edges!, uncache_edges!, gpu_transmit_cached!
export gpu_step_fused!, gpu_reset_rng!

end # module
