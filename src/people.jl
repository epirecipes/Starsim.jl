"""
People (agent population) management for Starsim.jl.

Mirrors Python starsim's `people.py`. Manages all agent states, handles
births (grow!) and deaths (request_death!, step_die!, finish_step!).
"""

"""
    People

The agent population. Stores all agent states and manages the active
UID set. Parametric design allows different array backends.

# Built-in states
- `uid` — unique agent identifiers
- `slot` — RNG slot indices
- `alive` — life status
- `age` — agent age in years
- `female` — biological sex
- `ti_dead` — scheduled death time
- `ti_removed` — removal time
- `scale` — per-agent scaling factor
- `parent` — parent UID (for births)

# Example
```julia
people = People(10_000)
init_people!(people)
```
"""
mutable struct People
    n_agents_init::Int

    # Index states (not regular agent states)
    uid::StateVector{Int64, Vector{Int64}}
    slot::StateVector{Int64, Vector{Int64}}

    # Built-in states
    alive::StateVector{Bool, Vector{Bool}}
    age::StateVector{Float64, Vector{Float64}}
    female::StateVector{Bool, Vector{Bool}}
    ti_dead::StateVector{Float64, Vector{Float64}}
    ti_removed::StateVector{Float64, Vector{Float64}}
    scale::StateVector{Float64, Vector{Float64}}
    parent::StateVector{Int64, Vector{Int64}}

    # Active UIDs (shared reference with all states)
    auids::UIDs{Vector{Int}}
    auids_ref::Base.RefValue{UIDs{Vector{Int}}}

    # Module states registry
    module_states::Vector{StateVector}

    # Next UID to assign
    next_uid::Int

    # CRN slot assignment
    slot_scale::Float64  # 0.0 = CRN disabled, >0 = CRN enabled (slot range scale)
    slot_rng::StableRNG  # RNG for drawing newborn slots

    # Reference to sim (set during init)
    initialized::Bool
end

function People(n::Int=10_000; slot_scale::Float64=0.0)
    auids_ref = Ref(UIDs())
    People(
        n,
        IndexState(:uid),
        IndexState(:slot),
        BoolState(:alive; default=true),
        FloatState(:age; default=0.0),
        BoolState(:female; default=false),
        FloatState(:ti_dead; default=Inf),
        FloatState(:ti_removed; default=Inf),
        FloatState(:scale; default=1.0),
        IntState(:parent; default=INT_NAN),
        UIDs(),
        auids_ref,
        StateVector[],
        1,
        slot_scale,
        StableRNG(0),
        false
    )
end

function Base.show(io::IO, p::People)
    n_alive = length(p.auids)
    print(io, "People(n_init=$(p.n_agents_init), n_alive=$n_alive, next_uid=$(p.next_uid))")
end

Base.length(p::People) = length(p.auids)

"""
    init_people!(people::People; use_aging::Bool=true)

Initialize the People object with `n_agents_init` agents.
"""
function init_people!(people::People; use_aging::Bool=true)
    n = people.n_agents_init
    uids = UIDs(collect(1:n))
    people.auids = uids
    people.auids_ref[] = uids
    people.next_uid = n + 1

    # Pick up global slot_scale if not set explicitly
    if people.slot_scale == 0.0 && crn_enabled()
        people.slot_scale = get_slot_scale()
    end

    # Initialize all built-in states
    for s in builtin_states(people)
        init_state!(s, n, people.auids_ref)
    end

    # Set UIDs
    people.uid.raw .= 1:n
    # Set slots
    people.slot.raw .= 1:n
    # Set alive
    people.alive.raw .= true
    # Set sex (roughly 50/50)
    rng = StableRNG(12345)
    people.female.raw .= rand(rng, Bool, n)
    # Set age (uniform 0-60 if aging, matching Python starsim default)
    if use_aging
        people.age.raw .= rand(rng, n) .* 60.0
    end
    # Set defaults
    people.ti_dead.raw .= Inf
    people.ti_removed.raw .= Inf
    people.scale.raw .= 1.0
    people.parent.raw .= INT_NAN

    # Seed the slot RNG
    people.slot_rng = StableRNG(hash(:slot_assignment) ⊻ UInt64(54321))

    people.initialized = true
    return people
end

"""Get all built-in state vectors."""
function builtin_states(p::People)
    return [p.uid, p.slot, p.alive, p.age, p.female, p.ti_dead, p.ti_removed, p.scale, p.parent]
end

"""Get all states (built-in + module states)."""
function all_states(p::People)
    return vcat(builtin_states(p), p.module_states)
end

"""
    add_module_state!(people::People, state::StateVector)

Register a module's state with the People object so it is grown on births.
"""
function add_module_state!(people::People, state::StateVector)
    push!(people.module_states, state)
    if people.initialized
        init_state!(state, people.uid.len_tot, people.auids_ref)
        # Initialize values for existing agents
        init_vals!(state, people.auids)
    end
    return people
end

"""
    grow!(people::People, n::Int; parent_uids::Union{UIDs, Nothing}=nothing)

Add `n` new agents (births/immigration). Returns UIDs of new agents.

When CRN is enabled (`slot_scale > 0`), newborn slots are drawn from
`Uniform(N_init, slot_scale * N_init)` to avoid sequential assignment
and maintain CRN invariance. If `parent_uids` are provided and CRN is
enabled, the slot RNG is seeded per-parent for reproducibility.
"""
function grow!(people::People, n::Int; parent_uids::Union{UIDs, Nothing}=nothing)
    n == 0 && return UIDs()

    # Generate new UIDs
    new_uids = UIDs(collect(people.next_uid:people.next_uid + n - 1))
    people.next_uid += n

    # Grow all states
    for s in all_states(people)
        grow!(s, new_uids)
    end

    # Set defaults for new agents
    people.uid[new_uids] = new_uids.values
    people.alive[new_uids] = true
    people.age[new_uids] = 0.0
    people.ti_dead[new_uids] = Inf
    people.ti_removed[new_uids] = Inf
    people.scale[new_uids] = 1.0

    # CRN-safe slot assignment
    if people.slot_scale > 0.0
        N = people.n_agents_init
        lo = N + 1
        hi = Int(round(people.slot_scale * N))
        hi = max(hi, lo + 1)  # ensure valid range
        for (i, u) in enumerate(new_uids.values)
            people.slot.raw[u] = lo + rand(people.slot_rng, 0:(hi - lo))
        end
    else
        people.slot[new_uids] = new_uids.values
    end

    # Update active UIDs
    people.auids = UIDs(vcat(people.auids.values, new_uids.values))
    people.auids_ref[] = people.auids

    return new_uids
end

"""
    request_death!(people::People, death_uids::UIDs, ti::Int)

Schedule deaths for the given UIDs at the current timestep.
"""
function request_death!(people::People, death_uids::UIDs, ti::Int)
    people.ti_dead[death_uids] = Float64(ti)
    return people
end

"""
    step_die!(people::People, ti::Int, diseases::Vector)

Process deaths: identify who dies, call disease step_die!, mark as dead.
Returns UIDs of agents who died.
"""
function step_die!(people::People, ti::Int, diseases::AbstractVector)
    ti_f = Float64(ti)
    active = people.auids.values
    ti_dead_raw = people.ti_dead.raw
    alive_raw = people.alive.raw

    # Quick check: any deaths this timestep?
    has_deaths = false
    @inbounds for u in active
        if alive_raw[u] && ti_dead_raw[u] <= ti_f
            has_deaths = true
            break
        end
    end

    if !has_deaths
        return UIDs(Int[])
    end

    # Collect death UIDs
    death_vals = Int[]
    @inbounds for u in active
        if alive_raw[u] && ti_dead_raw[u] <= ti_f
            push!(death_vals, u)
        end
    end
    alive_deaths = UIDs(death_vals)

    for disease in diseases
        step_die!(disease, alive_deaths)
    end
    people.alive[alive_deaths] = false
    return alive_deaths
end

"""
    update_results!(people::People, ti::Int, results::Results)

Update people-level results (n_alive, etc.).
"""
function update_people_results!(people::People, ti::Int, results::Results)
    if haskey(results, :n_alive)
        results[:n_alive][ti] = Float64(length(people.auids))
    end
    return
end

"""
    finish_step!(people::People, dt::Float64, use_aging::Bool)

Remove dead agents from active UIDs, age the population.
"""
function finish_step!(people::People, dt::Float64, use_aging::Bool)
    active = people.auids.values
    alive_raw = people.alive.raw

    # Only rebuild auids if someone died
    all_alive = true
    @inbounds for u in active
        if !alive_raw[u]
            all_alive = false
            break
        end
    end

    if !all_alive
        new_vals = Int[]
        sizehint!(new_vals, length(active))
        @inbounds for u in active
            if alive_raw[u]
                push!(new_vals, u)
            end
        end
        people.auids = UIDs(new_vals)
        people.auids_ref[] = people.auids
    end

    # Age the population
    if use_aging
        @inbounds for u in people.auids.values
            people.age.raw[u] += dt
        end
    end
    return people
end

export People, init_people!, add_module_state!, grow!, request_death!,
       step_die!, update_people_results!, finish_step!, all_states, builtin_states
