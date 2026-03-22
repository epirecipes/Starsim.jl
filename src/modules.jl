"""
Module base for Starsim.jl.

Defines the module lifecycle interface via multiple dispatch.
All simulation components (diseases, networks, etc.) implement
these methods.
"""

# ============================================================================
# Module data — mutable companion to the abstract type
# ============================================================================

"""
    ModuleData

Common mutable state carried by every module instance. Stored as a field
rather than inherited (Julia uses composition over inheritance).

# Fields
- `name::Symbol` — short key-like name
- `label::String` — human-readable label
- `pars::Pars` — module parameters
- `t::Timeline` — module-specific timeline (may differ from sim)
- `results::Results` — module results
- `states::Vector{StateVector}` — registered agent states
- `dists::Vector{AbstractStarsimDist}` — registered distributions
- `initialized::Bool`
"""
mutable struct ModuleData
    name::Symbol
    label::String
    pars::Pars
    t::Timeline
    results::Results
    states::Vector{StateVector}
    dists::Vector{AbstractStarsimDist}
    initialized::Bool
end

function ModuleData(name::Symbol; label::String=string(name))
    ModuleData(name, label, Pars(), Timeline(), Results(),
               StateVector[], AbstractStarsimDist[], false)
end

export ModuleData

# ============================================================================
# Module interface — lifecycle methods via multiple dispatch
# ============================================================================

"""
    module_data(mod::AbstractModule) → ModuleData

Return the ModuleData for a module. Every concrete module must implement this.
"""
function module_data end

"""
    module_name(mod::AbstractModule) → Symbol

Return the module's name.
"""
module_name(mod::AbstractModule) = module_data(mod).name

"""
    module_pars(mod::AbstractModule) → Pars

Return the module's parameters.
"""
module_pars(mod::AbstractModule) = module_data(mod).pars

"""
    module_results(mod::AbstractModule) → Results

Return the module's results.
"""
module_results(mod::AbstractModule) = module_data(mod).results

"""
    module_states(mod::AbstractModule) → Vector{StateVector}

Return the module's registered states.
"""
module_states(mod::AbstractModule) = module_data(mod).states

"""
    module_timeline(mod::AbstractModule) → Timeline

Return the module's timeline.
"""
module_timeline(mod::AbstractModule) = module_data(mod).t

export module_data, module_name, module_pars, module_results, module_states, module_timeline

# ============================================================================
# State and result definition helpers
# ============================================================================

"""
    define_states!(mod::AbstractModule, states::StateVector...)

Register agent states with the module. These will be added to People
during initialization.
"""
function define_states!(mod::AbstractModule, states::StateVector...)
    md = module_data(mod)
    for s in states
        push!(md.states, s)
    end
    return mod
end

"""
    define_results!(mod::AbstractModule, results::Result...)

Register results with the module.
"""
function define_results!(mod::AbstractModule, results::Result...)
    md = module_data(mod)
    for r in results
        r.module_name = md.name
        push!(md.results, r)
    end
    return mod
end

"""
    define_pars!(mod::AbstractModule, pairs::Pair...)

Define default parameters for the module.
"""
function define_pars!(mod::AbstractModule, pairs::Pair...)
    md = module_data(mod)
    for (k, v) in pairs
        md.pars[Symbol(k)] = v
    end
    return mod
end

export define_states!, define_results!, define_pars!

# ============================================================================
# Lifecycle methods — default implementations (overridden by concrete types)
# ============================================================================

"""
    init_pre!(mod::AbstractModule, sim)

Link the module to the simulation, register states with People,
initialize distributions. Called during `init!(sim)`.
"""
function init_pre!(mod::AbstractModule, sim)
    md = module_data(mod)
    md.t = Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    # Register states with People
    for s in md.states
        add_module_state!(sim.people, s)
    end

    md.initialized = true
    return mod
end

"""
    init_post!(mod::AbstractModule, sim)

Initialize state values after all modules are linked.
Called after `init_pre!` for all modules.
"""
function init_post!(mod::AbstractModule, sim)
    return mod
end

"""
    init_results!(mod::AbstractModule)

Initialize result arrays. Called after states are set up.
"""
function init_results!(mod::AbstractModule)
    md = module_data(mod)
    npts = md.t.npts

    # Auto-generate results for BoolStates
    for s in md.states
        if s isa StateVector{Bool}
            r = Result(Symbol("n_", s.name);
                       label="Number $(s.label)",
                       module_name=md.name,
                       npts=npts,
                       scale=true)
            push!(md.results, r)
        end
    end
    return mod
end

"""
    start_step!(mod::AbstractModule, sim)

Pre-step initialization (jump distributions, etc.).
"""
function start_step!(mod::AbstractModule, sim)
    md = module_data(mod)
    # Jump distributions for this timestep
    for d in md.dists
        jump_dt!(d, md.t.ti)
    end
    return mod
end

"""
    step!(mod::AbstractModule, sim)

Main module logic for a single timestep.
"""
function step!(mod::AbstractModule, sim)
    return mod
end

"""
    step_state!(mod::AbstractModule, sim)

Disease state transitions (before transmission). Default no-op.
"""
function step_state!(mod::AbstractModule, sim)
    return mod
end

"""
    step_die!(mod::AbstractModule, uids::UIDs)

Handle state changes upon agent death. Default no-op.
"""
function step_die!(mod::AbstractModule, death_uids::UIDs)
    return mod
end

"""
    update_results!(mod::AbstractModule, sim)

Update module results for the current timestep.
"""
function update_results!(mod::AbstractModule, sim)
    md = module_data(mod)
    ti = md.t.ti

    # Auto-update BoolState counts
    for s in md.states
        if s isa StateVector{Bool}
            rkey = Symbol("n_", s.name)
            if haskey(md.results, rkey)
                md.results[rkey][ti] = Float64(length(uids(s)))
            end
        end
    end
    return mod
end

"""
    finish_step!(mod::AbstractModule, sim)

Post-step cleanup, advance module time.
"""
function finish_step!(mod::AbstractModule, sim)
    md = module_data(mod)
    advance!(md.t)
    return mod
end

"""
    finalize!(mod::AbstractModule)
    finalize!(mod::AbstractModule, sim)

Final cleanup after simulation completes. The two-argument form passes the
simulation for analyzers that need to collect data from other modules.
"""
function finalize!(mod::AbstractModule)
    return mod
end

function finalize!(mod::AbstractModule, sim)
    return finalize!(mod)
end

"""
    set_prognoses!(disease, target, source, sim)

Set disease prognoses for a newly infected agent. Called during transmission.
Custom diseases should extend this method.
"""
function set_prognoses! end

export init_pre!, init_post!, init_results!, start_step!, step!, step_state!,
       step_die!, update_results!, finish_step!, finalize!, set_prognoses!
