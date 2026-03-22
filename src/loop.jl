"""
Integration loop for Starsim.jl.

Mirrors Python starsim's `loop.py`. Orchestrates the 16-step per-timestep
module execution order.
"""

"""Type alias for loop step functions. Uses FunctionWrapper for type-stable dispatch."""
const StepFn = FunctionWrapper{Nothing, Tuple{Any}}

"""
    StepEntry

A single step in the integration loop: a function to call with the sim.
Uses `FunctionWrapper` instead of bare `Function` to avoid dynamic dispatch
overhead in the inner loop.
"""
struct StepEntry
    order::Int
    label::String
    fn::StepFn
end

"""
    Loop

The integration loop. Holds the ordered list of step entries and executes
them in sequence for each timestep.

The 16-step order mirrors Python starsim exactly:
1. sim.start_step
2. all modules.start_step
3. custom.step
4. demographics.step
5. diseases.step_state
6. connectors.step
7. networks.step
8. interventions.step
9. diseases.step (transmission)
10. people.step_die
11. people.update_results
12. all modules.update_results
13. analyzers.step
14. all modules.finish_step
15. people.finish_step
16. sim.finish_step
"""
mutable struct Loop
    steps::Vector{StepEntry}
    ti::Int
    npts::Int
end

Loop() = Loop(StepEntry[], 0, 0)

function Base.show(io::IO, l::Loop)
    print(io, "Loop($(length(l.steps)) steps, ti=$(l.ti)/$(l.npts))")
end

"""
    build_loop!(loop::Loop, sim)

Construct the integration loop from the simulation's modules.
Follows Python starsim's 16-step integration order exactly.
"""
function build_loop!(loop::Loop, sim)
    empty!(loop.steps)
    order = 0

    # 1. sim.start_step — reset per-timestep state
    order += 1
    push!(loop.steps, StepEntry(order, "sim.start_step",
        StepFn(s -> nothing)))

    # 2. all modules.start_step — jump distributions
    for (_, mod) in all_modules(sim)
        order += 1
        let m = mod
            push!(loop.steps, StepEntry(order, "$(module_name(m)).start_step",
                StepFn(s -> (start_step!(m, s); nothing))))
        end
    end

    # 3. extra_modules.step — generic modules (ODE, compartmental, etc.)
    for (_, mod) in sim.extra_modules
        order += 1
        let m = mod
            push!(loop.steps, StepEntry(order, "$(module_name(m)).step",
                StepFn(s -> (step!(m, s); nothing))))
        end
    end

    # 4. demographics.step — births, deaths
    for (_, demo) in sim.demographics
        order += 1
        let d = demo
            push!(loop.steps, StepEntry(order, "$(module_name(d)).step",
                StepFn(s -> (step!(d, s); nothing))))
        end
    end

    # 5. diseases.step_state — state transitions (before transmission)
    for (_, dis) in sim.diseases
        order += 1
        let d = dis
            push!(loop.steps, StepEntry(order, "$(module_name(d)).step_state",
                StepFn(s -> (step_state!(d, s); nothing))))
        end
    end

    # 6. connectors.step — cross-module interactions
    for (_, conn) in sim.connectors
        order += 1
        let c = conn
            push!(loop.steps, StepEntry(order, "$(module_name(c)).step",
                StepFn(s -> (step!(c, s); nothing))))
        end
    end

    # 7. networks.step — update edges
    for (_, net) in sim.networks
        order += 1
        let n = net
            push!(loop.steps, StepEntry(order, "$(module_name(n)).step",
                StepFn(s -> (step!(n, s); nothing))))
        end
    end

    # 8. interventions.step
    for (_, iv) in sim.interventions
        order += 1
        let i = iv
            push!(loop.steps, StepEntry(order, "$(module_name(i)).step",
                StepFn(s -> (step!(i, s); nothing))))
        end
    end

    # 9. diseases.step — transmission
    for (_, dis) in sim.diseases
        order += 1
        let d = dis
            push!(loop.steps, StepEntry(order, "$(module_name(d)).step",
                StepFn(s -> (step!(d, s); nothing))))
        end
    end

    # 10. people.step_die
    order += 1
    let disease_vec = collect(values(sim.diseases))
        push!(loop.steps, StepEntry(order, "people.step_die",
            StepFn(s -> (step_die!(s.people, s.loop.ti, disease_vec); nothing))))
    end

    # 11. people.update_results (placeholder)
    order += 1
    push!(loop.steps, StepEntry(order, "people.update_results",
        StepFn(s -> (update_people_results!(s.people, s.loop.ti, s.results); nothing))))

    # 12. all modules.update_results
    for (_, mod) in all_modules(sim)
        order += 1
        let m = mod
            push!(loop.steps, StepEntry(order, "$(module_name(m)).update_results",
                StepFn(s -> (update_results!(m, s); nothing))))
        end
    end

    # 13. analyzers.step
    for (_, ana) in sim.analyzers
        order += 1
        let a = ana
            push!(loop.steps, StepEntry(order, "$(module_name(a)).step",
                StepFn(s -> (step!(a, s); nothing))))
        end
    end

    # 14. all modules.finish_step
    for (_, mod) in all_modules(sim)
        order += 1
        let m = mod
            push!(loop.steps, StepEntry(order, "$(module_name(m)).finish_step",
                StepFn(s -> (finish_step!(m, s); nothing))))
        end
    end

    # 15. people.finish_step — remove dead, age
    order += 1
    push!(loop.steps, StepEntry(order, "people.finish_step",
        StepFn(s -> (finish_step!(s.people, s.pars.dt, s.pars.use_aging); nothing))))

    # 16. sim.finish_step — advance time
    order += 1
    push!(loop.steps, StepEntry(order, "sim.finish_step",
        StepFn(s -> (s.loop.ti += 1; nothing))))

    return loop
end

"""
    run_loop!(loop::Loop, sim; verbose::Int=1)

Execute the integration loop for all timesteps.
"""
function run_loop!(loop::Loop, sim; verbose::Int=1)
    loop.npts = sim.t.npts
    loop.ti = 1

    for ti in 1:loop.npts
        loop.ti = ti

        if verbose >= 2
            year = sim.pars.start + (ti - 1) * sim.pars.dt
            println("  Step $ti / $(loop.npts) (year=$(round(year, digits=2)))")
        end

        for entry in loop.steps
            entry.fn(sim)
        end
    end

    if verbose >= 1
        println("Simulation complete ($(loop.npts) steps)")
    end

    return loop
end

export Loop, StepEntry, build_loop!, run_loop!
