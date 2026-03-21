"""
STI-specific sexual networks.
Port of Python `stisim.networks.StructuredSexual` and related.

StructuredSexual provides:
- Risk groups (low/medium/high) with different contact rates
- Age-matched partnership formation
- Duration-based relationships
- Concurrency (multiple simultaneous partners)
"""

# ============================================================================
# StructuredSexual <: AbstractNetwork
# ============================================================================

"""
    StructuredSexual <: AbstractNetwork

Structured sexual network with risk groups, age-matching, and concurrency.

# Keyword arguments
- `name::Symbol` — network name (default `:structuredsexual`)
- `n_risk_groups::Int` — number of risk groups (default 3: low, medium, high)
- `risk_dist::Vector{Float64}` — distribution across risk groups (default [0.5, 0.3, 0.2])
- `contact_rates::Vector{Float64}` — mean partners per year by risk group (default [1.0, 3.0, 10.0])
- `mean_dur::Real` — mean partnership duration in years (default 2.0)
- `participation_rate::Real` — fraction of pop in partnerships (default 0.5)
- `age_lo::Real` — minimum age for sexual activity (default 15)
- `age_hi::Real` — maximum age for sexual activity (default 65)
- `age_diff_mean::Real` — mean M-F age difference (default 3.0)
- `age_diff_std::Real` — std of age difference (default 3.0)
- `concurrency::Real` — probability of concurrent partnerships (default 0.1)
"""
mutable struct StructuredSexual <: Starsim.AbstractNetwork
    data::Starsim.NetworkData

    # Risk group assignment (per-agent)
    risk_group::Starsim.StateVector{Int64, Vector{Int64}}

    # Parameters
    n_risk_groups::Int
    risk_dist::Vector{Float64}
    contact_rates::Vector{Float64}
    mean_dur::Float64
    participation_rate::Float64
    p_pair_form::Float64       # Per-woman probability of forming a partnership each dt (Python default 0.5)
    age_lo::Float64
    age_hi::Float64
    age_diff_mean::Float64
    age_diff_std::Float64
    concurrency::Float64
    acts_per_year::Float64     # Mean acts per partnership per year (Python default 80)
    acts_std::Float64          # Std of acts per year (Python default 30)

    # Partnership tracking
    partner_durations::Vector{Float64}
    partnership_start::Vector{Float64}

    rng::StableRNG
end

function StructuredSexual(;
    name::Symbol             = :structuredsexual,
    n_risk_groups::Int       = 3,
    risk_dist::Vector{Float64} = [0.5, 0.3, 0.2],
    contact_rates::Vector{Float64} = [1.0, 3.0, 10.0],
    mean_dur::Real           = 2.0,
    participation_rate::Real = 0.5,
    p_pair_form::Real        = 0.5,    # Python default: 0.5 per woman per dt
    age_lo::Real             = 15.0,
    age_hi::Real             = 65.0,
    age_diff_mean::Real      = 3.0,
    age_diff_std::Real       = 3.0,
    concurrency::Real        = 0.1,
    acts_per_year::Real      = 80.0,   # Python default: lognorm_ex(80, 30)
    acts_std::Real           = 30.0,
)
    md = Starsim.ModuleData(name; label="Structured sexual network")
    nd = Starsim.NetworkData(md, Starsim.Edges(), true)

    StructuredSexual(
        nd,
        Starsim.IntState(:risk_group; default=0, label="Risk group"),
        n_risk_groups, risk_dist, contact_rates,
        Float64(mean_dur), Float64(participation_rate),
        Float64(p_pair_form),
        Float64(age_lo), Float64(age_hi),
        Float64(age_diff_mean), Float64(age_diff_std),
        Float64(concurrency),
        Float64(acts_per_year), Float64(acts_std),
        Float64[], Float64[],
        StableRNG(0),
    )
end

Starsim.network_data(net::StructuredSexual) = net.data

function Starsim.init_pre!(net::StructuredSexual, sim)
    md = Starsim.module_data(net)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))

    Starsim.add_module_state!(sim.people, net.risk_group)

    md.initialized = true

    # Assign risk groups
    _assign_risk_groups!(net, sim.people)

    # Form initial partnerships (use full participation rate, not dt-scaled)
    _form_structured_partnerships!(net, sim; initial=true)

    return net
end

"""Assign risk groups based on risk_dist."""
function _assign_risk_groups!(net::StructuredSexual, people::Starsim.People)
    active = people.auids.values
    cumprob = cumsum(net.risk_dist)
    for u in active
        r = rand(net.rng)
        rg = 0
        for (i, cp) in enumerate(cumprob)
            if r <= cp
                rg = i - 1  # 0-indexed risk groups
                break
            end
        end
        net.risk_group.raw[u] = rg
    end
    return net
end

"""Form partnerships with age-matching and risk-group-based rates."""
function _form_structured_partnerships!(net::StructuredSexual, sim; initial::Bool=false)
    people = sim.people
    active = people.auids.values
    dt = sim.pars.dt
    age_raw = people.age.raw
    female_raw = people.female.raw
    edges = net.data.edges

    # Build set of agents already in partnerships (for concurrency control)
    partnered = Set{Int}()
    for i in 1:length(edges)
        push!(partnered, edges.p1[i])
        push!(partnered, edges.p2[i])
    end

    # Find eligible agents (age-appropriate and not already partnered or under concurrency)
    eligible_males = Int[]
    eligible_females = Int[]
    for u in active
        age = age_raw[u]
        if age >= net.age_lo && age <= net.age_hi
            if !(u in partnered) || (rand(net.rng) < net.concurrency)
                if female_raw[u]
                    push!(eligible_females, u)
                else
                    push!(eligible_males, u)
                end
            end
        end
    end

    isempty(eligible_males) && return net
    isempty(eligible_females) && return net

    # At init: create full steady-state partnerships; ongoing: turnover-based formation
    if initial
        n_pairs_target = max(1, Int(round(min(length(eligible_males), length(eligible_females)) * net.participation_rate)))
    else
        n_eligible_min = min(length(eligible_males), length(eligible_females))
        frac = net.participation_rate * dt / net.mean_dur
        n_pairs_target = max(1, Int(round(n_eligible_min * frac)))
    end

    # Shuffle and pair with age-matching
    m_perm = randperm(net.rng, length(eligible_males))
    f_perm = randperm(net.rng, length(eligible_females))

    new_p1 = Int[]
    new_p2 = Int[]
    new_dur = Float64[]
    new_acts = Float64[]
    used_females = Set{Int}()

    ti_f = initial ? 0.0 : Float64(Starsim.module_data(net).t.ti)

    for mi in 1:min(n_pairs_target, length(eligible_males))
        m = eligible_males[m_perm[mi]]
        m_age = age_raw[m]

        target_age = m_age - net.age_diff_mean + randn(net.rng) * net.age_diff_std
        best_f = 0
        best_diff = Inf

        n_candidates = min(20, length(eligible_females))
        for ci in 1:n_candidates
            fi = f_perm[((mi + ci - 2) % length(eligible_females)) + 1]
            f = eligible_females[fi]
            if f in used_females
                continue
            end
            diff = abs(age_raw[f] - target_age)
            if diff < best_diff
                best_diff = diff
                best_f = f
            end
        end

        if best_f > 0 && best_diff < 15.0
            push!(new_p1, m)
            push!(new_p2, best_f)
            push!(used_females, best_f)
            dur = max(dt, net.mean_dur + randn(net.rng) * net.mean_dur * 0.5)
            push!(new_dur, dur)
            acts_raw = max(1.0, net.acts_per_year + randn(net.rng) * net.acts_std)
            push!(new_acts, acts_raw)
        end
    end

    if !isempty(new_p1)
        n = length(new_p1)
        Starsim.add_edges!(net.data.edges, new_p1, new_p2,
                           ones(Float64, n), new_acts)
        append!(net.partner_durations, new_dur)
        append!(net.partnership_start, fill(ti_f, n))
    end

    return net
end

function Starsim.step!(net::StructuredSexual, sim)
    md = Starsim.module_data(net)
    ti = md.t.ti
    dt = sim.pars.dt
    ti_f = Float64(ti)

    # Dissolve expired partnerships
    if !isempty(net.partner_durations)
        keep = Int[]
        for i in 1:length(net.partner_durations)
            # Elapsed time in years = (current_ti - start_ti) * dt
            elapsed_years = (ti_f - net.partnership_start[i]) * dt
            if elapsed_years < net.partner_durations[i]
                push!(keep, i)
            end
        end

        if length(keep) < length(net.partner_durations)
            edges = net.data.edges
            edges.p1 = edges.p1[keep]
            edges.p2 = edges.p2[keep]
            edges.beta = edges.beta[keep]
            edges.acts = edges.acts[keep]
            net.partner_durations = net.partner_durations[keep]
            net.partnership_start = net.partnership_start[keep]
        end
    end

    # Form new partnerships to replace dissolved ones
    _form_structured_partnerships!(net, sim)

    return net
end
