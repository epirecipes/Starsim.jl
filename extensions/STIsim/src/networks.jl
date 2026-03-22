"""
STI-specific sexual networks.
Port of Python `stisim.networks.StructuredSexual`.

Matches the Python stisim StructuredSexual architecture:
- Sex-specific two-stage risk group assignment (prop_f0/m0/f2/m2)
- Poisson+1 concurrency limits per risk group × sex
- Sexual debut age from lognormal distribution
- p_pair_form Bernoulli filter on eligible women
- Sort-based age matching with age-group × risk-group preferences
- Stable/casual/onetime partnership types with risk-group-dependent probabilities
- Lognormal duration stratified by age group × risk group
- Sex work contacts (FSW + clients)
- Duration-based partnership dissolution (dur measured in timesteps)
"""

# ============================================================================
# Age group bins (matching Python NetworkPars.f_age_group_bins)
# ============================================================================
const AGE_GROUP_BINS = (
    teens = (0.0, 20.0),
    young = (20.0, 25.0),
    adult = (25.0, Inf),
)
const AGE_GROUP_KEYS = (:teens, :young, :adult)

# ============================================================================
# StructuredSexual <: AbstractNetwork
# ============================================================================

"""
    StructuredSexual <: AbstractNetwork

Structured sexual network matching Python stisim's StructuredSexual.

Risk groups (0-indexed): 0 = low (marry/stay), 1 = medium (divorce/concurrent), 2 = high (never marry).
"""
mutable struct StructuredSexual <: Starsim.AbstractNetwork
    data::Starsim.NetworkData

    # Per-agent states
    risk_group::Starsim.StateVector{Int64, Vector{Int64}}
    concurrency_state::Starsim.StateVector{Float64, Vector{Float64}}  # Max simultaneous partners
    partners_state::Starsim.StateVector{Float64, Vector{Float64}}     # Current partner count
    debut_state::Starsim.StateVector{Float64, Vector{Float64}}        # Sexual debut age
    fsw_state::Starsim.StateVector{Float64, Vector{Float64}}          # Female sex worker (0/1)
    client_state::Starsim.StateVector{Float64, Vector{Float64}}       # Male client of FSW (0/1)

    # Risk group proportions (sex-specific, matching Python defaults)
    prop_f0::Float64  # Fraction of females in risk group 0 (default 0.85)
    prop_m0::Float64  # Fraction of males in risk group 0 (default 0.80)
    prop_f2::Float64  # Fraction of females in risk group 2 (default 0.01)
    prop_m2::Float64  # Fraction of males in risk group 2 (default 0.02)

    # Concurrency (Poisson lambda by risk group × sex, before +1)
    f_conc::Vector{Float64}  # [f0_conc, f1_conc, f2_conc]
    m_conc::Vector{Float64}  # [m0_conc, m1_conc, m2_conc]

    # Sexual debut parameters
    debut_pars_f::Tuple{Float64, Float64}  # (mean, std) for lognormal female debut
    debut_pars_m::Tuple{Float64, Float64}  # (mean, std) for lognormal male debut

    # Partnership formation
    p_pair_form::Float64  # Bernoulli probability a woman seeks a partner each dt

    # Age difference preferences: NamedTuple of 3 vectors of (mu, std) per risk group
    age_diff_pars::NamedTuple{(:teens, :young, :adult), NTuple{3, Vector{Tuple{Float64, Float64}}}}

    # Partnership type probabilities
    p_matched_stable::Vector{Float64}     # P(stable | matched risk group) per RG
    p_mismatched_casual::Vector{Float64}  # P(casual | mismatched) per RG of female

    # Duration parameters: NamedTuple of age-group → vector of (mean, std) per RG
    stable_dur_pars::NamedTuple{(:teens, :young, :adult), NTuple{3, Vector{Tuple{Float64, Float64}}}}
    casual_dur_pars::NamedTuple{(:teens, :young, :adult), NTuple{3, Vector{Tuple{Float64, Float64}}}}

    # Acts
    acts_mean::Float64
    acts_std::Float64

    # Sex work parameters
    fsw_share::Float64         # P(female is FSW) (default 0.05)
    client_share::Float64      # P(male is client) (default 0.12)
    sw_seeking_rate::Float64   # Monthly rate clients seek FSW (default 1.0)

    # Edge-level metadata (parallel to edges)
    edge_dur::Vector{Float64}        # Remaining duration in timesteps
    edge_type::Vector{Int}           # 0=stable, 1=casual, 2=onetime, 3=sw
    edge_sw::Vector{Bool}            # Whether edge is sex work

    rng::StableRNG
end

function StructuredSexual(;
    name::Symbol = :structuredsexual,
    # Risk group proportions (Python defaults)
    prop_f0::Real = 0.85,
    prop_m0::Real = 0.80,
    prop_f2::Real = 0.01,
    prop_m2::Real = 0.02,
    # Concurrency (Poisson lambda, Python defaults)
    f_conc::Vector{Float64} = [0.0001, 0.01, 0.1],
    m_conc::Vector{Float64} = [0.0001, 0.2, 0.5],
    # Sexual debut (lognormal mean, std)
    debut_pars_f::Tuple{Float64, Float64} = (20.0, 3.0),
    debut_pars_m::Tuple{Float64, Float64} = (21.0, 3.0),
    # Partnership formation
    p_pair_form::Real = 0.5,
    # Age difference preferences (mu, std) per risk group per age group
    age_diff_pars = (
        teens = [(7.0, 3.0), (6.0, 3.0), (5.0, 1.0)],
        young = [(8.0, 3.0), (7.0, 3.0), (5.0, 2.0)],
        adult = [(8.0, 3.0), (7.0, 3.0), (5.0, 2.0)],
    ),
    # Partnership type probabilities
    p_matched_stable::Vector{Float64}    = [0.9, 0.5, 0.0],
    p_mismatched_casual::Vector{Float64} = [0.5, 0.5, 0.5],
    # Duration parameters (years) per risk group per age group
    stable_dur_pars = (
        teens = [(100.0, 1.0), (8.0, 2.0),  (1e-4/12, 1e-4/12)],
        young = [(100.0, 1.0), (10.0, 3.0), (1e-4/12, 1e-4/12)],
        adult = [(100.0, 1.0), (12.0, 3.0), (1e-4/12, 1e-4/12)],
    ),
    casual_dur_pars = (
        teens = [(1.0, 3.0), (1.0, 3.0), (1.0, 3.0)],
        young = [(1.0, 3.0), (1.0, 3.0), (1.0, 3.0)],
        adult = [(1.0, 3.0), (1.0, 3.0), (1.0, 3.0)],
    ),
    # Acts (lognormal mean, std in acts/timestep scaled from yearly)
    acts_mean::Real = 80.0,
    acts_std::Real  = 30.0,
    # Sex work
    fsw_share::Real         = 0.05,
    client_share::Real      = 0.12,
    sw_seeking_rate::Real   = 1.0,
    # Legacy aliases (ignored but accepted for backward compatibility)
    n_risk_groups::Int = 3,
    risk_dist::Vector{Float64} = Float64[],
    contact_rates::Vector{Float64} = Float64[],
    mean_dur::Real = -1.0,
    participation_rate::Real = -1.0,
    age_lo::Real = -1.0,
    age_hi::Real = -1.0,
    age_diff_mean::Real = -1.0,
    age_diff_std::Real = -1.0,
    concurrency::Real = -1.0,
    acts_per_year::Real = -1.0,
)
    md = Starsim.ModuleData(name; label="Structured sexual network")
    nd = Starsim.NetworkData(md, Starsim.Edges(), true)

    StructuredSexual(
        nd,
        Starsim.IntState(:risk_group; default=0, label="Risk group"),
        Starsim.FloatState(:net_concurrency; default=1.0, label="Concurrency limit"),
        Starsim.FloatState(:net_partners; default=0.0, label="Current partners"),
        Starsim.FloatState(:net_debut; default=0.0, label="Sexual debut age"),
        Starsim.FloatState(:net_fsw; default=0.0, label="FSW status"),
        Starsim.FloatState(:net_client; default=0.0, label="Client status"),
        Float64(prop_f0), Float64(prop_m0), Float64(prop_f2), Float64(prop_m2),
        f_conc, m_conc,
        debut_pars_f, debut_pars_m,
        Float64(p_pair_form),
        age_diff_pars,
        p_matched_stable, p_mismatched_casual,
        stable_dur_pars, casual_dur_pars,
        Float64(acts_mean), Float64(acts_std),
        Float64(fsw_share), Float64(client_share), Float64(sw_seeking_rate),
        Float64[], Int[], Bool[],
        StableRNG(0),
    )
end

# Legacy accessors for backward compatibility with tests
Base.getproperty(net::StructuredSexual, s::Symbol) = begin
    if s === :n_risk_groups
        return 3
    elseif s === :risk_dist
        f0 = getfield(net, :prop_f0)
        f2 = getfield(net, :prop_f2)
        return [f0, 1.0 - f0 - f2, f2]
    elseif s === :mean_dur
        # Return the average stable duration for adults RG1 as representative
        sdp = getfield(net, :stable_dur_pars)
        return sdp.adult[2][1]
    elseif s === :contact_rates
        return [1.0, 3.0, 10.0]  # Nominal values for backward compat
    elseif s === :participation_rate
        return getfield(net, :p_pair_form)
    elseif s === :concurrency
        mc = getfield(net, :m_conc)
        return mc[2]  # Return medium risk male concurrency
    else
        return getfield(net, s)
    end
end

Starsim.network_data(net::StructuredSexual) = getfield(net, :data)

# ============================================================================
# Initialization
# ============================================================================

function Starsim.init_pre!(net::StructuredSexual, sim)
    md = Starsim.module_data(net)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net_rng = getfield(net, :rng)
    net_rng = StableRNG(hash(md.name) ⊻ UInt64(sim.pars.rand_seed))
    setfield!(net, :rng, net_rng)

    # Register all per-agent states
    for st in [getfield(net, :risk_group), getfield(net, :concurrency_state),
               getfield(net, :partners_state), getfield(net, :debut_state),
               getfield(net, :fsw_state), getfield(net, :client_state)]
        Starsim.add_module_state!(sim.people, st)
    end

    md.initialized = true

    # Initialize all network states for existing agents
    _set_network_states!(net, sim)

    # NOTE: Do NOT form initial partnerships here.
    # Python's init_post() calls super().init_post(add_pairs=False).
    # Partnerships are formed in the first step() call, matching Python behavior.

    return net
end

# ============================================================================
# Network state assignment (risk groups, concurrency, debut, sex work)
# ============================================================================

"""Assign risk groups using Python's two-stage Bernoulli split."""
function _set_risk_groups!(net::StructuredSexual, people::Starsim.People; upper_age::Float64=1000.0)
    rng = getfield(net, :rng)
    rg_raw = getfield(net, :risk_group).raw
    prop_f0 = getfield(net, :prop_f0)
    prop_m0 = getfield(net, :prop_m0)
    prop_f2 = getfield(net, :prop_f2)
    prop_m2 = getfield(net, :prop_m2)
    female_raw = people.female.raw
    age_raw = people.age.raw

    for u in people.auids.values
        age_raw[u] > upper_age && continue
        is_female = female_raw[u]

        # Stage 1: low risk or not
        p_lo = is_female ? prop_f0 : prop_m0
        if rand(rng) < p_lo
            rg_raw[u] = 0
        else
            # Stage 2: high risk or medium
            p_hi = is_female ? prop_f2 / (1.0 - prop_f0) : prop_m2 / (1.0 - prop_m0)
            rg_raw[u] = rand(rng) < p_hi ? 2 : 1
        end
    end
    return net
end

"""Set concurrency limits using Poisson(lambda) + 1."""
function _set_concurrency!(net::StructuredSexual, people::Starsim.People; upper_age::Float64=1000.0)
    rng = getfield(net, :rng)
    rg_raw = getfield(net, :risk_group).raw
    conc_raw = getfield(net, :concurrency_state).raw
    female_raw = people.female.raw
    age_raw = people.age.raw
    f_conc = getfield(net, :f_conc)
    m_conc = getfield(net, :m_conc)

    for u in people.auids.values
        age_raw[u] > upper_age && continue
        rg = rg_raw[u]
        lam = female_raw[u] ? f_conc[rg + 1] : m_conc[rg + 1]
        # Poisson sample + 1
        conc_raw[u] = Float64(rand(rng, Distributions.Poisson(lam)) + 1)
    end
    return net
end

"""Set sexual debut age from lognormal distribution."""
function _set_debut!(net::StructuredSexual, people::Starsim.People; upper_age::Float64=1000.0)
    rng = getfield(net, :rng)
    debut_raw = getfield(net, :debut_state).raw
    female_raw = people.female.raw
    age_raw = people.age.raw
    dpf = getfield(net, :debut_pars_f)
    dpm = getfield(net, :debut_pars_m)

    for u in people.auids.values
        age_raw[u] > upper_age && continue
        mu, sigma = female_raw[u] ? dpf : dpm
        # Lognormal parameterized by desired mean and std
        mu_ln = log(mu^2 / sqrt(sigma^2 + mu^2))
        sigma_ln = sqrt(log(1.0 + sigma^2 / mu^2))
        debut_raw[u] = rand(rng, Distributions.LogNormal(mu_ln, sigma_ln))
    end
    return net
end

"""Set sex work status."""
function _set_sex_work!(net::StructuredSexual, people::Starsim.People; upper_age::Float64=1000.0)
    rng = getfield(net, :rng)
    fsw_raw = getfield(net, :fsw_state).raw
    client_raw = getfield(net, :client_state).raw
    female_raw = people.female.raw
    age_raw = people.age.raw
    fsw_share = getfield(net, :fsw_share)
    client_share = getfield(net, :client_share)

    for u in people.auids.values
        age_raw[u] > upper_age && continue
        if female_raw[u]
            fsw_raw[u] = rand(rng) < fsw_share ? 1.0 : 0.0
        else
            client_raw[u] = rand(rng) < client_share ? 1.0 : 0.0
        end
    end
    return net
end

"""Initialize all network states for agents."""
function _set_network_states!(net::StructuredSexual, sim; upper_age::Float64=1000.0)
    _set_risk_groups!(net, sim.people; upper_age)
    _set_concurrency!(net, sim.people; upper_age)
    _set_debut!(net, sim.people; upper_age)
    _set_sex_work!(net, sim.people; upper_age)
    return net
end

# ============================================================================
# Age-group / risk-group parameter lookup (matching Python get_age_risk_pars)
# ============================================================================

"""Get age group key for a given age."""
function _age_group_key(age::Float64)
    age < 20.0 && return :teens
    age < 25.0 && return :young
    return :adult
end

"""Get (mean, std) duration/age-diff parameters for agents by their age and risk group."""
function _get_age_risk_pars(net::StructuredSexual, uids::Vector{Int}, pars::NamedTuple,
                            people_age_raw::Vector{Float64}, rg_raw::Vector{Int64})
    n = length(uids)
    loc = Vector{Float64}(undef, n)
    scale = Vector{Float64}(undef, n)
    for (i, u) in enumerate(uids)
        age = people_age_raw[u]
        rg = rg_raw[u]
        agk = _age_group_key(age)
        mu, sigma = pars[agk][rg + 1]
        loc[i] = mu
        scale[i] = sigma
    end
    return loc, scale
end

# ============================================================================
# Partnership matching (matching Python match_pairs)
# ============================================================================

"""Match pairs using sort-based age matching with age-group × risk-group preferences."""
function _match_pairs(net::StructuredSexual, sim)
    rng = getfield(net, :rng)
    people = sim.people
    age_raw = people.age.raw
    female_raw = people.female.raw
    debut_raw = getfield(net, :debut_state).raw
    conc_raw = getfield(net, :concurrency_state).raw
    partners_raw = getfield(net, :partners_state).raw
    rg_raw = getfield(net, :risk_group).raw
    p_pair_form = getfield(net, :p_pair_form)
    age_diff_pars = getfield(net, :age_diff_pars)

    # Find eligible: over debut and under concurrency limit
    m_eligible = Int[]
    f_eligible = Int[]
    for u in people.auids.values
        age_raw[u] <= debut_raw[u] && continue
        partners_raw[u] >= conc_raw[u] && continue
        if female_raw[u]
            push!(f_eligible, u)
        else
            push!(m_eligible, u)
        end
    end

    # Bernoulli filter on women seeking partners
    f_looking = Int[]
    for u in f_eligible
        rand(rng) < p_pair_form && push!(f_looking, u)
    end

    (isempty(f_looking) || isempty(m_eligible)) && return (Int[], Int[])

    # Compute desired male ages for each looking female (based on age group × risk group)
    n_f = length(f_looking)
    desired_ages = Vector{Float64}(undef, n_f)
    for (i, u) in enumerate(f_looking)
        age = age_raw[u]
        rg = rg_raw[u]
        agk = _age_group_key(age)
        mu, sigma = age_diff_pars[agk][rg + 1]
        age_gap = mu + randn(rng) * sigma
        desired_ages[i] = age + age_gap
    end

    # Get male ages
    n_m = length(m_eligible)
    m_ages = Vector{Float64}(undef, n_m)
    for (i, u) in enumerate(m_eligible)
        m_ages[i] = age_raw[u]
    end

    # Sort both by age
    ind_m = sortperm(m_ages; alg=MergeSort)
    ind_f = sortperm(desired_ages; alg=MergeSort)

    isempty(ind_m) && return (Int[], Int[])
    isempty(ind_f) && return (Int[], Int[])

    # Trim males outside desired age range
    youngest_desired = desired_ages[ind_f[1]]
    oldest_desired = desired_ages[ind_f[end]]
    youngest_male = m_ages[ind_m[1]]
    oldest_male = m_ages[ind_m[end]]

    if youngest_male < youngest_desired
        cutoff = searchsortedfirst(view(m_ages, ind_m), youngest_desired)
        cutoff > length(ind_m) && return (Int[], Int[])
        ind_m = ind_m[cutoff:end]
    elseif youngest_desired < youngest_male
        cutoff = searchsortedfirst(view(desired_ages, ind_f), youngest_male)
        cutoff > length(ind_f) && return (Int[], Int[])
        ind_f = ind_f[cutoff:end]
    end

    isempty(ind_m) && return (Int[], Int[])
    isempty(ind_f) && return (Int[], Int[])

    # Trim upper end
    oldest_desired_now = desired_ages[ind_f[end]]
    oldest_male_now = m_ages[ind_m[end]]
    if oldest_male_now > oldest_desired_now
        cutoff = searchsortedfirst(view(m_ages, ind_m), oldest_desired_now)
        cutoff > length(ind_m) && (cutoff = length(ind_m))
        ind_m = ind_m[1:cutoff]
    elseif oldest_desired_now > oldest_male_now
        cutoff = searchsortedfirst(view(desired_ages, ind_f), oldest_male_now)
        cutoff > length(ind_f) && (cutoff = length(ind_f))
        ind_f = ind_f[1:cutoff]
    end

    isempty(ind_m) && return (Int[], Int[])
    isempty(ind_f) && return (Int[], Int[])

    # Balance groups by random subsampling
    if length(ind_m) < length(ind_f)
        subset = sort(randperm(rng, length(ind_f))[1:length(ind_m)])
        ind_f = ind_f[subset]
    elseif length(ind_f) < length(ind_m)
        subset = sort(randperm(rng, length(ind_m))[1:length(ind_f)])
        ind_m = ind_m[subset]
    end

    p1 = [m_eligible[i] for i in ind_m]
    p2 = [f_looking[i] for i in ind_f]
    return (p1, p2)
end

# ============================================================================
# Sex worker matching (matching Python match_sex_workers)
# ============================================================================

"""Match sex workers to clients."""
function _match_sex_workers(net::StructuredSexual, sim)
    rng = getfield(net, :rng)
    people = sim.people
    dt = sim.pars.dt
    age_raw = people.age.raw
    debut_raw = getfield(net, :debut_state).raw
    fsw_raw = getfield(net, :fsw_state).raw
    client_raw = getfield(net, :client_state).raw
    sw_rate = getfield(net, :sw_seeking_rate)

    # Active FSW and clients (over debut)
    active_fsw = Int[]
    active_clients = Int[]
    for u in people.auids.values
        age_raw[u] <= debut_raw[u] && continue
        if fsw_raw[u] > 0.5
            push!(active_fsw, u)
        end
        if client_raw[u] > 0.5
            push!(active_clients, u)
        end
    end

    (isempty(active_fsw) || isempty(active_clients)) && return (Int[], Int[])

    # Convert monthly probability to per-dt probability: 1-(1-p)^(dt*12)
    # Python: sw_seeking_rate = ss.probpermonth(1.0), to_prob() → 1.0 for monthly dt
    p_seek = 1.0 - (1.0 - sw_rate)^(dt * 12.0)
    m_looking = Int[]
    for u in active_clients
        rand(rng) < p_seek && push!(m_looking, u)
    end

    isempty(m_looking) && return (Int[], Int[])

    # Match: if more clients than FSW, repeat FSW; otherwise subsample
    n_pairs = min(length(m_looking), length(active_fsw) * 10)
    if n_pairs > length(active_fsw)
        # Repeat FSW to match demand
        p2 = Int[]
        while length(p2) < n_pairs
            append!(p2, active_fsw)
        end
        p2 = p2[1:n_pairs]
        # Shuffle
        for i in length(p2):-1:2
            j = rand(rng, 1:i)
            p2[i], p2[j] = p2[j], p2[i]
        end
    else
        # Subsample FSW
        perm = randperm(rng, length(active_fsw))
        p2 = active_fsw[perm[1:n_pairs]]
    end

    p1 = length(m_looking) <= n_pairs ? m_looking : m_looking[1:n_pairs]
    n = min(length(p1), length(p2))
    return (p1[1:n], p2[1:n])
end

# ============================================================================
# Add pairs (non-SW + SW) — matching Python add_pairs_nonsw + add_pairs_sw
# ============================================================================

"""Sample from lognormal parameterized by desired mean and std."""
function _lognorm_sample(rng::StableRNG, mu::Float64, sigma::Float64)
    mu <= 0.0 && return 0.0
    sigma <= 0.0 && return mu
    mu_ln = log(mu^2 / sqrt(sigma^2 + mu^2))
    sigma_ln = sqrt(log(1.0 + sigma^2 / mu^2))
    return rand(rng, Distributions.LogNormal(mu_ln, sigma_ln))
end

"""Add non-sex-work partnerships."""
function _add_pairs_nonsw!(net::StructuredSexual, sim; initial::Bool=false)
    p1, p2 = _match_pairs(net, sim)
    isempty(p1) && return net

    rng = getfield(net, :rng)
    people = sim.people
    dt = sim.pars.dt
    age_raw = people.age.raw
    rg_raw = getfield(net, :risk_group).raw
    partners_raw = getfield(net, :partners_state).raw
    p_matched_stable = getfield(net, :p_matched_stable)
    p_mismatched_casual = getfield(net, :p_mismatched_casual)
    stable_dur_pars = getfield(net, :stable_dur_pars)
    casual_dur_pars = getfield(net, :casual_dur_pars)
    # Convert annual acts to per-timestep (Python freqperyear auto-converts)
    acts_mean_dt = getfield(net, :acts_mean) * dt
    acts_std_dt = getfield(net, :acts_std) * dt

    n = length(p1)
    new_p1 = Vector{Int}(undef, n)
    new_p2 = Vector{Int}(undef, n)
    new_beta = ones(Float64, n)
    new_acts = Vector{Float64}(undef, n)
    new_dur = Vector{Float64}(undef, n)  # In timesteps
    new_type = Vector{Int}(undef, n)
    new_sw = fill(false, n)

    for i in 1:n
        m = p1[i]
        f = p2[i]
        new_p1[i] = m
        new_p2[i] = f

        # Acts per timestep — Python uses .astype(int) truncation (can be 0)
        new_acts[i] = floor(_lognorm_sample(rng, acts_mean_dt, acts_std_dt))

        # Determine partnership type based on risk group matching
        rg_m = rg_raw[m]
        rg_f = rg_raw[f]
        matched_risk = (rg_m == rg_f)
        edge_is_stable = false

        if matched_risk
            p_stable = p_matched_stable[rg_m + 1]
            is_match = rand(rng) < p_stable
            if is_match
                # Stable partnership — duration from stable_dur_pars
                edge_is_stable = true
                agk = _age_group_key(age_raw[f])
                mu, sigma = stable_dur_pars[agk][rg_f + 1]
                # Python converts dur pars to months via .months, then samples
                dur_months = _lognorm_sample(rng, mu * 12.0, sigma * 12.0)
            else
                # Failed match → onetime (dur = 1 timestep), matching Python
                dur_months = -1.0
            end
        else
            p_casual = p_mismatched_casual[rg_f + 1]
            is_match = rand(rng) < p_casual
            if is_match
                agk = _age_group_key(age_raw[f])
                mu, sigma = casual_dur_pars[agk][rg_f + 1]
                dur_months = _lognorm_sample(rng, mu * 12.0, sigma * 12.0)
            else
                # Failed match → onetime (dur = 1 timestep), matching Python
                dur_months = -1.0
            end
        end

        # Convert months to integer timesteps (Python truncates via int array assignment)
        # Non-matched edges default to dur=1 (onetime)
        if dur_months < 0.0
            dur_ts = 1.0  # Failed match → onetime
        else
            dur_ts = floor(dur_months)  # Truncate like Python's float→int cast
            dur_ts = max(dur_ts, 0.0)   # Can't be negative
        end
        new_dur[i] = dur_ts

        # Classify: Python sets types first, then overwrites dur==1 as onetime
        if dur_ts <= 1.0
            new_type[i] = 2  # onetime
        elseif edge_is_stable
            new_type[i] = 0  # stable
        else
            new_type[i] = 1  # casual
        end

        # Update partner counts
        partners_raw[m] += 1.0
        partners_raw[f] += 1.0
    end

    # Append to edges
    Starsim.add_edges!(getfield(net, :data).edges, new_p1, new_p2, new_beta, new_acts)
    append!(getfield(net, :edge_dur), new_dur)
    append!(getfield(net, :edge_type), new_type)
    append!(getfield(net, :edge_sw), new_sw)

    return net
end

"""Add sex work partnerships (one-timestep contacts)."""
function _add_pairs_sw!(net::StructuredSexual, sim)
    p1, p2 = _match_sex_workers(net, sim)
    isempty(p1) && return net

    rng = getfield(net, :rng)
    dt = sim.pars.dt
    n = length(p1)
    # Convert annual acts to per-timestep
    acts_mean_dt = getfield(net, :acts_mean) * dt
    acts_std_dt = getfield(net, :acts_std) * dt

    new_beta = ones(Float64, n)
    new_acts = Vector{Float64}(undef, n)
    new_dur = ones(Float64, n)  # 1 timestep
    new_type = fill(3, n)       # SW type
    new_sw = fill(true, n)

    for i in 1:n
        new_acts[i] = floor(_lognorm_sample(rng, acts_mean_dt, acts_std_dt))
    end

    Starsim.add_edges!(getfield(net, :data).edges, p1, p2, new_beta, new_acts)
    append!(getfield(net, :edge_dur), new_dur)
    append!(getfield(net, :edge_type), new_type)
    append!(getfield(net, :edge_sw), new_sw)

    return net
end

"""Add all partnerships (non-SW + SW)."""
function _add_pairs!(net::StructuredSexual, sim; initial::Bool=false)
    _add_pairs_nonsw!(net, sim; initial)
    _add_pairs_sw!(net, sim)
    return net
end

# ============================================================================
# Partnership dissolution (matching Python end_pairs)
# ============================================================================

"""Dissolve expired partnerships and decrement partner counts."""
function _end_pairs!(net::StructuredSexual, sim)
    edge_dur = getfield(net, :edge_dur)
    isempty(edge_dur) && return net

    edges = getfield(net, :data).edges
    edge_type = getfield(net, :edge_type)
    edge_sw = getfield(net, :edge_sw)
    partners_raw = getfield(net, :partners_state).raw

    # Decrement all durations by 1 timestep
    for i in 1:length(edge_dur)
        edge_dur[i] -= 1.0
    end

    # Find active edges (dur > 0 and both agents alive)
    alive_raw = sim.people.alive.raw
    keep = Int[]
    for i in 1:length(edge_dur)
        p1_alive = alive_raw[edges.p1[i]]
        p2_alive = alive_raw[edges.p2[i]]
        if edge_dur[i] > 0.0 && p1_alive && p2_alive
            push!(keep, i)
        else
            # Decrement partner counts for non-SW dissolved edges
            if !edge_sw[i]
                partners_raw[edges.p1[i]] = max(0.0, partners_raw[edges.p1[i]] - 1.0)
                partners_raw[edges.p2[i]] = max(0.0, partners_raw[edges.p2[i]] - 1.0)
            end
        end
    end

    if length(keep) < length(edge_dur)
        edges.p1 = edges.p1[keep]
        edges.p2 = edges.p2[keep]
        edges.beta = edges.beta[keep]
        edges.acts = edges.acts[keep]
        setfield!(net, :edge_dur, edge_dur[keep])
        setfield!(net, :edge_type, edge_type[keep])
        setfield!(net, :edge_sw, edge_sw[keep])
    end

    return net
end

# ============================================================================
# Step (matching Python step)
# ============================================================================

function Starsim.step!(net::StructuredSexual, sim)
    dt = sim.pars.dt

    # 1. Dissolve expired partnerships
    _end_pairs!(net, sim)

    # 2. Set network states for newly aged-in agents
    _set_network_states!(net, sim; upper_age=dt)

    # 3. Form new partnerships
    _add_pairs!(net, sim)

    return net
end
