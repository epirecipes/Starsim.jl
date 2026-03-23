"""
HPV-specific persistent partnership network.

Faithfully mirrors Python hpvsim's partnership model:
- Partnerships persist for a drawn duration (not reformed every timestep)
- Two layers: marital (m) and casual (c) with different parameters
- Per-partnership acts drawn at formation, age-scaled each timestep
- Dissolution when duration expires or partner dies
- New partnerships formed only for underpartnered agents
- Age-assortative mixing via mixing matrices
"""

using Distributions: NegativeBinomial, LogNormal, Poisson

# ============================================================================
# Per-partnership data (persistent edge storage)
# ============================================================================

"""Storage for persistent partnerships within one network layer."""
mutable struct PartnershipLayer
    p1::Vector{Int}              # Male UIDs
    p2::Vector{Int}              # Female UIDs
    acts::Vector{Float64}        # Base acts per year (drawn at formation)
    dur::Vector{Float64}         # Partnership duration in years
    start_time::Vector{Float64}  # Start time (years)
    end_time::Vector{Float64}    # End time = start + dur
end

PartnershipLayer() = PartnershipLayer(Int[], Int[], Float64[], Float64[], Float64[], Float64[])

Base.length(pl::PartnershipLayer) = length(pl.p1)
Base.isempty(pl::PartnershipLayer) = isempty(pl.p1)

function remove_indices!(pl::PartnershipLayer, mask::BitVector)
    keep = .!mask
    pl.p1 = pl.p1[keep]
    pl.p2 = pl.p2[keep]
    pl.acts = pl.acts[keep]
    pl.dur = pl.dur[keep]
    pl.start_time = pl.start_time[keep]
    pl.end_time = pl.end_time[keep]
    return pl
end

# ============================================================================
# HPVSexualNet — persistent multi-layer sexual network
# ============================================================================

"""
    HPVSexualNet <: Starsim.AbstractNetwork

Persistent partnership sexual network matching Python hpvsim structure.

Each layer (marital, casual) maintains edge lists with per-partnership acts
drawn at formation from NegBinomial, durations drawn from NegBin/LogNormal,
and age-scaled acts each timestep.

# Parameters (matching Python hpvsim/parameters.py)
- Marital: acts~NB(80,40)/yr, dur~NB(80,3) yrs, condoms=0.01
- Casual:  acts~NB(50,5)/yr,  dur~LN(1,2) yrs, condoms=0.20
- Condom efficacy: 0.5
"""
mutable struct HPVSexualNet <: Starsim.AbstractNetwork
    data::Starsim.NetworkData
    layers::Dict{Symbol, PartnershipLayer}

    # Per-layer distribution parameters
    acts_par1::Dict{Symbol, Float64}
    acts_par2::Dict{Symbol, Float64}
    dur_par1::Dict{Symbol, Float64}
    dur_par2::Dict{Symbol, Float64}
    dur_dist_type::Dict{Symbol, Symbol}    # :neg_binomial or :lognormal
    condom_usage::Dict{Symbol, Float64}
    condom_efficacy::Float64

    # Age-dependent act scaling (piecewise linear)
    age_act_peak::Dict{Symbol, Float64}
    age_act_retirement::Dict{Symbol, Float64}
    age_act_debut_ratio::Dict{Symbol, Float64}
    age_act_retirement_ratio::Dict{Symbol, Float64}

    # Mixing and participation
    mixing_matrices::Dict{Symbol, Matrix{Float64}}
    age_bins::Vector{Float64}
    female_participation::Dict{Symbol, Vector{Float64}}
    male_participation::Dict{Symbol, Vector{Float64}}

    # Per-agent concurrency tracking (indexed by UID)
    current_partners::Dict{Symbol, Vector{Int}}
    desired_partners::Dict{Symbol, Vector{Int}}

    # Cross-layer concurrency limits (matching Python hpvsim)
    f_cross_layer::Float64   # P(female with other-layer partner can seek in this layer)
    m_cross_layer::Float64   # P(male with other-layer partner can seek in this layer)

    # Per-agent debut ages (drawn from sex-specific Normal distributions)
    debut_ages::Vector{Float64}
    debut_f_mean::Float64    # Female debut: Normal(15.0, 2.1)
    debut_f_std::Float64
    debut_m_mean::Float64    # Male debut: Normal(17.6, 1.8)
    debut_m_std::Float64

    initial_partnerships_formed::Bool  # Lazy init flag
    _last_initialized_uid::Int         # Track which UIDs have been initialized
    _edge_condom_factors::Vector{Float64}  # Per-edge condom factor, rebuilt each step
    rng::StableRNG
end

function HPVSexualNet(;
    name::Symbol = :sexual,
    condom_efficacy::Real = 0.5,
    debut_f_mean::Real = 15.0,
    debut_f_std::Real = 2.1,
    debut_m_mean::Real = 17.6,
    debut_m_std::Real = 1.8,
)
    md = Starsim.ModuleData(name; label="HPV sexual network")
    nd = Starsim.NetworkData(md, Starsim.Edges(), true)

    lks = [:m, :c]
    acts_par1 = Dict(:m => 80.0, :c => 50.0)
    acts_par2 = Dict(:m => 40.0, :c => 5.0)
    dur_par1  = Dict(:m => 80.0, :c => 1.0)
    dur_par2  = Dict(:m => 3.0,  :c => 2.0)
    dur_dist  = Dict(:m => :neg_binomial, :c => :lognormal)
    condoms   = Dict(:m => 0.01, :c => 0.20)

    peak    = Dict(:m => 30.0, :c => 25.0)
    retire  = Dict(:m => 100.0, :c => 100.0)
    d_ratio = Dict(:m => 0.5, :c => 0.5)
    r_ratio = Dict(:m => 0.1, :c => 0.1)

    mm = Dict(:m => MARITAL_MIXING_MATRIX, :c => CASUAL_MIXING_MATRIX)
    mp = default_layer_probs(network_type=:marital)
    cp = default_layer_probs(network_type=:casual)
    fp = Dict(:m => mp[:female], :c => cp[:female])
    mlp = Dict(:m => mp[:male],   :c => cp[:male])

    HPVSexualNet(
        nd,
        Dict(lk => PartnershipLayer() for lk in lks),
        acts_par1, acts_par2, dur_par1, dur_par2, dur_dist, condoms,
        Float64(condom_efficacy),
        peak, retire, d_ratio, r_ratio,
        mm, AGE_MIXING_BINS, fp, mlp,
        Dict(lk => Int[] for lk in lks),
        Dict(lk => Int[] for lk in lks),
        0.05,   # f_cross_layer (Python default)
        0.30,   # m_cross_layer (Python default)
        Float64[],  # debut_ages (populated at init)
        Float64(debut_f_mean),
        Float64(debut_f_std),
        Float64(debut_m_mean),
        Float64(debut_m_std),
        false,  # initial_partnerships_formed
        0,      # _last_initialized_uid
        Float64[],  # _edge_condom_factors
        StableRNG(0),
    )
end

Starsim.network_data(net::HPVSexualNet) = net.data

# ============================================================================
# Lifecycle
# ============================================================================

function Starsim.init_pre!(net::HPVSexualNet, sim)
    md = Starsim.module_data(net)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)

    n = sim.pars.n_agents * 2
    for lk in keys(net.layers)
        net.current_partners[lk] = zeros(Int, n)
        net.desired_partners[lk] = zeros(Int, n)
    end

    # Draw per-agent debut ages from sex-specific Normal distributions (matching Python)
    # Note: ages may be reassigned in init_post! by demographics (HPVDeaths),
    # but debut_ages depend only on sex which doesn't change.
    net.debut_ages = zeros(Float64, n)
    rng = net.rng
    for u in sim.people.auids.values
        u > n && continue
        if sim.people.female.raw[u]
            net.debut_ages[u] = net.debut_f_mean + net.debut_f_std * randn(rng)
        else
            net.debut_ages[u] = net.debut_m_mean + net.debut_m_std * randn(rng)
        end
    end

    _init_desired_partners!(net, sim.people, sim.people.auids.values)

    net._last_initialized_uid = sim.people.next_uid - 1
    md.initialized = true

    # Don't form partnerships here — ages may be reassigned by demographics init_post!
    # Partnerships will form at the first step! call when ages are correct.
    return net
end

# Fixed layer ordering: marital first, then casual (matching Python hpvsim)
const LAYER_ORDER = [:m, :c]

function Starsim.step!(net::HPVSexualNet, sim)
    now = Starsim.now(sim.t)
    dt = sim.pars.dt

    # Ensure arrays are large enough for any newly born agents and init their properties
    _grow_for_new_agents!(net, sim.people)

    if !net.initial_partnerships_formed
        # Python hpvsim creates partnerships during make_people() and again at step 0.
        # Julia forms once here, then the normal dissolve+create cycle handles subsequent steps.
        for lk in LAYER_ORDER
            haskey(net.layers, lk) || continue
            _form_new_partnerships!(net, lk, sim.people, now, dt)
        end
        _rebuild_edges!(net, sim.people)
        net.initial_partnerships_formed = true
        if get(ENV, "HPVSIM_DEBUG_NET", "") == "1"
            _debug_count(net, sim.people, "INIT", now)
        end
        return net
    end

    # Match Python: dissolve ALL layers first, then form ALL layers.
    # Cross-layer eligibility depends on partnership state at formation time.
    for lk in LAYER_ORDER
        haskey(net.layers, lk) || continue
        _dissolve_partnerships!(net, lk, sim.people, now)
    end
    for lk in LAYER_ORDER
        haskey(net.layers, lk) || continue
        _form_new_partnerships!(net, lk, sim.people, now, dt)
    end
    _rebuild_edges!(net, sim.people)
    if get(ENV, "HPVSIM_DEBUG_NET", "") == "1"
        _debug_count(net, sim.people, "STEP", now)
    end
    return net
end

"""Grow arrays and initialize desired_partners/debut_ages for any new agents."""
function _grow_for_new_agents!(net::HPVSexualNet, people::Starsim.People)
    max_uid = people.next_uid - 1
    max_uid <= 0 && return

    # Ensure capacity for all UIDs
    _ensure_capacity!(net, max_uid)

    # Initialize properties for any agents beyond last initialized UID
    last_init = net._last_initialized_uid
    last_init >= max_uid && return

    rng = net.rng
    new_uids = Int[]
    for u in (last_init+1):max_uid
        if u <= length(people.alive.raw) && people.alive.raw[u]
            push!(new_uids, u)
            if people.female.raw[u]
                net.debut_ages[u] = net.debut_f_mean + net.debut_f_std * randn(rng)
            else
                net.debut_ages[u] = net.debut_m_mean + net.debut_m_std * randn(rng)
            end
        end
    end

    # Initialize desired partners for new agents
    if !isempty(new_uids)
        _init_desired_partners!(net, people, new_uids)
    end

    net._last_initialized_uid = max_uid
    return
end

function _debug_count(net::HPVSexualNet, people, label, now)
    alive = people.auids.values
    for lk in LAYER_ORDER
        haskey(net.current_partners, lk) || continue
        cur = net.current_partners[lk]
        n = sum(cur[u] for u in alive if u <= length(cur)) ÷ 2
        @info "$label t=$now $lk=$n pop=$(length(alive))"
    end
end

# ============================================================================
# Internal: Desired partners
# ============================================================================

function _init_desired_partners!(net::HPVSexualNet, people::Starsim.People, uids::Vector{Int})
    rng = net.rng
    for lk in keys(net.layers)
        dp = net.desired_partners[lk]
        for u in uids
            u > length(dp) && continue
            if lk == :m
                # Marital: poisson1(0.01) for both sexes → Poisson(0.01) + 1
                dp[u] = 1 + rand(rng, Poisson(0.01))
            else
                # Casual: sex-specific (matching Python)
                if people.female.raw[u]
                    # Female casual: poisson(1) → just Poisson(1), can be 0
                    dp[u] = rand(rng, Poisson(1.0))
                else
                    # Male casual: poisson1(0.5) → Poisson(0.5) + 1
                    dp[u] = 1 + rand(rng, Poisson(0.5))
                end
            end
        end
    end
    return
end

function _ensure_capacity!(net::HPVSexualNet, max_uid::Int)
    for lk in keys(net.layers)
        cur = net.current_partners[lk]
        old_len = length(cur)
        if max_uid > old_len
            new_len = max(max_uid + 100, old_len * 2)
            resize!(net.current_partners[lk], new_len)
            resize!(net.desired_partners[lk], new_len)
            net.current_partners[lk][(old_len+1):new_len] .= 0
            net.desired_partners[lk][(old_len+1):new_len] .= 0
        end
    end
    # Also resize debut_ages
    old_len = length(net.debut_ages)
    if max_uid > old_len
        new_len = max(max_uid + 100, old_len * 2)
        resize!(net.debut_ages, new_len)
        net.debut_ages[(old_len+1):new_len] .= 0.0
    end
    return
end

# ============================================================================
# Internal: Dissolution
# ============================================================================

function _dissolve_partnerships!(net::HPVSexualNet, lk::Symbol, people::Starsim.People, now::Float64)
    layer = net.layers[lk]
    isempty(layer) && return

    alive_raw = people.alive.raw
    n = length(layer)
    to_remove = falses(n)
    cur = net.current_partners[lk]

    @inbounds for i in 1:n
        if !alive_raw[layer.p1[i]] || !alive_raw[layer.p2[i]] || now >= layer.end_time[i]
            to_remove[i] = true
            p1 = layer.p1[i]; p2 = layer.p2[i]
            if p1 <= length(cur); cur[p1] = max(0, cur[p1] - 1); end
            if p2 <= length(cur); cur[p2] = max(0, cur[p2] - 1); end
        end
    end

    any(to_remove) && remove_indices!(layer, to_remove)
    return
end

# ============================================================================
# Internal: Formation
# ============================================================================

function _form_new_partnerships!(net::HPVSexualNet, lk::Symbol, people::Starsim.People, now::Float64, dt::Float64)
    active = people.auids.values
    bins = net.age_bins
    nb = length(bins)
    mm = net.mixing_matrices[lk]  # mm[male_bin, female_bin]
    rng = net.rng
    f_part = net.female_participation[lk]
    m_part = net.male_participation[lk]
    cur = net.current_partners[lk]
    des = net.desired_partners[lk]
    debut_ages = net.debut_ages

    # Cross-layer concurrency: check if agent has partners in OTHER layers
    other_layers = [olK for olK in keys(net.layers) if olK != lk]

    # -- Step 1: determine eligible agents --
    f_eligible_all = Int[]
    m_eligible_all = Int[]

    @inbounds for u in active
        age = people.age.raw[u]
        u > length(debut_ages) && continue
        age < debut_ages[u] && continue
        u > length(cur) && continue
        cur[u] >= des[u] && continue

        # Cross-layer concurrency check (matching Python f/m_cross_layer)
        has_other = false
        for ol in other_layers
            ol_cur = net.current_partners[ol]
            if u <= length(ol_cur) && ol_cur[u] > 0
                has_other = true
                break
            end
        end
        if has_other
            cross_prob = people.female.raw[u] ? net.f_cross_layer : net.m_cross_layer
            rand(rng) < cross_prob || continue
        end

        if people.female.raw[u]
            push!(f_eligible_all, u)
        else
            push!(m_eligible_all, u)
        end
    end

    # -- Step 2: participation filter for males (globally, matching Python) --
    m_participants = Int[]
    m_age_bins = Int[]  # 1-based age bin index for each participating male
    for u in m_eligible_all
        bi = _age_bin_index(people.age.raw[u], bins)
        rate = bi <= length(m_part) ? m_part[bi] : 0.0
        if rand(rng) < rate
            push!(m_participants, u)
            push!(m_age_bins, bi)
        end
    end

    # -- Step 3: participation filter for females → bin them --
    f_participants = Int[]
    f_age_bins_arr = Int[]
    for u in f_eligible_all
        bi = _age_bin_index(people.age.raw[u], bins)
        rate = bi <= length(f_part) ? f_part[bi] : 0.0
        if rand(rng) < rate
            push!(f_participants, u)
            push!(f_age_bins_arr, bi)
        end
    end

    # -- Step 4: female-driven matching (matching Python's create_edgelist direction) --
    # Iterate through female age bins in shuffled order. Each female picks a male
    # from the weighted pool. Males are removed after selection. This matches
    # Python's direction (female-driven) while using sequential draws for
    # efficiency.
    n_m = length(m_participants)
    m_available = trues(n_m)  # availability flags for participating males

    new_p1 = Int[]  # male UIDs
    new_p2 = Int[]  # female UIDs

    # Build female lists per age bin
    f_bin_lists = [Int[] for _ in 1:nb]
    for (i, bi) in enumerate(f_age_bins_arr)
        push!(f_bin_lists[bi], f_participants[i])
    end

    # Shuffle female order within each bin, then iterate bins in shuffled order
    bin_order = [bi for bi in 1:nb if !isempty(f_bin_lists[bi])]
    shuffle!(rng, bin_order)

    for fi in bin_order
        fems = f_bin_lists[fi]
        shuffle!(rng, fems)

        for f_uid in fems
            # Compute weights: mixing[male_bin, fi] * availability
            total_w = 0.0
            @inbounds for j in 1:n_m
                m_available[j] || continue
                total_w += mm[m_age_bins[j], fi]
            end
            total_w <= 0.0 && continue

            # Draw a male proportional to mixing weights
            target = rand(rng) * total_w
            cum = 0.0
            m_idx = 0
            @inbounds for j in 1:n_m
                m_available[j] || continue
                cum += mm[m_age_bins[j], fi]
                if cum >= target
                    m_idx = j
                    break
                end
            end
            m_idx == 0 && continue

            m_uid = m_participants[m_idx]
            push!(new_p1, m_uid)
            push!(new_p2, f_uid)
            m_available[m_idx] = false  # remove from pool
            cur[m_uid] += 1
            cur[f_uid] += 1
        end
    end

    # Draw per-partnership properties
    layer = net.layers[lk]
    n_new = length(new_p1)
    n_new == 0 && return

    a1 = net.acts_par1[lk]; a2 = net.acts_par2[lk]
    d1 = net.dur_par1[lk];  d2 = net.dur_par2[lk]

    peak    = net.age_act_peak[lk]
    retire  = net.age_act_retirement[lk]
    d_ratio = net.age_act_debut_ratio[lk]
    r_ratio = net.age_act_retirement_ratio[lk]

    for i in 1:n_new
        # Draw acts/year from NegBinomial(r=par2, p=par2/(par1+par2))
        acts_drawn = if a2 > 0 && a1 > 0
            p_nb = a2 / (a1 + a2)
            max(1.0, Float64(rand(rng, NegativeBinomial(a2, p_nb))))
        else
            a1
        end

        # Draw duration
        dur_drawn = if net.dur_dist_type[lk] == :neg_binomial
            p_nb = d2 / (d1 + d2)
            max(dt, Float64(rand(rng, NegativeBinomial(d2, p_nb))))
        else
            # Python hpvsim hpu.sample: par1=mean, par2=std of the lognormal distribution
            # Convert to underlying normal parameters (matching lines 311-312 of hpvsim/utils.py)
            mu    = log(d1^2 / sqrt(d2^2 + d1^2))
            sigma = sqrt(log(d2^2 / d1^2 + 1))
            max(dt, rand(rng, LogNormal(mu, sigma)))
        end

        # Age-scale acts ONCE at creation (matching Python: age_scale_acts in make_contacts)
        m_uid = new_p1[i]
        f_uid = new_p2[i]
        avg_age   = (people.age.raw[m_uid] + people.age.raw[f_uid]) / 2.0
        avg_debut = (net.debut_ages[m_uid] + net.debut_ages[f_uid]) / 2.0
        scaled_acts = age_scale_acts(acts_drawn, avg_age, avg_debut, peak, retire, d_ratio, r_ratio)

        # Python filters out zero-act partnerships: keep_inds = scaled_acts > 0
        scaled_acts <= 0.0 && continue

        push!(layer.p1, m_uid)
        push!(layer.p2, f_uid)
        push!(layer.acts, scaled_acts)
        push!(layer.dur, dur_drawn)
        push!(layer.start_time, now)
        push!(layer.end_time, now + dur_drawn)
    end
    return
end

# ============================================================================
# Internal: Rebuild Starsim edges from persistent layers
# ============================================================================

"""Rebuild Starsim Edges from all layers with age-scaled acts per timestep.

Edge beta stores acts_per_year (age-scaled) for use by transmission.
Condom factor is baked into the stored beta as a negative flag:
  beta > 0 → acts_per_year for edges with condom_factor
  We encode condom_factor into a separate array stored on the network.
"""
function _rebuild_edges!(net::HPVSexualNet, people::Starsim.People)
    Starsim.clear_edges!(net.data.edges)

    all_p1 = Int[]
    all_p2 = Int[]
    all_beta = Float64[]
    # Store per-edge condom factor: (1 - condom_usage * condom_efficacy)
    condom_factors = Float64[]

    for (lk, layer) in net.layers
        cf = 1.0 - net.condom_usage[lk] * net.condom_efficacy

        @inbounds for i in 1:length(layer)
            m_uid = layer.p1[i]
            f_uid = layer.p2[i]
            # Acts were already age-scaled at creation time (matching Python)
            push!(all_p1, m_uid)
            push!(all_p2, f_uid)
            push!(all_beta, layer.acts[i])
            push!(condom_factors, cf)
        end
    end

    if !isempty(all_p1)
        Starsim.add_edges!(net.data.edges, all_p1, all_p2, all_beta)
    end
    # Store per-edge condom factors for transmission code
    net._edge_condom_factors = condom_factors
    return
end

# ============================================================================
# Age-dependent act scaling (matching Python hpvsim population.py)
# ============================================================================

"""Scale acts by age. Three phases: debut→peak (rise), peak→retirement (fall), ≥retirement (zero)."""
function age_scale_acts(acts::Float64, avg_age::Float64, debut::Float64,
                        peak::Float64, retirement::Float64,
                        debut_ratio::Float64, retirement_ratio::Float64)
    avg_age < debut && return 0.0
    avg_age >= retirement && return 0.0

    if avg_age <= peak
        denom = peak - debut
        denom <= 0.0 && return acts
        scale = debut_ratio + (1.0 - debut_ratio) * (avg_age - debut) / denom
    else
        denom = peak - retirement
        denom >= 0.0 && return acts * retirement_ratio
        scale = retirement_ratio + (1.0 - retirement_ratio) * (avg_age - retirement) / denom
    end
    return acts * clamp(scale, 0.0, 1.0)
end

# ============================================================================
# HPVNet convenience constructor
# ============================================================================

"""
    HPVNet(; kwargs...) → HPVSexualNet or MFNet

Convenience constructor for an HPV-appropriate sexual network.
Returns a persistent partnership HPVSexualNet matching Python hpvsim.
Pass `use_age_mixing=false` to get a simple MFNet instead.
"""
function HPVNet(; use_age_mixing::Bool=true, mean_dur::Union{Nothing,Real}=nothing, kwargs...)
    if !use_age_mixing
        # Fallback to simple MFNet
        if mean_dur !== nothing
            return Starsim.MFNet(; mean_dur=Float64(mean_dur), kwargs...)
        else
            return Starsim.MFNet(; kwargs...)
        end
    end
    # Filter out unsupported kwargs for HPVSexualNet
    supported = (:name, :condom_efficacy, :debut_f_mean, :debut_f_std, :debut_m_mean, :debut_m_std)
    filtered = Dict(k => v for (k, v) in kwargs if k in supported)
    return HPVSexualNet(; filtered...)
end

# ============================================================================
# Helper
# ============================================================================

"""Assign each agent to an age bin index (1-based)."""
function _age_bin_index(age::Float64, bins::Vector{Float64})
    nb = length(bins)
    @inbounds for i in nb:-1:1
        age >= bins[i] && return i
    end
    return 1
end

# ============================================================================
# Age-structured sexual mixing defaults (from Python hpvsim/parameters.py)
# ============================================================================

const AGE_MIXING_BINS = Float64[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

const MARITAL_MIXING_MATRIX = Float64[
#    0   5  10  15  20  25  30  35  40  45  50  55  60  65  70  75
     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0;  #  0
     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0;  #  5
     0   0  .1   0   0   0   0   0   0   0   0   0   0   0   0   0;  # 10
     0   0  .1  .1   0   0   0   0   0   0   0   0   0   0   0   0;  # 15
     0   0  .1  .1  .1  .1   0   0   0   0   0   0   0   0   0   0;  # 20
     0   0  .5  .1  .5  .1  .1   0   0   0   0   0   0   0   0   0;  # 25
     0   0   1  .5  .5  .5  .5  .1   0   0   0   0   0   0   0   0;  # 30
     0   0  .5   1   1  .5   1   1  .5   0   0   0   0   0   0   0;  # 35
     0   0   0  .5   1   1   1   1   1  .5   0   0   0   0   0   0;  # 40
     0   0   0   0  .1   1   1   2   1   1  .5   0   0   0   0   0;  # 45
     0   0   0   0   0  .1   1   1   1   1   2  .5   0   0   0   0;  # 50
     0   0   0   0   0   0  .1   1   1   1   1   2  .5   0   0   0;  # 55
     0   0   0   0   0   0   0  .1  .5   1   1   1   2  .5   0   0;  # 60
     0   0   0   0   0   0   0   0   0   0   1   1   1   2  .5   0;  # 65
     0   0   0   0   0   0   0   0   0   0   0   1   1   1   1  .5;  # 70
     0   0   0   0   0   0   0   0   0   0   0   0   1   1   1   1;  # 75
]

const CASUAL_MIXING_MATRIX = Float64[
#    0   5  10  15  20  25  30  35  40  45  50  55  60  65  70  75
     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0;  #  0
     0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0;  #  5
     0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0;  # 10
     0   0   1   1   1   1   0   0   0   0   0   0   0   0   0   0;  # 15
     0   0   1   1   1   1   1   0   0   0   0   0   0   0   0   0;  # 20
     0   0  .5   1   1   1   1   1   0   0   0   0   0   0   0   0;  # 25
     0   0   0  .5   1   1   1  .5   0   0   0   0   0   0   0   0;  # 30
     0   0   0  .5   1   1   1   1  .5   0   0   0   0   0   0   0;  # 35
     0   0   0   0  .5   1   1   1   1  .5   0   0   0   0   0   0;  # 40
     0   0   0   0   0   1   1   1   1   1  .5   0   0   0   0   0;  # 45
     0   0   0   0   0  .5   1   1   1   1   1  .5   0   0   0   0;  # 50
     0   0   0   0   0   0   0   1   1   1   1   1  .5   0   0   0;  # 55
     0   0   0   0   0   0   0   0   1   1   1   1   1  .5   0   0;  # 60
     0   0   0   0   0   0   0   0   0   1   1   1   1   2  .5   0;  # 65
     0   0   0   0   0   0   0   0   0   0   1   1   1   1   1  .5;  # 70
     0   0   0   0   0   0   0   0   0   0   0   1   1   1   1   1;  # 75
]

function default_layer_probs(; network_type::Symbol=:marital)
    if network_type == :marital
        return Dict(
            :age_bins => AGE_MIXING_BINS,
            :female => [0, 0, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
            :male   => [0, 0, 0.01, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
        )
    else
        return Dict(
            :age_bins => AGE_MIXING_BINS,
            :female => [0, 0, 0.2, 0.6, 0.8, 0.6, 0.4, 0.4, 0.4, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            :male   => [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.6, 0.2, 0.1, 0.05, 0.02, 0.02, 0.02],
        )
    end
end
