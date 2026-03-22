"""
HPV-specific network configuration.

HPV is sexually transmitted, so we use Starsim's MFNet (male-female network)
with HPV-appropriate defaults. This file provides convenience constructors
and age-structured mixing defaults from the Python hpvsim.
"""

# ============================================================================
# HPVSexualNet — age-assortative male-female sexual network
# ============================================================================

"""
    HPVSexualNet <: Starsim.AbstractNetwork

Age-assortative heterosexual partnership network for HPV transmission.
Partners are selected with preference for similar age groups using
mixing matrices from the Python hpvsim.

# Keyword arguments
- `name::Symbol` — network name (default `:sexual`)
- `mean_dur::Float64` — mean partnership duration in years (default 2.0)
- `participation_rate::Float64` — base fraction sexually active (default 0.8)
- `mixing_matrix::Matrix{Float64}` — male(row) × female(col) mixing preferences
- `age_bins::Vector{Float64}` — age bin boundaries
- `female_participation::Vector{Float64}` — age-specific female participation rates
- `male_participation::Vector{Float64}` — age-specific male participation rates
"""
mutable struct HPVSexualNet <: Starsim.AbstractNetwork
    data::Starsim.NetworkData
    mean_dur::Float64
    participation_rate::Float64
    mixing_matrix::Matrix{Float64}
    age_bins::Vector{Float64}
    female_participation::Vector{Float64}
    male_participation::Vector{Float64}
    rng::StableRNG
end

function HPVSexualNet(;
    name::Symbol = :sexual,
    mean_dur::Real = 2.0,
    participation_rate::Real = 0.8,
    mixing_matrix::Matrix{Float64} = MARITAL_MIXING_MATRIX,
    age_bins::Vector{Float64} = AGE_MIXING_BINS,
    female_participation::Vector{Float64} = default_layer_probs(network_type=:marital)[:female],
    male_participation::Vector{Float64} = default_layer_probs(network_type=:marital)[:male],
)
    md = Starsim.ModuleData(name; label="HPV sexual network")
    nd = Starsim.NetworkData(md, Starsim.Edges(), true)
    HPVSexualNet(nd, Float64(mean_dur), Float64(participation_rate),
                 mixing_matrix, age_bins, female_participation, male_participation,
                 StableRNG(0))
end

Starsim.network_data(net::HPVSexualNet) = net.data

function Starsim.init_pre!(net::HPVSexualNet, sim)
    md = Starsim.module_data(net)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    net.rng = StableRNG(hash(md.name) ⊻ sim.pars.rand_seed)
    md.initialized = true
    _form_age_partnerships!(net, sim.people)
    return net
end

function Starsim.step!(net::HPVSexualNet, sim)
    Starsim.clear_edges!(net.data.edges)
    _form_age_partnerships!(net, sim.people)
    return net
end

"""Assign each agent to an age bin index (1-based). Returns 0 for unmatched."""
function _age_bin_index(age::Float64, bins::Vector{Float64})
    nb = length(bins)
    @inbounds for i in nb:-1:1
        if age >= bins[i]
            return i
        end
    end
    return 1
end

"""Form age-assortative male-female partnerships using mixing matrix."""
function _form_age_partnerships!(net::HPVSexualNet, people::Starsim.People)
    active = people.auids.values
    bins = net.age_bins
    nb = length(bins)
    mm = net.mixing_matrix
    rng = net.rng

    # Separate males and females by age bin
    male_bins   = [Int[] for _ in 1:nb]
    female_bins = [Int[] for _ in 1:nb]

    @inbounds for u in active
        age = people.age.raw[u]
        bi = _age_bin_index(age, bins)
        if people.female.raw[u]
            push!(female_bins[bi], u)
        else
            push!(male_bins[bi], u)
        end
    end

    # For each male age bin, form partnerships weighted by mixing matrix
    all_p1 = Int[]
    all_p2 = Int[]

    for mi in 1:nb
        males_in_bin = male_bins[mi]
        isempty(males_in_bin) && continue

        # Age-specific male participation
        m_part = mi <= length(net.male_participation) ? net.male_participation[mi] : net.participation_rate
        n_seeking = max(1, Int(round(length(males_in_bin) * m_part)))
        n_seeking = min(n_seeking, length(males_in_bin))

        # Shuffle and take seeking males
        perm = randperm(rng, length(males_in_bin))
        seeking_males = males_in_bin[perm[1:n_seeking]]

        # Compute mixing weights for each female bin
        weights = Float64[]
        for fi in 1:nb
            w = mm[mi, fi]
            # Weight by available females adjusted for participation
            f_part = fi <= length(net.female_participation) ? net.female_participation[fi] : net.participation_rate
            n_avail = Int(round(length(female_bins[fi]) * f_part))
            push!(weights, w * n_avail)
        end
        total_w = sum(weights)
        total_w <= 0.0 && continue

        # Normalize to cumulative probabilities
        cum_weights = cumsum(weights) ./ total_w

        # For each seeking male, pick a female bin then a random female
        for m_uid in seeking_males
            r = rand(rng)
            fi = 1
            @inbounds for j in 1:nb
                if r <= cum_weights[j]
                    fi = j
                    break
                end
            end

            isempty(female_bins[fi]) && continue
            # Pick a random female from the chosen bin
            f_idx = rand(rng, 1:length(female_bins[fi]))
            f_uid = female_bins[fi][f_idx]

            push!(all_p1, m_uid)
            push!(all_p2, f_uid)
        end
    end

    if !isempty(all_p1)
        Starsim.add_edges!(net.data.edges, all_p1, all_p2)
    end
    return net
end

# ============================================================================
# HPVNet — backward-compatible convenience constructor
# ============================================================================

"""
    HPVNet(; kwargs...) → HPVSexualNet

Convenience constructor for an HPV-appropriate sexual network.
Returns an age-assortative HPVSexualNet by default.

# Keyword arguments
- `mean_dur::Float64` — mean partnership duration in years (default 2.0)
- `participation_rate::Float64` — fraction sexually active (default 0.8)
- `name::Symbol` — network name (default :sexual)
- `use_age_mixing::Bool` — use age-assortative mixing (default true)
"""
function HPVNet(;
    mean_dur::Real = 2.0,
    participation_rate::Real = 0.8,
    name::Symbol = :sexual,
    use_age_mixing::Bool = true,
)
    if use_age_mixing
        return HPVSexualNet(;
            name = name,
            mean_dur = Float64(mean_dur),
            participation_rate = Float64(participation_rate),
        )
    else
        return Starsim.MFNet(;
            name = name,
            mean_dur = Float64(mean_dur),
            participation_rate = Float64(participation_rate),
        )
    end
end

# ============================================================================
# Age-structured sexual mixing defaults (from Python hpvsim/parameters.py)
# ============================================================================

"""
Default age bins for sexual mixing matrices (5-year bands from 0 to 75+).
"""
const AGE_MIXING_BINS = Float64[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]

"""
Default marital/stable partnership mixing matrix (males in rows, females in columns).
Higher values indicate more likely partnerships between age groups.
From Python hpvsim `get_mixing(network='default')`.
"""
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

"""
Default casual partnership mixing matrix.
From Python hpvsim `get_mixing(network='default')`.
"""
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

"""
    default_layer_probs(; network_type=:marital)

Return age-dependent sexual activity probabilities for males and females.
First row = age bins, second row = female participation, third row = male participation.
"""
function default_layer_probs(; network_type::Symbol=:marital)
    if network_type == :marital
        return Dict(
            :age_bins => AGE_MIXING_BINS,
            :female => [0, 0, 0.01, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
            :male   => [0, 0, 0.01, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01],
        )
    else  # :casual
        return Dict(
            :age_bins => AGE_MIXING_BINS,
            :female => [0, 0, 0.2, 0.6, 0.8, 0.6, 0.4, 0.4, 0.4, 0.1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            :male   => [0, 0, 0.2, 0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.6, 0.2, 0.1, 0.05, 0.02, 0.02, 0.02],
        )
    end
end
