"""
HPV-specific network configuration.

HPV is sexually transmitted, so we use Starsim's MFNet (male-female network)
with HPV-appropriate defaults. This file provides convenience constructors
and age-structured mixing defaults from the Python hpvsim.
"""

"""
    HPVNet(; kwargs...) → MFNet

Convenience constructor for an HPV-appropriate sexual network.
Wraps Starsim's MFNet with HPV-relevant defaults.

# Keyword arguments
- `mean_dur::Float64` — mean partnership duration in years (default 2.0)
- `participation_rate::Float64` — fraction sexually active (default 0.8)
- `name::Symbol` — network name (default :sexual)
"""
function HPVNet(;
    mean_dur::Real = 2.0,
    participation_rate::Real = 0.8,
    name::Symbol = :sexual,
)
    return Starsim.MFNet(;
        name = name,
        mean_dur = Float64(mean_dur),
        participation_rate = Float64(participation_rate),
    )
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
