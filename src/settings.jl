"""
Global simulation options for Starsim.jl.

Provides a mutable `Options` struct controlling global behavior such as
CRN (Common Random Numbers) support, verbosity, and numeric precision.
"""

# ============================================================================
# Options — global simulation settings
# ============================================================================

"""
    Options

Global settings for Starsim.jl simulations. Controls CRN (Common Random
Numbers), verbosity, and numeric precision.

# Fields
- `verbose::Float64` — default verbosity level (0.0 = silent, 1.0 = normal)
- `single_rng::Bool` — if `true`, use a single centralized RNG (not CRN-safe)
- `precision::Int` — numeric precision (32 or 64)
- `slot_scale::Float64` — CRN slot scale factor; 0.0 = CRN disabled, >0 = enabled.
  Controls the range for newborn slot assignment: `Uniform(N, slot_scale * N)`.

# CRN overview

When `slot_scale > 0`, Common Random Numbers mode is active:
- Each stochastic decision gets its own seeded PRNG stream.
- Random draws are indexed by agent `slot` (not UID), ensuring that
  adding/removing agents doesn't shift other agents' draws.
- Newborn slots are drawn from a wide range to avoid collisions.
- Pairwise transmission uses XOR combining for CRN-safe edge randomness.

# Example
```julia
# Enable CRN globally
Starsim.OPTIONS.slot_scale = 5.0

# Check if CRN is enabled
Starsim.crn_enabled()
```
"""
mutable struct Options
    verbose::Float64
    single_rng::Bool
    precision::Int
    slot_scale::Float64
end

"""Global simulation options singleton."""
const OPTIONS = Options(0.1, false, 64, 0.0)

"""
    crn_enabled() → Bool

Return `true` if Common Random Numbers mode is active (slot_scale > 0).
"""
crn_enabled() = OPTIONS.slot_scale > 0.0

"""
    get_slot_scale() → Float64

Return the current CRN slot scale factor.
"""
get_slot_scale() = OPTIONS.slot_scale

export Options, OPTIONS, crn_enabled, get_slot_scale
