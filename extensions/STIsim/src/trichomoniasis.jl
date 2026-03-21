"""
Trichomoniasis — SEIS-based STI.
Port of Python `stisim.diseases.trichomoniasis.Trichomoniasis`.
"""

"""
    Trichomoniasis(; kwargs...) → SEIS

Trichomonas vaginalis SEIS model. Wraps the base SEIS with
trichomoniasis-specific default parameters.

# Keyword arguments
- `init_prev::Real` — initial prevalence (default 0.03)
- `beta_m2f::Real` — per-act m→f (default 0.20)
- `rel_beta_f2m::Real` — relative f→m (default 0.5)
- `beta_m2m::Real` — per-act MSM (default 0.10, lower for trich)
- `dur_inf::Real` — infection duration in years (default 0.5)
- `p_symp_f::Real` — symptomatic probability female (default 0.30)
- `p_symp_m::Real` — symptomatic probability male (default 0.20)
"""
function Trichomoniasis(;
    name::Symbol       = :trichomoniasis,
    init_prev::Real    = 0.03,
    beta_m2f::Real     = 0.20,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.10,
    dur_exp::Real      = 1/52,
    p_symp_f::Real     = 0.30,
    p_symp_m::Real     = 0.20,
    dur_inf::Real      = 0.5,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.05,
    eff_condom::Real   = 0.9,
)
    SEIS(;
        name       = name,
        init_prev  = init_prev,
        beta_m2f   = beta_m2f,
        rel_beta_f2m = rel_beta_f2m,
        beta_m2m   = beta_m2m,
        dur_exp    = dur_exp,
        p_symp_f   = p_symp_f,
        p_symp_m   = p_symp_m,
        dur_inf    = dur_inf,
        dur_inf_std = dur_inf_std,
        p_pid      = p_pid,
        eff_condom = eff_condom,
        label      = "Trichomoniasis",
    )
end
