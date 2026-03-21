"""
Gonorrhea — SEIS-based STI with drug resistance potential.
Port of Python `stisim.diseases.gonorrhea.Gonorrhea`.
"""

"""
    Gonorrhea(; kwargs...) → SEIS

Neisseria gonorrhoeae SEIS model. Wraps the base SEIS with
gonorrhea-specific default parameters.

# Keyword arguments
- `init_prev::Real` — initial prevalence (default 0.02)
- `beta_m2f::Real` — per-act m→f (default 0.20)
- `rel_beta_f2m::Real` — relative f→m (default 0.5)
- `beta_m2m::Real` — per-act MSM (default 0.20)
- `dur_exp::Real` — exposed duration (default 1/52)
- `p_symp_f::Real` — symptomatic probability female (default 0.20)
- `p_symp_m::Real` — symptomatic probability male (default 0.80)
- `dur_inf::Real` — infection duration in years (default 0.5)
- `p_pid::Real` — PID probability (default 0.15)
"""
function Gonorrhea(;
    name::Symbol       = :gonorrhea,
    init_prev::Real    = 0.02,
    beta_m2f::Real     = 0.20,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.20,
    dur_exp::Real      = 1/52,
    p_symp_f::Real     = 0.20,
    p_symp_m::Real     = 0.80,
    dur_inf::Real      = 0.5,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.15,
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
        label      = "Gonorrhea",
    )
end
