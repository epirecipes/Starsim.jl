"""
Gonorrhea — SEIS-based STI with drug resistance potential.
Port of Python `stisim.diseases.gonorrhea.Gonorrhea`.
"""

"""
    Gonorrhea(; kwargs...) → SEIS

Neisseria gonorrhoeae SEIS model. Wraps the base SEIS with
gonorrhea-specific default parameters matching Python stisim.

# Keyword arguments
- `init_prev::Real` — initial prevalence (default 0.01)
- `beta_m2f::Real` — per-act m→f (default 0.06)
- `rel_beta_f2m::Real` — relative f→m (default 0.5)
- `beta_m2m::Real` — per-act MSM (default 0.06)
- `dur_exp::Real` — exposed duration (default 0.0; gonorrhea has no exposed period)
- `p_symp_f::Real` — symptomatic probability female (default 0.35)
- `p_symp_m::Real` — symptomatic probability male (default 0.65)
- `dur_inf::Real` — fallback infection duration in years (default 0.5)
- `p_pid::Real` — PID probability (default 0.0; Python stisim default)
"""
function Gonorrhea(;
    name::Symbol       = :gonorrhea,
    init_prev::Real    = 0.01,
    beta_m2f::Real     = 0.06,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.06,
    dur_exp::Real      = 0.0,
    p_symp_f::Real     = 0.35,
    p_symp_m::Real     = 0.65,
    dur_inf::Real      = 0.5,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.0,
    eff_condom::Real   = 0.9,
    # Sex-specific clearance (years) — Python: months(X)/12
    dur_asymp2clear_f::Real     = 8.0/12,   # months(8)
    dur_asymp2clear_f_std::Real = 2.0/12,   # months(2)
    dur_asymp2clear_m::Real     = 6.0/12,   # months(6)
    dur_asymp2clear_m_std::Real = 3.0/12,   # months(3)
    dur_symp2clear_f::Real      = 9.0/12,   # months(9)
    dur_symp2clear_f_std::Real  = 2.0/12,   # months(2)
    dur_symp2clear_m::Real      = 6.0/12,   # months(6)
    dur_symp2clear_m_std::Real  = 3.0/12,   # months(3)
    # Sex-specific presymptomatic period (years) — Python: weeks(X)/52
    dur_presymp_f::Real         = 1.0/52,   # weeks(1)
    dur_presymp_f_std::Real     = 12.0/52,  # weeks(12)
    dur_presymp_m::Real         = 0.25/52,  # weeks(0.25)
    dur_presymp_m_std::Real     = 1.0/52,   # weeks(1)
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
        dur_asymp2clear_f     = dur_asymp2clear_f,
        dur_asymp2clear_f_std = dur_asymp2clear_f_std,
        dur_asymp2clear_m     = dur_asymp2clear_m,
        dur_asymp2clear_m_std = dur_asymp2clear_m_std,
        dur_symp2clear_f      = dur_symp2clear_f,
        dur_symp2clear_f_std  = dur_symp2clear_f_std,
        dur_symp2clear_m      = dur_symp2clear_m,
        dur_symp2clear_m_std  = dur_symp2clear_m_std,
        dur_presymp_f         = dur_presymp_f,
        dur_presymp_f_std     = dur_presymp_f_std,
        dur_presymp_m         = dur_presymp_m,
        dur_presymp_m_std     = dur_presymp_m_std,
        label      = "Gonorrhea",
    )
end
