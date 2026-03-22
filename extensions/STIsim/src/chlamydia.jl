"""
Chlamydia — SEIS-based STI with PID.
Port of Python `stisim.diseases.chlamydia.Chlamydia`.
"""

"""
    Chlamydia(; kwargs...) → SEIS

Chlamydia trachomatis SEIS model. Wraps the base SEIS with
chlamydia-specific default parameters matching Python stisim.

# Keyword arguments
- `init_prev::Real` — initial prevalence (default 0.03)
- `beta_m2f::Real` — per-act male-to-female (default 0.06)
- `rel_beta_f2m::Real` — relative female-to-male (default 0.5)
- `beta_m2m::Real` — per-act MSM (default 0.06)
- `dur_exp::Real` — exposed duration (default 1/52, ~1 week)
- `p_symp_f::Real` — symptomatic probability female (default 0.20)
- `p_symp_m::Real` — symptomatic probability male (default 0.54)
- `dur_inf::Real` — fallback infection duration in years (default 1.0)
- `p_pid::Real` — PID probability (default 0.2)
- `eff_condom::Real` — condom efficacy (default 0.0; Python stisim default)
"""
function Chlamydia(;
    name::Symbol       = :chlamydia,
    init_prev::Real    = 0.03,
    beta_m2f::Real     = 0.06,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.06,
    dur_exp::Real      = 1/52,
    p_symp_f::Real     = 0.20,
    p_symp_m::Real     = 0.54,
    dur_inf::Real      = 1.0,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.2,
    eff_condom::Real   = 0.0,
    # Sex-specific clearance (years) — Python: months(X)/12
    dur_asymp2clear_f::Real     = 18.0/12,  # months(18)
    dur_asymp2clear_f_std::Real = 1.0/12,   # months(1)
    dur_asymp2clear_m::Real     = 12.0/12,  # months(12)
    dur_asymp2clear_m_std::Real = 1.0/12,   # months(1)
    dur_symp2clear_f::Real      = 18.0/12,  # months(18)
    dur_symp2clear_f_std::Real  = 1.0/12,   # months(1)
    dur_symp2clear_m::Real      = 12.0/12,  # months(12)
    dur_symp2clear_m_std::Real  = 1.0/12,   # months(1)
    # Sex-specific presymptomatic period (years) — Python: weeks(X)/52
    dur_presymp_f::Real         = 1.0/52,   # weeks(1)
    dur_presymp_f_std::Real     = 10.0/52,  # weeks(10)
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
        label      = "Chlamydia",
    )
end
