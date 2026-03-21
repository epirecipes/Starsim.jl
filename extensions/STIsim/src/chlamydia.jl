"""
Chlamydia — SEIS-based STI with PID.
Port of Python `stisim.diseases.chlamydia.Chlamydia`.
"""

"""
    Chlamydia <: AbstractInfection

Chlamydia trachomatis SEIS model. Wraps the base SEIS with
chlamydia-specific default parameters.

# Keyword arguments
- `init_prev::Real` — initial prevalence (default 0.03)
- `beta_m2f::Real` — per-act male-to-female (default 0.16)
- `rel_beta_f2m::Real` — relative female-to-male (default 0.5)
- `beta_m2m::Real` — per-act MSM (default 0.16)
- `dur_exp::Real` — exposed duration (default 1/52)
- `p_symp_f::Real` — symptomatic probability female (default 0.25)
- `p_symp_m::Real` — symptomatic probability male (default 0.5)
- `dur_inf::Real` — infection duration in years (default 1.0)
- `p_pid::Real` — PID probability (default 0.2)
"""
function Chlamydia(;
    name::Symbol       = :chlamydia,
    init_prev::Real    = 0.03,
    beta_m2f::Real     = 0.16,
    rel_beta_f2m::Real = 0.5,
    beta_m2m::Real     = 0.16,
    dur_exp::Real      = 1/52,
    p_symp_f::Real     = 0.25,
    p_symp_m::Real     = 0.50,
    dur_inf::Real      = 1.0,
    dur_inf_std::Real  = 0.1,
    p_pid::Real        = 0.2,
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
        label      = "Chlamydia",
    )
end
