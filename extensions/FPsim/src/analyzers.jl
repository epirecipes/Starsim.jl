"""
FPsim analyzers — population metrics, ASFR, TFR, CPR tracking.
"""

# ============================================================================
# Age bins for ASFR calculation
# ============================================================================
const ASFR_BINS = Float64[10, 15, 20, 25, 30, 35, 40, 45, 50]
const N_ASFR_BINS = length(ASFR_BINS) - 1

"""Map an age to an ASFR bin index (1-based), or 0 if outside range."""
function asfr_bin(age::Real)
    for i in 1:N_ASFR_BINS
        if age >= ASFR_BINS[i] && age < ASFR_BINS[i+1]
            return i
        end
    end
    return 0
end

# ============================================================================
# FPAnalyzer
# ============================================================================

"""
    FPAnalyzer <: AbstractAnalyzer

Family planning metrics analyzer. Tracks:
- Number of women (reproductive age)
- ASFR (age-specific fertility rates) by 5-year age bin
- TFR (total fertility rate)
- CPR (contraceptive prevalence rate)
- Population growth rate
- Method mix
"""
mutable struct FPAnalyzer <: Starsim.AbstractAnalyzer
    data::Starsim.ModuleData
    # Accumulators for ASFR (rolling 12-month sums)
    births_by_bin::Matrix{Float64}   # (N_ASFR_BINS, npts)
    women_by_bin::Matrix{Float64}    # (N_ASFR_BINS, npts)
end

function FPAnalyzer(; name::Symbol = :fp_analyzer)
    md = Starsim.ModuleData(name; label="FP analyzer")
    FPAnalyzer(md, zeros(0, 0), zeros(0, 0))
end

Starsim.module_data(a::FPAnalyzer) = a.data

function Starsim.init_pre!(a::FPAnalyzer, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    npts = md.t.npts
    Starsim.define_results!(a,
        Starsim.Result(:n_women; npts=npts, label="Women (15-49)", scale=false),
        Starsim.Result(:n_women_total; npts=npts, label="Women (total)", scale=false),
        Starsim.Result(:pop_growth_rate; npts=npts, label="Pop growth rate", scale=false),
        Starsim.Result(:tfr; npts=npts, label="TFR", scale=false),
        Starsim.Result(:cpr; npts=npts, label="CPR", scale=false),
        Starsim.Result(:mcpr; npts=npts, label="mCPR", scale=false),
    )

    # Add per-bin ASFR results
    for i in 1:N_ASFR_BINS
        lo = Int(ASFR_BINS[i])
        hi = Int(ASFR_BINS[i+1])
        rname = Symbol("asfr_$(lo)_$(hi)")
        Starsim.define_results!(a,
            Starsim.Result(rname; npts=npts, label="ASFR $lo-$hi", scale=false),
        )
    end

    a.births_by_bin = zeros(N_ASFR_BINS, npts)
    a.women_by_bin = zeros(N_ASFR_BINS, npts)

    md.initialized = true
    return a
end

function Starsim.step!(a::FPAnalyzer, sim)
    md = Starsim.module_data(a)
    ti = md.t.ti
    ti > md.t.npts && return a

    active = sim.people.auids.values
    fpmod = nothing
    for (_, c) in sim.connectors
        if c isa FPmod
            fpmod = c
            break
        end
    end

    n_women_15_49 = 0
    n_women_total = 0
    n_on_contra = 0
    n_modern_contra = 0
    n_eligible = 0  # sexually active women 15-49

    @inbounds for u in active
        !sim.people.female.raw[u] && continue
        age = sim.people.age.raw[u]
        n_women_total += 1

        if age >= 15.0 && age < 50.0
            n_women_15_49 += 1
            bin = asfr_bin(age)
            if bin > 0
                a.women_by_bin[bin, ti] += 1.0
            end
        end

        if fpmod !== nothing && age >= 15.0 && age < 50.0
            if fpmod.sexually_active.raw[u] || fpmod.pregnant.raw[u] || fpmod.postpartum.raw[u]
                n_eligible += 1
            end
            if fpmod.on_contra.raw[u]
                n_on_contra += 1
                midx = Int(fpmod.method_idx.raw[u])
                if midx >= 1 && midx <= length(fpmod.methods)
                    if fpmod.methods[midx].modern
                        n_modern_contra += 1
                    end
                end
            end
        end
    end

    # Count births by age bin from FPmod results (use parity change)
    if fpmod !== nothing
        fpmd = Starsim.module_data(fpmod)
        if haskey(fpmd.results, :n_births)
            total_births = fpmd.results[:n_births][ti]
            # Distribute births across age bins based on pregnant women's ages
            @inbounds for u in active
                !sim.people.female.raw[u] && continue
                # Identify women who just gave birth (postpartum with months_pp near 0)
                if fpmod.postpartum.raw[u] && fpmod.months_postpartum.raw[u] < sim.pars.dt * MPY + 0.5
                    age = sim.people.age.raw[u]
                    bin = asfr_bin(age)
                    if bin > 0
                        a.births_by_bin[bin, ti] += 1.0
                    end
                end
            end
        end
    end

    # Store results
    res = md.results
    res[:n_women][ti] = Float64(n_women_15_49)
    res[:n_women_total][ti] = Float64(n_women_total)

    # CPR (proportion of eligible women using contraception)
    if n_women_15_49 > 0
        res[:cpr][ti] = Float64(n_on_contra) / Float64(n_women_15_49)
        res[:mcpr][ti] = Float64(n_modern_contra) / Float64(n_women_15_49)
    end

    # ASFR and TFR
    tfr = 0.0
    for i in 1:N_ASFR_BINS
        lo = Int(ASFR_BINS[i])
        hi = Int(ASFR_BINS[i+1])
        rname = Symbol("asfr_$(lo)_$(hi)")
        bin_width = hi - lo
        births = a.births_by_bin[i, ti]
        women = a.women_by_bin[i, ti]
        asfr_val = women > 0 ? (births / women) * (1.0 / sim.pars.dt) * 1000.0 : 0.0
        if haskey(res, rname)
            res[rname][ti] = asfr_val
        end
        tfr += asfr_val * bin_width / 1000.0
    end
    res[:tfr][ti] = tfr

    # Pop growth rate
    if ti > 1
        prev_pop = res[:n_women_total][ti - 1]
        if prev_pop > 0
            res[:pop_growth_rate][ti] = (Float64(n_women_total) - prev_pop) / prev_pop / sim.pars.dt
        end
    end

    return a
end
