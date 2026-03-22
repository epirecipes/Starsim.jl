"""
Cross-validation of Julia HPVsim against Python hpvsim.

Runs four matching scenarios in Julia, then reads the Python JSON
results (produced by cross_validation.py) and prints side-by-side
comparison tables.

Scenarios:
  1. Single genotype HPV16 (basic)
  2. Multi-genotype HPV16 + HPV18
  3. HPV16+18 with prophylactic vaccination
  4. HPV16 with screening programme

Usage:
  # First, run the Python side:
  #   python cross_validation.py
  # Then run the Julia side:
  #   julia --project=../../ cross_validation.jl
"""

using HPVsim
using Starsim
using Statistics: mean
using JSON3

const OUT_DIR = joinpath(@__DIR__, "cross_validation_results")
mkpath(OUT_DIR)

const N_AGENTS = 5_000
const SIM_START = 2000.0
const SIM_STOP  = 2050.0
const SIM_DT    = 0.25
const SEED      = 42

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Extract results dict from a completed Julia HPVSim."""
function extract_julia_results(sim; genotype_names=Symbol[])
    npts = 0
    results = Dict{String, Any}()

    # Per-genotype results
    total_prev   = nothing
    total_cin    = nothing
    total_cancer = nothing

    for (name, dis) in sim.diseases
        md   = Starsim.module_data(dis)
        prev = md.results[:prevalence].values
        npts = length(prev)

        if total_prev === nothing
            total_prev   = zeros(npts)
            total_cin    = zeros(npts)
            total_cancer = zeros(npts)
        end

        total_prev   .+= prev
        total_cin    .+= md.results[:cin_prevalence].values
        total_cancer .+= md.results[:n_cancerous].values

        sname = String(name)
        results["hpv_prev_$sname"]     = copy(prev)
        results["cin_prev_$sname"]     = copy(md.results[:cin_prevalence].values)
        results["n_cin1_$sname"]       = copy(md.results[:n_cin1].values)
        results["n_cin2_$sname"]       = copy(md.results[:n_cin2].values)
        results["n_cin3_$sname"]       = copy(md.results[:n_cin3].values)
        results["n_cancerous_$sname"]  = copy(md.results[:n_cancerous].values)
    end

    # Aggregate
    n_genotypes = length(sim.diseases)
    results["hpv_prev"]     = total_prev
    results["cin_prev"]     = total_cin
    results["n_cancerous"]  = total_cancer
    results["n_genotypes"]  = n_genotypes
    results["npts"]         = npts

    # Year vector
    years = [SIM_START + (i - 1) * SIM_DT for i in 1:npts]
    results["year"] = years

    return results
end

"""Load Python JSON results."""
function load_python_results(scenario_name)
    path = joinpath(OUT_DIR, "$(scenario_name)_python.json")
    if !isfile(path)
        @warn "Python results not found at $path — run cross_validation.py first"
        return nothing
    end
    json_str = read(path, String)
    return JSON3.read(json_str, Dict{String, Any})
end

"""Save Julia results as JSON."""
function save_julia_results(scenario_name, data)
    path = joinpath(OUT_DIR, "$(scenario_name)_julia.json")
    open(path, "w") do f
        JSON3.pretty(f, data)
    end
    println("  Saved → $path")
end

# ── Printing helpers ────────────────────────────────────────────────────────

function print_header(title)
    println()
    println("╔", "═"^68, "╗")
    println("║ ", rpad(title, 67), "║")
    println("╚", "═"^68, "╝")
end

function print_comparison_header()
    println(rpad("Metric", 35), rpad("Julia", 14), rpad("Python", 14), "Match?")
    println("-"^72)
end

function compare_row(label, jval, pval; tol=0.5, fmt="%.4f")
    jstr = @eval @sprintf($fmt, $jval)
    pstr = @eval @sprintf($fmt, $pval)

    # Qualitative check: within tolerance factor or both near zero
    if jval ≈ 0.0 && pval ≈ 0.0
        match = "✓ (both ~0)"
    elseif pval ≈ 0.0
        match = jval < 0.01 ? "~ (both small)" : "✗"
    else
        ratio = jval / pval
        if 1/tol ≤ ratio ≤ tol
            match = "~ qualitative"
        elseif 0.5 ≤ ratio ≤ 2.0
            match = "~ order-of-mag"
        else
            match = "✗ divergent"
        end
    end
    println("  ", rpad(label, 33), rpad(jstr, 14), rpad(pstr, 14), match)
end

# Use Printf directly to avoid eval overhead
using Printf

function compare_row_f(label, jval, pval; tol=2.0)
    jstr = @sprintf("%.4f", jval)
    pstr = @sprintf("%.4f", pval)

    if abs(jval) < 1e-8 && abs(pval) < 1e-8
        match = "✓ both ≈0"
    elseif abs(pval) < 1e-8
        match = abs(jval) < 0.01 ? "~ both small" : "✗"
    else
        ratio = jval / pval
        if 1/tol ≤ ratio ≤ tol
            match = "✓ qualitative"
        elseif 0.2 ≤ ratio ≤ 5.0
            match = "~ same order"
        else
            match = "✗ divergent"
        end
    end
    println("  ", rpad(label, 33), rpad(jstr, 14), rpad(pstr, 14), match)
end

function compare_row_pct(label, jval, pval)
    jstr = @sprintf("%.1f%%", jval)
    pstr = @sprintf("%.1f%%", pval)
    both_pos = jval > 0 && pval > 0
    match = both_pos ? "✓ both show effect" : "✗"
    println("  ", rpad(label, 33), rpad(jstr, 14), rpad(pstr, 14), match)
end

# ── Scenario 1: Single genotype HPV16 ──────────────────────────────────────

function scenario1()
    print_header("Scenario 1: Single genotype HPV16 (basic)")

    sim = HPVSim(
        genotypes    = [GenotypeDef(:hpv16; init_prev=0.05)],
        n_agents     = N_AGENTS,
        start        = SIM_START,
        stop         = SIM_STOP,
        dt           = SIM_DT,
        rand_seed    = SEED,
        use_immunity = false,
        verbose      = 0,
    )
    Starsim.run!(sim)

    jres = extract_julia_results(sim)
    save_julia_results("scenario1", jres)

    # Load Python results
    pres = load_python_results("scenario1")

    println("\n  Julia results:")
    println("    HPV prev (final): $(@sprintf("%.4f", jres["hpv_prev"][end]))")
    println("    CIN prev (final): $(@sprintf("%.6f", jres["cin_prev"][end]))")
    println("    Cancer count (final): $(jres["n_cancerous"][end])")
    println("    Timesteps: $(jres["npts"])")

    if pres !== nothing
        println("\n  ── Comparison ──")
        print_comparison_header()

        npts_j = jres["npts"]
        npts_p = length(pres["hpv_prev"])
        # Sample at comparable time points: year 10, 25, final
        yr10_j = min(round(Int, 10/SIM_DT) + 1, npts_j)
        yr25_j = min(round(Int, 25/SIM_DT) + 1, npts_j)
        yr10_p = min(11, npts_p)   # Python has annual output
        yr25_p = min(26, npts_p)

        compare_row_f("HPV prev (year 10)",  jres["hpv_prev"][yr10_j], pres["hpv_prev"][yr10_p])
        compare_row_f("HPV prev (year 25)",  jres["hpv_prev"][yr25_j], pres["hpv_prev"][yr25_p])
        compare_row_f("HPV prev (final)",    jres["hpv_prev"][end],    pres["hpv_prev"][end])
        compare_row_f("CIN prev (final)",    jres["cin_prev"][end],    pres["cin_prev"][end])
    end

    return jres
end

# ── Scenario 2: Multi-genotype HPV16 + HPV18 ──────────────────────────────

function scenario2()
    print_header("Scenario 2: Multi-genotype HPV16+18")

    sim = HPVSim(
        genotypes    = [
            GenotypeDef(:hpv16; init_prev=0.02),
            GenotypeDef(:hpv18; init_prev=0.015),
        ],
        n_agents     = N_AGENTS,
        start        = SIM_START,
        stop         = SIM_STOP,
        dt           = SIM_DT,
        rand_seed    = SEED,
        use_immunity = true,
        verbose      = 0,
    )
    Starsim.run!(sim)

    jres = extract_julia_results(sim)
    save_julia_results("scenario2", jres)

    pres = load_python_results("scenario2")

    println("\n  Julia results:")
    println("    Total HPV prev (final): $(@sprintf("%.4f", jres["hpv_prev"][end]))")
    println("    HPV16 prev (final): $(@sprintf("%.4f", jres["hpv_prev_hpv16"][end]))")
    println("    HPV18 prev (final): $(@sprintf("%.4f", jres["hpv_prev_hpv18"][end]))")
    println("    CIN prev (final): $(@sprintf("%.6f", jres["cin_prev"][end]))")

    if pres !== nothing
        println("\n  ── Comparison ──")
        print_comparison_header()
        compare_row_f("Total HPV prev (final)", jres["hpv_prev"][end],         pres["hpv_prev"][end])
        compare_row_f("HPV16 prev (final)",     jres["hpv_prev_hpv16"][end],   pres["hpv_prev_hpv16"][end])
        compare_row_f("HPV18 prev (final)",     jres["hpv_prev_hpv18"][end],   pres["hpv_prev_hpv18"][end])
        compare_row_f("CIN prev (final)",       jres["cin_prev"][end],         pres["cin_prev"][end])
    end

    return jres
end

# ── Scenario 3: Vaccination ───────────────────────────────────────────────

function scenario3()
    print_header("Scenario 3: HPV16+18 with vaccination")

    # NOTE: Julia HPVsim has a fixed agent population (no births/deaths).
    # Agents initialised at age 0–80 age over the simulation. In Python
    # hpvsim, demographics continuously inject young agents, so routine
    # vaccination of 9–14-year-olds works naturally.
    #
    # To produce a meaningful vaccination comparison, the Julia scenario
    # vaccinates from the start of the simulation (year 2000) and targets
    # ages 0–30 so agents are caught before they age out. This tests
    # whether the *mechanism* (vaccine → reduced susceptibility → lower
    # prevalence) works, even though the delivery strategy differs.

    genotype_defs = [
        GenotypeDef(:hpv16; init_prev=0.02),
        GenotypeDef(:hpv18; init_prev=0.015),
    ]

    # Baseline (no vaccination)
    sim_base = HPVSim(
        genotypes    = genotype_defs,
        n_agents     = N_AGENTS,
        start        = SIM_START,
        stop         = SIM_STOP,
        dt           = SIM_DT,
        rand_seed    = SEED,
        use_immunity = true,
        verbose      = 0,
    )
    Starsim.run!(sim_base)
    base_res = extract_julia_results(sim_base)

    # With vaccination — start from year 0, target wide age range
    # (compensates for no demographic turnover in Julia)
    vax = HPVVaccination(
        start_year        = SIM_START,
        covered_genotypes = [:hpv16, :hpv18],
        min_age           = 0.0,
        max_age           = 30.0,
        uptake_prob       = 0.9,
        sex               = :both,
    )
    sim_vx = HPVSim(
        genotypes     = genotype_defs,
        n_agents      = N_AGENTS,
        start         = SIM_START,
        stop          = SIM_STOP,
        dt            = SIM_DT,
        rand_seed     = SEED,
        use_immunity  = true,
        interventions = [vax],
        verbose       = 0,
    )
    Starsim.run!(sim_vx)
    vx_res = extract_julia_results(sim_vx)

    # Compute reduction
    base_final = base_res["hpv_prev"][end]
    vx_final   = vx_res["hpv_prev"][end]
    reduction  = base_final > 0 ? (1 - vx_final / base_final) * 100 : 0.0

    # Vaccination counts
    vax_mod = sim_vx.interventions[:hpv_vax]
    md_vax  = Starsim.module_data(vax_mod)
    n_vacc  = sum(md_vax.results[:n_vaccinated].values)

    vx_res["baseline_hpv_prev"] = base_res["hpv_prev"]
    vx_res["hpv_prev_reduction_pct"] = reduction
    vx_res["total_vaccinated"] = n_vacc
    save_julia_results("scenario3", vx_res)

    pres = load_python_results("scenario3")

    println("\n  Julia results:")
    println("    Baseline HPV prev (final): $(@sprintf("%.4f", base_final))")
    println("    Vaccinated HPV prev (final): $(@sprintf("%.4f", vx_final))")
    println("    Reduction: $(@sprintf("%.1f", reduction))%")
    println("    Total vaccinated: $(n_vacc)")

    if pres !== nothing
        py_reduction = pres["hpv_prev_reduction_pct"]
        println("\n  ── Comparison ──")
        print_comparison_header()
        compare_row_f("Baseline HPV prev",      base_final, pres["baseline_hpv_prev"][end])
        compare_row_f("Vaccinated HPV prev",     vx_final,   pres["hpv_prev"][end])
        compare_row_pct("HPV prev reduction",    reduction,  py_reduction)
    end

    return vx_res
end

# ── Scenario 4: Screening ─────────────────────────────────────────────────

function scenario4()
    print_header("Scenario 4: HPV16 with screening")

    genotype_defs = [GenotypeDef(:hpv16; init_prev=0.05)]

    # Baseline (no screening)
    sim_base = HPVSim(
        genotypes    = genotype_defs,
        n_agents     = N_AGENTS,
        start        = SIM_START,
        stop         = SIM_STOP,
        dt           = SIM_DT,
        rand_seed    = SEED,
        use_immunity = false,
        verbose      = 0,
    )
    Starsim.run!(sim_base)
    base_res = extract_julia_results(sim_base)

    # With screening starting at year 20
    scr = HPVScreening(
        test_type   = :hpv_dna,
        start_year  = SIM_START + 20,
        screen_prob = 0.10,
        min_age     = 25.0,
        max_age     = 65.0,
    )
    sim_scr = HPVSim(
        genotypes     = genotype_defs,
        n_agents      = N_AGENTS,
        start         = SIM_START,
        stop          = SIM_STOP,
        dt            = SIM_DT,
        rand_seed     = SEED,
        use_immunity  = false,
        interventions = [scr],
        verbose       = 0,
    )
    Starsim.run!(sim_scr)
    scr_res = extract_julia_results(sim_scr)

    # Screening module stats
    scr_mod = sim_scr.interventions[:hpv_screening]
    md_scr  = Starsim.module_data(scr_mod)
    total_screened = sum(md_scr.results[:n_screened].values)
    total_detected = sum(md_scr.results[:n_detected].values)
    total_treated  = sum(md_scr.results[:n_treated].values)

    # CIN reduction
    base_cin = base_res["cin_prev"][end]
    scr_cin  = scr_res["cin_prev"][end]
    cin_red  = base_cin > 0 ? (1 - scr_cin / base_cin) * 100 : 0.0

    scr_res["baseline_hpv_prev"]       = base_res["hpv_prev"]
    scr_res["baseline_cin_prev"]       = base_res["cin_prev"]
    scr_res["cin_prev_reduction_pct"]  = cin_red
    scr_res["total_screened"]          = total_screened
    scr_res["total_detected"]          = total_detected
    scr_res["total_treated"]           = total_treated
    save_julia_results("scenario4", scr_res)

    pres = load_python_results("scenario4")

    println("\n  Julia results:")
    println("    Baseline CIN prev (final): $(@sprintf("%.6f", base_cin))")
    println("    Screened CIN prev (final): $(@sprintf("%.6f", scr_cin))")
    println("    CIN reduction: $(@sprintf("%.1f", cin_red))%")
    println("    Total screened: $total_screened")
    println("    Total detected: $total_detected")
    println("    Total treated:  $total_treated")

    if pres !== nothing
        py_cin_red = pres["cin_prev_reduction_pct"]
        println("\n  ── Comparison ──")
        print_comparison_header()
        compare_row_f("Baseline CIN prev",  base_cin, pres["baseline_cin_prev"][end])
        compare_row_f("Screened CIN prev",   scr_cin,  pres["cin_prev"][end])
        compare_row_pct("CIN prev reduction", cin_red,  py_cin_red)
        compare_row_f("Baseline HPV prev",  base_res["hpv_prev"][end], pres["baseline_hpv_prev"][end])
        compare_row_f("Screened HPV prev",   scr_res["hpv_prev"][end],  pres["hpv_prev"][end])
    end

    return scr_res
end

# ── Grand summary ──────────────────────────────────────────────────────────

function print_grand_summary(r1, r2, r3, r4)
    pres1 = load_python_results("scenario1")
    pres2 = load_python_results("scenario2")
    pres3 = load_python_results("scenario3")
    pres4 = load_python_results("scenario4")

    has_python = pres1 !== nothing

    println("\n", "="^72)
    println("  CROSS-VALIDATION SUMMARY: Julia HPVsim vs Python hpvsim")
    println("="^72)
    println()
    println("  Note: Exact match NOT expected. The Julia and Python implementations")
    println("  differ in population dynamics (demographics vs fixed agents), network")
    println("  structure, and stochastic internals. We assess QUALITATIVE agreement:")
    println("  similar prevalence levels, correct trends, interventions showing effect.")
    println()

    if !has_python
        println("  ⚠ Python results not found. Run cross_validation.py first.")
        println("  Showing Julia-only results.\n")
    end

    # Table header
    println(rpad("Scenario / Metric", 40), rpad("Julia", 12), has_python ? rpad("Python", 12) : "", "Assessment")
    println("-"^(has_python ? 80 : 60))

    function row(label, jval; pval=nothing, fmt="%.4f", assessment="")
        jstr = @sprintf("%s", @eval @sprintf($fmt, $jval))
        if pval !== nothing
            pstr = @sprintf("%s", @eval @sprintf($fmt, $pval))
            println("  ", rpad(label, 38), rpad(jstr, 12), rpad(pstr, 12), assessment)
        else
            println("  ", rpad(label, 38), rpad(jstr, 12), assessment)
        end
    end

    # Scenario 1
    println("\n[1] Single HPV16")
    jv = r1["hpv_prev"][end]
    pv = has_python ? pres1["hpv_prev"][end] : nothing
    assess = has_python ? (0.05 < jv < 0.5 && 0.05 < pv < 0.5 ? "✓ Both show endemic HPV" : "?") : ""
    row("HPV prevalence (final)", jv; pval=pv, assessment=assess)

    jv = r1["cin_prev"][end]
    pv = has_python ? pres1["cin_prev"][end] : nothing
    assess = has_python ? (jv > 0 && pv > 0 ? "✓ Both show CIN" : "?") : ""
    row("CIN prevalence (final)", jv; pval=pv, fmt="%.6f", assessment=assess)

    # Scenario 2
    println("\n[2] Multi-genotype HPV16+18")
    jv = r2["hpv_prev"][end]
    pv = has_python ? pres2["hpv_prev"][end] : nothing
    assess = has_python ? (jv > 0 && pv > 0 ? "✓ Both show HPV" : "?") : ""
    row("Total HPV prev (final)", jv; pval=pv, assessment=assess)

    jv = r2["hpv_prev_hpv16"][end]
    pv = has_python ? pres2["hpv_prev_hpv16"][end] : nothing
    assess = has_python ? (jv > 0 && pv > 0 ? "✓ HPV16 present" : "?") : ""
    row("HPV16 prev (final)", jv; pval=pv, assessment=assess)

    jv16 = r2["hpv_prev_hpv16"][end]
    jv18 = r2["hpv_prev_hpv18"][end]
    if has_python
        pv16 = pres2["hpv_prev_hpv16"][end]
        pv18 = pres2["hpv_prev_hpv18"][end]
        j_higher = jv16 > jv18 ? "16" : "18"
        p_higher = pv16 > pv18 ? "16" : "18"
        assess = j_higher == p_higher ? "✓ Same dominant genotype" : "~ Different ranking"
        println("  ", rpad("Dominant genotype", 38), rpad("HPV$j_higher", 12), rpad("HPV$p_higher", 12), assess)
    end

    # Scenario 3
    println("\n[3] Vaccination")
    println("  (Julia: vax from year 0 ages 0-30; Python: routine vax ages 9-14 from year 20)")
    jred = r3["hpv_prev_reduction_pct"]
    pred = has_python ? pres3["hpv_prev_reduction_pct"] : nothing
    assess = has_python ? (jred > 0 && pred > 0 ? "✓ Both show vaccine effect" : (jred > 0 ? "~ Julia shows effect" : "✗ Julia shows no effect")) : ""
    row("HPV prev reduction (%)", jred; pval=pred, fmt="%.1f", assessment=assess)

    # Scenario 4
    println("\n[4] Screening")
    jred = r4["cin_prev_reduction_pct"]
    pred = has_python ? pres4["cin_prev_reduction_pct"] : nothing
    assess = has_python ? (jred > 0 && pred > 0 ? "✓ Both show screening effect" : (jred > 0 ? "~ Julia shows effect" : "✗")) : ""
    row("CIN prev reduction (%)", jred; pval=pred, fmt="%.1f", assessment=assess)

    jscr = r4["total_screened"]
    row("Total screened (Julia)", Float64(jscr); fmt="%.0f", assessment="")
    jtrt = r4["total_treated"]
    row("Total treated (Julia)", Float64(jtrt); fmt="%.0f", assessment="")

    println("\n", "="^(has_python ? 80 : 60))
    println("  QUALITATIVE CRITERIA:")
    println("    ✓ = Both models agree on the qualitative pattern")
    println("    ~ = Partial agreement (magnitudes differ, trend matches)")
    println("    ✗ = Disagreement requiring investigation")
    println()
    println("  KEY ARCHITECTURAL DIFFERENCES:")
    println("    • Python hpvsim: scaled population (~25k scale factor),")
    println("      location demographics (Nigeria default), complex sexual networks")
    println("    • Julia HPVsim: fixed-size agent population, simplified M/F network,")
    println("      rate-based CIN progression (no demographic scaling)")
    println("    • Absolute magnitudes WILL differ; trends should match")
    println("="^(has_python ? 80 : 60))
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    println("╔══════════════════════════════════════════════════════════════════════╗")
    println("║     Julia HPVsim Cross-Validation Against Python hpvsim            ║")
    println("║     N=$N_AGENTS agents, $(Int(SIM_STOP - SIM_START)) years, dt=$SIM_DT                            ║")
    println("╚══════════════════════════════════════════════════════════════════════╝")

    r1 = scenario1()
    r2 = scenario2()
    r3 = scenario3()
    r4 = scenario4()

    print_grand_summary(r1, r2, r3, r4)
end

main()
