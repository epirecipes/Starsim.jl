#!/usr/bin/env julia
#=
Cross-validation: Julia RotaABM vs Python rotasim
Runs matching scenarios and saves results as JSON for comparison.

Usage:
    cd code/Starsim.jl
    julia --project=extensions/RotaABM extensions/RotaABM/test/cross_validation.jl
=#

# ── Setup ──────────────────────────────────────────────────────────────────
starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using RotaABM
using Printf

const RESULTS_FILE = joinpath(@__DIR__, "crossval_results_julia.json")
const SEED = 42

# ── Helpers ────────────────────────────────────────────────────────────────

peak_prevalence(d) = maximum(Starsim.module_results(d)[:prevalence].values)

function mean_prevalence(d)
    prev = Starsim.module_results(d)[:prevalence].values
    return sum(prev) / length(prev)
end

function attack_rate(disease, n_agents)
    return sum(Starsim.module_results(disease)[:new_infections].values) / n_agents
end

function total_new_infections(sim)
    total = 0.0
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        total += sum(Starsim.module_results(d)[:new_infections].values)
    end
    return total
end

function count_active_strains(sim)
    n = 0
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        sum(Starsim.module_results(d)[:new_infections].values) > 0 && (n += 1)
    end
    return n
end

function cum_infections_by_age(sim; bins=[1.0, 5.0, Inf], labels=["<1y", "1-4y", "5+y"])
    active = sim.people.auids.values
    counts = zeros(Float64, length(bins))
    for u in active
        total_inf = 0.0
        for (_, d) in sim.diseases
            d isa Rotavirus || continue
            total_inf += d.n_infections.raw[u]
        end
        total_inf > 0 || continue
        age = sim.people.age.raw[u]
        idx = searchsortedfirst(bins, age)
        idx = min(idx, length(bins))
        counts[idx] += total_inf
    end
    return Dict(zip(labels, counts))
end

# ── Minimal JSON writer ───────────────────────────────────────────────────

function dict_to_json(d::Dict, indent=0)
    buf = IOBuffer()
    _write_json(buf, d, indent)
    return String(take!(buf))
end

function _write_json(io, d::Dict, indent)
    pad = " " ^ indent
    println(io, "{")
    ks = collect(keys(d))
    for (i, k) in enumerate(ks)
        print(io, pad, "  \"", k, "\": ")
        _write_json(io, d[k], indent + 2)
        i < length(ks) && print(io, ",")
        println(io)
    end
    print(io, pad, "}")
end
_write_json(io, v::Bool, _) = print(io, v ? "true" : "false")
_write_json(io, v::AbstractFloat, _) = @printf(io, "%.6f", v)
_write_json(io, v::Integer, _) = print(io, v)
_write_json(io, v::AbstractString, _) = print(io, "\"", v, "\"")
function _write_json(io, v::AbstractVector, indent)
    print(io, "[")
    for (i, x) in enumerate(v)
        _write_json(io, x, indent)
        i < length(v) && print(io, ", ")
    end
    print(io, "]")
end

# ── Scenario 1: Single strain ─────────────────────────────────────────────

function scenario1(; n_agents=5000, years=20)
    println("\n", "="^60)
    println("Scenario 1: Single strain (G1P8), $n_agents agents, $years years")
    println("="^60)

    sim = RotaSim(
        scenario = Dict{String,Any}(
            "strains" => Dict{Tuple{Int,Int}, Dict{String,Float64}}(
                (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.01)
            ),
            "default_fitness" => 1.0
        ),
        n_agents  = n_agents,
        stop      = Float64(years),
        dt        = Starsim.days(1),
        rand_seed = SEED,
        analyzers = [StrainStats(), EventStats()],
    )
    run!(sim; verbose=0)

    d = sim.diseases[:G1P8]
    pp = peak_prevalence(d)
    mp = mean_prevalence(d)
    ar = attack_rate(d, n_agents)

    println("  Peak prevalence:        $(round(pp, digits=4))")
    println("  Mean prevalence:        $(round(mp, digits=4))")
    println("  Cumulative attack rate: $(round(ar, digits=4))")

    return Dict{String,Any}(
        "peak_prevalence" => pp,
        "mean_prevalence" => mp,
        "attack_rate"     => ar,
    )
end

# ── Scenario 2: Multi-strain with reassortment ────────────────────────────

function scenario2(; n_agents=5000, years=20)
    println("\n", "="^60)
    println("Scenario 2: Multi-strain + reassortment, $n_agents agents, $years years")
    println("="^60)

    sim = RotaSim(
        scenario  = "simple",
        n_agents  = n_agents,
        stop      = Float64(years),
        dt        = Starsim.days(1),
        rand_seed = SEED,
        analyzers = [StrainStats(), EventStats()],
    )
    run!(sim; verbose=0)

    n_active = count_active_strains(sim)
    total_inf = total_new_infections(sim)
    total_ar = total_inf / n_agents

    # Per-strain peak prevalence
    strain_peaks = Dict{String,Float64}()
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        strain_peaks["G$(d.G)P$(d.P)"] = peak_prevalence(d)
    end

    # Total peak prevalence across strains
    npts = length(Starsim.module_results(first(values(sim.diseases)))[:prevalence].values)
    total_prev = zeros(npts)
    for (_, d) in sim.diseases
        d isa Rotavirus || continue
        total_prev .+= Starsim.module_results(d)[:prevalence].values
    end
    total_peak = maximum(total_prev)

    # Reassortment events
    reassort_total = 0.0
    for (_, c) in sim.connectors
        c isa RotaReassortmentConnector || continue
        reassort_total = sum(Starsim.module_results(c)[:n_reassortments].values)
    end

    println("  Active strains:         $n_active")
    println("  Total peak prevalence:  $(round(total_peak, digits=4))")
    println("  Total attack rate:      $(round(total_ar, digits=4))")
    println("  Reassortment events:    $(Int(reassort_total))")
    for (s, p) in sort(collect(strain_peaks), by=x->x[2], rev=true)
        println("    $s peak: $(round(p, digits=4))")
    end

    return Dict{String,Any}(
        "active_strains"      => n_active,
        "total_peak_prev"     => total_peak,
        "total_attack_rate"   => total_ar,
        "reassortment_events" => reassort_total,
    )
end

# ── Scenario 3: Vaccination (Rotarix) ─────────────────────────────────────

function scenario3(; n_agents=10000, years=20)
    println("\n", "="^60)
    println("Scenario 3: Vaccination (Rotarix at year 5), $n_agents agents, $years years")
    println("  Demographics enabled (births/deaths) for infant vaccination")
    println("="^60)

    demographics = [Starsim.Births(birth_rate=20.0), Starsim.Deaths(death_rate=10.0)]

    # --- No vaccination ---
    sim_novax = RotaSim(
        scenario  = "simple",
        n_agents  = n_agents,
        stop      = Float64(years),
        dt        = Starsim.days(1),
        rand_seed = SEED,
        analyzers = [StrainStats(), EventStats()],
        demographics = demographics,
    )
    run!(sim_novax; verbose=0)

    # --- With Rotarix at year 5 (fresh module instances) ---
    vax = Rotarix(start_year=5.0, uptake_prob=0.8)
    demographics2 = [Starsim.Births(birth_rate=20.0), Starsim.Deaths(death_rate=10.0)]
    sim_vax = RotaSim(
        scenario  = "simple",
        n_agents  = n_agents,
        stop      = Float64(years),
        dt        = Starsim.days(1),
        rand_seed = SEED,
        analyzers = [StrainStats(), EventStats()],
        demographics = demographics2,
        interventions = [vax],
    )
    run!(sim_vax; verbose=0)

    inf_novax = total_new_infections(sim_novax)
    inf_vax   = total_new_infections(sim_vax)
    reduction = inf_novax > 0 ? (inf_novax - inf_vax) / inf_novax * 100 : 0.0

    g1p8_novax_peak = peak_prevalence(sim_novax.diseases[:G1P8])
    g1p8_vax_peak   = peak_prevalence(sim_vax.diseases[:G1P8])
    g1p8_reduction  = g1p8_novax_peak > 0 ? (g1p8_novax_peak - g1p8_vax_peak) / g1p8_novax_peak * 100 : 0.0

    # Mean prevalence post-year-5
    npts = length(Starsim.module_results(first(values(sim_novax.diseases)))[:prevalence].values)
    post_vax_start = min(Int(round(5.0 / sim_novax.pars.dt)) + 1, npts)

    mean_prev_novax = 0.0; mean_prev_vax = 0.0
    for (_, d) in sim_novax.diseases
        d isa Rotavirus || continue
        prev = Starsim.module_results(d)[:prevalence].values
        mean_prev_novax += sum(prev[post_vax_start:end]) / length(prev[post_vax_start:end])
    end
    for (_, d) in sim_vax.diseases
        d isa Rotavirus || continue
        prev = Starsim.module_results(d)[:prevalence].values
        mean_prev_vax += sum(prev[post_vax_start:end]) / length(prev[post_vax_start:end])
    end
    mean_prev_reduction = mean_prev_novax > 0 ? (mean_prev_novax - mean_prev_vax) / mean_prev_novax * 100 : 0.0

    vax_summary = get_vaccination_summary(vax, sim_vax)

    println("  Total infections (no vax): $(Int(inf_novax))")
    println("  Total infections (vax):    $(Int(inf_vax))")
    println("  Overall reduction:         $(round(reduction, digits=1))%")
    println("  G1P8 peak (no vax):        $(round(g1p8_novax_peak, digits=4))")
    println("  G1P8 peak (vax):           $(round(g1p8_vax_peak, digits=4))")
    println("  G1P8 peak reduction:       $(round(g1p8_reduction, digits=1))%")
    println("  Mean prev post-Y5 (no vax):$(round(mean_prev_novax, digits=4))")
    println("  Mean prev post-Y5 (vax):   $(round(mean_prev_vax, digits=4))")
    println("  Mean prev reduction:       $(round(mean_prev_reduction, digits=1))%")
    println("  Vaccinated agents:         $(vax_summary["received_any_dose"])")
    println("  Completed schedule:        $(vax_summary["completed_schedule"])")

    return Dict{String,Any}(
        "infections_novax"          => inf_novax,
        "infections_vax"            => inf_vax,
        "overall_reduction_pct"     => reduction,
        "g1p8_peak_novax"           => g1p8_novax_peak,
        "g1p8_peak_vax"             => g1p8_vax_peak,
        "g1p8_peak_reduction_pct"   => g1p8_reduction,
        "mean_prev_reduction_pct"   => mean_prev_reduction,
        "vaccinated_agents"         => vax_summary["received_any_dose"],
        "completed_schedule"        => vax_summary["completed_schedule"],
    )
end

# ── Scenario 4: Age-structured ────────────────────────────────────────────

function scenario4(; n_agents=10000, years=10)
    println("\n", "="^60)
    println("Scenario 4: Age-structured, $n_agents agents, $years years")
    println("  Demographics enabled for age diversity")
    println("="^60)

    sim = RotaSim(
        scenario  = "simple",
        n_agents  = n_agents,
        stop      = Float64(years),
        dt        = Starsim.days(1),
        rand_seed = SEED,
        analyzers = [StrainStats(), EventStats(), AgeStats()],
        demographics = [Starsim.Births(birth_rate=20.0), Starsim.Deaths(death_rate=10.0)],
    )
    run!(sim; verbose=0)

    age_inf = cum_infections_by_age(sim; bins=[1.0, 5.0, Inf], labels=["<1y", "1-4y", "5+y"])
    total_inf = sum(values(age_inf))
    age_frac = Dict(k => total_inf > 0 ? v / total_inf : 0.0 for (k, v) in age_inf)

    active = sim.people.auids.values
    pop_bins = Dict("<1y" => 0, "1-4y" => 0, "5+y" => 0)
    for u in active
        age = sim.people.age.raw[u]
        if age < 1.0;       pop_bins["<1y"]  += 1
        elseif age < 5.0;   pop_bins["1-4y"] += 1
        else;                pop_bins["5+y"]  += 1
        end
    end
    pop_total = Float64(length(active))
    pop_frac = Dict(k => v / pop_total for (k, v) in pop_bins)

    age_ar = Dict(k => pop_bins[k] > 0 ? age_inf[k] / pop_bins[k] : 0.0 for k in ["<1y", "1-4y", "5+y"])

    println("  Cumulative infections by age:")
    for k in ["<1y", "1-4y", "5+y"]
        println("    $k: $(round(age_inf[k], digits=0)) ($(round(age_frac[k]*100, digits=1))% of total)")
    end
    println("  Population at end:")
    for k in ["<1y", "1-4y", "5+y"]
        println("    $k: $(pop_bins[k]) ($(round(pop_frac[k]*100, digits=1))%)")
    end
    println("  Attack rate by age:")
    for k in ["<1y", "1-4y", "5+y"]
        println("    $k: $(round(age_ar[k], digits=2))")
    end

    return Dict{String,Any}(
        "infection_frac_lt1"   => age_frac["<1y"],
        "infection_frac_1to4"  => age_frac["1-4y"],
        "infection_frac_5plus" => age_frac["5+y"],
        "attack_rate_lt1"      => age_ar["<1y"],
        "attack_rate_1to4"     => age_ar["1-4y"],
        "attack_rate_5plus"    => age_ar["5+y"],
        "pop_lt1"              => pop_bins["<1y"],
        "pop_1to4"             => pop_bins["1-4y"],
        "pop_5plus"            => pop_bins["5+y"],
    )
end

# ── Main ───────────────────────────────────────────────────────────────────

function main()
    println("Julia RotaABM Cross-Validation")
    println("="^60)

    t0 = time()
    r1 = scenario1()
    r2 = scenario2()
    r3 = scenario3()
    r4 = scenario4()
    elapsed = round(time() - t0, digits=1)
    println("\nTotal elapsed: $(elapsed)s")

    # Flatten strain_peaks for JSON
    s2_flat = Dict{String,Any}(
        "active_strains"      => r2["active_strains"],
        "total_peak_prev"     => r2["total_peak_prev"],
        "total_attack_rate"   => r2["total_attack_rate"],
        "reassortment_events" => r2["reassortment_events"],
    )
    if haskey(r2, "strain_peaks")
        for (k, v) in r2["strain_peaks"]
            s2_flat["peak_$k"] = v
        end
    end

    results = Dict{String,Any}(
        "scenario1" => r1,
        "scenario2" => s2_flat,
        "scenario3" => Dict{String,Any}(
            k => v for (k, v) in r3 if v isa Number
        ),
        "scenario4" => Dict{String,Any}(
            k => v for (k, v) in r4 if v isa Number
        ),
    )

    open(RESULTS_FILE, "w") do f
        write(f, dict_to_json(results))
    end
    println("\nResults saved to: $RESULTS_FILE")
end

main()
