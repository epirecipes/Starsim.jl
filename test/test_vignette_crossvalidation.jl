# Cross-validation tests: compare Julia Starsim.jl against Python starsim v3.2.2
# for key vignette scenarios.
#
# Loads pre-generated Python reference data (JSON), runs equivalent Julia
# simulations, and compares results quantitatively.  Different RNG streams
# make exact agreement impossible; tests use generous stochastic tolerances
# and focus on conservation laws, directional effects, and order-of-magnitude
# agreement in peak prevalence / attack rate.
#
# Usage:
#   julia --project=. test/test_vignette_crossvalidation.jl
#   # or include from runtests.jl

using Test
using Starsim
using JSON3
using Statistics
using Printf
using LinearAlgebra

const VREF_PATH = joinpath(@__DIR__, "python_ref", "vignette_reference.json")
const PYTHON    = joinpath(@__DIR__, "python_ref", ".venv", "bin", "python")
const GENSCRIPT = joinpath(@__DIR__, "python_ref", "generate_vignette_reference.py")

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Report a comparison line and return the relative difference."""
function compare(label::String, jl_val::Float64, py_val::Float64)
    if py_val == 0.0
        diff_pct = jl_val == 0.0 ? 0.0 : Inf
    else
        diff_pct = 100.0 * abs(jl_val - py_val) / abs(py_val)
    end
    @printf("    %-30s  Julia: %8.4f  Python: %8.4f  diff: %5.1f%%\n",
            label, jl_val, py_val, diff_pct)
    return diff_pct
end

"""Check that two values are close within relative tolerance, printing diagnostics."""
function check_close(label, jl_val, py_val; rtol=0.15, atol=0.0)
    compare(label, Float64(jl_val), Float64(py_val))
    if py_val == 0.0
        @test abs(jl_val) <= atol + 0.01
    else
        @test isapprox(jl_val, py_val; rtol=rtol, atol=atol)
    end
end

"""Print timing info."""
function print_elapsed(name::String, t0::Float64)
    elapsed = time() - t0
    @printf("    [%s completed in %.1fs]\n", name, elapsed)
end

# ── Generate / load reference data ──────────────────────────────────────────

function ensure_reference_data()
    if !isfile(VREF_PATH)
        if isfile(PYTHON) && isfile(GENSCRIPT)
            println("  Generating Python reference data...")
            run(`$PYTHON $GENSCRIPT`)
        else
            error("Python reference data not found at $VREF_PATH and cannot generate " *
                  "(need $PYTHON and $GENSCRIPT)")
        end
    end
    return JSON3.read(read(VREF_PATH, String))
end

# ── Test suite ──────────────────────────────────────────────────────────────

@testset "Vignette Cross-Validation (Julia ↔ Python)" begin

    ref = ensure_reference_data()
    println("  Reference data: starsim v$(ref[:starsim_version])")

    # ────────────────────────────────────────────────────────────────────────
    # V01: Introduction — Basic SIR
    # ────────────────────────────────────────────────────────────────────────
    @testset "V01 Introduction — Basic SIR" begin _t0 = time()
        py = ref[:v01_introduction]

        sim = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :sir, :prevalence)
        n_rec = get_result(sim, :sir, :n_recovered)
        n_sus = get_result(sim, :sir, :n_susceptible)
        n_inf = get_result(sim, :sir, :n_infected)

        jl_peak   = maximum(prev)
        jl_attack = n_rec[end] / 5000.0

        # Peak prevalence: both should show major epidemic
        check_close("peak prevalence", jl_peak, py[:peak_prevalence]; rtol=0.15)
        check_close("attack rate",     jl_attack, py[:attack_rate];   rtol=0.10)

        # Final state: epidemic should burn out
        @test prev[end] < 0.01  # SIR should burn out

        # Conservation: S + I + R ≈ N
        for i in 1:length(n_sus)
            total = n_sus[i] + n_inf[i] + n_rec[i]
            @test abs(total - 5000.0) < 2.0
        end

        # Peak timing: should be in similar range (within ±15 timesteps)
        jl_peak_day = argmax(prev) - 1  # 0-indexed to match Python
        @test abs(jl_peak_day - py[:peak_day]) <= 15
        @printf("    %-30s  Julia: %8d     Python: %8d\n",
                "peak day", jl_peak_day, py[:peak_day])

        # Prevalence curve correlation
        py_prev = Float64.(py[:prevalence])
        n = min(length(prev), length(py_prev))
        corr = cor(prev[1:n], py_prev[1:n])
        @printf("    %-30s  %.4f\n", "prevalence correlation", corr)
        @test corr > 0.80  # Prevalence curves should be highly correlated
        print_elapsed("V01 Introduction — Basic SIR", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V02: Building a Model — SIR with higher beta
    # ────────────────────────────────────────────────────────────────────────
    @testset "V02 Building a Model — SIR β=0.1" begin _t0 = time()
        py = ref[:v02_building_a_model]

        sim = Sim(
            n_agents  = 2000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.1, dur_inf=10.0, init_prev=0.02),
            dt        = 1.0,
            stop      = 180.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :sir, :prevalence)
        n_rec = get_result(sim, :sir, :n_recovered)

        jl_peak   = maximum(prev)
        jl_attack = n_rec[end] / 2000.0

        check_close("peak prevalence", jl_peak,   py[:peak_prevalence]; rtol=0.10)
        check_close("attack rate",     jl_attack, py[:attack_rate];     rtol=0.10)

        # Prevalence curve shape: should rise then fall
        mid = length(prev) ÷ 2
        @test maximum(prev[1:mid]) > prev[end]  # Epidemic should peak then decline

        # Compare prevalence time series (correlation)
        py_prev = Float64.(py[:prevalence])
        n = min(length(prev), length(py_prev))
        corr = cor(prev[1:n], py_prev[1:n])
        @printf("    %-30s  %.4f\n", "prevalence correlation", corr)
        @test corr > 0.85  # Prevalence curves should be highly correlated
        print_elapsed("V02 Building a Model — SIR β=0.1", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V03: Demographics — SIR with births/deaths
    # ────────────────────────────────────────────────────────────────────────
    @testset "V03 Demographics — Births & Deaths" begin _t0 = time()
        py = ref[:v03_demographics]

        # Baseline (no demographics)
        sim_base = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_base)
        jl_base_peak = maximum(get_result(sim_base, :sir, :prevalence))
        check_close("base peak prevalence", jl_base_peak, py[:base_peak_prevalence]; rtol=0.15)

        # With demographics (birth_rate=20, death_rate=15)
        sim_demo = Sim(
            n_agents     = 5000,
            networks     = RandomNet(n_contacts=10),
            diseases     = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            demographics = [Births(birth_rate=20.0), Deaths(death_rate=15.0)],
            dt           = 1.0,
            stop         = 365.0,
            rand_seed    = 42,
            verbose      = 0,
        )
        run!(sim_demo)

        jl_alive  = get_result(sim_demo, :n_alive)[end]
        jl_births = sum(get_result(sim_demo, :births))
        jl_deaths = sum(get_result(sim_demo, :deaths))

        # Population should grow (births > deaths with these rates)
        @test jl_alive > 5000.0  # Population should grow with birth_rate=20, death_rate=15
        check_close("final population", jl_alive,  py[:demo_final_alive];  rtol=0.15)
        check_close("total births",     jl_births, py[:demo_total_births]; rtol=0.15)
        check_close("total deaths",     jl_deaths, py[:demo_total_deaths]; rtol=0.15)

        # Growth vs decline direction (short duration to avoid population explosion)
        sim_grow = Sim(
            n_agents     = 5000,
            networks     = RandomNet(n_contacts=10),
            diseases     = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            demographics = [Births(birth_rate=30.0), Deaths(death_rate=10.0)],
            dt           = 1.0,
            stop         = 50.0,
            rand_seed    = 42,
            verbose      = 0,
        )
        run!(sim_grow)

        sim_decline = Sim(
            n_agents     = 5000,
            networks     = RandomNet(n_contacts=10),
            diseases     = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            demographics = [Births(birth_rate=10.0), Deaths(death_rate=30.0)],
            dt           = 1.0,
            stop         = 50.0,
            rand_seed    = 42,
            verbose      = 0,
        )
        run!(sim_decline)

        jl_grow_alive    = get_result(sim_grow, :n_alive)[end]
        jl_decline_alive = get_result(sim_decline, :n_alive)[end]

        # Direction: grow > initial > decline
        @test jl_grow_alive > 5000.0  # Growth scenario should increase population
        @test jl_decline_alive < 5000.0  # Decline scenario should decrease population
        @test jl_grow_alive > jl_decline_alive

        check_close("growth final pop",  jl_grow_alive,    py[:grow_final_alive];    rtol=0.20)
        check_close("decline final pop", jl_decline_alive, py[:decline_final_alive]; rtol=0.20)
        print_elapsed("V03 Demographics — Births & Deaths", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V04: Diseases — Multiple disease types
    # ────────────────────────────────────────────────────────────────────────
    @testset "V04 Diseases — SIR vs SIS" begin _t0 = time()
        py = ref[:v04_diseases]

        # SIR
        sim_sir = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_sir)
        jl_sir_peak   = maximum(get_result(sim_sir, :sir, :prevalence))
        jl_sir_attack = get_result(sim_sir, :sir, :n_recovered)[end] / 5000.0
        check_close("SIR peak prevalence", jl_sir_peak,   py[:sir_peak_prevalence]; rtol=0.15)
        check_close("SIR attack rate",     jl_sir_attack, py[:sir_attack_rate];     rtol=0.10)

        # SIS
        sim_sis = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIS(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_sis)
        sis_prev = get_result(sim_sis, :sis, :prevalence)
        jl_sis_peak  = maximum(sis_prev)
        jl_sis_final = sis_prev[end]
        check_close("SIS peak prevalence",  jl_sis_peak,  py[:sis_peak_prevalence];  rtol=0.15)
        check_close("SIS final prevalence", jl_sis_final, py[:sis_final_prevalence]; rtol=0.25, atol=0.40)

        # SIS should reach endemic equilibrium
        @test jl_sis_final > 0.05  # SIS should have endemic equilibrium

        # SIR should burn out, SIS should persist
        sir_final = get_result(sim_sir, :sir, :prevalence)[end]
        @test sir_final < jl_sis_final  # SIR burns out while SIS persists
        print_elapsed("V04 Diseases — SIR vs SIS", _t0)
    end

    @testset "V04 Diseases — Multi-disease" begin _t0 = time()
        py = ref[:v04_diseases]

        sim_multi = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = [
                SIR(name=:flu,  beta=0.08, dur_inf=7.0, init_prev=0.01),
                SIS(name=:cold, beta=0.05, dur_inf=5.0, init_prev=0.02),
            ],
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_multi)

        jl_flu_peak   = maximum(get_result(sim_multi, :flu, :prevalence))
        cold_prev     = get_result(sim_multi, :cold, :prevalence)
        jl_cold_peak  = maximum(cold_prev)
        jl_cold_final = cold_prev[end]

        check_close("flu peak prevalence",   jl_flu_peak,   py[:multi_flu_peak];  rtol=0.15)
        check_close("cold peak prevalence",  jl_cold_peak,  py[:multi_cold_peak]; rtol=0.20, atol=0.25)
        check_close("cold final prevalence", jl_cold_final, py[:multi_cold_final]; rtol=0.30, atol=0.45)

        # Flu (SIR, higher beta) should peak higher than cold (SIS, lower beta)
        @test jl_flu_peak > jl_cold_peak  # Flu should peak higher than cold
        print_elapsed("V04 Diseases — Multi-disease", _t0)
    end

    @testset "V04 Diseases — Beta sensitivity" begin _t0 = time()
        py = ref[:v04_diseases]
        py_peaks = py[:beta_peaks]

        jl_peaks = Dict{String,Float64}()
        for b in [0.02, 0.05, 0.10]
            local sim = Sim(
                n_agents  = 5000,
                networks  = RandomNet(n_contacts=10),
                diseases  = SIR(beta=b, dur_inf=10.0, init_prev=0.01),
                dt        = 1.0,
                stop      = 365.0,
                rand_seed = 42,
                verbose   = 0,
            )
            run!(sim)
            local peak = maximum(get_result(sim, :sir, :prevalence))
            jl_peaks[string(b)] = peak
            check_close("beta=$(b) peak prev", peak, Float64(py_peaks[Symbol(string(b))]); rtol=0.15, atol=0.20)
        end

        # Monotonicity: higher beta -> higher peak
        @test jl_peaks["0.05"] > jl_peaks["0.02"]
        @test jl_peaks["0.1"]  > jl_peaks["0.05"]
        print_elapsed("V04 Diseases — Beta sensitivity", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V05: Networks — Different network types
    # ────────────────────────────────────────────────────────────────────────
    @testset "V05 Networks — RandomNet vs StaticNet" begin _t0 = time()
        py = ref[:v05_networks]

        # RandomNet
        sim_random = Sim(
            n_agents  = 2000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_random)
        jl_random_peak   = maximum(get_result(sim_random, :sir, :prevalence))
        jl_random_attack = get_result(sim_random, :sir, :n_recovered)[end] / 2000.0
        check_close("RandomNet peak prev",   jl_random_peak,   py[:random][:peak_prevalence]; rtol=0.20)
        check_close("RandomNet attack rate", jl_random_attack, py[:random][:attack_rate];     rtol=0.10)

        # StaticNet
        sim_static = Sim(
            n_agents  = 2000,
            networks  = StaticNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_static)
        jl_static_peak   = maximum(get_result(sim_static, :sir, :prevalence))
        jl_static_attack = get_result(sim_static, :sir, :n_recovered)[end] / 2000.0
        check_close("StaticNet peak prev",   jl_static_peak,   py[:static][:peak_prevalence]; rtol=0.20)
        check_close("StaticNet attack rate", jl_static_attack, py[:static][:attack_rate];     rtol=0.10)

        # Both should produce substantial epidemics
        @test jl_random_peak > 0.3  # RandomNet should produce substantial epidemic
        @test jl_static_peak > 0.3  # StaticNet should produce substantial epidemic
        print_elapsed("V05 Networks — RandomNet vs StaticNet", _t0)
    end

    @testset "V05 Networks — MixingPool" begin _t0 = time()
        py = ref[:v05_networks]

        sim_mp = Sim(
            n_agents  = 2000,
            networks  = MixingPool(contact_rate=10.0),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_mp)
        jl_mp_peak   = maximum(get_result(sim_mp, :sir, :prevalence))
        jl_mp_attack = get_result(sim_mp, :sir, :n_recovered)[end] / 2000.0
        check_close("MixingPool peak prev",   jl_mp_peak,   py[:mixingpool][:peak_prevalence]; rtol=0.15)
        check_close("MixingPool attack rate", jl_mp_attack, py[:mixingpool][:attack_rate];     rtol=0.10)

        @test jl_mp_peak > 0.3  # MixingPool should produce substantial epidemic
        print_elapsed("V05 Networks — MixingPool", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V12: MultiSim — Mean prevalence across runs
    # ────────────────────────────────────────────────────────────────────────
    @testset "V12 MultiSim — Mean prevalence" begin _t0 = time()
        py = ref[:v12_multisim]

        base = Sim(
            n_agents  = 3000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        msim = MultiSim(base; n_runs=5)
        run!(msim; verbose=0)

        m = mean_result(msim, :sir_prevalence)
        jl_mean_peak = maximum(m)

        check_close("mean peak prevalence", jl_mean_peak, py[:mean_peak_prevalence]; rtol=0.20)

        # Mean attack rate
        jl_attacks = Float64[]
        for s in msim.sims
            local rec = get_result(s, :sir, :n_recovered)
            push!(jl_attacks, rec[end] / 3000.0)
        end
        jl_mean_attack = mean(jl_attacks)
        check_close("mean attack rate", jl_mean_attack, py[:mean_attack_rate]; rtol=0.10)

        # Individual peak variability
        jl_peaks = [maximum(get_result(s, :sir, :prevalence)) for s in msim.sims]
        py_peaks = Float64.(py[:individual_peaks])
        @printf("    %-30s  Julia: %s\n", "individual peaks",
                join([@sprintf("%.3f", p) for p in jl_peaks], ", "))
        @printf("    %-30s  Python: %s\n", "",
                join([@sprintf("%.3f", p) for p in py_peaks], ", "))

        # Runs should be consistent (CV < 10%)
        cv = std(jl_peaks) / mean(jl_peaks)
        @test cv < 0.10  # MultiSim peaks should be consistent (CV=$(round(cv, digits=3)))

        # Compare ranges
        jl_range = maximum(jl_peaks) - minimum(jl_peaks)
        py_range = maximum(py_peaks) - minimum(py_peaks)
        @printf("    %-30s  Julia: %.4f  Python: %.4f\n", "peak range", jl_range, py_range)
        print_elapsed("V12 MultiSim — Mean prevalence", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V13: Calibration — Grid search recovers known beta
    # ────────────────────────────────────────────────────────────────────────
    @testset "V13 Calibration — Grid search" begin _t0 = time()
        py = ref[:v13_calibration]

        # Generate target data
        target_sim = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(target_sim)
        target = get_result(target_sim, :sir, :prevalence)

        # Grid search
        betas  = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
        losses = Float64[]
        for b in betas
            local sim = Sim(
                n_agents  = 5000,
                networks  = RandomNet(n_contacts=10),
                diseases  = SIR(beta=b, dur_inf=10.0, init_prev=0.01),
                dt        = 1.0,
                stop      = 200.0,
                rand_seed = 42,
                verbose   = 0,
            )
            run!(sim)
            local p = get_result(sim, :sir, :prevalence)
            local n = min(length(p), length(target))
            local mse = mean((p[1:n] .- target[1:n]).^2)
            push!(losses, mse)
            @printf("      beta=%.2f: MSE=%.6f\n", b, mse)
        end

        best_idx  = argmin(losses)
        best_beta = betas[best_idx]
        best_mse  = losses[best_idx]

        @printf("    %-30s  Julia: %.2f  Python: %.2f\n",
                "best beta", best_beta, py[:best_beta])

        # Best beta should be 0.05 (matches target)
        @test best_beta == py[:true_beta]  # Grid search should recover true beta
        @test best_mse < 1e-6  # Self-match MSE should be ~0

        # Both should have minimum at same beta
        py_losses = Float64.(py[:losses])
        py_best_idx = argmin(py_losses)
        @test betas[best_idx] == py[:betas][py_best_idx]  # Both find same optimal beta

        # Loss landscape shape (rank correlation)
        jl_ranks = sortperm(sortperm(losses))
        py_ranks = sortperm(sortperm(py_losses))
        rank_corr = cor(Float64.(jl_ranks), Float64.(py_ranks))
        @printf("    %-30s  %.4f\n", "loss rank correlation", rank_corr)
        @test rank_corr > 0.8  # Loss landscape should have similar shape
        print_elapsed("V13 Calibration — Grid search", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V06: Interventions — Routine Vaccination
    # ────────────────────────────────────────────────────────────────────────
    @testset "V06 Interventions — Routine Vaccination" begin _t0 = time()
        py = ref[:v06_interventions]

        # Baseline (no intervention)
        sim_base = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_base)

        # Routine vaccination (efficacy=0.9, prob=0.01 per day)
        vaccine = Vx(efficacy=0.9)
        routine = RoutineDelivery(product=vaccine, prob=0.01, disease_name=:sir)
        sim_vx = Sim(
            n_agents      = 5000,
            networks      = RandomNet(n_contacts=10),
            diseases      = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            interventions = routine,
            dt            = 1.0,
            stop          = 365.0,
            rand_seed     = 42,
            verbose       = 0,
        )
        run!(sim_vx)

        prev_base = get_result(sim_base, :sir, :prevalence)
        prev_vx   = get_result(sim_vx, :sir, :prevalence)

        jl_base_peak   = maximum(prev_base)
        jl_vx_peak     = maximum(prev_vx)
        jl_base_attack = get_result(sim_base, :sir, :n_recovered)[end] / 5000.0
        jl_vx_attack   = get_result(sim_vx, :sir, :n_recovered)[end] / 5000.0

        check_close("base peak prevalence", jl_base_peak,   py[:base_peak_prevalence]; rtol=0.15)
        check_close("vx peak prevalence",   jl_vx_peak,     py[:vx_peak_prevalence];   rtol=0.20)
        check_close("base attack rate",     jl_base_attack, py[:base_attack_rate];     rtol=0.10)
        check_close("vx attack rate",       jl_vx_attack,   py[:vx_attack_rate];       rtol=0.15)

        # Directional: vaccination should reduce the epidemic
        @test jl_vx_peak < jl_base_peak       # Vaccine reduces peak
        @test jl_vx_attack <= jl_base_attack   # Vaccine reduces attack rate

        # Prevalence curves should correlate with Python
        py_base_prev = Float64.(py[:base_prevalence])
        n = min(length(prev_base), length(py_base_prev))
        corr = cor(prev_base[1:n], py_base_prev[1:n])
        @printf("    %-30s  %.4f\n", "base prevalence correlation", corr)
        @test corr > 0.80
        print_elapsed("V06 Interventions — Routine Vaccination", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V07: Connectors — Seasonality
    # ────────────────────────────────────────────────────────────────────────
    @testset "V07 Connectors — Seasonality" begin _t0 = time()
        py = ref[:v07_connectors]

        # SIS baseline (constant beta, 2 years)
        sim_base = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIS(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 730.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_base)

        # SIS with seasonality (amplitude=0.5, peak_day=0)
        seasonal = Seasonality(disease_name=:sis, amplitude=0.5, peak_day=0)
        sim_s = Sim(
            n_agents   = 5000,
            networks   = RandomNet(n_contacts=10),
            diseases   = SIS(beta=0.05, dur_inf=10.0, init_prev=0.01),
            connectors = seasonal,
            dt         = 1.0,
            stop       = 730.0,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim_s)

        prev_base = get_result(sim_base, :sis, :prevalence)
        prev_s    = get_result(sim_s, :sis, :prevalence)

        jl_base_peak      = maximum(prev_base)
        jl_base_final     = prev_base[end]
        jl_seasonal_peak  = maximum(prev_s)
        jl_seasonal_final = prev_s[end]

        check_close("base peak prevalence",     jl_base_peak,     py[:base_peak_prevalence];     rtol=0.15)
        check_close("base final prevalence",    jl_base_final,    py[:base_final_prevalence];    rtol=0.30, atol=0.40)
        check_close("seasonal peak prevalence", jl_seasonal_peak, py[:seasonal_peak_prevalence]; rtol=0.20)

        # SIS should reach endemic equilibrium (prevalence > 0.05 at end)
        @test jl_base_final > 0.05  # SIS endemic equilibrium

        # Both should produce substantial epidemics
        @test jl_base_peak > 0.10     # Baseline should have epidemic
        @test jl_seasonal_peak > 0.10 # Seasonal should have epidemic

        # Prevalence correlation with Python baselines
        py_base_prev = Float64.(py[:base_prevalence])
        n = min(length(prev_base), length(py_base_prev))
        corr = cor(prev_base[1:n], py_base_prev[1:n])
        @printf("    %-30s  %.4f\n", "base prevalence correlation", corr)
        @test corr > 0.70
        print_elapsed("V07 Connectors — Seasonality", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V08: Measles — SEIR Outbreak
    # ────────────────────────────────────────────────────────────────────────
    @testset "V08 Measles — SEIR Outbreak" begin _t0 = time()
        py = ref[:v08_measles]

        # Julia uses SEIR; Python reference uses SIR(dur_inf=19=8+11)
        sim = Sim(
            n_agents  = 10000,
            networks  = RandomNet(n_contacts=15),
            diseases  = SEIR(beta=0.3, dur_exp=8.0, dur_inf=11.0, init_prev=0.001),
            dt        = 1.0,
            stop      = 180.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :seir, :prevalence)
        n_rec = get_result(sim, :seir, :n_recovered)
        n_sus = get_result(sim, :seir, :n_susceptible)
        n_exp = get_result(sim, :seir, :n_exposed)
        n_inf = get_result(sim, :seir, :n_infected)

        jl_peak     = maximum(prev)
        jl_peak_day = argmax(prev) - 1  # 0-indexed
        jl_attack   = n_rec[end] / 10000.0

        # Structural: both should produce large measles-like epidemics
        @test jl_peak > 0.10      # Large epidemic expected
        @test jl_attack > 0.50    # High attack rate for measles R₀

        # SEIR vs SIR: generous comparison (different models)
        check_close("peak prevalence", jl_peak,   py[:peak_prevalence]; rtol=0.50, atol=0.30)
        check_close("attack rate",     jl_attack, py[:attack_rate];     rtol=0.30)

        # Peak timing
        @printf("    %-30s  Julia SEIR: %8d     Python SIR: %8d\n",
                "peak day", jl_peak_day, py[:peak_day])

        # Conservation: S + E + I + R ≈ N
        for i in 1:length(n_sus)
            local total = n_sus[i] + n_exp[i] + n_inf[i] + n_rec[i]
            @test abs(total - 10000.0) < 2.0
        end

        # Exposed peak should precede infected peak
        exp_peak_day = argmax(n_exp)
        inf_peak_day = argmax(n_inf)
        @printf("    %-30s  exposed: %d  infected: %d\n",
                "peak timestep", exp_peak_day, inf_peak_day)
        @test exp_peak_day <= inf_peak_day
        print_elapsed("V08 Measles — SEIR Outbreak", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V09: Cholera — High-Beta Epidemic (SIR proxy)
    # ────────────────────────────────────────────────────────────────────────
    @testset "V09 Cholera — High-Beta Epidemic" begin _t0 = time()
        py = ref[:v09_cholera]

        # Simplified SIR proxy matching cholera-scale dynamics
        sim = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=8),
            diseases  = SIR(beta=0.5, dur_inf=5.0, init_prev=0.005),
            dt        = 1.0,
            stop      = 200.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :sir, :prevalence)
        n_rec = get_result(sim, :sir, :n_recovered)
        n_sus = get_result(sim, :sir, :n_susceptible)
        n_inf = get_result(sim, :sir, :n_infected)

        jl_peak     = maximum(prev)
        jl_peak_day = argmax(prev) - 1
        jl_attack   = n_rec[end] / 5000.0

        check_close("peak prevalence", jl_peak,   py[:peak_prevalence]; rtol=0.15)
        check_close("attack rate",     jl_attack, py[:attack_rate];     rtol=0.10)

        # Structural: high-beta should produce fast, large epidemic
        @test jl_peak > 0.10
        @test jl_attack > 0.50
        @test jl_peak_day < 50  # Fast epidemic with high beta

        # Epidemic should burn out
        @test prev[end] < 0.01

        # Peak timing
        @printf("    %-30s  Julia: %8d     Python: %8d\n",
                "peak day", jl_peak_day, py[:peak_day])
        @test abs(jl_peak_day - py[:peak_day]) <= 10

        # Conservation: S + I + R ≈ N
        for i in 1:length(n_sus)
            local total = n_sus[i] + n_inf[i] + n_rec[i]
            @test abs(total - 5000.0) < 2.0
        end

        # Prevalence correlation
        py_prev = Float64.(py[:prevalence])
        n = min(length(prev), length(py_prev))
        corr = cor(prev[1:n], py_prev[1:n])
        @printf("    %-30s  %.4f\n", "prevalence correlation", corr)
        @test corr > 0.85
        print_elapsed("V09 Cholera — High-Beta Epidemic", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V10: Ebola — Slow-Spread Epidemic (SIR proxy)
    # ────────────────────────────────────────────────────────────────────────
    @testset "V10 Ebola — Slow-Spread Epidemic" begin _t0 = time()
        py = ref[:v10_ebola]

        # Simplified SIR proxy matching ebola-scale dynamics
        sim = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=5),
            diseases  = SIR(beta=0.5, dur_inf=20.0, init_prev=0.005),
            dt        = 1.0,
            stop      = 300.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :sir, :prevalence)
        n_rec = get_result(sim, :sir, :n_recovered)
        n_sus = get_result(sim, :sir, :n_susceptible)
        n_inf = get_result(sim, :sir, :n_infected)

        jl_peak     = maximum(prev)
        jl_peak_day = argmax(prev) - 1
        jl_attack   = n_rec[end] / 5000.0

        check_close("peak prevalence", jl_peak,   py[:peak_prevalence]; rtol=0.15)
        check_close("attack rate",     jl_attack, py[:attack_rate];     rtol=0.10)

        # Structural: epidemic should occur and burn out
        @test jl_peak > 0.10
        @test jl_attack > 0.30
        @test prev[end] < 0.05

        # Peak timing
        @printf("    %-30s  Julia: %8d     Python: %8d\n",
                "peak day", jl_peak_day, py[:peak_day])
        @test abs(jl_peak_day - py[:peak_day]) <= 15

        # Conservation
        for i in 1:length(n_sus)
            local total = n_sus[i] + n_inf[i] + n_rec[i]
            @test abs(total - 5000.0) < 2.0
        end

        # Prevalence correlation
        py_prev = Float64.(py[:prevalence])
        n = min(length(prev), length(py_prev))
        corr = cor(prev[1:n], py_prev[1:n])
        @printf("    %-30s  %.4f\n", "prevalence correlation", corr)
        @test corr > 0.85
        print_elapsed("V10 Ebola — Slow-Spread Epidemic", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V11: HIV — SIR on MFNet Sexual Network
    # ────────────────────────────────────────────────────────────────────────
    @testset "V11 HIV — SIR on MFNet" begin _t0 = time()
        py = ref[:v11_hiv]

        sim = Sim(
            n_agents  = 5000,
            networks  = MFNet(),
            diseases  = SIR(beta=Dict(:mf => 0.08), dur_inf=200.0, init_prev=0.05),
            dt        = 1.0,
            stop      = 365.0 * 5,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        prev  = get_result(sim, :sir, :prevalence)
        n_rec = get_result(sim, :sir, :n_recovered)

        jl_peak   = maximum(prev)
        jl_attack = n_rec[end] / 5000.0
        jl_final  = prev[end]

        # MFNet implementations differ between Julia and Python (different
        # partnership formation, contact rates), so use generous tolerances
        # and focus on structural properties.
        check_close("peak prevalence", jl_peak,   py[:peak_prevalence]; rtol=0.50, atol=0.35)
        check_close("attack rate",     jl_attack, py[:attack_rate];     rtol=0.50, atol=0.35)

        # Structural: MFNet should produce a substantial epidemic
        @test jl_peak > 0.05    # Epidemic occurs on sexual network
        @test jl_attack > 0.10  # Non-trivial attack rate

        # SIR should eventually burn out
        @test jl_final < 0.05

        # Prevalence curve shape: should rise then fall
        mid = length(prev) ÷ 4
        @test maximum(prev[1:mid]) > prev[end]  # Early peak, then decline
        print_elapsed("V11 HIV — SIR on MFNet", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V14: Malaria — Multi-patch Ross-Macdonald ODE
    # ────────────────────────────────────────────────────────────────────────
    @testset "V14 Malaria — Multi-patch ODE" begin _t0 = time()
        py = ref[:v14_malaria]

        # ── MalariaODE module (deterministic, from vignette 14) ──
        function _analytical_mosquito_density(ivector, hvector, pij, a, b, c, mu, r, tau)
            xvector = ivector ./ r
            gvector = xvector .* r ./ (1 .- xvector)
            n = length(xvector)
            fvector = zeros(n)
            for i in 1:n
                k_i = sum(pij[:, i] .* xvector .* hvector) / sum(pij[:, i] .* hvector)
                fvector[i] = b * c * k_i / (a * c * k_i / mu + 1)
            end
            fmatrix = Diagonal(fvector)
            cvector = (pij * fmatrix) \ gvector
            return cvector .* mu ./ a^2 ./ exp(-mu * tau)
        end

        function _seasonal_sinusoidal(t; amplitude=0.8, peak_day=180)
            day_of_year = t % 365
            return max(0.0, 1 + amplitude * sin(2π * (day_of_year - peak_day + 365/4) / 365))
        end

        # Direct Euler integration (no starsim wrapper — pure math)
        a, b_param, c_param = 0.3, 0.1, 0.214
        r, mu, tau = 1.0/150.0, 1.0/10.0, 10.0
        hvector = [5000.0, 10000.0, 8000.0]
        ivector = [0.001, 0.003, 0.002]
        pij = [0.85 0.10 0.05;
               0.08 0.80 0.12;
               0.06 0.09 0.85]

        m_base = _analytical_mosquito_density(ivector, hvector, pij, a, b_param, c_param, mu, r, tau)
        m_base = max.(m_base, 1e-10)

        X = ivector ./ r
        C_cum = zeros(3)
        nsteps = 365 * 5
        dt_ode = 1.0

        X_history = zeros(nsteps + 1, 3)
        X_history[1, :] .= X

        for step in 1:nsteps
            t = (step - 1) * dt_ode
            ms = m_base .* _seasonal_sinusoidal(t)
            k = (pij' * (X .* hvector)) ./ (pij' * hvector)
            Z_numer = a^2 * b_param * c_param * exp(-mu * tau) .* k
            Z_denom = a * c_param .* k .+ mu
            dC = pij * (ms .* Z_numer ./ Z_denom) .* (1 .- X)
            dX = dC .- r .* X
            X .+= dX .* dt_ode
            C_cum .+= dC .* dt_ode
            X_history[step + 1, :] .= X
        end

        py_final = Float64.(py[:final_prevalence])
        py_peak  = Float64.(py[:peak_prevalence])
        py_mean  = Float64.(py[:mean_prevalence])
        py_cum   = Float64.(py[:final_cumulative])

        # Deterministic ODE: tight tolerance (same algorithm, same params)
        for i in 1:3
            check_close("patch $i final prevalence", X[i],                          py_final[i]; rtol=0.01)
            check_close("patch $i peak prevalence",  maximum(X_history[:, i]),       py_peak[i];  rtol=0.01)
            check_close("patch $i mean prevalence",  mean(X_history[:, i]),          py_mean[i];  rtol=0.01)
            check_close("patch $i cumul. incidence",  C_cum[i],                      py_cum[i];   rtol=0.01)
        end

        # Structural: all patches should have non-zero prevalence
        for i in 1:3
            @test X[i] > 0.01  # Malaria persists in all patches
        end

        # Seasonal oscillation: prevalence should vary over time
        for i in 1:3
            local pmax = maximum(X_history[:, i])
            local pmin = minimum(X_history[365:end, i])  # After initial transient
            @test pmax > 1.5 * pmin  # Seasonal variation creates substantial oscillation
        end
        print_elapsed("V14 Malaria — Multi-patch ODE", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # V04 bonus: SEIR model (Julia built-in)
    # ────────────────────────────────────────────────────────────────────────
    @testset "V04 SEIR — Peak timing & magnitude" begin _t0 = time()
        # SEIR is built into Julia but not Python starsim v3.2.2.
        # Verify structural properties: latent period delays and reduces peak.
        sim = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SEIR(beta=0.05, dur_exp=5.0, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim)

        n_exp = get_result(sim, :seir, :n_exposed)
        n_inf = get_result(sim, :seir, :n_infected)
        prev  = get_result(sim, :seir, :prevalence)

        # Exposed peak should precede infected peak (latent period effect)
        exp_peak = argmax(n_exp)
        inf_peak = argmax(n_inf)
        @printf("    %-30s  exposed: %d  infected: %d\n", "peak timestep", exp_peak, inf_peak)
        @test exp_peak <= inf_peak  # Exposed peak should precede infected peak

        # Conservation: S + E + I + R = N
        n_sus = get_result(sim, :seir, :n_susceptible)
        n_rec = get_result(sim, :seir, :n_recovered)
        for i in 1:length(n_sus)
            local total = n_sus[i] + n_exp[i] + n_inf[i] + n_rec[i]
            @test abs(total - 5000.0) < 2.0
        end

        # Compare against SIR: SEIR peak <= SIR peak (latent period effect)
        sim_sir = Sim(
            n_agents  = 5000,
            networks  = RandomNet(n_contacts=10),
            diseases  = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            dt        = 1.0,
            stop      = 365.0,
            rand_seed = 42,
            verbose   = 0,
        )
        run!(sim_sir)
        sir_prev = get_result(sim_sir, :sir, :prevalence)

        jl_seir_peak = maximum(prev)
        jl_sir_peak  = maximum(sir_prev)
        @printf("    %-30s  SEIR: %.4f  SIR: %.4f\n",
                "peak prevalence", jl_seir_peak, jl_sir_peak)

        # SEIR peak should not greatly exceed SIR
        @test jl_seir_peak <= jl_sir_peak + 0.05

        # Epidemic should still occur
        @test jl_seir_peak > 0.1  # SEIR should produce a substantial epidemic
        print_elapsed("V04 SEIR — Peak timing & magnitude", _t0)
    end

    # ────────────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────────────
    println("\n  All vignette cross-validation tests complete")

end # top-level testset
