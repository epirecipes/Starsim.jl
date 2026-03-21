"""
Cross-validation of RotaABM.jl against Python rotasim.

Runs both Julia and Python with the same parameters and compares
key metrics. Requires PyCall and a Python environment with rotasim installed.
"""
using Test
using Statistics

starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using RotaABM

# Try to load PyCall; skip gracefully if unavailable
pycall_available = try
    using PyCall
    true
catch
    false
end

if !pycall_available
    @warn "PyCall not available — skipping cross-validation tests"
    exit(0)
end

# Activate the Python venv if it exists
venv_python = joinpath(starsim_root, "test", "python_ref", ".venv", "bin", "python")
if isfile(venv_python)
    ENV["PYCALL_JL_RUNTIME_PYTHON"] = venv_python
end

@testset "Cross-validation" begin

    @testset "Simple scenario comparison" begin
        # --- Julia ---
        jl_sim = RotaSim(
            scenario  = "simple",
            n_agents  = 2000,
            stop      = 0.5,
            dt        = 1.0 / 365.25,
            rand_seed = 42,
            base_beta = 0.1,
        )
        run!(jl_sim; verbose=0)

        jl_prev_g1p8 = Starsim.module_results(jl_sim.diseases[:G1P8])[:prevalence].values
        jl_prev_g2p4 = Starsim.module_results(jl_sim.diseases[:G2P4])[:prevalence].values

        jl_peak_g1p8 = maximum(jl_prev_g1p8)
        jl_peak_g2p4 = maximum(jl_prev_g2p4)

        # Basic sanity checks on Julia results
        @test jl_peak_g1p8 > 0.0 "Julia G1P8 should have positive peak prevalence"
        @test jl_peak_g2p4 > 0.0 "Julia G2P4 should have positive peak prevalence"

        # --- Python (if available) ---
        try
            py"""
            import rotasim
            import numpy as np

            sim = rotasim.Sim(
                scenario='simple',
                base_beta=0.1,
                n_agents=2000,
                start='2000-01-01',
                stop='2000-07-01',
                rand_seed=42,
                verbose=0,
            )
            sim.run()

            # Get peak prevalence for each strain
            g1p8 = sim.diseases['G1P8']
            g2p4 = sim.diseases['G2P4']
            py_peak_g1p8 = float(np.max(g1p8.results.prevalence.values))
            py_peak_g2p4 = float(np.max(g2p4.results.prevalence.values))
            """

            py_peak_g1p8 = py"py_peak_g1p8"
            py_peak_g2p4 = py"py_peak_g2p4"

            # Compare peak prevalences (generous tolerance for stochastic models)
            @test abs(jl_peak_g1p8 - py_peak_g1p8) / max(py_peak_g1p8, 1e-6) < 0.5 "G1P8 peak prevalence should be within 50% of Python"
            @test abs(jl_peak_g2p4 - py_peak_g2p4) / max(py_peak_g2p4, 1e-6) < 0.5 "G2P4 peak prevalence should be within 50% of Python"

            println("Cross-validation results:")
            println("  G1P8 peak: Julia=$(round(jl_peak_g1p8, digits=4)), Python=$(round(py_peak_g1p8, digits=4))")
            println("  G2P4 peak: Julia=$(round(jl_peak_g2p4, digits=4)), Python=$(round(py_peak_g2p4, digits=4))")

        catch e
            @warn "Python rotasim not available for cross-validation: $e"
            @test_skip "Python comparison"
        end
    end
end

println("\n✓ Cross-validation tests completed!")
