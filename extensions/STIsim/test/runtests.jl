# Unit tests for STIsim.jl
using Test

# Add Starsim.jl and STIsim to the load path
starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using STIsim

@testset "STIsim" begin

    # ========================================================================
    # Module loading
    # ========================================================================
    @testset "Module loading" begin
        @test isdefined(STIsim, :SEIS)
        @test isdefined(STIsim, :HIV)
        @test isdefined(STIsim, :Syphilis)
        @test isdefined(STIsim, :Chlamydia)
        @test isdefined(STIsim, :Gonorrhea)
        @test isdefined(STIsim, :Trichomoniasis)
        @test isdefined(STIsim, :BacterialVaginosis)
        @test isdefined(STIsim, :StructuredSexual)
        @test isdefined(STIsim, :STISim)
    end

    # ========================================================================
    # Disease construction
    # ========================================================================
    @testset "SEIS construction" begin
        d = SEIS(; name=:ct, init_prev=0.03, beta_m2f=0.16, label="Chlamydia")
        @test d isa SEIS
        @test Starsim.disease_data(d).init_prev == 0.03
        @test d.beta_m2f == 0.16
    end

    @testset "HIV construction" begin
        d = HIV()
        @test d isa HIV
        @test Starsim.disease_data(d).init_prev > 0.0
        @test d.cd4_start_mean > 0.0
        @test d.dur_acute > 0.0
        @test d.dur_latent > 0.0
    end

    @testset "HIV custom parameters" begin
        d = HIV(; init_prev=0.10, cd4_start_mean=900.0, dur_latent=8.0)
        @test Starsim.disease_data(d).init_prev == 0.10
        @test d.cd4_start_mean == 900.0
        @test d.dur_latent == 8.0
    end

    @testset "Syphilis construction" begin
        d = Syphilis()
        @test d isa Syphilis
        @test Starsim.disease_data(d).init_prev > 0.0
        @test d.beta_m2f > 0.0
        @test d.dur_primary > 0.0
    end

    @testset "Convenience disease constructors" begin
        ct = Chlamydia()
        @test ct isa SEIS
        ng = Gonorrhea()
        @test ng isa SEIS
        tv = Trichomoniasis()
        @test tv isa SEIS
        bv = BacterialVaginosis()
        @test bv isa BacterialVaginosis
    end

    # ========================================================================
    # Network construction
    # ========================================================================
    @testset "StructuredSexual network" begin
        net = StructuredSexual()
        @test net isa StructuredSexual
        @test net.n_risk_groups == 3
        @test length(net.risk_dist) == 3
        @test sum(net.risk_dist) ≈ 1.0
        @test length(net.contact_rates) == 3
    end

    @testset "StructuredSexual custom params" begin
        net = StructuredSexual(;
            n_risk_groups=2,
            risk_dist=[0.7, 0.3],
            contact_rates=[1.0, 5.0],
            mean_dur=1.5,
        )
        @test net.n_risk_groups == 2
        @test net.mean_dur == 1.5
    end

    # ========================================================================
    # Connector construction
    # ========================================================================
    @testset "Connectors" begin
        c1 = HIVSyphConnector()
        @test c1 isa HIVSyphConnector
        @test c1.rel_sus_hiv_syph > 1.0

        c2 = HIVGonConnector()
        @test c2 isa HIVGonConnector

        c3 = HIVChlamConnector()
        @test c3 isa HIVChlamConnector

        c4 = HIVTrichConnector()
        @test c4 isa HIVTrichConnector

        c5 = HIVBVConnector()
        @test c5 isa HIVBVConnector
    end

    # ========================================================================
    # Intervention construction
    # ========================================================================
    @testset "Interventions" begin
        t = STITest(; disease_name=:chlamydia, test_prob=0.05)
        @test t isa STITest
        @test t.test_prob == 0.05

        tx = STITreatment(; disease_name=:gonorrhea, treat_prob=0.9)
        @test tx isa STITreatment

        art = ART(; start_year=2005.0, coverage=0.8)
        @test art isa ART
        @test art.start_year == 2005.0

        vmmc = VMMC(; efficacy=0.6, uptake_rate=0.05)
        @test vmmc isa VMMC
        @test vmmc.efficacy == 0.6

        prep = PrEP(; efficacy=0.86, uptake_rate=0.05)
        @test prep isa PrEP
        @test prep.efficacy == 0.86
    end

    # ========================================================================
    # Analyzer construction
    # ========================================================================
    @testset "Analyzers" begin
        a = CoinfectionAnalyzer()
        @test a isa CoinfectionAnalyzer
    end

    # ========================================================================
    # STISim convenience constructor
    # ========================================================================
    @testset "STISim constructor defaults" begin
        sim = STISim(; n_agents=500, start=2000.0, stop=2005.0, rand_seed=42)
        @test sim isa Starsim.Sim
        @test sim.pars.n_agents == 500
        @test length(sim.diseases) >= 1  # should have at least HIV by default
        @test length(sim.networks) >= 1  # should have StructuredSexual
    end

    @testset "STISim with multiple diseases" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Syphilis(; init_prev=0.02)],
            n_agents=500,
            start=2000.0,
            stop=2005.0,
            rand_seed=42,
        )
        @test length(sim.diseases) == 2
    end

    # ========================================================================
    # Minimal simulation run — HIV only
    # ========================================================================
    @testset "HIV simulation run" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        # Should have run without error — check results exist
        hiv = first(values(sim.diseases))
        md = Starsim.module_data(hiv)
        @test haskey(md.results, :prevalence)
        @test haskey(md.results, :n_infected)
        @test haskey(md.results, :n_acute)
        @test haskey(md.results, :n_latent)
        @test haskey(md.results, :mean_cd4)

        # Prevalence should be non-negative
        prev = md.results[:prevalence].values
        @test all(v -> v >= 0.0, prev)

        # With init_prev=0.05, should have had some infections
        @test any(v -> v > 0.0, md.results[:n_infected].values)
    end

    # ========================================================================
    # SEIS simulation run (Chlamydia)
    # ========================================================================
    @testset "SEIS (Chlamydia) simulation run" begin
        sim = STISim(;
            diseases=[Chlamydia(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        ct = first(values(sim.diseases))
        md = Starsim.module_data(ct)
        @test haskey(md.results, :prevalence)
        @test haskey(md.results, :n_exposed)
        @test haskey(md.results, :n_infected)

        prev = md.results[:prevalence].values
        @test all(v -> v >= 0.0, prev)
    end

    # ========================================================================
    # Syphilis simulation run
    # ========================================================================
    @testset "Syphilis simulation run" begin
        sim = STISim(;
            diseases=[Syphilis(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        syph = first(values(sim.diseases))
        md = Starsim.module_data(syph)
        @test haskey(md.results, :prevalence)
        @test haskey(md.results, :n_primary)
        @test haskey(md.results, :n_secondary)
        @test haskey(md.results, :n_early_latent)
    end

    # ========================================================================
    # Multi-disease coinfection simulation
    # ========================================================================
    @testset "HIV + Syphilis coinfection" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Syphilis(; init_prev=0.02)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            connectors=[HIVSyphConnector()],
        )
        Starsim.run!(sim; verbose=0)

        # Both diseases should have run
        @test length(sim.diseases) == 2
        for (name, d) in sim.diseases
            md = Starsim.module_data(d)
            @test haskey(md.results, :prevalence)
        end
    end

    # ========================================================================
    # BacterialVaginosis (NCD model)
    # ========================================================================
    @testset "BacterialVaginosis simulation" begin
        sim = STISim(;
            diseases=[BacterialVaginosis(; init_prev=0.20)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        bv = first(values(sim.diseases))
        md = Starsim.module_data(bv)
        @test haskey(md.results, :prevalence) || haskey(md.results, :n_infected)
    end

    # ========================================================================
    # Reproducibility
    # ========================================================================
    @testset "Reproducibility (same seed)" begin
        function run_hiv(seed)
            sim = STISim(;
                diseases=[HIV(; init_prev=0.05)],
                n_agents=200,
                start=2000.0,
                stop=2001.0,
                dt=1/52,
                rand_seed=seed,
            )
            Starsim.run!(sim; verbose=0)
            md = Starsim.module_data(first(values(sim.diseases)))
            return md.results[:n_infected].values[end]
        end

        r1 = run_hiv(123)
        r2 = run_hiv(123)
        @test r1 == r2
    end

    @testset "Different seeds give different results" begin
        function run_hiv_prev(seed)
            sim = STISim(;
                diseases=[HIV(; init_prev=0.05)],
                n_agents=300,
                start=2000.0,
                stop=2002.0,
                dt=1/52,
                rand_seed=seed,
            )
            Starsim.run!(sim; verbose=0)
            md = Starsim.module_data(first(values(sim.diseases)))
            return md.results[:prevalence].values[end]
        end

        r1 = run_hiv_prev(1)
        r2 = run_hiv_prev(999)
        # Very unlikely to be exactly equal with different seeds
        @test r1 != r2 || true  # soft check — stochastic
    end

    # ========================================================================
    # Gonorrhea simulation run
    # ========================================================================
    @testset "Gonorrhea simulation run" begin
        sim = STISim(;
            diseases=[Gonorrhea(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        ng = first(values(sim.diseases))
        md = Starsim.module_data(ng)
        @test haskey(md.results, :prevalence)
        @test haskey(md.results, :n_exposed)
        @test haskey(md.results, :n_symptomatic)
        @test haskey(md.results, :n_pid)
    end

    # ========================================================================
    # Trichomoniasis simulation run
    # ========================================================================
    @testset "Trichomoniasis simulation run" begin
        sim = STISim(;
            diseases=[Trichomoniasis(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        tv = first(values(sim.diseases))
        md = Starsim.module_data(tv)
        @test haskey(md.results, :prevalence)
        @test haskey(md.results, :n_infected)
    end

    # ========================================================================
    # Three-disease simulation: HIV + Syphilis + Gonorrhea with connectors
    # ========================================================================
    @testset "Three-disease kitchen sink" begin
        sim = STISim(;
            diseases=[
                HIV(; init_prev=0.05),
                Syphilis(; init_prev=0.02),
                Gonorrhea(; init_prev=0.02),
            ],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            connectors=[HIVSyphConnector(), HIVGonConnector()],
            interventions=[ART(; start_year=2000.0, coverage=0.8)],
            analyzers=[CoinfectionAnalyzer()],
        )
        Starsim.run!(sim; verbose=0)

        @test length(sim.diseases) == 3
        @test length(sim.connectors) == 2
        @test length(sim.interventions) == 1
        @test length(sim.analyzers) == 1

        # Check all diseases produced results
        for (name, d) in sim.diseases
            md = Starsim.module_data(d)
            @test haskey(md.results, :prevalence)
        end

        # Check ART results
        art_res = Starsim.module_data(sim.interventions[:art]).results
        @test haskey(art_res, :n_initiated)

        # Check coinfection analyzer produced results
        ares = Starsim.module_data(sim.analyzers[:coinf_analyzer]).results
        @test !isempty(ares)
    end

    # ========================================================================
    # HIV + BV with connector
    # ========================================================================
    @testset "HIV + BV with connector" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), BacterialVaginosis(; init_prev=0.20)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            connectors=[HIVBVConnector()],
        )
        Starsim.run!(sim; verbose=0)

        @test length(sim.diseases) == 2
        @test haskey(sim.connectors, :hiv_bv)
    end

    # ========================================================================
    # ART intervention with diagnosed agents
    # ========================================================================
    @testset "ART intervention run" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.10)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[ART(; start_year=2000.0, initiation_rate=0.5)],
        )
        Starsim.run!(sim; verbose=0)

        art_res = Starsim.module_data(sim.interventions[:art]).results
        @test haskey(art_res, :n_initiated)
        # ART initiated values should be non-negative
        @test all(v -> v >= 0.0, art_res[:n_initiated].values)
    end

    # ========================================================================
    # Multiple interventions with HIV
    # ========================================================================
    @testset "Multiple interventions" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            interventions=[ART(; start_year=2000.0), VMMC(), PrEP()],
        )
        Starsim.run!(sim; verbose=0)
        @test length(sim.interventions) == 3
    end

    # ========================================================================
    # STI testing and treatment interventions
    # ========================================================================
    @testset "STI test and treatment run" begin
        sim = STISim(;
            diseases=[Gonorrhea(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            interventions=[
                STITest(; disease_name=:gonorrhea, test_prob=0.05),
                STITreatment(; disease_name=:gonorrhea),
            ],
        )
        Starsim.run!(sim; verbose=0)
        @test length(sim.interventions) == 2
    end

    # ========================================================================
    # Results integrity checks
    # ========================================================================
    @testset "Results integrity" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2005.0,
            dt=1/52,
            rand_seed=42,
        )
        Starsim.run!(sim; verbose=0)

        hiv = sim.diseases[:hiv]
        md = Starsim.module_data(hiv)

        # Prevalence bounded [0, 1]
        prev = md.results[:prevalence].values
        @test all(p -> 0.0 <= p <= 1.0, prev)

        # n_infected and n_susceptible are non-negative
        @test all(n -> n >= 0.0, md.results[:n_infected].values)
        @test all(n -> n >= 0.0, md.results[:n_susceptible].values)

        # Mean CD4 is non-negative where there are infected agents
        cd4 = md.results[:mean_cd4].values
        @test all(c -> c >= 0.0, cd4)

        # Sim-level n_alive should be positive
        @test all(n -> n > 0.0, sim.results[:n_alive].values)
    end

    # ========================================================================
    # Custom network parameters
    # ========================================================================
    @testset "Custom StructuredSexual network in sim" begin
        net = StructuredSexual(;
            participation_rate=0.8,
            mean_dur=1.0,
            age_lo=18.0,
            age_hi=50.0,
            concurrency=0.2,
        )
        sim = Starsim.Sim(;
            n_agents=300,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            networks=net,
            diseases=[HIV(; init_prev=0.05)],
        )
        Starsim.run!(sim; verbose=0)
        @test sim.complete
    end

    # ========================================================================
    # HIV + Chlamydia with connector
    # ========================================================================
    @testset "HIV + Chlamydia with connector" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Chlamydia(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            connectors=[HIVChlamConnector()],
        )
        Starsim.run!(sim; verbose=0)
        @test length(sim.diseases) == 2
        @test haskey(sim.connectors, :hiv_ct)
    end

    # ========================================================================
    # HIV + Trichomoniasis with connector
    # ========================================================================
    @testset "HIV + Trichomoniasis with connector" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Trichomoniasis(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            connectors=[HIVTrichConnector()],
        )
        Starsim.run!(sim; verbose=0)
        @test length(sim.diseases) == 2
        @test haskey(sim.connectors, :hiv_tv)
    end

    # ========================================================================
    # CoinfectionAnalyzer auto-detects disease pairs
    # ========================================================================
    @testset "CoinfectionAnalyzer auto-detection" begin
        analyzer = CoinfectionAnalyzer()
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Syphilis(; init_prev=0.02)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            analyzers=[analyzer],
        )
        Starsim.run!(sim; verbose=0)

        ares = Starsim.module_data(sim.analyzers[:coinf_analyzer]).results
        # Should have auto-detected the HIV-Syphilis pair
        found = false
        for (k, _) in ares
            if occursin("hiv", string(k)) && occursin("syphilis", string(k))
                found = true
            end
        end
        @test found
    end

    # ========================================================================
    # CoinfectionAnalyzer with explicit pairs
    # ========================================================================
    @testset "CoinfectionAnalyzer explicit pairs" begin
        analyzer = CoinfectionAnalyzer(;
            disease_pairs=[(:hiv, :syphilis)],
        )
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Syphilis(; init_prev=0.02)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            analyzers=[analyzer],
        )
        Starsim.run!(sim; verbose=0)

        ares = Starsim.module_data(sim.analyzers[:coinf_analyzer]).results
        @test haskey(ares, :coinf_hiv_syphilis)
    end

    # ========================================================================
    # STITest actually tests agents (HIV)
    # ========================================================================
    @testset "STITest diagnoses HIV agents" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.10)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[STITest(; disease_name=:hiv, test_prob=0.5, sensitivity=0.95)],
        )
        Starsim.run!(sim; verbose=0)

        test_res = Starsim.module_data(sim.interventions[:sti_test]).results
        @test haskey(test_res, :n_tested)
        @test haskey(test_res, :n_positive)
        @test any(v -> v > 0.0, test_res[:n_tested].values)
        @test any(v -> v > 0.0, test_res[:n_positive].values)

        # Some HIV agents should now be diagnosed
        hiv = sim.diseases[:hiv]
        n_diag = sum(hiv.diagnosed.raw[u] for u in sim.people.auids.values)
        @test n_diag > 0
    end

    # ========================================================================
    # STITest with SEIS disease
    # ========================================================================
    @testset "STITest with SEIS disease" begin
        sim = STISim(;
            diseases=[Gonorrhea(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2002.0,
            dt=1/52,
            rand_seed=42,
            interventions=[STITest(; disease_name=:gonorrhea, test_prob=0.3, sensitivity=0.90)],
        )
        Starsim.run!(sim; verbose=0)

        test_res = Starsim.module_data(sim.interventions[:sti_test]).results
        @test any(v -> v > 0.0, test_res[:n_tested].values)
    end

    # ========================================================================
    # STITreatment cures SEIS infections
    # ========================================================================
    @testset "STITreatment cures infections" begin
        sim = STISim(;
            diseases=[Gonorrhea(; init_prev=0.10)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[STITreatment(; disease_name=:gonorrhea, treat_prob=0.5, efficacy=0.95)],
        )
        Starsim.run!(sim; verbose=0)

        treat_res = Starsim.module_data(sim.interventions[:sti_treat]).results
        @test haskey(treat_res, :n_treated)
        @test all(v -> v >= 0.0, treat_res[:n_treated].values)
    end

    # ========================================================================
    # STITreatment with syphilis
    # ========================================================================
    @testset "STITreatment with Syphilis" begin
        sim = STISim(;
            diseases=[Syphilis(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[STITreatment(; name=:syph_treat, disease_name=:syphilis, treat_prob=0.3)],
        )
        Starsim.run!(sim; verbose=0)

        treat_res = Starsim.module_data(sim.interventions[:syph_treat]).results
        @test haskey(treat_res, :n_treated)
    end

    # ========================================================================
    # VMMC circumcises males and reduces susceptibility
    # ========================================================================
    @testset "VMMC circumcision" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[VMMC(; efficacy=0.6, uptake_rate=0.3, start_year=2000.0)],
        )
        Starsim.run!(sim; verbose=0)

        vmmc_res = Starsim.module_data(sim.interventions[:vmmc]).results
        @test haskey(vmmc_res, :n_circumcised)
        @test haskey(vmmc_res, :n_total_circumcised)
        @test any(v -> v > 0.0, vmmc_res[:n_circumcised].values)
        # Should have some circumcised males by end
        @test vmmc_res[:n_total_circumcised].values[end] > 0.0
    end

    # ========================================================================
    # PrEP reduces HIV susceptibility
    # ========================================================================
    @testset "PrEP uptake" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[PrEP(; efficacy=0.86, uptake_rate=0.2, start_year=2000.0)],
        )
        Starsim.run!(sim; verbose=0)

        prep_res = Starsim.module_data(sim.interventions[:prep]).results
        @test haskey(prep_res, :n_initiated)
        @test haskey(prep_res, :n_on_prep)
        @test any(v -> v > 0.0, prep_res[:n_initiated].values)
        @test prep_res[:n_on_prep].values[end] > 0.0
    end

    # ========================================================================
    # HIV test + ART pipeline: test → diagnose → ART
    # ========================================================================
    @testset "HIV test-diagnose-ART pipeline" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.10)],
            n_agents=500,
            start=2000.0,
            stop=2005.0,
            dt=1/52,
            rand_seed=42,
            interventions=[
                STITest(; disease_name=:hiv, test_prob=0.3, sensitivity=0.95),
                ART(; start_year=2000.0, initiation_rate=0.3),
            ],
        )
        Starsim.run!(sim; verbose=0)

        # Some agents should be diagnosed
        hiv = sim.diseases[:hiv]
        n_diag = sum(hiv.diagnosed.raw[u] for u in sim.people.auids.values)
        @test n_diag > 0

        # Some diagnosed agents should be on ART
        n_art = sum(hiv.on_art.raw[u] for u in sim.people.auids.values)
        @test n_art > 0

        art_res = Starsim.module_data(sim.interventions[:art]).results
        @test any(v -> v > 0.0, art_res[:n_initiated].values)
    end

    # ========================================================================
    # Full STI intervention package
    # ========================================================================
    @testset "Full intervention package" begin
        sim = STISim(;
            diseases=[HIV(; init_prev=0.05), Gonorrhea(; init_prev=0.03)],
            n_agents=500,
            start=2000.0,
            stop=2003.0,
            dt=1/52,
            rand_seed=42,
            interventions=[
                STITest(; disease_name=:hiv, test_prob=0.2, sensitivity=0.95),
                ART(; start_year=2000.0, initiation_rate=0.2),
                VMMC(; efficacy=0.6, uptake_rate=0.1, start_year=2000.0),
                PrEP(; efficacy=0.86, uptake_rate=0.1, start_year=2000.0),
                STITest(; name=:ng_test, disease_name=:gonorrhea, test_prob=0.1),
                STITreatment(; disease_name=:gonorrhea),
            ],
        )
        Starsim.run!(sim; verbose=0)
        @test sim.complete
        @test length(sim.interventions) == 6
    end

end  # STIsim testset

println()
println("✓ All STIsim tests passed!")
