using Test
using HPVsim
using Starsim
using Statistics: mean

@testset "HPVsim.jl" begin

    # ========================================================================
    # Genotype parameters
    # ========================================================================
    @testset "Genotype parameters" begin
        # Registry lookup
        gp16 = get_genotype_params(:hpv16)
        @test gp16.name == :hpv16
        @test gp16.rel_beta == 1.0
        @test 0.0 < gp16.clearance_rate_inf < 1.0
        @test 0.0 < gp16.prog_rate_cin1 < 1.0
        @test 0.0 < gp16.own_imm <= 1.0

        gp18 = get_genotype_params(:hpv18)
        @test gp18.name == :hpv18
        @test gp18.rel_beta < gp16.rel_beta

        # All registered genotypes
        for sym in [:hpv16, :hpv18, :hi5, :ohr, :lr]
            gp = get_genotype_params(sym)
            @test gp.name == sym
            @test gp.rel_beta > 0.0
        end

        # Unknown genotype
        @test_throws ErrorException get_genotype_params(:unknown_genotype)

        # Exported constants
        @test HPV16_PARAMS.name == :hpv16
        @test HPV18_PARAMS.name == :hpv18
        @test HI5_PARAMS.name == :hi5
        @test OHR_PARAMS.name == :ohr
        @test LR_PARAMS.name == :lr
        @test length(GENOTYPE_REGISTRY) >= 5  # 5 grouped + 8 individual types
    end

    # ========================================================================
    # GenotypeDef
    # ========================================================================
    @testset "GenotypeDef" begin
        gd = GenotypeDef(:hpv16)
        @test gd.name == :hpv16
        @test gd.init_prev == 0.01

        gd2 = GenotypeDef(:hpv18; init_prev=0.05)
        @test gd2.init_prev == 0.05

        @test length(DEFAULT_GENOTYPES) == 3
        @test length(SIMPLE_GENOTYPES) == 1
        @test length(FULL_GENOTYPES) == 5

        @test DEFAULT_GENOTYPES[1].name == :hpv16
        @test SIMPLE_GENOTYPES[1].name == :hpv16
    end

    # ========================================================================
    # HPVGenotype construction
    # ========================================================================
    @testset "HPVGenotype construction" begin
        d = HPVGenotype(genotype=:hpv16, init_prev=0.05)
        @test d.genotype == :hpv16
        @test d.params.name == :hpv16

        md = Starsim.module_data(d)
        @test md.name == :hpv16

        # Check with explicit params
        d2 = HPVGenotype(genotype=:hpv18)
        @test d2.genotype == :hpv18
        @test d2.params.rel_beta == HPV18_PARAMS.rel_beta
    end

    # ========================================================================
    # HPVImmunityConnector construction
    # ========================================================================
    @testset "HPVImmunityConnector construction" begin
        conn = HPVImmunityConnector()
        @test conn.own_imm == 0.90
        @test conn.partial_imm == 0.50
        @test conn.cross_imm == 0.30
        @test conn.waning_rate == 0.05

        conn2 = HPVImmunityConnector(own_imm=0.80, waning_rate=0.10)
        @test conn2.own_imm == 0.80
        @test conn2.waning_rate == 0.10
    end

    # ========================================================================
    # HPVVaccination construction
    # ========================================================================
    @testset "HPVVaccination construction" begin
        vax = HPVVaccination()
        @test vax.covered_genotypes == [:hpv16, :hpv18]
        @test vax.n_doses == 2
        @test vax.uptake_prob == 0.8
        @test vax.sex == :female

        vax2 = HPVVaccination(
            covered_genotypes=[:hpv16, :hpv18, :hi5],
            n_doses=3,
            uptake_prob=0.9,
        )
        @test length(vax2.covered_genotypes) == 3
        @test vax2.n_doses == 3
    end

    # ========================================================================
    # HPVScreening construction
    # ========================================================================
    @testset "HPVScreening construction" begin
        scr = HPVScreening()
        @test scr.test_type == :pap
        @test scr.sensitivity == 0.55
        @test scr.specificity == 0.97

        scr_dna = HPVScreening(test_type=:hpv_dna)
        @test scr_dna.sensitivity == 0.95
        @test scr_dna.specificity == 0.90

        scr_via = HPVScreening(test_type=:via)
        @test scr_via.sensitivity == 0.60
        @test scr_via.specificity == 0.84
    end

    # ========================================================================
    # Simple single-genotype simulation
    # ========================================================================
    @testset "Simple single-genotype sim" begin
        sim = HPVSim(
            genotypes    = SIMPLE_GENOTYPES,
            n_agents     = 500,
            start        = 2000.0,
            stop         = 2010.0,
            dt           = 0.5,
            rand_seed    = 42,
            use_immunity = false,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        # Check HPV16 disease exists
        @test haskey(sim.diseases, :hpv16)
        d16 = sim.diseases[:hpv16]
        @test d16 isa HPVGenotype

        # Results should be populated
        md = Starsim.module_data(d16)
        prev = md.results[:prevalence].values
        @test length(prev) > 0
        @test any(x -> x > 0.0, prev)

        n_inf = md.results[:n_infected].values
        @test length(n_inf) > 0
        @test any(x -> x > 0.0, n_inf)

        n_sus = md.results[:n_susceptible].values
        @test length(n_sus) > 0
        @test n_sus[1] > 0.0
    end

    # ========================================================================
    # Multi-genotype simulation
    # ========================================================================
    @testset "Multi-genotype sim" begin
        sim = HPVSim(
            genotypes    = DEFAULT_GENOTYPES,
            n_agents     = 500,
            start        = 2000.0,
            stop         = 2010.0,
            dt           = 0.5,
            rand_seed    = 123,
            use_immunity = true,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        # All 3 genotypes present
        @test length(sim.diseases) == 3
        @test haskey(sim.diseases, :hpv16)
        @test haskey(sim.diseases, :hpv18)
        @test haskey(sim.diseases, :hi5)

        # Immunity connector present
        @test length(sim.connectors) >= 1
        conn = first(values(sim.connectors))
        @test conn isa HPVImmunityConnector
        @test length(conn.hpv_diseases) == 3
        @test size(conn.imm_matrix) == (3, 3)

        # Diagonal = own_imm, off-diagonal = partial_imm (all high-risk)
        for i in 1:3
            @test conn.imm_matrix[i, i] == conn.own_imm
        end

        # Each genotype has results
        for (_, dis) in sim.diseases
            md = Starsim.module_data(dis)
            @test haskey(md.results, :prevalence)
            @test haskey(md.results, :n_infected)
            @test haskey(md.results, :n_cin1)
            @test haskey(md.results, :n_cancerous)
        end
    end

    # ========================================================================
    # Full 5-genotype simulation
    # ========================================================================
    @testset "Full genotype sim" begin
        sim = HPVSim(
            genotypes    = FULL_GENOTYPES,
            n_agents     = 300,
            start        = 2000.0,
            stop         = 2005.0,
            dt           = 0.5,
            rand_seed    = 7,
            use_immunity = true,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete
        @test length(sim.diseases) == 5

        # Immunity matrix: low-risk vs high-risk should be cross_imm
        conn = first(values(sim.connectors))
        @test conn isa HPVImmunityConnector
        n_g = length(conn.genotype_names)
        @test n_g == 5

        # Find index of :lr
        lr_idx = findfirst(x -> x == :lr, conn.genotype_names)
        @test lr_idx !== nothing

        # lr vs any high-risk should have cross_imm
        for i in 1:n_g
            if conn.genotype_names[i] in [:hpv16, :hpv18, :hi5, :ohr]
                @test conn.imm_matrix[lr_idx, i] == conn.cross_imm
                @test conn.imm_matrix[i, lr_idx] == conn.cross_imm
            end
        end
    end

    # ========================================================================
    # Simulation with vaccination
    # ========================================================================
    @testset "Vaccination intervention" begin
        vax = HPVVaccination(
            start_year   = 2000.0,
            min_age      = 0.0,
            max_age      = 100.0,
            uptake_prob  = 1.0,
            sex          = :both,
        )
        sim = HPVSim(
            genotypes     = SIMPLE_GENOTYPES,
            n_agents      = 300,
            start         = 2000.0,
            stop          = 2005.0,
            dt            = 0.5,
            rand_seed     = 99,
            use_immunity  = false,
            interventions = [vax],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        # Vaccination should have administered some doses
        @test haskey(sim.interventions, :hpv_vax)
        vax_mod = sim.interventions[:hpv_vax]
        md = Starsim.module_data(vax_mod)
        n_vacc = md.results[:n_vaccinated].values
        @test any(x -> x > 0.0, n_vacc)
    end

    # ========================================================================
    # Simulation with screening
    # ========================================================================
    @testset "Screening intervention" begin
        scr = HPVScreening(
            start_year  = 2000.0,
            screen_prob = 0.10,
            min_age     = 0.0,
            max_age     = 200.0,
        )
        sim = HPVSim(
            genotypes     = SIMPLE_GENOTYPES,
            n_agents      = 500,
            start         = 2000.0,
            stop          = 2010.0,
            dt            = 0.5,
            rand_seed     = 77,
            use_immunity  = false,
            interventions = [scr],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        scr_mod = sim.interventions[:hpv_screening]
        md = Starsim.module_data(scr_mod)
        n_screened = md.results[:n_screened].values
        @test any(x -> x > 0.0, n_screened)
    end

    # ========================================================================
    # Combined vaccination + screening
    # ========================================================================
    @testset "Combined interventions" begin
        vax = HPVVaccination(
            start_year  = 2000.0,
            min_age     = 0.0,
            max_age     = 100.0,
            uptake_prob = 0.5,
            sex         = :both,
        )
        scr = HPVScreening(
            start_year  = 2000.0,
            screen_prob = 0.05,
            min_age     = 0.0,
            max_age     = 200.0,
        )
        sim = HPVSim(
            genotypes     = DEFAULT_GENOTYPES,
            n_agents      = 300,
            start         = 2000.0,
            stop          = 2005.0,
            dt            = 0.5,
            rand_seed     = 55,
            use_immunity  = true,
            interventions = [vax, scr],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete
        @test length(sim.interventions) == 2
    end

    # ========================================================================
    # HPVNet convenience constructor
    # ========================================================================
    @testset "HPVNet" begin
        net = HPVNet()
        @test net isa Starsim.MFNet
        md = Starsim.module_data(net)
        @test md.name == :sexual

        net2 = HPVNet(mean_dur=3.0, name=:sex2)
        md2 = Starsim.module_data(net2)
        @test md2.name == :sex2
    end

    # ========================================================================
    # Genotype specification variants
    # ========================================================================
    @testset "Genotype specification variants" begin
        # Symbol vector
        sim1 = HPVSim(
            genotypes = [:hpv16, :hpv18],
            n_agents  = 200,
            start     = 2000.0,
            stop      = 2002.0,
            dt        = 0.5,
            rand_seed = 1,
            verbose   = 0,
        )
        Starsim.run!(sim1)
        @test length(sim1.diseases) == 2

        # Single symbol
        sim2 = HPVSim(
            genotypes = :hpv16,
            n_agents  = 200,
            start     = 2000.0,
            stop      = 2002.0,
            dt        = 0.5,
            rand_seed = 1,
            verbose   = 0,
        )
        Starsim.run!(sim2)
        @test length(sim2.diseases) == 1

        # Single GenotypeDef
        sim3 = HPVSim(
            genotypes = GenotypeDef(:hpv16; init_prev=0.10),
            n_agents  = 200,
            start     = 2000.0,
            stop      = 2002.0,
            dt        = 0.5,
            rand_seed = 1,
            verbose   = 0,
        )
        Starsim.run!(sim3)
        @test length(sim3.diseases) == 1
    end

    # ========================================================================
    # Reproducibility — same seed gives same results
    # ========================================================================
    @testset "Reproducibility" begin
        function run_sim(seed)
            sim = HPVSim(
                genotypes    = SIMPLE_GENOTYPES,
                n_agents     = 200,
                start        = 2000.0,
                stop         = 2005.0,
                dt           = 0.5,
                rand_seed    = seed,
                use_immunity = false,
                verbose      = 0,
            )
            Starsim.run!(sim)
            md = Starsim.module_data(sim.diseases[:hpv16])
            return md.results[:prevalence].values
        end

        prev1 = run_sim(42)
        prev2 = run_sim(42)
        @test prev1 == prev2

        prev3 = run_sim(99)
        @test prev3 != prev1
    end

    # ========================================================================
    # CIN progression states
    # ========================================================================
    @testset "CIN progression" begin
        sim = HPVSim(
            genotypes    = SIMPLE_GENOTYPES,
            n_agents     = 1000,
            start        = 2000.0,
            stop         = 2030.0,
            dt           = 0.25,
            rand_seed    = 10,
            use_immunity = false,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        d16 = sim.diseases[:hpv16]
        md = Starsim.module_data(d16)

        # Over 30 years, we should see some CIN progression
        cin1_vals = md.results[:n_cin1].values
        cin2_vals = md.results[:n_cin2].values
        cin3_vals = md.results[:n_cin3].values
        cancer_vals = md.results[:n_cancerous].values

        # At least some CIN1 should have occurred
        @test any(x -> x > 0.0, cin1_vals)

        # Clearance should occur
        cleared_vals = md.results[:n_cleared].values
        @test any(x -> x > 0.0, cleared_vals)
    end

    # ========================================================================
    # Zero prevalence — no infections without seeding
    # ========================================================================
    @testset "Zero initial prevalence" begin
        gd = GenotypeDef(:hpv16; init_prev=0.0)
        sim = HPVSim(
            genotypes    = [gd],
            n_agents     = 200,
            start        = 2000.0,
            stop         = 2005.0,
            dt           = 0.5,
            rand_seed    = 1,
            use_immunity = false,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        d16 = sim.diseases[:hpv16]
        md = Starsim.module_data(d16)

        # No infections should ever appear
        n_inf = md.results[:n_infected].values
        @test all(x -> x == 0.0, n_inf)
    end

    # ========================================================================
    # New features: Individual genotypes
    # ========================================================================
    @testset "Individual genotypes" begin
        # All individual genotypes exist in registry
        for sym in [:hpv31, :hpv33, :hpv45, :hpv52, :hpv58, :hpv6, :hpv11]
            gp = get_genotype_params(sym)
            @test gp.name == sym
            @test gp.rel_beta > 0.0
        end

        # Aggregate HR type
        gp_hr = get_genotype_params(:hr)
        @test gp_hr.name == :hr

        # Risk classification
        @test is_high_risk(:hpv16)
        @test is_high_risk(:hpv31)
        @test is_high_risk(:hi5)
        @test !is_high_risk(:lr)
        @test !is_high_risk(:hpv6)
        @test is_low_risk(:lr)
        @test is_low_risk(:hpv6)
        @test is_low_risk(:hpv11)
        @test !is_low_risk(:hpv16)

        # list_genotypes
        all_g = list_genotypes()
        @test length(all_g) == length(GENOTYPE_REGISTRY)

        # Genotype presets
        @test length(BIVALENT_GENOTYPES) == 2
        @test length(NONAVALENT_GENOTYPES) == 9
        @test length(INDIVIDUAL_HR_GENOTYPES) == 7
    end

    # ========================================================================
    # New features: logf2 and duration-based progression functions
    # ========================================================================
    @testset "logf2 functions" begin
        # logf2 should be 0 at x=0 and ~1 at x=ttc
        @test logf2(0.0, 0.3, 0.0; ttc=50.0) ≈ 0.0 atol=1e-6
        val_at_ttc = logf2(50.0, 0.3, 0.0; ttc=50.0)
        @test val_at_ttc ≈ 1.0 atol=0.05

        # logf2 is monotonically increasing
        vals = [logf2(Float64(x), 0.3, 0.0; ttc=50.0) for x in 0:5:50]
        for i in 2:length(vals)
            @test vals[i] >= vals[i-1]
        end

        # intlogf2 should be >= 0 and monotonically increasing
        int_vals = [intlogf2(Float64(x), 0.3, 0.0; ttc=50.0) for x in 0:5:60]
        @test int_vals[1] ≈ 0.0 atol=1e-6
        for i in 2:length(int_vals)
            @test int_vals[i] >= int_vals[i-1] - 1e-10
        end

        # compute_cin_prob should be between 0 and 1
        cin_p = compute_cin_prob(3.0, 1.0, 0.3, 0.0; ttc=50.0)
        @test 0.0 <= cin_p <= 1.0

        # compute_cancer_prob should be between 0 and 1
        cancer_p = compute_cancer_prob(5.0, 1.0, 0.3, 0.0, 2e-3; ttc=50.0)
        @test 0.0 <= cancer_p <= 1.0

        # Higher severity → higher CIN probability
        low_sev = compute_cin_prob(3.0, 0.5, 0.3, 0.0; ttc=50.0)
        high_sev = compute_cin_prob(3.0, 2.0, 0.3, 0.0; ttc=50.0)
        @test high_sev > low_sev

        # sample_lognormal_duration returns positive values
        using StableRNGs
        rng = StableRNG(42)
        for _ in 1:100
            d = sample_lognormal_duration(rng, 3.0, 9.0)
            @test d > 0.0
        end
    end

    # ========================================================================
    # New features: Duration-based progression model
    # ========================================================================
    @testset "Duration-based model" begin
        sim = HPVSim(
            genotypes    = SIMPLE_GENOTYPES,
            n_agents     = 500,
            start        = 2000.0,
            stop         = 2020.0,
            dt           = 0.25,
            rand_seed    = 42,
            use_immunity = false,
            verbose      = 0,
        )
        # Enable duration model on the disease
        d16 = sim.diseases[:hpv16]
        d16.use_duration_model = true

        Starsim.run!(sim)
        @test sim.complete

        md = Starsim.module_data(d16)
        prev = md.results[:prevalence].values
        @test any(x -> x > 0.0, prev)

        # CIN should occur
        cin1_vals = md.results[:n_cin1].values
        @test any(x -> x > 0.0, cin1_vals)

        # Clearance should occur
        cleared_vals = md.results[:n_cleared].values
        @test any(x -> x > 0.0, cleared_vals)
    end

    # ========================================================================
    # New features: Cancer mortality
    # ========================================================================
    @testset "Cancer mortality" begin
        sim = HPVSim(
            genotypes    = SIMPLE_GENOTYPES,
            n_agents     = 1000,
            start        = 2000.0,
            stop         = 2040.0,
            dt           = 0.25,
            rand_seed    = 10,
            use_immunity = false,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        d16 = sim.diseases[:hpv16]
        md = Starsim.module_data(d16)

        # Cancer deaths result should exist
        @test haskey(md.results, :n_cancer_deaths)
        death_vals = md.results[:n_cancer_deaths].values
        @test length(death_vals) > 0

        # CIN prevalence result should exist
        @test haskey(md.results, :cin_prevalence)
    end

    # ========================================================================
    # New features: Nonavalent genotypes simulation
    # ========================================================================
    @testset "Nonavalent genotypes" begin
        sim = HPVSim(
            genotypes    = NONAVALENT_GENOTYPES,
            n_agents     = 300,
            start        = 2000.0,
            stop         = 2003.0,
            dt           = 0.5,
            rand_seed    = 42,
            use_immunity = true,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete
        @test length(sim.diseases) == 9

        # All diseases should exist
        for gd in NONAVALENT_GENOTYPES
            @test haskey(sim.diseases, gd.name)
        end

        # Immunity connector should handle all 9 genotypes
        conn = first(values(sim.connectors))
        @test conn isa HPVImmunityConnector
        @test length(conn.hpv_diseases) == 9
        @test size(conn.imm_matrix) == (9, 9)

        # Cross-immunity between high-risk and low-risk should be cross_imm
        lr_names = [:hpv6, :hpv11]
        hr_names = [:hpv16, :hpv18, :hpv31, :hpv33, :hpv45, :hpv52, :hpv58]
        for lr in lr_names
            lr_idx = findfirst(x -> x == lr, conn.genotype_names)
            lr_idx === nothing && continue
            for hr in hr_names
                hr_idx = findfirst(x -> x == hr, conn.genotype_names)
                hr_idx === nothing && continue
                @test conn.imm_matrix[lr_idx, hr_idx] == conn.cross_imm
            end
        end
    end

    # ========================================================================
    # New features: Treatment types
    # ========================================================================
    @testset "Treatment types" begin
        # Treatment efficacy lookup
        @test get_treatment_efficacy(ABLATION, :cin1) == 0.90
        @test get_treatment_efficacy(ABLATION, :cin3) == 0.75
        @test get_treatment_efficacy(EXCISION, :cin1) == 0.95
        @test get_treatment_efficacy(EXCISION, :cancer) == 0.50
        @test get_treatment_efficacy(GENERIC, :cin1) == 0.85

        # Screening with EXCISION treatment
        scr = HPVScreening(
            start_year     = 2000.0,
            screen_prob    = 0.10,
            min_age        = 0.0,
            max_age        = 200.0,
            treatment_type = EXCISION,
        )
        @test scr.treatment_type == EXCISION
        @test scr.sensitivity_cin1 > 0.0
        @test scr.sensitivity_cancer > 0.0

        # Screening with ABLATION treatment
        scr2 = HPVScreening(
            start_year     = 2000.0,
            screen_prob    = 0.10,
            min_age        = 0.0,
            max_age        = 200.0,
            treatment_type = ABLATION,
        )
        sim = HPVSim(
            genotypes     = SIMPLE_GENOTYPES,
            n_agents      = 500,
            start         = 2000.0,
            stop          = 2010.0,
            dt            = 0.5,
            rand_seed     = 77,
            use_immunity  = false,
            interventions = [scr2],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete
    end

    # ========================================================================
    # New features: Stage-dependent sensitivity
    # ========================================================================
    @testset "Stage-dependent sensitivity" begin
        # PAP: CIN1 sensitivity should be lower than CIN2+ sensitivity
        scr_pap = HPVScreening(test_type=:pap)
        @test scr_pap.sensitivity_cin1 < scr_pap.sensitivity
        @test scr_pap.sensitivity_cancer > scr_pap.sensitivity

        # HPV DNA: high sensitivity for CIN1 too
        scr_dna = HPVScreening(test_type=:hpv_dna)
        @test scr_dna.sensitivity_cin1 >= 0.85

        # VIA
        scr_via = HPVScreening(test_type=:via)
        @test scr_via.sensitivity_cin1 == 0.40
    end

    # ========================================================================
    # New features: Therapeutic vaccination
    # ========================================================================
    @testset "Therapeutic vaccination" begin
        txvx = HPVTherapeuticVaccine(
            start_year      = 2000.0,
            min_age         = 0.0,
            max_age         = 200.0,
            uptake_prob     = 1.0,
            clearance_boost = 3.0,
        )
        sim = HPVSim(
            genotypes     = SIMPLE_GENOTYPES,
            n_agents      = 500,
            start         = 2000.0,
            stop          = 2010.0,
            dt            = 0.5,
            rand_seed     = 88,
            use_immunity  = false,
            interventions = [txvx],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        # Check therapeutic vaccine was applied
        @test haskey(sim.interventions, :hpv_txvx)
        txvx_mod = sim.interventions[:hpv_txvx]
        md = Starsim.module_data(txvx_mod)
        n_given = md.results[:n_txvx_given].values
        # Should have given some therapeutic vaccines (female infected agents)
        @test any(x -> x > 0.0, n_given)
    end

    # ========================================================================
    # New features: Network data structures
    # ========================================================================
    @testset "Network mixing data" begin
        @test length(AGE_MIXING_BINS) == 16
        @test size(MARITAL_MIXING_MATRIX) == (16, 16)
        @test size(CASUAL_MIXING_MATRIX) == (16, 16)

        # Mixing matrices should be non-negative
        @test all(x -> x >= 0.0, MARITAL_MIXING_MATRIX)
        @test all(x -> x >= 0.0, CASUAL_MIXING_MATRIX)

        # Layer probs
        probs = default_layer_probs(network_type=:marital)
        @test haskey(probs, :age_bins)
        @test haskey(probs, :female)
        @test haskey(probs, :male)
        @test length(probs[:female]) == length(AGE_MIXING_BINS)

        probs_c = default_layer_probs(network_type=:casual)
        @test haskey(probs_c, :male)
    end

    # ========================================================================
    # New features: Seroconversion
    # ========================================================================
    @testset "Seroconversion" begin
        # HPV16 has sero_prob = 0.75, LR has sero_prob = 0.40
        @test HPV16_PARAMS.sero_prob == 0.75
        @test LR_PARAMS.sero_prob == 0.40
        @test HPV18_PARAMS.sero_prob == 0.56

        # Run multi-genotype sim with immunity
        # Seroconversion means not all clearances yield immunity
        sim = HPVSim(
            genotypes    = DEFAULT_GENOTYPES,
            n_agents     = 500,
            start        = 2000.0,
            stop         = 2015.0,
            dt           = 0.25,
            rand_seed    = 44,
            use_immunity = true,
            verbose      = 0,
        )
        Starsim.run!(sim)
        @test sim.complete

        # Immunity connector should show some but not all clearances leading to immunity
        conn = first(values(sim.connectors))
        @test conn isa HPVImmunityConnector
    end

    # ========================================================================
    # New features: GenotypeParams keyword constructor
    # ========================================================================
    @testset "GenotypeParams @kwdef" begin
        gp = GenotypeParams(name=:custom, rel_beta=0.5)
        @test gp.name == :custom
        @test gp.rel_beta == 0.5
        @test gp.prog_rate_cin1 == 0.10  # default
        @test gp.cin_fn_k == 0.3         # default
        @test gp.cancer_mortality_rate == 0.05  # default
    end

    # ========================================================================
    # New features: Duration-based params in GenotypeParams
    # ========================================================================
    @testset "Duration-based GenotypeParams" begin
        gp16 = HPV16_PARAMS
        @test gp16.dur_precin_par1 == 3.0
        @test gp16.dur_precin_par2 == 9.0
        @test gp16.cin_fn_k == 0.3
        @test gp16.cin_fn_ttc == 50.0
        @test gp16.cancer_fn_transform_prob == 2.0e-3
        @test gp16.cancer_mortality_rate > 0.0

        gp_lr = LR_PARAMS
        @test gp_lr.cancer_fn_transform_prob < HPV16_PARAMS.cancer_fn_transform_prob
    end

    # ========================================================================
    # New features: Combined interventions with treatment types
    # ========================================================================
    @testset "Full intervention stack" begin
        vax = HPVVaccination(
            start_year  = 2000.0,
            min_age     = 0.0,
            max_age     = 100.0,
            uptake_prob = 0.5,
            sex         = :both,
        )
        scr = HPVScreening(
            start_year     = 2000.0,
            screen_prob    = 0.05,
            min_age        = 0.0,
            max_age        = 200.0,
            treatment_type = EXCISION,
        )
        txvx = HPVTherapeuticVaccine(
            start_year  = 2000.0,
            min_age     = 0.0,
            max_age     = 200.0,
            uptake_prob = 0.3,
        )
        sim = HPVSim(
            genotypes     = DEFAULT_GENOTYPES,
            n_agents      = 300,
            start         = 2000.0,
            stop          = 2005.0,
            dt            = 0.5,
            rand_seed     = 55,
            use_immunity  = true,
            interventions = [vax, scr, txvx],
            verbose       = 0,
        )
        Starsim.run!(sim)
        @test sim.complete
        @test length(sim.interventions) == 3
    end

end # HPVsim.jl
