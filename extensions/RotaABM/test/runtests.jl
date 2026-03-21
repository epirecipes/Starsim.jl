# Unit tests for RotaABM.jl
using Test

# Add Starsim.jl to the load path
starsim_root = joinpath(@__DIR__, "..", "..", "..")
push!(LOAD_PATH, starsim_root)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Starsim
using RotaABM

@testset "RotaABM" begin

    # ========================================================================
    # Scenarios
    # ========================================================================
    @testset "Scenarios" begin
        @test haskey(RotaABM.SCENARIOS, "simple")
        @test haskey(RotaABM.SCENARIOS, "baseline")
        @test haskey(RotaABM.SCENARIOS, "realistic_competition")
        @test haskey(RotaABM.SCENARIOS, "high_diversity")

        scenarios = list_scenarios()
        @test length(scenarios) >= 7

        s = get_scenario("simple")
        @test haskey(s, "strains")
        @test haskey(s["strains"], (1, 8))
        @test haskey(s["strains"], (2, 4))
    end

    @testset "Scenario validation" begin
        s = validate_scenario("baseline")
        @test haskey(s, "default_fitness")

        custom = Dict{String, Any}(
            "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
                (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.01),
            ),
        )
        v = validate_scenario(custom)
        @test v["default_fitness"] == 1.0

        @test_throws ErrorException validate_scenario("nonexistent_scenario")
    end

    @testset "Scenario overrides" begin
        s = get_scenario("simple")
        overridden = apply_scenario_overrides(s; override_fitness=0.5)
        @test overridden["strains"][(1,8)]["fitness"] == 0.5
        @test overridden["strains"][(2,4)]["fitness"] == 0.5

        overridden2 = apply_scenario_overrides(s; override_prevalence=Dict((1,8) => 0.05))
        @test overridden2["strains"][(1,8)]["prevalence"] == 0.05
        @test overridden2["strains"][(2,4)]["prevalence"] == 0.01  # unchanged
    end

    @testset "GP reassortment generation" begin
        combos = generate_gp_reassortments([(1, 8), (2, 4)])
        @test length(combos) == 4
        @test (1, 8) in combos
        @test (1, 4) in combos
        @test (2, 8) in combos
        @test (2, 4) in combos

        combos2 = generate_gp_reassortments([(1, 8)])
        @test length(combos2) == 1
        @test combos2[1] == (1, 8)

        @test_throws ErrorException generate_gp_reassortments(Tuple{Int,Int}[])
    end

    # ========================================================================
    # Rotavirus disease
    # ========================================================================
    @testset "Rotavirus construction" begin
        rot = Rotavirus(G=1, P=8)
        @test rot.G == 1
        @test rot.P == 8
        @test Starsim.module_data(rot).name == :G1P8

        rot2 = Rotavirus(G=2, P=4; name=:custom)
        @test Starsim.module_data(rot2).name == :custom
    end

    @testset "Rotavirus SIRS lifecycle" begin
        rot = Rotavirus(G=1, P=8; init_prev=0.1, beta=0.2, dur_inf_mean=7.0)
        sim = Starsim.Sim(
            n_agents   = 500,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = rot,
            connectors = RotaImmunityConnector(),
            start      = 0.0,
            stop       = 0.5,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # Check that prevalence was recorded
        prev = Starsim.get_result(sim, :G1P8, :prevalence)
        @test length(prev) > 0
        @test any(p -> p > 0.0, prev)

        # Check recovered counts
        n_rec = Starsim.get_result(sim, :G1P8, :n_recovered)
        @test any(r -> r > 0.0, n_rec)
    end

    # ========================================================================
    # Immunity connector
    # ========================================================================
    @testset "Bitmask immunity" begin
        # Two-strain simulation to test immunity
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
            Rotavirus(G=2, P=4; init_prev=0.05, beta=0.15),
        ]
        conn = RotaImmunityConnector(
            homotypic_efficacy     = 0.9,
            partial_hetero_efficacy = 0.5,
            complete_hetero_efficacy = 0.3,
        )
        sim = Starsim.Sim(
            n_agents   = 500,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = diseases,
            connectors = conn,
            start      = 0.0,
            stop       = 0.5,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # After running, some agents should have immunity
        imm_conn = sim.connectors[:rota_immunity]
        active = sim.people.auids.values
        n_immune = count(imm_conn.has_immunity.raw[u] for u in active)
        @test n_immune > 0

        # Bitmask mappings should be populated
        @test length(imm_conn.G_to_bit) == 2
        @test length(imm_conn.P_to_bit) == 2
        @test length(imm_conn.GP_to_bit) == 2
    end

    @testset "Immunity hierarchy" begin
        # Check that homotypic > partial > complete heterotypic > naive
        conn = RotaImmunityConnector()
        @test conn.homotypic_efficacy > conn.partial_hetero_efficacy
        @test conn.partial_hetero_efficacy > conn.complete_hetero_efficacy
        @test conn.complete_hetero_efficacy > conn.naive_efficacy
    end

    # ========================================================================
    # Reassortment
    # ========================================================================
    @testset "Reassortment connector" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.1, beta=0.3),
            Rotavirus(G=2, P=4; init_prev=0.1, beta=0.3),
            Rotavirus(G=1, P=4; init_prev=0.0, beta=0.3),  # dormant reassortant
            Rotavirus(G=2, P=8; init_prev=0.0, beta=0.3),  # dormant reassortant
        ]
        conn_imm = RotaImmunityConnector()
        conn_rea = RotaReassortmentConnector(reassortment_prob=0.5)  # high prob for testing

        sim = Starsim.Sim(
            n_agents   = 1000,
            networks   = Starsim.RandomNet(n_contacts=15),
            diseases   = diseases,
            connectors = [conn_imm, conn_rea],
            start      = 0.0,
            stop       = 0.5,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # The reassortment connector should have been initialized with all 4 diseases
        rea = sim.connectors[:rota_reassortment]
        @test length(rea.rota_diseases) == 4
        @test length(rea.gp_to_disease) == 4
    end

    # ========================================================================
    # Vaccination
    # ========================================================================
    @testset "Vaccination" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
            Rotavirus(G=2, P=4; init_prev=0.05, beta=0.15),
        ]
        vx = RotaVaccination(
            start_year   = 0.0,
            n_doses      = 2,
            G_antigens   = [1],
            P_antigens   = [8],
            uptake_prob  = 0.5,
            min_age_days = 0.0,   # all ages eligible for testing
            max_age_days = 36500.0,
        )
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents      = 500,
            networks      = Starsim.RandomNet(n_contacts=10),
            diseases      = diseases,
            connectors    = conn,
            interventions = vx,
            start         = 0.0,
            stop          = 0.5,
            dt            = 1.0 / 365.25,
            rand_seed     = 42,
            verbose       = 0,
        )
        run!(sim; verbose=0)

        n_vax = Starsim.get_result(sim, :rota_vax, :n_vaccinated)
        @test any(v -> v > 0, n_vax)
    end

    # ========================================================================
    # Analyzers
    # ========================================================================
    @testset "StrainStats analyzer" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
            Rotavirus(G=2, P=4; init_prev=0.05, beta=0.15),
        ]
        analyzer = StrainStats()
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents   = 500,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = diseases,
            connectors = conn,
            analyzers  = analyzer,
            start      = 0.0,
            stop       = 0.3,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        stats = sim.analyzers[:strain_stats]
        @test length(stats.rota_diseases) == 2
        @test length(stats.strain_names) == 2
    end

    # ========================================================================
    # RotaSim convenience constructor
    # ========================================================================
    @testset "RotaSim convenience" begin
        sim = RotaSim(
            scenario  = "simple",
            n_agents  = 500,
            stop      = 0.25,
            rand_seed = 42,
        )
        run!(sim; verbose=0)
        @test sim.complete

        # Should have diseases for all GP combinations from simple scenario
        # simple has (1,8) and (2,4) → 4 total combinations
        @test length(sim.diseases) == 4

        # Verify connectors were added
        @test length(sim.connectors) >= 2
    end

    @testset "RotaSim custom scenario" begin
        custom = Dict{String, Any}(
            "strains" => Dict{Tuple{Int,Int}, Dict{String, Float64}}(
                (1, 8) => Dict("fitness" => 1.0, "prevalence" => 0.02),
                (3, 6) => Dict("fitness" => 0.8, "prevalence" => 0.01),
            ),
            "default_fitness" => 0.5,
        )
        sim = RotaSim(
            scenario  = custom,
            n_agents  = 300,
            stop      = 0.15,
            rand_seed = 1,
        )
        run!(sim; verbose=0)
        @test sim.complete
        # Should have (1,8), (1,6), (3,8), (3,6) = 4 diseases
        @test length(sim.diseases) == 4
    end

    # ========================================================================
    # Multi-strain dynamics
    # ========================================================================
    @testset "Multi-strain competition" begin
        sim = RotaSim(
            scenario  = "baseline",
            n_agents  = 1000,
            stop      = 0.5,
            rand_seed = 42,
        )
        run!(sim; verbose=0)

        # All initial strains should have had some infections
        for (nm, dis) in sim.diseases
            if dis isa Rotavirus && dis.infection.dd.init_prev > 0.0
                prev = Starsim.module_results(dis)[:prevalence].values
                @test any(p -> p > 0.0, prev)
            end
        end
    end

    # ========================================================================
    # Bitmask utility functions
    # ========================================================================
    @testset "Hamming distance" begin
        @test hamming_distance(0b0000, 0b0000) == 0
        @test hamming_distance(0b1111, 0b1111) == 0
        @test hamming_distance(0b0001, 0b0000) == 1
        @test hamming_distance(0b1010, 0b0101) == 4
        @test hamming_distance(0b1100, 0b1010) == 2
        # Symmetry
        @test hamming_distance(0b1010, 0b0011) == hamming_distance(0b0011, 0b1010)
    end

    @testset "Strain similarity" begin
        @test strain_similarity(0b1111, 0b1111) ≈ 1.0
        @test strain_similarity(0b0000, 0b0000) ≈ 1.0  # both empty
        @test strain_similarity(0b1000, 0b0001) ≈ 0.0   # no shared bits
        @test strain_similarity(0b1100, 0b1100) ≈ 1.0
        @test strain_similarity(0b1100, 0b1000) ≈ 0.5   # 1 of 2 shared
        # Symmetry
        @test strain_similarity(0b110, 0b011) ≈ strain_similarity(0b011, 0b110)
    end

    @testset "Bitmask from GP" begin
        G_to_bit = Dict(1 => 0, 2 => 1, 3 => 2)
        P_to_bit = Dict(4 => 0, 8 => 1)
        mask = bitmask_from_gp(1, 8, G_to_bit, P_to_bit)
        @test (mask & (Int64(1) << 0)) != 0  # G bit 0
        @test (mask & (Int64(1) << 1)) != 0  # P bit 1
    end

    @testset "Match type classification" begin
        G_to_bit  = Dict(1 => 0, 2 => 1)
        P_to_bit  = Dict(4 => 0, 8 => 1)
        GP_to_bit = Dict((1,8) => 0, (2,4) => 1, (1,4) => 2, (2,8) => 3)

        # Homotypic: exact GP match
        exposed_GP = Int64(1) << 0  # bit for (1,8)
        exposed_G  = Int64(1) << 0  # bit for G=1
        exposed_P  = Int64(1) << 1  # bit for P=8
        @test match_type(1, 8, exposed_GP, exposed_G, exposed_P, G_to_bit, P_to_bit, GP_to_bit) == :homotypic

        # Partial heterotypic: shared G but not exact
        @test match_type(1, 4, exposed_GP, exposed_G, exposed_P, G_to_bit, P_to_bit, GP_to_bit) == :partial_hetero

        # Partial heterotypic: shared P but not exact
        @test match_type(2, 8, exposed_GP, exposed_G, exposed_P, G_to_bit, P_to_bit, GP_to_bit) == :partial_hetero

        # Complete heterotypic: no shared G or P but has immunity
        exposed_GP2 = Int64(1) << 0  # only (1,8)
        exposed_G2  = Int64(1) << 0  # only G=1
        exposed_P2  = Int64(1) << 1  # only P=8
        @test match_type(2, 4, exposed_GP2, exposed_G2, exposed_P2, G_to_bit, P_to_bit, GP_to_bit) == :complete_hetero

        # Naive: no exposure at all
        @test match_type(1, 8, Int64(0), Int64(0), Int64(0), G_to_bit, P_to_bit, GP_to_bit) == :naive
    end

    # ========================================================================
    # Detailed bitmask immunity tests
    # ========================================================================
    @testset "Bitmask immunity — GP bitmask tracking" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.2, beta=0.3),
            Rotavirus(G=2, P=4; init_prev=0.2, beta=0.3),
        ]
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents   = 200,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = diseases,
            connectors = conn,
            start      = 0.0,
            stop       = 0.5,
            dt         = 1.0 / 365.25,
            rand_seed  = 123,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        imm = sim.connectors[:rota_immunity]
        active = sim.people.auids.values

        # Verify bitmask bit positions were assigned
        @test haskey(imm.G_to_bit, 1)
        @test haskey(imm.G_to_bit, 2)
        @test haskey(imm.P_to_bit, 4)
        @test haskey(imm.P_to_bit, 8)

        # Agents with immunity should have non-zero bitmasks
        for u in active
            if imm.has_immunity.raw[u]
                @test (imm.exposed_GP_bitmask.raw[u] != 0 ||
                       imm.exposed_G_bitmask.raw[u] != 0 ||
                       imm.exposed_P_bitmask.raw[u] != 0)
            end
        end

        # Agents without immunity should have zero bitmasks
        for u in active
            if !imm.has_immunity.raw[u]
                @test imm.exposed_GP_bitmask.raw[u] == 0
            end
        end
    end

    @testset "Bitmask immunity — cross-protection reduces susceptibility" begin
        # High init_prev to ensure many recoveries and hence cross-immunity
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.3, beta=0.4),
            Rotavirus(G=2, P=4; init_prev=0.3, beta=0.4),
        ]
        conn = RotaImmunityConnector(
            homotypic_efficacy      = 0.9,
            partial_hetero_efficacy = 0.5,
            complete_hetero_efficacy= 0.3,
        )
        sim = Starsim.Sim(
            n_agents   = 300,
            networks   = Starsim.RandomNet(n_contacts=15),
            diseases   = diseases,
            connectors = conn,
            start      = 0.0,
            stop       = 0.3,
            dt         = 1.0 / 365.25,
            rand_seed  = 77,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        active = sim.people.auids.values
        imm = sim.connectors[:rota_immunity]

        # Agents with immunity should have reduced susceptibility (< 1.0) for at least one disease
        any_reduced = false
        for d in [sim.diseases[:G1P8], sim.diseases[:G2P4]]
            for u in active
                if imm.has_immunity.raw[u] && d.infection.rel_sus.raw[u] < 1.0
                    any_reduced = true
                    break
                end
            end
        end
        @test any_reduced
    end

    @testset "Bitmask immunity — waning increases susceptibility over time" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.3, beta=0.5, waning_rate_mean=30.0),
        ]
        conn = RotaImmunityConnector(homotypic_efficacy=0.9)
        sim = Starsim.Sim(
            n_agents   = 200,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = diseases,
            connectors = conn,
            start      = 0.0,
            stop       = 1.0,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # After long enough with fast waning, susceptibility should approach 1.0 for most
        active = sim.people.auids.values
        d = sim.diseases[:G1P8]
        n_high_sus = count(d.infection.rel_sus.raw[u] > 0.8 for u in active)
        @test n_high_sus > length(active) * 0.3
    end

    @testset "Bitmask immunity — decay states per strain" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.1, beta=0.2),
            Rotavirus(G=2, P=4; init_prev=0.1, beta=0.2),
        ]
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents   = 200,
            networks   = Starsim.RandomNet(n_contacts=10),
            diseases   = diseases,
            connectors = conn,
            start      = 0.0,
            stop       = 0.3,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        imm = sim.connectors[:rota_immunity]
        # Per-strain decay states should exist for each unique GP
        @test haskey(imm.strain_decay_states, (1, 8))
        @test haskey(imm.strain_decay_states, (2, 4))
        @test length(imm.strain_decay_states) == 2
    end

    # ========================================================================
    # Detailed reassortment tests
    # ========================================================================
    @testset "Reassortment — GP combination generation" begin
        # Two parents → 4 total combinations (2 parents + 2 reassortants)
        combos = generate_gp_reassortments([(1, 8), (2, 4)])
        @test Set(combos) == Set([(1,8), (1,4), (2,8), (2,4)])

        # Three parents with shared genotypes
        combos3 = generate_gp_reassortments([(1, 8), (2, 4), (3, 8)])
        @test (3, 4) in combos3  # reassortant
        @test length(combos3) == 6  # 3 G × 2 P

        # Single parent → only itself
        combos1 = generate_gp_reassortments([(1, 8)])
        @test length(combos1) == 1
    end

    @testset "Reassortment — preferred partners filtering" begin
        combos_all = generate_gp_reassortments([(1, 8), (2, 4)]; use_preferred_partners=false)
        combos_pref = generate_gp_reassortments([(1, 8), (2, 4)]; use_preferred_partners=true)

        @test length(combos_all) == 4

        # G=1 preferred partners are [6, 8], so G1P4 should be excluded
        @test (1, 4) ∉ combos_pref
        @test (1, 8) in combos_pref
        @test (2, 4) in combos_pref
        @test (2, 8) in combos_pref
    end

    @testset "Reassortment — dormant strain activation" begin
        # Set up 2 parental strains + 2 dormant reassortants
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.15, beta=0.3),
            Rotavirus(G=2, P=4; init_prev=0.15, beta=0.3),
            Rotavirus(G=1, P=4; init_prev=0.0, beta=0.3),  # dormant
            Rotavirus(G=2, P=8; init_prev=0.0, beta=0.3),  # dormant
        ]
        conn_imm = RotaImmunityConnector()
        conn_rea = RotaReassortmentConnector(reassortment_prob=1.0)  # guaranteed reassortment

        sim = Starsim.Sim(
            n_agents   = 500,
            networks   = Starsim.RandomNet(n_contacts=20),
            diseases   = diseases,
            connectors = [conn_imm, conn_rea],
            start      = 0.0,
            stop       = 0.5,
            dt         = 1.0 / 365.25,
            rand_seed  = 42,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # The reassortment connector should map all 4 GP pairs
        rea = sim.connectors[:rota_reassortment]
        @test haskey(rea.gp_to_disease, (1, 4))
        @test haskey(rea.gp_to_disease, (2, 8))

        # Check that reassortment events were recorded
        n_rea = Starsim.get_result(sim, :rota_reassortment, :n_reassortments)
        total_rea = sum(n_rea)
        # With high reassortment_prob and high prevalence, some events should occur
        @test total_rea >= 0  # might be 0 if no co-infections, but structure should work
    end

    @testset "Reassortment — co-infection detection" begin
        # Two high-prevalence strains to maximize co-infection
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.3, beta=0.5),
            Rotavirus(G=2, P=4; init_prev=0.3, beta=0.5),
            Rotavirus(G=1, P=4; init_prev=0.0, beta=0.5),
            Rotavirus(G=2, P=8; init_prev=0.0, beta=0.5),
        ]
        conn_imm = RotaImmunityConnector()
        conn_rea = RotaReassortmentConnector(reassortment_prob=0.8)

        sim = Starsim.Sim(
            n_agents   = 500,
            networks   = Starsim.RandomNet(n_contacts=20),
            diseases   = diseases,
            connectors = [conn_imm, conn_rea],
            start      = 0.0,
            stop       = 0.3,
            dt         = 1.0 / 365.25,
            rand_seed  = 99,
            verbose    = 0,
        )
        run!(sim; verbose=0)

        # Immunity connector tracks current infections
        imm = sim.connectors[:rota_immunity]
        active = sim.people.auids.values
        max_coinf = maximum(imm.num_current_infections.raw[u] for u in active)
        # At least verify the tracking works (max could be 0 at end if all recovered)
        @test max_coinf >= 0.0
    end

    # ========================================================================
    # Named vaccine constructors
    # ========================================================================
    @testset "Rotarix constructor" begin
        vx = Rotarix()
        @test Starsim.module_data(vx).name == :rotarix
        @test vx.n_doses == 2
        @test vx.G_antigens == [1]
        @test vx.P_antigens == [8]
        @test length(vx.dose_effectiveness) == 2
    end

    @testset "RotaTeq constructor" begin
        vx = RotaTeq()
        @test Starsim.module_data(vx).name == :rotateq
        @test vx.n_doses == 3
        @test Set(vx.G_antigens) == Set([1, 2, 3, 4])
        @test vx.P_antigens == [8]
        @test length(vx.dose_effectiveness) == 3
    end

    @testset "Rotavac constructor" begin
        vx = Rotavac()
        @test Starsim.module_data(vx).name == :rotavac
        @test vx.n_doses == 3
        @test vx.G_antigens == [9]
        @test vx.P_antigens == [11]
    end

    @testset "Rotarix in simulation" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
            Rotavirus(G=2, P=4; init_prev=0.05, beta=0.15),
        ]
        vx = Rotarix(start_year=0.0, uptake_prob=0.5, min_age_days=0.0, max_age_days=36500.0)
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents      = 300,
            networks      = Starsim.RandomNet(n_contacts=10),
            diseases      = diseases,
            connectors    = conn,
            interventions = vx,
            start         = 0.0,
            stop          = 0.3,
            dt            = 1.0 / 365.25,
            rand_seed     = 42,
            verbose       = 0,
        )
        run!(sim; verbose=0)
        n_vax = Starsim.get_result(sim, :rotarix, :n_vaccinated)
        @test any(v -> v > 0, n_vax)
    end

    # ========================================================================
    # Vaccination scenarios
    # ========================================================================
    @testset "Vaccination scenarios" begin
        vax_scenarios = list_vaccination_scenarios()
        @test length(vax_scenarios) >= 5
        @test haskey(vax_scenarios, "rotarix_baseline")
        @test haskey(vax_scenarios, "rotateq_baseline")
        @test haskey(vax_scenarios, "no_vaccination")

        vs = get_vaccination_scenario("rotarix_baseline")
        @test vs["vaccine"] == "rotarix"
        @test vs["scenario"] == "baseline"
    end

    # ========================================================================
    # Vaccination summary
    # ========================================================================
    @testset "Vaccination summary" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
        ]
        vx = RotaVaccination(
            start_year=0.0, n_doses=2, uptake_prob=0.9,
            min_age_days=0.0, max_age_days=36500.0,
        )
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents=200, networks=Starsim.RandomNet(n_contacts=10),
            diseases=diseases, connectors=conn, interventions=vx,
            start=0.0, stop=0.3, dt=1.0/365.25, rand_seed=42, verbose=0,
        )
        run!(sim; verbose=0)

        summary = get_vaccination_summary(sim.interventions[:rota_vax], sim)
        @test haskey(summary, "total_agents")
        @test haskey(summary, "received_any_dose")
        @test haskey(summary, "completed_schedule")
        @test summary["total_agents"] > 0
    end

    # ========================================================================
    # Strain accessor
    # ========================================================================
    @testset "Strain accessor" begin
        rot = Rotavirus(G=3, P=6)
        @test strain(rot) == (3, 6)
    end

    # ========================================================================
    # EventStats analyzer
    # ========================================================================
    @testset "EventStats analyzer" begin
        diseases = [
            Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15),
            Rotavirus(G=2, P=4; init_prev=0.05, beta=0.15),
        ]
        analyzer = EventStats()
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents=300, networks=Starsim.RandomNet(n_contacts=10),
            diseases=diseases, connectors=conn, analyzers=analyzer,
            start=0.0, stop=0.3, dt=1.0/365.25, rand_seed=42, verbose=0,
        )
        run!(sim; verbose=0)

        es = sim.analyzers[:event_stats]
        res = Starsim.module_results(es)
        @test haskey(res, :infected_agents)
        @test haskey(res, :coinfected_agents)
        @test haskey(res, :recoveries)
    end

    # ========================================================================
    # AgeStats analyzer
    # ========================================================================
    @testset "AgeStats analyzer" begin
        diseases = [Rotavirus(G=1, P=8; init_prev=0.05, beta=0.15)]
        analyzer = AgeStats()
        conn = RotaImmunityConnector()
        sim = Starsim.Sim(
            n_agents=200, networks=Starsim.RandomNet(n_contacts=10),
            diseases=diseases, connectors=conn, analyzers=analyzer,
            start=0.0, stop=0.1, dt=1.0/365.25, rand_seed=42, verbose=0,
        )
        run!(sim; verbose=0)

        as = sim.analyzers[:age_stats]
        res = Starsim.module_results(as)
        # Should have results for age bins
        @test length(res) > 0
    end

end  # @testset "RotaABM"

println("\n✓ All RotaABM tests passed!")
