using Test
using Starsim
using DataFrames: nrow
using SparseArrays: nnz
using Graphs
using Statistics: mean

@testset "Starsim.jl" begin

    @testset "States" begin
        # UIDs
        u1 = UIDs([1, 2, 3])
        u2 = UIDs([2, 3, 4])
        @test length(u1) == 3
        @test intersect(u1, u2) == UIDs([2, 3])
        @test union(u1, u2) == UIDs([1, 2, 3, 4])
        @test setdiff(u1, u2) == UIDs([1])
        @test isempty(UIDs())

        # StateVector
        s = FloatState(:test; default=0.0)
        @test s.name == :test
        @test !s.initialized
    end

    @testset "Time" begin
        d = Duration(10.0, :days)
        @test to_years(d) ≈ 10.0 / 365.25 atol=0.001
        @test to_days(d) ≈ 10.0

        r = Rate(0.1, :peryear)
        p = to_prob(r, 1.0)
        @test 0.0 < p < 1.0

        tl = Timeline(start=0.0, stop=10.0, dt=1.0)
        @test tl.npts >= 10
    end

    @testset "Distributions" begin
        d = bernoulli(p=0.5)
        @test d isa BernoulliDist
        @test d.p == 0.5

        d2 = ss_normal(loc=0.0, scale=1.0)
        @test d2 isa StarsimDist
    end

    @testset "Parameters" begin
        p = Pars(:beta => 0.05, :n => 100)
        @test p[:beta] == 0.05
        update!(p, :beta => 0.1)
        @test p[:beta] == 0.1

        sp = SimPars(n_agents=1000, dt=0.1)
        @test sp.n_agents == 1000
        @test sp.dt == 0.1
    end

    @testset "Results" begin
        r = Result(:test; npts=10, scale=false)
        @test length(r) == 10
        r[1] = 5.0
        @test r[1] == 5.0

        rs = Results()
        push!(rs, r)
        @test haskey(rs, :test)
    end

    @testset "People" begin
        p = People(100)
        init_people!(p)
        @test length(p) == 100
        @test length(p.auids) == 100
        @test all(p.alive.raw[p.auids.values])

        # Grow
        new_uids = grow!(p, 10)
        @test length(new_uids) == 10
        @test length(p) == 110
    end

    @testset "SIR Simulation" begin
        sim = Sim(
            n_agents = 1000,
            networks = RandomNet(n_contacts=10),
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 180.0,
            verbose = 0,
        )
        run!(sim)

        @test sim.complete
        prev = get_result(sim, :sir, :prevalence)
        @test length(prev) > 0
        n_sus = get_result(sim, :sir, :n_susceptible)
        n_inf = get_result(sim, :sir, :n_infected)
        n_rec = get_result(sim, :sir, :n_recovered)

        # Basic sanity: S + I + R + dead = n_agents at each step
        for i in 1:length(n_sus)
            total = n_sus[i] + n_inf[i] + n_rec[i]
            @test total <= 1000.0 + 2.0  # Can decrease due to disease deaths
        end

        # Higher beta should produce more infections (use low betas to avoid saturation)
        sim_low = Sim(
            n_agents = 1000,
            networks = RandomNet(n_contacts=10),
            diseases = SIR(beta=0.005, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 180.0,
            rand_seed = 42,
            verbose = 0,
        )
        run!(sim_low)
        sim_high = Sim(
            n_agents = 1000,
            networks = RandomNet(n_contacts=10),
            diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 180.0,
            rand_seed = 42,
            verbose = 0,
        )
        run!(sim_high)
        n_rec_low = get_result(sim_low, :sir, :n_recovered)
        n_rec_high = get_result(sim_high, :sir, :n_recovered)
        @test n_rec_high[end] > n_rec_low[end]
    end

    @testset "SIS Simulation" begin
        sim = Sim(
            n_agents = 500,
            networks = RandomNet(n_contacts=5),
            diseases = SIS(beta=0.1, dur_inf=20.0, init_prev=0.05),
            dt = 1.0,
            stop = 180.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete
        prev = get_result(sim, :sis, :prevalence)
        @test length(prev) > 0
    end

    @testset "Demographics" begin
        sim = Sim(
            n_agents = 500,
            networks = RandomNet(n_contacts=5),
            diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
            demographics = [Births(birth_rate=20.0), Deaths(death_rate=10.0)],
            dt = 1.0,
            stop = 365.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete
        births = get_result(sim, :births)
        deaths = get_result(sim, :deaths)
        @test sum(births) > 0
        @test sum(deaths) > 0
    end

    @testset "MultiSim" begin
        base = Sim(
            n_agents = 500,
            networks = RandomNet(n_contacts=5),
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 60.0,
            verbose = 0,
        )

        # Basic run
        msim = MultiSim(base; n_runs=3)
        run!(msim; verbose=0)
        @test msim.complete
        @test length(msim.sims) == 3
        @test !isempty(msim.results)
        @test length(msim) == 3

        # Result keys exist
        rkeys = result_keys(msim)
        @test :sir_prevalence in rkeys
        @test :sir_n_infected in rkeys

        # mean_result
        m = mean_result(msim, :sir_prevalence)
        @test length(m) == base.t.npts
        @test all(isfinite, m)

        # quantile_result
        q10 = quantile_result(msim, :sir_prevalence, 0.1)
        q90 = quantile_result(msim, :sir_prevalence, 0.9)
        @test all(q10 .<= q90)

        # reduce! — median mode (default)
        reduce!(msim)
        @test msim.which === :reduced
        @test haskey(msim.reduced, :sir_prevalence)
        rr = msim.reduced[:sir_prevalence]
        @test length(rr.values) == base.t.npts
        @test all(rr.low .<= rr.values)
        @test all(rr.values .<= rr.high)

        # reduce! — mean mode
        reduce!(msim; use_mean=true, bounds=2.0)
        rr2 = msim.reduced[:sir_prevalence]
        @test length(rr2.low) == length(rr2.high)

        # MultiSim from vector of sims
        sims = [deepcopy(base) for _ in 1:4]
        for (i, s) in enumerate(sims); s.pars.rand_seed = i; end
        msim2 = MultiSim(sims)
        run!(msim2; verbose=0, parallel=false)
        @test msim2.complete
        @test msim2.n_runs == 4
    end

    @testset "Save/Load" begin
        sim = Sim(
            n_agents = 200,
            networks = RandomNet(n_contacts=5),
            diseases = SIR(beta=0.1, init_prev=0.05),
            dt = 1.0,
            stop = 10.0,
            verbose = 0,
        )
        run!(sim)

        # save/load round-trip
        tmpfile = tempname() * ".jls"
        save_sim(tmpfile, sim)
        @test isfile(tmpfile)
        sim2 = load_sim(tmpfile)
        @test sim2 isa Sim
        @test sim2.complete
        @test sim2.pars.n_agents == 200
        rm(tmpfile)

        # to_json
        d = to_json(sim)
        @test haskey(d, "pars")
        @test haskey(d, "results")
        @test d["pars"]["n_agents"] == 200
        @test haskey(d["results"], "time")

        # to_json with file
        tmpjson = tempname() * ".json"
        to_json(sim; filename=tmpjson)
        @test isfile(tmpjson)
        rm(tmpjson)
    end

    @testset "DataFrame export" begin
        sim = Sim(
            n_agents = 200,
            networks = RandomNet(n_contacts=5),
            diseases = SIR(beta=0.1, init_prev=0.05),
            dt = 1.0,
            stop = 5.0,
            verbose = 0,
        )
        run!(sim)
        df = to_dataframe(sim)
        @test :time in propertynames(df)
        @test nrow(df) > 0
    end

    @testset "Interventions" begin
        vx = Vx(efficacy=0.9)
        routine = RoutineDelivery(product=vx, prob=0.05, disease_name=:sir)

        sim = Sim(
            n_agents = 500,
            networks = RandomNet(n_contacts=5),
            diseases = SIR(beta=0.1, init_prev=0.05),
            interventions = [routine],
            dt = 1.0,
            stop = 5.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete
        delivered = get_result(sim, :n_delivered)
        @test sum(delivered) > 0
    end

    @testset "Demo" begin
        sim = demo(n_agents=200, verbose=0)
        @test sim.complete
    end

    @testset "InfectionLog" begin
        il = InfectionLog()
        sim = Sim(
            n_agents = 500,
            networks = RandomNet(n_contacts=10),
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.05),
            analyzers = [il],
            dt = 1.0,
            stop = 30.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete

        # Check events were collected
        @test haskey(il.events, :sir)
        events = il.events[:sir]
        @test length(events) > 0

        # Check graph was built
        @test haskey(il.graph, :sir)
        g = il.graph[:sir]
        @test nv(g) == sim.people.next_uid - 1
        @test ne(g) > 0

        # Check DataFrame export
        df = to_dataframe(il)
        @test :source in propertynames(df)
        @test :target in propertynames(df)
        @test :t in propertynames(df)
        @test nrow(df) == length(events)

        # Check TransmissionEvent fields
        ev = events[1]
        @test ev.target > 0
        @test ev.t >= 1
    end

    @testset "CRN — Common Random Numbers" begin
        # Test MultiRandom XOR combining
        mr = MultiRandom(:test_trans)
        init_dist!(mr; base_seed=42, trace="test.trans")
        @test mr.initialized

        # Draw pairwise random numbers
        src = UIDs([1, 2, 3, 4, 5])
        trg = UIDs([6, 7, 8, 9, 10])
        rands = multi_rvs(mr, src, trg)
        @test length(rands) == 5
        @test all(0.0 .<= rands .<= 1.0)

        # Determinism: same UIDs give same results after re-init
        init_dist!(mr; base_seed=42, trace="test.trans")
        jump_dt!(mr, 1)
        rands2 = multi_rvs(mr, src, trg)
        init_dist!(mr; base_seed=42, trace="test.trans")
        jump_dt!(mr, 1)
        rands3 = multi_rvs(mr, src, trg)
        @test rands2 == rands3

        # Different timesteps give different results
        init_dist!(mr; base_seed=42, trace="test.trans")
        jump_dt!(mr, 1)
        r_t1 = multi_rvs(mr, src, trg)
        init_dist!(mr; base_seed=42, trace="test.trans")
        jump_dt!(mr, 2)
        r_t2 = multi_rvs(mr, src, trg)
        @test r_t1 != r_t2

        # Test combine_rvs directly
        v1 = [0.1, 0.5, 0.9]
        v2 = [0.3, 0.7, 0.2]
        combined = combine_rvs([v1, v2])
        @test length(combined) == 3
        @test all(0.0 .<= combined .<= 1.0)

        # Test CRN-enabled simulation runs successfully
        old_scale = Starsim.OPTIONS.slot_scale
        Starsim.OPTIONS.slot_scale = 5.0
        try
            sim = Sim(
                n_agents = 500,
                networks = RandomNet(n_contacts=5),
                diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.05),
                dt = 1.0,
                stop = 60.0,
                verbose = 0,
            )
            run!(sim)
            @test sim.complete
            prev = get_result(sim, :sir, :prevalence)
            @test length(prev) > 0

            sim2 = Sim(
                n_agents = 50,
                networks = RandomNet(n_contacts=2),
                diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.0),
                dt = 1.0,
                stop = 5.0,
                rand_seed = 123,
                verbose = 0,
            )
            init!(sim2)
            sir = sim2.diseases[:sir]
            uids1 = UIDs([2, 5, 11])
            uids2 = UIDs([11, 2, 5])
            draws1 = Starsim._sample_recovery_draws(sir, sim2, uids1, 3)
            draws2 = Starsim._sample_recovery_draws(sir, sim2, uids2, 3)
            map1 = Dict(uid => draws1[i] for (i, uid) in pairs(uids1.values))
            map2 = Dict(uid => draws2[i] for (i, uid) in pairs(uids2.values))
            @test map1 == map2

            extended = UIDs([2, 5, 11, 17])
            draws3 = Starsim._sample_recovery_draws(sir, sim2, extended, 3)
            map3 = Dict(uid => draws3[i] for (i, uid) in pairs(extended.values))
            @test map1[2] == map3[2]
            @test map1[5] == map3[5]
            @test map1[11] == map3[11]
        finally
            Starsim.OPTIONS.slot_scale = old_scale
        end
    end

    @testset "Settings" begin
        @test Starsim.OPTIONS isa Starsim.Options
        @test Starsim.OPTIONS.slot_scale == 0.0  # default CRN disabled
        @test !crn_enabled()

        # Enable CRN
        old = Starsim.OPTIONS.slot_scale
        Starsim.OPTIONS.slot_scale = 5.0
        @test crn_enabled()
        Starsim.OPTIONS.slot_scale = old
    end

    @testset "Graphs.jl interop" begin
        e = Edges([1, 2, 3], [4, 5, 6])
        g = to_graph(e)
        @test nv(g) == 6
        @test ne(g) == 3

        dg = to_digraph(e)
        @test ne(dg) == 3

        A = to_adjacency_matrix(e)
        @test size(A) == (6, 6)
        @test nnz(A) == 6  # bidirectional

        # Round-trip: graph → edges
        e2 = Edges()
        from_graph!(e2, g)
        @test length(e2) == 3
    end

    @testset "Additional distributions" begin
        d1 = ss_uniform(low=0.0, high=10.0)
        init_dist!(d1; base_seed=42)
        v1 = rvs(d1, 100)
        @test all(0.0 .<= v1 .<= 10.0)

        d2 = ss_exponential(scale=2.0)
        init_dist!(d2; base_seed=42)
        v2 = rvs(d2, 100)
        @test all(v2 .>= 0.0)

        d3 = ss_poisson(lam=5.0)
        init_dist!(d3; base_seed=42)
        v3 = rvs(d3, 100)
        @test all(v3 .>= 0)

        d4 = ss_lognormal(mean=1.0, sigma=0.5)
        init_dist!(d4; base_seed=42)
        v4 = rvs(d4, 100)
        @test all(v4 .> 0.0)

        # New distribution types
        d5 = ss_beta(a=2.0, b=5.0)
        init_dist!(d5; base_seed=42)
        v5 = rvs(d5, 100)
        @test all(0.0 .<= v5 .<= 1.0)

        d6 = ss_weibull(c=2.0, scale=3.0)
        init_dist!(d6; base_seed=42)
        v6 = rvs(d6, 100)
        @test all(v6 .>= 0.0)

        d7 = ss_gamma(a=2.0, scale=1.5)
        init_dist!(d7; base_seed=42)
        v7 = rvs(d7, 100)
        @test all(v7 .>= 0.0)

        d8 = ss_constant(value=42.0)
        init_dist!(d8; base_seed=0)
        v8 = rvs(d8, 10)
        @test all(v8 .== 42.0)

        d9 = ss_choice(a=[1.0, 2.0, 3.0])
        init_dist!(d9; base_seed=42)
        v9 = rvs(d9, 100)
        @test all(v -> v in [1.0, 2.0, 3.0], v9)
    end

    @testset "SEIR Simulation" begin
        sim = Sim(
            n_agents = 1000,
            networks = RandomNet(n_contacts=10),
            diseases = SEIR(beta=0.3, dur_exp=5.0, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 180.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete

        n_sus = get_result(sim, :seir, :n_susceptible)
        n_exp = get_result(sim, :seir, :n_exposed)
        n_inf = get_result(sim, :seir, :n_infected)
        n_rec = get_result(sim, :seir, :n_recovered)
        prev = get_result(sim, :seir, :prevalence)

        # Conservation: S + E + I + R ≈ N
        for i in 1:length(n_sus)
            total = n_sus[i] + n_exp[i] + n_inf[i] + n_rec[i]
            @test abs(total - 1000.0) < 2.0
        end

        # Peak prevalence should be non-trivial
        @test maximum(prev) > 0.01

        # Exposed peak should precede infected peak
        exp_peak_idx = argmax(n_exp)
        inf_peak_idx = argmax(n_inf)
        @test exp_peak_idx <= inf_peak_idx
    end

    @testset "StaticNet" begin
        sim = Sim(
            n_agents = 200,
            networks = StaticNet(n_contacts=6),
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 60.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete
    end

    @testset "MSMNet" begin
        sim = Sim(
            n_agents = 200,
            networks = MSMNet(participation_rate=0.3),
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.05),
            dt = 1.0,
            stop = 60.0,
            verbose = 0,
        )
        run!(sim)
        @test sim.complete
    end

    @testset "PrenatalNet" begin
        net = PrenatalNet()
        @test network_data(net).mod.name == :prenatal
        # add_pairs! manually
        add_pairs!(net, [1, 2], [3, 4])
        @test length(network_edges(net)) == 2
    end

    @testset "PostnatalNet" begin
        net = PostnatalNet(dur=5.0)
        @test net.dur == 5.0
        @test network_data(net).mod.name == :postnatal
    end

    @testset "BreastfeedingNet" begin
        net = BreastfeedingNet()
        @test network_data(net).mod.name == :breastfeeding
    end

    @testset "HouseholdNet" begin
        using DataFrames
        hh_data = DataFrame(
            hh_id = [1, 2, 3, 4, 5],
            ages = ["30, 5", "45, 20, 10", "60", "25, 30, 8, 3", "40, 15"],
        )
        net = HouseholdNet(hh_data=hh_data)
        sim = Sim(
            n_agents = 20,
            networks = net,
            diseases = SIR(beta=0.1, dur_inf=10.0, init_prev=0.1),
            dt = 1.0, stop = 10.0, verbose = 0,
        )
        run!(sim)
        @test sim.complete
        # Check that household_ids were assigned
        @test length(net.household_ids) >= 20
        @test any(net.household_ids .> 0)
        # Check that edges were created (pairwise within households)
        @test length(network_edges(net)) > 0
    end

    @testset "Convenience constructors" begin
        # Test routine_vx
        rv = routine_vx(prob=0.05, efficacy=0.8)
        @test rv isa RoutineDelivery
        @test rv.prob == 0.05
        @test rv.iv.product isa Vx
        @test rv.iv.product.efficacy == 0.8

        # Test campaign_vx
        cv = campaign_vx(years=[2025.0], coverage=0.9, efficacy=0.95)
        @test cv isa CampaignDelivery
        @test cv.coverage == 0.9
        @test cv.iv.product.efficacy == 0.95

        # Test simple_vx (alias for routine)
        sv = simple_vx(prob=0.03, efficacy=0.7)
        @test sv isa RoutineDelivery
        @test sv.prob == 0.03

        # Test routine_screening
        rs = routine_screening(prob=0.2, sensitivity=0.9, specificity=0.95)
        @test rs isa RoutineDelivery
        @test rs.iv.product isa Dx
        @test rs.iv.product.sensitivity == 0.9
        @test rs.iv.product.specificity == 0.95

        # Test campaign_screening
        cs = campaign_screening(years=[5.0], coverage=0.6, sensitivity=0.8, specificity=0.9)
        @test cs isa CampaignDelivery
        @test cs.iv.product isa Dx

        # Test in a simulation
        sim = Sim(n_agents=500, diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
                  networks=RandomNet(n_contacts=10), interventions=[routine_vx(prob=0.02)],
                  start=0.0, stop=50.0, rand_seed=1, verbose=0)
        run!(sim)
        @test sim.complete
        prev = get_result(sim, :sir, :prevalence)
        @test maximum(prev) < 1.0  # vaccination should keep prevalence bounded
    end

    @testset "Utility functions" begin
        # Rate/probability conversions
        @test rate_prob(0.1, 1.0) ≈ 1 - exp(-0.1)
        @test rate_prob(0.0, 1.0) == 0.0
        @test rate_prob(1.0, 0.0) == 0.0

        @test time_prob(0.5, 1.0, 2.0) ≈ 1 - (1 - 0.5)^2
        @test time_prob(0.0, 1.0, 1.0) == 0.0
        @test time_prob(1.0, 1.0, 1.0) == 1.0

        r = prob_rate(0.5, 1.0)
        @test r > 0.0
        @test rate_prob(r, 1.0) ≈ 0.5 atol=1e-10  # round-trip

        # mock_sim
        ms = mock_sim()
        @test ms isa Sim
        @test ms.pars.n_agents == 100
        run!(ms; verbose=0)
        @test ms.complete

        # diff_sims and check_sims_match
        s1 = mock_sim(n_agents=200)
        run!(s1; verbose=0)
        s2 = mock_sim(n_agents=200)
        run!(s2; verbose=0)
        # Same seed → identical results
        @test check_sims_match(s1, s2)
        d = diff_sims(s1, s2)
        @test !isempty(d)
        for (_, v) in d
            @test v.max_diff == 0.0
        end
    end

    @testset "Distribution aliases" begin
        # ss_dur
        d1 = ss_dur(mean=5.0)
        @test d1 isa StarsimDist
        init_dist!(d1; base_seed=42)
        v1 = rvs(d1, 1000)
        @test all(v1 .>= 0.0)
        @test abs(mean(v1) - 5.0) < 1.0  # generous tolerance

        # ss_randint
        d2 = ss_randint(low=1, high=6)
        @test d2 isa ChoiceDist
        init_dist!(d2; base_seed=42)
        v2 = rvs(d2, 100)
        @test all(v -> v in 1.0:6.0, v2)
    end

    @testset "Catlab composition extension" begin
        using Catlab

        # EpiNet creation
        sir_net = EpiNet([:S, :I, :R], [:infection => (:S => :I), :recovery => (:I => :R)])
        @test nparts(sir_net, :S) == 3
        @test nparts(sir_net, :T) == 2

        sis_net = EpiNet([:S, :I], [:infection => (:S => :I), :recovery => (:I => :S)])
        @test nparts(sis_net, :S) == 2

        # EpiSharer creation
        sir = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01)
        net = RandomNet(n_contacts=10)
        ds = EpiSharer(:disease, sir)
        ns = EpiSharer(:network, net)
        @test ds !== nothing  # Successfully created

        # UWD construction
        uwd = epi_uwd([ds, ns])
        @test nparts(uwd, :Box) == 2
        @test nparts(uwd, :Junction) >= 1

        # compose_epi produces a working simulation
        sim = compose_epi([ds, ns]; n_agents=500, stop=30, rand_seed=1)
        @test sim isa Sim
        run!(sim; verbose=0)
        @test sim.complete

        prev = get_result(sim, :sir, :prevalence)
        @test length(prev) > 0

        # Composed sim matches manual sim
        sim2 = Sim(n_agents=500, diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.01),
                   networks=RandomNet(n_contacts=10), stop=30, rand_seed=1)
        run!(sim2; verbose=0)
        prev2 = get_result(sim2, :sir, :prevalence)
        @test maximum(abs.(prev .- prev2)) < 1e-10 # Composed sim must match manual sim exactly

        # to_sim from EpiNet
        sim3 = to_sim(sir_net; n_agents=500, beta=0.05, dur_inf=10.0,
                       networks=RandomNet(n_contacts=10), stop=30, rand_seed=1)
        run!(sim3; verbose=0)
        @test sim3.complete

        # OpenEpiNet
        open_net = OpenEpiNet(sir_net, [[1]])
        @test haskey(open_net, :net)
        @test haskey(open_net, :legs)
        @test length(open_net.legs) == 1

        # Multi-module composition with demographics
        sir3 = SIR(beta=0.05, dur_inf=10.0, init_prev=0.01)
        net3 = RandomNet(n_contacts=10)
        births = Births(birth_rate=20.0)
        deaths = Deaths(death_rate=15.0)
        ds3 = EpiSharer(:disease, sir3)
        ns3 = EpiSharer(:network, net3)
        demog_s = EpiSharer(:demog, [births, deaths])
        sim4 = compose_epi([ds3, ns3, demog_s]; n_agents=500, stop=20, rand_seed=1)
        run!(sim4; verbose=0)
        @test sim4.complete
    end

    @testset "GPU backends (conditional smoke tests)" begin
        function try_using_gpu(pkg::Symbol)
            try
                Base.eval(Main, Expr(:toplevel, Expr(:using, pkg)))
                return true
            catch
                return false
            end
        end

        function gpu_backend_usable(pkg::Symbol)
            try_using_gpu(pkg) || return false
            mod = getproperty(Main, pkg)
            if isdefined(mod, :functional)
                return getproperty(mod, :functional)()
            else
                return true
            end
        end

        function run_gpu_smoke(backend::Symbol)
            sim = Sim(
                n_agents = 500,
                networks = RandomNet(n_contacts=5),
                diseases = SIR(beta=0.05, dur_inf=10.0, init_prev=0.05),
                dt = 1.0,
                stop = 20.0,
                rand_seed = 42,
                verbose = 0,
            )
            run!(sim; verbose=0, backend=backend)
            @test sim.complete
            prev = get_result(sim, :sir, :prevalence)
            @test length(prev) == sim.t.npts
            @test all(isfinite, prev)

            sim2 = Sim(
                n_agents = 200,
                networks = RandomNet(n_contacts=4),
                diseases = SIR(beta=0.05, dur_inf=8.0, init_prev=0.05),
                dt = 1.0,
                stop = 5.0,
                rand_seed = 7,
                verbose = 0,
            )
            init!(sim2)
            gsim = to_gpu(sim2; backend=backend)
            sim2_cpu = to_cpu(gsim)
            @test sim2_cpu === sim2
            @test count(sim2.people.alive.raw) == sim2.pars.n_agents
        end

        function gpu_prevalence_trace(backend::Symbol; crn::Bool=false)
            old_slot_scale = Starsim.OPTIONS.slot_scale
            Starsim.OPTIONS.slot_scale = crn ? 5.0 : 0.0
            try
                sim = Sim(
                    n_agents = 400,
                    networks = RandomNet(n_contacts=4),
                    diseases = SIR(beta=0.05, dur_inf=8.0, init_prev=0.05),
                    dt = 1.0,
                    stop = 15.0,
                    rand_seed = 123,
                    verbose = 0,
                )
                run!(sim; verbose=0, backend=backend)
                return copy(get_result(sim, :sir, :prevalence))
            finally
                Starsim.OPTIONS.slot_scale = old_slot_scale
            end
        end

        function run_gpu_reproducibility(backend::Symbol)
            prev1 = gpu_prevalence_trace(backend; crn=false)
            prev2 = gpu_prevalence_trace(backend; crn=false)
            @test prev1 == prev2

            prev_crn_1 = gpu_prevalence_trace(backend; crn=true)
            prev_crn_2 = gpu_prevalence_trace(backend; crn=true)
            @test prev_crn_1 == prev_crn_2
        end

        function make_recovery_parity_sim(disease)
            graph_fn = (n, rng) -> begin
                g = Graphs.SimpleGraph(n)
                Graphs.add_edge!(g, 1, 2)
                return g
            end

            return Sim(
                n_agents = 2,
                networks = StaticNet(graph_fn=graph_fn, n_contacts=1),
                diseases = disease,
                dt = 1.0,
                stop = 4.0,
                rand_seed = 123,
                verbose = 0,
            )
        end

        function run_cpu_gpu_parity(backend::Symbol, disease_name::Symbol, disease; cache_edges::Bool=false)
            old_slot_scale = Starsim.OPTIONS.slot_scale
            Starsim.OPTIONS.slot_scale = 5.0
            try
                sim_cpu = make_recovery_parity_sim(disease)
                sim_gpu = make_recovery_parity_sim(deepcopy(disease))
                run!(sim_cpu; verbose=0)
                if cache_edges
                    run_gpu!(sim_gpu; verbose=0, backend=backend, cache_edges=true)
                else
                    run!(sim_gpu; verbose=0, backend=backend)
                end

                @test get_result(sim_cpu, disease_name, :n_infected) == get_result(sim_gpu, disease_name, :n_infected)
                @test get_result(sim_cpu, disease_name, :prevalence) == get_result(sim_gpu, disease_name, :prevalence)

                cpu_disease = sim_cpu.diseases[disease_name]
                gpu_disease = sim_gpu.diseases[disease_name]
                @test cpu_disease.infection.infected.raw == gpu_disease.infection.infected.raw
                @test cpu_disease.ti_recovered.raw ≈ gpu_disease.ti_recovered.raw atol=1e-6

                 if cpu_disease isa SEIR
                     @test get_result(sim_cpu, disease_name, :n_exposed) == get_result(sim_gpu, disease_name, :n_exposed)
                     @test get_result(sim_cpu, disease_name, :n_recovered) == get_result(sim_gpu, disease_name, :n_recovered)
                     @test cpu_disease.exposed.raw == gpu_disease.exposed.raw
                     @test cpu_disease.recovered.raw == gpu_disease.recovered.raw
                     @test cpu_disease.ti_exposed.raw ≈ gpu_disease.ti_exposed.raw atol=1e-6
                 end
            finally
                Starsim.OPTIONS.slot_scale = old_slot_scale
            end
        end

        @testset "Metal" begin
            if Sys.isapple() && gpu_backend_usable(:Metal)
                run_gpu_smoke(:metal)
                run_gpu_reproducibility(:metal)
                run_cpu_gpu_parity(:metal, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:metal, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0))
                run_cpu_gpu_parity(:metal, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:metal, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
                run_cpu_gpu_parity(:metal, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0); cache_edges=true)
                run_cpu_gpu_parity(:metal, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
            else
                @test_skip "Metal backend unavailable"
            end
        end

        @testset "CUDA" begin
            if gpu_backend_usable(:CUDA)
                run_gpu_smoke(:cuda)
                run_gpu_reproducibility(:cuda)
                run_cpu_gpu_parity(:cuda, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:cuda, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0))
                run_cpu_gpu_parity(:cuda, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:cuda, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
                run_cpu_gpu_parity(:cuda, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0); cache_edges=true)
                run_cpu_gpu_parity(:cuda, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
            else
                @test_skip "CUDA backend unavailable"
            end
        end

        @testset "AMDGPU" begin
            if gpu_backend_usable(:AMDGPU)
                run_gpu_smoke(:amdgpu)
                run_gpu_reproducibility(:amdgpu)
                run_cpu_gpu_parity(:amdgpu, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:amdgpu, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0))
                run_cpu_gpu_parity(:amdgpu, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0))
                run_cpu_gpu_parity(:amdgpu, :sir, SIR(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
                run_cpu_gpu_parity(:amdgpu, :sis, SIS(beta=100.0, dur_inf=8.0, init_prev=0.5, p_death=0.0, waning=0.0); cache_edges=true)
                run_cpu_gpu_parity(:amdgpu, :seir, SEIR(beta=100.0, dur_exp=1.0, dur_inf=8.0, init_prev=0.5, p_death=0.0); cache_edges=true)
            else
                @test_skip "AMDGPU backend unavailable"
            end
        end
    end

    @testset "GPU backend resolution" begin
        sim = Sim(
            n_agents = 50,
            networks = RandomNet(n_contacts=2),
            diseases = SIR(beta=0.05, dur_inf=5.0, init_prev=0.05),
            dt = 1.0,
            stop = 2.0,
            verbose = 0,
        )

        loaded = Starsim._loaded_gpu_backends()
        if isempty(loaded)
            err = try
                run!(sim; backend=:gpu, verbose=0)
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("Load one of", sprint(showerror, err))

            for backend in (:metal, :cuda, :amdgpu)
                berr = try
                    run!(sim; backend=backend, verbose=0)
                    nothing
                catch e
                    e
                end
                @test berr isa ErrorException
                @test occursin("requires loading", sprint(showerror, berr))
            end
        else
            @test !isempty(loaded)
        end

        uerr = try
            run!(sim; backend=:definitely_not_a_backend, verbose=0)
            nothing
        catch e
            e
        end
        @test uerr isa ErrorException
        @test occursin("Unknown backend", sprint(showerror, uerr))
    end

    @testset "Makie extension (stubs)" begin
        sim = Sim(n_agents=200, diseases=SIR(beta=0.05, dur_inf=10.0, init_prev=0.1),
                  networks=RandomNet(n_contacts=4), start=0.0, stop=10.0, rand_seed=1)
        run!(sim; verbose=0)

        # Test the helper functions that the Makie extension relies on
        pr = Starsim._collect_plot_results(sim)
        @test length(pr) > 0
        @test pr[1] isa Pair{Symbol, Starsim.Result}

        tvec = Starsim._sim_tvec(sim)
        @test length(tvec) == sim.t.npts
        @test tvec[1] ≈ sim.pars.start

        # Stub functions are exported and callable (MethodError before extension loads)
        @test isdefined(Starsim, :plot_sim)
        @test isdefined(Starsim, :plot_disease)
        @test isdefined(Starsim, :plot_comparison)

        # Verify result fields the extension depends on
        _, first_result = pr[1]
        @test hasproperty(first_result, :values)
        @test hasproperty(first_result, :label)
        @test hasproperty(first_result, :auto_plot)
        @test first_result.auto_plot == true
    end

end
