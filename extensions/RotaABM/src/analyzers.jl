"""
Analyzers for multi-strain rotavirus simulations.
Port of Python `rotasim.analyzers`.
"""

# ============================================================================
# StrainStats
# ============================================================================

"""
    StrainStats <: AbstractAnalyzer

Tracks infection counts and proportions for each Rotavirus strain.
"""
mutable struct StrainStats <: Starsim.AbstractAnalyzer
    data::Starsim.AnalyzerData
    rota_diseases::Vector{Rotavirus}
    strain_names::Vector{String}
end

function StrainStats(; name::Symbol = :strain_stats)
    md = Starsim.ModuleData(name; label="Strain statistics")
    ad = Starsim.AnalyzerData(md)
    StrainStats(ad, Rotavirus[], String[])
end

Starsim.analyzer_data(a::StrainStats) = a.data

function Starsim.init_pre!(a::StrainStats, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    a.rota_diseases = Rotavirus[]
    a.strain_names  = String[]
    for (_, dis) in sim.diseases
        if dis isa Rotavirus
            push!(a.rota_diseases, dis)
            push!(a.strain_names, "($(dis.G), $(dis.P))")
        end
    end

    npts = md.t.npts
    for sn in a.strain_names
        Starsim.define_results!(a,
            Starsim.Result(Symbol(sn, " proportion"); npts=npts, label="$sn proportion", scale=false),
            Starsim.Result(Symbol(sn, " count"); npts=npts, label="$sn count"),
        )
    end

    md.initialized = true
    return a
end

function Starsim.step!(a::StrainStats, sim)
    isempty(a.rota_diseases) && return a
    md = Starsim.module_data(a)
    ti = md.t.ti
    res = Starsim.module_results(a)

    total_count = 0.0
    counts = Float64[]
    for d in a.rota_diseases
        c = Float64(count(d.infection.infected.raw[u] for u in sim.people.auids.values))
        push!(counts, c)
        total_count += c
    end

    for (i, sn) in enumerate(a.strain_names)
        cnt_key  = Symbol(sn, " count")
        prop_key = Symbol(sn, " proportion")
        if haskey(res, cnt_key) && ti <= length(res[cnt_key].values)
            res[cnt_key][ti] = counts[i]
        end
        if haskey(res, prop_key) && ti <= length(res[prop_key].values)
            res[prop_key][ti] = total_count > 0 ? counts[i] / total_count : 0.0
        end
    end

    return a
end

# ============================================================================
# EventStats
# ============================================================================

"""
    EventStats <: AbstractAnalyzer

Tracks key simulation events: infections, recoveries, infected/coinfected agents.
"""
mutable struct EventStats <: Starsim.AbstractAnalyzer
    data::Starsim.AnalyzerData
    rota_diseases::Vector{Rotavirus}
end

function EventStats(; name::Symbol = :event_stats)
    md = Starsim.ModuleData(name; label="Event statistics")
    ad = Starsim.AnalyzerData(md)
    EventStats(ad, Rotavirus[])
end

Starsim.analyzer_data(a::EventStats) = a.data

function Starsim.init_pre!(a::EventStats, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    a.rota_diseases = Rotavirus[]
    for (_, dis) in sim.diseases
        dis isa Rotavirus && push!(a.rota_diseases, dis)
    end

    npts = md.t.npts
    for ev in [:infected_agents, :coinfected_agents, :recoveries, :new_infections]
        Starsim.define_results!(a,
            Starsim.Result(ev; npts=npts, label=string(ev)),
        )
    end

    md.initialized = true
    return a
end

function Starsim.step!(a::EventStats, sim)
    isempty(a.rota_diseases) && return a
    md = Starsim.module_data(a)
    ti = md.t.ti
    res = Starsim.module_results(a)
    active = sim.people.auids.values

    # Count per-agent infection count
    n_infected = 0
    n_coinfected = 0
    for u in active
        cnt = 0
        for d in a.rota_diseases
            if d.infection.infected.raw[u]
                cnt += 1
            end
        end
        if cnt > 0; n_infected += 1; end
        if cnt > 1; n_coinfected += 1; end
    end

    if haskey(res, :infected_agents) && ti <= length(res[:infected_agents].values)
        res[:infected_agents][ti] = Float64(n_infected)
    end
    if haskey(res, :coinfected_agents) && ti <= length(res[:coinfected_agents].values)
        res[:coinfected_agents][ti] = Float64(n_coinfected)
    end

    # Sum recoveries / new_infections across diseases
    total_rec = 0.0
    total_inf = 0.0
    for d in a.rota_diseases
        dres = Starsim.module_results(d)
        if haskey(dres, :new_recovered) && ti <= length(dres[:new_recovered].values)
            total_rec += dres[:new_recovered][ti]
        end
    end
    if haskey(res, :recoveries) && ti <= length(res[:recoveries].values)
        res[:recoveries][ti] = total_rec
    end

    return a
end

# ============================================================================
# AgeStats
# ============================================================================

"""
    AgeStats <: AbstractAnalyzer

Tracks age distribution of infected population over time.
"""
mutable struct AgeStats <: Starsim.AbstractAnalyzer
    data::Starsim.AnalyzerData
    age_bins::Vector{Float64}
    age_labels::Vector{String}
    rota_diseases::Vector{Rotavirus}
end

function AgeStats(; name::Symbol = :age_stats)
    md = Starsim.ModuleData(name; label="Age statistics")
    ad = Starsim.AnalyzerData(md)
    age_bins   = [2/12, 4/12, 6/12, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    age_labels = ["0-2m", "2-4m", "4-6m", "6-12m", "12-24m", "24-36m", "36-48m", "48-60m", "60m+"]
    AgeStats(ad, age_bins, age_labels, Rotavirus[])
end

Starsim.analyzer_data(a::AgeStats) = a.data

function Starsim.init_pre!(a::AgeStats, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    a.rota_diseases = Rotavirus[]
    for (_, dis) in sim.diseases
        dis isa Rotavirus && push!(a.rota_diseases, dis)
    end

    npts = md.t.npts
    for lbl in a.age_labels
        Starsim.define_results!(a,
            Starsim.Result(Symbol(lbl); npts=npts, label=lbl),
        )
    end

    md.initialized = true
    return a
end

function Starsim.step!(a::AgeStats, sim)
    md = Starsim.module_data(a)
    ti = md.t.ti
    res = Starsim.module_results(a)
    active = sim.people.auids.values

    # Bin ages
    counts = zeros(Int, length(a.age_bins))
    for u in active
        age = sim.people.age.raw[u]
        bin = searchsortedfirst(a.age_bins, age)
        bin = min(bin, length(a.age_bins))
        counts[bin] += 1
    end

    for (i, lbl) in enumerate(a.age_labels)
        k = Symbol(lbl)
        if haskey(res, k) && ti <= length(res[k].values)
            res[k][ti] = Float64(counts[i])
        end
    end

    return a
end
