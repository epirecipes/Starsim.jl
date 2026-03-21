"""
STI-specific analyzers.
"""

# ============================================================================
# CoinfectionAnalyzer
# ============================================================================

"""
    CoinfectionAnalyzer <: AbstractAnalyzer

Track coinfection prevalence across STI diseases.
"""
mutable struct CoinfectionAnalyzer <: Starsim.AbstractAnalyzer
    data::Starsim.ModuleData
    disease_pairs::Vector{Tuple{Symbol, Symbol}}
end

function CoinfectionAnalyzer(;
    name::Symbol = :coinf_analyzer,
    disease_pairs::Vector{Tuple{Symbol, Symbol}} = Tuple{Symbol, Symbol}[],
)
    md = Starsim.ModuleData(name; label="Coinfection analyzer")
    CoinfectionAnalyzer(md, disease_pairs)
end

Starsim.module_data(a::CoinfectionAnalyzer) = a.data

function Starsim.init_pre!(a::CoinfectionAnalyzer, sim)
    md = Starsim.module_data(a)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)

    # Auto-detect disease pairs if not provided
    if isempty(a.disease_pairs)
        names = collect(keys(sim.diseases))
        for i in 1:length(names)
            for j in (i+1):length(names)
                push!(a.disease_pairs, (names[i], names[j]))
            end
        end
    end

    npts = md.t.npts
    for (d1, d2) in a.disease_pairs
        rname = Symbol("coinf_$(d1)_$(d2)")
        Starsim.define_results!(a,
            Starsim.Result(rname; npts=npts, label="Coinfected $d1+$d2", scale=false),
        )
    end

    md.initialized = true
    return a
end

function Starsim.step!(a::CoinfectionAnalyzer, sim)
    md = Starsim.module_data(a)
    ti = md.t.ti
    active = sim.people.auids.values

    for (d1, d2) in a.disease_pairs
        haskey(sim.diseases, d1) || continue
        haskey(sim.diseases, d2) || continue
        dis1 = sim.diseases[d1]
        dis2 = sim.diseases[d2]

        n_coinf = 0
        @inbounds for u in active
            if dis1.infection.infected.raw[u] && dis2.infection.infected.raw[u]
                n_coinf += 1
            end
        end

        rname = Symbol("coinf_$(d1)_$(d2)")
        res = md.results
        if haskey(res, rname) && ti <= length(res[rname].values)
            res[rname][ti] = Float64(n_coinf)
        end
    end

    return a
end
