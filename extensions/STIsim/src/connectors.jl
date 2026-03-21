"""
HIV-STI coinfection connectors.
Port of Python `stisim.connectors.hiv_sti`.

Each connector modifies susceptibility and/or transmissibility
of one disease based on infection status of another.
"""

# ============================================================================
# HIV-Syphilis connector
# ============================================================================

"""
    HIVSyphConnector <: AbstractConnector

Bidirectional HIV-Syphilis coinfection connector.

# Parameters
- `rel_sus_hiv_syph::Real` — syphilis increases HIV susceptibility (default 2.67)
- `rel_trans_hiv_syph::Real` — syphilis increases HIV transmissibility (default 1.2)
- `rel_sus_syph_hiv::Real` — HIV increases syphilis susceptibility (default 1.0)
- `rel_trans_syph_hiv::Real` — HIV increases syphilis transmissibility (default 1.0)
"""
mutable struct HIVSyphConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    hiv_name::Symbol
    syph_name::Symbol
    rel_sus_hiv_syph::Float64
    rel_trans_hiv_syph::Float64
    rel_sus_syph_hiv::Float64
    rel_trans_syph_hiv::Float64
end

function HIVSyphConnector(;
    name::Symbol = :hiv_syph,
    hiv_name::Symbol = :hiv,
    syph_name::Symbol = :syphilis,
    rel_sus_hiv_syph::Real = 2.67,
    rel_trans_hiv_syph::Real = 1.2,
    rel_sus_syph_hiv::Real = 1.0,
    rel_trans_syph_hiv::Real = 1.0,
)
    md = Starsim.ModuleData(name; label="HIV-Syphilis connector")
    cd = Starsim.ConnectorData(md)
    HIVSyphConnector(cd, hiv_name, syph_name,
        Float64(rel_sus_hiv_syph), Float64(rel_trans_hiv_syph),
        Float64(rel_sus_syph_hiv), Float64(rel_trans_syph_hiv))
end

Starsim.connector_data(c::HIVSyphConnector) = c.data

function Starsim.init_pre!(c::HIVSyphConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function Starsim.step!(c::HIVSyphConnector, sim)
    haskey(sim.diseases, c.hiv_name) || return c
    haskey(sim.diseases, c.syph_name) || return c

    hiv = sim.diseases[c.hiv_name]
    syph = sim.diseases[c.syph_name]

    for u in sim.people.auids.values
        # Syphilis → HIV effects
        if syph.infection.infected.raw[u]
            hiv.infection.rel_sus.raw[u] *= c.rel_sus_hiv_syph
            hiv.infection.rel_trans.raw[u] *= c.rel_trans_hiv_syph
        end
        # HIV → Syphilis effects
        if hiv.infection.infected.raw[u]
            syph.infection.rel_sus.raw[u] *= c.rel_sus_syph_hiv
            syph.infection.rel_trans.raw[u] *= c.rel_trans_syph_hiv
        end
    end
    return c
end

# ============================================================================
# HIV-Gonorrhea connector
# ============================================================================

"""
    HIVGonConnector <: AbstractConnector

HIV-Gonorrhea coinfection connector.

# Parameters
- `rel_sus_hiv_ng::Real` — gonorrhea increases HIV susceptibility (default 1.2)
- `rel_trans_hiv_ng::Real` — gonorrhea increases HIV transmissibility (default 1.2)
"""
mutable struct HIVGonConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    hiv_name::Symbol
    ng_name::Symbol
    rel_sus_hiv_ng::Float64
    rel_trans_hiv_ng::Float64
end

function HIVGonConnector(;
    name::Symbol = :hiv_ng,
    hiv_name::Symbol = :hiv,
    ng_name::Symbol = :gonorrhea,
    rel_sus_hiv_ng::Real = 1.2,
    rel_trans_hiv_ng::Real = 1.2,
)
    md = Starsim.ModuleData(name; label="HIV-Gonorrhea connector")
    cd = Starsim.ConnectorData(md)
    HIVGonConnector(cd, hiv_name, ng_name, Float64(rel_sus_hiv_ng), Float64(rel_trans_hiv_ng))
end

Starsim.connector_data(c::HIVGonConnector) = c.data

function Starsim.init_pre!(c::HIVGonConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function Starsim.step!(c::HIVGonConnector, sim)
    haskey(sim.diseases, c.hiv_name) || return c
    haskey(sim.diseases, c.ng_name) || return c

    hiv = sim.diseases[c.hiv_name]
    ng = sim.diseases[c.ng_name]

    for u in sim.people.auids.values
        if ng.infection.infected.raw[u]
            hiv.infection.rel_sus.raw[u] *= c.rel_sus_hiv_ng
            hiv.infection.rel_trans.raw[u] *= c.rel_trans_hiv_ng
        end
    end
    return c
end

# ============================================================================
# HIV-Chlamydia connector
# ============================================================================

"""
    HIVChlamConnector <: AbstractConnector

HIV-Chlamydia coinfection connector.
"""
mutable struct HIVChlamConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    hiv_name::Symbol
    ct_name::Symbol
    rel_sus_hiv_ct::Float64
end

function HIVChlamConnector(;
    name::Symbol = :hiv_ct,
    hiv_name::Symbol = :hiv,
    ct_name::Symbol = :chlamydia,
    rel_sus_hiv_ct::Real = 1.0,
)
    md = Starsim.ModuleData(name; label="HIV-Chlamydia connector")
    cd = Starsim.ConnectorData(md)
    HIVChlamConnector(cd, hiv_name, ct_name, Float64(rel_sus_hiv_ct))
end

Starsim.connector_data(c::HIVChlamConnector) = c.data

function Starsim.init_pre!(c::HIVChlamConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function Starsim.step!(c::HIVChlamConnector, sim)
    haskey(sim.diseases, c.hiv_name) || return c
    haskey(sim.diseases, c.ct_name) || return c

    hiv = sim.diseases[c.hiv_name]
    ct = sim.diseases[c.ct_name]

    for u in sim.people.auids.values
        if ct.infection.infected.raw[u]
            hiv.infection.rel_sus.raw[u] *= c.rel_sus_hiv_ct
        end
    end
    return c
end

# ============================================================================
# HIV-Trichomoniasis connector
# ============================================================================

"""
    HIVTrichConnector <: AbstractConnector

HIV-Trichomoniasis coinfection connector.
"""
mutable struct HIVTrichConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    hiv_name::Symbol
    tv_name::Symbol
    rel_sus_hiv_tv::Float64
end

function HIVTrichConnector(;
    name::Symbol = :hiv_tv,
    hiv_name::Symbol = :hiv,
    tv_name::Symbol = :trichomoniasis,
    rel_sus_hiv_tv::Real = 1.5,
)
    md = Starsim.ModuleData(name; label="HIV-Trichomoniasis connector")
    cd = Starsim.ConnectorData(md)
    HIVTrichConnector(cd, hiv_name, tv_name, Float64(rel_sus_hiv_tv))
end

Starsim.connector_data(c::HIVTrichConnector) = c.data

function Starsim.init_pre!(c::HIVTrichConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function Starsim.step!(c::HIVTrichConnector, sim)
    haskey(sim.diseases, c.hiv_name) || return c
    haskey(sim.diseases, c.tv_name) || return c

    hiv = sim.diseases[c.hiv_name]
    tv = sim.diseases[c.tv_name]

    for u in sim.people.auids.values
        if tv.infection.infected.raw[u]
            hiv.infection.rel_sus.raw[u] *= c.rel_sus_hiv_tv
        end
    end
    return c
end

# ============================================================================
# HIV-BV connector
# ============================================================================

"""
    HIVBVConnector <: AbstractConnector

HIV-Bacterial Vaginosis coinfection connector.
"""
mutable struct HIVBVConnector <: Starsim.AbstractConnector
    data::Starsim.ConnectorData
    hiv_name::Symbol
    bv_name::Symbol
    rel_sus_hiv_bv::Float64
    rel_trans_hiv_bv::Float64
end

function HIVBVConnector(;
    name::Symbol = :hiv_bv,
    hiv_name::Symbol = :hiv,
    bv_name::Symbol = :bv,
    rel_sus_hiv_bv::Real = 2.0,
    rel_trans_hiv_bv::Real = 2.0,
)
    md = Starsim.ModuleData(name; label="HIV-BV connector")
    cd = Starsim.ConnectorData(md)
    HIVBVConnector(cd, hiv_name, bv_name, Float64(rel_sus_hiv_bv), Float64(rel_trans_hiv_bv))
end

Starsim.connector_data(c::HIVBVConnector) = c.data

function Starsim.init_pre!(c::HIVBVConnector, sim)
    md = Starsim.module_data(c)
    md.t = Starsim.Timeline(start=sim.pars.start, stop=sim.pars.stop, dt=sim.pars.dt)
    md.initialized = true
    return c
end

function Starsim.step!(c::HIVBVConnector, sim)
    haskey(sim.diseases, c.hiv_name) || return c
    haskey(sim.diseases, c.bv_name) || return c

    hiv = sim.diseases[c.hiv_name]
    bv = sim.diseases[c.bv_name]

    bv_inf_raw = if hasproperty(bv, :infection)
        bv.infection.infected.raw
    else
        bv.infected.raw
    end
    for u in sim.people.auids.values
        if bv_inf_raw[u]
            hiv.infection.rel_sus.raw[u] *= c.rel_sus_hiv_bv
            hiv.infection.rel_trans.raw[u] *= c.rel_trans_hiv_bv
        end
    end
    return c
end
