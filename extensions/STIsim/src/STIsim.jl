"""
    STIsim

Sexually-transmitted infection agent-based models built on Starsim.jl.
Port of the Python `stisim` package with per-act transmission, SEIS diseases,
multi-stage HIV/syphilis, coinfection connectors, and STI-specific interventions.
"""
module STIsim

using Starsim
using Random
using StableRNGs
using Distributions
using Statistics
using OrderedCollections

# Base STI disease (SEIS pattern, per-act transmission)
include("sti_base.jl")

# Specific diseases
include("hiv.jl")
include("syphilis.jl")
include("chlamydia.jl")
include("gonorrhea.jl")
include("trichomoniasis.jl")
include("bv.jl")

# Networks
include("networks.jl")

# Connectors
include("connectors.jl")

# Interventions
include("interventions.jl")

# Analyzers
include("analyzers.jl")

# Convenience simulation constructor
include("sim.jl")

export # Base
       BaseSTI, SEIS,
       # Diseases
       HIV, Syphilis, Chlamydia, Gonorrhea, Trichomoniasis, BacterialVaginosis,
       # Networks
       StructuredSexual,
       # Connectors
       HIVSyphConnector, HIVGonConnector, HIVChlamConnector,
       HIVTrichConnector, HIVBVConnector,
       # Interventions
       STITest, STITreatment, ART, VMMC, PrEP,
       # Analyzers
       CoinfectionAnalyzer,
       # Sim
       STISim

end # module STIsim
