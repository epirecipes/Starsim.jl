using Documenter
using Starsim

makedocs(
    sitename = "Starsim.jl",
    modules = [Starsim],
    authors = "Simon Frost",
    remotes = nothing,
    warnonly = [:missing_docs, :cross_references, :autodocs_block],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Getting started" => "guide/getting_started.md",
            "Diseases" => "guide/diseases.md",
            "Networks" => "guide/networks.md",
            "Demographics" => "guide/demographics.md",
            "Interventions" => "guide/interventions.md",
            "CRN" => "guide/crn.md",
        ],
        "Vignettes" => [
            "Introduction" => "vignettes/01_introduction.md",
            "Building a model" => "vignettes/02_building_a_model.md",
            "Demographics" => "vignettes/03_demographics.md",
            "Disease models" => "vignettes/04_diseases.md",
            "Contact networks" => "vignettes/05_networks.md",
            "Interventions" => "vignettes/06_interventions.md",
            "Connectors" => "vignettes/07_connectors.md",
            "Measles (SEIR)" => "vignettes/08_measles.md",
            "Cholera (environmental)" => "vignettes/09_cholera.md",
            "Ebola (severity)" => "vignettes/10_ebola.md",
            "HIV (sexual networks)" => "vignettes/11_hiv.md",
            "MultiSim" => "vignettes/12_multisim.md",
            "Calibration" => "vignettes/13_calibration.md",
            "Malaria (metapopulation)" => "vignettes/14_malaria.md",
            "Composition (Catlab)" => "vignettes/15_composition.md",
            "Automatic differentiation" => "vignettes/16_automatic_differentiation.md",
            "GPU acceleration" => "vignettes/17_gpu.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/epirecipes/Starsim.jl.git",
    devbranch = "main",
)
