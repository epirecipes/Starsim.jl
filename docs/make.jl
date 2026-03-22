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
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/epirecipes/Starsim.jl.git",
    devbranch = "main",
)
