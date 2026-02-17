using Documenter
using StructuralSearchModels

makedocs(
    sitename="StructuralSearchModels.jl",
    authors = "Rafael P. Greminger",
    modules = [StructuralSearchModels],
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://rgreminger.github.io/StructuralSearchModels.jl/stable",
    ),
    pages = [
        "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials and Examples" => "tutorials.md",
        "Models" => "models.md",
        "Data" => "data_generation.md",
        "Estimation" => "estimation.md",
        "Post-Estimation" => "post_estimation.md",
        # "Public Interface" => "api.md",
        # "Contributing" => "contributing.md",
        ],
    clean = true, linkcheck = true,
        warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/rgreminger/StructuralSearchModels.jl.git",
    push_preview = true,
)

# makedocs(sitename="StructuralSearchModels.jl",
#     format = Documenter.LaTeX(),
#     pages = [
#         "index.md",
#         "Getting Started" => "getting_started.md",
#         "Tutorials and Examples" => "tutorials.md",
#         "Models" => "models.md",
#         "Data" => "data_generation.md",
#         "Estimation" => "estimation.md",
#         "Post-Estimation" => "post_estimation.md",
#         # "Public Interface" => "api.md",
#         # "Contributing" => "contributing.md",
#         ]
# )
