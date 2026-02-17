using Documenter
using StructuralSearchModels

makedocs(sitename="StructuralSearchModels.jl",
    format = Documenter.HTML(
        prettyurls = true,
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
        ]
)

deploydocs(
    repo = "github.com/rgreminger/StructuralSearchModels.jl.git",
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
