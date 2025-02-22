using Documenter
using StructuralSearchModels

makedocs(sitename="StructuralSearchModels.jl",
    pages = [
        "index.md",
        "Getting Started" => "getting_started.md",
        "Tutorials and Examples" => "tutorials.md",
        "How it Works" => "how_it_works.md",
        "Models" => "models.md",
        "API Reference" => "api.md",
        "Contributing" => "contributing.md",
        ]
)

