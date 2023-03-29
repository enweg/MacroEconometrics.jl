using MacroEconometrics
using Documenter

DocMeta.setdocmeta!(MacroEconometrics, :DocTestSetup, :(using MacroEconometrics); recursive=true)

makedocs(;
    modules=[MacroEconometrics],
    authors="Enrico Wegner",
    repo="https://github.com/enweg/MacroEconometrics.jl/blob/{commit}{path}#{line}",
    sitename="MacroEconometrics.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://enweg.github.io/MacroEconometrics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/enweg/MacroEconometrics.jl",
    devbranch="main",
)
