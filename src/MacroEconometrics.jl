module MacroEconometrics

using Documenter
using TSFrames

# Write your package code here.

export BayesianEstimated, Estimated
export FixedEstimated
include("estimated.jl")

export AbstractVectorAutoregression, AbstractVAREstimator
export simulate!, estimate!, predict, irf, make_companion_matrix, is_stable
export VAR
include("./VAR/types.jl")

end
