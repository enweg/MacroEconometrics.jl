module MacroEconometrics

using Documenter
using TSFrames
using LinearAlgebra

# Write your package code here.

export BayesianEstimated, Estimated
export FixedEstimated
include("estimated.jl")

export AbstractVectorAutoregression, AbstractVAREstimator
export simulate!, estimate!, predict, irf, make_companion_matrix, is_stable
export VAR
include("./VAR/utils.jl")
include("./VAR/types.jl")

end
