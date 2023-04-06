module MacroEconometrics

using Documenter
using TSFrames
using LinearAlgebra
using Dates
using Random, Distributions
using GLM: lm, coef, confint, residuals
using StatsModels: @formula, term
using Statistics
using TinyGibbs
using Parameters
using AbstractMCMC

# Write your package code here.
export MacroEconometricModel
export AbstractIdentificationMethod
include("./types.jl")

export BayesianEstimated, Estimated
export FixedEstimated
export FrequentistEstimated
include("estimated.jl")

export AbstractVectorAutoregression, AbstractVAREstimator
export simulate!, estimate!, predict, irf, make_companion_matrix, is_stable
export lag
export VAR
include("./VAR/utils.jl")
include("./VAR/types.jl")
include("./VAR/stability_checks.jl")
include("./VAR/companion_matrix.jl")
include("./VAR/simulation.jl")
export OlsVAREstimator
export IndependentNormalWishart, create_minnesota_params
include("./VAR/estimation.jl")


export ImpulseResponseFunction, StructuralImpulseResponseFunction
include("./IRF/types.jl")


end
