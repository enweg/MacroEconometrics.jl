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
export VAR, SVAR
include("./VAR/utils.jl")
include("./VAR/types.jl")
include("./VAR/stability_checks.jl")
include("./VAR/companion_matrix.jl")
include("./VAR/simulation.jl")

export ImpulseResponseFunction, StructuralImpulseResponseFunction
export to_impact_normalisation
export CholeskyVAR
include("./IRF/types.jl")
include("./VAR/irf.jl")

export OlsVAREstimator
export IndependentNormalWishart, create_minnesota_params
include("./VAR/estimation.jl")


export stack_last_dim
include("./utils.jl")


end
