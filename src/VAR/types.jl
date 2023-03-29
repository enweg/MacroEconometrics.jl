using DataFrames

abstract type AbstractVectorAutoregression end
abstract type AbstractVAREstimator end

function simulate!(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
function estimate!(var::AbstractVectorAutoregression, method::AbstractVAREstimator, args...; kwargs...) end
function predict(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
function irf(var::AbstractVectorAutoregression, horizon, args...; kwargs...) end
function make_companion_matrix(var::AbstractVectorAutoregression) end
function is_stable(var::AbstractVectorAutoregression) end

mutable struct VAR{E<:Estimated}
    n::Int
    p::Int
    B::E
    b0::E 
    Σ::E

    data::DataFrame
end