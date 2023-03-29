using MacroTools: @forward
"""
The idea is to abstract he model from the way it has been estimated. In it's
most essential form, a model is just a struct, with the same structure no matter
how is has been estimated. The only time that estimation matter is when
statistics about estimated quantities are asked for. Thus, any statistic that
needs to be estimated in a model should be of type Estimated. Estimated itself
can then be either Frequentist or Bayesian.
"""
abstract type Estimated end 

"""
Since conjugacy is incredible rare and often not actually plausible (the priors
are not plausible) in actual work, we will assume that all Bayesian estimated
quantities are just samples, and can thus be put in an array. Additional meta
data, such as warning information during sampling, etc can be put in the
metadata field.

## Behaviour

- Most basic iterations will directly be forwarded to the value field. As such,
  the type can be used like any other array, including multiplication with other arrays.
"""
struct BayesianEstimated{T, M}<:Estimated
    value::Array{T}
    metadata::M
end
@forward BayesianEstimated.value Base.getindex, Base.length, Base.size, Base.ndims, 
    Base.first, Base.last, Base.lastindex, Base.firstindex, Base.setindex!, 
    Base.eltype, Base.eachslice, Base.eachcol, Base.eachrow

ops = [:+, :-, :*, :/]
for op in ops
    eval(:(Base.$op(x::V, be::BayesianEstimated{T, M}) where {V,T, M} = Base.$op(x, be.value)))
    eval(:(Base.$op(be::BayesianEstimated{T, M}, args...) where {T, M} = Base.$op(be.value, args...)))
end

Base.broadcasted(f, be::BayesianEstimated{T, M}, args...) where {T, M} = Base.broadcasted(f, be.value, args...)
Base.broadcasted(f, x::V, be::BayesianEstimated{T, M}) where {V, T, M} = Base.broadcasted(f, x, be.value)
