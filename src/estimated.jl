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


""""
  Fixing the estimation.

Sometimes we want to fix a parameter to a specific value. Although this
parameter is no-longer technically estimated, it is still an estimable quantity
and thus is still a subtype of Estimated. 

## Fields

-`value::Array{T}`: The value of the quantity
"""
struct FixedEstimated{T}<:Estimated
  value::Array{T}
end
@forward FixedEstimated.value Base.getindex, Base.length, Base.size, Base.ndims, 
    Base.first, Base.last, Base.lastindex, Base.firstindex, Base.setindex!, 
    Base.eltype, Base.eachslice, Base.eachcol, Base.eachrow

ops = [:+, :-, :*, :/]
for op in ops
    eval(:(Base.$op(x::V, fe::FixedEstimated{T}) where {V, T} = Base.$op(x, fe.value)))
    eval(:(Base.$op(fe::FixedEstimated{T}, args...) where {T} = Base.$op(fe.value, args...)))
end

Base.broadcasted(f, fe::FixedEstimated{T}, args...) where {T} = Base.broadcasted(f, fe.value, args...)
Base.broadcasted(f, x::V, fe::FixedEstimated{T}) where {V, T} = Base.broadcasted(f, x, fe.value)


"""
Since conjugacy is incredibly rare and often not actually plausible (the priors
are not plausible) in actual work, we will assume that all Bayesian estimated
quantities are just samples, and can thus be put in an array. Additional meta
data, such as warning information during sampling, etc can be put in the
metadata field.

## Behaviour

- Most basic operations will directly be forwarded to the value field. As such,
  the type can be used like any other array, including multiplication with other
  arrays, iteration, etc.
  
## Fields

- `value::Array{T}`: The actual values of the estimated quantity. Should be an
  Array; If multiple chains have been used, then the chains should be stacked
  along the last dimension; That is, if only a single chain is being used, then
  the last dimension should be of length 1. So for a matrix B that is n×n, we
  would have an array of dimensions n×n×d×c where d are the number of draws, and
  c is the number of chains.
- `metadata::M`: Any additional data that one wishes to save relating to the
  estimation. This could be warnings from the sampling algorithms, etc.
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
Base.mapslices(f, be::BayesianEstimated{T, M}; dims=[ndims(be)-1, ndims(be)]) where {T, M} = Base.mapslices(f, be.value; dims=dims)

"""
Frequentist estimations usually consist of a point estimate and a confidence
interval. If Bootstrapping is used, the confidence interval must not be
symmetric. As such, having a separate lower and upper CI allows for asymmetric
CIs. 

## Behaviour

Most operations are directly forwarded to the value field. For example,
additiona, subtraction, ... are all forwarded to the `value` field. So is
indexing. 

## Fields

- `value::Array{T}`: The point estimates. 
- `ci_lower::Array{T}`: The lower end of the CI for each value in `values`
- `ci_upper::Array{T}`: The upper end of the CI for each value in `values`
- `metadata::M`: Any metadata. Should, for example, include the level of the CI.

"""
struct FrequentistEstimated{T, M} <: Estimated
  value::Array{T}
  ci_lower::Array{T}
  ci_upper::Array{T}
  metadata::M

  function FrequentistEstimated(value::Array{T}, ci_lower::Array{T}, ci_upper::Array{T}, metadata::M) where {T, M}
    @assert all(size(value) .== size(ci_lower))
    @assert all(size(value) .== size(ci_upper))
    @assert all(ci_lower .<= ci_upper)
    @assert all(ci_lower .<= value)

    return new{T, M}(value, ci_lower, ci_upper, metadata)
  end
end
@forward FrequentistEstimated.value Base.getindex, Base.length, Base.size, Base.ndims, 
    Base.first, Base.last, Base.lastindex, Base.firstindex, Base.setindex!, 
    Base.eltype, Base.eachslice, Base.eachcol, Base.eachrow
ops = [:+, :-, :*, :/]
for op in ops
    eval(:(Base.$op(x::V, be::FrequentistEstimated{T, M}) where {V,T, M} = Base.$op(x, be.value)))
    eval(:(Base.$op(be::FrequentistEstimated{T, M}, args...) where {T, M} = Base.$op(be.value, args...)))
end
Base.broadcasted(f, be::FrequentistEstimated{T, M}, args...) where {T, M} = Base.broadcasted(f, be.value, args...)
Base.broadcasted(f, x::V, be::FrequentistEstimated{T, M}) where {V, T, M} = Base.broadcasted(f, x, be.value)
Base.mapslices(f, be::FrequentistEstimated{T, M}; dims=[ndims(be)-1, ndims(be)]) where {T, M} = Base.mapslices(f, be.value; dims=dims)
