"""
- Should have a field called `variables::Vector{Symbol}` containing the
  variables 
- Should have a field called `irfs::E<:Estimated` containing the IRF estimates.
  Dimensions should be to×from×horizon
"""
abstract type AbstractImpulseResponseFunction end
abstract type AbstractIRFNormalisation end
"""
Variance of structural shocks is normalised to unity.
"""
struct IRFCovarianceNormalisation <: AbstractIRFNormalisation end
"""
Impact of structural shock on variable is normalised to unity. That is, each
structural shock has a variable which it increases by one unit if the shock
increases by one unit. 
"""
struct IRFImpactNormalisation <: AbstractIRFNormalisation end


struct ImpulseResponseFunction{E<:Estimated,M<:MacroEconometricModel} <: AbstractImpulseResponseFunction
    variables::Vector{Symbol}
    irfs::E
    model::M
end

struct StructuralImpulseResponseFunction{E<:Estimated,M<:MacroEconometricModel,N<:AbstractIRFNormalisation} <: AbstractImpulseResponseFunction
    variables::Vector{Symbol}
    irfs::E
    model::M
    normalisation::N
end

@doc raw"""

Move from covariance normalisation to impact normalisation. 

Given a SVAR of the form 

```math
A y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + \Sigma^{1/2}\varepsilon
```

that has been covariance normalised, such that ``\Sigma^{1/2}=I``, we can move
to a SVAR that is impact normalised - the diagonal of ``A`` consists of only
ones, by premultiplying all matriced by ``N = Diag(A)^{-1}`` where ``Diag(X)``
denotes the matrix of the diagonal of ``X``. The covariance matrix of the
structural errors is then given by ``NN'``. 

## Arguments 

- `sirfs::<:StructuralImpulseResponseFunction` obtained from a [`VAR`](@ref) or
  [`SVAR`](@ref) model and using any identification method relying on covariance
  normalisation. 
"""
function to_impact_normalisation(sirfs::S) where {S <: StructuralImpulseResponseFunction} end

# TODO: test me
function to_impact_normalisation(sirfs::StructuralImpulseResponseFunction{E, M, N}) where {E<:BayesianEstimated, M<:AbstractVectorAutoregression, N<:AbstractIRFNormalisation}
  irfs = copy(sirfs.irfs.value)
  horizons = size(irfs, 3)-1
  for chain in axes(irfs, 5)
    for draw in axes(irfs, 4)
      Nn = diagm(1 ./ diag(irfs[:, :, 1, draw, chain]))
      for h in 0:horizons
        irfs[:,:,h+1,draw,chain] = irfs[:,:,h+1,draw,chain] * Nn
      end
    end
  end
  return StructuralImpulseResponseFunction(
    sirfs.variables, 
    BayesianEstimated(irfs, nothing), 
    sirfs.model, 
    IRFImpactNormalisation()
  )
end
