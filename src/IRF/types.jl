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

struct StructuralImpulseResponseFunction{E<:Estimated,M<:MacroEconometricModel,N<:AbstractIRFNormalisation,I<:AbstractIdentificationMethod} <: AbstractImpulseResponseFunction
    variables::Vector{Symbol}
    irfs::E
    model::M
    normalisation::N
    identification::I
end

