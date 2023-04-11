
# TODO: write a test for this
"""
    _var_irf!(irfs::AbstractArray, B::AbstractMatrix)

Calculate the IRF of a reduced form VAR model with lag coefficient matrix `B`.
Save these results in `irfs`. `irfs` must have dimensions nvars×nvars×horizons
"""
function _var_irf!(irfs::AbstractArray, B::AbstractMatrix)
    companion_matrix = make_companion_matrix(B)
    n = size(irfs, 1)
    for i in 0:(size(irfs, 3)-1)
        irfs[:, :, i+1] = (companion_matrix^i)[1:n, 1:n]
    end
end

# TODO: write a test for this
"""
    ImpulseResponseFunction(var::VAR{E}, horizon::Int) where {E <: BayesianEstimated}

Calculate the reduced form IRFs of a [`VAR`](@ref) model. 

## Arguments

- `var::VAR{<:BayesianEstimated}`: A [`VAR`](@ref) model estimated using
  Bayesian estimation methods
- `horizon::Int`: Maximum IRF horizon. 

## Retruns

- Returns a tensor of dimension to×from×horizon×draw×chain. This can be
  flattened to a tensor of dimensions to×from×horizon×draw*chain using
  [`stack_last_dim`](@ref).
"""
function ImpulseResponseFunction(var::VAR{E}, horizon::Int) where {E <: BayesianEstimated}
    ndraws = size(var.B.value, 3)
    nchains = size(var.B.value, 4)
    irfs = Array{Float64}(undef, var.n, var.n, horizon+1, ndraws, nchains)
    for draw in 1:ndraws
        for chain in 1:nchains
            _var_irf!(view(irfs, :, :, :, draw, chain), var.B[:, :, draw, chain])
        end
    end
    irfs = BayesianEstimated(irfs, nothing)
    return ImpulseResponseFunction(Symbol.(names(var.data)), irfs, var)
end

struct CholeskyVAR <: AbstractIdentificationMethod end

# TODO: test this somehow
@doc raw"""
    StructuralImpulseResponseFunction(var::VAR{<:BayesianEstimated}, horizon::Int, identification_method::CholeskyVAR)

Estimate Structural Impulse Response Functions using a Cholesky decomposition. 

The reduced form covariance matrix is given by ``A^{-1}\Sigma A^{-1}'``. If we
normalise the structural covariance matrix ``\Sigma = I``, then the reduced form
covariance matrix is given by ``A^{-1}A^{-1}'``. Thus, if we assume that ``A``
takes a lower-triangular form, then we can estimate ``A`` and ``A^{-1}`` from
the reduced form covariance matrix by obtaining the lower-triangular cholesky
matrix of the reduced form covariance matrix. Since the covariance matrix is
positive semi-definite, this lower-triangular matrix is unique. 

## Arguments

- `var::VAR{<:BayesianEstimated}`: A [`VAR`](@ref) model estimated in a Bayesian
  way
- `horizon::Int`: Maximum horizon of IRFs. 
- `identification_method::CholeskyVAR`: Identification method.

## Returns

- Returns a tensor of dimension to×from×horizon×draw×chain. This can be
  flattened to a tensor of dimensions to×from×horizon×draw*chain using
  [`stack_last_dim`](@ref).

## References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818

"""
function StructuralImpulseResponseFunction(var::VAR{E}, horizon::Int, identification_method::CholeskyVAR) where {E <: BayesianEstimated}
    reduced_irfs = ImpulseResponseFunction(var, horizon).irfs.value
    Ainverses = mapslices(x -> cholesky(Hermitian(x)).L, var.Σ.value; dims=[1, 2])
    irfs = similar(reduced_irfs)
    for h in 0:horizon
        for draw in axes(reduced_irfs, 4)
            for chain in axes(reduced_irfs, 5)
                irfs[:, :, h+1, draw, chain] = reduced_irfs[:, :, h+1, draw, chain]*Ainverses[:, :, draw, chain]
            end
        end
    end
    return StructuralImpulseResponseFunction(
        Symbol.(names(var.data)), 
        BayesianEstimated(irfs, nothing),
        var, 
        IRFCovarianceNormalisation()
    )
end