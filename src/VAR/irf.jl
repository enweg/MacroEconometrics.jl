
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

