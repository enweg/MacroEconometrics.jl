@doc """
    make_companion_matrix(var::VAR{FixedEstimated{T}}) where {T}
    make_companion_matrix(var::VAR{BayesianEstimated{T, M}}) where {T, M}

Make the companion matrix corresponding to a VAR model. In case of Bayesian
estimated models, the companion matrix is constructed for each draw in each
chain.
"""
function make_companion_matrix(var::VAR{FixedEstimated{T}}) where {T}
    return make_companion_matrix(var.B.value)
end
function make_companion_matrix(var::VAR{BayesianEstimated{T, M}}) where {T, M}
    # This should always be four dimensional. 
    if ndims(var.B) != 4
        throw(ErrorException("B should have 4 dimensions, but has $(ndims(var.B))"))
    end
    return mapslices(make_companion_matrix, var.B.value; dims=[1, 2])
end