function is_stable(var::VAR{FixedEstimated{T}}) where {T}
    C = make_companion_matrix(var)
    return maximum(abs.(eigen(C).values)) < 1.0
end

@doc """
    is_stable(var::VAR{BayesianEstimated{T, M}}) where {T, M}

In case of Bayesian estimated VARs, the model is considered stable if and only
if the VAR is stable for all parameter draws. As such, a rather strong view is
taken here. 
"""
function is_stable(var::VAR{BayesianEstimated{T, M}}) where {T, M}
    C = make_companion_matrix(var)
    stable_tensor = mapslices(x -> maximum(abs.(eigen(x).values)) < 1.0, C; dims=[1, 2])
    return all(stable_tensor)
end