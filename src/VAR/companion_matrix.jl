@doc raw"""
    make_companion(B::AbstractMatrix{T}) where {T}
    
Create the VAR companion matrix.

Given a VAR of the form

```math
y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + \varepsilon_t
```

The companion matrix is given by 

```math
C = \begin{bmatrix}
    B_1 & B_2 & \dots & B_{p-1} & B_p \\
    I_n & O & \dots & O & O \\
    O & I_n & & O & O \\
    \vdots & & \ddots & \vdots & \vdots \\ 
    O & O & \dots & I_n & O
\end{bmatrix}
```

Thus, ``B`` is a ``np\times np`` matrix. 

## Arguments

-`B::AbstractMatrix{T}`: Lag matrix in the form required for a `VAR` model. See
    the documentation of `VAR`.


## References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818


"""
function make_companion_matrix(B::AbstractMatrix{T}) where {T}
    n = Int(size(B, 1))
    p = Int(size(B, 2) / n)
    ident = diagm(fill(T(1), n*(p-1)))
    companion_lower = hcat(ident, zeros(n*(p-1), n))
    companion = vcat(B, companion_lower)
    return companion
end

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