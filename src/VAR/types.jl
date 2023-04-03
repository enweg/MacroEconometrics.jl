abstract type AbstractVectorAutoregression end
abstract type AbstractVAREstimator end

function simulate!(var::AbstractVectorAutoregression, periods::Int, args...; kwargs...) end
function estimate!(var::AbstractVectorAutoregression, method::AbstractVAREstimator, args...; kwargs...) end
function predict(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
function irf(var::AbstractVectorAutoregression, horizon, args...; kwargs...) end

function make_companion_matrix(var::AbstractVectorAutoregression) end

@doc """
    is_stable(var::AbstractVectorAutoregression)

Check whether the VAR is stable. 

Given a companionmatrix C of a VAR, the VAR is considered stable if and only if
all eigenvalues of C are less than unity in absolute value.  

# Arguments

- `var::Estimated`: The VAR model

# References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818


"""
function is_stable(var::AbstractVectorAutoregression) end

@doc raw"""
    VAR(n::Int, p::Int, B::E, b0::E, Σ::E, data::TSFrame) where {E<:Estimated}
    VAR(data::TSFrame, p; type::Type{T}=BayesianEstimated)
    VAR(n::Int, p::Int, B::FixedEstimated{T}, b0::FixedEstimated{T}, Σ::FixedEstimated{T}) where {T}

Every VAR model is of the form

```math
y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + \varepsilon_t
```

where ``B_i`` is ``n \times n`` and ``b_0`` is an ``n \times 1`` vector. For the
disturbance terms, ``\varepsilon_t`` we generally assume ``\varepsilon_t \sim
N(0, \Sigma)``. This can therefore be equivelently written as 

```math
y_t = b_0 + B \tilde{y}_{t-1} + \varepsilon_t
```

where ``B`` is now ``n \times np`` and ``\tilde{y}_{t-1}=(y_{t-1}', ...,
y_{t-p}')'``.

## General Information

- VAR models are only approprite for regularly spaced observations. We therefore
  check whether this is the case here

## Fields

- `n::Int`: Number of variables in the VAR.
- `p::Int`: Numbe of lags; See the equations above.
- `B::Union{Nothing, E<:Estimated}`: Lag coefficient matrix; See the equations above; Should be
  a subtype of `Estimated`. 
- `b0::Union{Nothing, E<:Estimated}`: Intercept vector; See equations above.
- `Σ::Union{Nothing, E<:Estimated}`: Covariance matrix of disturbance terms; See equation above.
- `data::Union{Nothing, TSFrame}`: A time series data frame containing the data used for the
  VAR. Observations should be regularly spaced.

"""
mutable struct VAR{E<:Estimated}
    n::Int
    p::Int
    B::Union{Nothing, E}
    b0::Union{Nothing, E}
    Σ::Union{Nothing, E}

    data::Union{Nothing, TSFrame}

    function VAR(n::Int, p::Int, B::E, b0::E, Σ::E, data::TSFrame) where {E<:Estimated}
        # check if data is regular. VAR models are only appropriate for regularly spaced data
        regular = _check_regularity_data(data)
        if !regular
            throw(ErrorException("VAR has no known frequency"))
        end
        return new{E}(n, p, B, b0, Σ, data)
    end
    function VAR(data::TSFrame, p; type::Type{T}=BayesianEstimated) where {T<:Estimated}
        regular = _check_regularity_data(data)
        if !regular
            throw(ErrorException("VAR has no known frequency"))
        end
        n = nrow(data)
        # estimates do not yet exist, so set to zero
        B = b0 = Σ = nothing
        return new{type}(n, p, B, b0, Σ, data)
    end
    function VAR(n::Int, p::Int, B::FixedEstimated{T}, b0::FixedEstimated{T}, Σ::FixedEstimated{T}) where {T}
        return new{FixedEstimated{T}}(n, p, B, b0, Σ, nothing)
    end
    function VAR(n::Int, p::Int, B::BayesianEstimated{T, M}, b0::BayesianEstimated{T, M}, Σ::BayesianEstimated{T, M}) where {T, M}
        return new{BayesianEstimated{T, M}}(n, p, B, b0, Σ, nothing)
    end
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


"""

Simulate a Gaussian VAR

# Arguments

- `B::AbstractMatrix{T}`: Lag matrix of dimensions n×(n*p)
- `b0::AbstractVector{T}`: Vector of intercepts of dimension n×1
- `Σ::AbstractMatrix{T}`: Covariance matrix of errors of dimension n×n
- `initial::AbstractVector{T}`: Initial values of dimension (n*p)×1; So it is
  the initial value of the companion form.
- `periods::Int`: Number of periods to simulate for. 

# Keyword arguments

- `burnin::Int=100`: Number of periods to use as burnin. This will reduce the
  dependency on the initial conditions. 
- `rng::Random.AbstractRNG=Random.MersenneTwister()`: Random number generator. 

# Returns

- Returns a matrix `Y` of dimensions n×periods; Thus, rows are the variables,
  columns are the time points. 
"""
function _simulate_VAR(
    B::AbstractMatrix{T}, 
    b0::AbstractVector{T}, 
    Σ::AbstractMatrix{T},
    initial::AbstractVector{T}, 
    periods::Int; 
    burnin::Int = 100, 
    rng::Random.AbstractRNG=Random.MersenneTwister()
) where {T} 

    n = size(B, 1)
    p = Int(size(B, 2)/n)
    sim_b0 = vcat(b0, zeros(T, n*(p-1)))
    companion = make_companion_matrix(B)
    error_dist = MultivariateNormal(zeros(T, n), Σ)
    e = rand(rng, error_dist, periods+burnin)
    sim_e = vcat(e, zeros(T, n*(p-1), periods+burnin))

    Y = Matrix{T}(undef, n*p, periods+burnin+1)
    Y[:, 1] = initial
    for t in 2:(periods+burnin+1)
        Y[:, t] = sim_b0 + companion*Y[:, t-1] + sim_e[:, t-1]
    end
    Y = Y[1:n, end-periods+1:end]
end

"""

Simulate a Gaussian VAR. 

# Arguments

- `var::VAR{FixedEstimated{T}}`: A VAR model having `FixedEstimated`
  coefficients.
- `periods::Int`: Number of periods to simulate the VAR for

# Keyword Arguments

- `burnin::Int=100`: Number of periods to use as burnin; The longe the burnin,
  the less the dependence on the initial values. 
- `initial::AbstractVector{T}=zeros(T, var.n*var.p)`: Initial values. Should be
  in companion form, so a vector of length var.n×var.p
- `start_date::Date=Dates.today()`: The simulated data will be stored in a
  TSFrame including dates. Thus, we need a start date. 
- `frequency::DatePeriod=Dates.Quarter(1)`: Frequency of data. 
- `rng::Random.AbstractRNG=Random.MersenneTwister()`: Random Number Generator

# Returns

- Returns the same VAR model. The `data` field in the VAR model is overwritten
  witht he new simulated data.
"""
function simulate!(
    var::VAR{FixedEstimated{T}}, periods::Int; 
    burnin::Int=100, initial::AbstractVector{T}=zeros(T, var.n*var.p), 
    start_date::Date=Dates.today(), frequency::DatePeriod=Dates.Quarter(1), 
    rng::Random.AbstractRNG=Random.MersenneTwister()
) where {T}

    Y = _simulate_VAR(
        var.B.value, 
        var.b0.value, 
        var.Σ.value, 
        initial,
        periods; 
        burnin=burnin, 
        rng=rng
    )
    data = TSFrame(Y', start_date:frequency:(start_date+(periods-1)*frequency))
    var.data = data
    return var
end

    