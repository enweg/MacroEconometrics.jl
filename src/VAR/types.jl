abstract type AbstractVectorAutoregression <: MacroEconometricModel end
abstract type AbstractVAREstimator end

function simulate!(var::AbstractVectorAutoregression, periods::Int, args...; kwargs...) end

"""
    estimate!(var::AbstractVectorAutoregression, method::AbstractVAREstimator, args...; kwargs...)

Estimate a VAR using `method`.

## Methods

Depending on the type of VAR, the methods can differ. Methods can also differ
with regard to which statistical paradigm they are following. Some methods will
estimate the VAR using Frequentist methods, while others will estimate the VAR
using Bayesian methods. 

### Standard Gaussian VAR: [`VAR`](@ref)

This is represented by a standard [`VAR`](@ref) type. Methods currently include

- [`OlsVAREstimator`](@ref)
- [`IndependentNormalWishart`](@ref)

"""
function estimate!(var::AbstractVectorAutoregression, method::AbstractVAREstimator, args...; kwargs...) end
function predict(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
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
mutable struct VAR{E<:Estimated}<:AbstractVectorAutoregression
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
    function VAR(data::TSFrame, p::Int; type::Type{T}=BayesianEstimated) where {T<:Estimated}
        regular = _check_regularity_data(data)
        if !regular
            throw(ErrorException("VAR has no known frequency"))
        end
        n = ncol(data)
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

@doc raw"""
    SVAR(n::Int, p::Int, B::E, b0::E, Σ::E, data::TSFrame) where {E<:Estimated}
    SVAR(data::TSFrame, p; type::Type{T}=BayesianEstimated)
    SVAR(n::Int, p::Int, B::FixedEstimated{T}, b0::FixedEstimated{T}, Σ::FixedEstimated{T}) where {T}

Every SVAR model is of the form

```math
A y_t = b_0 + B_1 y_{t-1} + ... + B_p y_{t-p} + \varepsilon_t
```

where ``A`` is a ``n \times n`` contemporanous matrix, ``B_i`` is ``n \times n``
and ``b_0`` is an ``n \times 1`` vector. For the disturbance terms,
``\varepsilon_t`` we generally assume ``\varepsilon_t \sim N(0, \Sigma)``. This
can therefore be equivelently written as 

```math
A y_t = b_0 + B \tilde{y}_{t-1} + \varepsilon_t
```

where ``B`` is now ``n \times np`` and ``\tilde{y}_{t-1}=(y_{t-1}', ...,
y_{t-p}')'``.

## General Information

- SVAR models are only approprite for regularly spaced observations. We therefore
  check whether this is the case here

## Fields

- `n::Int`: Number of variables in the VAR.
- `p::Int`: Numbe of lags; See the equations above.
- `A::Union{Nothing, E<:Estimated}`: Contemporanous matrix; See the equations above;
- `B::Union{Nothing, E<:Estimated}`: Lag coefficient matrix; See the equations above; Should be
  a subtype of `Estimated`. 
- `b0::Union{Nothing, E<:Estimated}`: Intercept vector; See equations above.
- `Σ::Union{Nothing, E<:Estimated}`: Covariance matrix of disturbance terms; See equation above.
- `data::Union{Nothing, TSFrame}`: A time series data frame containing the data used for the
  VAR. Observations should be regularly spaced.

"""
mutable struct SVAR{E<:Estimated}<:AbstractVectorAutoregression
    n::Int 
    p::Int 
    A::Union{Nothing, E}
    B::Union{Nothing, E}
    b0::Union{Nothing, E}
    Σ::Union{Nothing, E}

    data::Union{Nothing, TSFrame}

    function SVAR(n::Int, p::Int, A::E, B::E, b0::E, Σ::E, data::TSFrame) where {E<:Estimated}
        regular = _check_regularity_data(data)
        if !regular
            throw(ErrorException("SVAR has no known frequency"))
        end
        return new{E}(n, p, A, B, b0, Σ, data) 
    end
    function SVAR(data::TSFrame, p::Int; type::Type{T}=BayesianEstimated) where {T<:Estimated}
        regular = _check_regularity_data(data)
        if !regular
            throw(ErrorException("SVAR has no known frequency"))
        end
        n = ncol(data)
        # estimates do not yet exist, so set to zero
        A = B = b0 = Σ = nothing
        return new{type}(n, p, A, B, b0, Σ, data)
    end
    function SVAR(n::Int, p::Int, A::FixedEstimated{T}, B::FixedEstimated{T}, b0::FixedEstimated{T}, Σ::FixedEstimated{T}) where {T}
        return new{FixedEstimated{T}}(n, p, A, B, b0, Σ, nothing)
    end
    function SVAR(n::Int, p::Int, A::BayesianEstimated{T, M}, B::BayesianEstimated{T, M}, b0::BayesianEstimated{T, M}, Σ::BayesianEstimated{T, M}) where {T, M}
        return new{BayesianEstimated{T, M}}(n, p, A, B, b0, Σ, nothing)
    end
end