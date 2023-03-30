abstract type AbstractVectorAutoregression end
abstract type AbstractVAREstimator end

function simulate!(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
function estimate!(var::AbstractVectorAutoregression, method::AbstractVAREstimator, args...; kwargs...) end
function predict(var::AbstractVectorAutoregression, periods, args...; kwargs...) end
function irf(var::AbstractVectorAutoregression, horizon, args...; kwargs...) end
function make_companion_matrix(var::AbstractVectorAutoregression) end
function is_stable(var::AbstractVectorAutoregression) end

const FREQUENCIES = [:day, :month, :quarter, :year]

@doc raw"""
    VAR model

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
- `B::E<:Estimated`: Lag coefficient matrix; See the equations above; Should be
  a subtype of `Estimated`. 
- `b0::E<:Estimated`: Intercept vector; See equations above.
- `Σ::E<:Estimated`: Covariance matrix of disturbance terms; See equation above.
- `data::TSFrame`: A time series data frame containing the data used for the
  VAR. Observations should be regularly spaced.

"""
mutable struct VAR{E<:Estimated}
    n::Int
    p::Int
    B::E
    b0::E 
    Σ::E

    data::TSFrame

    function VAR(n::Int, p::Int, B::E, b0::E, Σ::E, data::TSFrame; verbose = true) where {E<:Estimated}
        # check if data is regular. VAR models are only appropriate for regularly spaced data
        for freq in FREQUENCIES
            if isregular(data, freq)
                verbose ? @info("VAR is specified in frequency: $freq") : nothing 
                return new{E}(n, p, B, b0, Σ, data)
            end
        end
        throw(ErrorException("VAR has no known frequency"))
    end
end
