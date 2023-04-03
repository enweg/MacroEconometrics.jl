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