
"""
    OlsVAREstimator(intercept::Bool=true; confint_level=0.95)

OLS estimator for a VAR

Estimate VAR coefficients equation by equation using OLS. This uses the `lm` method in `GLS.jl`

## Fields

- `intercept::Bool`: Should an intercept be estimated? 
- `confint_level::Real`: Confidence level; Confidence intervals are symmetric and are using the asymptotic normal approximation. 

## References

- Lütkepohl, H. (2005). New introduction to multiple time series analysis. New York: Springer. (page 72 below equation 3.2.12)

"""
struct OlsVAREstimator <: AbstractVAREstimator
    intercept::Bool
    confint_level::Real
    OlsVAREstimator(intercept::Bool=true; confint_level=0.95) = new(intercept, confint_level)
end
function _ols_estimate_VAR_eq(var::VAR{E}, equation::Int, intercept::Bool=true, confint_level::Real=0.95) where {E<:FrequentistEstimated}
    variable = names(var.data)[equation]
    Y_lag = _lag_ts(var.data, 0:var.p)
    lagged_vars = filter(x -> contains(x, r"__L[1-9]+"), names(Y_lag))
    rhs_terms = term.(Symbol.(lagged_vars))
    if !intercept
        rhs_terms = vcat(ConstantTerm(-1), rhs_terms...)
    end
    f = term(Symbol("$(variable)__L0")) ~ foldl(+, rhs_terms)
    m = lm(f, Y_lag)

    coefs = coef(m)
    ci = confint(m; level=confint_level)
    ci_lower = ci[:, 1]
    ci_upper = ci[:, 2]
    resids = residuals(m)
    return coefs, ci_lower, ci_upper, resids
end
function estimate!(var::VAR{E}, method::OlsVAREstimator=OlsVAREstimator(), args...; kwargs...) where {E<:FrequentistEstimated}
    estimates = [_ols_estimate_VAR_eq(var, i, method.intercept, method.confint_level) for i in 1:var.n]
    coefs = reduce(vcat, [e[1]' for e in estimates])
    ci_lower = reduce(vcat, [e[2]' for e in estimates])
    ci_upper = reduce(vcat, [e[3]' for e in estimates])
    resids = reduce(hcat, [e[4] for e in estimates])

    meta = (; method=method)
    b0 = FrequentistEstimated(coefs[:, 1:method.intercept], ci_lower[:, 1:method.intercept], ci_upper[:, 1:method.intercept], meta)
    B = FrequentistEstimated(coefs[:, (method.intercept+1):end], ci_lower[:, (method.intercept+1):end], ci_upper[:, (method.intercept+1):end], meta)
    # We only provide a point estimate for the covariance matrix
    # corrected=true makes sure that scaling is (T-1)
    Σhat = Statistics.cov(resids; corrected=true)
    Σ = FrequentistEstimated(Σhat, meta)

    var.b0 = b0
    var.B = B
    var.Σ = Σ

    return var
end


@doc raw"""
Estimate a VAR using a Independent Normal-Wishart prior. 

Given a standard VAR, we can assume the following priors: 

```math
\begin{split}
\beta \sim N(\beta_0, V_0) \\
\Sigma^{-1} \sim Wishart(S_0^{-1}, \nu_0)
\end{split}
```

where ``\beta`` is the vector of all VAR coefficients (each equations parameters
stacked above each other) and where ``\Sigma`` is the covariance matrix of the
error terms.

This then results in the conditional posteriors

```math
\begin{split}
\beta | Y, \Sigma^{-1} &\sim N(\bar\beta, \bar{V}) \\
\bar{V} &= \left(V_0^{-1} + \sum_{t=1}^T Z_t'\Sigma^{-1}Z_t\right)^{-1} \\
\bar\beta &= \bar{V}\left(V_0^{-1}\beta_0 + \sum_{t=1}^T Z_t'\Sigma^{-1}y_t\right)
\end{split}
```

and 

```math
\Sigma^{-1}|Y, \beta &\sim Wishart(\bar{S}^{-1}, \bar\nu) \\
\bar\nu &= T + \nu_0 \\
\bar{S} &= S_0 + \sum_{t=1}^T (y_t - Z_t\beta)(y_t - Z_t\beta)'
```

where ``y_t`` is the outcome vector at time t, and where ``Z_t = I_n \otimes (1,
y_{t-1}', ..., y_{t-p}')``.

## Fields

- `prior_β::AbsatractVector{T}`: Prior mean of the coefficients. 
- `prior_V::AbstractMatrix{T}`: Prior covariance of coefficients. 
- `prior_S::AbstractMatrix{T}`: Prior scale matrix. 
- `prior_ν::T`: Prior degrees of freedom. 
- `intercept::Bool`: Estimate an intercept.

## References

- Koop, G., & Korobilis, D. (Eds.). (2010). Bayesian multivariate time series
  methods for emprirical macroeconomics. now.

"""
struct IndependentNormalWishart{T} <: AbstractVAREstimator
    prior_β::AbstractVector{T}
    prior_V::AbstractMatrix{T}
    prior_S::AbstractMatrix{T}
    prior_ν::T

    intercept::Bool
end
using Kronecker
@tiny_gibbs function VAR_independent_normal_wishart(prior_β, prior_V, prior_S, prior_ν, Z, Y)
    V_bar, beta_bar = _get_inw_beta_params_fast(Σinv, prior_β, prior_V, Z, Y)
    β ~ MultivariateNormal(beta_bar, Symmetric(V_bar))

    S_bar, nu_bar = _get_inw_Sigma_params_fast(β, prior_S, prior_ν, Z, Y)
    S_bar = cholesky(S_bar)
    Σinv ~ Wishart(nu_bar, inv(S_bar))
end
function _get_inw_beta_params_fast(Σinv, prior_β, prior_V, Z, Y)
    n = size(Σinv, 1)
    t = floor(Int, size(Y, 1) / n)
    It = Diagonal(diagm(ones(t)))
    variance = kronecker(It, Σinv)
    s = Z' * variance * Z
    V_bar = inv(inv(prior_V) + s)
    beta_bar = V_bar * (prior_V * prior_β + Z' * variance * Y)
    return V_bar, beta_bar
end
function _get_inw_beta_params(Σinv, prior_β, prior_V, Zts, yts)
    s = sum([Zt' * Σinv * Zt for Zt in Zts])
    V_bar = inv(inv(prior_V) + s)
    beta_bar = V_bar * (prior_V * prior_β + sum([Zt' * Σinv * yt for (Zt, yt) in zip(Zts, yts)]))
    return V_bar, beta_bar
end
function _get_inw_Sigma_params_fast(β, prior_S, prior_ν, Z, Y)
    n = size(prior_S, 1)
    t = floor(Int, size(Y, 1) / n)
    nu_bar = t + prior_ν
    resids = reshape(Y - Z*β, n, :)
    S_bar = prior_S + resids*resids'
    return S_bar, nu_bar
end
function _get_inw_Sigma_params(β, prior_S, prior_ν, Zts, yts)
    nu_bar = length(yts) + prior_ν
    S_bar = prior_S + sum([(yt - Zt * β) * (yt - Zt * β)' for (yt, Zt) in zip(yts, Zts)])
    return S_bar, nu_bar
end
function _construct_Z(var::VAR{E}, intercept::Bool=true) where {E<:Estimated}
    return reduce(vcat, _construct_Zts(var, intercept))
end
function _construct_Zts(var::VAR{E}, intercept::Bool=true) where {E<:Estimated}
    T = eltype(var.data[:, 1])
    lag_ts = _lag_ts(var.data, 1:var.p)
    lag_matrix = T.(Matrix(lag_ts))
    if intercept
        lag_matrix = hcat(ones(size(lag_matrix, 1)), lag_matrix)
    end
    eye = diagm(ones(var.n))
    return [kron(eye, zt') for zt in eachrow(lag_matrix)]
end
"""
For the sampling of VARs with Independent-Normal-Wishart priors ([`IndependentNormalWishart`](@ref)), the following
additional arguments are introduced: 

- `N::Int`: Number of samples
- `chains::Int=1`: Number of chains

Additionally, the following keyword arguments are introduced

- `rng::Random.AbstractRNG=Random.default_rng()`: RNG to be used
- `parallel::AbstractMCMC.AbstractMCMCEnsemble=AbstractMCMC.MCMCSeriel()`:
  Ensemble algorithm. This allows sampling using multiple threads or processes.

"""
function estimate!(var::VAR{E}, method::IndependentNormalWishart, N::Int, chains::Int=1; rng::Random.AbstractRNG=Random.default_rng(), parallel::AbstractMCMC.AbstractMCMCEnsemble=AbstractMCMC.MCMCSerial(), kwargs...) where {E<:BayesianEstimated}
    yts = eachrow(Matrix(var.data[var.p+1:end]))
    Y = reduce(vcat, yts)
    Z = _construct_Z(var, method.intercept)
    @unpack prior_β, prior_V, prior_S, prior_ν = method
    initial_values = Dict(
        :β => rand(rng, MultivariateNormal(prior_β, prior_V)),
        :Σinv => rand(rng, Wishart(prior_ν, prior_S))
    )
    sampler = VAR_independent_normal_wishart(initial_values, prior_β, prior_V, prior_S, prior_ν, Z, Y)
    samples = sample(rng, sampler, parallel, N, chains; chain_type=Dict, kwargs...)
    B = reshape(samples[:β], var.n * var.p + method.intercept, var.n, N, chains)
    B = permutedims(B, (2, 1, 3, 4))
    Σ = mapslices(inv, samples[:Σinv]; dims=[1, 2])
    var.b0 = BayesianEstimated(B[:, 1:method.intercept, :, :], method)
    var.B = BayesianEstimated(B[:, method.intercept+1:end, :, :], method)
    var.Σ = BayesianEstimated(Σ, method)
    return var
end

"""
    create_minnesota_params(
        var::VAR{E},
        λ::T,
        θ::T;
        mean_first_own_lag::T=1.0,
        mean_other_lags::T=0.0,
        include_intercept::Bool=true,
        mean_intercept::T=0.0,
        variance_intercept::T=10.0
    ) where {E<:Estimated,T}

Crate the mean and covariance of a minnesota prior on the coefficients of a VAR.
The parameter vector stacks the parameters of each equation of the VAR. Thus, if
B is the coefficient matrix of the VAR, being n×np+intercept, then the returned
coefficient vector is vec(B').

The returned values can then be used as the prior mean anf variance of the VAR coefficients in
the [`IndependentNormalWishart`](@ref) prior.

## Arguments

- `var::VAR{<:Estimated}`: a [`VAR`](@ref) model.
- `λ::<:Real`: prior standard deviation on the first own lag. 
- `θ::<:Real`: relative tightness of the prior standard deviation in other lags.
  0 < θ < 1

## Keyword Arguments

- `mean_first_own_lag::<:Real=1.0`: Mean on the first own lag. 
- `mean_other_lags::<:Real=0.0`: Mean on other lags than the first own lag. 
- `include_intercept::Bool=true`: Should an intercept be included. Intercepts
  are the first two elements in the returned mean vector. 
- `mean_intercept::<:Real=0.0`: Mean of the intercept. 
- `variance_intercept::<:Real=10.0`: Variance on the intercept. Commonly chosen
  quite large. 

## References

- Kilian, L., & Lütkepohl, H. (2017). Structural Vector Autoregressive Analysis:
  (1st ed.). Cambridge University Press. https://doi.org/10.1017/9781108164818

"""
function create_minnesota_params(
    var::VAR{E},
    λ::T,
    θ::T;
    mean_first_own_lag::T=1.0,
    mean_other_lags::T=0.0,
    include_intercept::Bool=true,
    mean_intercept::T=0.0,
    variance_intercept::T=10.0
) where {E<:Estimated,T<:Real}

    @assert 0 < θ < 1
    @assert λ > 0

    nparams = var.n^2 * var.p + var.n * include_intercept

    # Creating mean vector
    B_minnesota = Matrix(reshape(fill(mean_other_lags, nparams), var.n, var.n * var.p + include_intercept))
    if include_intercept
        B_minnesota[:, 1] .= mean_intercept
    end
    for i in 1:var.n
        B_minnesota[i, i+include_intercept] = mean_first_own_lag
    end
    b_minnesota = vec(B_minnesota')


    params_per_equation = var.n*var.p + include_intercept
    V_B_form = λ^2*θ^2*ones(T, var.n, params_per_equation)
    # sigmas of minnesota are chosen to be the unconditional variances
    # TODO: some others choose the standard deviation of univariate AR regressions
    vars = vec(Statistics.var(Matrix(var.data); dims=1))
    for i in 1:var.n
        # Multiplying each row by the variance of that variable
        V_B_form[i, :] .*= vars[i]
        # dividing by other variable sigmas
        V_B_form[i, include_intercept+1:end] ./= repeat(vars, var.p)
        # Cancelling out the θ^2 in own lags
        own_lags = (i+include_intercept):var.n:params_per_equation
        V_B_form[i, own_lags] ./= θ^2
        # dividing by the lag squared
        l2 = repeat(1:var.p; inner=var.n).^2
        V_B_form[i, include_intercept+1:end] ./= l2
        # overwriting the intercept variance
        if include_intercept
            V_B_form[i, 1] = variance_intercept
        end
    end
    V_minnesota = Diagonal(vec(V_B_form'))

    return b_minnesota, V_minnesota
end



