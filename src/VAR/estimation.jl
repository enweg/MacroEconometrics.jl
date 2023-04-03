
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
function _ols_estimate_VAR_eq(var::VAR{E}, equation::Int, intercept::Bool=true, confint_level::Real=0.95) where {E <: FrequentistEstimated}
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
function estimate!(var::VAR{E}, method::OlsVAREstimator = OlsVAREstimator(), args...; kwargs...) where {E<:FrequentistEstimated}
    estimates = [_ols_estimate_VAR_eq(var, i, method.intercept, method.confint_level) for i in 1:var.n]
    coefs = reduce(vcat, [e[1]' for e in estimates])
    ci_lower = reduce(vcat, [e[2]' for e in estimates])
    ci_upper = reduce(vcat, [e[3]' for e in estimates])
    resids = reduce(hcat, [e[4] for e in estimates])
    
    meta = (; method = method)
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
