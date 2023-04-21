@testset "VAR Frequentist Estimation" begin
    # Simulating data
    rng = StableRNG(123)
    n = 3
    p = 3
    B = FixedEstimated(0.2*randn(rng, n, n*p))
    b0 = FixedEstimated(rand(rng, n))
    Σ = FixedEstimated(rand(rng, Wishart(n+1, diagm(ones(n)))))
    model_sim = VAR(n, p, B, b0, Σ)

    inside_ci = 0
    comparisons = 0
    for i in 1:100
        simulate!(model_sim, 10_000; rng = rng)

        model_freq = VAR(model_sim.data, p; type=FrequentistEstimated)
        method = OlsVAREstimator(true; confint_level=0.95)
        estimate!(model_freq, method)
        
        inside_ci += sum((model_sim.B.value .<= model_freq.B.ci_upper) .&& (model_sim.B.value .>= model_freq.B.ci_lower))
        comparisons += length(model_freq.B)
    end

    # Technically this should be >= 0.95 if the Normal was exact, but it only holds
    # asymptotically
    @test inside_ci/comparisons > 0.9
end

@testset "VAR Independent Normal Wishart" begin

    # Testing the construction of Zts 
    data = reshape(1:10, 5, :)
    dates = Dates.Date("1996/11/19", dateformat"yyyy/mm/dd")
    dates = dates:Day(1):(dates + 4*Day(1))
    ts = TSFrame(data, dates)
    model = VAR(ts, 2)

    Z1 = [2 7 1 6 0 0 0 0; 0 0 0 0 2 7 1 6]
    Z2 = [3 8 2 7 0 0 0 0; 0 0 0 0 3 8 2 7]
    Z3 = [4 9 3 8 0 0 0 0; 0 0 0 0 4 9 3 8]
    Zts = MacroEconometrics._construct_Zts(model, false)
    @test all(Z1 .== Zts[1])
    @test all(Z2 .== Zts[2])
    @test all(Z3 .== Zts[3])

    # Testing the construction of Zts
    Z1 = [1 2 7 1 6 0 0 0 0 0; 0 0 0 0 0 1 2 7 1 6]
    Z2 = [1 3 8 2 7 0 0 0 0 0; 0 0 0 0 0 1 3 8 2 7]
    Z3 = [1 4 9 3 8 0 0 0 0 0; 0 0 0 0 0 1 4 9 3 8]
    Zts = MacroEconometrics._construct_Zts(model, true)
    @test all(Z1 .== Zts[1])
    @test all(Z2 .== Zts[2])
    @test all(Z3 .== Zts[3])

    # Testing the construction of the posterior distribution parameters
    intercept = true
    nparams = model.n^2*model.p + intercept*model.n
    prior_β = zeros(nparams)
    prior_V = diagm(ones(nparams))
    prior_nu = 10
    prior_S = diagm(ones(model.n))

    initial_Σinv = diagm(ones(model.n))
    initial_β = zeros(nparams)

    yts = eachrow(Matrix(model.data[model.p+1:end]))
    V_bar, beta_bar = MacroEconometrics._get_inw_beta_params(initial_Σinv, prior_β, prior_V, Zts, yts)
    yts = collect(yts)
    V_bar_manual = inv(I + Z1'*Z1 + Z2'*Z2 + Z3'*Z3)
    b_bar_manual = V_bar_manual*(Z1'*yts[1] + Z2'*yts[2] + Z3'*yts[3])
    @test all(V_bar .== V_bar_manual)
    @test all(beta_bar .== b_bar_manual)

    yts = eachrow(Matrix(model.data[model.p+1:end]))
    S_bar, nu_bar = MacroEconometrics._get_inw_Sigma_params(initial_β, prior_S, prior_nu, Zts, yts)
    yts = collect(yts)
    nu_bar_manual = 3 + prior_nu
    S_bar_manual = prior_S + (yts[1]-Z1*initial_β)*(yts[1]-Z1*initial_β)' + (yts[2]-Z2*initial_β)*(yts[2]-Z2*initial_β)' + (yts[3]-Z3*initial_β)*(yts[3]-Z3*initial_β)'
    @test nu_bar == nu_bar_manual
    @test all(S_bar .== S_bar_manual)

    # All that is left is to test the actual sampling. Since the sampling
    # procedure is stochastic it is difficult to test it. Everything that is
    # non-stoachstic in the procedure has been tested above though, and thus, we
    # will here only test whether the sampling is actually running. We will not
    # compare the values to anything. 
    rng = StableRNG(123)
    n = 3
    p = 3
    B = FixedEstimated(0.2*randn(rng, n, n*p))
    b0 = FixedEstimated(rand(rng, n))
    Σ = FixedEstimated(rand(rng, Wishart(n+1, diagm(ones(n)))))
    model_sim = VAR(n, p, B, b0, Σ)    
    simulate!(model_sim, 500; rng = rng)

    intercept = true
    n_params = model_sim.n^2*model_sim.p + model_sim.n
    method = IndependentNormalWishart(zeros(n_params), 10*diagm(ones(n_params)), diagm(ones(model_sim.n)), model_sim.n + 1.0, intercept)
    model_bayes = VAR(model_sim.data, p; type=BayesianEstimated)
    model_bayes = estimate!(model_bayes, method, 1000; rng = rng)
end

@testset "Minnesota paramters intercept=$(intercept)" for intercept in [true, false]
    n = 2
    p = 3
    # intercept = true
    λ = 1.5
    θ = 0.2
    mean_first_own_lag=1.0
    mean_other_lags=0.0
    mean_intercept=0.0
    variance_intercept=10.0

    rng = StableRNG(123)
    data = randn(rng, 100, n)
    dates = Dates.Date("1996/11/19", dateformat"yyyy/mm/dd")
    dates = dates:Day(1):(dates + Day(99))
    ts = TSFrame(data, dates)
    sigmas = sqrt.(vec(Statistics.var(data; dims=1)))
    model = VAR(ts, p; type=BayesianEstimated)

    b_minnesota, V_minnesota = create_minnesota_params(
        model,
        λ,
        θ;
        mean_first_own_lag=mean_first_own_lag,
        mean_other_lags=mean_other_lags,
        include_intercept=intercept,
        mean_intercept=mean_intercept,
        variance_intercept=variance_intercept
    )

    b_manual = [
        mean_intercept, mean_first_own_lag, mean_other_lags, mean_other_lags, mean_other_lags, mean_other_lags, mean_other_lags, 
        mean_intercept, mean_other_lags, mean_first_own_lag, mean_other_lags, mean_other_lags, mean_other_lags, mean_other_lags
    ]
    V_diag_manual = [
        variance_intercept, 
        (λ/1)^2, 
        (λ*θ*sigmas[1])^2/(1*sigmas[2])^2, 
        (λ/2)^2, 
        (λ*θ*sigmas[1])^2/(2*sigmas[2])^2, 
        (λ/3)^2, 
        (λ*θ*sigmas[1])^2/(3*sigmas[2])^2, 
        variance_intercept, 
        (λ*θ*sigmas[2])^2/(1*sigmas[1])^2, 
        (λ/1)^2, 
        (λ*θ*sigmas[2])^2/(2*sigmas[1])^2, 
        (λ/2)^2, 
        (λ*θ*sigmas[2])^2/(3*sigmas[1])^2, 
        (λ/3)^2
    ]
    if !intercept
        selector = vcat(2:7, 9:14)
        b_manual = b_manual[selector]
        V_diag_manual = V_diag_manual[selector]
    end
    V_manual = Diagonal(V_diag_manual)

    @test all(b_minnesota .== b_manual)
    @test all(V_minnesota .≈ V_manual)
end