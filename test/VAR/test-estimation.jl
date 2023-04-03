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