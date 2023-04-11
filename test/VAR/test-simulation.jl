@testset "VAR{FixedEstimated} simulation" begin
    # We will run this twice
    # If the first one does not indicate any problems, then we will return 
    # a pass, otherwise we try again. If the test fails again then this is
    # likely due to an actual implementation problem. 
    rng = StableRNG(123)
    pass = false
    for _ in 1:2
        n = 2
        p = 2
        B = FixedEstimated(0.5randn(rng, n, n*p))
        b0 = FixedEstimated(randn(rng, n))
        Σ = FixedEstimated(rand(rng, Wishart(n, diagm(ones(n)))))
        model = VAR(n, p, B, b0, Σ)

        simulate!(model, 1_000_000; rng = rng)

        lag_ts = MacroEconometrics._lag_ts(model.data, 1:2)
        ts = join(model.data, lag_ts; jointype=:JoinInner)

        # first equation
        eq1 = lm(@formula(x1 ~ 1 + x1__L1 + x2__L1 + x1__L2 + x2__L2), ts.coredata)
        eq2 = lm(@formula(x2 ~ 1 + x1__L1 + x2__L1 + x1__L2 + x2__L2), ts.coredata)
        deltas_eq1 = (vcat(b0[1], B[1,:]) .- coef(eq1)) ./ stderror(eq1)
        deltas_eq2 = (vcat(b0[2], B[2,:]) .- coef(eq2)) ./ stderror(eq2)
        p_values_eq1 = 2*cdf.(Normal(0, 1), -abs.(deltas_eq1))
        p_values_eq2 = 2*cdf.(Normal(0, 1), -abs.(deltas_eq2))
        if all(p_values_eq1 .> 0.05) && all(p_values_eq2 .> 0.05)
            pass = true
            @info "Smallest p-value is $(minimum(p_values_eq1)) and $(minimum(p_values_eq2))"
            @info "Maxumum absolute difference is $(maximum(abs.(deltas_eq1 .* stderror(eq1)))) and $(maximum(abs.(deltas_eq2 .* stderror(eq2))))"
            break
        end
    end
    @test pass
end

@testset "SVAR{FixedEstimated} simulation" begin
    rng = StableRNG(123)
    n = 3
    p = 2
    A = 0.4*rand(rng, n, n)
    A = FixedEstimated(Matrix(UnitLowerTriangular(A)))
    B = FixedEstimated(0.1*randn(rng, n, n*p))
    b0 = FixedEstimated(randn(rng, n))
    Σ = FixedEstimated(diagm(ones(n)))
    model = SVAR(n, p, A, B, b0, Σ)

    simulate!(model, 1_000_000; rng=rng)

    lag_ts = MacroEconometrics._lag_ts(model.data, 1:p)
    ts = join(model.data, lag_ts; jointype=:JoinInner)

    # Since A is unit lower triangular, the first structural equation should
    # only depend on the lagged coefficients
    eq1 = lm(@formula(x1 ~ 1 + x1__L1 + x2__L1 + x3__L1 + x1__L2 + x2__L2 + x3__L2), ts.coredata)
    deltas_eq1 = (vcat(b0[1], B[1, :]) .- coef(eq1)) ./ stderror(eq1)
    p_values_eq1 = 2*cdf.(Normal(0, 1), -abs.(deltas_eq1))
    @test all(p_values_eq1 .> 0.05)
    @info "Smallest p-value is $(minimum(p_values_eq1))"
end