@testset "VAR type all information" begin
    n = 3
    p = 2
    B = BayesianEstimated(randn(n, n*p), nothing)
    b0 = BayesianEstimated(randn(n, 1), nothing)
    Σ = BayesianEstimated(randn(n, n), nothing)

    dates = Date("1996-11-19"):Quarter(1):Date("2023-03-30")
    data = TSFrame(randn(length(dates), n), dates)
    model = VAR(n, p, B, b0, Σ, data)

    # All of the above should run without problems
    # If we remove a row, the data will no-longer be regular 
    # and hence the construction of the model should fail

    dates = vcat(dates[1:3], dates[5:end])
    data = TSFrame(randn(length(dates), n), dates)
    @test_throws ErrorException VAR(n, p, B, b0, Σ, data)
end

@testset "VAR type only data and lags" begin
    n = 3
    p = 2
    dates = Date("1996-11-19"):Quarter(1):Date("2023-03-30")
    data = TSFrame(randn(length(dates), n), dates)

    # model should have BayesianEstimated as parameter
    model = VAR(data, p)
    @test typeof(model) <: VAR{BayesianEstimated}

    # model should have FixedEstimated as parameter
    model = VAR(data, p; type=FixedEstimated)
    @test typeof(model) <: VAR{FixedEstimated}

    # model quantities should be overwritable
    B = FixedEstimated(randn(n, n*p))
    model.B = B
    @test model.B == B

    # error should be thrown when data is not regular
    dates = vcat(dates[1:3], dates[5:end])
    data = TSFrame(randn(length(dates), n), dates)
    @test_throws ErrorException VAR(data, p)
end

@testset "VAR no data" begin
    n = 5
    p = 10
    B = FixedEstimated(randn(n, n*p))    
    b0 = FixedEstimated(randn(n))
    Σ = FixedEstimated(randn(n, n))
    # only used to check if constructor works
    # there is nothing here to check yet
    VAR(n, p, B, b0, Σ)
end

# TODO: write a better test
@testset "VAR Companion Matrix" begin
    n = 3
    p = 2
    draws = 3
    chains = 2

    # FixedEstimated
    B = Array(reshape(1:(n*(n*p)), n, n*p))
    B = Float64.(B)
    B_fixed = FixedEstimated(B)
    b0_fixed = FixedEstimated(randn(n))
    Σ_fixed = FixedEstimated(randn(n, n))
    model_fixed = VAR(n, p, B_fixed, b0_fixed, Σ_fixed)
    C_fixed = make_companion_matrix(model_fixed)
    C_compare = make_companion_matrix(B)
    # We have another test to check whether 
    # companion matrices given a matrix are correct. So here we only need 
    # to check if these two match
    @test all(C_fixed .== C_compare)

    # BayesianEstimated
    B = Array(reshape(1:(n*(n*p)*draws*chains), n, n*p, draws, chains))
    B = Float64.(B)
    B_bayes = BayesianEstimated(B, nothing)
    b0_bayes = BayesianEstimated(randn(n, draws, chains), nothing)
    Σ_bayes = BayesianEstimated(randn(n, n, draws, chains), nothing)
    model_bayes = VAR(n, p, B_bayes, b0_bayes, Σ_bayes)
    # we know that companion matrix is correct given a matrix (we have a test)
    # for that.  Thus, we can simply compare these two here.
    C_bayes = make_companion_matrix(model_bayes)
    for (i, j) in zip(1:size(B, 3), 1:size(B, 4))
        C_compare = make_companion_matrix(B[:, :, i, j])
        @test all(C_compare .== C_bayes[:, :, i, j])
    end
end

@testset "VAR is_stable" begin
    n = 2
    p = 1
    B = [0.5 0.3; 0.2 0.5]
    b0 = [0.0, 0.0]
    Σ = [1.0 0.0; 0.0 1.0]

    # FixedEstimated
    B_fixed = FixedEstimated(B)
    b0_fixed = FixedEstimated(b0)
    Σ_fixed = FixedEstimated(Σ)
    model_fixed = VAR(n, p, B_fixed, b0_fixed, Σ_fixed)
    # Should be stable
    @test is_stable(model_fixed)
    # Should no-longer be stable
    model_fixed.B[1, 1] = 10.0
    @test !is_stable(model_fixed)

    # BayesianEstimated
    B = [0.5 0.3; 0.2 0.5]
    b0 = [0.0, 0.0]
    Σ = [1.0 0.0; 0.0 1.0]
    B_bayes = cat([B for _ in 1:3]...; dims=3)
    B_bayes = BayesianEstimated(reshape(B_bayes, size(B_bayes)..., 1), nothing)
    b0_bayes = BayesianEstimated(randn(n, 3, 1), nothing)
    Σ_bayes = BayesianEstimated(randn(n, n, 3, 1), nothing)
    model_bayes = VAR(n, p, B_bayes, b0_bayes, Σ_bayes)
    # should be stable since all slices are stable (they are the same)
    @test is_stable(model_bayes)
    # Should not be stable
    B_bayes[1, 1, 1, 1] = 10.0
    @test !is_stable(model_bayes)
end