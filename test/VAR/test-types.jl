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