@testset "Check regular" begin
    n = 3
    dates = Date("1996-11-19"):Quarter(1):Date("2023-03-30")
    data = TSFrame(randn(length(dates), n), dates)    
    # data should be regular
    @test MacroEconometrics._check_regularity_data(data)

    dates = vcat(dates[1:3], dates[5:end])
    data = TSFrame(randn(length(dates), n), dates)
    # data should not be regular
    @test !MacroEconometrics._check_regularity_data(data)
end


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