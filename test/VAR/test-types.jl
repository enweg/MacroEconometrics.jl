@testset "VAR type" begin
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
    @test_throws "VAR has no known" VAR(n, p, B, b0, Σ, data)
end