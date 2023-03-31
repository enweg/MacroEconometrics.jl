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

@testset "Make Companion Matrix" begin
    # Given a VAR(2) with two variables, we consider the lag matrix
    # B = [0.5 0.3 0.1 0.5; 0.1 0.9 -0.3 0]
    # Which should have a companion matrix of the form 
    # C = [0.5 0.3 0.1 0.5; 0.1 0.9 -0.3 0; 1 0 0 0; 0 1 0 0]
    B = [0.5 0.3 0.1 0.5; 0.1 0.9 -0.3 0]
    C = make_companion_matrix(B)
    @test all(C .== [0.5 0.3 0.1 0.5; 0.1 0.9 -0.3 0; 1 0 0 0; 0 1 0 0]) 
end