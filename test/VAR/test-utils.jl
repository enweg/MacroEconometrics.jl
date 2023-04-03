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

@testset "_lag_ts" begin
    x1 = collect(1:5)
    x2 = collect(11:15)
    dates = Dates.today():Day(1):(Dates.today()+(length(x1)-1)*Day(1))
    ts = TSFrame([x1 x2], dates)
    lag_ts = MacroEconometrics._lag_ts(ts, 0:2)

    expected_ts_data = [
        3 13 2 12 1 11; 
        4 14 3 13 2 12; 
        5 15 4 14 3 13
    ]
    @test all(Matrix(lag_ts) .== expected_ts_data)
    expected_ts_dates = [
        Dates.today()+2*Day(1), 
        Dates.today()+3*Day(1), 
        Dates.today()+4*Day(1)
    ]
    @test all(lag_ts.coredata[!, :Index] .== expected_ts_dates)
    expected_ts_names = [
        "x1__L0", 
        "x2__L0", 
        "x1__L1", 
        "x2__L1", 
        "x1__L2", 
        "x2__L2"
    ]
    @test all(names(lag_ts) .== expected_ts_names)
end