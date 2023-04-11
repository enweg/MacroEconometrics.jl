
@testset "stack Bayesian chains" begin
    n_draw = 1000
    n_chains = 3
    arr = randn(3, 3, n_draw, n_chains)
    arr_stacked = stack_last_dim(arr)
    @test all(arr_stacked[:, :, 1:n_draw] .== arr[:, :, :, 1])
    @test all(arr_stacked[:, :, n_draw+1:2*n_draw] .== arr[:, :, :, 2])
    @test all(arr_stacked[:, :, 2*n_draw+1:3*n_draw] .== arr[:, :, :, 3])

    arr = randn(3, n_draw, n_chains)
    arr_stacked = stack_last_dim(arr)
    @test all(arr_stacked[:, 1:n_draw] .== arr[:, :, 1])
    @test all(arr_stacked[:, n_draw+1:2*n_draw] .== arr[:, :, 2])
    @test all(arr_stacked[:, 2*n_draw+1:3*n_draw] .== arr[:, :, 3])

    arr = randn(n_draw, n_chains)
    arr_stacked = stack_last_dim(arr)
    @test all(arr_stacked[1:n_draw] .== arr[:, 1])
    @test all(arr_stacked[n_draw+1:2*n_draw] .== arr[:, 2])
    @test all(arr_stacked[2*n_draw+1:3*n_draw] .== arr[:, 3])
end