@testset "BayesianEstimated" begin
    be_1d = BayesianEstimated(rand(1), nothing)
    be_2d = BayesianEstimated(rand(3, 3), nothing)
    be_3d = BayesianEstimated(rand(3, 3, 3), nothing)

    # getindex
    @test be_1d[1] == be_1d.value[1]
    @test be_2d[1] == be_2d.value[1]
    @test be_2d[1, 3] == be_2d.value[1, 3]
    @test all(be_2d[1:3, :] .== be_2d.value[1:3, :])
    @test all(be_2d[:, 1:3] .== be_2d.value[:, 1:3])
    @test all(be_3d[:, 1, 1:2] .== be_3d.value[:, 1, 1:2])

    # length
    @test length(be_1d) == length(be_1d.value)
    @test length(be_2d) == length(be_2d.value)
    @test length(be_3d) == length(be_3d.value)

    # size
    @test all(size(be_1d) .== size(be_1d.value))
    @test all(size(be_2d) .== size(be_2d.value))
    @test all(size(be_3d) .== size(be_3d.value))

    # ndims
    @test ndims(be_1d) == ndims(be_1d.value)
    @test ndims(be_2d) == ndims(be_2d.value)
    @test ndims(be_3d) == ndims(be_3d.value)

    # first
    @test first(be_1d) == first(be_1d.value)
    @test first(be_2d) == first(be_2d.value)
    @test first(be_3d) == first(be_3d.value)

    # last
    @test last(be_1d) == last(be_1d.value)
    @test last(be_2d) == last(be_2d.value)
    @test last(be_3d) == last(be_3d.value)

    # lastindex
    @test lastindex(be_1d) == lastindex(be_1d.value)
    @test lastindex(be_2d) == lastindex(be_2d.value)
    @test lastindex(be_3d) == lastindex(be_3d.value)

    # firstindex
    @test firstindex(be_1d) == firstindex(be_1d.value)
    @test firstindex(be_2d) == firstindex(be_2d.value)
    @test firstindex(be_3d) == firstindex(be_3d.value)

    # setindex!
    be_1d[1] = 10.0
    @test be_1d.value[1] == 10.0
    be_2d[2, 3] = 10.0
    @test be_2d[2, 3] == 10.0
    be_3d[2, 2, 3] = 10.0
    @test be_3d[2, 2, 3] == 10.0

    # eltype
    @test eltype(be_1d) == Float64
    @test eltype(be_2d) == Float64
    @test eltype(be_3d) == Float64

    # eachcol
    for (c1, c2) in zip(eachcol(be_2d), eachcol(be_2d.value))
        @test c1 == c2
    end
    # eachrow
    for (r1, r2) in zip(eachrow(be_2d), eachrow(be_2d.value))
        @test r1 == r2 
    end
    # eachslice
    for (s1, s2) in zip(eachslice(be_3d; dims=3), eachslice(be_3d; dims=3))
        @test s1 == s2
    end


    r_2d = rand(3, 3)
    # basic operators
    ops = [:+, :-, :/, :*]
    for op in ops
        expr = :(@test all($op(be_2d.value, r_2d) .== $op(be_2d, r_2d)))
        eval(expr)
    end
    ops = [:.+, :.-, :./, :.*]
    for op in ops
        expr = :(@test all($op(be_2d.value, r_2d) .== $op(be_2d, r_2d)))
        eval(expr)
    end
end