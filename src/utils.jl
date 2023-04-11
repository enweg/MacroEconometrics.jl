
"""
    stack_last_dim(arr::AbstractArray)

Stack the last dimension into the second to last dimension. 

This is useful here, since we generally have the convention that the last
dimension reflects the chain, and the second to last dimension the draws. As
such, this function stacks the chains such that all draws of all chains are
considered together. 

## Arguments

- `arr::AbstractArray`

## Returns

- returns an array with one dimension less but with the second to last dimension
  enlarged by a factor equal the size of the last dimension of `arr`.
"""
function stack_last_dim(arr::AbstractArray)
    nd = ndims(arr)
    n_draws = size(arr, nd-1)
    n_chains = size(arr, nd)
    arr_stacked = Array{eltype(arr)}(undef, vcat(size(arr)[1:nd-2]..., n_draws*n_chains)...)
    for slice in 1:n_chains
        s = (slice-1)*n_draws+1
        e = s + n_draws - 1
        arr_stacked[fill(:, nd-2)..., s:e] = selectdim(arr, nd, slice)
    end
    return arr_stacked
end
