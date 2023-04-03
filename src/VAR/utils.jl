import TSFrames.lag

const FREQUENCIES = [:day, :month, :quarter, :year]
@doc """
     Check whether data is of a regular and known frequency.

## Arguments

- `data::TSFrame`: Time series data
"""
function _check_regularity_data(data::TSFrame) 
    for freq in FREQUENCIES
        if isregular(data, freq)
            return true
        end
    end
    return false
end

"""

If lag_value is a range or a vector of lags, then a vector of TSFrames will be
returned where each element of the vector is one lagged TSFrame.
"""
function lag(ts::TSFrame, lag_value::Union{OrdinalRange{T, T}, AbstractVector{T}}) where {T}
    lags = collect(lag_value)
    dfs = [TSFrames.lag(ts, l) for l in lags]
    return dfs
end
"""

Create a lagged TSFrame.

# Arguments

- `ts::TSFrame`: TSFrame to lag
- `lag_value::Union{OrdinalRange{T, T}, AbstractVector{T}}`: lags

# Keyword Arguments

- `name_attach::String="__L"`: String to attach to variable names
- `remove_fist::Bool=true`: Remove the first few rows tha will be `missing` for
  the lagged variables?
"""
function _lag_ts(
    ts::TSFrame, 
    lag_value::Union{OrdinalRange{T, T}, AbstractVector{T}}; 
    name_attach::String="__L", 
    remove_first::Bool=true
) where {T}
    lags = collect(lag_value)
    dfs = lag(ts, lag_value)
    dfs = [TSFrames.rename!(dfs[i], names(dfs[i]) .* "$(name_attach)$(l)") for (i,l) in zip(eachindex(dfs), lags)]
    df = join(dfs...; jointype=:JoinAll)
    if remove_first
        df = df[maximum(lags)+1:end]
    end
    return df
end