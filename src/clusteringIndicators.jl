

function calculate_clustering_indicators(df)
    #=
    Use the following for clustering indicators
    1)Log return
    log return​=ln(Pt​/Pt-1)​
    close to close of t=1, 5, 21 rolling windows
    =#
    df[!,:lnReturn1day] = logreturn(df[!,:close], 1)
    df[!,:lnReturn5day] = logreturn(df[!,:close], 5)
    df[!,:lnReturn21day] = logreturn(df[!,:close], 21)
    #=
    2) Rate of change
    ROC t​(n)=(​Pt-Pt−n)​/Pt−n​
    21 and 63 day
    =#
    df[!,:roc21day] = roc(df[!,:close], 21)
    df[!,:roc63day] = roc(df[!,:close], 63)


    #=
    3) Realized volatility
    σn​=sqrt(1/n−1∑(i=1:n)​(ri​−rˉ)^2)
    ​annuallized=σn​ \dot sqrt(252)
    ​5, 10, 21 day windows
    =#
    df[!,:realVol5day] = realized_volatility(df[!,:close], 5)
    df[!,:realVol10day] = realized_volatility(df[!,:close], 10)
    df[!,:realVol21day] = realized_volatility(df[!,:close], 21)

    #=
    4) Garman–Klass volatility (21-day)
    σ^2​=1/n∑(1:n​)[0.5(ln(​Hi/Li​​))^2−(2ln2−1)(ln(​Ci/Oi​​))^2]
    =#
    df[!,:gkVol21day] = gk_volatility(df[!,:open],df[!,:high],df[!,:low],df[!,:close], 21)

    #=
    5) Volatility-of-volatility 
    take stdev eqn (3) of RV over 21d
    =#
    df[!,:volOfVol21day] = vol_of_vol(df[!,:close], 21, 21)

    #=
    6) Moving average diff (10day-50day, 100day-20day)
    =#
    df[!,:maDiff10_50day] = ma_diff(df[!,:close], 10, 50)
    df[!,:maDiff20_100day] = ma_diff(df[!,:close], 20, 100)
    #=
    7)Volume Z score
    Zn=Vo-Vmean,n/Vstd,n
    =#
    df[!,:volumeZscore21day] = volume_zscore(df[!,:volume], 20)
    #=
    8) Amihud illiquidity
    Al=(abs(Pt-Pt-1)/Pt-1)/Pt*Vt
    - average this over a window
    =#
    df[!,:amihud21day] = amihud(df[!,:close],df[!,:volume], 21)
    
    return df
end

"""
    logreturn(prices, n::Int=1)

Compute n-period log returns for a vector of prices.
Returns a vector of length `length(prices) - n`.
"""
function logreturn(prices::AbstractVector, n::Int=1)
    N = length(prices)
    logret = Vector{Union{Float64, Missing}}(undef, N)  # allow missing
    logret[1:n] .= missing                               # first n entries have no data

    @inbounds for t in (n+1):N
        logret[t] = log(prices[t] / prices[t-n])
    end

    return logret
end

"""
    roc_df(prices::AbstractVector, n::Int=1)

Compute n-period Rate of Change (ROC) as a DataFrame column aligned with `prices`.
First `n` entries are `missing`.
"""
function roc(prices::AbstractVector, n::Int=1)
    N = length(prices)
    roc = Vector{Union{Float64, Missing}}(undef, N)   # allow missing
    roc[1:n] .= missing                                # first n entries have no data

    @inbounds for t in (n+1):N
        roc[t] = (prices[t] - prices[t-n]) / prices[t-n]
    end

    return roc
end

"""
    realized_vol_df(prices::AbstractVector, n::Int=21)

Compute n-period rolling realized volatility (standard deviation of log returns) as a DataFrame column aligned with `prices`.
First `n` entries are `missing`.
"""
function realized_volatility(prices::AbstractVector, n::Int=21)
    N = length(prices)
    vol = Vector{Union{Float64, Missing}}(undef, N)
    vol[1:n] .= missing  # first n entries have no data

    # compute log returns
    logrets = log.(prices[2:end] ./ prices[1:end-1])

    @inbounds for t in (n+1):N
        vol[t] = Statistics.std(logrets[(t-n):(t-1)])
    end

    return vol
end

"""
    gk_vol_df(open::AbstractVector, high::AbstractVector, low::AbstractVector, close::AbstractVector, n::Int=21)

Compute n-period rolling Garman-Klass volatility as a DataFrame column aligned with price vectors.
First `n` entries are missing.
"""
function gk_volatility(open::AbstractVector, high::AbstractVector, low::AbstractVector, close::AbstractVector, n::Int=21)
    N = length(close)
    @assert length(open) == N && length(high) == N && length(low) == N "OHLC vectors must have same length"
    
    gk_daily = Vector{Float64}(undef, N)
    @inbounds for t in 1:N
        log_hl = log(high[t] / low[t])
        log_co = log(close[t] / open[t])
        gk_daily[t] = sqrt(0.5 * log_hl^2 - (2*log(2) - 1) * log_co^2)
    end

    # rolling n-period GK volatility
    vol = Vector{Union{Float64, Missing}}(undef, N)
    vol[1:n] .= missing
    @inbounds for t in (n+1):N
        vol[t] = std(gk_daily[(t-n):(t-1)])
    end

    return vol
end

using DataFrames, Statistics

"""
    vol_of_vol_df(prices::AbstractVector, vol_window::Int=21, vov_window::Int=21)

Compute Vol-of-Vol (rolling std of realized volatility) and return a DataFrame aligned with `prices`.
- vol_window: window for realized volatility
- vov_window: window for Vol-of-Vol
"""
function vol_of_vol(prices::AbstractVector, vol_window::Int=21, vov_window::Int=21)
    N = length(prices)
    
    # Step 1: compute daily log returns
    logrets = log.(prices[2:end] ./ prices[1:end-1])
    
    # Step 2: compute rolling realized volatility
    rv = Vector{Union{Float64, Missing}}(undef, N)
    rv[1:vol_window] .= missing
    @inbounds for t in (vol_window+1):N
        rv[t] = std(logrets[(t-vol_window):(t-1)])
    end

    # Step 3: compute rolling Vol-of-Vol
    vov = Vector{Union{Float64, Missing}}(undef, N)
    vov[1:(vol_window + vov_window)] .= missing  # first entries missing
    @inbounds for t in (vol_window + vov_window + 1):N
        window_vals = skipmissing(rv[(t-vov_window):(t-1)])
        vov[t] = std(collect(window_vals))
    end

    return vov
end

using DataFrames, Statistics

"""
    ma_diff(prices::AbstractVector, short_n::Int=10, long_n::Int=50)

Compute moving average difference: MA(short_n) - MA(long_n)
Returns a DataFrame aligned with `prices`.
"""
function ma_diff(prices::AbstractVector, short_n::Int=10, long_n::Int=50)
    N = length(prices)
    @assert short_n < long_n "short_n must be less than long_n"

    short_ma = Vector{Union{Float64, Missing}}(undef, N)
    long_ma  = Vector{Union{Float64, Missing}}(undef, N)
    
    # compute rolling averages
    short_ma[1:short_n-1] .= missing
    long_ma[1:long_n-1] .= missing

    @inbounds for t in short_n:N
        short_ma[t] = mean(prices[(t-short_n+1):t])
    end

    @inbounds for t in long_n:N
        long_ma[t] = mean(prices[(t-long_n+1):t])
    end

    # compute MA difference
    ma_diff = Vector{Union{Float64, Missing}}(undef, N)
    @inbounds for t in 1:N
        if !ismissing(short_ma[t]) && !ismissing(long_ma[t])
            ma_diff[t] = short_ma[t] - long_ma[t]
        else
            ma_diff[t] = missing
        end
    end

    return ma_diff
end


"""
    volume_zscore_df(volume::AbstractVector, n::Int=21)

Compute rolling Z-score of volume over a window of n periods.
Returns a DataFrame aligned with the original volume vector.
"""
function volume_zscore(volume::AbstractVector, n::Int=21)
    N = length(volume)
    zscore = Vector{Union{Float64, Missing}}(undef, N)
    zscore[1:n-1] .= missing  # first n-1 entries missing

    @inbounds for t in n:N
        window = volume[(t-n+1):t]
        μ = mean(window)
        σ = std(window)
        zscore[t] = σ ≈ 0 ? 0.0 : (volume[t] - μ)/σ
    end

    return zscore
end

"""
    amihud(prices::AbstractVector, volume::AbstractVector, n::Int=21)

Compute n-period rolling Amihud Illiquidity Ratio.
Returns a DataFrame aligned with the original price vector.
- prices: daily close prices
- volume: daily volumes (number of shares)
- n: rolling window
"""
function amihud(prices::AbstractVector, volume::AbstractVector, n::Int=21)
    N = length(prices)
    @assert length(volume) == N "Prices and volume must have the same length"

    illiq = Vector{Union{Float64, Missing}}(undef, N)
    illiq[1:n] .= missing  # first n entries missing

    # compute daily absolute return / dollar volume
    daily_illiq = Vector{Float64}(undef, N-1)
    @inbounds for t in 2:N
        R = abs(prices[t] - prices[t-1]) / prices[t-1]
        dollar_vol = prices[t] * volume[t]
        daily_illiq[t-1] = dollar_vol ≈ 0 ? missing : R / dollar_vol
    end

    # compute rolling n-period mean
    @inbounds for t in (n+1):N
        window = daily_illiq[(t-n):(t-1)]
        illiq[t] = mean(skipmissing(window))
    end

    return illiq
end

"""
    zscore_df(df::DataFrame, cols::Vector{Symbol};
              suffix::AbstractString = :_z,
              ddof::Int = 1)

Standardize columns `cols` in `df` using z-score (mean/std) while ignoring `missing`.
Returns a tuple `(df_out, scaler)` where:
- `df_out` is a DataFrame with new columns named `col * suffix` containing z-scores (type `Union{Float64,Missing}`).
- `scaler` is a Dict{Symbol, Tuple{Float64, Float64}} mapping each column => (mean, std).

Parameters:
- `cols`: vector of column Symbols to standardize.
- `suffix`: suffix appended to original column name (default `:_z` -> `:feature_z`).
- `ddof`: delta degrees of freedom for std (1 for sample std).
"""
function zscore_df(df::DataFrame, cols::Vector{Symbol}; suffix::AbstractString=":z", ddof::Int=1)
    # ensure suffix is string and produce column names
    suf = typeof(suffix) <: Symbol ? string(suffix) : String(suffix)
    df_out = copy(df)
    scaler = Dict{Symbol, Tuple{Float64, Float64}}()

    for c in cols
        @assert hasproperty(df, c) "Column $(c) not found"

        col = df[!, c]
        # compute mean and std ignoring missing
        vals = collect(skipmissing(col))
        if isempty(vals)
            μ = NaN
            σ = NaN
        else
            μ = mean(vals)
            σ = std(vals; corrected = (ddof==1))
            # guard against zero std
            if σ == 0.0 || isnan(σ)
                σ = 1.0
            end
        end

        scaler[c] = (μ, σ)

        # create output column with same length, allow missing
        zcol = Vector{Union{Float64, Missing}}(undef, length(col))
        @inbounds for i in eachindex(col)
            if ismissing(col[i])
                zcol[i] = missing
            else
                zcol[i] = (float(col[i]) - μ) / σ
            end
        end

        newname = Symbol(string(c) * suf)
        df_out[!, newname] = zcol
    end

    return df_out, scaler
end
