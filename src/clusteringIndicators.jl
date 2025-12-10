

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
    df[!,:realVol5day] = annualized_realized_volatility(df[!,:close], 5)
    df[!,:realVol10day] = annualized_realized_volatility(df[!,:close], 10)
    df[!,:realVol21day] = annualized_realized_volatility(df[!,:close], 21)

    #=
    4) Garman–Klass volatility (21-day)
    σ^2​=1/n∑(1:n​)[0.5(ln(​Hi/Li​​))^2−(2ln2−1)(ln(​Ci/Oi​​))^2]
    =#
    df[!,:gkVol21day] = gk_volatility(df[!,:open],df[!,:high],df[!,:low],df[!,:close], 21)

    #=
    5) Volatility-of-volatility 
    take stdev eqn (3) of RV over 21d
    =#
    df[!,:volOfVol21day] = vol_of_vol(df[!,:close], 1, 21)

    #=
    6) Moving average diff (10day-50day, 100day-20day)
    =#
    df[!,:maDiff10_50day] = ma_diff(df[!,:close], 10, 50)
    df[!,:maDiff20_100day] = ma_diff(df[!,:close], 20, 100)
    #=
    7)Volume Z score
    Zn=Vo-Vmean,n/Vstd,n
    =#
    df[!,:volumeZscore21day] = volume_zscore(df[!,:volume], 21)
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
    daily_realized_vol_df(prices::AbstractVector, n::Int=21)

Compute n-period rolling realized volatility (standard deviation of log returns) as a DataFrame column aligned with `prices`.
First `n` entries are `missing`.
"""
function daily_realized_volatility(prices::AbstractVector, n::Int=21)
    N = length(prices)
    vol = Vector{Union{Float64, Missing}}(undef, N)
    vol[1:n] .= missing  # first n entries have no data

    # compute log returns
    logrets =logreturn(prices, 1)
    
    @inbounds for t in n:N
        vol[t] = sqrt(sum(logrets[t-n+1:t].^2)/n)
    end

    return vol
end

"""
    annualized_realized_vol_df(prices::AbstractVector, n::Int=21)

Compute n-period rolling realized volatility (standard deviation of log returns) as a DataFrame column aligned with `prices`.
First `n` entries are `missing`.
"""
function annualized_realized_volatility(prices::AbstractVector, n::Int=21)
    
     return  daily_realized_volatility(prices, n).*sqrt(252)

end

"""
    gk_volatility(open, high, low, close, n=21; annualization_factor=252.0)

Calculates the rolling Garman-Klass Volatility, which is an estimator of 
volatility that incorporates open, high, low, and close prices.

The function calculates the daily Garman-Klass variance, averages it over 
a rolling window of 'n' periods, takes the square root, and annualizes the result.

Arguments:
- `open`, `high`, `low`, `close`: AbstractVectors of price data (must be the same length).
- `n`: The look-back period for the rolling window (default is 21 trading days/periods).
- `annualization_factor`: Factor used to annualize the daily volatility (default 252.0).

Returns:
- A Vector{Union{Float64, Missing}} containing the annualized Garman-Klass volatility.
  The first 'n' elements will be 'missing' as there is not enough data.
"""
function gk_volatility(
    open::AbstractVector, 
    high::AbstractVector, 
    low::AbstractVector, 
    close::AbstractVector, 
    n::Int=21; 
    annualization_factor::Float64=252.0
)
    N = length(close)
    @assert N > 0 "Input vectors cannot be empty."
    @assert n > 0 "Lookback period 'n' must be positive."
    @assert length(open) == N && length(high) == N && length(low) == N "OHLC vectors must have the same length."

    # Pre-calculate the constant factor (2*ln(2) - 1)
    # This factor is used to weight the close-to-open return component.
    const_factor = 2 * log(2) - 1
    
    # 1. Calculate Daily Garman-Klass *Variance* Contributions
    # Variance is additive, so we must operate on variance, not volatility.
    gk_daily_variance = Vector{Float64}(undef, N)
    
    @inbounds for t in 1:N
        # Range Component (High/Low)
        log_hl = log(high[t] / low[t])
        range_variance = 0.5 * log_hl^2
        
        # Close-to-Open Component (Drift)
        log_co = log(close[t] / open[t])
        co_variance_correction = const_factor * log_co^2
        
        # Garman-Klass Daily Variance
        gk_daily_variance[t] = range_variance - co_variance_correction
    end

    # 2. Calculate Rolling n-period Annualized GK Volatility
    vol = Vector{Union{Float64, Missing}}(undef, N)
    
    # The first 'n' periods cannot be calculated
    vol[1:n] .= missing 
    
    # Calculate rolling volatility from the (n+1)-th period onward
    @inbounds for t in (n+1):N
        # Define the window: from t-n up to t-1
        window = gk_daily_variance[(t-n):(t-1)]
        
        # Calculate the average variance (mean of the daily variances)
        rolling_mean_variance = sum(window) / n 
        
        # Annualize (multiply by factor) and take the square root to get volatility
        # RV_GK = sqrt(AnnualizationFactor * RollingMeanVariance)
        vol[t] = sqrt(annualization_factor * rolling_mean_variance)
    end

    return vol
end

"""
    vol_of_vol_df(prices::AbstractVector, vol_window::Int=21, vov_window::Int=21)

Compute Vol-of-Vol (rolling std of realized volatility) and return a DataFrame aligned with `prices`.
- vol_window: window for realized volatility
- vov_window: window for Vol-of-Vol
"""
function vol_of_vol(prices::AbstractVector, vol_window::Int=1, vov_window::Int=21)
    
    vol=daily_realized_volatility(prices, vol_window)

    vov=daily_realized_volatility(vol, vov_window)

    return vov
end

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

function simple_mean(arr::AbstractVector)
    return sum(arr) / length(arr)
end

function simple_std(arr::AbstractVector)
    n = length(arr)
    # Use n-1 for sample standard deviation, which is standard in finance
    return sqrt(sum((arr .- simple_mean(arr)).^2) / (n - 1))
end

# --- 1. PROPERLY IMPLEMENTED VOLUME Z-SCORE FUNCTION ---

function volume_zscore(volume::AbstractVector, n::Int)
    N = length(volume)
    
    # Initialize the output vector with missing values.
    # The first 'n' entries must be missing because a rolling window of size 'n' 
    # must be complete, and the result is reported on the next day (n+1).
    zscore = Vector{Union{Float64, Missing}}(undef, N)
    zscore[1:n] .= missing  # First 'n' entries are missing

    # The first valid calculation is reported at t = n + 1 
    @inbounds for t in (n+1):N
        
        # --- Define Historical Window (t-n through t-1) ---
        # This window ensures n days of volume *prior* to the current day volume[t] 
        # are used for mean (μ) and standard deviation (σ). NON-LOOK-AHEAD COMPLIANT.
        window = volume[(t-n):(t-1)]
        
        # Calculate Historical Context (μ_t-1 and σ_t-1)
        μ = simple_mean(window) 
        σ = simple_std(window)  
        
        # Z-Score Calculation: (Current Volume - Historical Mean) / Historical StDev
        current_volume = volume[t]
        
        # Handle zero standard deviation: set Z-Score to 0 to prevent division by zero
        zscore[t] = σ ≈ 0 ? 0.0 : (current_volume - μ) / σ
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
