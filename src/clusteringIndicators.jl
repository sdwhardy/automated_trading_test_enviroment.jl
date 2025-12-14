

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
    logreturn(prices::AbstractVector, n::Int=1)

Compute the *n-period logarithmic return* of a price series.

The logarithmic return at time `t` is defined as the natural logarithm
of the ratio between the current price and the price `n` periods earlier.

Missing values are returned for the first `n` observations, as the return
cannot be computed due to insufficient historical data.

# Arguments
- `prices::AbstractVector`:
  Vector of strictly positive asset prices ordered in time.
- `n::Int=1`:
  Return horizon (number of periods). Must satisfy `n ≥ 1`.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Logarithmic returns of the same length as `prices`, with `missing`
  values in positions `1:n`.

# Definition
For t > n:

r_t = ln(P_t / P_(t - n))

Notes:

Log-returns are time-additive and commonly used in financial modeling.

Prices must be strictly positive.

Setting n > 1 computes multi-period log-returns.

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
    roc(prices::AbstractVector, n::Int=1)

Compute the *n-period Rate of Change (ROC)* of a price series.

The ROC measures the relative change in price over a specified number
of periods. It is expressed as a fraction of the price `n` periods ago.

Missing values are returned for the first `n` observations, since the
ROC cannot be computed without sufficient historical data.

# Arguments
- `prices::AbstractVector`:
  Vector of asset prices ordered in time.
- `n::Int=1`:
  Return horizon (number of periods). Must satisfy `n ≥ 1`.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Rate of Change series of the same length as `prices`, with `missing`
  values in positions `1:n`.

# Definition
For t > n:

ROC_t = (P_t - P_(t - n)) / P_(t - n)

Notes
ROC is a simple momentum indicator commonly used in technical analysis.

Values can be positive (price increase) or negative (price decrease).

Setting n > 1 computes multi-period ROC.
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
    daily_realized_volatility(prices::AbstractVector, n::Int=21)

Compute the *daily realized volatility* using a rolling window of
logarithmic returns.

The realized volatility at time `t` is defined as the square root of the
average of squared daily log-returns over the previous `n` periods.
Missing values are returned for the first `n` observations.

# Arguments
- `prices::AbstractVector`:
  Vector of strictly positive asset prices ordered in time.
- `n::Int=21`:
  Lookback window size (typically 21 trading days ≈ one month).

# Returns
- `Vector{Union{Float64, Missing}}`:
  Daily realized volatility series with `missing` values in positions `1:n`.

# Definition
Let r_t denote the one-period log-return:

r_t = ln(P_t / P_{t-1})

For t > n, the realized volatility is defined as:

σ_t = sqrt((1 / n) * sum(r_{t-i}^2)),  for i = 1, …, n

# Notes
- This is a non-parametric volatility estimator.
- It assumes zero mean returns over the window.
- Often used as a proxy for latent volatility in financial models.
"""
function daily_realized_volatility(prices::AbstractVector, n::Int=21)
    N = length(prices)
    vol = Vector{Union{Float64, Missing}}(undef, N)
    vol[1:n] .= missing  # first n entries have no data

    # compute log returns
    logrets = logreturn(prices, 1)

    @inbounds for t in n+1:N
        # take the last n valid log returns ending at t-1
        window = skipmissing(logrets[(t-n):(t-1)])
        vol[t] = sqrt(sum(window.^2)/n)
    end

    return vol
end


"""
    annualized_realized_volatility(prices::AbstractVector, n::Int=21)

Compute the *annualized realized volatility* from daily log returns.

This function scales the daily realized volatility by the square root
of the number of trading periods per year, assuming independent and
identically distributed daily returns.

Missing values are returned for the first `n` observations.

# Arguments
- `prices::AbstractVector`:
  Vector of strictly positive asset prices ordered in time.
- `n::Int=21`:
  Lookback window size used to compute daily realized volatility.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Annualized realized volatility series.

# Definition
Let σ_t^d denote the daily realized volatility computed over the past
`n` days. The annualized realized volatility is defined as:

σ_t = sqrt(A) * σ_t^d

where A is the number of trading periods per year (typically A = 252).

# Notes
- Assumes no serial correlation in daily returns.
- Annualization is performed after volatility (not variance) estimation.
- Widely used for comparability across assets and horizons.
"""
function annualized_realized_volatility(prices::AbstractVector, n::Int=21)
    
     return  daily_realized_volatility(prices, n).*sqrt(252)

end

"""
    gk_volatility(
        open::AbstractVector,
        high::AbstractVector,
        low::AbstractVector,
        close::AbstractVector,
        n::Int=21;
        annualization_factor::Float64=252.0
    )

Compute the rolling *Garman–Klass (GK) volatility estimator* using OHLC data.

The Garman–Klass estimator is an efficient, non-parametric volatility
measure that exploits the daily high–low price range while correcting
for opening and closing price drift. The estimator is first computed
as a daily variance contribution, then averaged over a rolling window
and annualized.

Missing values are returned for the first `n` observations.

# Arguments
- `open::AbstractVector`:
  Opening prices.
- `high::AbstractVector`:
  High prices.
- `low::AbstractVector`:
  Low prices.
- `close::AbstractVector`:
  Closing prices.
- `n::Int=21`:
  Lookback window length (typically 21 trading days).
- `annualization_factor::Float64=252.0`:
  Scaling factor to annualize volatility.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Annualized Garman–Klass volatility series.

# Definition
Let O_t, H_t, L_t, and C_t denote the open, high, low, and close prices.
The daily Garman–Klass *variance* is defined as:

v_t = 0.5 * ln(H_t / L_t)^2 - (2 * ln(2) - 1) * ln(C_t / O_t)^2

For t > n, the rolling annualized volatility is:

σ_t = sqrt(A * (1 / n) * sum(v_{t-i})),  for i = 1, …, n

where A is the annualization factor.

# Notes
- Variance (not volatility) is averaged over time.
- Assumes zero overnight drift.
- More efficient than close-to-close volatility under ideal conditions.
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
    vol_of_vol(prices::AbstractVector, vol_window::Int=1, vov_window::Int=21)

Compute the *volatility of volatility* (VoV) from realized volatility.

This function first computes daily realized volatility using a rolling
window of length `vol_window`, then estimates the volatility of that
volatility as the standard deviation of realized volatility over a
second rolling window of length `vov_window`.

Missing values are returned for periods where insufficient data is
available.

# Arguments
- `prices::AbstractVector`:
  Vector of strictly positive asset prices ordered in time.
- `vol_window::Int=1`:
  Lookback window used to compute daily realized volatility.
- `vov_window::Int=21`:
  Lookback window used to compute volatility of volatility.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Volatility-of-volatility time series.

# Definition
Let σ_t^d denote daily realized volatility computed over `vol_window`
periods. The volatility of volatility is defined as the rolling standard
deviation of σ_t^d over `vov_window` observations:

v_t = std(σ_{t-i}^d),  for i = 0, …, vov_window - 1

# Notes
- Measures variability of volatility rather than price returns.
- Often interpreted as a proxy for volatility regime instability.
- Used in risk management, volatility forecasting, and option pricing.
"""
function vol_of_vol(prices::AbstractVector, vol_window::Int=1, vov_window::Int=21)
    
    vol=daily_realized_volatility(prices, vol_window)

    N = length(prices)

    vov = Vector{Union{Float64,Missing}}(undef, N)
    vov[1:(vol_window + vov_window - 1)] .= missing

    @inbounds for t in (vol_window + vov_window):N
        window = skipmissing(vol[(t - vov_window + 1):t])
        vov[t] = std(collect(window))
    end

    return vov
end

"""
    ma_diff(prices::AbstractVector, short_n::Int=10, long_n::Int=50)

Compute the *moving average difference* (MA difference) indicator.

This indicator measures the difference between a short-term and a
long-term simple moving average of prices. It is commonly used to
identify trend direction and momentum by comparing recent price behavior
to longer-term trends.

Missing values are returned until both moving averages are defined.

# Arguments
- `prices::AbstractVector`:
  Vector of asset prices ordered in time.
- `short_n::Int=10`:
  Window length of the short-term moving average.
- `long_n::Int=50`:
  Window length of the long-term moving average. Must satisfy
  `short_n < long_n`.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Time series of moving average differences.

# Definition
Let S_t denote the short-term moving average and L_t the long-term moving
average:

S_t = (1 / short_n) * sum(P_{t-i}),  i = 0, …, short_n - 1  
L_t = (1 / long_n)  * sum(P_{t-i}),  i = 0, …, long_n  - 1  

The moving average difference is then defined as:

D_t = S_t - L_t

# Notes
- Positive values indicate upward momentum.
- Negative values indicate downward momentum.
- Closely related to trend-following and crossover strategies.
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

"""
    volume_zscore(volume::AbstractVector, n::Int)

Compute the *volume z-score* over a rolling window.

This indicator standardizes the current trading volume relative to its
recent historical distribution by subtracting the rolling mean and
dividing by the rolling standard deviation. It measures how unusual
current volume is compared to recent activity.

Missing values are returned for the first `n` observations.

# Arguments
- `volume::AbstractVector`:
  Vector of traded volumes ordered in time.
- `n::Int`:
  Lookback window length used to compute the rolling mean and standard deviation.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Volume z-score time series.

# Definition
Let V_t denote trading volume at time t. Define the rolling mean μ_t and
standard deviation σ_t over the previous n observations:

μ_t = (1 / n) * sum(V_{t-i})  
σ_t = std(V_{t-i}),   i = 1, …, n

The volume z-score is then defined as:

Z_t = (V_t - μ_t) / σ_t

If σ_t is numerically zero, the z-score is set to zero.

# Notes
- Highlights unusually high or low trading activity.
- Commonly used to detect breakouts, accumulation, or distribution.
- Assumes volume observations are approximately stationary over the window.
"""
function volume_zscore(volume::AbstractVector, n::Int)
    N = length(volume)

    zscore = Vector{Union{Float64, Missing}}(undef, N)
    zscore[1:n] .= missing

    @inbounds for t in (n+1):N
        # historical window
        window = skipmissing(volume[(t-n):(t-1)])

        μ = simple_mean(collect(window))
        σ = simple_std(collect(window))

        current_volume = volume[t]

        zscore[t] = σ ≈ 0 ? 0.0 : (current_volume - μ) / σ
    end

    return zscore
end


"""
    amihud(prices::AbstractVector, volume::AbstractVector, n::Int=21)

Compute the *Amihud illiquidity measure* using price and volume data.

The Amihud illiquidity indicator measures the price impact of trading by
relating absolute returns to traded dollar volume. Higher values indicate
lower market liquidity, as small trading volumes are associated with
larger price movements.

Missing values are returned for the first `n` observations.

# Arguments
- `prices::AbstractVector`:
  Vector of asset prices ordered in time.
- `volume::AbstractVector`:
  Vector of traded volumes ordered in time.
- `n::Int=21`:
  Lookback window length used to compute rolling illiquidity.

# Returns
- `Vector{Union{Float64, Missing}}`:
  Amihud illiquidity time series.

# Definition
Let P_t denote the asset price and V_t the traded volume at time t.
Define the daily return magnitude and dollar volume as:

R_t = |P_t - P_{t-1}| / P_{t-1}  
DV_t = P_t * V_t

The daily illiquidity contribution is

I_t = R_t / DV_t

For t > n, the Amihud illiquidity measure is the rolling mean:

A_t = (1 / n) * sum(I_{t-i}),  for i = 1, …, n

# Notes
- Captures price impact per unit of trading volume.
- Higher values indicate lower liquidity.
- Widely used in empirical asset pricing and market microstructure.
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


function signed_log(x)
    return sign(x) * log1p(abs(x))
end

"""
    zscore_df(df::DataFrame, cols::Vector{Symbol}, ddof::Int=1)

Standardize selected DataFrame columns using z-score normalization.

For each column in `cols`, this function computes the mean and standard
deviation (ignoring missing values) and rescales the data to have zero
mean and unit variance. Missing values are preserved. The function
returns both the standardized DataFrame and the fitted scaling
parameters.

# Arguments
- `df::DataFrame`:
  Input DataFrame containing the data.
- `cols::Vector{Symbol}`:
  Columns to be standardized.
- `ddof::Int=1`:
  Degrees of freedom used in the standard deviation.
  If `ddof == 1`, the sample standard deviation is used.
  If `ddof == 0`, the population standard deviation is used.

# Returns
- `df_out::DataFrame`:
  Copy of the input DataFrame with standardized columns.
- `scaler::Dict{Symbol, Tuple{Float64, Float64}}`:
  Dictionary mapping each column to its `(mean, std)` used for scaling.

# Definition
Let X_{i,c} denote the value of column c at row i. For each column c,
the standardized value is defined as

Z_{i,c} = (X_{i,c} - μ_c) / σ_c

where μ_c and σ_c are the mean and standard deviation of column c,
computed ignoring missing values.

If σ_c is zero or undefined, it is replaced by 1 to avoid division by zero.

# Notes
- Standardization is performed column-wise.
- Scaling parameters are reusable for out-of-sample data.
- Commonly used as preprocessing for clustering and regression models.
"""
function zscore_df(df::DataFrame, cols::Vector{Symbol}, ddof::Int=1)
    # ensure suffix is string and produce column names
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

        df_out[!, Symbol(string(c))] = zcol
    end

    return df_out, scaler
end


function mean_and_std(df::DataFrame, cols::Vector{Symbol}, ddof::Int=1)
    # ensure suffix is string and produce column names
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
    end

    return scaler
end