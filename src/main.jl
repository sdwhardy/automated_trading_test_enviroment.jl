import automated_trading_test_environment as ATTE

# --------------------------------------------------------------------------------------------------
# Data definitions
# --------------------------------------------------------------------------------------------------
# OHLCVT fields:
#   Open   : first traded price in the interval
#   High   : highest traded price
#   Low    : lowest traded price
#   Close  : final traded price
#   Volume : total traded volume
#   Trades : number of individual trades
#
# Supported intervals (minutes):
#   1, 5, 15, 30, 60, 240, 720 (12h), 1440 (24h)
#
# Data coverage:
#   BTC : 2013-10-06T21:34:00+02:00 → 2025-03-31T23:59:00+02:00
#   ETH : 2015-08-07T14:03:00+02:00 → 2025-03-31T23:59:00+02:00
# --------------------------------------------------------------------------------------------------


####################################################################################################
# BTC — Daily interval
####################################################################################################

pair     = "XBTUSD"
interval = "1440"

# Load OHLCVT dataframe
OHLCVT_df = ATTE.get_pair_interval_df(pair, interval)


# --------------------------------------------------------------------------------------------------
# Feature engineering for clustering and regime analysis
# --------------------------------------------------------------------------------------------------

# 1) Log returns
#    ln(P_t / P_{t-n}) for multiple rolling horizons
OHLCVT_df[!, :lnReturn1day]  = ATTE.logreturn(OHLCVT_df[!, :close], 1)
OHLCVT_df[!, :lnReturn5day]  = ATTE.logreturn(OHLCVT_df[!, :close], 5)
OHLCVT_df[!, :lnReturn21day] = ATTE.logreturn(OHLCVT_df[!, :close], 21)


# 2) Rate of Change (ROC)
#    (P_t - P_{t-n}) / P_{t-n}
OHLCVT_df[!, :roc21day] = ATTE.roc(OHLCVT_df[!, :close], 21)
OHLCVT_df[!, :roc63day] = ATTE.roc(OHLCVT_df[!, :close], 63)


# 3) Realized volatility (annualized)
#    σ_n = sqrt(1/(n-1) * Σ (r_i - r̄)^2) * sqrt(252)
OHLCVT_df[!, :realVol5day]  = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 5)
OHLCVT_df[!, :realVol10day] = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 10)
OHLCVT_df[!, :realVol21day] = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 21)


# 4) Garman–Klass volatility (21-day)
#    σ² = (1/n) Σ [0.5 ln(H/L)² − (2 ln 2 − 1) ln(C/O)²]
OHLCVT_df[!, :gkVol21day] = ATTE.gk_volatility(
    OHLCVT_df[!, :open],
    OHLCVT_df[!, :high],
    OHLCVT_df[!, :low],
    OHLCVT_df[!, :close],
    21
)


# 5) Volatility-of-volatility
#    Standard deviation of realized volatility over a 21-day window
OHLCVT_df[!, :volOfVol21day] = ATTE.vol_of_vol(OHLCVT_df[!, :close], 1, 21)


# 6) Moving-average differentials
OHLCVT_df[!, :maDiff10_50day]  = ATTE.ma_diff(OHLCVT_df[!, :close], 10, 50)
OHLCVT_df[!, :maDiff20_100day] = ATTE.ma_diff(OHLCVT_df[!, :close], 20, 100)


# 7) Volume Z-score
#    Z_n = (V_t − mean_n) / std_n
OHLCVT_df[!, :volumeZscore21day] = ATTE.volume_zscore(OHLCVT_df[!, :volume], 21)


# 8) Amihud illiquidity (21-day average)
#    |P_t − P_{t−1}| / (P_{t−1} * P_t * V_t)
OHLCVT_df[!, :amihud21day] = ATTE.amihud(
    OHLCVT_df[!, :close],
    OHLCVT_df[!, :volume],
    21
)


# --------------------------------------------------------------------------------------------------
# Pre-normalization nonlinear transforms
# --------------------------------------------------------------------------------------------------

# Signed log transform for heavy-tailed ROC features
OHLCVT_df[!, :roc21day] .= ATTE.signed_log.(OHLCVT_df[!, :roc21day])
OHLCVT_df[!, :roc63day] .= ATTE.signed_log.(OHLCVT_df[!, :roc63day])

# Log transform for strictly positive Amihud measure
OHLCVT_df[!, :amihud21day] = log.(1.0 .+ OHLCVT_df[!, :amihud21day])


# --------------------------------------------------------------------------------------------------
# Z-score normalization
# --------------------------------------------------------------------------------------------------

exclude = [
    :timestamp, :open, :high, :low, :close,
    :volume, :trades, :volumeZscore21day
]

cols = Symbol.(names(OHLCVT_df, ATTE.Not(exclude)))

OHLCVT_df, OHLCVT_dict = ATTE.zscore_df(OHLCVT_df, cols)
OHLCVT_dict            = ATTE.mean_and_std(OHLCVT_df, cols)


# --------------------------------------------------------------------------------------------------
# Feature validation and PCA
# --------------------------------------------------------------------------------------------------

feature_cols = [
    :lnReturn1day,
    :lnReturn5day,
    :lnReturn21day,
    :roc21day,
    :roc63day,
    :realVol5day,
    :realVol10day,
    :realVol21day,
    :gkVol21day,
    :volOfVol21day,
    :maDiff10_50day,
    :maDiff20_100day,
    :volumeZscore21day,
    :amihud21day
]

results  = ATTE.validate_features(OHLCVT_df, feature_cols)
pca_dict = ATTE.PCA(OHLCVT_df, feature_cols)

df_pca = ATTE.percent_explained_PCA(
    pca_dict["pca_df"],
    pca_dict["eigen_values"],
    pca_dict["nonmissing_idx"]
)






#NOTE - organize and explain PCA in typst


#=ts = OHLCVT["XBTUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

#=for col in cols
    _mn=minimum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    _mx=maximum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    println("Range of ", col, " is: ", _mn, " - ", _mx)
end=#

