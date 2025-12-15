import automated_trading_test_environment as ATTE

#Open - the first traded price
#High - the highest traded price
#Low - the lowest traded price
#Close - the final traded price
#Volume - the total volume traded by all trades
#Trades - the number of individual trades
#1, 5, 15, 30, 60, 240, 720 (12 hour) and 1440 (24 hour) minute intervals
#BTC Data spans: 2013-10-06T21:34:00+02:00 - 2025-03-31T23:59:00+02:00
#ETH Data spans: 2015-08-07T14:03:00+02:00 - 2025-03-31T23:59:00+02:00


#################################################################################################################
######################################################### BTC ###################################################
#################################################################################################################

OHLCVT=Dict()
pair="XBTUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>ATTE.get_pair_interval_df(pair,interval),"dict"=>Dict())))

#=ts = OHLCVT["XBTUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

OHLCVT["XBTUSD"]["1440"]["df"]=ATTE.calculate_clustering_indicators(OHLCVT["XBTUSD"]["1440"]["df"])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], ATTE.Not(exclude)))

#=for col in cols
    _mn=minimum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    _mx=maximum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    println("Range of ", col, " is: ", _mn, " - ", _mx)
end=#

#=Z score Normalization Transform=#
OHLCVT["XBTUSD"]["1440"]["df"][!,:roc21day] .= ATTE.signed_log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:roc21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:roc63day] .= ATTE.signed_log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:roc63day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], ATTE.Not(exclude)))
OHLCVT["XBTUSD"]["1440"]["df"], OHLCVT["XBTUSD"]["1440"]["dict"]=ATTE.zscore_df(OHLCVT["XBTUSD"]["1440"]["df"],cols)
OHLCVT["XBTUSD"]["1440"]["dict"]=ATTE.mean_and_std(OHLCVT["XBTUSD"]["1440"]["df"], cols)



# ----------------------------
# 1. Define your feature columns
# ----------------------------
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

results=ATTE.validate_features(OHLCVT["XBTUSD"]["1440"]["df"], feature_cols)

pca_dict=ATTE.PCA(OHLCVT["XBTUSD"]["1440"]["df"], feature_cols)

df_pca = ATTE.percent_explained_PCA(pca_dict["pca_df"],pca_dict["eigen_values"], pca_dict["nonmissing_idx"])

#NOTE - organize and explain PCA in typst