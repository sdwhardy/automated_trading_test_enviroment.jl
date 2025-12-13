using CSV, DataFrames, Dates, TimeZones, Statistics
include("clusteringIndicators.jl")
include("data.jl")
include("feature_validation.jl")

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
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))

#=ts = OHLCVT["XBTUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

OHLCVT["XBTUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["XBTUSD"]["1440"]["df"])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], Not(exclude)))

#=for col in cols
    _mn=minimum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    _mx=maximum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    println("Range of ", col, " is: ", _mn, " - ", _mx)
end=#

#=Z score Normalization Transform=#
OHLCVT["XBTUSD"]["1440"]["df"][!,:roc21day] .= signed_log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:roc21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:roc63day] .= signed_log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:roc63day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["XBTUSD"]["1440"]["df"], OHLCVT["XBTUSD"]["1440"]["dict"]=zscore_df(OHLCVT["XBTUSD"]["1440"]["df"],cols)
OHLCVT["XBTUSD"]["1440"]["dict"]=mean_and_std(OHLCVT["XBTUSD"]["1440"]["df"], cols)

results=validate_features(OHLCVT["XBTUSD"]["1440"]["df"])


#=validation
The mean should be within [-0.05, +0.05]
The std should be within [0.98, 1.02]
=#

#################################################################################################################
#################################### ETH ########################################################################
#################################################################################################################

OHLCVT=Dict()
pair="ETHUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))

#=ts = OHLCVT["ETHUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

OHLCVT["ETHUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"]["df"])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades]
cols = Symbol.(names(OHLCVT["ETHUSD"]["1440"]["df"], Not(exclude)))

for col in cols
    _mn=minimum(collect(skipmissing(OHLCVT["ETHUSD"]["1440"]["df"][!,col])))
    _mx=maximum(collect(skipmissing(OHLCVT["ETHUSD"]["1440"]["df"][!,col])))
    println("Range of ", col, " is: ", _mn, " - ", _mx)
end

#=Z score Normalization Transform 
Amihud illiquidity → log(1 + amihud)
Volatility and vol-of-vol:
stabilized with → log(vol) then z-score
remainder z score directly=#

OHLCVT["ETHUSD"]["1440"]["df"][!,:roc21day] .= signed_log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:roc21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:roc63day] .= signed_log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:roc63day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["ETHUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["ETHUSD"]["1440"]["df"],OHLCVT["ETHUSD"]["1440"]["dict"]=zscore_df(OHLCVT["ETHUSD"]["1440"]["df"],cols)
OHLCVT["ETHUSD"]["1440"]["dict"]=mean_and_std(OHLCVT["ETHUSD"]["1440"]["df"], cols)
