using CSV, DataFrames, Dates, TimeZones, Statistics
include("clusteringIndicators.jl")
include("data.jl")


#Open - the first traded price
#High - the highest traded price
#Low - the lowest traded price
#Close - the final traded price
#Volume - the total volume traded by all trades
#Trades - the number of individual trades
#1, 5, 15, 30, 60, 240, 720 (12 hour) and 1440 (24 hour) minute intervals
#BTC Data spans: 2013-10-06T21:34:00+02:00 - 2025-03-31T23:59:00+02:00
#ETH Data spans: 2015-08-07T14:03:00+02:00 - 2025-03-31T23:59:00+02:00


OHLCVT=Dict()
pair="ETHUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))
#interval="720"
#push!(OHLCVT[pair],interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict()))

#=ts = OHLCVT["ETHUSD"]["1440"][!,:timestamp][end]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

OHLCVT["ETHUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"]["df"])

#=NOTE: next step Validate!!!!!!!!!!!!
validation:
1) random calc of raw values 

69
149
1512
2289
3582
=#

#=
Transform 
Amihud illiquidity:
→ into log(1 + amihud)

Volume:
You already have a Z-score, so no need to standardize again.

Volatility and vol-of-vol:
Often stabilized with
→ log(vol)
→ then z-score


Apply z-score standardization to all remaining raw features
=#

OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["ETHUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["ETHUSD"]["1440"]["df"],OHLCVT["ETHUSD"]["1440"]["dict"]=zscore_df(OHLCVT["ETHUSD"]["1440"]["df"],cols)


#################################################################################################################

pair="XBTUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))
#interval="720"
#push!(OHLCVT[pair],interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict()))


OHLCVT["XBTUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["XBTUSD"]["1440"]["df"])

#=validation:
1) random calc of raw values 

69
149
1512
2289
3582
=#

#=
Transform 
Amihud illiquidity:
→ into log(1 + amihud)

Volume:
You already have a Z-score, so no need to standardize again.

Volatility and vol-of-vol:
Often stabilized with
→ log(vol)
→ then z-score


Apply z-score standardization to all remaining raw features
=#
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol5day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol5day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol10day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol10day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:gkVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:gkVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:volOfVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:volOfVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["XBTUSD"]["1440"]["df"],OHLCVT["XBTUSD"]["1440"]["dict"]=zscore_df(OHLCVT["XBTUSD"]["1440"]["df"],cols)

#=validation
The mean should be within [-0.05, +0.05]
The std should be within [0.98, 1.02]
=#
