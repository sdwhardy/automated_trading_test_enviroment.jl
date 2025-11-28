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
push!(OHLCVT,pair=>Dict(interval=>get_pair_interval_df(pair,interval)))
interval="720"
push!(OHLCVT[pair],interval=>get_pair_interval_df(pair,interval))
    



pair="XBTUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>get_pair_interval_df(pair,interval)))
interval="720"
push!(OHLCVT[pair],interval=>get_pair_interval_df(pair,interval))



#=ts = OHLCVT["ETHUSD"]["1440"][!,:timestamp][end]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#
OHLCVT["ETHUSD"]["1440"]=calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"])
OHLCVT["XBTUSD"]["1440"]=calculate_clustering_indicators(OHLCVT["XBTUSD"]["1440"])

